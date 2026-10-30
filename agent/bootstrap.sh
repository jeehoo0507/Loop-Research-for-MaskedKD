#!/usr/bin/env bash
# One command from an empty server to a running research loop. HUMAN starts it once:
#
#   bash agent/bootstrap.sh          # re-launches itself in tmux session "boot"; safe to re-run (skips finished steps)
#
# Steps (stops at the first failure and says why in agent/BOOT_STATUS.md and agent/logs/boot.log):
#   check tools/logins -> setup -> lock -> GPU smoke -> teacher fine-tune -> bench (FLOP check)
#   -> epochs by rule -> commit + relock -> CIFAR gate (waits hours) -> baselines -> loop
#   + in the background: ImageNet-100 preparation and its own gate/baselines (agent/stage2_boot.sh)
# If the CIFAR gate fails it stops: switching to the fallback setting is a human decision (docs/SETUP.md).
set -u
cd "$(dirname "$0")/.."
if [ -z "${TMUX:-}" ] && [ -z "${BOOT_NO_TMUX:-}" ]; then
  command -v tmux >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq tmux; }
  tmux has-session -t boot 2>/dev/null && { echo "already running: tmux attach -t boot"; exit 0; }
  tmux new -d -s boot "bash agent/bootstrap.sh; echo; echo '[boot] finished; press enter'; read"
  echo "started in tmux. Watch: tmux attach -t boot   (detach: Ctrl-b d)   or: tail -f agent/logs/boot.log"
  exit 0
fi

PY=.venv/bin/python
S=agent/.state; mkdir -p "$S" agent/logs
LOG=agent/logs/boot.log
say()  { echo "[boot $(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
status() { { echo "- $(date '+%Y-%m-%d %H:%M') $*"; } >> agent/BOOT_STATUS.md; say "$*"; }
fail() { status "STOPPED: $*"; git add agent/BOOT_STATUS.md >/dev/null 2>&1; git commit -qm "boot: stopped" >/dev/null 2>&1; git push -q >/dev/null 2>&1; exit 1; }
done_step() { [ -f "$S/boot_$1" ]; }
mark() { touch "$S/boot_$1"; }
[ -f agent/BOOT_STATUS.md ] || printf '# BOOT_STATUS (agent/bootstrap.sh)\n\n' > agent/BOOT_STATUS.md

# ---------- 0. tools, logins, disk ----------
say "checking tools"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG" || fail "no nvidia-smi / GPU"
command -v claude >/dev/null || fail "claude CLI not found. Install Claude Code, run 'claude' once to log in, then re-run"
git push --dry-run -q >/dev/null 2>&1 || fail "git push not authorized. Run: gh auth login && gh auth setup-git"
AUDITOR=claude
if command -v codex >/dev/null && timeout 180 codex exec -m "${CODEX_MODEL:-gpt-6-astra}" -c model_reasoning_effort="\"${CODEX_EFFORT:-high}\"" \
     -s read-only "Reply with the single word OK" 2>/dev/null | grep -q OK; then
  AUDITOR=codex
  if bwrap --ro-bind / / true 2>/dev/null; then status "Codex works: GPT audits, adds 2 ideas a day, and is the emergency researcher"
  else status "Codex works without a shell here (container blocks its sandbox): GPT audits and adds ideas from files the harness attaches; no emergency researcher"; fi
else
  status "Codex not available: Claude audits; no emergency researcher (optional, see docs/SETUP.md)"
fi
avail_gb=$(( $(df -Pk "${DATA_ROOT:-.}" 2>/dev/null | awk 'NR==2{print $4}') / 1048576 ))
STAGE2=1; [ "$avail_gb" -ge 40 ] || { STAGE2=0; status "only ${avail_gb} GB free (< 40): ImageNet-100 stage skipped for now"; }

# ---------- 1. environment + lock ----------
done_step setup || { say "setup"; bash setup.sh >>"$LOG" 2>&1 || fail "setup.sh failed (see $LOG)"; mark setup; }
bash agent/relock.sh >>"$LOG" 2>&1 || fail "tests failed during relock (see $LOG)"
tmux has-session -t sched 2>/dev/null || bash agent/start.sh sched >>"$LOG" 2>&1

# ---------- 2. GPU smoke (minutes) ----------
if ! done_step data; then   # download CIFAR-100 once, before parallel jobs could race on it
  $PY -c "from harness.data import load_split; load_split('train'); load_split('test')" >>"$LOG" 2>&1 || fail "CIFAR-100 download failed"
  mark data
fi
if ! done_step smoke; then
  say "GPU smoke"
  ids=$($PY -m harness.submit --smoke) || fail "smoke submit failed"
  $PY -m harness.submit --wait $ids | tee -a "$LOG" || fail "GPU smoke failed: see runs/<id>/error.txt"
  mark smoke; status "GPU smoke passed"
fi

# ---------- 3. teacher ----------
if [ ! -f state/teacher.json ]; then
  say "teacher fine-tune (CIFAR-100, DeiT-B, label smoothing 0)"
  $PY -m harness.finetune_teacher >>"$LOG" 2>&1 || fail "teacher fine-tune failed (see $LOG)"
fi
status "teacher val acc: $($PY -c 'import json;print(json.load(open("state/teacher.json"))["val_acc"])')"

# ---------- 4. bench + epochs ----------
if [ ! -f state/epochs.json ]; then
  say "bench (FLOP counter check + speed)"
  $PY -m harness.bench >>"$LOG" 2>&1; rc=$?
  [ $rc -eq 2 ] && fail "FLOP counter check failed (see state/bench.json flop_check). Do not continue until fixed"
  [ $rc -ne 0 ] && fail "bench failed (see $LOG)"
  $PY -m harness.boot epochs-from-bench >>"$LOG" 2>&1 || fail "could not derive epochs"
  status "epochs: $(cat state/epochs.json | tr -d '\n ')"
  git add state/bench.json state/teacher.json state/epochs.json agent/BOOT_STATUS.md
  git commit -qm "boot: teacher, bench, epochs" && git push -q
  bash agent/relock.sh >>"$LOG" 2>&1 || fail "relock failed"
  tmux kill-session -t sched 2>/dev/null; bash agent/start.sh sched >>"$LOG" 2>&1   # reload MAX_JOBS
fi

# ---------- 5. CIFAR gate (escalates by itself) ----------
# seed 0 -> + seeds 1, 2 (judged on the mean: one seed can miss by noise) -> 2x epochs -> 4x epochs.
# Longer training is the known lever for KD > CE (Beyer et al. 2022). Only after 4x does a human decide.
gate_ok() { $PY -m harness.report --gate >/dev/null 2>&1; }
run_gate() {  # $1 = id file name, rest = submit args; reuses runs that already exist
  local f="$S/$1"; shift
  [ -s "$f" ] || $PY -m harness.submit --gate "$@" > "$f" || fail "gate submit failed"
  $PY -m harness.submit --wait $(cat "$f") | tee -a "$LOG" || fail "a gate run failed: see runs/<id>/error.txt"
  status "CIFAR gate: $($PY -m harness.report --gate)"
}
for scale in 1 2 4; do
  gate_ok && break
  if [ "$scale" -gt 1 ]; then
    [ "$(cat "$S/gate_scale" 2>/dev/null || echo 1)" -ge "$scale" ] || {
      $PY -m harness.boot scale-epochs "$scale" >>"$LOG" 2>&1 || fail "could not scale epochs"
      git add state/epochs.json agent/BOOT_STATUS.md && git commit -qm "boot: gate failed, epochs x$scale" && git push -q
      bash agent/relock.sh >>"$LOG" 2>&1 || fail "relock failed"
      tmux kill-session -t sched 2>/dev/null; bash agent/start.sh sched >>"$LOG" 2>&1
      echo "$scale" > "$S/gate_scale"
      status "CIFAR gate failed at shorter training: epochs x$scale -> $(tr -d '\n ' < state/epochs.json)"
    }
  fi
  idf=gate_ids; [ "$scale" -gt 1 ] && idf="gate_ids_x$scale"
  gate_ok || run_gate "$idf" --seeds 0
  gate_ok || run_gate "${idf}_s12" --seeds 1,2
done
gate_ok || fail "CIFAR gate FAILED at 4x epochs on seeds 0-2: $($PY -m harness.report --gate). Fallback (human decision): DeiT-B teacher + pretrained student, docs/SETUP.md step 5"
status "CIFAR gate PASSED: $($PY -m harness.report --gate)"

# ---------- 6. baselines, stage 2, loop ----------
if ! done_step baselines; then
  $PY -m harness.submit --baselines > "$S/cifar_baseline_ids" || fail "baselines submit failed"
  mark baselines; status "CIFAR baselines queued ($(wc -l < "$S/cifar_baseline_ids") jobs)"
fi
$PY -m harness.report >/dev/null
git add -A && git commit -qm "boot: CIFAR gate passed, baselines queued" ; git push -q
if [ "$STAGE2" = 1 ] && ! tmux has-session -t stage2 2>/dev/null; then
  tmux new -d -s stage2 "bash agent/stage2_boot.sh"
  status "ImageNet-100 preparation started in background (tmux: stage2)"
fi
echo "$AUDITOR" > "$S/boot_auditor"
AUDITOR=$AUDITOR bash agent/start.sh >>"$LOG" 2>&1
status "research loop started (auditor: $AUDITOR). Your part from here: agent/QUEUE.md, agent/STOP"
git add agent/BOOT_STATUS.md && git commit -qm "boot: loop started" && git push -q
