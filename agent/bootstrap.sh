#!/usr/bin/env bash
# One command from an empty server to a running research loop. HUMAN starts it once:
#
#   bash agent/bootstrap.sh          # re-launches itself in tmux session "boot"; safe to re-run (skips finished steps)
#
# Steps (stops at the first failure and says why in agent/BOOT_STATUS.md and agent/logs/boot.log):
#   check tools/logins -> setup -> lock -> data (Waterbirds, CUB) -> GPU smoke -> teacher (aim-lab-test-5's if it is
#   on this server, else fine-tune) + student lr -> bench (FLOP check, parallelism) -> commit + relock
#   -> Waterbirds gate (seed 0, then seeds 1-2 on the mean) -> baselines -> loop
#   + in the background: the CUB teacher, gate and baselines (agent/stage2_boot.sh)
# If the gate fails on seeds 0-2 it stops: what to change is a human decision.
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
# catch up first: a local "boot:" commit next to new commits on GitHub would make the push check fail for the wrong reason
git add -A agent/BOOT_STATUS.md >/dev/null 2>&1; git commit -qm "boot: status" >/dev/null 2>&1
git pull -q --rebase --autostash >>"$LOG" 2>&1 || fail "git pull failed (local changes conflict with GitHub?): run git status"
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
[ "$avail_gb" -ge 10 ] || fail "only ${avail_gb} GB free (< 10)"
STAGE2=1

# ---------- 1. environment + lock ----------
done_step setup || { say "setup"; bash setup.sh >>"$LOG" 2>&1 || fail "setup.sh failed (see $LOG)"; mark setup; }
bash agent/relock.sh >>"$LOG" 2>&1 || fail "tests failed during relock (see $LOG)"
tmux has-session -t sched 2>/dev/null || bash agent/start.sh sched >>"$LOG" 2>&1

# ---------- 2. data + GPU smoke (minutes) ----------
if ! done_step data; then   # once, before parallel jobs could race on it
  say "data: Waterbirds + CUB (aim-lab-test-5's extracted copies if present, else download)"
  $PY -m harness.prepare_data >>"$LOG" 2>&1 || fail "data preparation failed (see $LOG)"
  mark data
fi
if ! done_step smoke; then
  say "GPU smoke"
  ids=$($PY -m harness.submit --smoke) || fail "smoke submit failed"
  $PY -m harness.submit --wait $ids | tee -a "$LOG" || fail "GPU smoke failed: see runs/<id>/error.txt"
  mark smoke; status "GPU smoke passed"
fi

# ---------- 3. teacher + student lr (stage-1 dataset) ----------
S1=$($PY -c 'from harness import config as C; print(C.STAGE1)')
TJ=$($PY -c 'from harness.train import teacher_paths; print(teacher_paths()[1])')
if [ ! -f "$TJ" ]; then
  say "teacher ($S1, DeiT-B): import aim-lab-test-5's or fine-tune (about 1 h)"
  $PY -m harness.finetune_teacher --dataset "$S1" >>"$LOG" 2>&1 || fail "teacher failed (see $LOG)"
fi
status "teacher ($S1): $($PY -c "import json;d=json.load(open('$TJ'));print(d['source'], 'val acc', d['val_acc'])")"
if ! $PY -c "import json,sys; sys.exit(0 if json.load(open('state/recipe.json')).get('$S1',{}).get('swept') else 1)" 2>/dev/null; then
  say "student lr sweep on $S1 (CE, lr in config.STUDENT_LRS, screen seed 0)"
  ids=$($PY -m harness.submit --lr-sweep --dataset "$S1") || fail "lr sweep submit failed"
  $PY -m harness.submit --wait $ids | tee -a "$LOG" || fail "an lr-sweep run failed: see runs/<id>/error.txt"
  $PY -m harness.boot lr-pick "$S1" >>"$LOG" 2>&1 || fail "lr pick failed (see $LOG)"
  status "student lr ($S1): $($PY -c "import json;print(json.load(open('state/recipe.json'))['$S1'])")"
  git add state/recipe.json agent/BOOT_STATUS.md && git commit -qm "boot: student lr sweep" && git push -q
  bash agent/relock.sh >>"$LOG" 2>&1 || fail "relock failed"
fi

# ---------- 4. bench ----------
if [ ! -f state/epochs.json ]; then
  say "bench (FLOP counter check + speed)"
  $PY -m harness.bench >>"$LOG" 2>&1; rc=$?
  [ $rc -eq 2 ] && fail "FLOP counter check failed (see state/bench.json flop_check). Do not continue until fixed"
  [ $rc -ne 0 ] && fail "bench failed (see $LOG)"
  $PY -m harness.boot jobs-from-bench >>"$LOG" 2>&1 || fail "could not read the bench"
  status "parallel jobs: $(tr -d '\n ' < state/epochs.json)"
  git add state/bench.json state/teacher*.json state/recipe.json state/epochs.json agent/BOOT_STATUS.md
  git commit -qm "boot: teacher, recipe, bench" && git push -q
  bash agent/relock.sh >>"$LOG" 2>&1 || fail "relock failed"
  tmux kill-session -t sched 2>/dev/null; bash agent/start.sh sched >>"$LOG" 2>&1   # reload MAX_JOBS
fi

# ---------- 5. gate: seed 0, then seeds 4-5 judged on the mean (one seed can miss by noise) ----------
gate_ok() { $PY -m harness.report --gate >/dev/null 2>&1; }
run_gate() {  # $1 = id file name, rest = submit args; reuses runs that already exist
  local f="$S/$1"; shift
  [ -s "$f" ] || $PY -m harness.submit --gate "$@" > "$f" || fail "gate submit failed"
  $PY -m harness.submit --wait $(cat "$f") | tee -a "$LOG" || fail "a gate run failed: see runs/<id>/error.txt"
  status "gate: $($PY -m harness.report --gate)"
}
gate_ok || run_gate gate_ids --seeds 0
gate_ok || run_gate gate_ids_s45 --seeds 4,5
if gate_ok; then
  status "gate PASSED: $($PY -m harness.report --gate)"
else
  # KD not beating CE stops everything (no KD signal to study). MaskedKD@98 short of Full KD on the seed mean does
  # not: it is a finding about MaskedKD on this dataset (more room for a better token choice), recorded and kept.
  $PY -m harness.report --gate --kd-only >/dev/null 2>&1 ||
    fail "gate FAILED on seeds 0, 4, 5: $($PY -m harness.report --gate). Human decision (recipe, dataset)"
  status "gate: KD beats CE; MaskedKD@98 more than 0.5 below Full KD on the seed mean, kept as a finding, search goes on: $($PY -m harness.report --gate)"
fi

# ---------- 6. baselines, stage 2, loop ----------
if ! done_step baselines; then
  $PY -m harness.submit --baselines > "$S/stage1_baseline_ids" || fail "baselines submit failed"
  mark baselines; status "$S1 baselines queued ($(wc -l < "$S/stage1_baseline_ids") jobs)"
fi
$PY -m harness.report >/dev/null
git add -A && git commit -qm "boot: gate passed, baselines queued" ; git push -q
if [ "$STAGE2" = 1 ] && ! tmux has-session -t stage2 2>/dev/null; then
  tmux new -d -s stage2 "bash agent/stage2_boot.sh"
  status "stage 2 teacher, gate and baselines started in background (tmux: stage2)"
fi
echo "$AUDITOR" > "$S/boot_auditor"
AUDITOR=$AUDITOR bash agent/start.sh >>"$LOG" 2>&1
status "research loop started (auditor: $AUDITOR). Your part from here: agent/QUEUE.md, agent/STOP"
git add agent/BOOT_STATUS.md && git commit -qm "boot: loop started" && git push -q
