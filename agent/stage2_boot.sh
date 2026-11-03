#!/usr/bin/env bash
# Stage 2 (config.STAGE2, the check dataset), started in the background by agent/bootstrap.sh, or by agent/loop.sh when
# config.STAGE2 names a dataset this script has not finished yet (2026-11-02: Waterbirds -> Aircraft). Safe to re-run.
#   data -> teacher (aim-lab-test-5's or fine-tune) + student lr -> relock (loop paused briefly)
#   -> stage-2 gate (final tier, one GPU slot) -> baselines at the budgets stage 1 found informative
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
S=agent/.state; LOG=agent/logs/stage2_boot.log
say()  { echo "[stage2 $(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
status() { echo "- $(date '+%Y-%m-%d %H:%M') [stage2] $*" >> agent/BOOT_STATUS.md; say "$*"; }
# $1: the dataset to set up (default config.STAGE2; agent/loop.sh also passes the held-out datasets, config.HOLDOUT)
S2="${1:-$($PY -c 'from harness import config as C; print(C.STAGE2)')}"
HOLD=$($PY -c "from harness import config as C; print(int('$S2' in C.HOLDOUT))")
# markers per dataset: agent/loop.sh starts this script again only while neither exists
fail() { status "STOPPED: $*"; touch "$S/stage2_failed_$S2"; exit 1; }
G2="$S/gate2_ids_$S2"; B2="$S/stage2_baseline_ids_$S2"
$PY -m harness.prepare_data --datasets "$S2" >>"$LOG" 2>&1 || fail "$S2 data preparation failed (see $LOG)"
TJ=$($PY -c "from harness.train import teacher_paths; print(teacher_paths('$S2')[1])")

if [ ! -f "$TJ" ]; then
  # DeiT-B fine-tuning needs the whole GPU (2026-11-02: OOM next to 4 CUB jobs) and writes state/recipe.json, a locked
  # file: stop the loop and new job starts first, wait for running jobs, and undo everything if it fails.
  status "$S2 teacher: pausing the loop and new job starts until the running jobs finish, then fine-tune (about 1-2 h)"
  touch agent/STOP "$S/sched_hold"
  while tmux has-session -t agent 2>/dev/null || ls queue/running/*.json >/dev/null 2>&1; do sleep 60; done
  if ! $PY -m harness.finetune_teacher --dataset "$S2" >>"$LOG" 2>&1; then
    git checkout -- state/recipe.json; rm -f "$TJ" "${TJ%.json}.pt"
    rm -f agent/STOP "$S/sched_hold"
    AUDITOR="$(cat "$S/boot_auditor" 2>/dev/null || echo claude)" bash agent/start.sh >>"$LOG" 2>&1
    fail "$S2 teacher failed (see $LOG); recipe restored, loop and scheduler resumed"
  fi
  status "$S2 teacher: $($PY -c "import json;d=json.load(open('$TJ'));print(d['source'], 'val acc', d['val_acc'])")"
  # teacher and recipe are locked: commit, relock, restart
  git add "$TJ" state/recipe.json && git commit -qm "stage2: $S2 teacher and student lr" && git push -q
  bash agent/relock.sh >>"$LOG" 2>&1 || fail "relock failed"
  rm -f agent/STOP "$S/sched_hold"
  AUDITOR="$(cat "$S/boot_auditor" 2>/dev/null || echo claude)" bash agent/start.sh >>"$LOG" 2>&1
  status "loop restarted after stage-2 relock"
fi

if ! $PY -m harness.report --gate --dataset "$S2" >/dev/null 2>&1; then
  [ -s "$G2" ] || $PY -m harness.submit --gate --dataset "$S2" > "$G2" || fail "gate submit failed"
  status "$S2 gate submitted (one GPU slot)"
  $PY -m harness.submit --wait $(cat "$G2") | tee -a "$LOG" || fail "a $S2 gate run failed"
fi
# Same rule as stage 1 (agent/bootstrap.sh): only KD not beating CE stops; MaskedKD@98 below Full is a finding.
if msg=$($PY -m harness.report --gate --dataset "$S2"); then
  status "$S2 gate PASSED: $msg"
else
  $PY -m harness.report --gate --kd-only --dataset "$S2" >/dev/null 2>&1 ||
    fail "$S2 gate FAILED: $msg (stage 2 stays closed; human decision)"
  status "$S2 gate: KD beats CE; MaskedKD@98 more than 0.5 below Full KD, kept as a finding, stage 2 goes on: $msg"
fi

if [ "$HOLD" = 1 ]; then
  status "$S2 is a held-out check: no baseline grid; the agent submits the frozen candidate and its references"
elif [ ! -s "$B2" ]; then
  [ -s "$S/stage1_baseline_ids" ] && $PY -m harness.submit --wait $(cat "$S/stage1_baseline_ids") >/dev/null
  keeps=$($PY -m harness.boot pick-keeps)
  $PY -m harness.submit --baselines --dataset "$S2" --keeps "$keeps" > "$B2" 2>>"$LOG" ||
    { rm -f "$B2"; fail "baselines submit failed: $(tail -1 "$LOG")"; }
  status "$S2 baselines queued at K=$keeps ($(wc -l < "$B2") jobs)"
fi
touch "$S/stage2_done_$S2"
