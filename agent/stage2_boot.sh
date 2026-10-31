#!/usr/bin/env bash
# Stage 2 (config.STAGE2, the check dataset), started in the background by agent/bootstrap.sh. Safe to re-run.
#   teacher (aim-lab-test-5's or fine-tune) + student lr -> relock (loop paused briefly)
#   -> stage-2 gate (final tier, one GPU slot) -> baselines at the budgets stage 1 found informative
# The data were prepared by bootstrap (harness.prepare_data does both datasets).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
S=agent/.state; LOG=agent/logs/stage2_boot.log
say()  { echo "[stage2 $(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
status() { echo "- $(date '+%Y-%m-%d %H:%M') [stage2] $*" >> agent/BOOT_STATUS.md; say "$*"; }
fail() { status "STOPPED: $*"; exit 1; }
S2=$($PY -c 'from harness import config as C; print(C.STAGE2)')
TJ=$($PY -c "from harness.train import teacher_paths; print(teacher_paths('$S2')[1])")

if [ ! -f "$TJ" ]; then
  say "$S2 teacher: import aim-lab-test-5's or fine-tune (about 1 h)"
  $PY -m harness.finetune_teacher --dataset "$S2" >>"$LOG" 2>&1 || fail "$S2 teacher failed (see $LOG)"
  status "$S2 teacher: $($PY -c "import json;d=json.load(open('$TJ'));print(d['source'], 'val acc', d['val_acc'])")"
  # teacher and recipe are locked: pause the loop between rounds, commit, relock, restart
  touch agent/STOP
  while tmux has-session -t agent 2>/dev/null; do sleep 30; done
  git add "$TJ" state/recipe.json && git commit -qm "stage2: $S2 teacher and student lr" && git push -q
  bash agent/relock.sh >>"$LOG" 2>&1 || fail "relock failed"
  rm -f agent/STOP
  AUDITOR="$(cat "$S/boot_auditor" 2>/dev/null || echo claude)" bash agent/start.sh >>"$LOG" 2>&1
  status "loop restarted after stage-2 relock"
fi

if ! $PY -m harness.report --gate --dataset "$S2" >/dev/null 2>&1; then
  [ -s "$S/gate2_ids" ] || $PY -m harness.submit --gate --dataset "$S2" > "$S/gate2_ids" || fail "gate submit failed"
  status "$S2 gate submitted (one GPU slot)"
  $PY -m harness.submit --wait $(cat "$S/gate2_ids") | tee -a "$LOG" || fail "a $S2 gate run failed"
fi
# Same rule as stage 1 (agent/bootstrap.sh): only KD not beating CE stops; MaskedKD@98 below Full is a finding.
if msg=$($PY -m harness.report --gate --dataset "$S2"); then
  status "$S2 gate PASSED: $msg"
else
  $PY -m harness.report --gate --kd-only --dataset "$S2" >/dev/null 2>&1 ||
    fail "$S2 gate FAILED: $msg (stage 2 stays closed; human decision)"
  status "$S2 gate: KD beats CE; MaskedKD@98 more than 0.5 below Full KD, kept as a finding, stage 2 goes on: $msg"
fi

if [ ! -s "$S/stage2_baseline_ids" ]; then
  [ -s "$S/stage1_baseline_ids" ] && $PY -m harness.submit --wait $(cat "$S/stage1_baseline_ids") >/dev/null
  keeps=$($PY -m harness.boot pick-keeps)
  $PY -m harness.submit --baselines --dataset "$S2" --keeps "$keeps" > "$S/stage2_baseline_ids" || fail "baselines submit failed"
  status "$S2 baselines queued at K=$keeps ($(wc -l < "$S/stage2_baseline_ids") jobs)"
fi
