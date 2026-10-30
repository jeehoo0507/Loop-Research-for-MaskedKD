#!/usr/bin/env bash
# Stage 2 (CUB-200-2011), started in the background by agent/bootstrap.sh. Safe to re-run.
#   teacher (aim-lab-test-5's or fine-tune) + student lr -> relock (loop paused briefly)
#   -> CUB gate (final seed 1, one GPU slot) -> baselines at the budgets Waterbirds found informative
# The data were prepared by bootstrap (harness.prepare_data does both datasets).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
S=agent/.state; LOG=agent/logs/stage2_boot.log
say()  { echo "[stage2 $(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
status() { echo "- $(date '+%Y-%m-%d %H:%M') [stage2] $*" >> agent/BOOT_STATUS.md; say "$*"; }
fail() { status "STOPPED: $*"; exit 1; }

if [ ! -f state/teacher_cub.json ]; then
  say "CUB teacher: import aim-lab-test-5's or fine-tune (about 1 h)"
  $PY -m harness.finetune_teacher --dataset cub >>"$LOG" 2>&1 || fail "CUB teacher failed (see $LOG)"
  status "CUB teacher: $($PY -c 'import json;d=json.load(open("state/teacher_cub.json"));print(d["source"], "val acc", d["val_acc"])')"
  # teacher and recipe are locked: pause the loop between rounds, commit, relock, restart
  touch agent/STOP
  while tmux has-session -t agent 2>/dev/null; do sleep 30; done
  git add state/teacher_cub.json state/recipe.json && git commit -qm "stage2: CUB teacher and student lr" && git push -q
  bash agent/relock.sh >>"$LOG" 2>&1 || fail "relock failed"
  rm -f agent/STOP
  AUDITOR="$(cat "$S/boot_auditor" 2>/dev/null || echo claude)" bash agent/start.sh >>"$LOG" 2>&1
  status "loop restarted after stage-2 relock"
fi

if ! $PY -m harness.report --gate --dataset cub >/dev/null 2>&1; then
  [ -s "$S/gate2_ids" ] || $PY -m harness.submit --gate --dataset cub > "$S/gate2_ids" || fail "gate submit failed"
  status "CUB gate submitted (one GPU slot)"
  $PY -m harness.submit --wait $(cat "$S/gate2_ids") | tee -a "$LOG" || fail "a CUB gate run failed"
  msg=$($PY -m harness.report --gate --dataset cub) || fail "CUB gate FAILED: $msg (stage 2 stays closed; human decision)"
fi
status "CUB gate PASSED"

if [ ! -s "$S/stage2_baseline_ids" ]; then
  [ -s "$S/stage1_baseline_ids" ] && $PY -m harness.submit --wait $(cat "$S/stage1_baseline_ids") >/dev/null
  keeps=$($PY -m harness.boot pick-keeps)
  $PY -m harness.submit --baselines --dataset cub --keeps "$keeps" > "$S/stage2_baseline_ids" || fail "baselines submit failed"
  status "CUB baselines queued at K=$keeps ($(wc -l < "$S/stage2_baseline_ids") jobs)"
fi
