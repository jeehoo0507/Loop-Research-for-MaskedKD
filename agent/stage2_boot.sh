#!/usr/bin/env bash
# Stage 2 (ImageNet-100) preparation, started in the background by agent/bootstrap.sh. Safe to re-run.
#   download + convert -> teacher -> smoke (throughput) -> epochs by rule -> relock (loop paused briefly)
#   -> ImageNet-100 gate (one GPU slot, hours) -> baselines at the budgets CIFAR found informative
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
S=agent/.state; LOG=agent/logs/stage2_boot.log
say()  { echo "[stage2 $(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
status() { echo "- $(date '+%Y-%m-%d %H:%M') [stage2] $*" >> agent/BOOT_STATUS.md; say "$*"; }
fail() { status "STOPPED: $*"; exit 1; }

D="${DATA_ROOT:-data}/in100"
if [ ! -f "$D/meta.json" ]; then
  say "download + convert ImageNet-100 (about 8.4 GB in, 25 GB out)"
  $PY -m harness.prepare_stage2 >>"$LOG" 2>&1 || fail "prepare_stage2 failed (see $LOG)"
fi
if [ ! -f state/teacher_in100.json ]; then
  $PY -m harness.prepare_stage2 --teacher >>"$LOG" 2>&1 || fail "ImageNet-100 teacher failed (see $LOG)"
fi
status "ImageNet-100 ready; teacher val acc $($PY -c 'import json;print(json.load(open("state/teacher_in100.json"))["val_acc"])')"

if ! $PY -c 'import json,sys;sys.exit(0 if json.load(open("state/epochs.json"))["epochs"].get("in100") else 1)' 2>/dev/null; then
  ids=$($PY -m harness.submit --smoke --dataset in100) || fail "smoke submit failed"
  $PY -m harness.submit --wait $ids | tee -a "$LOG" || fail "ImageNet-100 smoke failed: see runs/<id>/error.txt"
  mk=$(echo "$ids" | grep maskedkd | head -1)
  $PY -m harness.boot epochs-in100 "$mk" >>"$LOG" 2>&1 || fail "could not derive ImageNet-100 epochs"
  status "ImageNet-100 epochs: $(tr -d '\n ' < state/epochs.json)"
  # the epochs file is locked: pause the loop between rounds, commit, relock, restart
  touch agent/STOP
  while tmux has-session -t agent 2>/dev/null; do sleep 30; done
  git add state/epochs.json state/teacher_in100.json && git commit -qm "stage2: ImageNet-100 teacher and epochs" && git push -q
  bash agent/relock.sh >>"$LOG" 2>&1 || fail "relock failed"
  rm -f agent/STOP
  AUDITOR="$(cat "$S/boot_auditor" 2>/dev/null || echo claude)" bash agent/start.sh >>"$LOG" 2>&1
  status "loop restarted after stage-2 relock"
fi

if ! $PY -m harness.report --gate --dataset in100 >/dev/null 2>&1; then
  [ -s "$S/gate2_ids" ] || $PY -m harness.submit --gate --dataset in100 > "$S/gate2_ids" || fail "gate submit failed"
  status "ImageNet-100 gate submitted (one GPU slot; long)"
  $PY -m harness.submit --wait $(cat "$S/gate2_ids") | tee -a "$LOG" || fail "an ImageNet-100 gate run failed"
  msg=$($PY -m harness.report --gate --dataset in100) || fail "ImageNet-100 gate FAILED: $msg (stage 2 stays closed; human decision)"
fi
status "ImageNet-100 gate PASSED"

if [ ! -s "$S/in100_baseline_ids" ]; then
  [ -s "$S/cifar_baseline_ids" ] && $PY -m harness.submit --wait $(cat "$S/cifar_baseline_ids") >/dev/null
  keeps=$($PY -m harness.boot pick-keeps)
  $PY -m harness.submit --baselines --dataset in100 --keeps "$keeps" > "$S/in100_baseline_ids" || fail "baselines submit failed"
  status "ImageNet-100 baselines queued at K=$keeps ($(wc -l < "$S/in100_baseline_ids") jobs)"
fi
