#!/usr/bin/env bash
# HUMAN, once, after config.STAGE1 / STAGE2 changed (2026-10-31: search moves to CUB, Waterbirds becomes the check).
# Keeps every run (both datasets stay valid; runs at a superseded lr drop out of the tables by themselves), stops the
# loop and the stage scripts, forgets which gate/baseline steps were done, and re-runs bootstrap:
#   lr sweep on the new stage 1 -> relock -> stage-1 gate -> baselines -> loop, and stage 2 in the background.
#   git pull --rebase --autostash && bash agent/switch_stage.sh
set -u
cd "$(dirname "$0")/.."
S=agent/.state
touch agent/STOP
for s in agent boot stage2; do tmux kill-session -t "$s" 2>/dev/null; done
rm -f agent/STOP
rm -f "$S"/boot_baselines "$S"/gate_ids* "$S"/gate2_ids "$S"/stage1_baseline_ids "$S"/stage2_baseline_ids
S1=$(.venv/bin/python -c 'from harness import config as C; print(C.STAGE1)')
{ echo; echo "- $(date '+%Y-%m-%d %H:%M') SWITCH: stage 1 is now $S1 (decisions on mean accuracy; WGA diagnostic only)"; } >> agent/BOOT_STATUS.md
git add -A && git commit -qm "switch: stage 1 = $S1" ; git pull -q --rebase --autostash; git push -q
bash agent/bootstrap.sh
