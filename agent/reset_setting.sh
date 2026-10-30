#!/usr/bin/env bash
# HUMAN, once, after the experiment setting in harness/config.py changed (teacher, student, recipe).
# Stops everything, moves every result of the old setting to archive/<time>/ (kept in git, checkpoints excluded),
# resets the agent's notes, and re-runs agent/bootstrap.sh from the smoke step (setup and downloads are kept).
#
#   git pull && bash agent/reset_setting.sh
set -u
cd "$(dirname "$0")/.."
S=agent/.state
A="archive/$(date -u +%Y%m%d_%H%M)"
echo "[reset] stopping loop, scheduler, stage 2 and bootstrap"
touch agent/STOP
for s in agent boot stage2 sched; do tmux kill-session -t "$s" 2>/dev/null; done
pkill -f "python -m harness" 2>/dev/null; sleep 3

echo "[reset] archiving the old setting to $A"
mkdir -p "$A/state"
for p in runs queue analysis_out RESULTS_TABLE.md figures PROGRESS.md RESULTS.md PROPOSALS.md DAILY.md AUDIT.md \
         agent/AUDIT_REQUEST.md agent/INVALID.md; do
  [ -e "$p" ] && mkdir -p "$A/$(dirname "$p")" && mv "$p" "$A/$p"
done
for p in state/teacher.pt state/teacher.json state/teacher_in100.pt state/teacher_in100.json state/bench.json state/epochs.json; do
  [ -e "$p" ] && mv "$p" "$A/$p"
done

printf '# PROGRESS\n\n(에이전트가 라운드마다 갱신)\n' > PROGRESS.md
printf '# RESULTS\n\n## 확정 (감사 PASS)\n\n## 잠정 (screen)\n\n## 실패한 방향\n\n## 분석에서 배운 것\n' > RESULTS.md
printf '# PROPOSALS\n\n' > PROPOSALS.md
printf '# AUDIT (agent/loop.sh가 씀)\n' > AUDIT.md
: > agent/AUDIT_REQUEST.md

# keep: environment, CIFAR download. redo: smoke (models changed), teacher, bench, gate, baselines, stage 2 steps
find "$S" -maxdepth 1 -type f ! -name boot_setup ! -name boot_data -delete 2>/dev/null
{ echo; echo "- $(date '+%Y-%m-%d %H:%M') RESET: setting changed (harness/config.py); old results in $A"; } >> agent/BOOT_STATUS.md
rm -f agent/STOP

git add -A && git commit -qm "reset: new setting (DeiT-B teacher, pretrained DeiT-Ti student); old results in $A" && git push -q
echo "[reset] done; starting bootstrap"
bash agent/bootstrap.sh
