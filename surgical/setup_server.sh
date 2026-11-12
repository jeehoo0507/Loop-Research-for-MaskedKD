#!/usr/bin/env bash
# A server, new or old, made ready for the surgical study. Safe to run again. It never starts the agent loop.
#
#   sudo apt-get update -qq && sudo apt-get install -y -qq git tmux curl
#   git clone https://github.com/jeehoo0507/Loop-Research-for-MaskedKD.git && cd Loop-Research-for-MaskedKD
#   git checkout surgical && bash surgical/setup_server.sh          # set up, then stop
#   bash surgical/setup_server.sh --go                                # set up, then smoke -> teachers -> all runs
#
# Steps: GPU and disk check -> Python env (setup.sh: uv, Python 3.11, torch cu124, CPU tests) -> lock ->
# GPU scheduler in tmux -> what this server has (teachers, data). Then it prints what to run next.
set -eu
cd "$(dirname "$0")/.."
say() { echo "[setup $(date '+%H:%M:%S')] $*"; }

command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || { say "no GPU visible (nvidia-smi failed)"; exit 1; }
command -v tmux >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq tmux; }
free_gb=$(df -Pk . | awk 'NR==2 {print int($4/1048576)}')
say "free disk here: ${free_gb} GiB (the four default datasets need about 15 GiB)"

if [ -x .venv/bin/python ] && .venv/bin/python -c "import torch, timm, pyarrow" 2>/dev/null; then
  say "Python env: already there"
else
  say "Python env: bash setup.sh (about 10 minutes)"
  bash setup.sh
fi
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'torch does not see the GPU'"

touch agent/STOP   # the agent loop must not come back while this study runs (the scheduler revives it otherwise)
if tmux has-session -t agent 2>/dev/null; then
  say "the agent loop is running in tmux session 'agent'. This study runs without it: touch agent/STOP, wait, run this again"
  exit 1
fi
if [ -n "$(git status --porcelain -- harness tests)" ]; then
  say "harness/ or tests/ have uncommitted changes: commit or discard them first (the lock is taken from committed code)"
  exit 1
fi
say "lock: bash agent/relock.sh (runs the CPU tests)"
bash agent/relock.sh
# a scheduler started before this checkout would keep the old code: restart it
tmux kill-session -t sched 2>/dev/null || true
bash agent/start.sh sched >/dev/null
say "GPU scheduler running (tmux attach -t sched)"

.venv/bin/python -m surgical.smoke --env-only
if [ "${1:-}" = "--go" ]; then
  if tmux has-session -t surgical 2>/dev/null; then
    say "autorun is already running (tmux attach -t surgical)"
  else
    tmux new -d -s surgical '.venv/bin/python -m surgical.autorun 2>&1 | tee -a surgical/autorun.log'
    say "autorun started in tmux 'surgical': smoke -> teachers -> gate + grid. Watch: tail -f surgical/STATUS.md"
  fi
  exit 0
fi
cat <<'NEXT'

Next:
  .venv/bin/python -m surgical.smoke        # downloads, speed, time estimate, one end-to-end job per dataset
  cat surgical_smoke_report.md
Then, per dataset whose teacher is not here (see the table above):
  .venv/bin/python -m harness.finetune_teacher --dataset <dataset>     # whole GPU, no jobs running
  git add state/ && git commit -m "surgical: <dataset> teacher" && bash agent/relock.sh
and surgical/README.md from "the KD-gain gate" on.
NEXT
