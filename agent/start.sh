#!/usr/bin/env bash
# Start (or restart after a pod restart) the GPU scheduler and the agent loop in tmux. Idempotent.
#   bash agent/start.sh              # scheduler + loop
#   bash agent/start.sh sched        # scheduler only (for teacher/bench/gate phase)
set -u
cd "$(dirname "$0")/.."
mkdir -p agent/logs
command -v tmux >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq tmux; }
tmux has-session -t sched 2>/dev/null || tmux new -d -s sched '.venv/bin/python -m harness.scheduler 2>&1 | tee -a agent/logs/sched.log'
if [ "${1:-all}" = "all" ]; then
  tmux has-session -t agent 2>/dev/null || tmux new -d -s agent "AUDITOR=${AUDITOR:-claude} bash agent/loop.sh 2>&1 | tee -a agent/logs/loop_stdout.log"
fi
tmux ls
