#!/usr/bin/env bash
# HUMAN ONLY. Run after you reviewed a change to harness/, tests/ or the agent prompts on this server.
# A change pushed to GitHub is adopted by `python -m harness.lock --check` itself (loop.sh runs it after every pull);
# this script is for changes made on the server, or the first lock.
# Runs the tests, then stores the new digest outside the repo (~/.maskedkd_lock).
set -eu
cd "$(dirname "$0")/.."
MASKEDKD_FAKE_DATA=1 .venv/bin/python -m pytest -q tests
d="$(.venv/bin/python -m harness.lock)"
echo "$d" > "${MASKEDKD_LOCK:-$HOME/.maskedkd_lock}"
echo "locked: $d"
echo "If agent/loop.sh is running, restart it so its revert point (LOCK_REF) is this commit."
