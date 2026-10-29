#!/usr/bin/env bash
# Unattended research loop. Run inside tmux (see README.md) so it survives a closed terminal.
#
#   bash agent/loop.sh                 # researcher only (auditor = claude, read-only, separate session)
#   AUDITOR=codex bash agent/loop.sh   # GPT (Codex CLI) audits claims and adds 2 ideas a day
#   AUDITOR=none  bash agent/loop.sh   # no audit; claims stay UNAUDITED
#
# Each round is a fresh `claude -p` session. Continuity lives in files: PROGRESS.md, DAILY.md,
# RESULTS.md, PROPOSALS.md, RESULTS_TABLE.md, runs/, git log. Human controls: agent/QUEUE.md (instructions,
# read first every round), agent/STOP (exit after this round), agent/MODEL (force a model id).
set -u
cd "$(dirname "$0")/.."
END_DATE="${END_DATE:-2026-11-05}"        # no new rounds on/after this date
ROUND_SLEEP="${ROUND_SLEEP:-900}"         # seconds between rounds (screen jobs take ~1 h)
MODEL_DEFAULT="${MODEL:-claude-opus-5-5}"
MODEL_HARD="${MODEL_HARD:-claude-opus-5-5}"
AUDITOR="${AUDITOR:-claude}"              # claude | codex | none
CODEX_CMD="${CODEX_CMD:-codex exec -s read-only}"
REMOTE="${REMOTE:-origin}"
PY=.venv/bin/python
LOCKED=(harness tests agent/loop.sh agent/relock.sh agent/AGENT_PROMPT.md agent/AUDIT_PROMPT.md requirements.txt)

pick_model() {
  if [ -s agent/MODEL ]; then cat agent/MODEL; return; fi
  if [ ! -f agent/.last_day ] || [ "$(cat agent/.last_day)" != "$(date +%F)" ]; then echo "$MODEL_HARD"; return; fi
  if grep -q '^STUCK' PROGRESS.md 2>/dev/null || [ "${last_rc:-0}" != "0" ]; then echo "$MODEL_HARD"; return; fi
  echo "$MODEL_DEFAULT"
}

mkdir -p agent/logs
if ! $PY -m harness.lock --check; then
  echo "[loop] harness lock mismatch at start. A human must review the harness and run: bash agent/relock.sh"; exit 1
fi
if ! $PY -m harness.report --gate >/dev/null; then
  echo "[loop] reproduction gate not passed (python -m harness.report --gate). Loop not started."; exit 1
fi
LOCK_REF="$(git rev-parse HEAD)"

ROUND_PROMPT='이 세션은 무인 실행이다. 사람에게 질문하거나 확인을 요청하지 마라. 결정이 필요하면 가장 보수적인 선택을 하고 PROGRESS.md에 "결정:"으로 이유와 함께 기록하라. 같은 문제로 두 번 막히면 PROGRESS.md 첫 줄에 "STUCK: <문제>"를 쓰고 다른 작업으로 넘어가라. 먼저 agent/QUEUE.md(사람 지시, 최우선), git log -15, PROGRESS.md, RESULTS_TABLE.md, RESULTS.md, PROPOSALS.md, AUDIT.md를 읽어라. 그다음 agent/AGENT_PROMPT.md의 목적과 규칙에 따라 이어서 작업하라. 라운드 끝에 반드시 PROGRESS.md 갱신, git commit, git push.'
ALLOWED="Read,Edit,Write,Glob,Grep,Bash($PY -m harness.submit:*),Bash($PY -m harness.report:*),Bash($PY -m harness.compare:*),Bash($PY -m harness.trysel:*),Bash($PY -m harness.lock:*),Bash(git:*),Bash(ls:*),Bash(cat:*),Bash(head:*),Bash(tail:*),Bash(wc:*),Bash(grep:*),Bash(nvidia-smi:*),Bash(date:*)"
AUDIT_ALLOWED="Read,Glob,Grep,Bash($PY -m harness.report:*),Bash($PY -m harness.lock:*),Bash($PY -m harness.compare:*),Bash(git log:*),Bash(git show:*),Bash(git diff:*),Bash(ls:*),Bash(cat:*),Bash(head:*),Bash(tail:*),Bash(grep:*)"

run_audit() {  # appends verdicts to AUDIT.md for whatever is in agent/AUDIT_REQUEST.md
  [ -s agent/AUDIT_REQUEST.md ] || return 0
  local ts out; ts="$(date +%Y%m%d_%H%M%S)"; out="agent/logs/audit_${ts}.log"
  if [ "$AUDITOR" = "none" ]; then
    { echo; echo "## $(date '+%F %H:%M') UNAUDITED (AUDITOR=none)"; cat agent/AUDIT_REQUEST.md; } >> AUDIT.md
  else
    local prompt; prompt="$(cat agent/AUDIT_PROMPT.md)

--- AUDIT_REQUEST.md ---
$(cat agent/AUDIT_REQUEST.md)"
    if [ "$AUDITOR" = "codex" ]; then
      $CODEX_CMD "$prompt" > "$out" 2>&1
    else
      claude -p "$prompt" --model "$MODEL_HARD" --allowedTools "$AUDIT_ALLOWED" --output-format text > "$out" 2>&1
    fi
    { echo; echo "## $(date '+%F %H:%M') audit by $AUDITOR"; echo; echo "### request"; cat agent/AUDIT_REQUEST.md;
      echo; echo "### verdict"; grep -E '^(VERDICT|CLAIM|REASON)' "$out" || tail -40 "$out"; } >> AUDIT.md
  fi
  cat agent/AUDIT_REQUEST.md >> agent/logs/audited_requests.md
  : > agent/AUDIT_REQUEST.md
}

daily_ideas() {  # Codex only: 2 independent ideas a day, without reading the researcher's notes
  [ "$AUDITOR" = "codex" ] || return 0
  [ "$(cat agent/.ideas_day 2>/dev/null)" = "$(date +%F)" ] && return 0
  local out="agent/logs/ideas_$(date +%Y%m%d).log"
  $CODEX_CMD "$(cat agent/IDEAS_PROMPT.md)" > "$out" 2>&1
  { echo; echo "## $(date +%F) ideas from Codex (independent)"; sed -n '/^IDEA/,$p' "$out" | head -30; } >> PROPOSALS.md
  date +%F > agent/.ideas_day
}

round=0
while true; do
  if [ -f agent/STOP ]; then echo "[loop] STOP file found, exiting"; break; fi
  if [[ ! "$(date +%F)" < "$END_DATE" ]]; then echo "[loop] reached $END_DATE, exiting"; break; fi
  round=$((round + 1))
  ts="$(date +%Y%m%d_%H%M%S)"
  log="agent/logs/round_${ts}.log"
  git pull --rebase "$REMOTE" main >>"$log" 2>&1 || true
  if ! $PY -m harness.lock --check >>"$log" 2>&1; then
    # locked files changed by a pull: a human edited the harness without relocking on this server
    echo "[loop] lock mismatch after pull; pausing until a human runs agent/relock.sh" | tee -a agent/logs/loop.log
    sleep "$ROUND_SLEEP"; continue
  fi
  MODEL="$(pick_model)"
  echo "[loop] round $round start $ts model=$MODEL auditor=$AUDITOR" | tee -a agent/logs/loop.log
  date +%F > agent/.last_day
  # auditor switch without restarting: echo codex > ~/.maskedkd_auditor (outside the repo, so the agent can't flip it)
  if [ -s "$HOME/.maskedkd_auditor" ]; then AUDITOR="$(tr -d '[:space:]' < "$HOME/.maskedkd_auditor")"; fi
  daily_ideas
  claude -p "$ROUND_PROMPT" \
    --model "$MODEL" \
    --permission-mode acceptEdits \
    --allowedTools "$ALLOWED" \
    --output-format text >>"$log" 2>&1
  rc=$?; last_rc=$rc
  echo "[loop] round $round end rc=$rc $(date +%Y%m%d_%H%M%S)" | tee -a agent/logs/loop.log
  if ! $PY -m harness.lock --check >>"$log" 2>&1; then
    git checkout "$LOCK_REF" -- "${LOCKED[@]}" >>"$log" 2>&1
    git clean -fdq -- harness tests >>"$log" 2>&1   # files the agent added under locked dirs
    { echo; echo "## round $round ($ts): locked files were modified by the agent and reverted to $LOCK_REF";
      echo "Jobs submitted in this round need re-checking."; } >> agent/INVALID.md
    echo "[loop] TAMPER: reverted locked files" | tee -a agent/logs/loop.log
  fi
  $PY -m harness.report >/dev/null 2>>"$log" || true
  run_audit
  git add -A . >>"$log" 2>&1
  git diff --cached --quiet || git commit -q -m "agent: round $round autosave ($ts)" >>"$log" 2>&1
  git push "$REMOTE" HEAD:main >>"$log" 2>&1 || true
  sleep "$ROUND_SLEEP"
done
