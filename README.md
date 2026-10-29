# Loop Research for MaskedKD

해석가능성 기법으로 teacher 입력 토큰을 골라, MaskedKD(ECCV 2024)보다 **적은 teacher FLOPs로 같은 정확도**를 내는 방법을
무인 에이전트 루프로 탐색한다.

- 기준 논문: [MaskedKD](https://arxiv.org/abs/2302.10494), 코드 [effl-lab/MaskedKD](https://github.com/effl-lab/MaskedKD)
- 설정: CIFAR-100 224px (train 45k / val 5k / test 10k), teacher DeiT-S (ImageNet → fine-tune, 라벨 스무딩 0), student DeiT-Ti scratch
- 주 지표: Full KD 대비 val 1%p 이내를 유지하는 **최소 평균 teacher 토큰 수**. 비용은 학습 1스텝 전체 GFLOPs/img 실측(선택기 포함)

## 구조

| 경로 | 누가 | 내용 |
|---|---|---|
| `harness/` | 사람 (잠금) | 데이터(GPU 상주·GPU 증강), 학습, 손실, FLOPs, 평가, 작업 큐, 스케줄러, 보고서 |
| `tests/` | 사람 (잠금) | MaskedKD 공식 코드와의 동치성, 사기 방지 검사 |
| `selectors/` | 에이전트 | 선택기. 여기만 바꿀 수 있다 |
| `agent/` | 사람 | 루프, 지침, 감사 프롬프트, QUEUE(사람 지시) |
| `runs/<id>/` | harness | `metrics.json`(유일한 숫자 출처), `log.txt` |
| `RESULTS_TABLE.md` | harness | 결과 표와 주 지표 (자동 생성) |

## 사기 방지

1. **잠금:** `harness/`, `tests/`, 루프와 프롬프트의 sha256을 저장소 **밖**(`~/.maskedkd_lock`)에 저장한다. 에이전트가 바꾸면 라운드 후 되돌리고 `agent/INVALID.md`에 기록한다. 잠금이 어긋나면 스케줄러가 작업을 시작하지 않는다.
2. **숫자는 harness만 쓴다:** `metrics.json`에 harness 해시, 선택기 sha, teacher sha, git hash가 붙는다.
3. **FLOPs 실측:** `torch.utils.flop_counter`로 실제 스텝을 잰다. 선택기 안에서 몰래 돌린 연산도 잡힌다(테스트로 확인).
4. **선택기 검사:** 정적 검사(허용 import만, 파일·teacher·라벨·`.backward()` 금지, META 필수)와 실행 중 검사(예산 초과, 남은 gradient).
5. **test 격리:** 에이전트는 val만 쓴다. `harness.test_eval`은 사람이 마지막에 한 번만 쓴다.
6. **감사:** 주장은 `agent/AUDIT_REQUEST.md`에 올리고, 별도 세션이 읽기 전용으로 검토한다. 기본은 Claude 세션이고, `AUDITOR=codex`면 GPT(Codex CLI)가 맡는다. GPT는 **선택 사항**이다.

## 서버에서 처음부터 (A5000 1장)

**자세한 단계별 가이드와 GPT 감사관 붙이는 법: [docs/SETUP.md](docs/SETUP.md)**

```bash
git clone https://github.com/jeehoo0507/Loop-Research-for-MaskedKD.git && cd Loop-Research-for-MaskedKD
bash setup.sh                      # venv + torch cu124 + CPU 테스트
gh auth login && gh auth setup-git # push 권한
claude                             # 한 번 로그인하고 종료
# (선택) GPT 감사관: npm i -g @openai/codex && codex login
```

### 1. 잠금 → teacher → 속도 측정

```bash
bash agent/relock.sh                               # 테스트 통과 후 잠금 저장
.venv/bin/python -m harness.finetune_teacher       # state/teacher.pt, teacher.json (val 정확도·sha256)
.venv/bin/python -m harness.bench                  # state/bench.json + 권장 에폭·동시 작업 수
```

`harness/config.py`의 `TIER_EPOCHS`에서 `screen`(1회 ≤ 1시간)과 `final`을 bench 권장값으로 채우고, `MAX_JOBS`를 맞춘 뒤:

```bash
git add -A && git commit -m "set tier epochs from bench" && git push
bash agent/relock.sh
```

### 2. 재현 관문

```bash
bash agent/start.sh sched                          # GPU 스케줄러만 tmux로 시작
.venv/bin/python -m harness.submit --gate          # CE, Full KD, MaskedKD@98 (seed 0, screen)
.venv/bin/python -m harness.submit --status        # 진행 확인
.venv/bin/python -m harness.report --gate          # PASS여야 루프를 시작할 수 있다
```

기준: Full KD ≥ CE + 1%p, |MaskedKD@98 − Full KD| ≤ 0.5%p.
실패하면 루프를 시작하지 않는다. 대안 설정(teacher DeiT-B, student ImageNet 사전학습)으로 `harness/config.py`를 고치고 다시 relock한다.

### 3. 루프 시작

```bash
git add -A && git commit -m "gate passed" && git push
bash agent/start.sh                                # 스케줄러 + 에이전트 루프 (터미널 닫아도 계속)
# GPT 감사관을 쓰려면: AUDITOR=codex bash agent/start.sh
```

파드가 재시작되면 `bash agent/start.sh`만 다시 실행한다. 큐의 작업은 체크포인트에서 이어서 돈다.

## 사람이 관여하기

| 하고 싶은 것 | 방법 |
|---|---|
| 지시 | `agent/QUEUE.md`에 적고 push. 다음 라운드가 최우선으로 읽는다 |
| 멈춤 | `touch agent/STOP` (현재 라운드 후 종료), 즉시: `tmux kill-session -t agent` |
| 모델 강제 | `echo claude-fable-5-1 > agent/MODEL` (지우면 기본 Opus) |
| 보기 | `tmux attach -t agent`, `tail -f agent/logs/loop.log`, `cat RESULTS_TABLE.md` |
| harness 수정 | 서버에서 고치고 commit → `bash agent/relock.sh` → 루프 재시작 |
| 최종 test | `.venv/bin/python -m harness.test_eval --runs <id> ...` (사람만) |

- 루프 종료: `END_DATE`(기본 2026-11-05) 또는 `agent/STOP`.
- 라운드 간격 15분, 기본 모델 Opus. 하루의 첫 라운드와 `STUCK` 이후 라운드는 `MODEL_HARD`를 쓴다.
