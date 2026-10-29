# 세팅 가이드 (A5000 서버, 처음부터)

## 빠른 시작: 명령 한 줄

빈 서버에서 로그인 세 가지만 하고 bootstrap을 실행하면, 루프 시작까지 혼자 간다.

```bash
sudo apt-get update -qq && sudo apt-get install -y -qq tmux gh git
gh auth login && gh auth setup-git                 # GitHub push 권한
claude                                              # 한 번 로그인하고 /exit (Claude Code가 없으면 먼저 설치)
# (선택) GPT: Node.js + sudo npm i -g @openai/codex && codex login --device-auth
git clone https://github.com/jeehoo0507/Loop-Research-for-MaskedKD.git && cd Loop-Research-for-MaskedKD
bash agent/bootstrap.sh
```

bootstrap은 tmux 세션 `boot` 안에서 아래를 순서대로 하고, 실패하면 그 자리에서 멈춘다.
진행 상황은 `agent/BOOT_STATUS.md`(GitHub에도 push됨)와 `tail -f agent/logs/boot.log`로 본다.

| 단계 | 내용 | 시간 (대략) |
|---|---|---|
| 확인 | GPU, claude, git push 권한, Codex(있으면 GPT 감사·비상 연구자), 디스크 40GB | 1분 |
| 환경 | `setup.sh` + 잠금 | 10분 |
| 스모크 | 모든 방법을 30스텝씩 | 몇 분 |
| teacher | CIFAR-100 fine-tune | 측정 전 |
| bench | FLOPs 계량기 검사 + 속도 → 에폭 자동 결정(screen = 1시간, final = 2배) → 잠금 | 15분 |
| CIFAR 관문 | CE, Full KD, MaskedKD@98 | screen 1회 시간 |
| 시작 | 기준선 20개 제출, ImageNet-100 준비(백그라운드), 루프 시작 | — |

- **관문이 실패하면** 루프를 시작하지 않고 멈춘다. 대안 설정(아래 5단계)은 네가 정한다.
- 중간에 멈췄으면 원인을 고치고 `bash agent/bootstrap.sh`를 다시 실행한다. 끝난 단계는 건너뛴다.
- ImageNet-100 쪽(`agent/stage2_boot.sh`, tmux `stage2`)은 받기·변환 → teacher → 스모크로 에폭 결정(1회 12시간 이내) → 잠금(루프 잠깐 멈춤) → 관문 → 기준선 순서로 혼자 진행한다.

아래는 bootstrap이 하는 일을 손으로 할 때의 단계별 설명이다.

순서대로 따라 하면 된다. 각 단계 끝의 **확인**이 통과해야 다음으로 간다.
모든 명령은 저장소 폴더(`~/Loop-Research-for-MaskedKD`) 안에서 실행한다.

| 단계 | 하는 일 | 걸리는 시간 (대략) |
|---|---|---|
| 0 | 서버 도구 설치, 로그인 | 10분 |
| 1 | 저장소와 Python 환경 | 10분 |
| 2 | 잠금 | 1분 |
| 3 | teacher fine-tune | 측정 전 (bench 후 판단) |
| 4 | 속도 측정 → 에폭 결정 | 15분 |
| 5 | 재현 관문 (CE, Full KD, MaskedKD@98) | screen 1회 시간 |
| 6 | 루프 시작 | 1분 |
| (선택) | GPT 감사관 붙이기 | 10분 |

---

## 0. 서버 도구 (Coder 워크스페이스)

```bash
sudo apt-get update -qq && sudo apt-get install -y -qq tmux gh git
gh auth login            # GitHub.com → HTTPS → 브라우저 또는 코드 입력
gh auth setup-git        # git push가 gh 인증을 쓰게
claude                   # 한 번 열어 로그인만 하고 /exit
```

파드가 재시작되면 tmux와 gh가 지워질 수 있다. 그때는 위 첫 줄과 `gh auth login`만 다시 하면 된다.

## 1. 저장소와 환경

```bash
cd ~ && git clone https://github.com/jeehoo0507/Loop-Research-for-MaskedKD.git
cd Loop-Research-for-MaskedKD
bash setup.sh
```

`setup.sh`는 다음을 순서대로 한다.
1. uv가 없으면 설치한다.
2. Python 3.11 `.venv`를 만든다.
3. torch 2.5.1(CUDA 12.4), timm 1.0.15를 설치한다.
4. CPU 테스트를 돌린다.

**확인:** 마지막에 `torch 2.5.1+cu124 cuda True NVIDIA RTX A5000`과 `9 passed`가 나와야 한다.

- `cuda False`가 나오면 드라이버와 CUDA 12.4 휠이 호환되는지 `nvidia-smi`로 본다.
- 데이터는 기본으로 저장소 안 `data/`에 받는다. 다른 디스크를 쓰려면 이후 모든 명령 전에 `export DATA_ROOT=/경로`를 한다.

## 2. 잠금

```bash
bash agent/relock.sh
```

테스트를 다시 돌린 뒤 harness 해시를 `~/.maskedkd_lock`(저장소 밖)에 저장한다.

**확인:** `.venv/bin/python -m harness.lock --check`가 `OK`를 출력해야 한다.

## 3. teacher fine-tune (한 번만)

```bash
tmux new -s teacher '.venv/bin/python -m harness.finetune_teacher 2>&1 | tee agent/logs/teacher.log'
# 빠져나오기: Ctrl-b 다음 d / 다시 보기: tmux attach -t teacher
```

처음 실행하면 CIFAR-100(약 170MB)과 DeiT-S ImageNet 가중치를 받는다.

**확인:** `state/teacher.json`의 `val_acc`가 충분히 높아야 한다. 내 추정은 88% 이상이고, 측정한 값은 아니다. 낮으면 진행하지 말고 원인부터 본다.

## 4. 속도 측정 → 에폭 결정

```bash
.venv/bin/python -m harness.bench
```

먼저 **FLOPs 계량기 자체를 검사**한다: teacher DeiT-S(196토큰)의 측정값이 공식값 9.2 GFLOPs(4.6 GMACs)와 10% 넘게 다르면 `FLOP CHECK FAILED`를 출력하고 종료 코드 2로 끝난다. 이 경우 relock하지 말고 원인을 먼저 본다(모든 결과의 비용 축이 이 계량기에 달려 있다).

그다음 단독 실행과 동시 2~4개 실행의 처리량, VRAM을 재서 `state/bench.json`에 쓰고 권장값을 출력한다.
- `screen_epochs_for_1h_at_that_parallelism`: 거르기 1회를 1시간 안에 끝내는 에폭 수
- `parallel_jobs`: 동시에 돌릴 작업 수

[harness/config.py](../harness/config.py)를 고친다.

```python
TIER_EPOCHS = {
    "cifar100": {"smoke": 1, "screen": <권장값>, "final": <screen의 1.5~2배, 남은 날짜에 맞게>},
    "in100": {"smoke": 1, "final": <4-2단계에서 정함>},
}
MAX_JOBS = <parallel_jobs>
```

그다음 commit, push하고 다시 잠근다.

```bash
git add -A && git commit -m "tier epochs from bench" && git push
bash agent/relock.sh
```

### 4-1. GPU 스모크 (몇 분)

CPU에서만 검증된 코드라, 오래 걸리는 관문 전에 CUDA 쪽 오류를 먼저 잡는다. 작업마다 30스텝, val 500장만 쓴다.

```bash
bash agent/start.sh sched                       # GPU 스케줄러만 tmux(sched)로 시작
.venv/bin/python -m harness.submit --smoke      # ce, full, maskedkd, random, grid, center, tome, oracle
.venv/bin/python -m harness.submit --status     # 모두 rc=0이어야 한다. 실패하면 runs/<id>/error.txt
```

### 4-2. ImageNet-100 준비 (2단계용, CIFAR가 도는 동안 해도 된다)

```bash
tmux new -d -s prep '.venv/bin/python -m harness.prepare_stage2 2>&1 | tee agent/logs/prep.log'   # 받기 약 8.4GB, 변환 결과 약 25GB
.venv/bin/python -m harness.prepare_stage2 --teacher      # teacher: ImageNet-1k DeiT-S의 100클래스 행. val 정확도 출력
.venv/bin/python -m harness.submit --smoke --dataset in100
```

- 데이터: Hugging Face 공개 데이터셋 `clane9/imagenet-100` (로그인 불필요). ImageNet 이미지라 연구용 이용 조건이 적용된다.
- 디스크 여유를 먼저 확인한다(`df -h`). 위치를 바꾸려면 모든 명령 앞에 `export DATA_ROOT=/경로`.
- in100 스모크 로그의 `img_per_s`로 1에폭 시간(약 121,689장 ÷ img/s)을 계산해 `TIER_EPOCHS["in100"]["final"]`을 정한다. 1회가 `STAGE2_TIME_LIMIT_H`(24시간) 안에 들어와야 한다.

## 5. 재현 관문

```bash
.venv/bin/python -m harness.submit --gate      # ce / full@196 / maskedkd@98, seed 0, screen
.venv/bin/python -m harness.submit --status    # 진행 상황
```

세 작업이 끝나면:

```bash
.venv/bin/python -m harness.report --gate
```

**확인:** 두 항목이 모두 PASS여야 한다.
- Full KD ≥ CE + 1%p
- |MaskedKD@98 − Full KD| ≤ 0.5%p

**실패하면 루프를 시작하지 않는다.** 대안 설정으로 바꾼다.
1. [harness/config.py](../harness/config.py)에서 `TEACHER = "deit_base_patch16_224.fb_in1k"`로 바꾸고, student를 사전학습으로 만들도록 [harness/models.py](../harness/models.py)의 `make_student`를 `pretrained=True`로 바꾼다.
2. `state/teacher.*`를 지우고 3단계부터 다시 한다. commit 후 relock도 다시 한다.

통과하면 결과를 commit한다(감사관이 비교에 쓴다).

```bash
.venv/bin/python -m harness.report && git add -A && git commit -m "gate passed" && git push
```

### 5-1. 기준선과 잡음 측정 (관문 직후, 한 번)

```bash
.venv/bin/python -m harness.submit --baselines
```

작업 20개를 큐에 넣는다.
- `maskedkd@98`, `full`을 시드 1·2로: 시드 간 편차(σ)를 재서 승격 기준 max(0.5, 2σ)를 정한다.
- `maskedkd`, `random`, `grid`, `center`, `tome`을 K = 49, 25, 12에서: 토큰 선택이 성적을 가르는 K를 찾고, 공간 규칙(`grid` 고르게, `center` 가운데)과 teacher 경량화 기법(ToMe)이라는 비교 기준을 만든다. 주장은 같은 K에서 이 다섯을 모두 이겨야 한다.
- `oracle` K=12: 상한(분석용).

결과는 `RESULTS_TABLE.md`의 "Noise", "Where token choice matters" 줄에 자동으로 나온다. 에이전트의 final 결과가 생기면 맨 위 "Significant only"(CI가 0을 벗어난 비교만)와 "Core comparison"에 자동으로 정리된다. 루프는 이 작업들이 도는 동안 시작해도 된다.

### 5-2. ImageNet-100 관문과 기준선 (GPU 한 칸에서 천천히)

```bash
.venv/bin/python -m harness.submit --gate --dataset in100              # ce/full/maskedkd@98, final seed 1
# 관문 통과 후 (report의 Stage 2 Gate가 PASS):
.venv/bin/python -m harness.submit --baselines --dataset in100 --keeps 49,25   # 시드 2·3 + 격자 + oracle
```

- 2단계 작업은 동시에 1개, 가장 낮은 우선순위로만 돈다. CIFAR 탐색을 막지 않는다.
- `--keeps`는 CIFAR의 "Where token choice matters"에서 차이가 뚜렷했던 K로 정한다.
- 관문이 통과하기 전에는 에이전트의 방법이 2단계에 제출되지 않는다(harness가 거부).

## 6. 루프 시작

```bash
bash agent/start.sh                 # 스케줄러 + 에이전트 루프 (tmux: sched, agent)
```

`loop.sh`는 시작할 때 잠금과 관문을 다시 확인하고, 하나라도 실패하면 시작하지 않는다.
터미널을 닫아도 계속 돈다.

**확인:**
- `tmux ls`에 `sched`와 `agent`가 둘 다 보인다.
- 15분 안에 `agent/logs/loop.log`에 `round 1 start`가 찍힌다.
- 첫 라운드가 끝나면 GitHub에 `agent: round 1` 커밋이 올라온다.

### 운영 중

| 상황 | 명령 |
|---|---|
| 지시하기 | `agent/QUEUE.md`에 적고 push (로컬에서 해도 된다) |
| 결과 보기 | `cat RESULTS_TABLE.md`, `.venv/bin/python -m harness.submit --status` |
| 라운드 로그 | `tail -f agent/logs/loop.log`, `ls -t agent/logs/round_*.log \| head -1 \| xargs tail -50` |
| 멈춤 | `touch agent/STOP` (다음 라운드 전에 종료). 학습은 스케줄러가 계속한다 |
| 모델 강제 | `echo claude-fable-5-1 > agent/MODEL` (지우면 Opus로 복귀) |
| 파드 재시작 후 | `sudo apt-get install -y -qq tmux gh && bash agent/start.sh` (작업은 체크포인트에서 재개) |
| harness 수정 | 서버에서 고치고 commit → `bash agent/relock.sh` → `tmux kill-session -t agent && bash agent/start.sh` |
| 최종 test | `.venv/bin/python -m harness.test_eval --runs <id> ...` (사람만, 마지막에 한 번) |

### 문제 해결

- **`lock mismatch after pull; pausing`**: 누군가 잠금 파일을 바꿔 push했다. 바뀐 내용을 확인한 뒤 `bash agent/relock.sh` → 루프를 재시작한다.
- **`agent/INVALID.md`에 기록이 생김**: 에이전트가 잠금 파일을 고쳐서 되돌려졌다는 뜻이다. 그 라운드에 제출된 작업은 감사관이 다시 본다.
- **작업이 시작되지 않음**: `tmux attach -t sched`에서 `LOCK` 메시지나 VRAM 부족(`free`, `need`)을 확인한다.
- **작업 실패**: `runs/<id>/error.txt`, `runs/<id>/stdout.txt`를 본다.

---

## (선택) GPT 붙이기: 비상 연구자 + 감사관

GPT는 필수가 아니다. Codex CLI가 설치되어 로그인돼 있으면 다음 두 역할에 쓰인다.

**1) 비상 연구자 (자동).** Claude 라운드가 실패하면(종료 코드 ≠ 0, 또는 출력에 usage/rate limit) 같은 라운드를 GPT가 이어받는다.
- 명령: `codex exec -m gpt-6-astra -c model_reasoning_effort="high" -s workspace-write`
- 지침(`AGENT_PROMPT.md`)이 같고, 잠금 검사·되돌림도 똑같이 적용된다.
- 실패 후 60분(`CLAUDE_RETRY`)이 지나면 Claude를 다시 시도하고, 성공하면 Claude로 돌아온다.
- 커밋과 PROGRESS에 `[claude]` / `[codex]`가 붙는다.
- GPT가 낸 주장은 **Claude가 감사할 때까지 확정되지 않는다.** Claude가 멈춰 있으면 감사 요청은 그대로 대기한다.
- 모델 바꾸기: `CODEX_MODEL=<모델> CODEX_EFFORT=<low|medium|high|xhigh> bash agent/start.sh`
  (모델 이름은 외부 자료 기준이다. 계정에서 안 보이면 `codex exec -m gpt-6-astra "OK"`로 먼저 확인)

Codex가 없거나 로그인이 끊겨 있으면, Claude가 멈춘 동안 연구 라운드도 멈춘다. GPU 큐는 스케줄러가 계속 돌린다.

**2) 감사관 (선택).** `AUDITOR=codex`이면 Claude가 낸 주장을 GPT가 읽기 전용으로 감사하고, 하루 1번 독립 아이디어 2개를 `PROPOSALS.md`에 올린다. 기본값(`claude`)은 별도 Claude 세션이 감사한다.

### 1) 설치

Node.js가 없으면 먼저 설치한다.

```bash
node -v || { curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt-get install -y -qq nodejs; }
sudo npm i -g @openai/codex
codex --version
```

### 2) 로그인 (셋 중 하나)

서버에는 브라우저가 없어서 일반 `codex login`이 안 될 수 있다.

**(a) 기기 코드 로그인 (ChatGPT 계정, 권장)**

```bash
codex login --device-auth
```

출력된 링크를 내 컴퓨터 브라우저에서 열고 코드를 입력한다. ChatGPT 보안 설정에서 기기 코드 로그인이 켜져 있어야 한다. 베타 기능이다.

**(b) API 키**

```bash
read -s OPENAI_API_KEY && export OPENAI_API_KEY
printenv OPENAI_API_KEY | codex login --with-api-key
```

`read -s`는 키를 화면과 셸 기록에 남기지 않는다. 키는 어디에도 붙여 넣어 공유하지 마라.

**(c) 내 컴퓨터의 로그인 정보 복사**

내 컴퓨터에서 `codex login`을 한 뒤 `~/.codex/auth.json`을 서버의 같은 경로로 복사한다. 이 파일은 비밀번호와 같으니 git에 올리지 마라.

### 3) 동작 확인

```bash
codex exec -m gpt-6-astra -c model_reasoning_effort="high" -s read-only "Reply with the single word OK"
```

`OK`가 나오면 된다. 감사관은 `-s read-only`(파일 수정 불가)로, 비상 연구자는 `-s workspace-write`(저장소 안만 쓰기)로 돈다.

### 4) 켜기 / 끄기

**루프를 시작할 때 켜기:**

```bash
AUDITOR=codex bash agent/start.sh
```

**이미 도는 루프에서 바꾸기 (재시작 없이, 다음 라운드부터 적용):**

```bash
echo codex  > ~/.maskedkd_auditor     # GPT 감사
echo claude > ~/.maskedkd_auditor     # Claude 감사 (기본)
echo none   > ~/.maskedkd_auditor     # 감사 끔 (주장은 UNAUDITED로 남음)
rm ~/.maskedkd_auditor                # 시작할 때 설정으로 복귀
```

이 파일이 저장소 밖에 있는 이유는 에이전트가 감사를 스스로 끌 수 없게 하기 위해서다.

**확인:**
- 감사 요청이 생긴 라운드가 끝나면 `AUDIT.md`에 `audit by codex` 블록이 추가된다.
- 원본 출력은 `agent/logs/audit_*.log`에 있다.
- 아이디어는 `PROPOSALS.md`의 `ideas from Codex` 절과 `agent/logs/ideas_*.log`에 있다.

### 참고
- Codex 명령을 바꾸려면(모델 지정 등) `CODEX_CMD`를 쓴다. 예: `CODEX_CMD="codex exec -s read-only -m <모델>" AUDITOR=codex bash agent/start.sh`
- 감사는 주장이 올라온 라운드에만 돌아서, 사용량은 하루 몇 번 수준이다.
- GPT 로그인이 끊기면 감사 로그에 오류가 남고 `AUDIT.md`에는 판정이 없다. 그 주장은 확정되지 않는다. 다시 로그인하면 다음 요청부터 정상으로 돈다.
