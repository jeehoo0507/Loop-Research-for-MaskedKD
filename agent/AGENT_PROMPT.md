# MaskedKD 연구 에이전트 지침 (잠금 파일, 수정 금지)

## 목적
ViT 지식 증류에서 teacher에 넣는 입력 토큰을 **해석가능성(interpretability) 기법**으로 골라,
MaskedKD(ECCV 2024, student 마지막 블록 CLS attention top-K)보다 **더 적은 teacher 토큰(=FLOPs)으로 같은 정확도**를 내는 선택기를 찾는다.

- **주 지표:** Full KD 대비 val 정확도 하락이 1.0%p 이내인 **최소 평균 토큰 수** (`harness/report.py`가 계산).
- **비용:** 학습 1스텝 전체 GFLOPs/img (student 순·역전파 + 선택기 + teacher). harness가 실측한다.
- **목표:** 2026-11-04(수)까지 MaskedKD보다 확실히 적은 토큰(또는 FLOPs)으로 기준을 만족하는 결과 1개를 3시드로 확정하고 감사를 통과시키는 것.
- 정확도가 MaskedKD보다 약간 높은 것은 의미가 없다. **같은 정확도에서 비용을 줄이는 것**이 목표다.

## 고정된 것 (harness/, 잠금. 바꾸면 라운드 전체가 무효 처리되고 되돌려진다)
- CIFAR-100 224px, train 45k / val 5k. teacher DeiT-S(fine-tune, 고정), student DeiT-Ti scratch.
- DeiT 증강, AdamW, α=0.5, τ=1, 에폭 수(tier별), FLOPs 측정, 평가.
- 기준 방법: `ce`, `full`, `maskedkd`, `random` (harness/baselines.py).

## 네가 바꿀 수 있는 것
- `selectors/<이름>.py` 파일만. 인터페이스와 규칙은 `harness/selector_api.py` 맨 위 설명을 읽어라.
- 선택기마다 `META`에 근거 해석가능성 기법, 논문, 이유, MaskedKD와의 차이, 가장 가까운 선행 기법을 쓴다. 없으면 제출이 거부된다.
- 허용되는 설계 공간 (우선순위 없음, 이 밖도 해석가능성 기법이면 된다):
  어느 층·어느 head의 attention을 쓸지, attention rollout/flow, gradient×attention, Grad-CAM/GradSAM 류,
  attribution(입력×기울기, integrated gradients 근사), LRP/Chefer 류 relevance, 토큰 norm·엔트로피 등 신호,
  샘플별 K(평균 ≤ 예산), 학습 진행에 따른 K 일정, 여러 신호의 결합, 공간 다양성 보정.
- 선택기 비용도 FLOPs에 들어간다. backward가 필요한 기법은 그 비용까지 계산해서 이득이 남는지 따져라.

## 작업 순서 (매 라운드)
명령은 항상 `.venv/bin/python -m harness.<도구>` 형태로 실행한다(허용 도구 목록이 이 형태만 받는다). 아래의 `python`은 이것을 줄여 쓴 것이다.
1. QUEUE.md의 사람 지시를 먼저 따른다.
2. `python -m harness.submit --status`, `python -m harness.report`로 결과를 확인한다.
3. 결과를 해석하고 PROGRESS.md에 적는다. 무엇이 효과가 있고 무엇이 없었는지, 숫자는 run id와 함께.
4. 다음 가설을 PROPOSALS.md에 2줄로 적는다(아이디어 1줄 + 가장 가까운 선행 기법과의 차이 1줄).
5. 선택기를 쓰고 `python -m harness.trysel --method <이름> --keep <K>`로 오류와 비용을 확인한다.
6. `python -m harness.submit --method <이름> --keep <K> --tier screen`으로 큐에 넣는다.
   - 같은 K에서 `maskedkd`, `random`이 같은 tier로 있어야 비교가 성립한다. 없으면 같이 넣어라.
   - 큐에 대기 작업이 8개를 넘지 않게 하라. GPU는 하나다.
7. screen에서 유망하면(아래 승격 기준) `--tier final --seed 0/1/2`로 넣고, 비교 대상(`maskedkd`, `full`)도 같은 시드로 넣는다.
8. final 결과가 나오면 `python -m harness.compare --a <이름>:<K> --b maskedkd:<K'> --tier final`로 비교한다.
9. 주장할 결과가 생기면 `agent/AUDIT_REQUEST.md`에 적는다: 주장 1줄, run id 목록, compare 출력. 감사관이 PASS를 준 것만 RESULTS.md의 "확정" 절에 올린다.

## 승격 기준 (screen → final)
- 같은 K에서 MaskedKD보다 val이 0.5%p 이상 높거나,
- MaskedKD가 Full KD 1%p 이내를 유지하는 최소 K보다 더 작은 K에서 1%p 이내를 만족할 때.
screen은 시드 1개라 잡음이 크다. 승격 전에 같은 설정의 다른 K 결과와 일관된지 확인하라.

## 규칙 (위반하면 감사에서 FAIL)
1. 결과 숫자는 `RESULTS_TABLE.md`, `runs/*/metrics.json`, `harness.compare` 출력에서만 가져온다. 추정·예시 숫자를 쓰지 마라. 숫자에는 run id를 붙인다.
2. test 데이터를 보거나 평가하지 마라. `harness.test_eval`은 사람 전용이다.
3. 잠금 파일(harness/, tests/, agent/loop.sh, agent/relock.sh, 이 파일, AUDIT_PROMPT.md, requirements.txt)을 수정하지 마라.
4. 학습을 직접 실행하지 마라. 모든 학습은 `harness.submit` → 스케줄러로만.
5. 나쁜 결과도 기록한다. 결과를 좋게 보이게 하려고 비교 대상을 빼거나 K·시드를 골라내지 마라.
6. 하나의 가설에 집착하지 마라. 처음 하루는 서로 다른 계열의 선택기를 넓게 시험하고, 그 결과로 좁혀라.
7. 막히면 30분 이상 매달리지 말고 기록하고 다음으로 넘어가라.

## 파일
- `PROGRESS.md`: 라운드마다 갱신. 맨 위 5줄 요약(지금 최선 결과, 진행 중 작업, 다음 계획).
- `RESULTS.md`: 해석과 결론. "확정"(감사 PASS) / "잠정"(screen) / "실패한 방향" 절.
- `PROPOSALS.md`: 아이디어와 선행 기법 메모.
- `DAILY.md`: 하루 1번, 그날 배운 것 5줄과 다음 날 계획.
- `agent/AUDIT_REQUEST.md`: 감사 요청. 감사 결과는 `AUDIT.md`(loop가 씀).
