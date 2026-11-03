# 사람 지시 (매 라운드 최우선)


## 2026-10-31 17:30 KST — 타당성 있는 탐색으로 전환 (최우선)
사람 판단: 지금 최선인 suffdisc 계열(suffdisc, suffstride, attnwin)은 `center`를 부모로 하고, 가운데에서 최대 몇 칸만 움직이는 원판을 고른다. Waterbirds·CUB는 새가 대체로 가운데 있어서, 이 규칙은 메커니즘이 아니라 데이터셋의 위치 편향을 이용한다. 같은 총비용에서 center@49가 suffdisc@12를 이기는 것(H-W32)도 같은 이유로 보인다. 이런 방법은 일반성이 없고 노벨티가 없다.

1. AGENT_PROMPT의 새 절 "타당성 요건"과 AUDIT_PROMPT 16·17항을 먼저 읽어라 (이번 커밋에서 추가됨).
2. suffdisc 계열은 주장 대상에서 뺀다. 진행 중인 suffdisc·suffstride job은 끝까지 돌게 두고 기록만 한다. RESULTS.md "분석에서 배운 것"에 남길 것: (a) MaskedKD의 마지막 층 attention은 토큰을 줄이면 최악 그룹이 크게 무너진다 (WGA −40%p 수준), (b) 가린 teacher의 출력 충실도(KL)가 student WGA를 잘 예측한다 (analysis_proxy_validation, Spearman 0.94), (c) 위치를 고정한 원판 규칙이 이 데이터에서 강한 이유는 위치 편향이다.
3. 위 (a)(b)는 근거로 계속 써도 된다. 다음 방법은 위치 사전지식 없이 토큰별 점수로 만들고, 그 점수가 왜 teachability를 반영하는지 분석으로 먼저 보여라. 무엇을 할지는 네가 정한다.
4. harness 변경 (이번 커밋): 감사 증거에 전체 job 이력(모든 상태)이 들어간다 → STUCK이던 감사 요청을 다시 낼 수 있다(단, suffdisc 계열 주장은 2번에 따라 내지 마라). screen에서 seed 1·2·3은 이제 금지다(final과 같은 run이 되므로). 잡음 seed는 4·5다. CUB 관문은 "MaskedKD@98 ≥ Full − 0.5, 시드 평균"으로 바뀌어 2단계가 열릴 수 있다.

## 2026-10-31 18:30 KST — 계획 재정비: 탐색은 CUB, 판정은 평균 정확도 + FLOPs (위 17:30보다 우선)
사람 판단: 이 연구의 목표는 강건성이 아니라 **KD의 FLOPs 효율**이다. Waterbirds를 쓴 이유는 KD > CE를 찾기 쉬워서였을 뿐이다. 그런데 주 지표에 WGA가 들어가 있어서 연구가 "지름길 편향" 쪽으로 흘렀다. 또 2클래스 데이터는 teacher soft label에 클래스 관계 정보가 거의 없어서 KD 연구로 약하다.

이번 커밋으로 바뀐 것 (AGENT_PROMPT·AUDIT_PROMPT 갱신됨, 먼저 읽어라):
1. **1단계(탐색) = CUB-200-2011, 2단계(확인) = Waterbirds.** 서버가 CUB에서 student lr sweep → CUB 관문 → CUB 기준선을 자동으로 돌린다. 그동안 GPU가 차 있으니, 기준선이 나오기 전에는 분석과 설계에 집중해라.
2. **주 지표 = 평균 정확도(허용 오차 = KD 이득의 절반, 최대 1%p) + 그때의 학습 1스텝 총 GFLOPs.** WGA는 진단값으로만 표에 남는다. 주장은 **시드 3개 이상 평균**으로만 한다 (Primary metric의 claim-grade 줄).
3. **위 17:30 지시 중 유지:** 위치 사전지식 금지, 토큰별 점수, 근거 먼저, 사전 등록, 같은 총비용 비교. suffdisc 계열은 주장 대상이 아니다.
4. **위 17:30 지시 중 바뀐 것:** 근거로 쓰라고 한 (a)(b)는 WGA에 관한 것이라 주 질문의 근거가 아니다. 다음 방법의 근거는 "가린 teacher가 student에게 주는 KD 신호(soft label)를 얼마나 보존하는가"처럼 **평균 정확도와 FLOPs에 연결되는 것**이어야 한다. 무엇을 분석할지는 네가 정한다.
5. Waterbirds에서 끝난 run은 2단계 표에 그대로 남는다. 새 Waterbirds 실험은 2단계 규칙(관문 + CUB final + 감사 PASS) 뒤에만 낼 수 있다.

## 2026-10-31 20:40 KST (21:00 고침) — 참고 정보 + TrustMask 1회 시험
방향은 네가 근거로 정한다. 아래 지시 하나만 따른다.
- 읽을 만한 이론: Allen-Zhu & Li, ICLR 2023 (arXiv 2012.09816), KD의 multi-view 가설. 가린 teacher가 무엇을 잃는지 설명하는 근거로 쓸 수 있으면 써라.
- 사람의 이전 실험 (TrustMask, Waterbirds, 이 harness 밖이라 근거로는 약함): rollout 선택에서 "중요하지 않은 토큰을 빼는 필터"와 "이미지마다 남길 개수를 다르게 하는 것"(배치 평균은 K 유지)을 해 봤다. 분석이 부족했던 저단계 시도다.

**지시 (이것 하나만):** CUB 기준선이 나온 뒤, screen run **딱 1개**를 TrustMask를 발전시킨 선택기에 쓴다. 그대로 옮기지 말고, 아이디어 기반만 가져와 네 분석으로 다듬어라 (사전 등록, 같은 K의 rollout·maskedkd와 비교). 결과가 어떻든 그 1개로 끝내고 원래 탐색으로 돌아간다. 이기면 다른 후보와 똑같은 규칙(seed 추가, 같은 총비용 비교, 감사)으로만 이어 간다.
- 그 run 전에 분석 하나를 먼저 해 볼 만하다 (선택): CUB에서 K = 98·49·25·12마다 "student만 (rollout)", "teacher가 고르기 (teacher rollout 상위 K)", "teacher가 거르기 (teacher 지도의 허용 영역 안에서 student가 고르기)"의 TCKD 보존. MaskedKD 논문은 teacher로 고르기가 나빴다고 했고, TrustMask에서는 teacher로 거르기가 나았다. 작은 K에서만 거르기가 앞서는 교차점이 보이면 그 run에 teacher 지도를 써도 된다. 학습 초반에만 강제하고 student와 teacher 지도가 충분히 겹치면 student에게 넘기는 방식도 후보다.
- harness 변경 (이번 커밋, 서버가 스스로 받아들임): `ctx.teacher_map` (META `"uses_teacher_cache": True`)과 기준선 `teacher_topk`가 생겼다. 규칙은 AGENT_PROMPT 타당성 요건 2번의 예외 항목을 읽어라. 쓰지 않아도 된다.

## 2026-10-31 22:20 KST — CUB 관문의 MaskedKD 조건
CUB seed 0에서 KD 이득은 +3.5%p (Full 82.17 vs CE 78.67)로 충분하다. MaskedKD@98은 Full보다 1.17 낮았다 (기준 −0.5). seed 0·4·5 평균으로도 MaskedKD 조건만 못 맞추면 부트는 멈추지 않고 진행한다 (BOOT_STATUS에 "kept as a finding"). 그렇다면 "CUB에서는 MaskedKD가 토큰 절반에서도 손해를 본다"는 것이 기록할 결과다. 비교 기준은 그대로 Full KD의 허용 오차다.

## 2026-11-01 KST — 연구 질문 전환: student 학습 상태에 따른 이미지별 teacher 토큰 예산 (위 지시 전부보다 우선)
사람 판단 (멘토 방향 "해석 가능성 기법으로 KD를 효율적으로, Efficiency + Teachability" 기준):
- rollout을 조금씩 바꾸는 탐색(lateroll, sinkfree, priorroll, adaroll 등)은 여기서 멈춘다. 이득이 이미지 몇 장 수준이고, 층 범위 조절은 EViT 계열 토큰 가지치기에서 흔한 설정이라 노벨티가 없다.
- 새 주 질문: **student가 아직 못 배운 이미지에 teacher 토큰을 더 주고 이미 배운 이미지에는 덜 준다(배치·run 평균은 같은 예산). 토큰은 해석 기법으로 고른다. 같은 평균 teacher 비용에서 고정 K(MaskedKD, lateroll)보다 나은가?** MaskedKD는 "어떤 토큰"을 정하고, 이 질문은 "각 이미지에 몇 개"를 student의 학습 상태로 정한다.
- 데이터: CUB만으로 판단한다. Waterbirds는 KD 이득(+1.25%p)이 작아 판단에 쓰지 않는다(2단계 기록은 참고로만 남긴다). 루프 종료일은 2026-12-31로 늘렸다.

### 선행연구 경계 (사람이 문헌 확인함). 이것들과의 차이를 먼저 적고 시작해라
- Dynamic KD (Li et al., EMNLP 2021): student 불확실성으로 teacher에 물을 샘플을 고른다(샘플 단위, 전부 아니면 없음, NLP).
- Masters (arXiv 2512.22238): teacher 마스킹 비율을 학습 단계별로 바꾼다(시간 스케줄, 이미지별 아님, VLM).
- SBAM (ECCV 2024): 이미지마다 마스킹 비율을 saliency로 정한다(MIM, KD 아님, 학습 상태 아님).
- EGAD (arXiv 2605.01732), EA-KD: 엔트로피로 샘플·토큰별 증류 손실 가중치를 바꾼다(teacher 비용은 그대로).
- Explanation-guided KD (EGKD 등): teacher와 student 설명의 일치를 손실로 쓴다(예산 신호로 쓰지 않음).
- MaskedKD: 고정 비율. 이미지별 비율 후속은 찾지 못했다. 빈자리는 "ViT supervised KD에서 student 학습 상태(특히 attribution 일치도)로 이미지별 teacher 토큰 수를 연속 배분"이다.

### 이번 커밋의 harness 변경 (서버가 테스트 통과 후 스스로 받아들임). selector_api.py 문서를 먼저 읽어라
- 이미지별 예산: bool keep에 이미지마다 다른 개수를 둘 수 있다. 0개면 그 이미지는 teacher 없이 CE만 쓴다. 배치 평균은 2×예산까지, run 전체 평균은 예산 이하(넘으면 over_budget으로 기록). FLOPs는 run 전체에 걸친 여러 구간에서 잰다.
- `select`는 `(keep, kd_weight[B])`를 돌려줄 수 있다. kd_weight 평균은 teacher를 받은 이미지에서 1 이하다.
- `observe(fb)`: teacher forward 뒤 이미지별 KD 손실(fb.kd, teacher 없는 이미지는 NaN)과 fb.sample_idx(학습셋 행 번호)를 받는다. 이미지별 학습 상태를 epoch을 넘어 기억할 수 있다(teacher 추가 비용 없음, resume 뒤에는 상태가 비어 있음). `ctx.sample_idx`도 있다.
- 도우미: `allocate(weights, budget, kmin, kmax)`, `keep_by_counts(scores, counts)`.
- 새 기준선 5개(`config.ALLOC_REFS`): `rand_alloc`(무작위 배분, 대조군), `dynkd`(Dynamic KD식 전부/없음), `sal_ratio`(SBAM식 saliency 비율), `kd_sched`(Masters식 시간 스케줄), `ent_weight`(EGAD식 손실 가중, 고정 K). META에 `"per_image_budget": True`를 넣은 방법은 RESULTS의 core 비교에서 이 5개와도 같은 평균 예산으로 비교된다. **이 5개를 모두 이겨야 기여가 있다.**

### lateroll STUCK에 대한 답
lateroll은 주장(새 방법)으로 올리지 않는다. 대신 **가장 강한 고정 K 기준선**으로 확정한다. lateroll@29 final seed 1·2·3을 내라(주장이 아니라 비교 기준이므로 Parent check 문턱과 무관). maskedkd@29·rollout@29 final은 final_refs로 함께 들어간다. 주장에 대한 Parent check 규칙은 그대로다.

### 할 일 (순서대로)
0단계 — 생사 판정 (분석만, 사전 등록 후 제출. analysis API의 teacher_forward_keepmask와 student 체크포인트를 써라):
- H1: 이미지마다 필요한 토큰 수가 크게 다른가? 학습된 student(예: maskedkd@29, lateroll@29의 중간·끝 체크포인트)로 이미지별 "K를 늘릴 때 가린 teacher의 KD 신호가 Full teacher에 가까워지는 곡선"을 재고, 같은 품질에 필요한 K의 이미지 간 분포를 본다.
- H2: 그 필요량을 싼 신호로 예측할 수 있나? 후보: (a) student 예측 엔트로피·KD 손실, (b) student attention과 teacher 캐시 지도(teacher_map)의 일치도 — 해석 가능성 기반 Teachability 신호, (c) 무작위.
- H3: 같은 평균 K에서 연속 배분(필요량 순서로 토큰 수를 나눔)이 전부/없음 배분(Dynamic KD식)과 고정 K보다 KD 신호를 더 보존할 여지가 있나? 오라클 배분으로 상한을 먼저 재라.
- 판정: H1이 아니면(이미지 간 필요량이 비슷하면) 또는 H3에서 오라클 배분도 고정 K와 구별되지 않으면 이 방향을 끝낸다. PROGRESS 첫 줄에 "방향 종료: <이유>"를 쓰고, 그때는 아래 '종료 시'로 간다.
1단계 — 방법 (0단계 통과 시): 이미지별 예산 선택기를 설계한다(META `"per_image_budget": True`). 배분 규칙은 0단계 근거로 네가 정한다(공식 강제 없음). screen seed 0·4·5, 평균 K = 29(필요하면 49)에서 고정 K(maskedkd·rollout·lateroll)와 ALLOC_REFS 5개를 모두 같은 seed로 돌려 비교한다. 이기면 final 규칙대로.
2단계 — 멘토 보고용 정리 (진행 중 계속): Efficiency = 평균 teacher GFLOPs 대비 정확도 곡선. Teachability = 어떤 이미지가 예산을 받는가, "teacher가 중요하게 보는 것 ≠ student가 배울 수 있는 것"을 이미지 단위로 보인다. 멘토가 말한 "너무 달라도, 너무 같아도 안 된다"(student–teacher attribution 일치도와 KD 이득의 뒤집힌 U자)를 직접 검증해라.
종료 시 (0단계 실패): 새 rollout 변형을 만들지 말고, 지금까지의 고정 K 결과(K별 곡선, teacher_topk < student 기반, TrustMask 재현 실패)를 Teachability 분석으로 정리한 뒤 사람 지시를 기다린다.

### 보충 (같은 날): 예산 신호가 핵심 질문이다. 한쪽으로 몰지 마라
멘토의 질문("해석 기법이 고른 것이 student가 배우기 좋은 정보인가")에 답하는 부분은 **각 이미지에 몇 개를 줄지 정하는 신호**다. 어떤 토큰을 줄지는 기존 해석 기법으로 충분하다.
- 신호 후보를 두 갈래로 **같은 조건**(같은 평균 K, 같은 토큰 선택 규칙, 같은 seed)에서 비교하라.
  - 해석 기반: student attention과 teacher 지도의 일치도, student attribution의 퍼짐·집중도, 층 사이 attribution 변화 등
  - 비해석 기반: student 예측 엔트로피, observe로 받은 KD 손실 이력 등 (dynkd·ent_weight도 이 갈래다)
- 어느 쪽이 이기든 그대로 기록한다. 해석 기반이 이기면 멘토 질문에 대한 긍정적 답이다. 지거나 같으면 "teachability는 해석 신호로 잡히지 않고 학습 상태 신호로 잡힌다"는 것이 결과다. 둘 다 보고 가치가 있다.
- 결과를 보고 신호를 고르지 마라. 0단계 사전 등록에 두 갈래의 후보와 판정 기준을 미리 적어라.

### 유지
사전 등록, 같은 평균 총비용 비교, seed 3개 이상 평균으로만 주장, 위치 사전지식 금지, GPU 놀리지 않기(대기 8개 이하).

## 2026-11-01 22:40 KST — H-C22 "방향 종료"에 대한 사람 판단: 전부/없음 집중을 학습으로 확인한다
H-C22의 실패 모양이 결과다: 고정 K=29에서 가린 teacher는 이미지 대부분(약 76%)에 라벨 이상을 주지 못하고, 배분 여유는 대부분 "일부 이미지에 teacher 전체, 나머지는 없음"(무작위 선택으로도)에서 나왔다. 맞다면 "적은 teacher 예산은 모든 이미지에 조금씩보다 소수 이미지에 통째로가 낫다"는 MaskedKD의 고정 비율 전제와 반대되는 결과다. 대리 지표(TCKD 보존)로 본 것이라 정확도로 확인한다.

이번 커밋: 기준선 `rand_dyn`(dynkd와 같은 전부/없음 분할, 이미지는 무작위) 추가. scheduler 좀비 수정(재시작 뒤 끝난 job을 거둬 done으로 기록) 포함.

1단계 (지금): CUB screen, 평균 K = 29, seed 0·4·5. `dynkd`·`rand_dyn` 각 3개를 낸다. 같은 seed의 maskedkd@29·rollout@29·lateroll@29는 이미 있다. 사전 등록에 적을 것: (P1) rand_dyn의 3-seed 평균이 maskedkd@29보다 높다, (P2) dynkd가 rand_dyn보다 높다(선택 신호가 의미 있음), 판정은 같은 seed 짝 비교와 3-seed 평균으로.
- P1이 틀리면: 대리 지표와 정확도가 어긋난 것으로 기록하고 이 방향을 닫는다(Teachability 정리에 반영).
- P1이 맞으면: K = 12에서도 같은 비교를 낸다(예산이 작을수록 차이가 커야 한다는 예측도 미리 적어라). 그다음 질문은 "어떤 이미지에 몰아줄지"다. 해석 기반 신호와 비해석 신호(엔트로피, KD 손실 이력)를 같은 전부/없음 틀에서 같은 조건으로 비교하라(위 '보충' 지시 그대로). 전부/없음과 "일부는 전체, 나머지는 작은 K" 같은 혼합도 후보다. 무엇을 낼지는 근거로 네가 정한다.
- P2가 틀려도(무작위 = 불확실성) P1이 맞으면 결과다: 이득은 집중 자체에서 오고, 선택 신호는 부차적이다.
선행연구 경계: Dynamic KD(EMNLP 2021)가 NLP에서 샘플 단위 teacher 선택을 했다. 여기서의 차이는 ViT supervised KD에서 같은 teacher FLOPs의 고정 비율 토큰 마스킹(MaskedKD)과 직접 비교한다는 것, 그리고 무작위 집중까지 분리해 본다는 것이다.
lateroll@29 final seed 2·3은 그대로 진행한다.

## 2026-11-02 08:50 KST — 사람 결정: 이미지별 예산 탐색 종료 → lateroll 정리 + Aircraft 이전 검증
라운드 159의 판단 요청에 대한 답이다. 이미지별 예산(전부/없음 집중) 방향은 여기서 닫는다. 새 선택기 탐색도 하지 않는다.

### 하네스 변경 (사람, 이 커밋)
- `config.STAGE2 = "aircraft"` (FGVC-Aircraft 2013b, 100 variants). 분할: 공식 test 3333, 공식 trainval을 클래스별 90/10 (CUB와 같은 규칙) → train 5967 / val 700. 아래 20 px 저작권 띠는 잘라낸다. Waterbirds run은 그대로 유효하고 RESULTS_TABLE의 "Earlier check" 절에 남는다.
- **설정은 루프가 사람 없이 한다** (`agent/loop.sh` → `agent/stage2_boot.sh`): 데이터 다운로드·준비 → teacher 미세조정 (DeiT-B, CUB와 같은 레시피, 약 1~2 h; 그동안 루프가 잠깐 멈췄다 relock 뒤 다시 켜진다) → 게이트 (final seed 1, Full KD ≥ CE + 1.0) → 기준선 (pick-keeps 예산). 학생 lr은 기본값 1e-4 (CUB 스윕이 고른 값과 같다). 진행 상황은 `agent/BOOT_STATUS.md`의 `[stage2]` 줄.
- `STAGE2_MAX_RUNNING` 1 → 3 (CUB 탐색이 끝나 Aircraft가 주 작업).
- 너는 이 설정 과정을 건드리지 않는다. 막히면 (`STOPPED` 줄) PROGRESS에 적고 사람을 기다린다.

### 할 일 (순서대로)
1. **Teachability 정리 (지금, GPU 불필요):** H-C22~H-C28을 RESULTS의 한 절로 정리한다. 결론은 판정 그대로 쓴다: 전부/없음 집중은 CE 수준, 선택 신호는 무작위를 넘지 못함, K=98에서 고정 K 우위 재현, K≤29 우위는 재현 불확실, H-C26 손실 비중 혼입 불확실. H-C28(CE·Full seed 6·7)이 끝나면 그 판정도 넣는다. 멘토 질문 "이미지당 토큰 몇 개"에 대한 답으로 읽히게 쓰되, 주장 범위를 넘지 않는다.
2. **lateroll 감사 요청 (지금):** stage 2에 agent 방법을 올리려면 AUDIT.md에 PASS가 필요하다 (`submit.stage2_rule`). lateroll@29 CUB final seed 1·2·3 (81.39 vs maskedkd 80.22, rollout 80.44, Full 81.72)을 `agent/AUDIT_REQUEST.md`에 AGENT_PROMPT 9번 형식으로 적는다. 감사관이 NEEDS-MORE를 주면 요구를 채운다.
3. **CUB Efficiency 곡선 마무리:** lateroll final이 K=29에만 있다. 곡선(평균 teacher GFLOPs 대비 정확도)에 필요한 최소한의 lateroll final (예: K=12·49, seed 1·2·3, `--tier final`이면 final_refs가 같은 K의 기준선을 같이 낸다)을 낼지 네가 정한다. 이미 있는 run은 다시 내지 않는다. Aircraft job이 시작되면 GPU를 그쪽에 양보한다 (stage 2 우선순위는 하네스가 정한다).
4. **Aircraft 이전 검증 (게이트 통과 + 감사 PASS 뒤):** lateroll을 **CUB에서 정한 그대로** (코드·sel_args·K 변경 없음 → `stage2_untouched`) Aircraft final seed 1·2·3에 낸다: `python -m harness.submit --method lateroll --keep 29 --seed <s> --tier final --dataset aircraft` (final_refs가 maskedkd·rollout·tome·full을 같은 seed로 낸다). 3번에서 다른 K를 CUB에서 확정했으면 그 K도 같은 방식으로.
   - **판정 (미리 적음, 결과 보고 바꾸지 않는다):** Aircraft에서 lateroll − maskedkd, lateroll − rollout의 seed 3개 평균이 둘 다 0보다 크고 짝지은 95% CI가 0을 포함하지 않으면 "CUB에서 고른 lateroll이 다른 데이터셋으로 이전된다". 평균은 양수지만 CI가 0을 포함하면 "경향은 같으나 불확실". 평균이 0 이하면 "CUB 특화"로 기록하고 주장 범위를 CUB로 줄인다. 셋 다 보고할 수 있는 결과다.
   - Aircraft 결과를 본 뒤 lateroll을 고치는 것(`stage2_tuned`)은 판정이 끝난 뒤에만, 별도 항목으로.
5. **게이트가 실패하면** (Aircraft에서 KD가 CE를 +1.0 넘지 못함): 그 자체를 기록하고 (KD 이득이 없는 데이터셋에서는 토큰 선택을 판정할 수 없다) 사람을 기다린다. 다른 데이터셋을 스스로 고르지 않는다.

### 금지
- 데이터셋별 상수나 데이터셋을 보고 바꾸는 분기를 selector에 넣지 않는다 (사람이 이 연구에서 가장 경계하는 것: "그 데이터셋에서만 통하는 마스킹").
- Aircraft val로 lateroll의 하이퍼파라미터를 고르지 않는다.

## 2026-11-02 14:10 KST — 감사 STUCK·Aircraft 다운로드에 대한 사람 조치
- **감사 STUCK 해제:** `harness/evidence.py`가 이제 "past locks" 절을 넣는다. 인용 run마다 metrics.json의 harness_digest를 그 run이 시작한 커밋에서 다시 계산해 MATCH 여부를 보이고, 그 뒤 잠금 파일을 바꾼 커밋 전부(작성자·날짜)와 diff stat을 보인다. AUDIT_PROMPT 3항도 이 경우를 인정하게 고쳤다. 로컬 확인: 3b594cc1646f(53616be)·8ac8a710a4ae(cc12489) 모두 MATCH, 그 뒤 변경은 사람 커밋 faeb3cd·8f67ca5와 이 커밋뿐. → lateroll@29 감사를 한 번 다시 내라 (요청서의 무결성 절은 짧게 줄이고 "past locks 절 참고"로 충분).
- **Aircraft 다운로드:** 서버가 robots.ox.ac.uk에 접속하지 못한다. 같은 아카이브의 Hugging Face 사본을 두 번째 URL로 넣었다. 사람이 서버에서 실패 표시를 지우고 stage2를 다시 켠다. 너는 계속 건드리지 않는다.

## 2026-11-02 14:30 KST — 감사 묶음 보강 (04:55 NEEDS-MORE에 대한 사람 조치)
`harness/evidence.py` audit 묶음에 (a) 인용 run 커밋 → HEAD의 잠금 파일 전체 diff 본문(tests 제외), (b) 인용된 agent 방법의 PROPOSALS 항목 전부와 `git log --follow selectors/<method>.py`를 넣었다. STUCK을 풀고 lateroll@29 감사를 한 번 더 내라. 이번에도 묶음 구성 때문에 막히면 다시 내지 말고 PROGRESS에 무엇이 빠졌는지 적어라.

## 2026-11-03 11:50 KST — 감사 묶음 잘림 수정 (12:32 NEEDS-MORE에 대한 사람 조치)
`harness/evidence.py` audit 묶음 순서를 바꿨다: 요청 → 잠금 → 인용 run metrics·selector 원본 → **인용 analysis 출력 전문** → PROPOSALS(요청이 id로 인용한 항목 H-C2b 등 전문 + 방법 이름이 나오는 항목) → past locks → 나머지. job 이력은 인용된 방법의 job만 모두 싣는다(전체 개수 줄은 유지). 지난 요청으로 재현해 보니 172k자로 180k 한도 안에 전부 들어간다. STUCK을 풀고 lateroll@29 감사를 한 번 더 내라. 요청서에 analysis id와 H-id를 그대로 적으면 묶음에 들어간다.

## 2026-11-03 12:10 KST — 사람 결정: lateroll을 감사 PASS 전에 Aircraft에서 돌린다
감사가 결과가 아니라 증거 묶음 문제로 하루 넘게 막혀 Aircraft 이전 검증(08:50 할 일 4)을 막고 있었다. `config.STAGE2_PREAUDIT`에 lateroll을 넣어 stage 2 제출을 허용했다 (CUB final seed 1·2·3 조건은 그대로).
- **지금 제출:** `python -m harness.submit --method lateroll --keep 29 --seed <s> --tier final --dataset aircraft`, s = 1, 2, 3. 코드·sel_args·K는 CUB 그대로 (`stage2_untouched`). final_refs가 같은 seed의 maskedkd·rollout·tome·full을 내고, 이미 있는 run은 건너뛴다.
- **판정은 08:50에 미리 적은 규칙 그대로** (lateroll − maskedkd, lateroll − rollout의 seed 3개 평균과 짝지은 95% CI → "이전된다" / "경향은 같으나 불확실" / "CUB 특화"). 결과를 보고 K·seed를 더하지 않는다.
- 감사 없이 돌린 결과이므로 RESULTS "확정"에는 올리지 않는다. 판정은 RESULTS의 Aircraft 절에 "감사 전"이라고 적는다. lateroll@29 감사는 묶음 수정(3e6a774) 뒤 한 번 더 낸다 (11:50 절).

## 2026-11-03 16:10 KST — 라운드 238 사람 조치 요청에 대한 답
- **stage 2 관문:** `submit.stage2_rule`이 이제 `gate(..., kd_only=True)`를 쓴다 (stage2_boot과 같은 규칙: MaskedKD@98 부족은 발견으로 남기고 stage 2는 진행). 거절당한 **H-A20 세 job (Aircraft lateroll@29 final seed 1·2·3)을 등록 그대로 다시 내라.** 우회하지 않은 판단은 옳았다.
- **감사 묶음:** 섹션별 한도 때문에 diff·analysis 출력·PROPOSALS·RESULTS_TABLE이 여전히 잘렸다. 묶음 한도를 400k로 올리고 섹션 한도도 올렸다. 지난 요청으로 재현하면 309k이고 잘린 곳은 0이다. STUCK을 풀고 lateroll@29 감사를 한 번 더 내라.

## 2026-11-03 16:50 KST — 사람 결정: 새 방향 = TrustMask 강화 (Aircraft에서 탐색, 11/4 09:00 KST까지 1차 결과)
지금까지의 lateroll·이미지별 예산 방향은 닫는다 (H-A20: Aircraft lateroll@29는 rollout 대비 −0.14 → CUB 특화로 기록됨). 사람의 이전 실험 노트 "TrustMask"를 기반으로 강화한다. 루프의 `selectors/trustfilt.py`(H-C14)가 그 기본형이다.

### TrustMask 요약 (사람 노트, Waterbirds, 10/28)
- 신호 두 종류만 쓴다: **공짜**(student forward에서 이미 계산된 attention·rollout)와 **상각**(학습 전 이미지당 1회 계산해 캐시한 teacher attribution, `ctx.teacher_map`). 선택기 비용은 FLOPs에 포함한다.
- 토큰 분류: shared (teacher 중요 + student 이미 봄) 유지 / gap (teacher 중요 + student 아직 못 봄) 유지 / pruned (teacher 중요하지 않음) 제거.
- Waterbirds 결과: 토큰 15%에서 rollout 대비 seed 5개 clean +0.61 (5/5), WGA +1.87 (4/5), corruption +2.75 (5/5). **gap을 넣으면 오히려 나빠짐** (gap 50% −6.28, 10→50% −5.14, gap 0% +1.40) → 지금 TrustMask는 "teacher로 거르는 필터"뿐이다.
- 사람의 관찰: "student attention의 신뢰도(사전학습 여부)에 따라 마스킹 방식이 달라야 한다. 전환 시점을 찾는 것이 중요하다."
- CUB에서 trustfilt@29는 seed 0·4·5 평균 81.33 < lateroll 82.00 (H-C14·16). 단 CUB는 KD 이득이 불확실해 이 비교는 노트의 전제("Full KD가 CE보다 확실히 높은 환경")를 어겼다.

### 하네스 변경 (사람, 이 커밋)
- `config.STAGE2_EXPLORE = True`: Aircraft에서 agent 방법을 바로 돌릴 수 있다 (라벨 `stage2_explore`). 이것은 **탐색 결과**다. 나중의 주장에는 바꾸지 않은 방법을 다른 데이터셋(CUB 또는 Waterbirds)에서 다시 확인해야 한다. Aircraft val로 고른 하이퍼파라미터는 그 사실을 적는다.
- `STAGE2_MAX_RUNNING` 3 → 4 (GPU 전부).

### 할 일 (순서대로, 11/4 00:00 UTC = 09:00 KST에 1차 정리)
1. **노벨티 확인 (지금, 웹 검색, GPU와 병행):** TrustMask와 가까운 선행 연구를 찾아 PROPOSALS에 표로 적는다. 최소: teacher-guided token masking/selection for KD, teacher·student attention 합의(agreement) 기반 토큰 선택, 캐시된 teacher attribution을 쓰는 KD, student attention 신뢰도에 따른 전환, MaskedKD(ECCV 2024)·SBAM(ECCV 2024)·Dynamic KD. 표: 연구 / 무엇으로 고르나 / 비용 / 우리와 다른 점. 같은 것이 있으면 그 사실을 맨 위에 적고, 남는 차이를 노벨티 후보로 적는다. 억지로 차이를 만들지 않는다.
2. **기준 재현 (GPU, Aircraft final K=29 seed 1·2·3):** `trustfilt` 기본형(H-C14 소스 그대로)과 `teacher_topk`(teacher만으로 고름, 비교에 필요). 같은 seed의 maskedkd·rollout·lateroll·CE·Full은 이미 있다.
3. **약점 진단 (분석 채널, GPU 짧게):** 판정 규칙을 결과 전에 PROPOSALS에 적는다.
   - (a) **student 신뢰도 곡선:** 학습 진행에 따라 student rollout과 teacher map이 얼마나 겹치는가 (top-K 겹침, 순위 상관), 그리고 student가 고른 토큰으로 가린 teacher의 fidelity(정답·soft label 유지). 사전학습 student와 그 snapshot(25%·50%)으로 본다. 전환점이 있는지가 질문이다.
   - (b) **gap이 왜 해로운가:** gap 토큰이 가린 teacher의 fidelity를 떨어뜨리는지(teacher 분포 밖 입력), 아니면 student가 그 토큰을 못 따라가는지 분리한다.
4. **강화 시도 (GPU, 한 번에 하나, 같은 seed 짝 비교):** 3의 근거로 고른다. 후보 예시 (강제 아님):
   - 신뢰도 기반 전환: student-teacher 겹침이 문턱보다 낮은 동안(또는 이미지)에는 teacher 필터를 강하게, 높아지면 student만.
   - 필터 폭 적응: 고정 1.5K 대신 이미지별 겹침으로 폭을 정함 (평균 K는 같게).
   - gap을 신뢰도가 오른 뒤에만 조금 넣기.
   기준: 같은 K·같은 seed에서 trustfilt 기본형과 rollout을 모두 이기고, 차이가 seed 편차보다 클 것.
5. **09:00 KST 아침 요약:** PROGRESS 맨 위에 "아침 요약" 한 덩어리: 노벨티 표 결론, 기준 재현 숫자, 진단 결과, 강화 시도 판정, 다음 할 일. 숫자는 RESULTS_TABLE·compare 출력만.

### 지키는 것
- 판정 규칙은 결과 전에 적고 결과를 보고 바꾸지 않는다. 결과 보고 seed를 더하지 않는다.
- 데이터셋별 상수·분기를 selector에 넣지 않는다.
- 비용은 선택기 FLOPs 포함 (캐시 상각 포함)으로 비교한다.

### 보강 (16:55 KST, 사람이 먼저 찾은 가까운 선행 연구 — 할 일 1의 표에 반드시 넣고 원문으로 확인)
- **LeaF, "Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning" (NeurIPS 2025, arXiv 2506.07851):** teacher와 student의 기울기를 비교해 "student는 주목하는데 teacher는 무시하는" 토큰을 찾아 학습 중 제거한다. **TrustMask의 pruned 규칙(student는 보지만 teacher가 무시하는 토큰 제거)과 겹친다.** 다만 LLM 추론의 정확도(교란 토큰 제거)가 목적이고, student 입력을 자른다. teacher 연산을 줄이려는 게 아니다. 차이를 이 축(목적: teacher FLOPs / 대상: teacher 입력 / 신호: 캐시된 attribution vs 기울기 / 영역: ViT 이미지 분류)으로 적는다.
- **MaskedKD (ECCV 2024, arXiv 2302.10494):** student 마지막 층 CLS attention으로 teacher 입력을 가린다. 저자는 이것이 "커리큘럼"처럼 작동한다고 해석한다 (초반에는 쉽고 나중에는 어려운 teacher 감독). → student 신뢰도 곡선(할 일 3a)은 이 해석과 직접 맞닿는다. 그 주장을 확인하거나 반박하는 쪽으로 설계한다.
- **Distilling Token-Pruned Pose Transformer (arXiv 2304.05548):** teacher attention으로 학습 초반 student의 토큰 위치 찾기를 돕는다 ("초반에는 teacher 신호가 낫다"는 주장과 가까움).
- 웹 검색 몇 번으로는 "student attention 신뢰도에 따라 teacher/student 신호를 바꾸는 KD 토큰 선택"과 "teacher-student 합의로 teacher 입력 토큰을 고르는 효율 KD"는 찾지 못했다. 네 검색으로 더 확인하고, 찾으면 그대로 적는다.

### 정정 (17:15 KST, 사람): TrustMask 판정의 비교 대상은 MaskedKD다
위 16:50 절 할 일 4의 기준("trustfilt 기본형과 rollout을 모두 이길 것")을 아래로 바꾼다. 아직 결과를 보기 전의 정정이다.
- **주 비교 = MaskedKD** (사람 노트의 성공 기준 그대로):
  - 같은 토큰 수에서 MaskedKD보다 정확도가 높고, 차이가 seed 편차보다 크다 (seed 1·2·3, 같은 seed 짝 차이와 95% CI).
  - 또는 같은 정확도에서 MaskedKD보다 토큰이 적다 (평균 teacher GFLOPs 기준, 선택기·캐시 비용 포함).
- trustfilt 기본형은 기대가 낮은 출발점이다. 이겨야 할 대상이 아니라 "무엇을 고쳐서 얼마나 좋아졌나"를 보는 기준점이다.
- rollout·lateroll·teacher_topk·random·CE·Full은 맥락으로 함께 보고한다. 판정 조건은 아니다. 다만 rollout에 지면 그 사실을 숨기지 않고 적는다 ("해석 기반 선택 자체의 이득인지, TrustMask만의 이득인지" 구분).
- K는 29를 먼저 본다. MaskedKD가 무너지는 낮은 K(노트: 15%에서 붕괴)가 TrustMask가 이길 수 있는 곳이다. 여유가 있으면 K=12·49도 같은 규칙으로 본다 (K를 결과 보고 고르지 않도록, 볼 K를 먼저 적는다).

### 정정 2 (17:25 KST, 사람): TrustMask가 이제 이 연구의 구조(architecture)다
- 앞으로 만드는 모든 선택기는 **TrustMask 틀 안에서** 만든다. 틀 = (1) 공짜 신호(student forward)와 상각 신호(캐시된 teacher attribution)만 쓴다, (2) 토큰을 shared / gap / pruned로 나눈다, (3) 이미지별 토큰 수를 조절할 수 있다, (4) 선택기 비용은 FLOPs에 넣는다. 이 틀 밖의 새 계열(rollout 층 바꾸기, 이미지별 몰아주기 등 이미 닫은 방향)은 만들지 않는다.
- 너의 일은 **TrustMask를 계속 강해지게 하는 것**이다. 매번: 약점 하나를 진단한다 → 그 약점을 고치는 구성요소 하나를 설계한다 (PROPOSALS에 근거와 판정 규칙을 먼저 적는다) → 같은 seed 짝으로 이전 TrustMask 버전과 MaskedKD에 비교한다 → 이기면 그 버전을 새 기준 TrustMask로 삼는다 (`selectors/trustmask_v<N>.py`, parent는 이전 버전). 지면 이유를 적고 다른 약점으로 간다.
- 기본형(trustfilt)은 v0이다. 기대가 낮은 출발점이므로 Aircraft seed 1·2·3을 한 번만 돌려 기준점으로 쓰고, 그 뒤로는 고치는 데 GPU를 쓴다.
- 먼저 볼 약점 (사람 노트에서): gap이 해롭다 (원래 아이디어의 절반이 빠짐), student attention 신뢰도에 따라 방식이 달라야 한다 (전환점), 이미지별 토큰 수 조절이 아직 없다 (노트의 "얼마나?" 절). 순서는 진단 결과로 네가 정한다.
- 판정은 정정 1 그대로 MaskedKD 대비다. 각 버전의 변경 이력(무엇을 왜 바꿨고 결과가 어땠나)을 RESULTS에 "TrustMask 버전" 표로 남긴다. 이 표가 나중에 "왜 이렇게 설계했나"의 근거가 된다.
