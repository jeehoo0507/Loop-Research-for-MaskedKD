# AUDIT (agent/loop.sh가 씀)

## 2026-10-31 01:17 audit by codex

### request
# AUDIT_REQUEST — [claude] 2026-10-31 라운드 38

## 주장 (1줄)
Waterbirds의 같은 teacher 토큰 예산 K=12에서, student sufficiency로 teacher 입력 원판 위치를 고르는 `suffdisc`가 final 3 seed(1·2·3)에서 CLAIM_REFS(maskedkd, rollout, tome, grid, center, random)를 모두 이긴다. WGA는 6개 모두에서 WIN이고 정확도는 어느 것에도 지지 않는다.

## 범위 (주장하지 않는 것)
- 'Full KD 허용 오차 안의 최소 K'(주 지표)는 주장하지 않는다. full@196 대비 정확도 −1.53 %p, WGA −5.02 %p다. CI는 0을 포함하지만 평균이 허용 오차(1.00/2.45) 밖이다.
- 같은 total GFLOPs에서 우위도 주장하지 않는다. 아래 caveat 2를 본다.

## run id
- suffdisc@12 (selector_sha 0f060023…): final_suffdisc_k12_s1_1031_002441_5d86 (90.08/65.41), final_suffdisc_k12_s2_1031_002448_664e (89.66/60.90), final_suffdisc_k12_s3_1031_002453_9ebe (89.16/60.90). 3-seed 89.63±0.46 / 62.40±2.60, 최악 그룹은 모두 2(물새×육지)다.
- 비교군 (RESULTS_TABLE final 행, 3-seed 정확도/WGA):
  - maskedkd 84.21/22.31: final_maskedkd_k12_s1_1030_201811_e49a, _s2_1030_212126_ad00, _s3_1030_220958_cc4e
  - rollout 88.10/50.88: final_rollout_k12_s1_1030_201811_6293, _s2_1030_212126_33d1, _s3_1030_220958_d961
  - tome 89.96/50.88 (not matched, teacher cost ×2.69): final_tome_k12_s1_1030_201811_9fb4, _s2_1030_212126_9500, _s3_1030_220958_918a
  - grid 83.10/35.34: final_grid_k12_s1_1030_201811_dc71, _s2_1030_212126_9336, _s3_1030_220958_14eb
  - center (parent) 88.69/53.64: final_center_k12_s1_1030_201811_341b, _s2_1030_212126_5486, _s3_1030_220958_413c
  - random 83.85/37.34: final_random_k12_s1_1030_201811_19bf, _s2_1030_212126_b5bc, _s3_1030_220958_e1db
  - full@196 91.16/67.42: final_full_k196_s1_1030_201811_b764, _s2_1030_212126_a1dc, _s3_1030_220958_8d7e

## Core comparison (RESULTS_TABLE, harness 자동 계산, seeds 1,2,3, paired 95% CI)
| vs | acc | WGA | 판정 |
|---|---|---|---|
| maskedkd@12 | +5.42 [+4.18, +6.66] WIN | +40.10 [+30.88, +49.32] WIN | WIN |
| rollout@12 | +1.53 [−2.52, +5.58] n.s. | +11.53 [+5.82, +17.23] WIN | WIN (WGA) |
| tome@12 | −0.33 [−2.57, +1.91] n.s. | +11.53 [+0.58, +22.47] WIN | WIN (WGA) |
| grid@12 | +6.53 [+5.51, +7.56] WIN | +27.06 [+20.59, +33.53] WIN | WIN |
| center@12 (parent) | +0.95 [−0.79, +2.68] n.s. | +8.77 [+4.06, +13.47] WIN | WIN (WGA) |
| random@12 | +5.78 [+3.43, +8.14] WIN | +25.06 [+17.51, +32.61] WIN | WIN |
| full@196 | −1.53 [−3.49, +0.43] n.s. | −5.02 [−12.56, +2.53] n.s. | — |

## harness.compare 출력 (정확도, 2026-10-31 라운드 38에 실행)
```
suffdisc:12 vs maskedkd:12  val acc A-B = +5.42 %p, 95% CI [+4.18, +6.66] (n=3)  GFLOPs A 15.02 B 10.07 ratio 1.491
suffdisc:12 vs rollout:12   val acc A-B = +1.53 %p, 95% CI [-2.52, +5.58] (n=3)  GFLOPs A 15.02 B 10.08 ratio 1.491
suffdisc:12 vs tome:12      val acc A-B = -0.33 %p, 95% CI [-2.57, +1.91] (n=3)  GFLOPs A 15.02 B 12.00 ratio 1.252
suffdisc:12 vs grid:12      val acc A-B = +6.53 %p, 95% CI [+5.51, +7.56] (n=3)  GFLOPs A 15.02 B 10.09 ratio 1.489
suffdisc:12 vs center:12    val acc A-B = +0.95 %p, 95% CI [-0.79, +2.68] (n=3)  GFLOPs A 15.02 B 10.09 ratio 1.489
suffdisc:12 vs random:12    val acc A-B = +5.78 %p, 95% CI [+3.43, +8.14] (n=3)  GFLOPs A 15.02 B 10.09 ratio 1.489
suffdisc:12 vs full:196     val acc A-B = -1.53 %p, 95% CI [-3.49, +0.43] (n=3)  GFLOPs A 15.02 B 42.77 ratio 0.351
```
(seed별 run 대응은 compare 출력의 seed 1/2/3 줄과 위 run id 목록이 같다. WGA CI는 compare가 출력하지 않아 RESULTS_TABLE Core comparison에서 가져왔다.)

## 근거 분석 id
- 설계 근거: analysis_proxy_validation_1030_170914_4668 (최악 그룹 masked-teacher KL이 student WGA 순위를 예측, |Spearman| 0.94, 16 run), analysis_window_fidelity_1030_165418_e276 (K=12에서 중앙 원판을 ≤2 패치 옮기면 teacher KL 0.694→0.049, attention 중심으로는 그 위치를 못 찾음), analysis_student_sufficiency_1030_173025_6f96 (student sufficiency가 teacher-best 위치를 약하게 가리킴).
- 승격 전 분해: analysis_suffdisc_decomp_1030_202612_878c. 겹침 suffdisc∩maskedkd 0.11~0.14, ∩rollout 0.22~0.24, ∩center 0.37~0.38. centre_frac 0.024~0.032라 원판이 거의 항상 중앙에서 옮겨진다. train/eval 픽 일치율은 0.34~0.43이다.
- 대체 실험 (parent suffdisc): jitdisc (같은 25개 후보 중 균등 무작위). screen_jitdisc_k12_s0_1030_195912_ead1, screen_jitdisc_k12_s1_1030_195921_48c7, 2-seed 88.53/53.01로 center 수준이다. 미리 정한 기준(≤56.6이면 신호)에 따라 이득은 '원판을 흔든 것'이 아니라 sufficiency 신호에서 온다.
- screen 짝 비교 (H-W22, suffdisc − center, seed 0~3): WGA +2.26/+12.03/+12.79/+13.53, acc +0.58/+1.92/+1.83/+1.08 (run id는 PROGRESS 라운드 36).
- teachability: analysis_suff_teacher_align_1031_003305_e9b8 (H-W20). 최악 그룹에서 student sufficiency 순위와 teacher 충실도 순위의 Spearman이 같은 seed의 center student보다 +0.03~+0.09 높다(s0~s3). 이 정렬은 학습할수록 커진다(e25 0.235 → e50 0.269 → final 0.315). 그 pick에서 masked teacher WGA는 45.11~49.62이고, 고정 centre 원판은 41.35다. 해석: teacher는 소수 그룹에서 '물체 위의 작은 연속 영역 하나'를 볼 때 가장 충실하다. student 자신의 예측이 충분한 영역은 그 위치를 라벨 없이 추정한 값이다.

## 노벨티 (선행 연구 대비)
- MaskedKD·rollout은 attention 순위로 흩어진 토큰을 고른다. suffdisc는 attention 순위를 쓰지 않는다. ERASER sufficiency / RISE insertion식 'keep-only 충실도'를 student에서 재서 연속 원판의 위치를 정한다. 원래 설명 평가용이던 sufficiency를 KD teacher 입력 선택에 쓴 예는 검색(라운드 1·36)에서 찾지 못했다. 가까운 것은 arXiv 2411.17046(데이터 없는 증류의 중앙 편향 patch 합성, 실제 입력 선택은 아님)과 arXiv 2107.01378(patch 수준 manifold 증류)이다.
- center(공간 규칙만)와 jitdisc(같은 후보의 무작위)를 대조군으로 두어 이득이 신호에서 온다는 것을 분리했다.

## Caveats (감사관이 볼 것)
1. 공식 Parent check(screen seed 0)는 UNPROVEN이다(+0.58/+2.26). final 3-seed에서는 parent center 대비 WGA +8.77 [+4.06, +13.47] WIN이고, screen 4-seed 짝 비교도 모두 양수다.
2. **비용:** total 15.02 GFLOPs/img (selector 4.93 = 후보 25개 × student K-토큰 forward, teacher 2.45). maskedkd의 1.49배, tome의 1.25배다. 비슷한 total cost의 screen seed 0 결과는 rollout@29 89.41/64.28 (12.99 GFLOPs, 2-seed), rollout@49 89.66/64.66 (16.45 GFLOPs)로 suffdisc@12 final 89.63/62.40과 비슷하다(tier가 달라 직접 비교는 아니다). 그래서 주장은 'teacher 토큰 예산 K 고정'이라는 축에서만 한다. 'compute 대비 우위'는 주장하지 않는다.
3. tome@12는 not matched(teacher cost ×2.69)라 tome에 유리한 비교다. 그래도 total GFLOPs는 tome이 더 적다(12.00).
4. Search effort: 이 K에서 screen 시도가 많다(RESULTS_TABLE Search effort). suffdisc는 seed 0만 보고 한때 닫혔다가(라운드 11) seed 1 결과로 다시 열렸다. 그 뒤 판정 기준은 미리 정해 두었다(PROPOSALS 라운드 18·20·36·37).
5. 같은 계열의 다른 K(29·49) 결과는 없다. 이 주장은 K=12에 한정한다.
6. teachability 효과 크기는 작다(rho 차이 0.03~0.09). teacher-best까지의 거리도 크다(KL 0.46~0.56 대 0.04).

### verdict
CLAIM: <the claim, one line>
VERDICT: PASS | FAIL | NEEDS-MORE
REASON: <one or two lines, with run ids>
CLAIM: Waterbirds K=12에서 suffdisc가 final seeds 1·2·3으로 6개 비교군 모두에 WGA 우위를 보이며 정확도에서는 지지 않는다.
VERDICT: NEEDS-MORE
REASON: RESULTS_TABLE은 WGA 6승을 뒷받침하며 final_suffdisc_k12_s1_1031_002441_5d86 및 s2·s3의 수치·digest도 일치한다. 그러나 비교군 전체의 seed 2·3 metrics.json(예: final_tome_k12_s2_1030_212126_9500)이 빠져 완료·epoch·무결성을 검증할 수 없고, 전체 selector 실행 목록과 analysis_suffdisc_decomp_1030_202612_878c 등 설계·분해 분석 출력도 잘려 있다. 이 자료와 harness.test_eval 미사용 증거가 필요하다.
CLAIM: Waterbirds K=12에서 suffdisc가 final seeds 1·2·3으로 6개 비교군 모두에 WGA 우위를 보이며 정확도에서는 지지 않는다.
VERDICT: NEEDS-MORE
REASON: RESULTS_TABLE은 WGA 6승을 뒷받침하며 final_suffdisc_k12_s1_1031_002441_5d86 및 s2·s3의 수치·digest도 일치한다. 그러나 비교군 전체의 seed 2·3 metrics.json(예: final_tome_k12_s2_1030_212126_9500)이 빠져 완료·epoch·무결성을 검증할 수 없고, 전체 selector 실행 목록과 analysis_suffdisc_decomp_1030_202612_878c 등 설계·분해 분석 출력도 잘려 있다. 이 자료와 harness.test_eval 미사용 증거가 필요하다.
CLAIM: Waterbirds K=12에서 suffdisc가 final seeds 1·2·3으로 6개 비교군 모두에 WGA 우위를 보이며 정확도에서는 지지 않는다.
VERDICT: NEEDS-MORE
REASON: RESULTS_TABLE은 WGA 6승을 뒷받침하며 final_suffdisc_k12_s1_1031_002441_5d86 및 s2·s3의 수치·digest도 일치한다. 그러나 비교군 전체의 seed 2·3 metrics.json(예: final_tome_k12_s2_1030_212126_9500)이 빠져 완료·epoch·무결성을 검증할 수 없고, 전체 selector 실행 목록과 analysis_suffdisc_decomp_1030_202612_878c 등 설계·분해 분석 출력도 잘려 있다. 이 자료와 harness.test_eval 미사용 증거가 필요하다.

## 2026-10-31 01:27 audit by codex

### request
# AUDIT_REQUEST — [claude] 2026-10-31 라운드 39 (재요청, 1/3부)

## 왜 나눠서 요청하나
지난 감사(2026-10-31 01:17, codex, NEEDS-MORE)의 이유는 비교군 seed 2·3 metrics.json과 분석 출력이 증거 묶음에 없다는 것이었다. 원인은 두 가지다. (1) 지난 요청에서 seed 2·3 run id를 `_s2_…`처럼 줄여 써서 harness/evidence.py의 정규식이 잡지 못했다. (2) metrics.json은 run당 약 6.3k자이고, 묶음 한도는 120k자다. 비교군 21개 + suffdisc 3개를 한 번에 넣으면 한도를 넘는다.
그래서 같은 주장을 비교군 기준으로 세 부분으로 나눈다. 각 부분은 suffdisc final 3개, 비교군 2개의 final 3 seed 전체 id, 분석 1개만 붙여 스스로 검증할 수 있게 했다.
- 1/3부(이번): maskedkd, rollout + 분해 분석
- 2/3부: tome, grid + teachability 분석
- 3/3부: center(parent), random + 설계 근거 분석, 대체 실험(jitdisc)
세 부분이 모두 PASS일 때만 RESULTS.md "확정"에 전체 주장을 올린다.

## 주장 (1/3부, 1줄)
Waterbirds의 teacher 토큰 예산 K=12에서, `suffdisc`(student sufficiency로 teacher 입력 연속 원판의 위치를 고름)가 final seeds 1·2·3에서 `maskedkd`를 정확도와 WGA 모두에서 이기고, `rollout`을 WGA에서 이기며 정확도에서는 지지 않는다(n.s.).

## run id (모두 status done, epochs_done 100, harness_digest 033df9786a4a…)
| run | 정확도 | WGA |
|---|---|---|
| final_suffdisc_k12_s1_1031_002441_5d86 | 90.08 | 65.41 |
| final_suffdisc_k12_s2_1031_002448_664e | 89.66 | 60.90 |
| final_suffdisc_k12_s3_1031_002453_9ebe | 89.16 | 60.90 |
| final_maskedkd_k12_s1_1030_201811_e49a | 84.15 | 21.05 |
| final_maskedkd_k12_s2_1030_212126_ad00 | 84.24 | 23.31 |
| final_maskedkd_k12_s3_1030_220958_cc4e | 84.24 | 22.56 |
| final_rollout_k12_s1_1030_201811_6293 | 86.99 | 51.88 |
| final_rollout_k12_s2_1030_212126_33d1 | 87.99 | 51.88 |
| final_rollout_k12_s3_1030_220958_d961 | 89.32 | 48.87 |

## Core comparison (RESULTS_TABLE, harness 자동 계산, seeds 1,2,3, paired 95% CI)
- suffdisc@12 vs maskedkd@12: acc +5.42 [+4.18, +6.66] WIN, WGA +40.10 [+30.88, +49.32] WIN
- suffdisc@12 vs rollout@12: acc +1.53 [−2.52, +5.58] n.s., WGA +11.53 [+5.82, +17.23] WIN
- 정확도 차이 +1.53은 promotion threshold 0.76보다 크지만 CI가 0을 포함하므로 WIN으로 주장하지 않는다.

## 분해 (승격 전, 항목 15)
analysis_suffdisc_decomp_1030_202612_878c: 같은 이미지에서 suffdisc∩maskedkd 겹침 0.11~0.14, ∩rollout 0.22~0.24, ∩center 0.37~0.38(K 대비). centre_frac 0.024~0.032라 원판이 거의 항상 중앙에서 옮겨진다. 즉 suffdisc가 고르는 토큰은 maskedkd·rollout이 고르는 토큰과 대부분 다르다. 대체 실험(jitdisc, 같은 후보 중 무작위)은 3/3부에 붙인다.

## 비용과 tome (항목 10)
total 15.02 GFLOPs/img로 maskedkd(10.07)의 1.49배다(selector 4.93: 후보 원판 25개 × student K-토큰 forward). 그래서 주장은 'teacher 토큰 예산 K 고정' 축에만 한다. compute 대비 우위는 주장하지 않는다. tome@12 대비 정확도는 −0.33 [−2.57, +1.91] n.s.이고 WGA는 +11.53 [+0.58, +22.47] WIN이다(2/3부에서 검증). tome@12는 not matched(teacher 비용 ×2.69)이지만 total GFLOPs는 12.00으로 suffdisc보다 적다.

## 범위 (주장하지 않는 것)
- 주 지표('Full 허용 오차 안의 최소 K')는 주장하지 않는다. full@196 대비 acc −1.53 [−3.49, +0.43], WGA −5.02 [−12.56, +2.53](RESULTS_TABLE Core comparison). CI는 0을 포함하지만 평균이 허용 오차(1.00/2.45) 밖이다.
- K=12 한정이다. suffdisc@29 screen s0은 실행 중이다.

## 무결성 메모
- test_eval: harness/test_eval.py는 `final/test_*.json`에 결과를 쓴다. 저장소에 `final/` 디렉터리가 없다. agent 커밋 중 harness/·tests/를 바꾼 것은 없다(증거 묶음의 git log -- harness tests).
- Search effort: K=12에서 screen 시도가 많다. suffdisc는 seed 0에서 parent center 대비 UNPROVEN(+0.58/+2.26)이었고 seed 1 결과로 다시 열렸다. 그 뒤 screen 4-seed 짝 비교(center 대비 WGA +2.26/+12.03/+12.79/+13.53)를 미리 정한 기준으로 보고 final로 올렸다. final은 새 seed 1·2·3만 쓴다.

## 노벨티 (가장 가까운 선행)
MaskedKD·rollout은 attention 순위로 흩어진 토큰을 고른다. suffdisc는 ERASER sufficiency / RISE insertion식 'keep-only 충실도'를 student에서 재서 연속 원판의 위치를 정한다. 설명 평가용 sufficiency를 KD teacher 입력 선택에 쓴 예는 찾지 못했다. 가장 가까운 것은 arXiv 2411.17046(데이터 없는 증류의 중앙 편향 patch 합성)과 arXiv 2107.01378(patch 수준 manifold 증류)이다.

### verdict
CLAIM: <the claim, one line>
VERDICT: PASS | FAIL | NEEDS-MORE
REASON: <one or two lines, with run ids>
CLAIM: Waterbirds K=12에서 suffdisc는 final seeds 1·2·3의 maskedkd를 정확도·WGA 모두에서 이기고, rollout을 WGA에서 이기며 정확도에서는 지지 않는다(n.s.).
VERDICT: FAIL
REASON: 보고된 평균 차이와 WIN 수치는 일치하지만, rollout 정확도 CI [−2.52, +5.58]은 의미 있는 손실도 허용하므로 “지지 않는다”를 입증하지 못한다. 실제 final_suffdisc_k12_s3_1031_002453_9ebe의 정확도는 final_rollout_k12_s3_1030_220958_d961보다 낮다. “정확도 차이는 불확실하다”로 수정해야 한다.
CLAIM: Waterbirds K=12에서 suffdisc는 final seeds 1·2·3의 maskedkd를 정확도·WGA 모두에서 이기고, rollout을 WGA에서 이기며 정확도에서는 지지 않는다(n.s.).
VERDICT: FAIL
REASON: 보고된 평균 차이와 WIN 수치는 일치하지만, rollout 정확도 CI [−2.52, +5.58]은 의미 있는 손실도 허용하므로 “지지 않는다”를 입증하지 못한다. 실제 final_suffdisc_k12_s3_1031_002453_9ebe의 정확도는 final_rollout_k12_s3_1030_220958_d961보다 낮다. “정확도 차이는 불확실하다”로 수정해야 한다.
CLAIM: Waterbirds K=12에서 suffdisc는 final seeds 1·2·3의 maskedkd를 정확도·WGA 모두에서 이기고, rollout을 WGA에서 이기며 정확도에서는 지지 않는다(n.s.).
VERDICT: FAIL
REASON: 보고된 평균 차이와 WIN 수치는 일치하지만, rollout 정확도 CI [−2.52, +5.58]은 의미 있는 손실도 허용하므로 “지지 않는다”를 입증하지 못한다. 실제 final_suffdisc_k12_s3_1031_002453_9ebe의 정확도는 final_rollout_k12_s3_1030_220958_d961보다 낮다. “정확도 차이는 불확실하다”로 수정해야 한다.

## 2026-10-31 01:56 audit by codex

### request
# AUDIT_REQUEST — [claude] 2026-10-31 라운드 40 (1/3부 재요청, 문구 수정)

## 지난 판정과 고친 점
지난 판정(AUDIT.md 2026-10-31 01:27, codex)은 FAIL이었다. 평균 차이와 WIN 수치는 확인됐다. 문제는 "rollout에 정확도에서 지지 않는다(n.s.)"라는 문구였다. 감사관 지적대로, CI [−2.52, +5.58]은 의미 있는 손실도 허용한다. 또 seed 3에서는 suffdisc 정확도(final_suffdisc_k12_s3_1031_002453_9ebe 89.16)가 rollout(final_rollout_k12_s3_1030_220958_d961 89.32)보다 낮다.
그래서 주장에서 비열등성 표현을 뺐다. 정확도 차이는 "불확실하다"로 적는다. 수치, run id, 증거는 바꾸지 않았다.
나머지 부분(2/3부 tome·grid, 3/3부 center·random)도 같은 기준으로 쓴다. tome 대비 정확도(−0.33 [−2.57, +1.91])와 center 대비 정확도(+0.95, n.s.)도 "불확실"로 적을 것이다.

## 주장 (1/3부, 1줄)
Waterbirds의 teacher 토큰 예산 K=12에서, `suffdisc`(student sufficiency로 teacher 입력 연속 원판의 위치를 고름)는 final seeds 1·2·3에서 `maskedkd`를 정확도와 WGA 모두에서 이기고(paired 95% CI가 0 밖), `rollout`을 WGA에서 이긴다(+11.53 [+5.82, +17.23]). rollout 대비 정확도 차이는 불확실하다(+1.53 [−2.52, +5.58]이고 seed 3에서는 suffdisc가 −0.17 낮다).

## run id (모두 status done, epochs_done 100, harness_digest 033df9786a4a…)
| run | 정확도 | WGA |
|---|---|---|
| final_suffdisc_k12_s1_1031_002441_5d86 | 90.08 | 65.41 |
| final_suffdisc_k12_s2_1031_002448_664e | 89.66 | 60.90 |
| final_suffdisc_k12_s3_1031_002453_9ebe | 89.16 | 60.90 |
| final_maskedkd_k12_s1_1030_201811_e49a | 84.15 | 21.05 |
| final_maskedkd_k12_s2_1030_212126_ad00 | 84.24 | 23.31 |
| final_maskedkd_k12_s3_1030_220958_cc4e | 84.24 | 22.56 |
| final_rollout_k12_s1_1030_201811_6293 | 86.99 | 51.88 |
| final_rollout_k12_s2_1030_212126_33d1 | 87.99 | 51.88 |
| final_rollout_k12_s3_1030_220958_d961 | 89.32 | 48.87 |

## Core comparison (RESULTS_TABLE, harness 자동 계산, seeds 1,2,3, paired 95% CI)
- suffdisc@12 vs maskedkd@12: acc +5.42 [+4.18, +6.66] WIN, WGA +40.10 [+30.88, +49.32] WIN
- suffdisc@12 vs rollout@12: acc +1.53 [−2.52, +5.58] no difference(불확실), WGA +11.53 [+5.82, +17.23] WIN
- seed별 rollout 대비 정확도 차이: s1 +3.09, s2 +1.67, s3 −0.17. 3개 중 2개가 양수이지만 n=3이라 CI가 넓다. 비열등성은 주장하지 않는다.

## 분해 (승격 전)
analysis_suffdisc_decomp_1030_202612_878c: 같은 이미지에서 suffdisc가 고른 토큰과 겹치는 비율(K 대비)은 maskedkd 0.11~0.14, rollout 0.22~0.24, center 0.37~0.38이다. centre_frac은 0.024~0.032라 원판이 거의 항상 중앙에서 옮겨진다. suffdisc가 고르는 토큰은 maskedkd·rollout이 고르는 토큰과 대부분 다르다. 대체 실험(jitdisc, 같은 후보 중 무작위)은 3/3부에 붙인다.

## 비용
total 15.02 GFLOPs/img로 maskedkd(10.07)의 1.49배다(selector 4.93: 후보 원판 25개 × student K-토큰 forward). 주장은 'teacher 토큰 예산 K 고정' 축에만 한다. compute 대비 우위는 주장하지 않는다.
비용을 맞춘 변형 r=1(후보 9개, total 11.86 GFLOPs/img)의 screen 결과는 screen_suffdisc_k12_s0_1031_011600_6b15 88.57/55.64, screen_suffdisc_k12_s1_1031_011604_6add 88.99/56.39다. 같은 seed의 center screen(screen_center_k12_s0_1030_135953_c64e 88.74/56.39, screen_center_k12_s1_1030_185949_1766 87.99/51.88) 대비 WGA는 2-seed 평균 +1.88이다. promotion threshold 2.45에 못 미친다. 따라서 이득에는 ±2 탐색 범위(비용 1.49배)가 필요하다. 이 결과는 비용 caveat을 줄이지 못한다.

## 범위 (주장하지 않는 것)
- 주 지표('Full 허용 오차 안의 최소 K')는 주장하지 않는다. full@196 대비 acc −1.53 [−3.49, +0.43], WGA −5.02 [−12.56, +2.53]다. CI는 0을 포함하지만 평균이 허용 오차(1.00/2.45) 밖이다.
- rollout 대비 정확도 우위나 비열등성은 주장하지 않는다.
- K=12 한정이다. suffdisc@29·@49 screen s0은 실행 중이다.

## 무결성 메모
- test_eval: harness/test_eval.py는 `final/test_*.json`에 결과를 쓴다. 저장소에 `final/` 디렉터리가 없다. agent 커밋 중 harness/·tests/를 바꾼 것은 없다.
- Search effort: K=12에서 screen 시도가 많다. suffdisc는 seed 0에서 parent center 대비 UNPROVEN(+0.58/+2.26)이었다. 그 뒤 screen 4-seed 짝 비교(center 대비 WGA +2.26/+12.03/+12.79/+13.53)를 미리 정한 기준으로 보고 final로 올렸다. final은 새 seed 1·2·3만 쓴다.

## 노벨티 (가장 가까운 선행)
MaskedKD·rollout은 attention 순위로 흩어진 토큰을 고른다. suffdisc는 ERASER sufficiency / RISE insertion식 'keep-only 충실도'를 student에서 재서 연속 원판의 위치를 정한다. 설명 평가용 sufficiency를 KD teacher 입력 선택에 쓴 예는 찾지 못했다. 가장 가까운 것은 arXiv 2411.17046(데이터 없는 증류의 중앙 편향 patch 합성)과 arXiv 2107.01378(patch 수준 manifold 증류)이다.

### verdict
CLAIM: <the claim, one line>
VERDICT: PASS | FAIL | NEEDS-MORE
REASON: <one or two lines, with run ids>
CLAIM: Waterbirds K=12에서 suffdisc는 final seeds 1·2·3의 maskedkd 대비 정확도·WGA, rollout 대비 WGA에서 이기며, rollout 대비 정확도 차이는 불확실하다.
VERDICT: NEEDS-MORE
REASON: final_suffdisc_k12_s1_1031_002441_5d86, final_suffdisc_k12_s2_1031_002448_664e, final_suffdisc_k12_s3_1031_002453_9ebe와 maskedkd·rollout 비교군의 완료 상태·수치·digest는 일치한다. 그러나 필수 분해 분석 analysis_suffdisc_decomp_1030_202612_878c의 실제 출력이 누락됐고, RESULTS_TABLE이 잘려 전체 suffdisc 탐색 이력을 검증할 수 없다.
CLAIM: Waterbirds K=12에서 suffdisc는 final seeds 1·2·3의 maskedkd 대비 정확도·WGA, rollout 대비 WGA에서 이기며, rollout 대비 정확도 차이는 불확실하다.
VERDICT: NEEDS-MORE
REASON: final_suffdisc_k12_s1_1031_002441_5d86, final_suffdisc_k12_s2_1031_002448_664e, final_suffdisc_k12_s3_1031_002453_9ebe와 maskedkd·rollout 비교군의 완료 상태·수치·digest는 일치한다. 그러나 필수 분해 분석 analysis_suffdisc_decomp_1030_202612_878c의 실제 출력이 누락됐고, RESULTS_TABLE이 잘려 전체 suffdisc 탐색 이력을 검증할 수 없다.
CLAIM: Waterbirds K=12에서 suffdisc는 final seeds 1·2·3의 maskedkd 대비 정확도·WGA, rollout 대비 WGA에서 이기며, rollout 대비 정확도 차이는 불확실하다.
VERDICT: NEEDS-MORE
REASON: final_suffdisc_k12_s1_1031_002441_5d86, final_suffdisc_k12_s2_1031_002448_664e, final_suffdisc_k12_s3_1031_002453_9ebe와 maskedkd·rollout 비교군의 완료 상태·수치·digest는 일치한다. 그러나 필수 분해 분석 analysis_suffdisc_decomp_1030_202612_878c의 실제 출력이 누락됐고, RESULTS_TABLE이 잘려 전체 suffdisc 탐색 이력을 검증할 수 없다.

## 2026-10-31 02:11 audit by codex

### request
# AUDIT_REQUEST — [claude] 2026-10-31 라운드 41 (1/3부 재구성)

## 지난 판정과 고친 점
지난 판정(AUDIT.md 01:56, codex)은 NEEDS-MORE였다. 수치·상태·digest는 맞았다. 빠진 것은 두 가지였다. (1) 분해 분석 출력이 없었다. 원인은 묶음 한도 120k자다. 지난 요청은 run id 13개를 인용했고(run당 약 7k자), 맨 끝에 붙는 분석 출력이 잘렸다. (2) RESULTS_TABLE이 12k자에서 잘려 suffdisc screen 이력 행이 보이지 않았다.
이번 요청은 run 10개와 분석 1개만 인용한다. 구성은 suffdisc final 3개, maskedkd final 3개, 그리고 같은 설정의 suffdisc screen 4개 전부다. rollout 비교는 2/3부(tome과 함께)로 옮겼다.

## 주장 (1/3부, 1줄)
Waterbirds K=12에서 `suffdisc`(student sufficiency로 teacher 입력 연속 원판의 위치를 고름)는 final seeds 1·2·3에서 `maskedkd`를 정확도와 WGA 모두에서 이긴다(paired 95% CI가 0 밖). 같은 설정의 screen seeds 0~3 결과는 final에서도 유지된다.

## run id (모두 status done, epochs_done 100, harness_digest 033df9786a4a, suffdisc는 selector_sha 0f060023d021…)
| run | tier/seed | 정확도 | WGA |
|---|---|---|---|
| final_suffdisc_k12_s1_1031_002441_5d86 | final 1 | 90.08 | 65.41 |
| final_suffdisc_k12_s2_1031_002448_664e | final 2 | 89.66 | 60.90 |
| final_suffdisc_k12_s3_1031_002453_9ebe | final 3 | 89.16 | 60.90 |
| final_maskedkd_k12_s1_1030_201811_e49a | final 1 | 84.15 | 21.05 |
| final_maskedkd_k12_s2_1030_212126_ad00 | final 2 | 84.24 | 23.31 |
| final_maskedkd_k12_s3_1030_220958_cc4e | final 3 | 84.24 | 22.56 |
| screen_suffdisc_k12_s0_1030_173118_504a | screen 0 | 89.32 | 58.65 |
| screen_suffdisc_k12_s1_1030_185953_9455 | screen 1 | 89.91 | 63.91 |
| screen_suffdisc_k12_s2_1030_233019_98dd | screen 2 | 90.24 | 62.41 |
| screen_suffdisc_k12_s3_1030_233028_39ad | screen 3 | 89.99 | 68.42 |

## Core comparison (RESULTS_TABLE, seeds 1,2,3, paired 95% CI)
- suffdisc@12 vs maskedkd@12: acc +5.42 [+4.18, +6.66] WIN, WGA +40.10 [+30.88, +49.32] WIN. GFLOPs 비율 1.491.

## 탐색 이력 (항목 7)
"Search effort"에 suffdisc는 2 configs로 나온다. (a) 기본 설정(sel_args {}): 위 screen 4개, 평균 89.87/63.35. final 3-seed 평균은 89.63/62.40이다. 그래서 screen의 이득은 새 seed에서도 유지된다. (b) r=1(후보 9개, 비용 맞춤 시도): screen seed 0·1, 평균 88.78/56.02. 기본 설정보다 낮고 final로 올리지 않았다. 이 run id들은 한도 때문에 3/3부에 묶는다. 다른 K: suffdisc@29 screen seed 0은 90.16/61.65다. rollout@29 대비 WGA −2.63, center@29 대비 +1.50으로 미리 정한 기준 2.45에 못 미친다. 그래서 주장은 K=12에 한정한다. 자식 jitdisc(같은 후보 중 무작위 원판)는 3/3부의 대체 실험이다.

## 분해 (항목 15)
analysis_suffdisc_decomp_1030_202612_878c(summary.json): suffdisc가 고른 토큰 중 maskedkd와 겹치는 비율은 0.11~0.14다. rollout과는 0.22~0.24, center와는 0.37~0.38이다. centre_frac은 0.024~0.032라 원판은 거의 항상 중앙 밖으로 옮겨진다. 이득은 maskedkd 토큰을 다시 고르는 데서 오지 않는다. 대부분 maskedkd가 고르지 않은 연속 영역에서 온다.

## tome과 다른 기준선 (항목 10·13)
tome 대비 WGA는 +11.53 [+0.58, +22.47] WIN이고, 정확도 차이는 −0.33 [−2.57, +1.91]로 불확실하다. rollout·grid·center·random은 2/3부와 3/3부에서 다룬다. 이 1/3부의 PASS만으로 '해석 신호의 이득'을 주장하지 않는다.

## 비용과 범위
total 15.02 GFLOPs/img로 maskedkd(10.07)의 1.49배다. selector 비용은 4.93이고, 이것은 후보 원판 25개 × student K-토큰 forward다. 주장은 'teacher 토큰 예산 K 고정' 축에만 한다. compute 대비 우위와 주 지표('Full 허용 오차 안의 최소 K')는 주장하지 않는다. full 대비 차이는 acc −1.53, WGA −5.02이고 평균이 허용 오차 밖이다.

## 무결성
agent 커밋 중 harness/·tests/를 바꾼 것은 없다. test_eval은 쓰지 않았다(`final/` 디렉터리 없음).

## 노벨티
ERASER sufficiency / RISE insertion식 keep-only 충실도를 student에서 재서 연속 원판의 위치를 정한다. 설명 평가용 sufficiency를 KD teacher 입력 선택에 쓴 예는 찾지 못했다. 가장 가까운 선행은 MaskedKD(attention top-K), arXiv 2411.17046(중앙 편향 patch 합성), arXiv 2107.01378(patch 수준 증류)이다.

### verdict
CLAIM: <the claim, one line>
VERDICT: PASS | FAIL | NEEDS-MORE
REASON: <one or two lines, with run ids>

### verdict
CLAIM: <the claim, one line>
VERDICT: PASS | FAIL | NEEDS-MORE
REASON: <one or two lines, with run ids>
CLAIM: <the claim, one line>
VERDICT: PASS | FAIL | NEEDS-MORE
REASON: <one or two lines, with run ids>
CLAIM: <the claim, one line>
VERDICT: PASS | FAIL | NEEDS-MORE
REASON: <one or two lines, with run ids>
CLAIM: Waterbirds K=12에서 suffdisc는 final seeds 1·2·3에서 maskedkd를 정확도·WGA 모두 이기며, screen 결과가 final에서도 유지된다.
VERDICT: NEEDS-MORE
REASON: final_suffdisc_k12_s1_1031_002441_5d86, final_suffdisc_k12_s2_1031_002448_664e, final_suffdisc_k12_s3_1031_002453_9ebe와 대응 maskedkd 3개는 완료·100 epochs·digest·수치가 일치하며, +5.42/+40.10 및 CI는 지지된다. 그러나 Search effort는 주장한 2개가 아닌 3 configs이고 전체 suffdisc 이력이 여전히 잘려 탐색·예산 선택 검증이 불완전하다.
CLAIM: Waterbirds K=12에서 suffdisc는 final seeds 1·2·3에서 maskedkd를 정확도·WGA 모두 이기며, screen 결과가 final에서도 유지된다.
VERDICT: NEEDS-MORE
REASON: final_suffdisc_k12_s1_1031_002441_5d86, final_suffdisc_k12_s2_1031_002448_664e, final_suffdisc_k12_s3_1031_002453_9ebe와 대응 maskedkd 3개는 완료·100 epochs·digest·수치가 일치하며, +5.42/+40.10 및 CI는 지지된다. 그러나 Search effort는 주장한 2개가 아닌 3 configs이고 전체 suffdisc 이력이 여전히 잘려 탐색·예산 선택 검증이 불완전하다.
CLAIM: Waterbirds K=12에서 suffdisc는 final seeds 1·2·3에서 maskedkd를 정확도·WGA 모두 이기며, screen 결과가 final에서도 유지된다.
VERDICT: NEEDS-MORE
REASON: final_suffdisc_k12_s1_1031_002441_5d86, final_suffdisc_k12_s2_1031_002448_664e, final_suffdisc_k12_s3_1031_002453_9ebe와 대응 maskedkd 3개는 완료·100 epochs·digest·수치가 일치하며, +5.42/+40.10 및 CI는 지지된다. 그러나 Search effort는 주장한 2개가 아닌 3 configs이고 전체 suffdisc 이력이 여전히 잘려 탐색·예산 선택 검증이 불완전하다.

## 2026-10-31 02:20 audit by codex

### request
# AUDIT_REQUEST — [claude] 2026-10-31 라운드 42 (1/4부: suffdisc 탐색 이력만)

## 지난 판정과 고친 점
AUDIT.md 02:11(codex)은 NEEDS-MORE였다. maskedkd 대비 +5.42/+40.10과 CI는 확인됐다. 문제는 두 가지였다. (1) 내가 "2 configs"라고 썼는데 Search effort에는 3개(지금은 K=49가 끝나 4개)가 있었다. (2) RESULTS_TABLE이 12k자에서 잘려 suffdisc 이력 행이 보이지 않았다.
묶음은 run 약 11개까지만 담는다. 그래서 이번 부는 **suffdisc run 11개(screen 8개 전부 + final 3개)만** 인용한다. 비교군과 분석은 인용하지 않는다. maskedkd 비교(2/4부), rollout·tome(3/4부), grid·center·random·jitdisc(4/4부)는 이 부가 PASS한 뒤 따로 요청한다.

## 주장 (1줄)
`suffdisc`(selector_sha 0f060023d021…)의 Waterbirds 탐색 이력은 4 configs, screen 8 run이 전부다. final로 올린 것은 K=12 기본 설정 하나이고, 이 결정은 다른 config가 돌기 전에 내렸다. final 3-seed 평균(89.63/62.40)은 같은 설정의 screen 4-seed 평균(89.87/63.35)과 일관된다.

## suffdisc run 전부 (status done, epochs_done 100, harness_digest 033df9786a4a)
| config (Search effort) | run | 정확도 | WGA | GFLOPs |
|---|---|---|---|---|
| 1. K=12, sel_args {} | screen_suffdisc_k12_s0_1030_173118_504a | 89.32 | 58.65 | 15.02 |
| 1. | screen_suffdisc_k12_s1_1030_185953_9455 | 89.91 | 63.91 | 15.02 |
| 1. | screen_suffdisc_k12_s2_1030_233019_98dd | 90.24 | 62.41 | 15.02 |
| 1. | screen_suffdisc_k12_s3_1030_233028_39ad | 89.99 | 68.42 | 15.02 |
| 2. K=12, sel_args {"r": 1} | screen_suffdisc_k12_s0_1031_011600_6b15 | 88.57 | 55.64 | 11.86 |
| 2. | screen_suffdisc_k12_s1_1031_011604_6add | 88.99 | 56.39 | 11.86 |
| 3. K=29, sel_args {} | screen_suffdisc_k29_s0_1031_011608_3604 | 90.16 | 61.65 | 22.62 |
| 4. K=49, sel_args {} | screen_suffdisc_k49_s0_1031_012539_2b13 | 90.24 | 60.90 | 31.75 |
| final (config 1) | final_suffdisc_k12_s1_1031_002441_5d86 | 90.08 | 65.41 | 15.02 |
| final (config 1) | final_suffdisc_k12_s2_1031_002448_664e | 89.66 | 60.90 | 15.02 |
| final (config 1) | final_suffdisc_k12_s3_1031_002453_9ebe | 89.16 | 60.90 | 15.02 |

RESULTS_TABLE 89행 "suffdisc: 4 configs"가 위 config 1~4와 맞다. 표 139·140행(잘린 뒤쪽)도 config 1·2의 run id와 평균을 그대로 담고 있다.

## 시간 순서 (run id의 타임스탬프로 확인 가능)
config 1 screen(10-30 17:31~23:30) → final 제출(10-31 00:24) → config 2·3(01:16) → config 4(01:25). 따라서 final로 올릴 K와 설정을 고를 때 config 2~4의 결과는 없었다. 결과를 보고 K를 고른 것이 아니다.

## 다른 config의 결과 (숨기지 않음)
- config 2(r=1, 비용 맞춤): 평균 88.78/56.02다. config 1보다 낮아서 final로 올리지 않았다.
- config 3·4: RESULTS_TABLE Parent check 74·75행 기준 같은 seed의 center 대비 K=29는 acc +1.17·WGA −0.76, K=49는 acc +0.08·WGA +0.75다. WGA 이득이 K=12에서만 보인다. 그래서 이후 모든 부의 주장은 K=12에 한정한다.
- config 1의 Parent check(73행, seed 0만)는 center 대비 UNPROVEN이다(acc −0.17, WGA −0.75). center와의 4-seed 짝 비교와 final 3-seed 비교(WGA +8.77 [+4.06, +13.47])는 4/4부에서 다룬다.

## screen → final 일관성
config 1 screen 4-seed 평균은 89.87/63.35(RESULTS_TABLE 140행)이고 final 3-seed는 89.63/62.40(108행)이다. 차이는 −0.24/−0.95로 noise threshold(0.76/2.45, 10행) 안이다.

## 무결성
agent 커밋 중 harness/·tests/를 바꾼 것은 없다. test_eval은 쓰지 않았다.

### verdict
CLAIM: <the claim, one line>
VERDICT: PASS | FAIL | NEEDS-MORE
REASON: <one or two lines, with run ids>
CLAIM: suffdisc의 탐색은 4 configs·screen 8 runs가 전부이며, 다른 config 실행 전에 K=12 기본 설정만 final로 승격했고 screen/final 평균이 일관된다.
VERDICT: NEEDS-MORE
REASON: 제공된 11개 run의 완료 상태·100 epochs·digest·SHA·반올림 수치는 일치한다. final_suffdisc_k12_s1_1031_002441_5d86 등 final 3개의 평균 89.63/62.40과 screen 평균 89.87/63.35는 확인되며, run ID의 시각도 주장한 순서와 맞는다. 그러나 RESULTS_TABLE의 suffdisc screen 이력이 여전히 잘려 있어 “8개가 전부”인지 확인할 전체 run 목록이 필요하다.
CLAIM: suffdisc의 탐색은 4 configs·screen 8 runs가 전부이며, 다른 config 실행 전에 K=12 기본 설정만 final로 승격했고 screen/final 평균이 일관된다.
VERDICT: NEEDS-MORE
REASON: 제공된 11개 run의 완료 상태·100 epochs·digest·SHA·반올림 수치는 일치한다. final_suffdisc_k12_s1_1031_002441_5d86 등 final 3개의 평균 89.63/62.40과 screen 평균 89.87/63.35는 확인되며, run ID의 시각도 주장한 순서와 맞는다. 그러나 RESULTS_TABLE의 suffdisc screen 이력이 여전히 잘려 있어 “8개가 전부”인지 확인할 전체 run 목록이 필요하다.
CLAIM: suffdisc의 탐색은 4 configs·screen 8 runs가 전부이며, 다른 config 실행 전에 K=12 기본 설정만 final로 승격했고 screen/final 평균이 일관된다.
VERDICT: NEEDS-MORE
REASON: 제공된 11개 run의 완료 상태·100 epochs·digest·SHA·반올림 수치는 일치한다. final_suffdisc_k12_s1_1031_002441_5d86 등 final 3개의 평균 89.63/62.40과 screen 평균 89.87/63.35는 확인되며, run ID의 시각도 주장한 순서와 맞는다. 그러나 RESULTS_TABLE의 suffdisc screen 이력이 여전히 잘려 있어 “8개가 전부”인지 확인할 전체 run 목록이 필요하다.
