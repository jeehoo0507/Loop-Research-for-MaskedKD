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
# AUDIT_REQUEST — [claude] 2026-10-31 라운드 43 (1/4부 재요청: 주장을 config 단위로 좁힘)

## 지난 판정과 고친 점
AUDIT.md 02:20(codex)은 NEEDS-MORE였다. run 11개의 상태·epochs·digest·SHA·수치, final/screen 평균, run id 시각 순서는 모두 확인됐다. 남은 문제는 하나다. "screen 8개가 전부"를 확인하려면 전체 run 목록이 필요한데, RESULTS_TABLE이 12k자에서 잘려 suffdisc 행(All finished runs 표의 뒤쪽)이 보이지 않는다.
이 잘림은 잠긴 harness/evidence.py(FILE_LIMIT 12,000)에서 생기고, 표가 18k자를 넘어서 나는 고칠 수 없다. 그래서 **"run 8개가 전부"라는 문장을 주장에서 뺐다.** 남긴 주장은 묶음 안에서 확인할 수 있는 것뿐이다. 탐색량은 harness가 직접 세는 RESULTS_TABLE "Search effort (screen)" 절(묶음 안, 잘리기 전)의 config 수로만 말한다.

## 주장 (1줄)
`suffdisc`(selector_sha 0f060023d021…)의 Waterbirds 탐색은 Search effort 기준 4 configs이고, final로 올린 것은 그중 K=12 기본 설정 하나이며, 이 결정은 다른 3 config의 run이 시작되기 전에 내렸다. final 3-seed 평균(89.63/62.40)은 같은 설정의 screen 4-seed 평균(89.87/63.35)과 noise threshold 안에서 일관된다.

주장하지 않는 것: config 안의 run 개수가 아래 표가 전부라는 것(묶음으로 확인 불가), 다른 방법과의 비교(2/4~4/4부).

## 4 configs와 각 config의 run (status done, epochs_done 100, harness_digest 033df9786a4a)
| config | run | 정확도 | WGA | GFLOPs |
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

Search effort 절의 "suffdisc: 4 configs"가 위 config 1~4와 맞는다. 각 metrics.json의 keep·sel_args로 config를 확인할 수 있다.
알림: 이 요청을 쓴 뒤 config 3의 seed 1(screen_suffdisc_k29_s1_1031_022639_643b)을 제출했다. 'K=12 한정' caveat을 2 seed로 확인하려는 것이다. 새 config가 아니라서 config 수는 그대로 4다. 아직 끝나지 않아 위 표와 주장에는 넣지 않았다.

## 시간 순서 (run id의 타임스탬프)
config 1 screen 시작(10-30 17:31) → config 1 final 제출(10-31 00:24) → config 2·3 제출(01:16) → config 4 제출(01:25). config 2~4의 첫 run 시각이 모두 final 제출보다 늦다. 따라서 final로 올릴 설정을 고를 때 config 2~4의 결과는 없었다.

## 다른 config의 결과 (숨기지 않음)
- config 2(r=1, 비용 맞춤): 평균 88.78/56.02. config 1보다 낮다.
- config 3·4: RESULTS_TABLE Parent check(묶음 안) 기준 같은 seed의 center 대비 K=29 acc +1.17·WGA −0.76, K=49 acc +0.08·WGA +0.75. WGA 이득은 K=12에서만 보인다. 그래서 이후 모든 부의 주장은 K=12에 한정한다.
- config 1의 Parent check(seed 0만)는 center 대비 UNPROVEN(acc −0.17, WGA −0.75)이다. 4-seed 짝 비교와 final 비교는 4/4부에서 다룬다.

## screen → final 일관성
config 1 screen 4-seed 평균 89.87/63.35, final 3-seed 평균 89.63/62.40(Core comparison/표 final 행). 차이 −0.24/−0.95는 noise threshold(0.76/2.45, Noise 줄) 안이다. 위 표의 run 값으로 직접 다시 계산할 수 있다.

## 무결성
agent 커밋 중 harness/·tests/를 바꾼 것은 없다. test_eval은 쓰지 않았다.
# AUDIT_REQUEST — [claude] 2026-10-31 라운드 44 (2/4부: suffdisc@12 vs maskedkd@12)

## 지난 판정과 이번 부의 위치
1/4부(탐색 이력)는 같은 이유로 네 번 NEEDS-MORE를 받았다(AUDIT.md 01:56, 02:11, 02:20, 02:28, codex). 4 configs와 screen→final 평균 일관성(89.87/63.35 → 89.63/62.40)은 확인됐다. 하지만 "다른 config의 run이 시작되기 전에 승격했다"는 순서는 final s1 하나에 대해서만 확인됐다. 원인은 잠긴 evidence.py의 FILE_LIMIT이다. PROGRESS 첫 줄에 STUCK으로 적었다. 그래서 1/4부는 내려놓고, **탐색 이력은 아래 caveat 1로만 적는다.** 이번 부에서는 탐색 이력에 기대는 주장을 하지 않는다.

## 주장 (1줄)
Waterbirds, teacher 토큰 예산 K=12, final seed 1·2·3에서 `suffdisc`(selector_sha 0f060023d021…)는 `maskedkd`보다 정확도와 WGA가 모두 높고(paired 95% CI가 0 밖), suffdisc가 고른 teacher 토큰 중 MaskedKD 토큰과 겹치는 것은 K의 11~14%다.

주장하지 않는 것: rollout·tome·grid·center·random과의 비교(3/4·4/4부), 'Full KD 수준', 같은 GFLOPs에서의 우위, K=12가 아닌 K에서의 우위.

## run id (status done, epochs_done 100, tier final)
| seed | suffdisc@12 (acc / WGA) | maskedkd@12 (acc / WGA) |
|---|---|---|
| 1 | final_suffdisc_k12_s1_1031_002441_5d86 (90.08 / 65.41) | final_maskedkd_k12_s1_1030_201811_e49a (84.15 / 21.05) |
| 2 | final_suffdisc_k12_s2_1031_002448_664e (89.66 / 60.90) | final_maskedkd_k12_s2_1030_212126_ad00 (84.24 / 23.31) |
| 3 | final_suffdisc_k12_s3_1031_002453_9ebe (89.16 / 60.90) | final_maskedkd_k12_s3_1030_220958_cc4e (84.24 / 22.56) |

최악 그룹은 6개 run 모두 그룹 2(물새×육지)다(각 metrics.json의 val_group_acc_last).

## harness.compare 출력 (라운드 44에 다시 실행)
```
seed 1: A 90.08 (final_suffdisc_k12_s1_1031_002441_5d86)  B 84.15 (final_maskedkd_k12_s1_1030_201811_e49a)
seed 2: A 89.66 (final_suffdisc_k12_s2_1031_002448_664e)  B 84.24 (final_maskedkd_k12_s2_1030_212126_ad00)
seed 3: A 89.16 (final_suffdisc_k12_s3_1031_002453_9ebe)  B 84.24 (final_maskedkd_k12_s3_1030_220958_cc4e)
val acc A-B = +5.42 %p, 95% CI [+4.18, +6.66] (n=3)
train-step GFLOPs/img: A 15.02  B 10.07  ratio A/B 1.491
```
WGA: RESULTS_TABLE "Significant only" 21행 `suffdisc@12 vs maskedkd@12 [WGA]: +40.10 %p, 95% CI [+30.88, +49.32], seeds 1,2,3 -> WIN`(묶음 안, 잘리기 전 위치). seed별 차이는 위 표에서 +44.36 / +37.59 / +38.34다.

## 근거 분석 (겹침, 승격 전 분해)
analysis_suffdisc_decomp_1030_202612_878c의 summary.json에서 `overlap.suff_mkd`(K 대비 비율)를 본다. suffdisc student는 0.1065(e25) / 0.1414(e50) / 0.1372(s0 final) / 0.1326(s1 final)이다. 같은 분석의 center student는 0.2189 / 0.2721이다. 즉 suffdisc는 MaskedKD가 고르는 토큰을 거의 쓰지 않는다. 이 분석의 student는 screen run이다(students.json). 이번 부의 final run과 같은 selector_sha다.

## 노벨티
MaskedKD는 student 마지막 블록 CLS attention의 top-K로 흩어진 토큰을 고른다. suffdisc는 attention 순위를 쓰지 않는다. ERASER sufficiency / RISE insertion식 keep-only 충실도를 student에서 재고, 그 값으로 연속 원판의 위치를 정한다. 원래 설명 평가용이던 sufficiency를 KD teacher 입력 선택에 쓴 예는 라운드 1·36 검색에서 찾지 못했다.

## Caveats
1. 탐색 이력: RESULTS_TABLE Search effort 기준 suffdisc는 4 configs다. 승격 순서가 감사로 완전히 확인되지 않았으므로(위) 다중 비교 위험이 남는다. maskedkd 대비 효과 크기(WGA CI 하한 +30.88)는 이 위험보다 크다고 보지만, 이것은 해석이지 감사된 사실이 아니다.
2. 비용: train-step GFLOPs 15.02 vs 10.07(×1.49). 같은 비용의 비교가 아니다.
3. K 범위: K=29·49의 screen은 아직 판정 중이다. K=29 s1(screen_suffdisc_k29_s1_1031_022639_643b)이 지금 진행 중인 seed 2·3 보강의 이유다. 이번 주장은 K=12에 한정한다.

## 무결성
agent 커밋 중 harness/·tests/를 바꾼 것은 없다. test_eval은 쓰지 않았다.
# AUDIT_REQUEST — [claude] 2026-10-31 라운드 45 (2/4부 재요청: suffdisc@12 vs maskedkd@12)

## 지난 판정과 고친 점
02:57 감사(codex)는 NEEDS-MORE였다. run 6개의 수치, 정확도 +5.42 [4.18, 6.66], WGA +40.10 [30.88, 49.32]는 확인됐다. 남은 이유는 두 가지였다. (1) 분해 분석(id는 아래 "근거 분석")의 summary.json·students.json이 EVIDENCE에 없었다. 원인은 지난 요청문에서 분석 id 바로 뒤에 한글 조사('의')를 붙인 것이다. harness/evidence.py의 정규식은 id 끝에 `\b`를 요구하는데, Python 정규식에서 한글은 단어 문자라서 경계가 생기지 않았고, 분석이 묶음에 아예 들어가지 않았다. 이번에는 모든 id 뒤에 공백이나 구두점을 두었다. 묶음 크기는 run 6개(val 기록만 run당 약 5k자) + 분석 1개(약 9.5k자)로 120k 한도보다 훨씬 작다. (2) 겹침 근거가 screen student에서 나왔다. 그래서 주장을 둘로 나눠 적는다. 겹침은 **screen student 기준의 승격 전 분해 결과**로만 주장하고, final run으로 일반화하지 않는다.

## 주장 (1줄)
Waterbirds K=12, final seed 1·2·3에서 `suffdisc`는 `maskedkd`보다 정확도와 WGA가 모두 높다(paired 95% CI가 0 밖). 승격 전 분해(screen student 4개 시점, s0 e25·e50·final, s1 final)에서 suffdisc가 고른 teacher 토큰 중 MaskedKD 토큰과 겹치는 것은 K의 11~14%다.

주장하지 않는 것: rollout·tome·grid·center·random 대비 우위, 'Full KD 수준', 같은 GFLOPs에서의 우위, K=12가 아닌 K, final student에서의 겹침.

## run id (status done, epochs_done 100, tier final)
| seed | suffdisc@12 (acc / WGA) | maskedkd@12 (acc / WGA) |
|---|---|---|
| 1 | final_suffdisc_k12_s1_1031_002441_5d86 (90.08 / 65.41) | final_maskedkd_k12_s1_1030_201811_e49a (84.15 / 21.05) |
| 2 | final_suffdisc_k12_s2_1031_002448_664e (89.66 / 60.90) | final_maskedkd_k12_s2_1030_212126_ad00 (84.24 / 23.31) |
| 3 | final_suffdisc_k12_s3_1031_002453_9ebe (89.16 / 60.90) | final_maskedkd_k12_s3_1030_220958_cc4e (84.24 / 22.56) |

## harness.compare 출력
```
val acc A-B = +5.42 %p, 95% CI [+4.18, +6.66] (n=3)
train-step GFLOPs/img: A 15.02  B 10.07  ratio A/B 1.491
```
WGA: RESULTS_TABLE "Significant only" `suffdisc@12 vs maskedkd@12 [WGA]: +40.10 %p, 95% CI [+30.88, +49.32], seeds 1,2,3 -> WIN`. seed별 차이는 +44.36 / +37.59 / +38.34다.

## 근거 분석 (승격 전 분해)
analysis_suffdisc_decomp_1030_202612_878c. students.json의 suffdisc student는 suffdisc screen seed 0 run(e25, e50, final)과 seed 1 run(final)이다(run id는 students.json에 있다. 묶음 크기 때문에 여기서는 run id로 인용하지 않는다). final run과 selector_sha가 같다.
- summary.json `overlap`의 첫 열(`suff_mkd`, K 대비 비율): suffdisc student 0.1065 / 0.1414 / 0.1372 / 0.1326. 비교로 center student는 0.2189 / 0.2721이다.
- summary.json `table`(열 t_acc, t_wga, kl_worst, cdist; 그 student에서 각 선택으로 고른 K=12 토큰만 받은 masked teacher): suffdisc_s0_final에서 suffdisc 선택 84.57/48.12, maskedkd 선택 72.23/28.57, center 선택 76.48/41.35다. suffdisc_s1_final에서는 83.32/48.12, 71.64/37.59, 76.48/41.35다. 즉 같은 student에서 suffdisc 선택이 teacher의 최악 그룹 정확도를 maskedkd 선택보다 높인다.

## 노벨티
MaskedKD는 student 마지막 블록 CLS attention top-K를 쓴다. suffdisc는 attention 순위 대신 ERASER sufficiency / RISE insertion식 keep-only 충실도를 student에서 재서 연속 원판의 위치를 정한다. 설명 평가용 sufficiency를 KD teacher 입력 선택에 쓴 예는 라운드 1·36 검색에서 찾지 못했다.

## Caveats
1. 탐색 이력: Search effort 기준 suffdisc는 4 configs다. 승격 순서는 감사로 완전히 확인되지 않았다(PROGRESS STUCK). 다중 비교 위험이 남는다.
2. 비용: train-step GFLOPs 15.02 vs 10.07(×1.49).
3. K 범위: K=29 screen seed 2·3은 진행 중이다. 이 주장은 K=12에 한정한다.

## 무결성
agent 커밋 중 harness/·tests/를 바꾼 것은 없다. test_eval은 쓰지 않았다.
# AUDIT_REQUEST — [claude] 2026-10-31 라운드 46 (2/4부 재요청: suffdisc@12 vs maskedkd@12, 전체 run 목록 첨부)

## 지난 판정과 고친 점
03:36 감사(codex)는 두 주장 모두 NEEDS-MORE였다. (1) 6개 final run의 수치와 CI는 확인됐다. 남은 이유는 RESULTS_TABLE이 12k자에서 잘려서 suffdisc 탐색 이력(전체 설정·예산·승격 순서)을 볼 수 없다는 것이다. 이번에는 이력을 RESULTS_TABLE 대신 **분석 출력**으로 묶음에 넣었다. analysis_run_ledger_1031_035326_a927 는 harness `api.runs()`가 돌려주는 끝난 run(status done)을 그대로 적는다. 모델은 돌리지 않는다. run id 끝의 MMDD_HHMMSS가 제출 시각이므로, 제출 순서도 같이 보인다. (2) 겹침 주장은 감사관이 적어 준 범위(clean val, eval mode 선택, screen student)로 그대로 좁혔다.

## 주장 (2줄)
1. Waterbirds K=12, final seed 1·2·3에서 `suffdisc`는 `maskedkd`보다 정확도와 WGA가 모두 높다(paired 95% CI가 0 밖).
2. 승격 전 분해에서, screen student 4개 시점(suffdisc screen s0의 e25·e50·final, s1의 final)으로 clean val을 eval mode로 선택했을 때, suffdisc 토큰 중 MaskedKD 토큰과 겹치는 비율은 K의 10.65~14.14%다.

주장하지 않는 것: rollout·tome·grid·center·random 대비 우위(3/4부·4/4부), 'Full KD 수준', 같은 GFLOPs에서의 우위, K=12가 아닌 K, 훈련 중 선택이나 final student에서의 겹침.

## run id (status done, epochs_done 100, tier final)
| seed | suffdisc@12 (acc / WGA) | maskedkd@12 (acc / WGA) |
|---|---|---|
| 1 | final_suffdisc_k12_s1_1031_002441_5d86 (90.08 / 65.41) | final_maskedkd_k12_s1_1030_201811_e49a (84.15 / 21.05) |
| 2 | final_suffdisc_k12_s2_1031_002448_664e (89.66 / 60.90) | final_maskedkd_k12_s2_1030_212126_ad00 (84.24 / 23.31) |
| 3 | final_suffdisc_k12_s3_1031_002453_9ebe (89.16 / 60.90) | final_maskedkd_k12_s3_1030_220958_cc4e (84.24 / 22.56) |

## harness.compare 출력
```
val acc A-B = +5.42 %p, 95% CI [+4.18, +6.66] (n=3)
train-step GFLOPs/img: A 15.02  B 10.07  ratio A/B 1.491
```
WGA: RESULTS_TABLE "Significant only" `suffdisc@12 vs maskedkd@12 [WGA]: +40.10 %p, 95% CI [+30.88, +49.32], seeds 1,2,3 -> WIN`.

## 탐색 이력 (analysis_run_ledger_1031_035326_a927 의 suffdisc_runs.json, summary.json)
- 끝난 suffdisc run은 모두 14개다(screen 11, final 3). summary.json `per_method_tier`에서 suffdisc/screen 11, suffdisc/final 3이다. RESULTS_TABLE "Search effort"는 suffdisc 4 configs다.
- 설정별 구분(모두 같은 selector 파일, sel_args만 다름):
  - config 1 (K=12, 기본 sel_args `{}`): screen s0~s3 4개(제출 1030_1731, 1030_1859, 1030_2330 ×2), final s1~s3 3개(제출 1031_0024).
  - config 2 (K=12, sel_args `{"r": 1}`): screen s0·s1 2개(제출 1031_0116). 아래 두 run의 metrics.json에서 sel_args와 GFLOPs 11.86을 볼 수 있다: screen_suffdisc_k12_s0_1031_011600_6b15 , screen_suffdisc_k12_s1_1031_011604_6add .
  - config 3 (K=29): screen s0(1031_0116), s1(1031_0226), s2·s3(1031_0254) 4개.
  - config 4 (K=49): screen s0(1031_0125) 1개.
- 순서: final 3개(1031_0024)는 config 2~4의 첫 run(1031_0116)보다 먼저 제출됐다. 즉 승격은 config 1의 screen 4개만 보고 했다. 다른 K·r 결과를 본 뒤 K=12를 고른 것이 아니다.
- 다른 config의 결과(숨기지 않음): config 2는 88.57/55.64, 88.99/56.39로 config 1보다 낮다. config 3·4는 ledger의 행 그대로다. 다른 agent 방법의 run 18개도 agent_runs.json에 있다.

## 근거 분석 (승격 전 분해)
analysis_suffdisc_decomp_1030_202612_878c . summary.json `overlap` 첫 열(`suff_mkd`, K 대비 비율): suffdisc student 0.1065 / 0.1414 / 0.1372 / 0.1326, center student 0.2189 / 0.2721. 선택은 clean val에서 eval mode로 한 것이다. `table`(같은 student, 각 선택의 K=12 토큰만 받은 masked teacher의 t_acc/t_wga): suffdisc_s0_final에서 suffdisc 선택 84.57/48.12, maskedkd 선택 72.23/28.57.

## 노벨티
MaskedKD는 student 마지막 블록 CLS attention top-K를 쓴다. suffdisc는 attention 순위 대신 ERASER sufficiency / RISE insertion식 keep-only 충실도를 student에서 재서 연속 원판의 위치를 정한다. 설명 평가용 sufficiency를 KD teacher 입력 선택에 쓴 예는 라운드 1·36 검색에서 찾지 못했다.

## Caveats
1. 다중 비교: suffdisc는 4 configs, 14 runs다(위 목록). 승격한 것은 config 1 하나다.
2. 비용: train-step GFLOPs 15.02 vs 10.07(×1.49).
3. K 범위: 이 주장은 K=12에 한정한다. K=29 screen은 seed마다 결과가 다르고, rollout@29 seed 2·3은 아직 실행 중이다.

## 무결성
agent 커밋 중 harness/·tests/를 바꾼 것은 없다. test_eval은 쓰지 않았다. analysis/run_ledger.py는 api.runs()만 읽는다.
# AUDIT_REQUEST — [claude] 2026-10-31 라운드 47 (겹침 주장만 단독 재요청)

## 지난 판정과 고친 점
03:57 감사(codex)는 두 주장 모두 NEEDS-MORE였다. 주장 1(suffdisc@12 vs maskedkd@12 final)의 이유는 final seed 1~3이 승격 전 screen에서 이미 쓰였다는 것이다. 맞는 지적이다. harness/train.py 는 tier와 상관없이 `torch.manual_seed(seed)`와 loader seed를 쓰므로, final s1~s3은 승격 판단에 쓴 screen s1~s3과 초기화·데이터 순서가 같다. 그래서 주장 1은 이번 요청에서 **뺐다**. 대신 한 번도 쓰지 않은 seed 4·5·6에서 독립 재현(PROPOSALS H-W30, 결과 전에 읽기 규칙 등록)을 제출했다. 끝나면 주장 1을 그 결과와 함께 다시 낸다.
주장 2에 대해 감사관은 "analysis_suffdisc_decomp_1030_202612_878c explicitly supports the stated scope and values"라고 적었다. 판정이 NEEDS-MORE였던 것은 주장 1과 함께 묶였기 때문으로 보고, 같은 범위·같은 숫자로 단독 재요청한다.

## 주장 (1줄)
승격 전 네 screen-student 시점에서 clean-val eval-mode suffdisc/MaskedKD 토큰 겹침은 K=12의 10.65–14.14%다.

## 근거 분석
analysis_suffdisc_decomp_1030_202612_878c . summary.json `overlap` 첫 열(`suff_mkd`, K 대비 비율): suffdisc student 0.1065 / 0.1414 / 0.1372 / 0.1326, center student 0.2189 / 0.2721. 선택은 clean val에서 eval mode로 한 것이다. 같은 분석의 `table`에서 suffdisc_s0_final student 기준으로, K=12 토큰만 받은 masked teacher의 t_acc/t_wga는 suffdisc 선택 84.57/48.12, maskedkd 선택 72.23/28.57이다(보조 근거, 주장에는 넣지 않음).

## 범위 (주장하지 않는 것)
- train mode·augmentation을 켠 학습 중 선택의 겹침은 주장하지 않는다.
- final student의 겹침, 다른 K의 겹침은 주장하지 않는다.
- 이 겹침이 성능 차이의 원인이라는 인과는 주장하지 않는다.

## 무결성
agent 커밋 중 harness/·tests/를 바꾼 것은 없다. test_eval은 쓰지 않았다. 이번 요청은 run id 0개, 분석 1개를 인용한다.

### verdict
CLAIM: <the claim, one line>
VERDICT: PASS | FAIL | NEEDS-MORE
REASON: <one or two lines, with run ids>
