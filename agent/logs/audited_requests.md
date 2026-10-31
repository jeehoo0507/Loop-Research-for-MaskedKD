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
