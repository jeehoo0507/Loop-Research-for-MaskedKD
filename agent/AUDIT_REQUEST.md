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
