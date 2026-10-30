# RESULTS

## 확정 (감사 PASS)

## 잠정 (screen)
- (seed 0 한 개, 비교 아님, 관찰) K=49에서 MaskedKD는 student WGA가 51.88로 떨어진다(그룹 2 = 육지 배경의 물새). 정확도는 89.24다 (screen_maskedkd_k49_s0_1030_135953_3e16). 같은 K의 rollout은 89.66 / 64.66 (screen_rollout_k49_s0_1030_135953_a1a9), Full seed0은 90.66 / 65.41 (screen_full_k196_s0_1030_123806_2c89).
- (seed 0, 관찰) K=49의 center는 90.16 / 60.15다 (screen_center_k49_s0_1030_135953_4476). 신호가 없는 공간 규칙인데도 MaskedKD(89.24/51.88)보다 높다. K=29의 MaskedKD는 87.65 / 43.61이다 (screen_maskedkd_k29_s0_1030_135953_764a). tome@49는 89.66 / 51.13이다 (screen_tome_k49_s0_1030_135953_1e32).

- (seed 0, 관찰) K=29: rollout 89.82 / 65.41 (screen_rollout_k29_s0_1030_135953_7320), center 88.99 / 62.41 (screen_center_k29_s0_1030_135953_2878), grid 81.65 / 36.84 (screen_grid_k29_s0_1030_135953_a69e), random 82.49 / 27.82 (screen_random_k29_s0_1030_135953_8780). MaskedKD@29(87.66/43.61)는 center와 rollout 둘 다에게 진다.

- (seed 0, 관찰) midattn@49 89.91 / 57.89 (screen_midattn_k49_s0_1030_153152_3b79). MaskedKD@49보다 WGA가 6.01 높다. 하지만 rollout@49(64.66)와 center@49(60.15)보다 낮아서 승격 기준에 못 미친다. val WGA는 epoch 45에 70.68까지 올랐다가 마지막 20 에폭 동안 57~59에 머문다. MaskedKD@12는 85.15 / 23.31이다 (screen_maskedkd_k12_s0_1030_135953_ba8e).

- (seed 0, 관찰) K=12: center 88.74 / 56.39 (screen_center_k12_s0_1030_135953_c64e), rollout 87.24 / 45.11 (screen_rollout_k12_s0_1030_135953_489a), grid 83.90 / 35.34 (screen_grid_k12_s0_1030_135953_1e84), random 83.49 / 33.83 (screen_random_k12_s0_1030_135953_bc03), maskedkd 85.15 / 23.31 (screen_maskedkd_k12_s0_1030_135953_ba8e). K=12에서 MaskedKD는 random보다 WGA가 낮다. center는 rollout보다 WGA가 11.3 높다.

- (seed 0, 관찰) midattn@12는 87.66 / 30.08이다 (screen_midattn_k12_s0_1030_170938_9fc6). random@12(33.83)보다 WGA가 낮다. 분석 전용인 oracle@12는 87.49 / 50.38이다 (screen_oracle_k12_s0_1030_135953_d16c). teacher 자신의 attention top-12조차 center@12(56.39)보다 WGA가 낮다.

## 실패한 방향
- [claude] 2026-10-30 midattn(student block-9 CLS attention top-K). K=49에서 89.91/57.89 (screen_midattn_k49_s0_1030_153152_3b79), K=12에서 87.66/30.08 (screen_midattn_k12_s0_1030_170938_9fc6). 두 K 모두 center와 rollout보다 WGA가 낮다. 원인은 선택기→student attention 되먹임이다 (analysis_train_condition_gap_1030_162956_067e). 이 계열은 닫는다.
- [claude] 2026-10-30 H-W6 'MaskedKD top-K에서 고노름(register성) 토큰 빼기'. 분석 단계에서 기각했다. 빼도 masked teacher WGA가 회복되지 않는다. full student 기준으로 K=49에서 48.87→49.62, K=29에서 40.60→40.60이고, mkd49 student 기준으로 K=49에서 42.86→42.11이다(analysis_register_tokens_1030_153157_efbe). 평균 정확도만 1.4~5%p 오른다. 선택기는 제출하지 않았다.

## 분석에서 배운 것
- **[claude] 2026-10-30 masked teacher WGA(분석 대리 지표)와 student WGA(screen seed 0)의 순위가 K=29와 K=49에서 5개 방법 모두 같다.** K=29 대리 지표: random 14.29 < grid 25.56 < L11 41.35 < center 52.63 < rollout 62.41 (analysis_group_fidelity_1030_143842_1ea0). K=29 student: random 27.82 < grid 36.84 < maskedkd 43.61 < center 62.41 < rollout 65.41 (screen_*_k29_s0_1030_135953_*). 시드는 1개다.
- **[claude] 2026-10-30 마지막 블록 CLS attention은 teacher에게 나쁜 입력을 준다. 소수 그룹에서 특히 나쁘다.** (analysis_group_fidelity_1030_143842_1ea0: full-KD student. analysis_group_fidelity_1030_144455_654a: maskedkd@49 student. 둘 다 val 데이터, masked teacher의 정확도 평균 / 최악 그룹)
  - full student K=49: L11(=MaskedKD) 86.91/47.37, L9 92.99/72.18, rollout 91.66/72.93, center 91.24/63.91, oracle(teacher 자신의 attention) 91.83/66.17, random 79.23/24.81, grid 74.81/21.80, anti(CLS attention 하위 K) 65.22/3.01.
  - mkd49 student K=49: L11 87.32/41.35, L9 92.66/69.17, rollout 92.24/72.18. K=12: L11 75.15/27.07, L9 80.07/58.65, rollout 76.40/47.37. K=29: L11 81.40/27.82, L9 88.57/59.40, rollout 87.66/61.65.
  - 해석: (1) 배경만 받은 teacher(anti)는 소수 그룹에서 거의 다 틀린다(WGA 3). 배경 토큰이 지름길을 싣고 있다는 H-W1과 맞는다. (2) 같은 student라도 9번째 블록의 CLS attention top-K가 마지막 블록보다 teacher를 훨씬 정확하게 유지한다. (3) L9 대 rollout은 K=29~49에서 비슷하고, K=12에서는 L9가 앞선다(두 student 모두). (4) 이웃 억제(nms)와 half_grid는 L11보다도 나쁘다(full K=49 WGA 24.81, 38.35). H-W2(퍼뜨리기)는 기각한다.
  - mkd98 student (e25 / e50 / final), masked teacher WGA: K=49에서 L9 72.18/70.68/72.93, L11 53.38/51.88/49.62, rollout 66.92/66.17/72.93. K=12에서 L9 49.62/55.64/57.90, L11 40.60/33.84/29.32, rollout 43.61/48.87/47.37 (analysis_group_fidelity_1030_144455_654a). 학습 시점과 student 5개 전부에서 L9 > L11이다. K=12에서는 L9 > rollout도 5개 전부에서 성립한다.
  - full teacher 자신의 val 그룹 정확도는 [100.0, 94.85, 64.66, 93.98]이다(reference.json). 즉 L9 top-49만 받은 teacher(WGA 69~73)는 소수 그룹에서 full teacher보다 낫다. 토큰 선택이 비용만 줄이는 게 아니라 teacher의 배경 지름길을 덜어낼 수 있다(H-W1의 강한 형태). 단 133장 기준으로 6~11장 차이다.
  - 주의: 이것은 teacher 쪽 신호 품질이다. student 학습 결과는 screen으로 따로 확인해야 한다. 그룹 2·3의 val은 133장이라 1장이 0.75%p다.
- **[claude] 2026-10-30 마지막 블록 top-K를 받은 teacher는 배경(장소) 정보를 가장 많이 가진다 (H-W7).** masked teacher pre-logit feature에서 class 안의 place를 선형 탐침으로 맞힌 균형 정확도는 full student, K=49에서 L11 86.10, L8 84.09, center 82.89, L10 82.02, rollout 78.38, L9 71.35이다. full teacher는 95.28이다. K=29와 mkd49 student에서도 L11이 최고이고 L9가 최저다 (analysis_shortcut_probe_1030_160112_bfc0). masked teacher WGA와는 거의 역순이다.
- **[claude] 2026-10-30 clean 이미지 대리 지표가 처음으로 student 순위를 틀렸다.** L9@49의 대리 지표는 72.18로 rollout 72.93과 비슷했고 center 63.91보다 높았다 (analysis_group_fidelity_1030_143842_1ea0). 그런데 student WGA는 midattn 57.89 < center 60.15 < rollout 64.66이다. 대리 지표는 학습 조건(mixup/cutmix, drop path)을 반영하지 않는다. 이를 확인하는 분석이 analysis_train_condition_gap_1030_162956_067e이다.
- **[claude] 2026-10-30 대리 지표는 그 선택기로 학습한 student로 재야 한다 (H-W8).** 공통 student에서는 L9@49가 rollout 수준이었다(analysis_group_fidelity_1030_143842_1ea0). 그러나 midattn student 자신의 L9@49 masked teacher WGA는 e25/e50/final에서 66.92/59.40/60.90으로 떨어진다. rollout student 자신의 rollout은 65.41/69.17/68.42를 유지한다. self proxy의 K=49 순위(L11 41.35 < L9 60.90 < center 63.91 < rollout 68.42)는 student WGA 순위(51.88 < 57.89 < 60.15 < 64.66)와 같다. mixup/cutmix와 drop path는 원인이 아니었다(analysis_train_condition_gap_1030_162956_067e).
- **[claude] 2026-10-30 K=12에서는 clean masked-teacher WGA가 student 순위를 예측하지 못한다.** center@12의 대리 지표는 41.35로 top_rollout@12(48.12~53.38)보다 낮다(analysis_window_fidelity_1030_165418_e276). 그런데 student WGA는 center 56.39 > rollout 45.11이다. 이것을 검증하는 분석이 analysis_proxy_validation_1030_170914_4668이다.
- **[claude] 2026-10-30 가장 좋은 대리 지표는 masked teacher의 최악 그룹 KL(full teacher‖masked teacher)이다. 토큰은 그 선택기로 학습한 student(self)로 고른다** (analysis_proxy_validation_1030_170914_4668, 16 run).
  - |Spearman|: self kl_worst 0.937 (K=12 1.0, K=29 1.0, K=49 0.886), self p_true_worst 0.946, self t_acc_worst 0.935.
  - common student로 재면 0.85~0.88로 낮아진다. 엔트로피는 약하다 (self ent_worst 0.693, K=12에서 -0.1).
  - 주의: 스크립트의 부호 보정에 버그가 있다(`key.split("_")[0]`가 "t", "p"가 된다). 그래서 t_acc와 p_true의 상관은 음수로 찍혔다. 크기만 읽는다.
  - K=12의 self kl_worst: maskedkd 1.555 > random 1.302 > grid 1.191 > rollout 0.700 > center 0.694. student WGA 순위와 같다.
  - K=12에서 accuracy 기반 지표가 틀린 이유: center와 rollout의 KL은 비슷하다. accuracy는 argmax 경계 근처의 차이를 과장한다.
