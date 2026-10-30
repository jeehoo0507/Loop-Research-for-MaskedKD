# RESULTS

## 확정 (감사 PASS)

## 잠정 (screen)
- (seed 0 한 개, 비교 아님, 관찰) K=49에서 MaskedKD는 student WGA가 51.88로 떨어진다(그룹 2 = 육지 배경의 물새). 정확도는 89.24다 (screen_maskedkd_k49_s0_1030_135953_3e16). 같은 K의 rollout은 89.66 / 64.66 (screen_rollout_k49_s0_1030_135953_a1a9), Full seed0은 90.66 / 65.41 (screen_full_k196_s0_1030_123806_2c89).
- (seed 0, 관찰) K=49의 center는 90.16 / 60.15다 (screen_center_k49_s0_1030_135953_4476). 신호가 없는 공간 규칙인데도 MaskedKD(89.24/51.88)보다 높다. K=29의 MaskedKD는 87.65 / 43.61이다 (screen_maskedkd_k29_s0_1030_135953_764a). tome@49는 89.66 / 51.13이다 (screen_tome_k49_s0_1030_135953_1e32).

- (seed 0, 관찰) K=29: rollout 89.82 / 65.41 (screen_rollout_k29_s0_1030_135953_7320), center 88.99 / 62.41 (screen_center_k29_s0_1030_135953_2878), grid 81.65 / 36.84 (screen_grid_k29_s0_1030_135953_a69e), random 82.49 / 27.82 (screen_random_k29_s0_1030_135953_8780). MaskedKD@29(87.66/43.61)는 center와 rollout 둘 다에게 진다.

## 실패한 방향
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
