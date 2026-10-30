# RESULTS

## 확정 (감사 PASS)

## 잠정 (screen)
- (seed 0 한 개, 비교 아님, 관찰) K=49에서 MaskedKD는 student WGA가 51.88로 떨어진다(그룹 2 = 육지 배경의 물새). 정확도는 89.24다 (screen_maskedkd_k49_s0_1030_135953_3e16). 같은 K의 rollout은 89.66 / 64.66 (screen_rollout_k49_s0_1030_135953_a1a9), Full seed0은 90.66 / 65.41 (screen_full_k196_s0_1030_123806_2c89).

## 실패한 방향

## 분석에서 배운 것
- **[claude] 2026-10-30 마지막 블록 CLS attention은 teacher에게 나쁜 입력을 준다. 소수 그룹에서 특히 나쁘다.** (analysis_group_fidelity_1030_143842_1ea0: full-KD student. analysis_group_fidelity_1030_144455_654a: maskedkd@49 student. 둘 다 val 데이터, masked teacher의 정확도 평균 / 최악 그룹)
  - full student K=49: L11(=MaskedKD) 86.91/47.37, L9 92.99/72.18, rollout 91.66/72.93, center 91.24/63.91, oracle(teacher 자신의 attention) 91.83/66.17, random 79.23/24.81, grid 74.81/21.80, anti(CLS attention 하위 K) 65.22/3.01.
  - mkd49 student K=49: L11 87.32/41.35, L9 92.66/69.17, rollout 92.24/72.18. K=12: L11 75.15/27.07, L9 80.07/58.65, rollout 76.40/47.37. K=29: L11 81.40/27.82, L9 88.57/59.40, rollout 87.66/61.65.
  - 해석: (1) 배경만 받은 teacher(anti)는 소수 그룹에서 거의 다 틀린다(WGA 3). 배경 토큰이 지름길을 싣고 있다는 H-W1과 맞는다. (2) 같은 student라도 9번째 블록의 CLS attention top-K가 마지막 블록보다 teacher를 훨씬 정확하게 유지한다. (3) L9 대 rollout은 K=29~49에서 비슷하고, K=12에서는 L9가 앞선다(두 student 모두). (4) 이웃 억제(nms)와 half_grid는 L11보다도 나쁘다(full K=49 WGA 24.81, 38.35). H-W2(퍼뜨리기)는 기각한다.
  - 주의: 이것은 teacher 쪽 신호 품질이다. student 학습 결과는 screen으로 따로 확인해야 한다. 그룹 2·3의 val은 133장이라 1장이 0.75%p다.
