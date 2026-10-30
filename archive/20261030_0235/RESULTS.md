# RESULTS

## 확정 (감사 PASS)

## 잠정 (screen)

## 실패한 방향

## 분석에서 배운 것
- [claude] analysis_signal_fidelity_1030_014040_eb3d (val 1000장, student = screen_maskedkd_k98_s0_1029_233749_7ad0의 e7·e14·최종과 screen_full_k196_s0_1029_233749_5549 최종). 지표는 teacher 충실도 KL(T_full‖T_masked)로, 학습 결과가 아닌 대리지표다.
  - 모든 student·모든 K(12/25/49)에서 `center`가 가장 좋다 (K=49: fid_kl 0.905, masked teacher 정확도 0.600; mkd98_final 기준).
  - MaskedKD 신호(마지막 층 CLS attention)는 K=25·49에서 `random`보다 나쁘다 (K=49: 1.634 / 0.432 vs random 1.281 / 0.511). teacher 자신의 attention(oracle, 분석 전용)도 random보다 나쁘다(1.323). → 소수의 "주목받는" 토큰은 서로 몰려 있어 teacher가 문맥을 잃는다는 해석과 맞는다. 단 MaskedKD@98은 val에서 Full과 구별되지 않으므로(RESULTS_TABLE) 충실도가 teachability와 같다는 증거는 아직 없다.
  - 비공간 신호 중 최선은 student 10번 블록 출력의 토큰 노름 `toknorm_L-2` (K=49: 1.156 / 0.539, 중앙 거리 5.03로 random 5.33과 비슷, MaskedKD와 겹침 0.26 ≈ 우연)과 0층 CLS attention(1.204).
  - 기울기 기반(gradattn_last 1.679, GAE 1.437)과 student 엔트로피 기여도(entgrad_attn_last 1.516, entgrad_tok_L0 1.698)는 random보다 나쁘다 → H2·H4는 이 대리지표에서 근거 없음.
  - rollout은 K=49에서 최하위권(2.430).
