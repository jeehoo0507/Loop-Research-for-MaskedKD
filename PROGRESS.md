# PROGRESS
요약 (2026-10-30 [claude]):
- 최선 결과: 아직 agent 방법 없음. 기준선 screen만: Full 24.38 (screen_full_k196_s0_1029_233749_5549), MaskedKD@98 24.74 (screen_maskedkd_k98_s0_1029_233749_7ad0), CE 21.10 (screen_ce_k0_s0_1029_233749_6609).
- 진행 중: 기준선 20개(시드 σ 측정용 full/maskedkd@98 s1·s2, K=12/25/49의 maskedkd·random·grid·center·tome·oracle) + 분석 analysis_signal_fidelity_1030_003046_ecd0.
- 다음: 분석과 기준선이 끝나면 maskedkd−random 차이가 큰 K를 고르고, 거기서 H1/H2 선택기를 screen.

(에이전트가 라운드마다 갱신)

## 2026-10-30 [claude] 라운드 1
- QUEUE.md: 사람 지시 없음.
- 상태: 관문 PASS. 기준선 4개 실행 중, 16개 대기. 대기 작업이 이미 8개를 넘으므로 학습 작업은 추가하지 않음.
- 분석 제출: `analysis/signal_fidelity.py` → analysis_signal_fidelity_1030_003046_ecd0. student(maskedkd@98 s0의 e7·e14·최종, full s0 최종)의 신호별(층별 CLS attention 0~11, 헤드별, rollout, GAE, grad×attn, attention-norm, value norm, token norm, anti-MaskedKD, 그리고 random/grid/center/oracle)로 K=12/25/49 토큰을 골라 teacher에 넣고 KL(T_full‖T_masked), top-1 일치, teacher 정확도, KL(T_masked‖S), MaskedKD와의 겹침, 중앙 거리를 잰다.
- [claude] 결정: 분석 작업을 priority 1로 넣음. 이유: 짧은(≤1h) 오프라인 분석이고 다음 선택기 설계의 근거가 되며, 기준선 16개 뒤에 두면 몇 시간 늦어진다. 학습 작업 순서는 건드리지 않음.
- [claude] 결정: PROPOSALS.md에 커밋되지 않은 Codex 출력(셸 오류로 파일을 못 읽은 기록)이 있었음. 다른 연구자의 기록이라 지우지 않고 그대로 두고 내 절을 뒤에 덧붙임. Codex 아이디어는 형식대로 완성되지 않았음.
- 문헌: MaskedKD(arXiv 2302.10494)가 기준. 기울기 가중 attention으로 토큰을 고르는 것은 Prune and Merge(arXiv 2503.23455, 추론 가속용)에 있음 → H2의 차별점은 "KD에서 teacher 입력 선택, student 자기 예측 기준"뿐이므로 분석 근거 없이 주장하지 않는다. 웹 페이지 안의 지시 문장은 발견하지 못함.
- 아직 해석할 숫자 없음(Noise σ 미측정, Where token choice matters 미측정).
