# PROGRESS
요약 (2026-10-30 [claude] 라운드 3):
- 최선 결과: 아직 agent 방법 없음. screen: Full 24.45±0.10 (표 기준 s0·s1; s2 = 23.78, screen_full_k196_s2_1030_002458_e159), MaskedKD@98 24.46±0.24 (s0/s1/s2), CE 21.10 (screen_ce_k0_s0_1029_233749_6609).
- Noise: σ 0.21%p (pooled, dof 3) → 승격 기준 0.50%p (RESULTS_TABLE).
- 진행 중: 실행 maskedkd@49, random@49, 분석 analysis_signal_fidelity_1030_014040_eb3d / 대기 기준선 14개 + in100 final 3개.
- 다음: 분석(fid_kl·gap_kl)과 K=12/25/49 기준선으로 H2(gradattn)·H4(엔트로피 기여도) 중 하나를 screen.

(에이전트가 라운드마다 갱신)

## 2026-10-30 [claude] 라운드 1
- QUEUE.md: 사람 지시 없음.
- 상태: 관문 PASS. 기준선 4개 실행 중, 16개 대기. 대기 작업이 이미 8개를 넘으므로 학습 작업은 추가하지 않음.
- 분석 제출: `analysis/signal_fidelity.py` → analysis_signal_fidelity_1030_003046_ecd0. student(maskedkd@98 s0의 e7·e14·최종, full s0 최종)의 신호별(층별 CLS attention 0~11, 헤드별, rollout, GAE, grad×attn, attention-norm, value norm, token norm, anti-MaskedKD, 그리고 random/grid/center/oracle)로 K=12/25/49 토큰을 골라 teacher에 넣고 KL(T_full‖T_masked), top-1 일치, teacher 정확도, KL(T_masked‖S), MaskedKD와의 겹침, 중앙 거리를 잰다.
- [claude] 결정: 분석 작업을 priority 1로 넣음. 이유: 짧은(≤1h) 오프라인 분석이고 다음 선택기 설계의 근거가 되며, 기준선 16개 뒤에 두면 몇 시간 늦어진다. 학습 작업 순서는 건드리지 않음.
- [claude] 결정: PROPOSALS.md에 커밋되지 않은 Codex 출력(셸 오류로 파일을 못 읽은 기록)이 있었음. 다른 연구자의 기록이라 지우지 않고 그대로 두고 내 절을 뒤에 덧붙임. Codex 아이디어는 형식대로 완성되지 않았음.
- 문헌: MaskedKD(arXiv 2302.10494)가 기준. 기울기 가중 attention으로 토큰을 고르는 것은 Prune and Merge(arXiv 2503.23455, 추론 가속용)에 있음 → H2의 차별점은 "KD에서 teacher 입력 선택, student 자기 예측 기준"뿐이므로 분석 근거 없이 주장하지 않는다. 웹 페이지 안의 지시 문장은 발견하지 못함.
- 아직 해석할 숫자 없음(Noise σ 미측정, Where token choice matters 미측정).

## 2026-10-30 [claude] 라운드 2
- QUEUE.md: 사람 지시 없음.
- 새로 끝난 run: screen_maskedkd_k98_s1_1030_002458_b791 val 24.30 (s0 24.74). 시드 2개로 σ 0.31%p, 승격 기준 0.62%p. dof 1이라 s2(full·maskedkd) 결과가 나오면 다시 본다. 사람이 넣은 in100 smoke 4개가 끝남(smoke tier라 해석하지 않음).
- 분석 analysis_signal_fidelity_1030_003046_ecd0: 아직 대기(priority 1이지만 실행 중 작업 4개 뒤에 있음). analysis_out/가 없어 해석할 새 분석 결과 없음.
- 작업: H2 선택기 `selectors/gradattn.py` 작성(parent=maskedkd). trysel K=25: selector 0.146 GFLOPs/img, total 9.005 vs maskedkd 8.859 (+1.6%). 오류 없음.
- [claude] 결정: gradattn을 screen에 넣지 않음. 이유: 대기 작업이 20개로 상한 8개를 이미 넘었고, 분석 결과(gradattn_last의 fid_kl)와 "Where token choice matters"가 아직 없어 어느 K를 공략할지 근거가 없다. 가장 보수적인 선택.

## 2026-10-30 [claude] 라운드 3
- QUEUE.md: 사람 지시 없음.
- 새로 끝난 run: screen_full_k196_s1_1030_002458_ec36 val 24.52, screen_full_k196_s2_1030_002458_e159 val 23.78, screen_maskedkd_k98_s2_1030_002458_a96d val 24.34. Full의 시드 범위(23.78~24.52)와 MaskedKD@98(24.30~24.74)이 겹침 → K=98에서는 둘이 구별되지 않음. σ 0.21%p(dof 3), 승격 기준 0.50%p. in100 smoke는 해석하지 않음.
- 실수 발견 1: 라운드 1에서 분석을 priority 1로 넣었는데 scheduler는 priority가 **높은** 것부터 돌린다(harness/scheduler.py에서 `-priority`로 정렬). 기준선(5)보다 뒤라 맨 마지막 차례였다.
- 실수 발견 2: analysis/signal_fidelity.py는 student 파라미터를 얼린 뒤 attention에 autograd.grad를 걸어서 실패하는 코드였다. priority 6으로 다시 넣은 v1(analysis_signal_fidelity_1030_014003_ecd0)이 바로 시작돼 `RuntimeError: element 0 of tensors does not require grad`로 2.8초 만에 실패했다(analysis_out/에 기록).
- 수정: 입력 x에 requires_grad를 걸어 그래프의 leaf로 삼음. 같은 패스에 H4 신호를 추가: student 예측 엔트로피의 grad×attention(마지막 층)과 grad×input(0층 토큰). v2 = analysis_signal_fidelity_1030_014040_eb3d, priority 6, 01:41 시작.
- [claude] 결정: 분석을 priority 6(기준선 5보다 위)으로 넣음. 이유: GPU 1시간 이내의 오프라인 분석이고 다음 선택기 설계가 모두 이 결과에 달려 있다. 학습 작업은 한 칸만 밀린다. 대기 중이던 v0(…003046_ecd0) 파일은 내 작업이라 중복 실행을 막으려고 지움.
- [claude] 결정: 이번 라운드도 선택기를 screen에 넣지 않음. 대기 17개로 상한 8개를 넘었고, "Where token choice matters"가 아직 측정되지 않았다.
- 관찰: sched.log에 `LOCK harness digest ac7a07d9a881 vs locked 4e2095e93694; not starting jobs`가 한 번 찍힌 직후 작업이 다시 시작됨. 사람 커밋 c5f1404(stage2) 시점으로 보인다. 나는 harness/를 건드리지 않았다.
- 문헌(H4 노벨티): 검색 2회에서 student 엔트로피 기여도로 ViT KD의 teacher 입력 패치를 고르는 선행 연구는 찾지 못함. 가까운 것은 AD-TP(NeurIPS 2025, IG로 추론 가지치기), Fast SAM2(arXiv 2512.21333, MC dropout 불확실성), SelecTKD(LLM). 웹 페이지 안에 지시 문장은 없었다.
