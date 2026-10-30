# PROPOSALS


## 2026-10-30 ideas from Codex (independent)
IDEA 1 (from interpretability): grounded in a published interpretability method.

IDEA 2 (from an inspiring paper): first retrieve papers that could inspire a new answer to the research question,
the way researchers find the papers that actually advanced their projects (ScholarCatalyst, arXiv 2610.02202).
Search widely: at least 5 different queries, phrased by mechanism rather than by this field's keywords, and include
papers outside token pruning and distillation.
Pick the one paper whose idea transfers best, say why it inspires this, and build the method from it. The method must
still be an interpretability technique (it explains or reads the student's internal computation).

Format, exactly:
IDEA 1: <method name> — <how scores are computed, 2 lines> — basis: <interpretability paper> — closest prior token-pruning work and difference: <1 line> — extra FLOPs: <rough>
IDEA 2: <method name> — <how scores are computed, 2 lines> — inspired by: <paper, venue, year, arXiv id or URL> — why it transfers: <1 line> — queries tried: <the queries, separated by ;> — closest prior token-pruning work and difference: <1 line> — extra FLOPs: <rough>
codex
I’ll read only the three named files, then search by mechanism for two distinct methods that use the student’s internal computation and account for selector FLOPs.
web search: 
web search: interpretability attention value vector norm transformer explaining contextualization Kobayashi 2020 ...
web search: 
web search: sparse sensor placement optimal reconstruction QR pivoting dynamical systems ...
codex
The shell fails before reading files with “bwrap: Failed to make / slave: Permission denied.” Please paste the contents of harness/selector_api.py, harness/baselines.py, and RESULTS_TABLE.md here so I can check the proposals against your harness. I’ll continue the paper search meanwhile.
web search: 
web search: minimal sufficient explanations information bottleneck internal neural network activations ...
codex
The search has surfaced useful ideas from sparse sensing, quadrature, and data subset selection. I’m checking the closest pruning papers next: attention multiplied by value magnitude already appears in pruning, so that alone would not be a distinct proposal. The three local files remain unread because the shell cannot start.
web search: 
web search: Adaptive Token Sampling value norm attention significance score ECCV 2022 ...
web search: 
web search: "token" "pruning" "herding" ...
web search: 
web search: Token Pooling in Vision Transformers reconstruction error ICLR 2022 arxiv ...

## 2026-10-30 [claude] 가설 (분석 analysis_signal_fidelity_1030_003046_ecd0 결과 대기)
- H1 층 선택: 근거 — 아직 없음(분석 대기). 아이디어 — MaskedKD는 student 마지막 층 CLS attention만 쓴다. 층별 CLS attention이 masked teacher의 충실도 KL(T_full‖T_masked)을 얼마나 지키는지 재고, 마지막 층보다 나은 층이 있으면 그 층을 쓴다. 선행 대비 — MaskedKD(마지막 층 고정)의 단순 확장이므로 단독으로는 노벨티가 약함. parent=maskedkd로만 시험.
- H2 기울기 가중 attention / GAE(Chefer 2021) 선택: 근거 — Chefer et al. ICCV 2021, Hao et al. AAAI 2021. 아이디어 — student 자신의 예측 클래스 logit의 attention 기울기로 CLS attention을 가중해 "student 결정에 실제로 쓰이는 토큰"을 teacher에 보낸다(레이블·teacher 미사용). 선행 대비 — Prune and Merge(arXiv 2503.23455)는 추론 가속용으로 학습 중 기울기 가중 attention을 쓰고, KD에서 teacher 입력을 고르는 데 쓴 예는 못 찾음. 비용: student 역전파 1회 추가(학습 1스텝 대비 약 +student fwd×2) → K가 작을 때만 이득 가능.
- H3 attention-norm(Kobayashi 2020): ‖α·f(x)‖. 선행 — ATS(ECCV 2022)가 attention×value norm을 이미 씀 → 노벨티 낮음. 분석에서 비교용으로만 측정.
- 검증 계획: 분석의 fid_kl 순위가 기준선 screen(maskedkd/random/grid/center @12/25/49)의 val 순위와 맞는지 먼저 확인한다. 맞지 않으면 "teacher 충실도"는 teachability의 대리지표가 아니며(MaskedKD@98 24.74 > Full 24.38, run screen_maskedkd_k98_s0_1029_233749_7ad0 / screen_full_k196_s0_1029_233749_5549) 다른 지표(gap_kl 등)를 찾는다.
- [라운드 2] H2 구현: selectors/gradattn.py, 비용 +0.146 GFLOPs/img(trysel K=25). 제출 조건: 분석에서 gradattn_last의 fid_kl 또는 gap_kl이 cls_attn_L11(MaskedKD)보다 나은 K가 있고, 그 K에서 maskedkd−random 차이가 승격 기준(0.62%p) 이상일 것.

## 2026-10-30 [claude] 라운드 3
- H4 엔트로피 기여도 선택(student uncertainty attribution): 근거 — 아직 분석 대기(analysis_signal_fidelity_1030_014040_eb3d의 `entgrad_attn_last`, `entgrad_tok_L0`). 아이디어 — student 자기 예측 엔트로피의 기울기×입력(Shrikumar 2016 / Ancona ICLR 2018)으로 "student를 헷갈리게 하는 토큰"을 teacher에 보낸다. teachability 가설: KD 신호는 student가 불확실한 곳에서 가장 정보가 많다(gap_kl로 확인). MaskedKD(student가 이미 보는 곳)와 방향이 반대일 수 있음. 선행 대비 — 웹 검색(2건)에서 ViT KD의 teacher 입력 패치를 student 엔트로피 기여도로 고르는 예는 못 찾음. 가장 가까운 것: AD-TP(NeurIPS 2025 슬라이드, Integrated Gradients로 추론 가지치기), Fast SAM2(arXiv 2512.21333, MC dropout 불확실성으로 가지치기 회피), SelecTKD(LLM, 토큰별 손실 선택). 모두 추론 가속 또는 LLM이고, teacher 입력 선택이 아님. 비용: 역전파 1회(H2와 같은 수준).
- 판정 규칙: fid_kl만이 아니라 gap_kl도 본다. 엔트로피 신호는 fid_kl은 나쁘고 gap_kl은 클 수 있음 → 그 경우 기준선 screen 순위와 어느 지표가 맞는지로 결정.
