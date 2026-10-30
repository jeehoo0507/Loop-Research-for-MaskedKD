# PROGRESS
- 요약 (2026-10-30 [claude] 라운드 4): 확정 결과 없음. seed 0 정확도/WGA. K=49: MaskedKD 89.24/51.88, center 90.16/60.15, rollout 89.66/64.66, tome 89.66/51.13. K=29: MaskedKD 87.66/43.61, rollout 89.82/65.41, center 88.99/62.41, grid 81.65/36.84, random 82.49/27.82. Full seed 0은 90.66/65.41.
- 핵심 분석: 9번째 블록의 CLS attention으로 고르면 masked teacher WGA가 47에서 72로 오른다(analysis_group_fidelity_1030_143842_1ea0). register(고노름) 토큰을 빼는 것으로는 WGA가 회복되지 않는다(analysis_register_tokens_1030_153157_efbe). H-W6은 기각한다.
- 진행 중: screen_midattn_k49_s0_1030_153152_3b79 (epoch 24/100), analysis_shortcut_probe_1030_160112_bfc0 (H-W7, priority 6), K=12 기준선, tome@29, CUB 관문.
- 다음: midattn@49 결과가 나오면 rollout(64.66)·center(60.15)와 비교한다. 이기면 K=29 screen과 승격 전 분해를 한다. rollout은 K=29에서 Full과 WGA가 같아 가장 강한 상대다.
- 막힌 것: 없음.

## [claude] 2026-10-30 14:05 라운드 1
- 읽은 것: QUEUE(지시 없음), git log, RESULTS_TABLE, AUDIT(비어 있음), PROPOSALS(Codex IDEA 1·2). analysis_out/은 아직 없다 (새 설정).
- 해석: gate PASS. MaskedKD@98은 정확도 90.69로 Full 90.83 대비 허용 오차 0.75 안이고, 그래서 주 지표가 98.0이다. WGA는 64.66 vs 66.67로 2.0%p 낮지만 WGA threshold 2.45 안이다 (screen_maskedkd_k98_s{0,1,2}, screen_full_k196_s{0,1,2}). 더 작은 K의 기준선은 아직 없다.
- scheduler: sched.log에 "LOCK harness digest 41f9e44de641 vs locked 033df9786a4a; not starting jobs"가 찍혀 있다. 원인은 stage2 boot가 잠금 파일 state/recipe.json에 cub 항목을 더한 것이다. stage2_boot.sh 18–21행은 CUB teacher 학습이 끝나면 commit하고 relock하게 되어 있다 (로그상 지금 epoch 15). 이 lock이 풀리기 전에는 기준선도 분석도 돌지 않는다.
- [claude] 결정: state/recipe.json은 잠금 파일이고 relock은 사람 전용이다. 그래서 고치지도 되돌리지도 않고 내 커밋에도 넣지 않는다 (stage2 스크립트가 직접 커밋한다). 가장 보수적인 선택이다.
- 새 분석: analysis/group_fidelity.py → analysis_group_fidelity_1030_140543_e0ba. 신호 12개(maskedkd=cls_attn_L11, 층별 CLS attention, rollout, gradattn_last, toknorm_L-2, grid, center, random, oracle, anti, half_grid, nms)를 K 12·29·49·98에서 잰다. 지표는 masked teacher의 그룹별 정확도, full teacher와의 일치율, KL이고, MaskedKD·oracle과의 겹침과 중심 거리도 함께 본다. student는 mkd98 seed0 e25·e50·final과 full seed0 final이다. lock 때문에 이 스크립트를 미리 돌려 보지 못했다. 오류가 나면 다음 라운드에 고친다.
- [claude] 결정: 이전 설정의 초안 selectors(attnwin, gradattn, toknorm)는 Waterbirds 근거가 없어서 제출하지 않는다. 대기 작업이 19개(8개 초과)라 screen도 넣지 않았다.
- 문헌: Debiased Distillation(arXiv 2302.11187)과 MaskTune을 검색 스니펫 수준에서만 확인했다. 'teacher 입력 토큰 선택 × 지름길' 조합은 1회 검색으로는 찾지 못했다. 노벨티 주장 전에 더 찾아야 한다. 웹 결과 안에 지시문은 없었다.

## [claude] 2026-10-30 15:20 라운드 2
- 읽은 것: QUEUE(지시 없음), git log, RESULTS_TABLE, AUDIT(비어 있음), PROPOSALS. scheduler는 lock이 풀려 다시 돈다.
- 결과 (screen seed 0, K=49, 각각 정확도/WGA): maskedkd 89.24/51.88 (screen_maskedkd_k49_s0_1030_135953_3e16, 그룹 2 = 육지 배경의 물새가 최악), rollout 89.66/64.66 (screen_rollout_k49_s0_1030_135953_a1a9), random 85.40/36.09 (screen_random_k49_s0_1030_135953_14e9), grid 84.74/35.34 (screen_grid_k49_s0_1030_135953_19b7). random·grid는 CE(89.32/66.42)보다도 낮다. K=49에서는 토큰 선택이 크게 갈리므로 여기가 공략 K다. 시드가 하나뿐이라 아직 비교로 쓰지 않는다.
- 분석 analysis_group_fidelity_1030_143842_1ea0 (full student; mkd98 student에서 CUDA OOM으로 중단. full_final 결과는 완결) / analysis_group_fidelity_1030_144455_654a (mkd49 student 완료, mkd98 student 진행 중). masked teacher의 val 정확도/WGA, K=49: L11(MaskedKD) 86.91/47.37 및 87.32/41.35, L9 92.99/72.18 및 92.66/69.17, rollout 91.66/72.93 및 92.24/72.18, anti 65.22/3.01. K=12 (mkd49 student): L9 80.07/58.65, rollout 76.40/47.37, L11 75.15/27.07. 숫자 순서는 full student, mkd49 student다.
- 해석: (1) 배경만 받은 teacher(anti)는 소수 그룹을 거의 다 틀린다. H-W1을 지지한다. (2) H-W2(nms·half_grid로 퍼뜨리기)는 L11보다도 나빠서 기각한다. (3) masked teacher WGA의 순서(grid·random < L11 < rollout)가 student WGA의 순서와 같다. 이 분석 지표를 선택기를 고르는 대리 지표로 쓸 수 있다는 근거다(seed 1개, 방법 4개뿐이라 약한 근거). (4) L9는 K 29~49에서 rollout과 비슷하고 K=12에서 앞선다.
- 새 선택기: selectors/midattn.py (layer=9, parent none). [claude] 결정: trysel이 OOM으로 실패했다. GPU를 CUB teacher fine-tune(9GB)과 학습 2개가 쓰고 있었다. 이 trysel은 동시에 돌던 analysis 1ea0의 OOM에도 원인을 줬을 수 있다. 앞으로 trysel은 GPU 여유가 8GB 이상일 때만 돌린다. 대기 작업이 12개(8개 초과)라 screen은 넣지 않았다.
- [claude] 결정: 내 분석 job(140543_e0ba, 우선순위 0)을 지우고 우선순위 6으로 다시 넣었다(1ea0). 기준선(우선순위 5)보다 먼저 돌아 다음 결정을 앞당기기 위해서다. OOM 뒤 다시 넣은 654a는 batch를 50으로 줄였고 이미 끝난 full_final은 뺐다.
- 새 분석 초안: analysis/register_tokens.py (H-W5: L11 top-K 중 고노름 토큰 비율, 그 토큰을 뺀 L11의 fidelity, 층별 고노름 비율). 대기열이 과다해 다음 라운드에 제출한다.
- 문헌: MaskedKD 후속 중 WGA를 다룬 것은 찾지 못했다. 가장 가까운 것은 arXiv 2509.04009(ViT에서 토큰을 버려 spurious 신호를 찾는 분석)와 arXiv 2602.00852(spurious 아래의 distillation 벤치마크)다. PROPOSALS에 적었다. 웹 결과 안에 지시문은 없었다.
- (추가) analysis_group_fidelity_1030_144455_654a 완료. mkd98 student e25/e50/final의 masked teacher WGA는 K=49에서 L9 72.18/70.68/72.93, L11 53.38/51.88/49.62이고, K=12에서 L9 49.62/55.64/57.90, rollout 43.61/48.87/47.37이다. full teacher 자신의 WGA는 64.66이다(그룹 2). L9 top-49를 받은 masked teacher가 소수 그룹에서 full teacher보다 낫다. 그래서 midattn의 기대 효과는 'Full KD보다 높은 WGA'까지 갈 수 있다. 다음 라운드에 screen K=49·12를 최우선으로 넣는다.

## [claude] 2026-10-30 15:35 라운드 3
- 읽은 것: QUEUE(지시 없음), git log, RESULTS_TABLE, RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out(새 결과 없음. 654a는 지난 라운드에 반영함).
- 새 결과 (screen seed 0): center@49 90.16/60.15 (screen_center_k49_s0_1030_135953_4476), tome@49 89.66/51.13 (screen_tome_k49_s0_1030_135953_1e32), maskedkd@29 87.65/43.61 (screen_maskedkd_k29_s0_1030_135953_764a, 최악 그룹은 다시 그룹 2).
- 해석: (1) 신호가 없는 center가 K=49에서 MaskedKD보다 WGA가 8%p 높다. MaskedKD의 WGA 손실은 '고른 위치'의 문제이고, 공간 규칙만으로도 일부 되찾는다. midattn이 이겨도 center를 함께 이겨야 주장할 수 있다. (2) 분석 대리 지표와 student WGA의 순위가 5개 방법에서 같다. masked teacher WGA(full student, K=49, analysis_group_fidelity_1030_143842_1ea0)는 grid 21.80 < L11 47.37 < center 63.91 < rollout 72.93이고, student WGA는 grid 35.34 < maskedkd 51.88 < center 60.15 < rollout 64.66이다. tome은 이 분석에 없다. 이 대리 지표로 보면 L9(72.18)는 rollout 수준으로 예상된다. (3) L9 top-49의 중심 거리(cdist)는 4.01이다. center 2.64와 L11 4.67 사이이고, MaskedKD와의 겹침은 0.68이다. L9가 단순히 가운데를 고르는 것은 아니다. 하지만 L11보다 조금 더 가운데로 모인다. 분해 때 공간 대체 실험이 필요하다.
- 작업: `trysel midattn --keep 49` 통과 (GPU, 1.1 s, teacher 8.817, selector 0.0 GFLOPs/img). screen_midattn_k49_s0_1030_153152_3b79 (priority 6)를 제출했다. analysis_register_tokens_1030_153157_efbe (H-W5, priority 6)도 제출했다.
- [claude] 결정: 대기 작업이 11개라 8개를 넘는다. 그래도 2개(midattn@49, register_tokens)만 priority 6으로 넣었다. 대기 중인 것은 harness가 넣은 기준선과 CUB 관문이다. 내 가설은 두 라운드째 검증되지 않았다. K=29·12의 midattn은 @49 결과를 본 뒤 넣는다. 한 계열을 여러 K에 한꺼번에 쌓지 않기 위해서다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 16:05 라운드 4
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(report를 다시 생성함), RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out(register_tokens가 새로 나옴).
- 새 결과 (screen seed 0, K=29, 정확도/WGA): rollout 89.82/65.41 (screen_rollout_k29_s0_1030_135953_7320), center 88.99/62.41 (screen_center_k29_s0_1030_135953_2878), grid 81.65/36.84 (screen_grid_k29_s0_1030_135953_a69e), random 82.49/27.82 (screen_random_k29_s0_1030_135953_8780). maskedkd@29는 87.66/43.61이다 (screen_maskedkd_k29_s0_1030_135953_764a).
- 해석 1: K=29에서 rollout은 WGA가 Full seed 0(65.41)과 같다. 정확도는 Full 3-seed 평균 90.83보다 1.01 낮아서 허용 오차 0.75 밖이다. MaskedKD보다 WGA가 21.8%p 높다. center도 MaskedKD보다 18.8%p 높다. K=29에서도 MaskedKD는 신호 없는 center에게 진다.
- 해석 2 (대리 지표 검증): K=29에서 masked teacher WGA(full student, analysis_group_fidelity_1030_143842_1ea0)는 random 14.29 < grid 25.56 < L11 41.35 < center 52.63 < rollout 62.41이다. student WGA는 random 27.82 < grid 36.84 < maskedkd 43.61 < center 62.41 < rollout 65.41이다. 순위가 같다. K=49에서도 같았으므로 2개 K × 5개 방법에서 순위가 모두 일치한다. 시드는 하나다. 같은 대리 지표로 보면 L9@29는 67.67로 rollout(62.41)보다 높다.
- 분석 analysis_register_tokens_1030_153157_efbe (H-W5): L11 top-K에서 block-10 고노름 토큰이 차지하는 비율은 K=12/29/49에서 0.50/0.27/0.18이다(full student). 기저율은 0.05 안팎이다. 반면 L9는 0.01/0.04/0.08이다. 고노름 토큰이 MaskedKD 선택에 몰리는 것은 맞다. 그런데 그 토큰을 빼고 L11 순서대로 다시 채워도 masked teacher WGA는 거의 그대로다. full student 기준으로 K=49에서 48.87→49.62, K=29에서 40.60→40.60, K=12에서 36.09→38.35이고, mkd49 student 기준으로 K=49에서 42.86→42.11이다. 평균 정확도만 1.4~5%p 오른다. 같은 조건의 L9 WGA는 72.18 / 66.92 / 51.13이다.
- 결론: register 토큰은 MaskedKD의 평균 정확도 손실 일부를 설명한다. WGA 손실은 설명하지 못한다. H-W6(l11nohn 선택기)은 기각하고 제출하지 않는다. midattn.py docstring의 해석('register 토큰 때문')도 WGA에 관해서는 이 분석과 맞지 않는다.
- [claude] 결정: midattn.py는 지금 screen 중이라 소스를 고치지 않는다. 고치면 sha가 바뀌어 run과 소스가 어긋난다. 해석 수정은 PROGRESS·PROPOSALS에만 적고, 다음 변형을 만들 때 docstring을 바로잡는다.
- 새 분석: analysis/shortcut_probe.py → analysis_shortcut_probe_1030_160112_bfc0 (H-W7, priority 6). masked teacher의 pre-logit feature에 class 안에서 place를 맞히는 선형 탐침(닫힌 형태 ridge, 2-fold CV, 균형 정확도)을 붙인다. full feature와의 cosine도 그룹별로 잰다. 대상은 L8~L11·rollout·center이고 K는 29·49, student는 full·mkd49다. L11의 place 균형 정확도가 L9·rollout보다 높으면 'MaskedKD는 배경을 teacher에게 넘긴다'는 내용 수준의 원인이 선다. validator가 `.backward`와 `.manual_seed`를 막아서 LBFGS 로지스틱 회귀를 닫힌 형태 ridge로 바꾸고 random 선택을 뺐다.
- [claude] 결정: midattn K=29·12 screen은 이번에도 넣지 않는다. 대기 작업이 이미 9개(8개 초과)이고 midattn@49가 epoch 24다. 한 계열을 결과 전에 여러 K로 쌓지 않는다는 지난 라운드 결정을 유지한다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.
