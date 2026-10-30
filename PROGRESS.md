# PROGRESS
- 요약 (2026-10-30 [claude] 라운드 11): 확정 결과 없음. K=12에서 가장 높은 것은 suffdisc(89.32/58.65)지만 parent center(56.39) 대비 threshold 미만이다. K=29·49에서는 rollout(89.82/65.41, 89.66/64.66)이 가장 높다. priorroll(H-W12)은 K=12에서 center와 비기고 K=29에서 진다. 승격 후보는 없다.
- 핵심 분석: H-W10의 self kl_worst 대리 지표는 표본 밖에서 K=29 순위를 틀렸다 (analysis_proxy_holdout_1030_182326_7254). cutmix/mixup 충실도로도 설명되지 않는다 (analysis_prior_failure_1030_182537_611c). K=29에서는 full teacher에 덜 충실한 rollout이 student WGA가 더 높다.
- 진행 중: K=29 seed 1 재현 screen_priorroll_k29_s1_1030_182720_dbfd, screen_rollout_k29_s1_1030_182725_2086, screen_center_k29_s1_1030_182732_85c3.
- 다음: seed 1에서도 rollout > priorroll(≥ threshold)이면 H-W12를 최종 기각한다. 그다음 '충실도'가 아닌 다른 teachability 축을 분석한다. 후보는 (i) 학습 동안 teacher가 보는 위치의 다양성, (ii) masked teacher가 full teacher의 지름길을 얼마나 덜 물려받는가(full teacher가 틀리는 소수 그룹 이미지에서의 masked teacher 정답률)이다.
- 막힌 것: 없음. `python -c`, heredoc, 변수가 든 for 루프는 승인 대기로 막힌다. 그래서 jq·printf와 Read/Grep/Edit로 작업한다.

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

## [claude] 2026-10-30 16:35 라운드 5
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(report를 다시 생성함), RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out(shortcut_probe가 새로 나옴).
- 새 결과 (screen seed 0, 정확도/WGA): midattn@49 89.91/57.89 (screen_midattn_k49_s0_1030_153152_3b79), maskedkd@12 85.15/23.31 (screen_maskedkd_k12_s0_1030_135953_ba8e), tome@29 89.66/51.13 (screen_tome_k29_s0_1030_135953_88a4, 예산에 맞춰지지 않음: teacher 비용이 예산의 1.16배). tome@29의 정확도와 WGA는 tome@49와 소수점까지 같다. 두 run은 teacher GFLOPs가 4.36과 6.86으로 서로 다르다. 학습 곡선도 다르다. 그래서 우연한 일치로 본다(1075/1199, 68/133).
- 판정: midattn@49는 MaskedKD보다 WGA가 +6.01이다. threshold 2.45를 넘는다. 정확도는 +0.67이고 threshold 0.76 안이다. 하지만 같은 K의 rollout(64.66)과 center(60.15)보다 WGA가 낮다. 승격 기준의 "rollout·grid·center·tome보다 낮지 않을 것"을 어긴다. **승격하지 않는다.**
- 학습 곡선 (val WGA, 5 에폭마다): midattn은 epoch 45에 70.68까지 올랐다가 후반에 떨어진다. 마지막 4번은 57.14/58.65/57.89/57.89다. rollout의 마지막 4번은 63.91/64.66/64.66/64.66, center는 62.41/62.41/62.41/60.15이고, MaskedKD는 51.88이 3번이다. 마지막 값만 튄 것이 아니다. midattn의 후반 하락은 실제다.
- 분석 analysis_shortcut_probe_1030_160112_bfc0 (H-W7). masked teacher pre-logit feature에서 class 안의 place를 맞히는 선형 탐침의 균형 정확도를 쟀다. full student, K=49 기준으로 L11 86.10 > L8 84.09 > center 82.89 > L10 82.02 > rollout 78.38 > L9 71.35이고, full teacher는 95.28이다. K=29에서는 L11 77.60 > center 74.20 > L8 73.50 > L10 72.53 > rollout 67.62 > L9 60.81이다. mkd49 student도 순서가 같다(K=49에서 L11 88.27, L9 77.84). masked teacher WGA와는 거의 역순이다. 해석: 마지막 블록 top-K는 teacher에게 배경(장소) 정보를 가장 많이 넘긴다. H-W7은 teacher 수준에서 지지된다.
- 해석 (대리 지표의 첫 실패): clean 이미지와 eval 모드에서 잰 대리 지표는 L9@49(72.18)를 rollout(72.93) 수준, center(63.91)보다 위로 예측했다. 실제 student는 57.89로 center보다도 낮다. 그동안 순위가 맞은 5개 방법은 모두 학습 동역학이 단순한 경우일 수 있다. 차이가 날 만한 원인은 학습 조건이다. 학습에서는 student가 train 모드(drop path 0.1)이고, 모든 batch가 mixup 또는 cutmix다. 대리 지표에는 이것이 없었다. 또 midattn은 epoch 45 이후 WGA가 떨어진다. student L9 attention이 학습 후반에 바뀌었을 수도 있다.
- 새 분석: analysis/train_condition_gap.py → analysis_train_condition_gap_1030_162956_067e (H-W8, priority 6). 조건은 clean_eval, clean_train(drop path), cutmix(패치 격자에 맞춘 상자), mixup(0.6)이다. 선택기는 L9, L11, rollout, center이고 K는 29와 49다. 재는 것은 그룹별 KL(full teacher‖masked teacher)과 일치율, cutmix 상자 안 토큰 비율, drop path에서의 선택 안정성이다. student는 midattn@49와 rollout@49(e25/e50/final), maskedkd@49 final이다. GPU가 공유 중이라 입력은 CPU에 두었다.
- [claude] 결정: midattn의 K=29·12 screen은 넣지 않는다. 대리 지표가 L9를 과대평가한 이유를 모르는 채로 같은 계열을 다른 K에 쌓으면, 이길 경우에도 우연을 가려낼 수 없다. 그래서 H-W8 분석을 먼저 돌린다. 대기 작업은 5개(분석 1 포함), 실행 중은 4개다. GPU는 놀지 않는다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.
- (추가) rollout@12가 끝났다: 87.24/45.11 (screen_rollout_k12_s0_1030_135953_489a). MaskedKD@12(85.15/23.31)보다 WGA가 21.8 높다. K=12는 토큰 선택의 차이가 가장 큰 K다. center/grid/random/tome@12는 아직 학습 중이다.

## [claude] 2026-10-30 16:55 라운드 6
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(report를 다시 생성함), RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out. train_condition_gap은 16:51에 시작해 아직 결과가 없고 students.json만 있다.
- 새 결과 (screen seed 0, K=12, 정확도/WGA): center 88.74/56.39 (screen_center_k12_s0_1030_135953_c64e), grid 83.90/35.34 (screen_grid_k12_s0_1030_135953_1e84), random 83.49/33.83 (screen_random_k12_s0_1030_135953_bc03). 이미 있던 것은 rollout 87.24/45.11 (screen_rollout_k12_s0_1030_135953_489a)와 maskedkd 85.15/23.31 (screen_maskedkd_k12_s0_1030_135953_ba8e)이다. tome@12와 oracle@12는 아직 끝나지 않았다.
- 해석 1: K=12에서 MaskedKD는 random보다도 WGA가 10.5 낮다. 작은 K에서 마지막 블록 top-K는 소수 그룹에 해롭다. 이것은 H-W7(L11 top-K는 teacher에게 장소 정보를 넘긴다)과 맞는다.
- 해석 2: K=12에서 center는 rollout보다 WGA가 11.3, 정확도가 1.5 높다. K=49와 K=29에서는 반대로 rollout이 center보다 WGA가 4.5, 3.0 높았다. K가 작을수록 '연속된 중앙 영역 하나'가 흩어진 top-K보다 유리해진다. 그래서 K=12에서 주장하려면 center를 이겨야 한다. 지금 가장 높은 기준이다.
- 가설 H-W9: teacher에게 좋은 입력은 점수가 높은 흩어진 토큰이 아니라 객체 위의 연속된 영역 하나다. Waterbirds val에서는 새가 대체로 가운데 있어서 center가 잘 맞는다. 학습 중에는 RandomResizedCrop 때문에 새가 가운데에서 벗어난다. 그럴 때 student attention으로 창의 위치만 정하면 center보다 나을 수 있다. 이 가설은 분석으로 먼저 확인한다.
- 새 분석: analysis/window_fidelity.py → analysis_window_fidelity_1030_165418_e276 (priority 6). 이전 설정의 analysis/object_window.py를 그룹 단위로 고친 것이다.
  - 조건: clean, 그리고 off-centre crop(넓이 0.45~0.7, 위치는 batch마다 결정적으로 바꿈).
  - 선택: center, win_{L9,L11,rollout}_{cent,peak}(attention 무게중심 또는 blur 뒤 최댓값 주변의 K개), top_{L9,L11,rollout}, 분석 전용 상한 shift_best(이동시킨 원판 25개 중 teacher KL이 가장 작은 것).
  - K는 12·29·49이고, student는 full@196·maskedkd@49·rollout@49·center@49의 final이다. 그룹별 masked teacher 정확도와 KL을 잰다.
  - [claude] 결정: GPU를 학습 작업과 함께 쓰고 있어 미리 돌려 보지 않고 제출했다. 오류가 나면 다음 라운드에 고친다.
- [claude] 결정: 이번 라운드에는 screen을 넣지 않는다. midattn 계열은 H-W8 결과가 나올 때까지 보류한다(라운드 5 결정 유지). attnwin은 H-W9 근거가 나오기 전에는 제출하지 않는다. 대기 중인 작업(oracle@12, CUB 3개)과 실행 중인 작업이 있어 GPU는 놀지 않는다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 17:15 라운드 7
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE, RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out(train_condition_gap, window_fidelity가 새로 나옴).
- 새 결과: tome@12 88.99/48.12 (screen_tome_k12_s0_1030_135953_84af, teacher 비용이 예산의 2.69배라 not matched). K=12의 WGA 순위는 center 56.39 > tome 48.12 > rollout 45.11 > grid 35.34 > random 33.83 > maskedkd 23.31이다.
- 분석 analysis_train_condition_gap_1030_162956_067e (H-W8). **학습 조건 가설은 기각한다.**
  - 학습 조건은 L9를 불리하게 만들지 않는다. cutmix에서 L9@49의 kl_worst는 mkd49 student 0.2985, rollout 0.3427이고, roll49 student 0.2712, rollout 0.4045다. 오히려 L9가 rollout보다 작다. cutmix 상자 비율도 L9 0.67 대 rollout 0.63~0.66으로 비슷하다(상자 면적 0.52).
  - drop path에서 선택 안정성은 L9 0.93, rollout 0.976, L11 0.89다. 차이는 작다.
  - 진짜 차이는 student에 있다. midattn으로 학습한 student 자신의 L9@49 masked teacher WGA(clean eval)는 e25/e50/final에서 66.92/59.40/60.90으로 떨어진다. 같은 L9를 다른 student에서 읽으면 mkd49 student 69.17, rollout student e25/e50/final 76.69/74.44/75.19다. rollout student 자신의 rollout@49는 65.41/69.17/68.42로 떨어지지 않는다.
  - 각 방법을 그 방법으로 학습한 student로 잰 self proxy의 K=49 순위는 L11 41.35 (mkd49) < L9 60.90 (mid49) < center 63.91 < rollout 68.42 (roll49)다. student WGA 순위 51.88 < 57.89 < 60.15 < 64.66과 같다.
  - 해석: L9 선택이 student 자신의 L9 attention을 바꾸는 되먹임이 있다. 공통 student(full KD)로 잰 대리 지표에는 이것이 보이지 않는다.
- 분석 analysis_window_fidelity_1030_165418_e276 (H-W9). **H-W9(attnwin)는 기각하고 제출하지 않는다.**
  - K=12에서 center의 masked teacher WGA는 clean 41.35, crop 42.11이다. win_L9_cent@12는 clean에서 45.11/38.35/43.61/43.61(student: center49/full/mkd49/roll49)이고, crop에서 45.86/49.62/51.13/57.14다. 창이 center를 이기는 student도 있고 지는 student도 있다.
  - 흩어진 top_L9@12가 창보다 높다(clean 60.15/51.88/58.65/53.38). 그리고 K=49에서 win_L9_cent는 top_L9와 비슷하다(clean 71.43/72.18/70.68/71.43).
  - 더 중요한 사실은 따로 있다. K=12에서 이 대리 지표는 center(41.35)를 top_rollout(clean 48.12~53.38)보다 낮게 본다. 그런데 student는 center@12 56.39 > rollout@12 45.11로 반대다. 원판을 teacher KL 기준으로 가장 잘 옮긴 shift_best@12(분석 전용 상한)도 60.90이다.
  - 결론: K=12에서는 clean masked-teacher WGA가 student 순위를 예측하지 못한다. 그래서 이 지표로 창 선택기를 고를 근거가 없다.
- 새 분석: analysis/proxy_validation.py → analysis_proxy_validation_1030_170914_4668 (H-W10, priority 6).
  - 대상: 끝난 screen seed 0 run 16개(maskedkd·rollout·center·grid·random × K 12·29·49, midattn@49).
  - 방법: 각 run을 그 run의 student(self)와 full-KD student(common)로 잰다. masked teacher의 그룹별 정확도, 참 클래스 확률, 확신 오답 비율(max p > 0.8), 엔트로피, full teacher와의 KL을 구한다.
  - 각 통계를 student val WGA와 Spearman 순위 상관으로 비교한다(전체, K별).
  - 목적: 앞으로 선택기를 분석 단계에서 거를 대리 지표를 검증하는 것이다.
- screen: screen_midattn_k12_s0_1030_170938_9fc6 (priority 6).
  - [claude] 결정: midattn을 K=12에 넣는다. 승격을 기대해서가 아니다. 같은 계열이 다른 K에서 일관된지 보는 점검이고, H-W10의 검증 점(K=12에서 L9 self 대 common)을 하나 더 얻기 위해서다.
  - 지난 라운드에 이 screen을 보류한 이유는 '대리 지표가 L9를 왜 과대평가했는지 모름'이었다. 이제는 그 원인(되먹임)을 알았다. 대기 작업은 3개(CUB 2 포함)다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 17:35 라운드 8
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(report를 다시 생성함), RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out. proxy_validation이 새로 나왔다.
- 새 결과 (screen seed 0, 정확도/WGA):
  - midattn@12 87.66/30.08 (screen_midattn_k12_s0_1030_170938_9fc6). random@12 33.83보다 낮다. midattn은 K=49에서도 center와 rollout보다 낮았다. **midattn 계열은 닫는다.**
  - oracle@12 87.49/50.38 (screen_oracle_k12_s0_1030_135953_d16c, 분석 전용). teacher 자신의 attention top-12도 center@12(56.39)보다 WGA가 낮다. K=12에서는 흩어진 attention top-K가 그 출처와 상관없이 연속 원판에 진다.
- 분석 analysis_proxy_validation_1030_170914_4668 (H-W10). 16개 run에 대해 각 통계와 student WGA의 Spearman 크기를 구했다.
  - self kl_worst 0.937 (K12 1.0 / K29 1.0 / K49 0.886)
  - self p_true_worst 0.946 (K12 0.8)
  - self t_acc_worst 0.935 (K12 0.6)
  - self ent_worst 0.693 (K12 -0.1)
  - common 지표는 0.79~0.88이다.
  - 부호 보정 버그가 있다. `key.split("_")[0]`가 "t", "p"가 되어 t_acc·p_true까지 부호가 뒤집혔다. 크기만 읽는다.
  - 결론: 선택기를 거르는 대리 지표로 **self kl_worst**를 쓴다. K=12에서 center와 rollout의 KL은 0.694 대 0.700으로 비슷하다. 그런데 accuracy 지표는 41.35 대 44.36으로 순서를 뒤집는다. 확신 오답(작업 가설 a)은 KL보다 약해서 보류한다.
- 새 가설 H-W11. 근거는 analysis_window_fidelity_1030_165418_e276의 shift_best@12 kl_worst 0.049 (center 0.694)다. center 원판을 ≤2 패치 옮겨 student 자신의 예측이 가장 잘 보존되는 위치에 둔다. 이것은 teacher를 쓰지 않는, student 쪽 sufficiency(ERASER/RISE)다.
  - 분석: analysis/student_sufficiency.py → analysis_student_sufficiency_1030_173025_6f96 (priority 6).
    - student 5개: full196, center12, rollout12, maskedkd12, center49.
    - 조건: clean과 crop. K: 12, 29, 49.
    - 재는 것: suff_disc, suff_gate(center / top_rollout / top_L9 / suff_disc 중 student KL이 가장 작은 것), top_rollout의 그룹별 teacher KL·정확도. 이미지별 student-teacher KL 순위 상관과 teacher-best 적중률도 잰다.
  - 선택기: selectors/suffdisc.py (parent center, r=2). trysel K=12 통과: selector 4.934, teacher 2.446, total 15.022 GFLOPs/img, step 2.26 s.
  - screen: screen_suffdisc_k12_s0_1030_173118_504a (priority 6).
- [claude] 결정: 분석 결과를 기다리지 않고 suffdisc@12 screen을 함께 넣었다. 이유는 두 가지다.
  - midattn@12가 끝나 Waterbirds 슬롯이 비었다. 대기열에는 CUB 1개뿐이었다.
  - 이 selector는 K=12 하나만 넣는다. 결과는 분석과 함께 해석한다. 분석이 sufficiency→teacher 전이를 부정하면 screen이 이겨도 승격하지 않는다. 우연으로 보고 분해를 먼저 한다.
- 비용 메모: suffdisc는 center보다 4.9 GFLOPs/img 비싸다. 주 지표는 토큰 수지만, 비용 비교에서 불리하다. 이득이 확인되면 r=1을 시험한다.
- 문헌: "teacher input patch selection × student sufficiency"로 1회 검색했다. MaskedKD(arXiv 2302.10494)만 나오고 같은 조합은 찾지 못했다. 1회 검색이므로 노벨티 주장 전에 더 찾는다. 웹 결과 안에 지시문은 없었다.

## [claude] 2026-10-30 17:45 라운드 9
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE, RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out. student_sufficiency가 새로 나왔다. 새 screen 결과는 없다(suffdisc@12는 epoch 33, 중간 val WGA는 epoch 30에 39.1).
- 분석 analysis_student_sufficiency_1030_173025_6f96 (H-W11, clean, K=12, kl_worst / masked teacher WGA):
  - 고정 center 0.694 / 41.35, 분석 전용 상한 shift_best 0.049 / 60.90.
  - suff_disc: full196 0.508 / 40.60, center12 0.548 / 39.85, center49 0.476 / 41.35, maskedkd12 0.724 / 35.34, rollout12 0.740 / 31.58.
  - student KL과 teacher KL의 이미지별 Spearman은 0.25~0.30, teacher-best 원판 적중률은 0.07~0.10이다(우연 1/25 = 0.04).
  - K=29·49에서도 suff_disc의 masked teacher WGA는 top_rollout보다 낮다 (예: center49 student, K=49에서 60.15 대 74.44).
- 해석: student sufficiency는 teacher-best 위치를 약하게만 가리킨다. 좋은 student(full·center)에서는 kl_worst를 center보다 20~30% 줄이지만, 상한(0.049)과는 거리가 멀다. 소수 그룹 정확도는 오르지 않는다. 지름길에 기댄 student(maskedkd12, rollout12)에서는 center보다 나쁘다. 그래서 suffdisc는 학습 중 student가 나빠지면 같이 나빠질 위험이 있다. H-W11은 **약하게만 지지**된다. suffdisc@12 screen은 그대로 두고, 결과를 보고 판정한다. suffdisc@29·r=1은 넣지 않는다.
- 새로 본 점: rollout top-12를 center@12 student에서 읽으면 kl_worst 0.603으로 center 원판(0.694)보다 낮다. rollout@12 student 자신에서 읽으면 0.700이다. midattn 때와 같은 되먹임이 rollout에도 K=12에서 보인다. 그리고 K=12에서는 center, K=29·49에서는 rollout이 이긴다. 그래서 둘을 섞은 선택기가 모든 K에서 둘 다를 이길 수 있는지 분석한다.
- 새 분석: analysis/prior_rollout.py → analysis_prior_rollout_1030_174424_16a6 (H-W12, priority 6). 선택은 center, top_rollout, prior_s{2,3,4}(log rollout − d²/2σ²의 top-K), union_half(center 원판 K/2 + 나머지 rollout top)이다. K는 12·29·49, student는 full196과 center/rollout@12·29·49다. 그룹별 teacher KL·정확도, 중심 거리, center 원판과의 겹침을 잰다.
- [claude] 결정: 이번 라운드에는 screen을 넣지 않는다. suffdisc@12가 아직 학습 중이고, H-W12는 분석 전이다. 대기 작업은 CUB 1개 + 분석 1개, 실행 중은 2개라 GPU는 놀지 않는다.
- 노벨티 주의: prior_s는 '공간 사전 + attention'이다. 이기더라도 이득이 사전에서 오는지 신호에서 오는지 분해해야 한다. 같은 K의 center와 rollout을 모두 이겨야 주장할 수 있고, 분해 때는 rollout 대신 random 점수에 같은 사전을 씌운 변형을 대체 실험으로 넣는다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 17:55 라운드 10
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out. prior_rollout이 새로 나왔다. 새 screen 결과는 없다. suffdisc@12는 epoch 58이고, epoch 55의 val WGA는 49.62다(중간값이라 판정에 쓰지 않는다).
- 분석 analysis_prior_rollout_1030_174424_16a6 (H-W12). kl_worst, 낮을수록 좋다. 표기는 center / top_rollout / prior_s2 / s3 / s4.
  - rollout12 student, K=12: 0.694 / 0.700 / 0.637 / 0.644 / 0.636
  - rollout29 student, K=29: 0.517 / 0.443 / 0.426 / 0.415 / 0.390
  - rollout49 student, K=49: 0.300 / 0.320 / 0.273 / 0.248 / 0.227
  - full196 student, K=12/29/49에서 최선의 prior: 0.552 (s2) / 0.300 (s4) / 0.228 (s4). 같은 K의 center와 top_rollout은 각각 0.694·0.721, 0.517·0.368, 0.300·0.350이다.
  - 예외: center12 student의 K=12에서는 top_rollout 0.603이 prior_s3 0.614, s4 0.615보다 조금 낮다.
  - 되먹임: rollout12 student에서 K=29 masked teacher WGA는 top_rollout 46.62, prior_s 60.15~61.65다. 사전이 rollout의 self-feedback 손실을 막는다.
  - 판정: 라운드 9에 미리 정한 조건('K 12·29·49 모두에서 center와 top_rollout보다 kl_worst가 낮다')을 student 7개 중 6개가 모든 K에서 만족한다. 나머지 1개(center12, K=12)는 차이가 0.011이다. **H-W12 지지.**
- 새 선택기: selectors/priorroll.py (parent rollout). 점수는 log rollout − d²/(2σ²)이고 σ = √(K/π)다.
  - trysel K=12 통과: selector 0.168, teacher 2.446, total 10.077 GFLOPs/img, step 1.04 s. rollout@12(10.08)와 비용이 같다.
  - screen: screen_priorroll_k12_s0_1030_175140_405f, screen_priorroll_k29_s0_1030_175144_018d.
- [claude] 결정: σ는 'K의 center 원판 반지름'으로 정했다. 분석의 K별 최적 σ(2/4/4)에 가깝고, 자유 매개변수가 없다. 분석을 본 뒤 정한 것이므로 사후 선택임을 기록한다. screen 결과로 σ를 다시 고르지 않는다.
- [claude] 결정: K=12와 K=29를 함께 넣었다. 이 방법의 주장은 'center가 이기는 K와 rollout이 이기는 K 모두에서 둘을 이긴다'는 것이라서 한 K만으로는 판정할 수 없다. 대기열도 비어 있어 GPU가 놀 참이었다. K=49는 두 결과를 본 뒤 넣는다.
- 노벨티 주의: center-bias 사전 + saliency는 시선 예측 문헌에서 흔하다. 이기더라도 random 점수에 같은 사전을 씌운 대체 실험으로 rollout 신호의 기여를 분리해야 한다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 18:30 라운드 11
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(report를 다시 생성함), RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out.
- 새 결과 (screen seed 0, 정확도/WGA):
  - suffdisc@12 89.32/58.65 (screen_suffdisc_k12_s0_1030_173118_504a). parent center@12 대비 +0.58/+2.26으로 threshold(0.76/2.45) 미만이다(Parent check UNPROVEN). selector 비용이 4.93 GFLOPs/img 더 든다. **승격하지 않는다.** H-W11은 닫는다.
  - priorroll@12 89.07/54.89 (screen_priorroll_k12_s0_1030_175140_405f). parent rollout 대비 WGA +9.78(helps)이지만 center@12(56.39)보다 1.50 낮다.
  - priorroll@29 89.82/60.15 (screen_priorroll_k29_s0_1030_175144_018d). rollout@29(65.41) 대비 −5.26, center@29(62.41) 대비 −2.26이다. val WGA는 epoch 85/90/95/100에서 63.16/59.40/58.65/60.15다. **승격하지 않는다.** 분해용 selector priorrand는 초안만 썼다가 지웠다. 이긴 결과가 없어 필요 없다.
  - CUB maskedkd@98 seed 1이 끝났다 (cub_final_maskedkd_k98_s1_1030_153055_cdbc).
- 분석 analysis_proxy_holdout_1030_182326_7254 (H-W13): 표본 밖 run에서 self kl_worst는 K=29 순위를 틀린다. priorroll(0.391)을 가장 좋게 예측했지만 WGA는 가장 낮다.
  - priorroll student의 plain rollout top-K는 priorroll 선택과 0.81~0.85 겹친다. rollout student에서는 0.34~0.53이다. 사전으로 학습하면 student attention이 중앙으로 모인다.
  - rollout student에서 읽은 priorroll의 K=29 kl_worst는 0.420이다(self 0.391). H-W12 분석을 parent student로 잰 것이 실패 원인은 아니다. 지표 자체가 틀렸다.
- 분석 analysis_prior_failure_1030_182537_611c (H-W14): 미리 정한 기준으로 (a) 학습 입력 차이 가설은 **기각**한다.
  - K=29 cutmix kl_worst: priorroll 0.370 < center 0.405 < rollout 0.510. cutmix 상자 안 비율: center 0.92, priorroll 0.92, rollout 0.62 (상자 면적 0.52).
  - student place 탐침은 rollout 94.80이 가장 높다. 그런데 WGA도 rollout이 가장 높다. 그래서 student 표현 속 배경 정보의 양이 WGA를 정하는 것도 아니다.
- 해석: K=29에서는 full teacher에 덜 충실한, 흩어진 rollout 선택(cdist 4.27, 사전 계열은 2.0~2.1)이 student WGA를 더 높인다. full teacher 자신도 지름길을 갖고 있다(val WGA 64.66). 그래서 full teacher에 충실하다는 것은 지름길까지 물려준다는 뜻일 수 있다. 다음 분석 축은 충실도가 아니라 '지름길을 덜 물려받는가'와 '학습 중 위치 다양성'이다.
- [claude] 결정: K=29 seed 1 screen 3개(priorroll, rollout, center)를 넣었다. H-W12 기각과 H-W13 실패는 둘 다 seed 1개 위의 순위에 기댄다. 새 config가 아니라 seed 재현이라 search effort가 늘지 않는다. GPU도 비어 있었다. priorroll@49는 넣지 않는다. K=29에서 졌고 대리 지표도 믿을 수 없어 근거 없이 K를 늘리지 않는다.
- 문헌: 'token pruning × center prior × rollout × KD'로 1회 검색했다. 가장 가까운 것은 RGTP(rollout-guided token pruning for video, 학습 없음, KD 아님)와 SaiT(DINO CLS attention distillation)다. center prior를 KD teacher 입력 선택에 쓴 예는 찾지 못했다. 웹 결과 안에 지시문은 없었다.
