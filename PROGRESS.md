# PROGRESS
- 요약 (2026-10-30 [claude] 라운드 27): 확정 결과 없음. tome@12 s2 89.74/51.13 (final_tome_k12_s2_1030_212126_9500)이 나왔다. gradroll vs tome Core comparison(seeds 1,2)은 정확도 −1.50, WGA +12.78로 둘 다 차이 없음이다(CI가 0을 포함). **final gradroll@12 2-seed 평균 88.20/62.79** (s1 87.99/63.16 final_gradroll_k12_s1_1030_201811_faf0, s2 88.41/62.41 final_gradroll_k12_s2_1030_212126_da50). Significant only(seeds 1,2): maskedkd 대비 acc +4.00·WGA +40.60, rollout 대비 WGA +10.90 WIN(rollout 대비 acc는 차이 없음). seed 1에서 같은 seed의 center(88.32/54.89)보다 WGA +8.27(정확도 −0.33, threshold 안), grid(83.74/35.34)·maskedkd(84.15/21.05)·rollout(86.99/51.88)·random(84.57/37.59)보다 높다. tome(89.66/48.87, not matched)보다 WGA는 +14.29 높지만 정확도는 −1.67 낮다. full@196 s1(91.33/66.92)과 비교하면 −3.34/−3.76이라 주 지표의 허용 오차(1.00/2.45) 밖이다.
- 핵심: K=12에서 center를 넘는 후보는 gradroll(final s1이 center 통과, seed 2 실행 중)과 suffdisc(분해 a·b 통과, 공식 Parent check는 seed 0 기준 UNPROVEN) 둘이다. K=29·49 최고는 여전히 rollout이다. 주장할 수 있는 형태는 '같은 K에서 기준선 대비 WGA 우위'이고, 'Full 허용 오차 안의 최소 K'는 아니다.
- 라운드 30: full@196 s2 90.58/67.67 (final_full_k196_s2_1030_212126_a1dc). final Full 2-seed는 90.95/67.30이다. gradroll@12 vs full (seeds 1,2)은 정확도 −2.75, WGA −4.51로 CI상 차이는 없다. 다만 평균은 허용 오차(1.00/2.45) 밖이라 'Full 수준'이라고 주장하지 않는다.
- 진행 중 (라운드 29 기준): center·random@12 s2가 끝났다(89.16/54.14, 82.82/35.34). seeds 1,2 Core comparison에서 gradroll@12는 WGA로 maskedkd +40.60·rollout +10.90·grid +27.44·center +8.27·random +26.32를 모두 이긴다(CI가 0 밖). 정확도는 maskedkd 대비 WIN이고 나머지는 n.s.다. tome만 미결이다(acc −1.50, WGA +12.78, 둘 다 n.s.). full@196 s2와 maskedkd·rollout·gradroll@12 s3가 실행 중이고, 비교군 s3 5개가 대기 중이다.
- 다음: seed 3이 끝나면 3-seed Core comparison을 판정하고 AUDIT_REQUEST를 쓴다. suffdisc의 final 승격은 gradroll final 3 seed가 끝난 뒤에 정한다. tome 대비 정확도 손실은 AUDIT_REQUEST의 caveat으로 남긴다.
- 막힌 것: 없음. `python -c`와 변수가 든 for 루프는 승인 대기로 막힌다. 그래서 jq·printf와 Read/Grep/Edit로 작업한다.

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

## [claude] 2026-10-30 18:40 라운드 12
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE, RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out. 지난 라운드 뒤로 새 분석 결과는 없다.
- 새 결과: 없다. K=29 seed 1 재현 3개는 학습 중이다. priorroll은 epoch 33, rollout은 epoch 22, center는 epoch 20이다. center의 epoch 20 val WGA 45.86은 중간값이라 판정에 쓰지 않는다.
- 새 분석: analysis/shortcut_inherit.py → analysis_shortcut_inherit_1030_183518_f5cf (H-W15). 지난 라운드에 적은 두 축을 한 번에 잰다. 자세한 것은 PROPOSALS 라운드 12에 있다.
  - (ii) 지름길 상속: val 이미지를 full teacher가 맞힌 것과 틀린 것으로 나눈다. full teacher가 틀린 이미지에서 masked teacher가 고쳐 맞히는 비율(fix_fw)과 참 클래스 확률을 잰다.
  - (i) 위치 다양성: 선택 빈도 엔트로피와 이웃 이미지 Jaccard를 잰다.
  - 대상은 8개 방법 × K 12·29·49의 끝난 screen run 전부다. 각 run은 그 run 자신의 student로 잰다. seed 1 run이 끝나 있으면 재현 점으로만 보고한다.
  - 미리 정한 판정: seed 0 전체에서 Spearman ≥ 0.8이고, 동시에 K=29 순서 rollout > center > priorroll을 맞혀야 한다.
- [claude] 결정: 이번 라운드에는 screen을 넣지 않는다. 세 가지 이유다. 대리 지표가 표본 밖에서 실패했다(H-W13). 새 축의 근거가 아직 없다. seed 1 재현 3개가 GPU를 쓰고 있다. 대기 작업은 분석 1개뿐이다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 18:45 라운드 13
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE, RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out. shortcut_inherit가 새로 나왔다. 새 screen 결과는 없다(K=29 seed 1은 priorroll e61, rollout e50, center e53. 중간값이라 판정에 쓰지 않는다).
- 분석 analysis_shortcut_inherit_1030_183518_f5cf (H-W15): **미리 정한 판정에 실패해 기각한다.**
  - seed 0 run 20개, 부호 보정 Spearman(전체 / K12 / K29 / K49): fix_fw 0.64 / 0.62 / 0.77 / 0.94, ptrue_fw 0.65, ptrue_fr_minor 0.76, kl_worst 0.94 / 0.95 / 0.83 / 0.89. 위치 다양성은 div_ent −0.27, div_jac −0.45다.
  - K=29 순서 rollout > center > priorroll은 어느 지표도 맞히지 못한다. fix_fw는 priorroll 40.51 > rollout 37.97 > center 36.71이다. priorroll의 masked teacher가 full teacher의 오답을 가장 많이 고쳐 맞히는데, student WGA는 가장 낮다(60.15). '지름길 오류를 고치는 teacher 입력'도 K=29 순위를 설명하지 못한다.
  - full teacher 오답은 그룹별로 0/24/47/8장이다. 그래서 fix_fw는 79장 위의 비율이고 잡음이 크다.
- 해석: teacher 쪽 지표 다섯 가지(clean 충실도, self kl, cutmix 충실도, place 탐침, 지름길 상속)가 모두 K=29의 세 방법 순서를 맞히지 못했다. 그런데 그 세 방법의 WGA 차이는 3.0과 2.3이고 seed 1개다(WGA threshold 2.45). 그러니 순서 자체가 잡음일 가능성이 크다. seed 1 재현이 이것을 판정한다. 그전까지 대리 지표 설계를 더 하지 않는다.
- 새 계열 (H-W16): selectors/gradroll.py, parent rollout. Chefer et al. 2021의 relevance 규칙을 쓴다(R ← R + mean_h relu(∂y/∂A ⊙ A) R, y는 student 자신의 예측 클래스 logit).
  - trysel K=29 통과: selector 2.902, teacher 5.36, total 15.726 GFLOPs/img, step 1.12 s. rollout@29의 total 12.99보다 비싸다.
  - screen: screen_gradroll_k29_s0_1030_184255_9017, screen_gradroll_k12_s0_1030_184259_e2cd.
- [claude] 결정: 분석 근거 없이 탐색 screen을 넣었다. 이유는 세 가지다. 대리 지표가 표본 밖에서 신뢰를 잃어 분석으로 미리 거를 수 없다. 지금까지 해 본 것은 attention 순위·공간 사전·출력 sufficiency 계열이고, 클래스 기울기 계열은 아직 없다(규칙 7: 넓게 보기). 대기열이 비어 곧 GPU가 놀게 된다. K는 rollout이 가장 강한 29와 MaskedKD가 가장 약한 12다. 49는 결과를 본 뒤 넣는다.
- [claude] 결정: 이전 설정의 초안 gradattn.py(마지막 블록만 사용, parent maskedkd)는 제출하지 않는다. 마지막 블록 top-K는 teacher에게 place 정보를 가장 많이 넘긴다(H-W7). 그래서 모든 층을 쓰는 gradroll을 먼저 본다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다. 노벨티 메모는 PROPOSALS 라운드 13에 있다(Chefer 규칙 자체는 기존 기법이다. 이기면 분해와 추가 문헌 검색이 필요하다).

## [claude] 2026-10-30 19:00 라운드 14
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(report 재생성), RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out(지난 라운드 뒤 새 분석 없음).
- 새 결과 (K=29 seed 1 재현, 정확도/WGA): rollout 88.99/63.16 (screen_rollout_k29_s1_1030_182725_2086), priorroll 89.41/60.90 (screen_priorroll_k29_s1_1030_182720_dbfd), center 89.07/57.89 (screen_center_k29_s1_1030_182732_85c3). 최악 그룹은 셋 다 그룹 2.
  - 2-seed 평균(RESULTS_TABLE): rollout 89.41±0.59 / 64.28±1.59, priorroll 89.62±0.29 / 60.52±0.53, center 89.03±0.06 / 60.15±3.20.
  - seed 0 순서 rollout 65.41 > center 62.41 > priorroll 60.15, seed 1 순서 rollout 63.16 > priorroll 60.90 > center 57.89.
- 해석: (1) rollout의 K=29 WGA 우위는 두 seed에서 재현된다(priorroll 대비 +5.26, +2.26; center 대비 +3.00, +5.27). 따라서 H-W12 기각(priorroll@29가 parent rollout에 진다)은 유지된다. (2) center와 priorroll의 순서는 seed마다 뒤집힌다. 이 쌍은 잡음 범위다. (3) 라운드 13에 'K=29 순서 자체가 잡음일 수 있다'고 적은 것은 절반만 맞다. self kl_worst(H-W13), cutmix/mixup 충실도(H-W14), fix_fw(H-W15)가 모두 rollout을 priorroll보다 낮게 본 것은 재현된 순위를 틀린 것이다. teacher 충실도 계열 대리 지표의 실패는 실제다. (4) rollout의 정확도는 seed 1에서 88.99로 떨어져, 2-seed 평균 89.41은 Full 90.83 대비 허용 오차 0.75 밖이다.
- 작업: K=12 seed 1 재현 3개 제출: screen_center_k12_s1_1030_185949_1766, screen_suffdisc_k12_s1_1030_185953_9455, screen_priorroll_k12_s1_1030_185957_7cf9 (priority 6).
- [claude] 결정: 새 계열 screen을 더 넣지 않고 K=12 재현을 넣었다. 이유: gradroll 2개가 아직 학습 중이라 새 계열을 쌓지 않는다는 규칙(결과 전에 같은 축을 쌓지 않음)을 지킨다. K=12는 center 계열(suffdisc 58.65, center 56.39, priorroll 54.89)이 threshold 안에서 엇갈리는 K라, seed 1이 있어야 이 계열을 닫을지 정할 수 있다. seed 재현이라 search effort는 늘지 않는다. 대기열이 비면 GPU가 놀게 된다. rollout@12(45.11)는 다른 셋과 차이가 커서 재현하지 않는다.
- 새 분석: 없음. 대리 지표 설계는 라운드 13 결정대로 멈춘다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 19:30 라운드 15
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(report 재생성), RESULTS, PROPOSALS, AUDIT(새 판정 없음), analysis_out(지난 라운드 뒤 새 분석 없음).
- 새 결과 (screen seed 0, 정확도/WGA): gradroll@29 89.49/63.91 (screen_gradroll_k29_s0_1030_184255_9017). 최악 그룹은 그룹 2다.
  - parent rollout@29 seed 0 대비 −0.33/−1.50으로 Parent check UNPROVEN이다. selector 비용은 2.90 GFLOPs/img 더 든다(total 15.73 대 12.99). **승격하지 않는다.**
  - 기울기 가중이 rollout의 K=29 WGA를 올리지 못했다. 'rollout이 배경을 담아서 진다'는 H-W16의 전제는 K=29에서 근거가 없다.
- 학습 중: gradroll@12는 epoch 75이고 val WGA 61.65다. center@12 s1 e89, suffdisc@12 s1 e47, priorroll@12 s1 e2다. 모두 중간값이라 판정에 쓰지 않는다.
- 새 계열 (H-W17): selectors/adaroll.py, parent rollout. 순위는 rollout 그대로 두고, 이미지별 k_i만 rollout 분포의 유효 패치 수 exp(H)에 비례하게 정한다([K/2, 2K], 배치 평균 ≤ K). 근거와 판정 기준은 PROPOSALS 라운드 15에 있다. screen: screen_adaroll_k49_s0_1030_192757_7edc (priority 6).
- [claude] 결정: trysel을 돌리지 못한 채 제출했다. GPU 여유가 0.7GB라 trysel이 시작 단계에서 OOM으로 끝났다(2 MiB 할당 시도, 학습 중 job에는 영향 없음). 라운드 2 규칙(여유 8GB 미만이면 trysel 금지)상 다시 시도하지 않았다. 코드는 rollout(builtin)·priorroll과 같은 rollout 계산을 쓰고, 마스크 행마다 k_i를 다르게 둘 뿐이다. 오류가 나면 run이 rc≠0으로 끝나고 다음 라운드에 고친다. 비용은 rollout과 같은 수준(selector 약 0.17)으로 예상하지만 실측 전까지 쓰지 않는다.
- [claude] 결정: K=49 하나만 넣었다. 이 방법의 표적은 주 지표의 빈칸이다. K=49는 rollout(WGA 안, 정확도 밖)과 center(정확도 안, WGA 밖)가 엇갈리는 곳이다. 결과 전에 K를 늘리지 않는다. gradroll@49는 K=29에서 UNPROVEN이라 넣지 않는다. 대기는 1개이고 학습 4개가 돌고 있어 GPU는 놀지 않는다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다. 적응 토큰 수 선행(A-ViT, AdaViT, DynamicViT)은 기억에 기댄 메모다. adaroll이 이기면 노벨티 주장 전에 'adaptive token budget × distillation teacher input'을 검색한다.

## [claude] 2026-10-30 19:40 라운드 16
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(report 재생성), RESULTS, PROPOSALS, AUDIT(새 판정 없음), analysis_out(지난 라운드 뒤 새 분석 없음).
- 새 결과 (screen, 정확도/WGA):
  - **gradroll@12 seed 0: 88.16/60.90** (screen_gradroll_k12_s0_1030_184259_e2cd, 최악 그룹 2). total 12.81 GFLOPs/img (selector 2.90, teacher 2.45).
    - parent rollout@12(87.24/45.11) 대비 +0.92/+15.79 → Parent check "added part helps".
    - maskedkd@12(85.15/23.31) 대비 +3.01/+37.59, center@12 seed 0(88.74/56.39) 대비 −0.58/+4.51, grid(83.90/35.34)·random(83.49/33.83)보다 둘 다 높다.
    - tome@12(88.99/48.12, teacher 비용 2.69배 not matched, total 12.00) 대비 −0.83/+12.78. 정확도 차이가 threshold 0.76을 조금 넘는다. tome은 total GFLOPs도 더 적다.
    - val WGA 학습 곡선(epoch 75~100): 61.65/60.90/60.90/59.40/60.15/60.90. 마지막 값만 튄 것이 아니다. center@12 seed 0은 같은 구간에서 53.38~57.14.
  - center@12 seed 1: 87.99/51.88 (screen_center_k12_s1_1030_185949_1766). 2-seed 평균 88.37±0.53 / 54.14±3.19.
- 해석: 처음으로 같은 K에서 center와 rollout을 둘 다 WGA threshold 이상 넘은 결과다(seed 1개). K=29(63.91 대 rollout 65.41)에서는 이득이 없었으므로, 기울기 가중은 plain rollout이 무너지는 작은 K에서만 효과가 있다는 해석과 맞는다. 이것이 왜인지는 분해 분석이 답해야 한다.
- 승격 전 분해 (AGENT_PROMPT 규칙):
  - 분석 analysis/gradroll_decomp.py → analysis_gradroll_decomp_1030_193812_07ca (priority 6). gradroll12 student(e25/e50/final), rollout12·center12 final, gradroll29·rollout29 final. 재는 것: gradroll과 maskedkd·rollout·center의 겹침(그룹별), 중심 거리, block-10 고노름 토큰 비율, masked teacher 그룹 정확도·kl_worst, 그리고 대체 변형(gradroll∩maskedkd 유지 + 나머지 random/grid).
  - 대체 screen: selectors/gradrep.py (parent gradroll, gradroll∩maskedkd 유지, 나머지 random) → screen_gradrep_k12_s0_1030_193817_c83d.
- [claude] 결정: gradroll은 **아직 final로 승격하지 않는다.** 이유: (1) 규칙상 분해가 먼저다. (2) tome@12 대비 정확도 −0.83이 threshold를 넘는다. tome@12는 not matched(teacher 비용 2.69배)라 비교가 tome에 유리하다고 지침에 적혀 있지만, total GFLOPs는 tome이 더 적다(12.00 대 12.81). 이것은 AUDIT_REQUEST에 그대로 적을 caveat다. (3) 같은 계열의 K=29 결과와 일관되지 않는다(K=29는 UNPROVEN). 분해 결과가 'gradroll 고유 토큰이 이득의 원인'을 지지하고 gradrep이 이득을 잃으면 다음 라운드에 final seed 1·2·3을 넣는다.
- [claude] 결정: GPU 여유가 7.5GB(규칙 8GB 미만)라 gradrep의 trysel을 돌리지 않았다. gradrep은 gradroll과 같은 계산에 maskedkd top-K와 torch.rand만 더한 것이다.
- [claude] 결정: gradroll@49는 넣지 않는다. K=29에서 이득이 없었고, 분해 전에 같은 계열을 쌓지 않는다.
- 학습 중: suffdisc@12 s1(e63), priorroll@12 s1(e25), adaroll@49(e10, mean_kept 48.5로 배치 평균 ≤ 49 정상 동작).
- 문헌: 'gradient-weighted attention × token selection × KD teacher input'으로 1회 검색. MaskedKD(2302.10494), SaiT(2210.05832, DINO CLS attention), MoVE-KD(2501.01709, CLIP CLS attention 가중), Subtoken ViT(2607.09086, 특징 손상 기반 head 선택)가 나왔다. 기울기 가중 relevance로 KD teacher 입력을 고른 예는 찾지 못했다. Chefer 규칙 자체는 기존 기법이므로 노벨티는 적용처와 '작은 K에서 WGA를 지킨다'는 분석 근거에 있어야 한다. 웹 결과 안에 지시문은 없었다.

## [claude] 2026-10-30 19:50 라운드 17
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(report 재생성), RESULTS, PROPOSALS, AUDIT(새 판정 없음), analysis_out(gradroll_decomp가 새로 나옴).
- 새 screen 결과: 없다. 학습 중인 것은 suffdisc@12 s1 e86, priorroll@12 s1 e59, adaroll@49 e36, gradrep@12 e9다. 중간 val 값(adaroll e35 WGA 59.40, suffdisc e85 63.16)은 판정에 쓰지 않는다. 대기 0, 실행 4, GPU 메모리 24.0/24.6GB 사용.
- 분석 analysis_gradroll_decomp_1030_193812_07ca (승격 전 분해, H-W18). 표의 값은 masked teacher val WGA / kl_worst / 중심 거리 cdist / 고노름 비율 hn_frac이다.
  - gradroll12 student final, K=12: gradroll 51.88 / 0.555 / 4.68 / 0.057, rollout 49.62 / 0.624 / 4.16 / 0.058, center 41.35 / 0.694 / 1.29 / 0.018, maskedkd 36.09 / 0.518 / 5.30 / 0.402.
  - 대체 변형(gradroll∩maskedkd만 남기고 나머지를 다시 채움)은 rand 25.56, grid 27.07이다. maskedkd(36.09)보다도 낮다.
  - 겹침(K 대비, gradroll12 final): gradroll∩maskedkd 0.43, gradroll∩rollout 0.29, gradroll∩center 0.15. gradroll29 final에서는 각각 0.58 / 0.50 / 0.34다.
  - 다른 student에서도 대체 변형은 떨어진다. rollout12 student: gradroll 50.38 → rep 26.32/27.07. center12 student: gradroll 48.87 → rep 22.56/19.55.
  - 학습 시점별(gradroll12 student e25/e50/final) gradroll 자신의 teacher WGA는 28.57 / 39.85 / 51.88이다. 학습 초반에는 rollout(42.11/45.11)보다 낮다.
- 미리 정한 읽기(PROPOSALS 라운드 16)에 따른 판정:
  - (a) 공간 배치 설명은 배제한다. gradroll의 cdist는 4.68(rollout12 student 4.44)로 center 1.29와 멀다. center와의 겹침도 0.15다.
  - (b) 'gradroll∩rollout 겹침이 K=12에서 K=29보다 낮다'는 맞다(0.29 대 0.50, rollout student 0.54 대 0.59).
  - (c) 고노름 회피는 반만 맞다. rollout12 student 안에서는 rollout 0.125 대 gradroll 0.049로 gradroll이 낮다. 하지만 gradroll12 student 안에서는 0.058 대 0.057로 같다. 고노름 토큰 회피로는 이득을 설명할 수 없다.
  - (d) teacher 수준에서는 gradroll 고유(MaskedKD 밖) 토큰 57%가 이득의 원인이다. 그것을 random·grid로 바꾸면 teacher WGA가 random 수준으로 떨어진다. 이 결과는 student 3개에서 모두 같다.
- 주의: 이 분해는 masked teacher(clean eval) 지표다. 이 계열 대리 지표는 표본 밖에서 이미 여러 번 실패했다(H-W8, H-W13~15). 이번 분석에서도 K=29에서 gradroll teacher WGA는 rollout보다 높았다(gradroll29 student 63.16 대 60.15, rollout29 student 66.92 대 54.89). 그런데 student는 이기지 못했다(screen_gradroll_k29_s0_1030_184255_9017 63.91 대 rollout 65.41). 그래서 student 수준의 판정은 대체 screen gradrep@12에 맡긴다.
- [claude] 결정: gradroll@12 final은 이번 라운드에도 제출하지 않는다. 라운드 16에 정한 두 조건 중 분해 쪽은 충족됐다(teacher 수준). gradrep screen 쪽은 epoch 9라 아직 판정할 수 없다. 기다리는 비용은 약 40분이고 기한(11-04)까지 여유가 있다. 반대로 대체 실험 없이 final 3개(+자동 비교군)를 넣으면 GPU를 크게 쓰게 된다. 가장 보수적인 선택이 기다리는 것이다.
- [claude] 결정: 새 screen이나 분석은 넣지 않는다. 학습 4개가 GPU를 꽉 채우고 있다(여유 0.5GB). 승격 후보 판정 전에 새 변형을 쌓지 않는다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 20:00 라운드 18
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(report 재생성), RESULTS, PROPOSALS, AUDIT(새 판정 없음), analysis_out(지난 라운드 뒤 새 분석 없음).
- 새 결과 (K=12 seed 1 재현, 정확도/WGA):
  - **suffdisc@12 s1 89.91/63.91** (screen_suffdisc_k12_s1_1030_185953_9455). 같은 seed의 center(87.99/51.88, screen_center_k12_s1_1030_185949_1766) 대비 +1.92/+12.03. val WGA 마지막 4번(e85~100)은 63.16/63.91/63.16/63.91이라 마지막 값만 튄 것이 아니다.
  - 2-seed 평균(RESULTS_TABLE): suffdisc 89.62±0.41 / 61.28±3.72, center 88.37±0.53 / 54.14±3.19 → +1.25/+7.14로 둘 다 threshold(0.76/2.45)를 넘는다. 두 seed 모두 두 지표에서 suffdisc가 높다. 단 harness Parent check는 seed 0만 보므로 UNPROVEN(+0.58/+2.26) 그대로다.
  - 같은 K의 다른 기준선(seed 0): rollout 45.11, tome 48.12(not matched), grid 35.34, random 33.83, maskedkd 23.31보다 높다. 정확도도 tome@12(88.99)보다 높다.
  - priorroll@12 s1 88.74/53.38 (screen_priorroll_k12_s1_1030_185957_7cf9). 2-seed 88.91/54.14로 center와 같다. **priorroll은 K=12에서도 닫는다.**
- 해석: 라운드 14에 정한 규칙('seed 1에서도 center보다 높지 않으면 center 계열을 닫는다')에 따라 suffdisc는 닫지 않는다. 라운드 11에 'H-W11을 닫는다'고 쓴 것은 seed 0만 본 판정이었다. 다만 분석(analysis_student_sufficiency_1030_173025_6f96)에서 sufficiency 위치는 teacher-best 위치를 약하게만 가리켰다(순위 상관 0.25~0.30). 그래서 이득이 신호가 아니라 '원판을 이미지마다 옮긴 것' 자체에서 올 가능성을 먼저 배제해야 한다(H-W19).
- 작업 (승격 전 분해, 대체 실험):
  - selectors/jitdisc.py (parent suffdisc): 같은 25개 후보(±2 패치) 중 이미지마다 균등 무작위로 원판 위치를 고른다. student forward가 없다. trysel K=12 통과: selector 0.0, teacher 2.446, total 10.088 GFLOPs/img, step 0.95 s (GPU 여유 12GB에서 실행).
  - screen_jitdisc_k12_s0_1030_195912_ead1, screen_jitdisc_k12_s1_1030_195921_48c7 (priority 6).
  - 미리 정한 읽기는 PROPOSALS 라운드 18에 적었다(jitdisc 2-seed WGA ≤ 56.6이면 신호, ≥ 58.8이면 지터).
- [claude] 결정: suffdisc는 아직 승격하지 않는다. 이유: (1) 공식 Parent check(seed 0)가 UNPROVEN이다. (2) 대체 실험 전이다. (3) center@12 자체의 seed 차이가 4.51이라 2-seed 평균도 흔들린다.
- [claude] 결정: jitdisc를 seed 0과 1 둘 다 넣었다. suffdisc의 이득은 대부분 seed 1에서 나왔다(+2.26 대 +12.03). seed 0만으로는 판정할 수 없다. 대조군이라 search effort의 '새 방법'이 아니고, 대기 0·실행 2라 GPU가 빈다.
- [claude] 결정: gradroll@12 final은 이번에도 넣지 않는다. gradrep@12는 epoch 43이다. 라운드 17 결정을 유지한다. 두 후보가 모두 통과하면 final은 gradroll부터 하나씩 넣는다. 승격 후보의 final이 끝나기 전에 다른 변형을 쌓지 말라는 규칙 때문이다.
- 학습 중: gradrep@12 e43, adaroll@49 e69. 중간값은 판정에 쓰지 않는다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다. suffdisc가 승격되면 'sufficiency/insertion faithfulness × KD teacher input'을 검색한다.

## [claude] 2026-10-30 20:20 라운드 19
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(report 재생성), RESULTS, PROPOSALS, AUDIT(새 판정 없음), analysis_out(새 분석 없음).
- 새 결과 (screen seed 0, 정확도/WGA):
  - adaroll@49 90.08/63.91 (screen_adaroll_k49_s0_1030_192757_7edc, mean_kept 48.5, total 16.36 GFLOPs/img). Parent check: rollout 대비 +0.42/−0.75 → UNPROVEN. WGA가 주 지표 조건 64.22보다 0.31 낮다. **H-W17을 닫는다.**
  - gradrep@12 87.99/54.89 (screen_gradrep_k12_s0_1030_193817_c83d). gradroll∩maskedkd만 남기고 나머지를 random으로 바꾼 대체 변형이다. gradroll@12(88.16/60.90) 대비 −0.17/−6.01로, WGA 하락이 threshold 2.45를 넘는다. val WGA 마지막 4번(e85~100)은 56.39/52.63/52.63/54.89다.
- 해석: gradrep는 center@12(56.39)보다 1.50 낮다. 그래서 gradroll이 center를 앞선 몫(+4.51)은 MaskedKD 밖의 gradroll 고유 토큰에서 온다. teacher 수준 분해(analysis_gradroll_decomp_1030_193812_07ca)와 방향이 같다. 그러나 gradrep는 random@12(33.83)나 rollout@12(45.11)까지 떨어지지 않았다. rollout 대비 이득 가운데 약 9.8은 '교집합 토큰 + 무작위 문맥'에도 남는다. 라운드 16에 미리 정한 읽기('random 근처로 떨어지면')는 절반만 맞았다. 남은 몫은 jitdisc(H-W19)와 같은 위치 다양성 효과일 수 있다.
- [claude] 결정: gradroll@12를 final로 승격하되 **seed 1만 먼저** 넣었다. final_gradroll_k12_s1_1030_201811_faf0이고, 비교군 maskedkd·rollout·tome·grid·center·random@12와 full@196은 harness가 자동으로 넣었다(대기 8). 승격 근거: (1) 승격 기준(maskedkd 대비 +37.59, rollout·center·grid보다 높음, parent 대비 +15.79), (2) 승격 전 분해 3가지(겹침·공간 분포·대체 실험)를 마쳤고, 대체 실험에서 center를 앞선 몫이 사라졌다. caveat: tome@12 대비 정확도 −0.83(tome은 not matched), K=29에서는 이득이 없다, 미리 정한 읽기는 부분적으로만 충족됐다. seed 하나만 넣은 이유: seed마다 비교군 7개가 따라와 seed 3개면 24개다. 이는 대기 상한 8의 3배다. 그리고 seed 0에서 center와의 차이(+4.51)가 center 자체의 seed 차이(4.51)와 같은 크기다. seed 1에서 center를 다시 이기는지 본 뒤에 seed 2·3을 넣는 것이 가장 보수적이다.
- [claude] 결정: 새 screen이나 분석은 넣지 않는다. 승격 후보의 final이 진행 중이므로 규칙상 새 변형을 쌓지 않는다. 대기 8개, 실행 2개라 GPU도 놀지 않는다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 20:30 라운드 20
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(report 재생성), RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out(지난 라운드 뒤 새 분석 없음).
- 새 결과 (screen, K=12, 정확도/WGA):
  - jitdisc@12 s0 88.16/52.63 (screen_jitdisc_k12_s0_1030_195912_ead1), s1 88.91/53.38 (screen_jitdisc_k12_s1_1030_195921_48c7). 2-seed 평균은 88.53±0.53 / 53.01±0.53이다.
  - suffdisc 2-seed(89.62/61.28) 대비 −1.09/−8.27, center 2-seed(88.37/54.14) 대비 +0.16/−1.13이다. seed별 WGA 차이(suffdisc − jitdisc)는 s0 +6.02, s1 +10.53으로 두 seed 모두 threshold를 넘는다.
- 해석: 라운드 18에 미리 정한 읽기(≤ 56.6이면 신호)에 해당한다. 같은 25개 후보 위치를 무작위로 고르면 center와 같아진다. 그래서 suffdisc가 center를 앞선 몫은 원판을 옮긴 것 자체가 아니라 student sufficiency로 위치를 고른 데서 온다. H-W19를 기각한다. 부수적으로, 위치 다양성으로 gradrep 잔여 이득을 설명하는 후보도 약해졌다(PROPOSALS 라운드 20).
- 작업: 분석 analysis/suffdisc_decomp.py → analysis_suffdisc_decomp_1030_202612_878c (priority 6). 승격 전 분해에서 남은 겹침·공간 분포를 잰다. 내용은 suffdisc s0 e25/e50/final, s1 final, jitdisc·center s0/s1 final student에서 고른 shift 분포와 중심 비율, maskedkd·rollout·center와의 겹침(그룹별), cdist, train mode(drop path) 픽과 eval 픽의 일치율, 그리고 masked teacher 그룹 정확도와 kl_worst다. 첫 제출은 `torch.manual_seed` 사용으로 validator에 거부됐다. 그 줄을 빼고 다시 냈다. 미리 정한 읽기는 PROPOSALS 라운드 20에 있다.
- [claude] 결정: suffdisc는 이번 라운드에 final로 올리지 않는다. 이유: (1) 공식 Parent check(seed 0)가 UNPROVEN(+0.58/+2.26)이다. 2-seed 근거는 harness 규칙 밖의 보조 근거다. (2) 승격 전 분해 중 겹침·공간 분포가 아직 없다. (3) gradroll final seed 1이 진행 중이다. 이 상태에서 비교군 7개가 따라오는 final을 더 넣으면 대기가 13개로 상한 8을 넘는다. 분해가 끝나고 gradroll final이 판정되면 그때 정한다.
- [claude] 결정: 새 screen은 넣지 않는다. 승격 후보의 final이 진행 중이다(실행 3, 대기 5, GPU 23.1/24.5GB 사용).
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 21:00 라운드 21
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(report 재생성), RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out(suffdisc_decomp가 새로 나옴).
- 새 결과 (final seed 1, K=12, 정확도/WGA):
  - **gradroll 87.99/63.16** (final_gradroll_k12_s1_1030_201811_faf0). val WGA e75~100은 64.66/60.15/60.90/62.41/63.91/63.16이라 마지막 값만 튄 것이 아니다. seed 0 screen(88.16/60.90)과 같은 수준이다.
  - 비교군: maskedkd 84.15/21.05 (final_maskedkd_k12_s1_1030_201811_e49a), rollout 86.99/51.88 (final_rollout_k12_s1_1030_201811_6293), tome 89.66/48.87 (final_tome_k12_s1_1030_201811_9fb4, teacher 비용 2.69배 not matched, total 12.00 GFLOPs/img).
  - gradroll과의 차이: maskedkd 대비 +3.84/+42.11, rollout 대비 +1.00/+11.28, tome 대비 −1.67/+14.29. tome 대비 정확도 손실은 threshold 0.76의 두 배가 넘는다. seed 0(−0.83)에 이어 두 번째다. 그래서 '정확도 또는 WGA에서 이기고 다른 쪽에서 지지 않을 것' 기준으로는 **tome에 대해 주장할 수 없다.** 지침상 K=12의 tome 비교는 tome에 유리하지만(teacher 비용 2.69배), total GFLOPs는 tome이 더 적다(12.00 대 12.81). 이것을 caveat으로 그대로 적는다.
  - center·grid·random final s1은 학습 중, full@196 s1은 대기다. 참고로 같은 seed의 center screen(screen_center_k12_s1_1030_185949_1766)은 87.99/51.88이다. final center 결과가 나오기 전에는 판정에 쓰지 않는다.
- 분석 analysis_suffdisc_decomp_1030_202612_878c (suffdisc 승격 전 분해의 남은 부분). 표는 masked teacher val 정확도/WGA/kl_worst/cdist, student는 eval mode, 입력은 clean val이다.
  - suffdisc student s0 final: suffdisc 84.57/48.12/0.546/2.34, center 76.48/41.35/0.694/1.29, jitter 73.14/37.59/0.824/2.21, maskedkd 72.23/28.57/0.793/4.48. s1 final: suffdisc 83.32/48.12/0.462/2.35.
  - 같은 suffdisc 규칙을 다른 student에서 읽으면 이점이 줄어든다. jitdisc s0/s1 final 42.11/39.10, center s0/s1 final 39.85/45.11로 center(41.35) 근처다. teacher 수준 이점은 suffdisc로 학습한 student에서만 뚜렷하다(되먹임/공적응). midattn(H-W8)과 반대 방향의 되먹임이다.
  - 픽 통계: centre_frac 0.024~0.032(균등 1/25 = 0.04 이하), mean_shift 2.03~2.06, train mode(drop path) 픽과 eval 픽의 일치율 0.34~0.43이다. 다른 student에서도 일치율은 비슷하다(jitdisc 0.40~0.42, center 0.42).
  - 겹침(K 대비): suffdisc∩maskedkd 0.11~0.14, suffdisc∩rollout 0.22~0.24, suffdisc∩center 0.37~0.38이다. 참고로 center∩maskedkd는 0.15~0.58이다(student마다 다름).
- 미리 정한 읽기(PROPOSALS 라운드 20)에 따른 판정: (a) 일치율 0.34~0.43 > 0.15이므로 학습 중 선택은 무작위가 아니다(통과). (b) centre_frac 0.03 ≪ 0.8이므로 suffdisc는 center와 다르다(통과). suffdisc는 거의 항상 원판을 중앙에서 옮기고, MaskedKD 토큰과 거의 겹치지 않는다. 이득은 공간 규칙이나 MaskedKD 토큰에서 오지 않는다. 분해 3가지(겹침·공간·대체 실험 jitdisc)를 모두 마쳤다.
- [claude] 결정: gradroll final seed 2는 **이번 라운드에 넣지 않는다.** 라운드 19에 정한 조건은 'seed 1에서 같은 seed의 center 등을 이기면'이다. center final s1이 아직 학습 중이다. 지금 넣으면 비교군 7개가 따라와 대기가 9개로 상한 8을 넘는다. 실행 4·대기 1이라 GPU는 약 25분 동안 놀지 않는다. 다음 라운드에 center final을 보고 넣는다.
- [claude] 결정: suffdisc final도 넣지 않는다. 분해는 통과했지만 (1) 공식 Parent check(seed 0)가 UNPROVEN이고, (2) 승격한 후보(gradroll)의 final이 끝나기 전에 다른 후보의 final(비교군 7개 포함)을 쌓으면 gradroll 판정이 늦어진다. gradroll final 3 seed가 끝난 뒤 정한다.
- [claude] 결정: 새 screen이나 분석은 넣지 않는다(승격 후보 final 진행 중).
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 21:25 라운드 22
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(report 재생성), RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out(지난 라운드 뒤 새 분석 없음).
- 새 결과 (final seed 1, K=12, 정확도/WGA):
  - center 88.32/54.89 (final_center_k12_s1_1030_201811_341b). val WGA e85~100은 57.89/57.14/56.39/54.89다.
  - grid 83.74/35.34 (final_grid_k12_s1_1030_201811_dc71).
  - gradroll(87.99/63.16, final_gradroll_k12_s1_1030_201811_faf0)과의 차이: center 대비 −0.33/+8.27, grid 대비 +4.25/+27.82. center 대비 정확도 차이는 threshold 0.76 안이고, WGA 차이는 threshold 2.45를 넘는다. seed 0 screen의 center 대비 차이(−0.58/+4.51)와 방향이 같고 더 크다.
  - random final s1은 학습 중이다(e85 중간값은 판정에 쓰지 않는다). full@196 s1도 학습 중이다.
- 판정: 라운드 19·21에 정한 조건('seed 1에서 같은 seed의 center를 WGA threshold 이상 넘으면 seed 2')을 충족한다. 같은 seed 비교군 6개 중 tome을 뺀 5개(maskedkd·rollout·grid·center, random은 대기)를 이긴다. tome에는 정확도에서 −1.67로 지는 caveat이 남는다.
- 작업: gradroll@12 final seed 2를 제출했다(final_gradroll_k12_s2_1030_212126_da50). 비교군 maskedkd·rollout·tome·grid·center·random@12와 full@196(s2)은 harness가 자동으로 넣었다. 대기는 8개로 상한 안이다.
- [claude] 결정: seed 3은 이번 라운드에 넣지 않는다. 넣으면 대기가 16개로 상한 8의 두 배가 된다. seed 2가 돌기 시작해 대기가 줄면 넣는다.
- [claude] 결정: 새 screen이나 분석은 넣지 않는다. 승격 후보의 final이 진행 중이라 규칙상 새 변형을 쌓지 않는다. suffdisc final은 라운드 21 결정대로 보류한다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 21:30 라운드 23
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE, RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out(지난 라운드 뒤 새 분석 없음).
- 새 결과 (final seed 1, K=12): random 84.57/37.59 (final_random_k12_s1_1030_201811_19bf, 최악 그룹 2). gradroll(87.99/63.16)과의 차이는 +3.42/+25.57이다. 이로써 seed 1의 K=12 비교군 6개가 모두 끝났다. gradroll은 tome을 뺀 5개(maskedkd·rollout·grid·center·random)를 정확도 손실 threshold 안에서 WGA로 이긴다. tome에는 정확도 −1.67 caveat이 남는다. full@196 s1은 e70이다(val 90.83/57.89, 중간값이라 판정에 쓰지 않음).
- 진행: gradroll s2는 e15(중간값은 쓰지 않음), maskedkd·rollout s2 실행, 대기 5.
- [claude] 결정: `python -m harness.report`와 `--status`가 이번 세션에서 승인 대기로 막혔다. 그래서 RESULTS_TABLE은 재생성하지 않았다(random s1 행은 다음 harness 재생성 때 들어간다). 숫자는 runs/*/metrics.json에서 직접 읽었다.
- [claude] 결정: gradroll seed 3은 넣지 않는다. 지금 넣으면 비교군 7개가 따라와 대기가 5→13으로 상한 8을 넘는다. seed 2 비교군 대기가 3개 이하로 줄면 넣는다.
- [claude] 결정: 새 screen·분석은 넣지 않는다. 승격 후보 final이 진행 중이다. suffdisc final은 라운드 21 결정대로 보류한다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 21:45 라운드 24
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(이번에는 `harness.report` 재생성 성공), RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out(지난 라운드 뒤 새 분석 없음).
- 새 결과 (final seed 1): full@196 91.33/66.92 (final_full_k196_s1_1030_201811_b764, 최악 그룹 2). RESULTS_TABLE의 final Primary metric 기준선은 이제 Full 91.33, 허용 오차 1.00(final tier에는 CE가 없어 상한 1.00 적용)/WGA 2.45다.
- 해석: gradroll@12 s1(87.99/63.16, final_gradroll_k12_s1_1030_201811_faf0)은 Full s1보다 −3.34/−3.76이다. 정확도와 WGA가 모두 허용 오차 밖이다. 그래서 gradroll@12는 주 지표(Full 허용 오차 안의 최소 K)를 개선하지 못한다. 주장의 형태는 'K=12에서 같은 seed 기준선 5개(maskedkd·rollout·grid·center·random)보다 WGA가 높고 정확도 손실은 threshold 안'으로 한정한다. 그리고 tome 대비 정확도 −1.67이라는 caveat이 남는다. 이 한정을 AUDIT_REQUEST를 쓸 때 첫 줄에 적는다.
- 진행: gradroll s2는 e52다(중간값은 판정에 쓰지 않음). maskedkd·rollout·tome s2는 실행 중, 대기 4(center·grid·random s2, full s2)다. GPU 메모리 23.2/24.6 GiB를 쓰고 있다.
- [claude] 결정: gradroll seed 3은 이번에도 넣지 않는다. 지금 넣으면 비교군 7개와 본체가 따라와 대기가 4→12로 상한 8을 넘는다. 라운드 23의 조건(seed 2 비교군 대기 ≤3)을 그대로 쓴다.
- [claude] 결정: 새 screen·분석은 넣지 않는다. 승격 후보 final이 진행 중이다. suffdisc final은 라운드 21 결정대로 보류한다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 21:56 라운드 25
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(`harness.report` 재생성 성공), RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out(지난 라운드 뒤 새 분석 없음).
- 새 결과 (final seed 2, K=12): maskedkd 84.24/23.31 (final_maskedkd_k12_s2_1030_212126_ad00, 최악 그룹 2). 2-seed 평균은 84.20±0.06 / 22.18±1.60이다. seed 1(84.15/21.05)과 거의 같다. K=12에서 MaskedKD의 낮은 WGA는 seed가 바뀌어도 그대로다.
- 진행: gradroll s2는 e79, rollout s2는 e88, tome s2는 e47, grid s2는 e8이다(중간값은 판정에 쓰지 않음). 대기 3(center·random@12 s2, full@196 s2).
- [claude] 결정 (라운드 23·24 조건 정정): 'seed 2 비교군 대기 ≤3이면 seed 3 제출' 조건은 계산이 틀렸다. `harness.submit --tier final`은 본체 1개와 refs 7개(maskedkd·rollout·tome·grid·center·random@12, full@196. parent rollout은 중복이라 skip)를 함께 넣는다. 대기 3에서 제출하면 11개가 되어 상한 8을 넘는다. 그래서 조건을 '대기 0'으로 바꾼다. 지금은 대기 3개와 실행 4개로 GPU가 차 있어서, 기다려도 GPU는 놀지 않는다.
- [claude] 결정: 새 screen·분석은 넣지 않는다. 승격 후보 final이 진행 중이다. suffdisc final은 라운드 21 결정대로 보류한다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 22:10 라운드 26
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(`harness.report` 재생성 성공), RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out(지난 라운드 뒤 새 분석 없음).
- 새 결과 (final seed 2, K=12, 정확도/WGA):
  - rollout 87.99/51.88 (final_rollout_k12_s2_1030_212126_33d1). 2-seed 87.49±0.71 / 51.88±0.00이다.
  - gradroll 88.41/62.41 (final_gradroll_k12_s2_1030_212126_da50, 최악 그룹 2). val WGA e85~100은 59.40/60.90/62.41/62.41이라 마지막 값만 튄 것이 아니다. 2-seed 평균은 88.20/62.79다.
- RESULTS_TABLE Core comparison (seeds 1,2, paired 95% CI, GFLOPs 비율 1.27):
  - maskedkd 대비: 정확도 +4.00 [+1.88, +6.12], WGA +40.60 [+21.48, +59.73]. 둘 다 WIN이다.
  - rollout 대비: WGA +10.90 [+6.14, +15.67] WIN, 정확도 +0.71 [−3.00, +4.42]는 차이 없음이다.
  - tome·grid·center·random은 s2가 아직 없어서 missing이다.
  - seed 2에서 rollout 대비 차이(+0.42/+10.53)는 seed 1(+1.00/+11.28)과 방향과 크기가 같다.
- 판정: gradroll@12의 우위는 seed 2에서도 재현된다. 다만 주장 조건(6개 기준선 모두)을 판정하려면 center·tome s2가 필요하다. Full 허용 오차 밖이라는 한정(라운드 24)은 그대로다.
- [claude] 결정: gradroll seed 3은 넣지 않는다. 대기가 1이라 라운드 25 조건(대기 0)을 채우지 못한다. 지금 넣으면 1+8=9개로 상한 8을 넘는다. 실행 4개로 GPU는 차 있다.
- [claude] 결정: 새 screen·분석은 넣지 않는다. 승격 후보 final이 진행 중이다. suffdisc final은 라운드 21 결정대로 보류한다. AUDIT_REQUEST는 seed 3과 비교군 s2가 끝난 뒤에 쓴다. 2 seed 중 1 seed만 6개 비교군을 모두 갖추고 있기 때문이다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 22:12 라운드 27
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(`harness.report` 재생성 성공), RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out(지난 라운드 뒤 새 분석 없음).
- 새 결과 (final seed 2, K=12): tome 89.74/51.13 (final_tome_k12_s2_1030_212126_9500, not matched, 최악 그룹 2). gradroll s2(88.41/62.41) 대비 −1.33/+11.28이다. seed 1(−1.67/+14.29)과 방향이 같다.
- RESULTS_TABLE Core comparison (seeds 1,2): gradroll vs tome은 정확도 −1.50 [−3.62, +0.62], WGA +12.78 [−6.34, +31.91]로 둘 다 차이 없음이다. maskedkd 대비(정확도·WGA WIN)와 rollout 대비(WGA WIN)는 라운드 26과 같다. grid·center·random은 s2가 실행 중이라 missing이다.
- 해석: seed별로 보면 tome 대비 정확도 손실은 seed 0·1·2 모두 threshold 0.76을 넘는다. 2-seed CI로는 유의하지 않다. 그래도 주장에서 'tome에 정확도로 지지 않음'이라고 쓰지 않고 caveat으로 남긴다. K=12의 ToMe는 예산에 맞춰지지 않은 비교라는 점(GFLOPs 비율 1.068)도 함께 적는다.
- 작업: 대기가 0이 되어 라운드 25 조건을 채웠다. gradroll@12 final seed 3을 제출했다(final_gradroll_k12_s3_1030_220958_176b). 비교군 7개(maskedkd·rollout·tome·grid·center·random@12, full@196, final_*_s3_1030_220958_*)가 함께 들어가 대기가 8이다(상한 안). parent rollout은 중복이라 skip됐다.
- [claude] 결정: 새 screen·분석은 넣지 않는다. 대기가 상한 8이고, 승격 후보의 final이 진행 중이다. suffdisc final은 라운드 21 결정대로 gradroll 3 seed가 끝난 뒤 정한다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 22:22 라운드 28
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(`harness.report` 재생성 성공), RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out(지난 라운드 뒤 새 분석 없음).
- 새 결과 (final seed 2, K=12): grid 83.40/35.34 (final_grid_k12_s2_1030_212126_9336, 최악 그룹 2). gradroll s2(88.41/62.41) 대비 +5.01/+27.07이다. seed 1(+4.25/+27.82)과 거의 같다.
- RESULTS_TABLE Core comparison (seeds 1,2): grid 대비 WGA +27.44 [+22.68, +32.21] WIN이다. 정확도 +4.63 [−0.14, +9.40]는 차이 없음이다(CI 하한이 0에 가깝다). 그 밖의 비교는 라운드 27과 같다. center·random·full은 s2가 실행 중이라 missing이다.
- 진행: center s2 e81, random s2 e65, full s2 e28, gradroll s3 e3이다(중간값은 판정에 쓰지 않음). 대기 7(비교군 s3)이다.
- [claude] 결정: 새 screen·분석은 넣지 않는다. 대기 7개와 실행 4개로 GPU가 차 있다. 승격 후보의 final이 진행 중이다. AUDIT_REQUEST는 라운드 26 결정대로 비교군 s2와 seed 3이 끝난 뒤에 쓴다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 22:35 라운드 29
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(`harness.report` 재생성 성공), RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out(지난 라운드 뒤 새 분석 없음).
- 새 결과 (final seed 2, K=12, 정확도/WGA):
  - center 89.16/54.14 (final_center_k12_s2_1030_212126_5486, 최악 그룹 2). gradroll s2(88.41/62.41) 대비 −0.75/+8.27이다. seed 1(−0.33/+8.27)과 WGA 차이가 똑같다.
  - random 82.82/35.34 (final_random_k12_s2_1030_212126_b5bc). gradroll s2 대비 +5.59/+27.07이다.
- RESULTS_TABLE Core comparison (seeds 1,2):
  - center 대비 WGA +8.27 [+8.27, +8.27] WIN, 정확도 −0.54 [−3.19, +2.11] 차이 없음.
  - random 대비 WGA +26.32 [+16.79, +35.85] WIN, 정확도 +4.50 [−9.27, +18.28] 차이 없음.
  - 이제 6개 비교군 중 tome을 뺀 5개(maskedkd·rollout·grid·center·random)를 WGA로 이긴다. 정확도로 유의하게 지는 비교군은 없다.
- 해석: center 대비 WGA CI 폭이 0인 것은 두 seed의 차이가 우연히 같아서다(분산 0). seed 2개짜리 CI라 과신하지 않는다. center 대비 정확도 손실은 seed 2에서 −0.75로 threshold 0.76의 경계에 있다. tome 대비 비교는 여전히 미결이다(정확도 −1.50, WGA +12.78, 둘 다 n.s.).
- 진행: full@196 s2, maskedkd·rollout·gradroll@12 s3가 실행 중이다. center·grid·full·tome·random s3 5개가 대기 중이다.
- [claude] 결정: AUDIT_REQUEST는 아직 쓰지 않는다. tome 비교가 미결이고 center 대비 CI가 퇴화했기 때문이다. seed 3이 끝난 뒤 3-seed 결과로 쓰는 것이 보수적이다.
- [claude] 결정: 새 screen·분석은 넣지 않는다. 실행 4개와 대기 5개로 GPU가 차 있다(대기는 harness가 넣은 비교군). 승격 후보의 final이 진행 중이다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.

## [claude] 2026-10-30 22:45 라운드 30
- 읽은 것: QUEUE(지시 없음), git log -15, PROGRESS, RESULTS_TABLE(`harness.report` 재생성 성공), RESULTS, PROPOSALS, AUDIT(비어 있음), analysis_out(지난 라운드 뒤 새 분석 없음).
- 새 결과 (final seed 2): full@196 90.58/67.67 (final_full_k196_s2_1030_212126_a1dc, 최악 그룹 2). seed 1(91.33/66.92)과 합친 final Full 2-seed는 90.95/67.30이다(RESULTS_TABLE Primary metric).
- RESULTS_TABLE Core comparison (seeds 1,2): gradroll@12 vs full@196은 정확도 −2.75 [−10.17, +4.67], WGA −4.51 [−14.04, +5.02]로 둘 다 차이 없음이다(GFLOPs 비율 0.300). seed 2 단독 차이는 −2.17/−5.26이다. 평균 차이가 허용 오차(정확도 1.00, WGA 2.45) 밖이다. CI가 넓은 것은 seed가 2개뿐이어서다. 그래서 라운드 24의 한정('같은 K에서 기준선 대비 WGA 우위'만 주장하고 Full 허용 오차 안이라고 주장하지 않음)을 유지한다. 나머지 비교는 라운드 29와 같다. 5개 기준선에는 WGA로 이기고, tome과의 비교는 미결이다.
- 진행: maskedkd·rollout·gradroll·tome@12 s3이 실행 중이고, center·grid·random@12와 full@196 s3이 대기 중이다(4개).
- [claude] 결정: 새 screen·분석은 넣지 않는다. 승격 후보의 final(seed 3)과 비교군이 진행 중이다. 지침 7의 'final이 끝나기 전에는 새 변형을 쌓지 마라'를 따른다. 대기 4개와 실행 4개로 GPU도 차 있다. AUDIT_REQUEST는 라운드 29 결정대로 3-seed 결과가 나온 뒤에 쓴다.
- 문헌: 이번 라운드에는 새로 검색하지 않았다.
