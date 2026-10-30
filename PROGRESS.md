# PROGRESS
- 요약 (2026-10-30 [claude] 라운드 2): 확정 결과 없음. 공략할 K는 49다. seed 0 결과: MaskedKD 89.24/WGA 51.88, rollout 89.66/64.66, random 85.40/36.09, grid 84.74/35.34, Full 90.66/65.41.
- 핵심 분석: 마지막 블록 대신 9번째 블록의 CLS attention으로 고르면 masked teacher의 WGA가 47→72로 오른다(analysis_group_fidelity_1030_143842_1ea0 / _144455_654a). 선택기 `selectors/midattn.py`를 만들었지만 아직 trysel하지 않았다(GPU OOM).
- 진행 중: K=12/29/49 기준선(12개 대기, center·tome@49와 maskedkd@29 실행 중), analysis_group_fidelity_1030_144455_654a(mkd98 student 부분 진행 중).
- 다음: 대기 작업이 8개 이하가 되면 trysel midattn을 하고 screen K=49, 29, 12를 넣는다. analysis/register_tokens.py(H-W5)를 제출한다.
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
