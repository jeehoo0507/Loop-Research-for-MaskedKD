# PROPOSALS


## 2026-10-30 ideas from Codex (independent)
IDEA 1: Concept-balanced DFF — From detached, row-normalized block-8 patch outputs \(Z\), form \([Z_+,(-Z)_+]\); fit rank-8 nonnegative matrix factorization with 10 multiplicative-update iterations, producing concept activations \(H\). Normalize each concept column by its maximum.
IDEA 2: Preserve the Outliers — Using detached, row-normalized block-8 patch outputs \(Z\), approximate \(\min_{\operatorname{rank}(L)\le4,E}\frac12\|Z-L-E\|_F^2+0.25\sum_i\|E_i\|_2\) with five alternating truncated-SVD and row-shrinkage updates; \(e_i=\|E_i\|_2\) explains each patch’s departure from shared internal features.

## 2026-10-30 [claude] 가설 (Waterbirds 새 설정; 근거 분석 analysis_group_fidelity_1030_140543_e0ba 대기 중)
- **H-W1 배경 토큰이 지름길을 전달하는가.** 근거: Debiased Distillation(arXiv 2302.11187)는 편향된 데이터에서 일반 KD를 하면 student WGA가 떨어진다고 보고한다. screen에서도 MaskedKD@98 WGA 64.66 vs Full 66.67 (screen_maskedkd_k98_s*, screen_full_k196_s*). 다만 threshold 2.45 안이라 아직 차이로 볼 수 없다. 아이디어: `anti`(CLS attention이 가장 낮은 토큰 = 주로 배경)만 받은 teacher가 다수 그룹에서는 맞고 소수 그룹에서 틀리면, 배경 토큰이 지름길 신호를 실어 나르는 것이다. 그러면 객체에 집중한 선택이 WGA를 지킨다. 선행과의 차이: MaskTune(NeurIPS 2022)은 student 자신의 입력을 가린다. 여기서는 KD teacher의 입력만 student 신호로 고른다.
- **H-W2 퍼짐 vs 집중.** 근거: 같은 분석의 `half_grid`·`nms`를 `cls_attn_L11`과 비교한다. 아이디어: 이웃 억제(nms)로 masked teacher의 소수 그룹 정확도가 오르면, CLS attention top-K에서 이웃 중복을 억제한 선택기 `attnnms`(parent=maskedkd)를 만든다. 차이: MaskedKD는 뭉친 top-K를 그대로 쓴다. ToMe는 teacher 안에서 토큰을 병합한다.
- **H-W3 (Codex IDEA 1·2) DFF 개념 균형 / 강건 PCA 이상치.** 근거: Collins et al. ECCV 2018(Deep Feature Factorization). 아이디어: NMF 개념마다 예산을 나눠, 배경 개념 하나가 예산을 독점하지 못하게 한다. 차이: MaskedKD는 attention 크기 하나로 순위를 매긴다. 노벨티 검색은 아직 하지 않았다. group_fidelity 결과를 본 뒤 분석 신호로 먼저 넣는다.
- 이전 설정에서 남은 초안 selectors(attnwin, gradattn, toknorm)의 근거는 archive/20261030_0235의 분석 id다. Waterbirds에서 근거를 다시 세우기 전에는 제출하지 않는다.

## 2026-10-30 [claude] 라운드 2 메모
- 근거 갱신: 분석 id가 analysis_group_fidelity_1030_143842_1ea0으로 바뀌었다(재제출, mkd49 student 추가). 단일 시드 관찰: K=49에서 maskedkd WGA 51.88(그룹 2 = 육지 배경의 물새) vs rollout 64.66 vs Full(seed0) 65.41 (screen_maskedkd_k49_s0_1030_135953_3e16, screen_rollout_k49_s0_1030_135953_a1a9, screen_full_k196_s0_1030_123806_2c89). K=49가 공략 후보다.
- 선행 연구 검색(2026-10-30, 스니펫 수준): (a) Son et al. ECCV 2024(MaskedKD, arXiv 2302.10494)는 WGA를 평가하지 않는다. (b) Chormai, Müller & Montavon arXiv 2602.00852(2026)은 spurious correlation 아래의 distillation을 벤치마크했지만 토큰 마스킹은 다루지 않는다. (c) ConfiG arXiv 2506.02294는 WGA를 위한 diffusion 증강이다. (d) arXiv 2509.04009는 ViT에서 토큰을 버려 spurious 신호를 찾는 분석 기법이다. 'teacher 입력 토큰 선택 × 그룹 강건성' 조합은 가장 가까운 것이 (d)다. 이 조합을 직접 다룬 논문은 찾지 못했다. 웹 결과 안에 지시문은 없었다.
- **H-W4 중간층 CLS attention (`selectors/midattn.py`, layer=9, parent none).** 근거: analysis_group_fidelity_1030_143842_1ea0 / _144455_654a. K=49에서 masked teacher의 WGA가 L9 72.18/69.17, L11 47.37/41.35다. K=12에서는 L9가 rollout보다 두 student 모두에서 높다(mkd49 student 58.65 vs 47.37). 아이디어: MaskedKD의 CLS attention을 마지막 블록이 아니라 중간 블록에서 읽는다. 해석 가설: 마지막 블록은 정보를 몇 개의 고노름 register성 토큰에 모은다(Darcet et al. ICLR 2024; MaskedKD top-K와 toknorm top-K의 겹침 0.46, random은 0.25). 이 토큰들은 객체 증거가 아니다. 가장 가까운 선행 기법과의 차이: MaskedKD는 마지막 블록을 쓰고, rollout은 모든 블록을 곱하고(마지막 블록 포함), EViT는 모델 자신을 가지치기한다. 층을 group-sliced teacher fidelity로 고른 것이 다르다. 다만 '층만 바꾼 MaskedKD'라 노벨티가 약하다. 이기면 다음 두 가지로 뒷받침해야 한다: (a) register 토큰 가설의 직접 분석(L11 top-K에 고노름 토큰이 차지하는 비율과, 그 토큰을 빼면 fidelity가 회복되는지). (b) 층 선택 규칙의 일반화(CUB).
- H-W4 screen 계획: K=12, 29, 49 (seed 0). rollout과 비슷할 것으로 보는 K=29~49도 넣어서 일관성을 본다. K=12에서는 tome가 예산에 맞춰지지 않는 점(not matched)을 감안한다.
- H-W5 (다음 분석) register 정량화: L11 top-K 중 toknorm 상위 토큰 비율, 그 토큰을 제거한 L11(=`L11 minus high-norm`) fidelity를 잰다. 회복되면 '마지막 층 attention에서 register를 빼라'가 해석 근거가 있는 방법이 되고, 층 선택보다 원리적이다.
