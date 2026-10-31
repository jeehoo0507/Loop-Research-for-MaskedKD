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

## 2026-10-30 [claude] 라운드 3 메모
- center@49(90.16/60.15, screen_center_k49_s0_1030_135953_4476)가 MaskedKD@49(89.24/51.88)보다 강하다. H-W4가 center와 비슷하게 나오면 "L9는 가운데 쪽으로 모인 것뿐"이라는 설명을 배제할 수 없다. L9의 cdist는 4.01, center는 2.64, L11은 4.67이다(analysis_group_fidelity_1030_143842_1ea0). 분해 계획: midattn 선택 중 MaskedKD와 겹치지 않는 토큰을 center 순위 토큰으로 바꾼 변형(parent=midattn)을 넣는다.
- H-W6 (H-W5 결과에 따라): L11 CLS attention에서 block-10 고노름 토큰을 빼고 다시 채우는 선택기 `l11nohn`(parent=maskedkd). 근거: analysis_register_tokens_1030_153157_efbe의 `L11_nohn@49` t_wga가 L11보다 크게 오르면 넣는다. 선행과의 차이: Darcet et al.은 register 토큰을 더해 학습을 바꾼다. 여기서는 KD teacher 입력을 고를 때 그 토큰을 진단해서 뺀다.

## 2026-10-30 [claude] 라운드 4 메모
- H-W5 결과(analysis_register_tokens_1030_153157_efbe): MaskedKD top-K에는 고노름 토큰이 많다(K=49에서 0.18, 기저율 0.05). 하지만 그 토큰을 빼도 masked teacher WGA는 그대로다(48.87→49.62). **H-W6(l11nohn)은 기각한다.** H-W4(midattn)의 해석은 'register 회피'에서 '내용(객체 대 배경)'으로 바꿔 검증해야 한다.
- **H-W7 L11 top-K는 teacher에게 장소(배경) 정보를 넘기는가.** 근거: anti(하위 K) WGA 3.01(analysis_group_fidelity_1030_143842_1ea0), H-W5의 음성 결과. 아이디어: class 안에서 place를 맞히는 선형 탐침(Alain & Bengio 2017)을 masked teacher의 feature에 붙이고 선택기별로 비교한다(analysis_shortcut_probe_1030_160112_bfc0). L11 > L9·rollout이면, 선택기 설계 원리를 '장소 정보가 적은 토큰'으로 세울 수 있다. 다만 그룹 라벨이 필요 없는 student 내부 대리 지표여야 한다. 후보는 student 중간층 CLS attention과 마지막 층 CLS attention의 차이(L11 − L9)다. 마지막 층에서 새로 attention을 받는 토큰이 지름길 토큰이라는 가설이다. 선행과의 차이: MaskTune과 Debiased Distillation은 student 입력이나 손실을 바꾼다. 이 방법은 KD teacher 입력 선택으로 지름길을 줄이면서 teacher 비용도 함께 줄인다.
- 경계: rollout@29(89.82/65.41)는 이미 Full KD의 WGA와 같다. midattn이 이기려면 정확도에서 이기거나 더 작은 K(12)에서 차이를 내야 한다. 분석 대리 지표로는 K=12에서 L9 > rollout이 5개 student 모두에서 성립했다(analysis_group_fidelity_1030_144455_654a).

## 2026-10-30 [claude] 라운드 5 메모
- H-W7 결과(analysis_shortcut_probe_1030_160112_bfc0): masked teacher feature의 place 탐침 균형 정확도는 L11이 가장 높고 L9가 가장 낮다(K=49, full student 기준 86.10 vs 71.35). masked teacher WGA와는 거의 역순이다. teacher 수준에서는 지지된다. 그러나 midattn@49 student WGA는 57.89로 rollout 64.66과 center 60.15보다 낮다(screen_midattn_k49_s0_1030_153152_3b79). "teacher가 받는 place 정보가 적을수록 student WGA가 좋다"는 연결은 아직 서지 않았다.
- **H-W8 학습 조건 차이가 대리 지표를 틀리게 한다.** 근거: 위의 불일치, 그리고 midattn의 val WGA가 epoch 45의 70.68에서 후반 57~59로 떨어진 것(같은 run의 metrics.json). 아이디어: 학습 입력(cutmix/mixup)과 train 모드 student에서 그룹별 teacher fidelity를 다시 잰다(analysis_train_condition_gap_1030_162956_067e). L9만 cutmix 상자 비율이 치우치거나 drop path에서 불안정하면, 그것이 원인이다. 그러면 선택기 설계에 '혼합 입력에서도 두 이미지의 객체 증거를 모두 남기는 선택'이 필요하다. rollout은 모든 층을 곱해서 이 점에서 더 안정적일 수 있다. 선행과의 차이: TransMix(CVPR 2022)는 attention으로 mixup 라벨을 다시 매기는 기법이고 student 학습 쪽이다. 여기서는 혼합 입력에서 KD teacher 입력을 고르는 문제다.
- 다음 후보 (H-W8 결과에 따라): (a) L9와 rollout의 순위 합(parent=midattn 또는 rollout). (b) 학습 진행에 따라 층을 바꾸는 방식: 초반에는 L9, 후반에는 rollout(parent=midattn). midattn WGA가 후반에 떨어진 것에 근거한다. 이런 혼합은 구성요소를 쌓는 것이므로 parent check를 통과해야 한다.

## 2026-10-30 [claude] 라운드 6 메모
- **H-W9 student attention으로 위치만 정한 연속 창 (`selectors/attnwin.py`, parent=center).**
  - 근거: K=12 screen seed 0에서 center 88.74/56.39 > rollout 87.24/45.11 > random 83.49/33.83 > maskedkd 85.15/23.31이다(WGA 순, screen_*_k12_s0_1030_135953_*). K가 작을수록 연속 영역이 흩어진 top-K보다 유리하다.
  - 아이디어: attention map(L9 또는 rollout)을 순위 매기기가 아니라 위치 추정(무게중심 또는 blur 뒤 최댓값)에만 쓴다. 그 주변 K개를 원판 모양으로 teacher에게 준다. off-centre crop 조건의 masked teacher WGA가 center보다 높을 때만 제출한다(analysis_window_fidelity_1030_165418_e276).
  - 가장 가까운 선행 기법과의 차이: TS-CAM·WS-DAN·TransFG는 attention으로 모델 자신의 입력을 자르거나 부분을 고른다. 이 방법은 student attention으로 KD teacher의 입력 영역을 정해 teacher 비용을 줄인다. MaskedKD는 흩어진 top-K를 쓴다. 'attention-guided crop for distillation teacher'에 대한 노벨티 검색이 아직 필요하다.
- 주의: attnwin이 center를 이겨도, 그 이득이 '창을 옮긴 것' 자체에서 오는지 확인해야 한다. 분석의 shift_best(상한)와 비교한다. 분해 단계에서는 무작위로 옮긴 원판(shift_rand) 변형을 대체 실험으로 넣는다.

## 2026-10-30 [claude] 라운드 7 메모
- H-W8 판정 (analysis_train_condition_gap_1030_162956_067e): 학습 조건(mixup/cutmix, drop path) 가설은 기각한다.
  - 원인은 '선택기→student attention' 되먹임이다. midattn student 자신의 L9@49 masked teacher WGA는 66.92→59.40→60.90으로 떨어진다. rollout student에서 읽은 L9는 75.19다.
  - self proxy로 재면 K=49 student 순위가 맞는다.
- H-W9 판정 (analysis_window_fidelity_1030_165418_e276): 기각한다. 창은 center를 일관되게 이기지 못하고 흩어진 top_L9보다 낮다. attnwin은 제출하지 않는다.
- **H-W10 어떤 대리 지표가 student WGA를 예측하는가 (analysis_proxy_validation_1030_170914_4668).**
  - 근거: K=12에서 clean masked-teacher WGA는 center를 rollout보다 낮게 보지만 student는 반대다(center@12 56.39 vs rollout@12 45.11).
  - 아이디어: 참 클래스 확률, 확신 오답, 엔트로피를 self/common student로 재서 16개 run과 순위 상관을 낸다.
  - 작업 가설: 작은 K의 center는 teacher를 '덜 확신하게' 만든다. KD 목표가 틀릴 때도 부드러워서 student가 지름길 쪽으로 밀리지 않는다. 그렇다면 cwrong_worst나 ent_minor가 t_acc_worst보다 순위를 잘 맞힐 것이다.
  - 선행과의 차이: ROAR(Hooker et al. 2019)는 지운 뒤 다시 학습해 attribution을 평가한다. 여기서는 KD teacher 입력 선택을 그룹별 calibration으로 평가하고, 그것을 학습된 student와 대조해 검증한다.
- 다음 후보 (H-W10 결과에 따라):
  - (a) 확신 오답이 원인이면, 샘플마다 다른 K를 쓰는 선택기를 만든다. student 자신의 attention 집중도(엔트로피)로 '객체가 불확실한 이미지'를 판단해 그런 이미지에 더 많은 토큰을 준다. 배치 평균은 예산 이하로 유지한다.
  - (b) 되먹임이 원인이면, 학습으로 덜 바뀌는 신호를 쓴다. rollout은 self proxy에서 안정적이었다. 또는 ctx.progress에 따라 신호를 섞는다.
  - 두 경우 모두 parent check가 필요하다.

## 2026-10-30 [claude] 라운드 8 메모
- H-W10 판정 (analysis_proxy_validation_1030_170914_4668): 확신 오답이나 엔트로피가 아니라 **KL(full teacher‖masked teacher)의 최악 그룹 값**이 student WGA를 가장 잘 맞힌다 (self, |ρ| 0.937, K=12 1.0). 작업 가설 (a)는 근거가 약해서 보류한다.
- **H-W11 student sufficiency로 정한 원판 위치 (suffdisc, parent center).**
  - 근거: analysis_window_fidelity_1030_165418_e276에서 원판을 ≤2 패치 옮겨 teacher KL이 가장 작은 곳에 두면 K=12 kl_worst가 0.694에서 0.049로 줄어든다 (clean). 이것은 teacher를 쓰는 분석 전용 상한이다. attention 무게중심은 그 위치를 찾지 못했다.
  - 아이디어: 25개 위치 중 student 자신의 예측이 가장 잘 보존되는 원판(KL(student full‖student masked)이 가장 작은 것)을 teacher에게 준다. label도 teacher도 쓰지 않는다. 검증 분석은 analysis_student_sufficiency_1030_173025_6f96이다. student KL과 teacher KL의 이미지별 순위 상관, teacher-best 적중률, 그룹별 teacher KL을 잰다.
  - 선행과의 차이: ERASER sufficiency(DeYoung et al. 2020)와 RISE insertion(Petsiuk et al. 2018)은 주어진 마스크를 설명 평가용으로 점수 매긴다. 여기서는 student 쪽의 sufficiency로 teacher 입력 창을 고른다. center와 달리 이미지마다 위치가 바뀐다. MaskedKD·rollout과 달리 attention 순위를 쓰지 않는다.
  - 비용: trysel K=12 기준 selector 4.93 GFLOPs/img, 합계 15.02다 (center 10.09). r=1(9개 위치)이면 약 1/3이 된다.
  - 되먹임 위험: midattn과 달리 attention이 아니라 출력을 쓴다. 그래도 student가 원판 밖 정보를 덜 쓰도록 바뀔 수 있다. screen 뒤에 self proxy로 확인한다.

## 2026-10-30 [claude] 라운드 9 메모
- H-W11 판정 (analysis_student_sufficiency_1030_173025_6f96): 약한 지지. suff_disc는 좋은 student에서 K=12 teacher kl_worst를 0.694→0.48~0.55로 줄이지만 masked teacher WGA는 그대로다(39.85~41.35). 이미지별 student-teacher KL 순위 상관 0.25~0.30. 지름길 student에서는 center보다 나쁘다. suffdisc@12 screen 결과로 최종 판정한다.
- **H-W12 centre-prior rollout (`priorroll`, parent rollout, 아직 만들지 않음).**
  - 근거: screen seed 0에서 K=12는 center > rollout (56.39 vs 45.11), K=29·49는 rollout > center. center@12 student에서 읽은 rollout top-12의 kl_worst 0.603 < center 원판 0.694 (analysis_student_sufficiency_1030_173025_6f96). 분석 analysis_prior_rollout_1030_174424_16a6로 먼저 검증한다.
  - 아이디어: rollout 점수에 중앙 가우시안 사전을 곱해(로그 공간에서 더해) top-K를 고른다. K가 작을수록 사전이 연속성을 주고, K가 크면 rollout 순위가 남는다.
  - 가장 가까운 선행 기법과의 차이: MaskedKD·rollout은 사전 없는 attention 순위, center는 신호 없는 사전이다. saliency에 center-bias 사전을 더하는 것은 시선 예측(saliency prediction) 문헌에서 흔하다(예: Judd et al. 2009 center prior). KD teacher 입력 선택에 쓴 예는 아직 찾지 못했다. 이 조합은 노벨티가 약하므로, 분해(사전만 쓴 random+prior)로 신호의 기여를 보여야 한다.

## 2026-10-30 [claude] 라운드 10 메모
- H-W12 판정 (analysis_prior_rollout_1030_174424_16a6): 지지. rollout student 3개와 full student에서는 모든 K에서 prior_s가 center·top_rollout보다 kl_worst가 낮다. center12 student의 K=12만 예외다(0.614 대 0.603).
- **`priorroll` 제출 (selectors/priorroll.py, parent rollout).** σ = √(K/π), 곧 같은 K의 center 원판 반지름(K=12/29/49에서 1.95/3.04/3.95)이다. 분석에서 K별 최적 σ는 대체로 2/4/4였다. σ 규칙은 분석을 본 뒤 정했고, screen에서 고른 값이 아니다.
  - 근거: 위 분석. 아이디어: 사전으로 연속성을 주고, rollout 신호로 사전 안팎의 패치를 고른다.
  - 선행과의 차이: rollout(사전 없음)과 center(신호 없음)의 결합이다. 시선 예측의 center-bias(Judd et al. 2009)와 같은 형태라 노벨티는 약하다. 이기면 분해가 필수다: (1) random 점수에 같은 사전을 씌운 변형(`priorrand`)으로 신호의 기여를 본다. (2) 사전만 쓴 결과는 곧 center다. (3) MaskedKD와의 겹침, 중심 거리를 잰다.
  - parent check: 같은 K에서 rollout보다 승격 기준 이상 높아야 사전 부분이 효과가 있다고 본다. K=12에서는 rollout 45.11, center 56.39다. 그래서 실질 기준은 center를 넘는 것이다.

## 2026-10-30 [claude] 라운드 11 메모
- H-W12 판정 (screen): priorroll@12 89.07/54.89 (screen_priorroll_k12_s0_1030_175140_405f)는 center@12(56.39)와 비긴다. priorroll@29 89.82/60.15 (screen_priorroll_k29_s0_1030_175144_018d)는 rollout@29(65.41)와 center@29(62.41)보다 낮다. **기각.** suffdisc@12 89.32/58.65 (screen_suffdisc_k12_s0_1030_173118_504a)는 parent center 대비 threshold 미만이다(UNPROVEN). **H-W11도 닫는다.**
- **H-W13 대리 지표의 표본 밖 검증 (analysis_proxy_holdout_1030_182326_7254): 실패.** 각 run 자신의 student로 잰 self kl_worst는 K=29에서 priorroll 0.391 < rollout 0.443 < center 0.517이다. student WGA는 rollout 65.41 > center 62.41 > priorroll 60.15다. K=12에서는 suffdisc 0.533 < priorroll 0.605 < center 0.694 < rollout 0.700이고, student WGA는 suffdisc 58.65 > center 56.39 > priorroll 54.89 > rollout 45.11다(priorroll과 center의 차이 1.50은 threshold 안). 결론: H-W10에서 고른 대리 지표는 16개 run에 맞춘 것이고 표본 밖에서 K=29 순위를 틀린다. 앞으로 이 지표로 선택기를 미리 거르지 않는다. 'clean 이미지에서 teacher가 충실하다'는 것만으로는 teachability가 설명되지 않는다.
  - 되먹임 관찰: priorroll student 자신의 plain rollout top-K는 priorroll 선택과 0.81~0.85 겹친다. rollout student에서 같은 겹침은 0.34~0.53이다. 사전으로 학습하면 student attention이 중앙으로 모인다.
- **H-W14 (분석 analysis_prior_failure_1030_182537_611c).**
  - 근거: 위 실패. 학습 batch는 전부 cutmix/mixup이라서 compact한 중앙 선택은 붙여 넣은 이미지 쪽을 teacher에게서 지울 수 있다.
  - 아이디어: cutmix/mixup 입력에서의 kl_worst와 cutmix 상자 비율을 self student로 잰다. student의 place 탐침도 함께 잰다.
  - 선행과의 차이: analysis_train_condition_gap(H-W8)는 L9 대 rollout만 봤다. 여기서는 공간 사전 계열(center, priorroll, suffdisc)을 같은 조건으로 잰다. 미리 정한 판정: K=29에서 cutmix/mixup kl_worst 순서가 rollout < center ≤ priorroll이면 (a)가 실패를 설명한다.

## 2026-10-30 [claude] 라운드 12 메모
- **H-W15 지름길 상속 (분석 analysis_shortcut_inherit_1030_183518_f5cf).**
  - 근거: H-W13·H-W14 실패. K=29에서 full teacher에 덜 충실한 rollout(self kl_worst 0.443)이 priorroll(0.391)보다 student WGA가 높다(65.41 대 60.15). full teacher 자신도 지름길에 기댄다(val WGA 64.66).
  - 아이디어: val 이미지를 full teacher가 맞힌 것(FR)과 틀린 것(FW)으로 나눈다. 각 run 자신의 student로 고른 masked teacher가 FW 이미지를 고쳐 맞히는 비율(fix_fw), FW에서의 참 클래스 확률, 그리고 비용으로 FR 소수 그룹에서의 참 클래스 확률을 잰다. 위치 다양성(선택 빈도 엔트로피, 이웃 이미지 Jaccard)도 함께 잰다. 대상은 끝난 screen run 전부(8개 방법 × K 12·29·49)이다.
  - 미리 정한 판정: seed 0 run 전체에서 부호 보정 Spearman ≥ 0.8이고, K=29의 rollout > center > priorroll 순서를 맞히면 후보 축이다. 맞지 않으면 기각한다.
  - 선행과의 차이: KD에서 teacher의 편향 상속을 다룬 연구(Debiased Distillation, arXiv 2302.11187)는 teacher의 출력이나 학습 데이터를 바꾼다. 여기서는 teacher 입력 토큰 선택만으로 상속을 줄이는지를 묻는다. 지지되면 '충실도 최대'가 아니라 '지름길 오류 지점에서의 탈충실'을 목표로 하는 선택기를 설계한다.

## 2026-10-30 [claude] 라운드 13 메모
- H-W15 판정 (analysis_shortcut_inherit_1030_183518_f5cf): **기각.** fix_fw의 Spearman은 0.64이고 K=29 순서도 틀렸다(priorroll 40.51 > rollout 37.97 > center 36.71). 위치 다양성은 음의 상관이다(div_ent −0.27).
- **H-W16 class-gradient-weighted rollout (`gradroll`, parent rollout).**
  - 근거: 끝난 screen에서 rollout은 K=29·49 최고이고 K=12에서 center에 진다. rollout은 클래스와 무관한 신호라 배경을 그대로 담는다. H-W7(analysis_shortcut_probe_1030_160112_bfc0)에서 rollout masked teacher의 place 탐침은 78.38(full student, K=49)로 L9(71.35)보다 높다.
  - 아이디어: 층마다 attention을 student 자신의 예측 logit 기울기로 가중한다(음수 제거). 그 relevance를 전파해 teacher 입력 top-K를 고른다.
  - 선행과의 차이: Chefer et al. 2021은 설명용이다. Prune and Merge(arXiv 2503.23455)는 추론 가지치기에 기울기 가중 attention을 쓴다. 여기서는 KD 중 고정 teacher의 입력을 student 관점에서 고른다. 규칙 자체는 기존 기법이라 노벨티는 적용처에만 있다.
  - 판정: 같은 K에서 rollout보다 threshold 이상 높아야 하고, center·maskedkd 등도 넘어야 한다. 이기면 (1) rollout·maskedkd와의 겹침, (2) 겹치지 않는 토큰을 random으로 바꾼 대체 실험, (3) 중심 거리를 본다.

## 2026-10-30 [claude] 라운드 14 메모
- K=29 seed 1 재현: rollout 63.16 > priorroll 60.90 > center 57.89 (screen_*_k29_s1_*). rollout 우위는 재현되고, 대리 지표(H-W13~15)가 rollout을 낮게 본 실패도 실제다. 새 가설은 세우지 않는다. 남은 미해결 질문: '왜 full teacher에 덜 충실한, 흩어진 rollout 선택이 K=29·49에서 student WGA가 더 높은가'. gradroll(H-W16) 결과가 나오면 rollout 계열 안에서 이 질문을 다시 본다.
- K=12 seed 1 재현(center, suffdisc, priorroll)을 넣었다. suffdisc가 seed 1에서도 center보다 높지 않으면 center 계열(사전·sufficiency)을 K=12에서 닫는다.

## 2026-10-30 [claude] 라운드 15 메모
- H-W16 판정 (K=29): gradroll@29 89.49/63.91 (screen_gradroll_k29_s0_1030_184255_9017)는 parent rollout(65.41) 대비 −1.50으로 UNPROVEN이고 비용이 더 든다. K=12 결과(학습 중)를 보고 계열을 닫는다. K=49는 넣지 않는다.
- **H-W17 entropy-adaptive per-image budget rollout (`adaroll`, parent rollout).**
  - 근거: 주 지표의 빈칸. K=49에서 rollout은 WGA 허용 오차 안(64.66 ≥ 64.22)이지만 정확도가 0.42%p 모자라고(screen_rollout_k49_s0_1030_135953_a1a9), center는 정확도는 안(90.16)이지만 WGA가 밖이다(screen_center_k49_s0_1030_135953_4476). K=49에서 둘 다 통과하면 MaskedKD의 98보다 작은 K가 된다. 선택 '순위'를 바꾼 시도(prior, gradient, sufficiency, 층 선택)는 모두 rollout을 넘지 못했다. 그래서 순위는 그대로 두고 '이미지별 토큰 배분'만 바꾼다.
  - 아이디어: student rollout 분포의 유효 패치 수 exp(H)에 비례해 이미지별 k_i를 정한다([K/2, 2K]로 자르고 배치 평균 ≤ K). top-k_i rollout 패치를 teacher에 준다.
  - 선행과의 차이: A-ViT(Yin et al. 2022)·AdaViT(Meng et al. 2022)·DynamicViT는 추론 중인 모델 자신의 토큰 수를 바꾼다. 여기서는 student의 attention 퍼짐이 고정 KD teacher의 입력 수를 정한다. MaskedKD·rollout은 모든 이미지에 같은 K를 쓴다.
  - 판정: K=49에서 rollout 대비 threshold 이상(정확도 +0.76 또는 WGA +2.45, 다른 쪽에서 지지 않음), 그리고 center·maskedkd·tome·grid·random보다 낮지 않을 것. 정확도 ≥ 90.08과 WGA ≥ 64.22를 동시에 만족하면 주 지표 후보다. 이기면 분해: (1) k_i를 무작위로 섞은 변형(배분 신호 제거), (2) 이미지별 k_i와 그룹의 관계(분석 전용), (3) maskedkd 겹침·중심 거리.

## 2026-10-30 [claude] 라운드 16 메모
- H-W16 판정 (K=12): gradroll@12 88.16/60.90 (screen_gradroll_k12_s0_1030_184259_e2cd)는 parent rollout(45.11)보다 WGA +15.79, center(56.39)보다 +4.51. **승격 후보**지만 분해 전이다. K=29에서는 rollout과 비김(−1.50).
- **H-W18 (분해로 검증할 가설): 작은 K에서 plain rollout이 무너지는 이유는 rollout top-K가 소수 고노름/배경 토큰에 몰리기 때문이고, 예측 클래스 logit에 양의 기여를 하는 attention만 전파하면(gradroll) 그 토큰을 피한다.**
  - 근거: rollout@12 45.11 대 rollout@29 65.41의 급락, H-W5(analysis_register_tokens_1030_153157_efbe)의 고노름 토큰 쏠림, gradroll@12의 회복.
  - 분석: analysis_gradroll_decomp_1030_193812_07ca (hn_frac, 겹침, cdist, 대체 변형의 teacher 충실도). 미리 정한 읽기: gradroll의 hn_frac이 rollout보다 낮고, gradroll∩rollout 겹침이 K=12에서 K=29보다 낮으면 H-W18 지지. gradroll cdist가 center(≈1.6) 수준이면 공간 배치 설명을 배제할 수 없다.
  - 대체 screen: gradrep@12 (gradroll∩maskedkd 유지 + random). WGA가 random@12(33.83) 근처로 떨어지면 이득은 gradroll 고유 토큰에서 온다.
  - 선행과의 차이: Chefer et al. 2021은 설명용, Prune and Merge(2503.23455)는 추론 가지치기. MaskedKD·SaiT·MoVE-KD는 CLS attention만 쓴다. KD teacher 입력 선택에 클래스 기울기 relevance를 쓰고, 그 이득이 작은 K의 최악 그룹에서 나온다는 것이 차이다(1회 검색 기준).

## 2026-10-30 [claude] 라운드 17 메모
- H-W18 분해 결과 (analysis_gradroll_decomp_1030_193812_07ca): 공간 배치 설명은 배제한다(cdist 4.68 대 center 1.29). 'K=12에서 rollout과 덜 겹친다'는 맞다(0.29 대 K=29 0.50). 고노름 회피는 rollout12 student에서만 성립한다(0.049 대 0.125). gradroll12 student에서는 0.057 대 0.058이라 H-W18의 '고노름 토큰 회피' 부분은 지지되지 않는다. 남는 설명은 '클래스 logit에 양의 기여를 하는 비-MaskedKD 토큰'이다. 대체 변형에서 teacher WGA가 51.88에서 25.56으로 떨어진다.
- 다음에 확인할 것(gradrep 결과 뒤): gradroll 고유 토큰이 객체(새) 위인지 볼 것. val에는 분할 마스크가 없으므로 대안을 쓴다. 후보는 full teacher의 Chefer relevance나 oracle과의 겹침이다(분석 전용, 주장 근거로는 쓰지 않음). 그리고 gradroll이 학습 초반(e25 teacher WGA 28.57)에 오히려 나쁜 이유를 본다. ctx.progress로 초반에 rollout을 쓰는 변형은 승격 판정 뒤에만 검토한다.

## 2026-10-30 [claude] 라운드 18 메모
- H-W11 재개 (suffdisc): 라운드 14에 정한 규칙('seed 1에서도 center보다 높지 않으면 center 계열을 K=12에서 닫는다')에 따라 닫지 않는다. seed 1에서 suffdisc 63.91 대 center 51.88 (screen_suffdisc_k12_s1_1030_185953_9455, screen_center_k12_s1_1030_185949_1766). 2-seed 평균 +1.25 / +7.14. priorroll@12는 2-seed 평균이 center와 같아서(54.14 대 54.14) 닫는다.
- **H-W19 (분해로 검증할 가설): suffdisc의 이득은 sufficiency 신호가 아니라 원판을 이미지마다 옮긴 것(teacher 입력의 위치 다양성)에서 올 수 있다.**
  - 근거: analysis_student_sufficiency_1030_173025_6f96에서 sufficiency 위치와 teacher-best 위치의 상관은 약했다(이미지별 순위 상관 0.25~0.30). masked teacher WGA도 center와 같았다(39.85~41.35). 그런데 student WGA는 2-seed 평균으로 center보다 7.14 높다. 신호가 약한데 이득이 크면 신호 밖의 요인을 의심해야 한다.
  - 대체 screen: jitdisc@12 (parent suffdisc, 같은 25개 후보 위치 중 균등 무작위, student forward 없음), seed 0·1. 미리 정한 읽기: jitdisc 2-seed WGA가 center 2-seed(54.14) 쪽(≤ 56.6)이면 이득은 sufficiency 신호에서 온다. suffdisc(61.28) 쪽(≥ 58.8)이면 이득은 위치 지터에서 오고, 그 경우 suffdisc의 4.93 GFLOPs/img는 쓸모없는 비용이다. 그 사이면 판정하지 않는다.
  - 선행과의 차이: jitdisc는 방법 후보가 아니다. ROAR식 대체 대조군이다. jitdisc가 이기면 그것은 해석가능성 신호가 없는 공간 규칙이라 주장 대상이 아니다. 대신 'K=12에서 teacher 입력 위치를 흔들면 student WGA가 오른다'는 분석 결과로 기록한다.

## 2026-10-30 [claude] 라운드 19 메모
- H-W17 (adaroll) 닫음: adaroll@49 90.08/63.91 (screen_adaroll_k49_s0_1030_192757_7edc). parent rollout@49 대비 +0.42/−0.75로 UNPROVEN이다. 주 지표 조건(WGA ≥ 64.22)도 만족하지 못한다. 이미지별 예산 배분은 효과가 없다.
- H-W18 student 수준 판정: gradrep@12 54.89 (screen_gradrep_k12_s0_1030_193817_c83d) 대 gradroll 60.90 → −6.01. center(56.39)를 앞선 몫은 gradroll 고유 토큰에서 온다. 그러나 교집합+random도 rollout(45.11)보다 9.8 높다. 해석 후보는 'MaskedKD와 겹치는 4~5개 토큰(새 위)에 무작위 문맥을 더하면 작은 K에서 teacher 입력 위치가 다양해진다'이다. jitdisc(H-W19)와 같은 축이다. jitdisc 결과와 함께 다음 라운드에 본다.
- 다음 분석 후보(final seed 1 뒤): gradroll 고유 토큰이 객체 위인지 확인한다. 기준은 full teacher Chefer relevance top-K와의 겹침이다(분석 전용). 그리고 gradrep의 이득 중 rollout 대비 남은 부분을 '위치 다양성'으로 설명할 수 있는지 본다. 예측 지표는 학습 중 선택 위치 엔트로피다.

## 2026-10-30 [claude] 라운드 20 메모
- H-W19 판정 (미리 정한 읽기 그대로): jitdisc@12 2-seed 88.53/53.01 (screen_jitdisc_k12_s0_1030_195912_ead1, screen_jitdisc_k12_s1_1030_195921_48c7) ≤ 56.6 → **위치 지터 가설을 기각한다.** suffdisc 2-seed 61.28과의 차이 −8.27은 threshold 2.45의 3배가 넘는다. center 2-seed 54.14와는 같은 수준이다.
- 부수 결과: 무작위 위치 다양성 자체는 K=12에서 WGA를 올리지 않는다. 그래서 라운드 19에 적은 gradrep 잔여 이득(rollout 대비 +9.8)을 '위치 다양성'으로 설명하는 후보는 약해진다. gradrep는 MaskedKD 교집합 토큰(새 위)을 유지한다는 점이 jitdisc와 다르다. 따라서 남은 설명은 '객체 위 소수 토큰 + 문맥'이다(분석 미검증).
- suffdisc 승격 전 분해의 남은 부분(겹침·공간 분포·drop-path 잡음)은 analysis/suffdisc_decomp.py → analysis_suffdisc_decomp_1030_202612_878c. 미리 정한 읽기: (a) 학습 시 픽(train mode)과 eval 픽의 일치율이 1/25 근처(≤ 0.15)면 학습 중 선택은 사실상 무작위다. 그렇다면 jitdisc와의 차이를 신호로 설명할 수 없으니 승격하지 않고 재검토한다. (b) centre_frac가 0.8 이상이면 suffdisc는 center와 거의 같다. 그렇다면 이득의 출처를 다시 본다.
- 선행과의 차이(변화 없음): RISE/ERASER sufficiency는 설명 평가용이다. KD teacher 입력 위치 선택에 쓴 예는 찾지 못했다(검색은 승격 시 수행).

## 2026-10-30 [claude] 라운드 21 메모
- suffdisc 분해 판정 (analysis_suffdisc_decomp_1030_202612_878c): 읽기 (a) train/eval 픽 일치율 0.34~0.43 > 0.15 통과, (b) centre_frac 0.03 < 0.8 통과. 승격 전 분해 3가지가 끝났다. final은 gradroll final 3 seed 뒤에 정한다.
- 새 관찰 H-W20 (가설, 미검증): suffdisc의 teacher 수준 이점(masked teacher WGA 48.12 대 center 41.35)은 suffdisc로 학습한 student에서만 나타난다. jitdisc·center student에서 같은 규칙을 읽으면 39.10~45.11이다. 선택기와 student가 함께 적응(공적응)하는 것이 이득의 일부라는 뜻이다. midattn은 이 되먹임이 해로웠고(H-W8), suffdisc는 이로운 쪽이다.
  - 검증 아이디어: suffdisc student e25/e50/final의 이 값이 학습 중에 오르는지 본다(e25/e50은 이미 있다: 41.35/40.60 → final 48.12. 오른다). 다음 분석에서는 suffdisc student의 sufficiency 지도와 teacher-best 위치의 순위 상관이 학습 중 오르는지를 잰다(analysis_student_sufficiency_1030_173025_6f96의 0.25~0.30과 비교).
  - 선행과의 차이: 자기 설명으로 학습 입력을 고르는 되먹임은 attention-guided 학습(예: 설명 기반 정규화)에서 다루지만, KD teacher 입력 선택에서 이 되먹임의 방향(해로움/이로움)을 비교한 예는 찾지 못했다(검색 미수행, 승격 시 수행).
- gradroll final s1 caveat: tome@12 대비 정확도가 두 seed 모두 threshold 이상 낮다(−0.83, −1.67). CLAIM_REFS 기준으로 tome에 대해서는 WGA 승, 정확도 패다. 감사 요청 때 'tome은 not matched(teacher 비용 2.69배), total GFLOPs는 tome 12.00 < gradroll 12.81'을 그대로 적는다.

## 2026-10-30 [claude] 라운드 33 메모
- **H-W21 (분석 가설): gradroll@12 seed 3의 WGA 하락(51.88, final_gradroll_k12_s3_1030_220958_176b; s1 63.16, s2 62.41)은 (a) 학습 중 선택이 무너진 것(masked teacher WGA 하락, MaskedKD 쪽으로 이동)이거나 (b) 선택은 같고 student 자체가 group 2에서 약해진 것이다.**
  - 분석: analysis_gradroll_seeds_1030_232303_4bd3. 미리 정한 읽기: s3 student(e25/e50/final)의 gradroll masked teacher WGA가 s1·s2보다 5 이상 낮거나 grad∩maskedkd 겹침이 0.1 이상 높으면 (a). 그 차이가 없고 student 자신의 group 2 정확도만 낮으면 (b). (a)이면 선택 안정화(예: 학습 진행에 따른 rollout 혼합)가 다음 방법 후보이고, (b)이면 gradroll의 seed 분산은 방법 고유가 아니라 student 분산으로 보고 CI 해석만 바꾼다.
  - 선행과의 차이: Chefer et al. 2021의 gradient-weighted relevance는 설명 평가용이다. 학습 중 seed별 설명 안정성이 KD 입력 선택의 결과를 좌우하는지 본 예는 찾지 못했다(검색 미수행).

## 2026-10-30 [claude] 라운드 34 메모
- H-W21 결과(analysis_gradroll_seeds_1030_232303_4bd3): seed 3에서는 gradroll·rollout 둘 다 masked teacher WGA가 낮고 student WGA도 낮다. gradroll∩maskedkd 겹침은 seed 사이에 같다(0.37~0.40). seed 사이 선택 일치율은 0.50~0.55다. 선택 안정화 변형은 근거가 약해 보류한다.
- **H-W22 (검증 계획): suffdisc@12의 center 대비 이득이 seed 2·3에서도 유지되는가.**
  - 근거: screen s0 +0.58/+2.26, s1 +1.92/+12.03 (screen_suffdisc_k12_s{0,1}_*, screen_center_k12_s{0,1}_*). 분해 3가지를 통과했다(jitdisc, analysis_suffdisc_decomp_1030_202612_878c).
  - 제출: screen_suffdisc_k12_s2_1030_233019_98dd, screen_center_k12_s2_1030_233024_3dcf, screen_suffdisc_k12_s3_1030_233028_39ad, screen_center_k12_s3_1030_233032_bf21.
  - 미리 정한 읽기: 승격하려면 세 조건을 모두 채워야 한다. (1) 4 seed의 짝지은 WGA 차이(suffdisc − center) 평균이 2.45 이상이다. (2) 4 seed 중 3개 이상에서 그 차이가 양수다. (3) 정확도 차이 평균이 −0.76보다 크다. 이 경우 공식 Parent check(seed 0)가 UNPROVEN이라는 사실을 함께 적고 final s1을 제출한다. 셋 중 하나라도 어기면 suffdisc 계열을 닫는다.
  - 선행과의 차이: 라운드 17·18 메모와 같다. student 자신의 sufficiency(원판만 받았을 때의 예측 유지)로 teacher 입력 위치를 고른다. RISE·Extremal Perturbation은 설명용이고, MaskedKD는 attention 순위를 쓴다.

## 2026-10-31 ideas from Codex (independent)
IDEA 1: Concept Coverage (untested) — From cached student block-6 patch outputs \(Z\), form signed channels \(X=[\max(Z,0),\max(-Z,0)]\); fit rank-6 nonnegative matrix factorization \(X\approx HW\) using 15 updates, and normalize each concept map: \(u_{ic}=H_{ic}/(\sum_jH_{jc}+\epsilon)\).  
IDEA 2: Sparse-Residual Counterfactual Attribution (untested) — Scale cached block-6 patch features \(Z\) by their median norm \(a\), giving \(X\); approximately minimize \(\frac12\|X-L-E\|_F^2+0.1\sum_i\|E_i\|_2\), with \(\operatorname{rank}(L)\le4\), using 10 alternating randomized rank-4 approximations and rowwise soft-thresholding.  

## 2026-10-31 [claude] 라운드 35 메모
- **H-W23 (Codex IDEA 1을 채택): concept coverage.** student block-6 patch feature를 부호 분리 후 rank-6 NMF로 개념 지도로 나누고, 개념마다 log 체감 이득으로 K개를 탐욕적으로 고른다 (selectors/conceptcov.py, parent none, selector 0.029 GFLOPs/img, trysel 통과).
  - 근거: Collins, Achanta & Süsstrunk ECCV 2018 (DFF). 프로젝트 반대 근거도 있다: K=12에서 퍼뜨리는 규칙(nms·half_grid·grid·random)은 WGA를 잃었고(analysis_group_fidelity_1030_143842_1ea0, screen_grid/random_k12), 배경 정보를 teacher에 넘기면 WGA가 떨어진다(analysis_shortcut_probe_1030_160112_bfc0). 이 screen은 '개념 단위 커버리지'가 '공간 퍼뜨리기'와 다른지를 묻는다.
  - 제출: screen_conceptcov_k12_s0_1031_000621_b88e.
  - 미리 정한 읽기: seed 0 WGA가 center s0(56.39)보다 2.45 이상 높고 정확도가 center s0(88.74) −0.76 이상이면 seed 1 screen과 분해(겹침·cdist·배경 개념 비중)를 한다. WGA가 rollout s0(45.11) 이하이면 '개념 커버리지는 배경 개념에도 토큰을 주어 해롭다'로 읽고 계열을 닫는다. 그 사이이면 seed 1 하나만 더 보고 닫을지 정한다.
  - 선행과의 차이: Token Pooling(Marin et al. 2021)·ToMe는 재구성 오차나 유사도로 토큰을 합친다. DFF는 설명용 개념 지도다. 원래 patch를 개념 커버리지로 골라 KD teacher 입력으로 쓰는 예는 찾지 못했다(이번 라운드 검색 미수행; 승격 시 검색).

## 2026-10-31 [claude] 라운드 36 메모
- H-W22 결과는 통과다. screen 4 seed의 짝지은 차이(suffdisc − center)는 acc +0.58/+1.92/+1.83/+1.08, WGA +2.26/+12.03/+12.79/+13.53이다(screen_suffdisc_k12_s{0..3}_*, screen_center_k12_s{0..3}_*). final_suffdisc_k12_s{1,2,3}를 제출했다.
- 미리 정한 final 읽기:
  - Core comparison(seeds 1,2,3)에서 maskedkd·rollout·tome·grid·center·random을 모두 이기면(한 지표 WIN, 다른 지표 LOSS 없음) AUDIT_REQUEST를 쓴다.
  - center 대비 WGA가 n.s.이면 'student sufficiency 신호'라는 주장은 성립하지 않는다. 이 경우 suffdisc 계열을 실패한 방향으로 옮긴다.
  - 비용은 suffdisc가 tome보다 비싸다(15.02 대 12.00 GFLOPs/img). 그래서 tome에 이겨도 비용 caveat을 함께 적는다.
- 문헌(라운드 36 검색 1회): student sufficiency로 KD teacher 입력 영역을 고르는 선행은 찾지 못했다. 가장 가까운 것은 arXiv 2411.17046(데이터 없는 증류, 중앙 편향 patch 합성)과 arXiv 2107.01378(patch manifold 증류)이고, 둘 다 실제 입력 선택이 아니다.

## 2026-10-31 [claude] 라운드 37 메모
- H-W23 판정: conceptcov@12 s0 81.82/27.07 (screen_conceptcov_k12_s0_1031_000621_b88e) ≤ rollout s0 45.11 → **닫는다**(미리 정한 읽기). 개념 커버리지는 K=12에서 공간 퍼뜨리기(grid 35.34, random 33.83)보다도 나쁘다.
- **H-W20 (teachability 설명, 분석으로 검증): suffdisc의 이득은 student sufficiency 순위가 teacher 자신의 충실도 순위와 맞아 가는 공적응에서 온다.**
  - 근거: analysis_suffdisc_decomp_1030_202612_878c에서 suffdisc 규칙의 masked teacher WGA가 suffdisc student에서만 높았다(48.12 대 다른 student 39.10~45.11). analysis_student_sufficiency_1030_173025_6f96의 이미지별 student-teacher KL 순위 상관은 0.25~0.30이었다.
  - 분석: analysis/suff_teacher_align.py → analysis_suff_teacher_align_1031_003305_e9b8. 25개 원판에서 student KL과 teacher KL의 이미지별 Spearman(rho), argmin 일치(우연 0.04), student pick의 teacher KL과 masked teacher 그룹 정확도를 잰다. student는 suffdisc screen s0~s3(e25·e50·final), center s0~s3, jitdisc s0·s1이다.
  - 미리 정한 읽기: final에서 최악 그룹 rho가 같은 seed의 center student보다 0.05 이상 높은 seed가 4개 중 3개 이상이고, suffdisc s0~s3 평균 rho가 e25→final로 오르면 H-W20을 지지한다. 둘 다 아니면 'student sufficiency가 teacher 충실도를 따라간다'는 설명을 AUDIT_REQUEST에서 빼고, 이득의 기제를 미해결로 적는다.
  - 선행과의 차이: ERASER·RISE의 sufficiency는 주어진 설명을 평가하는 지표다. 여기서는 그것을 student 쪽에서 재서 teacher 입력 위치를 정하고, 그 순위가 teacher 쪽 순위와 학습 중 정렬되는지를 본다.
- H-W20 판정: **지지** (analysis_suff_teacher_align_1031_003305_e9b8). 최악 그룹 rho가 같은 seed의 center student보다 +0.030/+0.052/+0.092/+0.066 높고(3/4 seed ≥ 0.05), suffdisc 평균이 e25 0.235 → final 0.315로 오른다. 전체 그룹 rho에서는 차이가 없다. 한계는 PROGRESS 라운드 37을 볼 것.

## 2026-10-31 [claude] 라운드 38 메모
- suffdisc@12 final 판정: 라운드 36에 미리 정한 읽기를 충족한다(CLAIM_REFS 6개 모두 WGA WIN, 정확도 LOSS 없음, center 대비 WGA +8.77 [+4.06, +13.47]). AUDIT_REQUEST를 썼다.
- **H-W24 (비용 맞춤): suffdisc r=1.** 후보를 25개(±2)에서 9개(±1)로 줄인다. trysel K=12: selector 1.776, total 11.864 GFLOPs/img로 tome@12(12.00)보다 싸다.
  - 근거: AUDIT_REQUEST caveat 2·3(비용 1.49배). analysis_suffdisc_decomp_1030_202612_878c의 mean_shift가 2.03~2.06이라 ±2 끝까지 쓰는 픽이 많다. 그래서 r=1에서는 이득이 줄 수 있다.
  - 제출: screen_suffdisc_k12_s0_1031_011600_6b15, screen_suffdisc_k12_s1_1031_011604_6add (sel_args r=1).
  - 미리 정한 읽기: 2-seed WGA 평균이 같은 seed의 center screen(s0 56.39, s1 51.88)보다 2.45 이상 높고 정확도 손실이 0.76 이내이면, 비용 맞춘 변형으로 seed 2·3 screen을 본다. center와 차이가 2.45 미만이면 '이득에는 ±2 탐색 범위가 필요하다'로 기록한다. 그러면 비용 caveat은 그대로 남는다.
- **H-W25 (같은 계열의 다른 K): suffdisc@29 s0 screen** (screen_suffdisc_k29_s0_1031_011608_3604, total 22.62 GFLOPs/img). 미리 정한 읽기: rollout@29(2-seed 89.41/64.28)과 center@29(89.03/60.15)보다 WGA가 2.45 이상 높으면 K=29에서도 일관된 것으로 본다. 아니면 주장을 K=12에 한정한다는 caveat 5를 확정한다. 비용이 rollout@49(16.45)보다 커서 K=29에서는 승격하지 않는다.
- 선행과의 차이: 라운드 36 메모와 같다.

## 2026-10-31 [claude] 라운드 39 메모
- (라운드 40 판정) H-W24: r=1 2-seed가 center 대비 WGA +1.88로 기준 2.45에 못 미친다 → '이득에는 ±2 탐색 범위가 필요하다'. seed 2·3 screen은 넣지 않는다(RESULTS '실패한 방향').
- **H-W26 (같은 계열의 다른 K, 보강): suffdisc@49 s0 screen** (screen_suffdisc_k49_s0_1031_012539_2b13, 기본 r=2). 근거: AUDIT_REQUEST caveat 5(K=12 한정). 미리 정한 읽기: rollout@49(89.66/64.66)와 center@49(90.16/60.15) screen s0보다 WGA가 2.45 이상 높으면 K=49에서도 일관된 것으로 본다. 비슷하거나 낮으면 'suffdisc의 이득은 작은 K(연속 영역 하나가 유리한 K)에 한정된다'로 기록한다. 이것은 H-W9(작은 K에서 center가 rollout을 이김)와 맞는 예측이다. 비용이 크므로 K=49에서는 승격하지 않는다.
- 선행과의 차이: 라운드 36 메모와 같다.

## 2026-10-31 [claude] 라운드 41 메모
- (판정) H-W25: suffdisc@29 s0은 90.16/61.65다(screen_suffdisc_k29_s0_1031_011608_3604). rollout@29 대비 WGA −2.63, center@29 대비 +1.50으로 기준 2.45에 못 미친다. 그래서 caveat 5(K=12 한정)를 확정한다. 정확도는 rollout보다 +0.75 높지만, 이것은 threshold 0.76 안이다.
- **H-W27 (대체 실험 보강): jitdisc@12 screen seed 2·3** (screen_jitdisc_k12_s2_1031_020851_1187, screen_jitdisc_k12_s3_1031_020901_8cd5). 근거: 감사 항목 13·15. 이득이 sufficiency 순위에서 오는지, 아니면 원판을 중앙 밖으로 옮기는 것만으로 오는지 가려야 한다. 지금까지 seed 0·1의 결과는 다음과 같다. jitdisc 88.16/52.63, 88.91/53.38이고 기본 suffdisc 89.32/58.65, 89.91/63.91이다. WGA 차이는 −6.02와 −10.53이다. 미리 정한 읽기: 4-seed 짝 비교에서 jitdisc가 기본 suffdisc보다 WGA 평균 2.45 이상 낮고 4 seed 모두 음수이면 '이득은 sufficiency 순위에서 온다'를 3/3부에 쓴다. 아니면 '이득의 일부는 원판 이동 자체에서 온다'고 적는다.
- (판정) H-W27: 지지. jitdisc s2 88.49/54.89, s3 89.41/56.39(screen_jitdisc_k12_s2_1031_020851_1187, screen_jitdisc_k12_s3_1031_020901_8cd5). 같은 seed suffdisc 대비 WGA 4-seed 짝 차이 −6.02/−10.53/−7.52/−12.03, 평균 −9.03, 모두 음수 → '이득은 sufficiency 순위에서 온다'를 4/4부에 쓴다.
- **H-W28 (caveat 보강): suffdisc@29 screen seed 1** (screen_suffdisc_k29_s1_1031_022639_643b). 근거: H-W25가 seed 0 하나뿐이다. 비교: 같은 seed의 center@29 s1(screen_center_k29_s1_1030_182732_85c3)·rollout@29 s1(screen_rollout_k29_s1_1030_182725_2086). 미리 정한 읽기: 두 비교 모두 WGA +2.45 이상이 아니면 'K=12 한정' caveat을 2 seed로 유지한다. 새 config가 아니다.
- (판정) H-W26: suffdisc@49 s0은 90.24/60.90이다(screen_suffdisc_k49_s0_1031_012539_2b13, 31.75 GFLOPs/img). 같은 seed의 rollout@49(64.66) 대비 WGA −3.76, center@49(60.15) 대비 +0.75다. 기준 2.45에 못 미치고, 'suffdisc의 이득은 K=12에 한정된다'로 기록한다. H-W9의 예측(작은 K에서만 연속 영역이 유리)과 맞는다.

## 2026-10-31 [claude] 라운드 44 메모
- (판정) H-W28: **미리 정한 caveat 유지 조건이 깨졌다.** suffdisc@29 s1은 91.74/71.43이다(screen_suffdisc_k29_s1_1031_022639_643b). 같은 seed의 center@29 s1(89.07/57.89) 대비 acc +2.67·WGA +13.54, rollout@29 s1(88.99/63.16) 대비 +2.75/+8.27로, 두 비교 모두 WGA +2.45 이상이다. seed 0(−2.63 vs rollout)과 방향이 반대다. 2-seed 평균 90.95/66.54는 rollout 2-seed(89.41/64.28) 대비 +1.54/+2.26, center 2-seed(89.03/60.15) 대비 +1.92/+6.39다. → 'K=12 한정'은 'K=29는 seed에 따라 엇갈려 미정'으로 바꾼다. K=49(seed 0만)는 그대로 기준 미달이다.
- **H-W29 (K=29 판정): suffdisc·center·rollout@29 screen seed 2·3** (screen_suffdisc_k29_s2_1031_025411_5cac, screen_suffdisc_k29_s3_1031_025416_358a, screen_center_k29_s2_1031_025420_4cff, screen_center_k29_s3_1031_025424_fc02, screen_rollout_k29_s2_1031_025429_73a1, screen_rollout_k29_s3_1031_025433_3854). 근거: H-W25·H-W28이 반대로 나왔다. 미리 정한 읽기: 4-seed 짝 비교에서 suffdisc − rollout의 WGA 평균이 +2.45 이상이고 4 seed 중 3개 이상이 양수이면 'K=12·29에서 일관'으로 본다. 평균이 ±2.45 안이면 'K=29는 rollout과 구별되지 않음', −2.45 이하이면 'K=12 한정'으로 되돌린다. center도 같은 규칙을 쓴다. 새 config가 아니다(config 3의 seed 추가). K=29 final 승격은 이번 기한 안에서 하지 않는다. 주장은 K=12 final에 그대로 한정한다.
- (H-W29 중간) center@29 s2 89.49/62.41 (screen_center_k29_s2_1031_025420_4cff), s3 89.91/60.15 (screen_center_k29_s3_1031_025424_fc02). 같은 seed의 suffdisc·rollout은 실행 중이다. 판정은 4개가 끝난 뒤 217행 규칙으로 한다.
- (H-W29 center 쪽 판정, 라운드 46) suffdisc@29 s2 90.08/63.91 (screen_suffdisc_k29_s2_1031_025411_5cac), s3 90.33/65.41 (screen_suffdisc_k29_s3_1031_025416_358a). center 대비 같은 seed WGA 차이는 s0~s3 순서로 −0.76/+13.54/+1.50/+5.26이고, 평균 +4.89, 양수 3/4다. 규칙에 따라 center 대비 'K=12·29에서 일관'이다. rollout 쪽은 rollout@29 s2·s3가 끝난 뒤 판정한다.

## 2026-10-31 [claude] 라운드 47 메모
- (판정) H-W29 rollout 쪽: rollout@29 s2 88.74/63.16 (screen_rollout_k29_s2_1031_025429_73a1), s3 89.24/60.90 (screen_rollout_k29_s3_1031_025433_3854). suffdisc − rollout 같은 seed WGA 차이는 s0~s3 순서로 −3.76/+8.27/+0.75/+4.51이고 평균 +2.44, 양수 3/4다(acc 차이 +0.33/+2.75/+1.33/+1.08). 평균이 기준 +2.45에 0.01 못 미치므로 217행 규칙대로 **'K=29는 rollout과 구별되지 않음'**으로 판정한다. center 쪽은 'K=12·29에서 일관'(라운드 46). 라운드 41에 적은 s0 차이 −2.63은 metrics.json(65.41 vs 61.65)과 맞지 않는다. 이 표의 −3.76이 맞다.
- **H-W30 (새 seed 독립 재현, 미리 등록): suffdisc@12 vs maskedkd@12, screen seed 4·5·6.** 근거: 03:57 감사(codex)는 final seed 1~3이 승격 전 screen에서 이미 쓰였다고 지적했다. harness/train.py 188·207행은 tier와 상관없이 `torch.manual_seed(seed)`와 loader seed를 쓴다. 그래서 final s1~s3은 승격 판단에 쓴 screen s1~s3과 초기화·데이터 순서가 같고, 선택과 독립이 아니다. 아이디어: 한 번도 쓰지 않은 seed 4·5·6에서 같은 설정(config 1, sel_args `{}`)을 다시 돌린다. 이 결과는 이 항목을 쓴 뒤에 나온다. 미리 정한 읽기: 3 seed 모두 suffdisc − maskedkd의 acc와 WGA가 양수이고 WGA 평균이 +2.45 이상이면 '새 seed에서 재현', 하나라도 음수이면 '재현 실패'로 적고 주장을 내린다. 선행 기법과의 차이: 새 방법이 아니라 같은 설정의 재현이다(새 config 아님).
  - **H-W30 결과 (라운드 49 [claude]): 재현.** suffdisc − maskedkd (acc / WGA): s4 +5.50 / +37.59 (screen_suffdisc_k12_s4_1031_040422_fa2a vs screen_maskedkd_k12_s4_1031_040431_c049 ), s5 +5.42 / +45.86 (screen_suffdisc_k12_s5_1031_040435_4e00 vs screen_maskedkd_k12_s5_1031_040439_3cc2 ), s6 +4.17 / +35.34 (screen_suffdisc_k12_s6_1031_040444_bb51 vs screen_maskedkd_k12_s6_1031_040448_eb48 ). 3 seed 모두 양수이고 WGA 평균 +39.60 ≥ +2.45이다. 미리 정한 읽기에 따라 '새 seed에서 재현'으로 적는다.
- **H-W31 (새 seed 기준선, 미리 등록): suffdisc@12 vs rollout·center·tome·grid·random @12, screen seed 4·5·6.** 근거: H-W30과 같은 이유(final s1~s3이 승격 전 screen과 seed가 같음)로 3/4·4/4부 비교에도 선택과 독립인 seed가 필요하다. 아이디어: 기준선 5개를 seed 4·5·6에서 돌리고, 이미 끝난 suffdisc s4~s6과 짝지어 비교한다. 이 항목은 기준선 결과가 나오기 전에 썼다. 미리 정한 읽기: 기준선마다 3 seed 모두 suffdisc − 기준선 WGA가 양수이고, WGA 평균이 +2.45 이상이며, acc 평균이 −0.76 이상이면 '그 기준선 대비 재현'으로 적는다. 아니면 '재현 실패'로 적고 그 기준선 대비 주장을 내린다. 선행 기법과의 차이: 새 방법이 아니라 재현이다.
- **H-W32 (같은 비용 비교, 미리 등록): suffdisc@12 (15.0222 GFLOPs/img) vs maskedkd@49·center@49 (각 16.44·약 16.4 GFLOPs/img, suffdisc보다 비쌈), screen seed 4·5·6.** 근거: suffdisc@12는 maskedkd@12보다 ×1.49 비싸다(RESULTS_TABLE). 그래서 "같은 K 우위"가 "teacher 토큰을 더 쓴 것과 같은 비용"보다 나은지는 아직 모른다. maskedkd@49 seed 0은 89.24/51.88 (screen_maskedkd_k49_s0_1030_135953_3e16)이다. 아이디어: suffdisc보다 비용이 조금 더 큰 K=49에서 maskedkd와, 가장 강한 공간 기준선인 center를 새 seed 4~6에서 돌린다. 이미 끝난 suffdisc@12 s4~s6과 짝짓는다. 비교 상대에게 유리한 쪽(비용 더 큼)으로 잡은 보수적 비교다. 이 항목은 결과가 나오기 전에 썼다. 미리 정한 읽기: 기준선마다 3 seed 모두 suffdisc − 기준선 WGA가 양수이고, WGA 평균이 +2.45 이상이며, acc 평균이 −0.76 이상이면 '같은 비용 이하에서 우위'로 적는다. 아니면 '같은 비용 우위 없음'으로 적고 "비용 ×1.49" caveat을 유지한다. K=49의 random·grid·rollout·tome은 넣지 않는다(시간 절약). 그래서 이 비교는 CLAIM_REFS 비교가 아니라 비용 caveat에 대한 보조 근거다. 선행 기법과의 차이: 새 방법이 아니라 비용 통제다.
- **H-W33 (주 지표 새 seed 확인, 미리 등록): suffdisc@29 vs full@196, screen seed 4·5·6.** 근거: RESULTS_TABLE Primary metric에서 suffdisc의 최소 K는 29다(screen s0~s3, 90.58 ± 0.79 / 65.60 ± 4.18). 그런데 이 K도 승격 판단과 같은 seed에서 나왔다. 아이디어: suffdisc@29(sel_args `{}`)와 full@196을 새 seed 4~6에서 돌려 짝짓는다. 이 항목은 결과가 나오기 전에 썼다. 미리 정한 읽기: 3 seed 평균 suffdisc − full이 acc ≥ −0.75(screen 허용 오차)이고 WGA ≥ −2.45이면 'K=29에서 Full KD 허용 오차 이내 재현'으로 적는다. 아니면 '재현 실패'로 적고 주 지표 29를 screen seed 0~3 결과로만 표시한다. H-W32 job이 대기열에서 빠진 뒤(대기 ≤ 8 유지) 제출한다. 선행 기법과의 차이: 새 방법이 아니라 재현이다.
- **H-W34 (아이디어만, 미제출): suffdisc 선택기 비용을 줄여 같은 비용 비교를 다시 한다.** 근거: H-W32에서 center@49 s4 (screen_center_k49_s4_1031_064043_463c 90.91/64.66, 16.4588 GFLOPs/img)가 suffdisc@12 s4 (screen_suffdisc_k12_s4_1031_040422_fa2a 89.99/57.89, 15.0222)보다 앞섰다. suffdisc의 이득은 같은 K에서는 크지만 비용(선택기 25회 forward) 때문에 같은 비용에서는 사라질 수 있다. r=1(후보 9개, H-W24)은 이미 center와 구별되지 않았다(RESULTS 실패한 방향). 아이디어 후보: 후보 원판의 KL을 student 앞쪽 몇 블록까지만 계산하거나(early-exit sufficiency), 학습 진행에 따라 탐색 빈도를 줄인다. 선행 기법과의 차이: 비용 축소 변형일 뿐 새 신호가 아니다. H-W32 center s5·s6과 H-W33이 끝난 뒤, 대기열에 자리가 날 때만 판단한다.
  - [claude] 라운드 56 갱신 (제출 전에 등록): selectors/suffstride.py (parent suffdisc). 탐색 범위는 r=2 그대로 두고 이동 간격만 2로 둔다(후보 {-2,0,2}² = 9개). trysel 비용은 K=12에서 11.864 GFLOPs/img이다(selector 1.776). 이것은 r=1 run screen_suffdisc_k12_s0_1031_011600_6b15 (11.8641, 88.57/55.64)과 같은 비용이다. 제출: suffstride@12 screen seed 0 1개. 미리 정한 읽기: WGA가 center@12 s0 (screen_center_k12_s0_1030_135953_c64e 88.74/56.39)보다 +2.45 이상 높고 acc가 −0.76 이상이면, 그리고 suffdisc@12 s0 (screen_suffdisc_k12_s0_1030_173118_504a 89.32/58.65)보다 WGA가 −2.45 넘게 낮지 않으면 '비용 축소 후보'로 적는다. 그때만 새 seed(7 이상, final seed 1~3 아님)에서 같은 비용 center(약 K=22, 추정 비용 ≈ 11.8)와 짝 비교를 등록한다. 아니면 '실패한 방향'으로 옮긴다. 이 변형은 구성요소를 더하지 않으므로 parent check의 '부모보다 높아야 함'은 목표가 아니다. 목표는 비용을 낮추고도 부모와 구별되지 않는 것이다. screen 시도 1회가 늘어난다(Search effort).
  - [claude] 라운드 57 갱신: QUEUE(2026-10-31 17:30·18:30)에 따라 suffdisc 계열(suffdisc, suffstride)은 위치 사전지식을 쓰므로 주장 대상이 아니다. H-W33과 H-W34의 결과는 기록만 하고, 위의 승격·후속 등록 규칙은 적용하지 않는다.

## 1단계 CUB-200-2011 (2026-10-31 전환 후)
- **H-C1 (분석 설계만, 미제출): 가린 teacher가 soft label의 클래스 관계를 얼마나 보존하는가.** 근거: Waterbirds에서 full teacher‖masked teacher KL이 student 순위를 잘 예측했다(analysis_proxy_validation_1030_170914_4668 , Spearman 0.937). 다만 그것은 WGA 기준이고 2클래스였다. CUB는 200클래스라서 soft label의 비정답 클래스 분포(Hinton 2015의 dark knowledge)가 KD 이득의 원천일 수 있다. 계획: CUB full-KD student와 maskedkd student가 생기면, 같은 K(예: 98·49·25·12)에서 선택기별(maskedkd, rollout, random, grid, center, tome) masked teacher에 대해 (i) 전체 KL, (ii) 정답 클래스를 뺀 분포의 KL, (iii) 정답 확률 차이를 잰다. 그리고 이 값들이 CUB screen 평균 정확도 순위와 맞는지 본다. 이 분석으로 "teacher 출력 보존"이 평균 정확도의 근거가 되는지 먼저 확인한다. 맞으면 다음 방법은 위치 사전지식 없이 student 내부 계산(토큰별 attention·기울기)으로 그 보존량을 예측하는 토큰별 점수로 설계한다. 이 경우 근거 → 방법 순서가 된다. 선행 기법과의 차이: MaskedKD는 CLS attention 크기만 쓰고, 고른 토큰이 teacher 출력을 얼마나 보존하는지는 재지 않는다. 노벨티는 문헌 확인 전이라 아직 주장하지 않는다. 제출 시점: CUB lr sweep과 기준선 run이 나온 뒤다. 그 전에는 비교할 student가 없다.
  - **H-C1a (사전 등록, 라운드 58 [claude], 제출 전 커밋): 기술 분석 analysis/cub_softlabel_fidelity.py.** 지금 있는 CUB student 두 개(cub_final_full_k196_s1_1030_153055_d87b 82.83, cub_final_maskedkd_k98_s1_1030_153055_cdbc 83.50)로 CUB val 600장에서 잰다. 점수는 모두 토큰별이고 student 계산에서 나온다(위치 사전지식 없음): maskedkd(마지막 블록 CLS attention), rollout, attnvnorm(CLS attention × ‖W_o v‖, Kobayashi 2020), gxi_top1(top-1 logit의 grad×input), gxi_dark(2~11위 logit logsumexp의 grad×input). 비교용: center·grid·random(신호 없음), oracle(teacher attention, 상한, 주장 금지). K = 98·49·25·12. 재는 값: KL(full‖masked)과 그 DKD 분해(TCKD + NCKD, Zhao 2022), top-1 일치, top-5 겹침, maskedkd와의 겹침.
    - 예측 P1: maskedkd의 KL에서 NCKD(비정답 클래스 부분)가 K=49·25·12에서 절반을 넘는다. 이것이 맞으면 "200클래스 soft label의 손실은 주로 dark knowledge에서 난다"가 다음 방법의 근거가 된다.
    - 예측 P2: K=25·12에서 gxi_dark의 NCKD가 maskedkd보다 낮다(두 student 모두).
    - 기각: P2가 두 student 모두에서 K=25·12 둘 다 틀리면 "비정답 logit 기울기 점수" 아이디어는 버린다. P1이 틀리면(TCKD가 대부분) 근거는 정답 확률 보존 쪽으로 바꾼다.
    - 한계: 이 분석은 기술(descriptive)이다. student 정확도와의 순위 상관은 CUB 기준선 run이 생긴 뒤에만 잴 수 있다. 그래서 이 결과만으로 방법을 주장하지 않는다. center가 KL이 낮게 나와도 그것은 위치 편향(QUEUE 17:30 (c))이고 비교용일 뿐이다. gxi 점수는 student backward가 한 번 더 필요하므로, 방법으로 쓰면 그 비용을 FLOPs에 넣어 같은 총비용 기준선과 비교해야 한다(타당성 요건 5).
    - [claude] 라운드 62 판정 (analysis_cub_softlabel_fidelity_1031_082711_df82 , 미리 정한 규칙대로): **P1 틀림.** maskedkd KL 중 NCKD 비율은 full196_s1 student에서 K=49 0.193/0.445 = 43%, K=25 25%, K=12 20%이다. maskedkd98_s1 student에서는 41% / 25% / 19%이다. 손실의 대부분은 TCKD(정답 확률)다. 규칙에 따라 다음 방법의 근거는 "정답 확률 보존" 쪽으로 바꾼다. **P2 맞음.** gxi_dark NCKD와 maskedkd NCKD는 K=25에서 0.312 대 0.460, 0.314 대 0.443이다. K=12에서는 0.589 대 0.763, 0.613 대 0.726이다. 두 student, 두 K 모두 gxi_dark가 낮다. 그래서 기각 조건에 걸리지 않는다.
    - 단서 (판정을 바꾸지는 않음): P2의 우위는 gxi_dark만의 것이 아니다. NCKD는 gxi_top1(K=25 0.308 / 0.296)과 rollout(0.324 / 0.348)도 비슷하게 낮다. 전체 KL도 rollout ≈ gxi_dark ≈ gxi_top1 < attnvnorm < maskedkd로, K=49·25·12의 두 student에서 같은 묶음이 나온다(kl_rank.json). rollout은 harness 기준선(CLAIM_REFS)이고 backward가 필요 없다. gxi 계열은 backward가 한 번 더 든다. 따라서 이 분석에서는 gxi 점수가 rollout보다 KD 신호를 더 보존한다는 근거가 없다. teacher 자신의 마지막 블록 CLS attention(oracle)도 rollout보다 KL이 높다(full196 K=49 0.375 대 0.276). Waterbirds에서 본 "마지막 층 attention은 나쁜 teacher 입력" 결과가 200클래스에서도 같은 방향으로 나온다.
    - [claude] 결정: gxi 계열을 방법으로 등록하지 않는다. P2는 통과했지만, 같은 KL을 내는 더 싼 기준선(rollout)이 이미 있다. 같은 총비용 비교(타당성 요건 5)에서 이길 근거가 없다. 다음 후보는 H-C2(아래)에 아이디어로만 적는다. CUB 기준선 run이 없으니 student 정확도와의 상관은 아직 잴 수 없다.
- **H-C2 (아이디어만, 미제출): rollout을 넘는 TCKD 보존 점수가 있는가.** 근거: H-C1a에서 CUB 손실은 주로 TCKD였다(정답 확률 p_true가 떨어짐, 예: maskedkd@25 0.374 대 full 0.801). rollout이 위치 사전지식 없는 기준선 중 가장 강하다. 그러니 새 점수는 rollout보다 같은 비용에서 p_true를 더 지켜야 의미가 있다. 후보 (미검증): (1) rollout을 앞쪽 일부 블록만으로 잘라 student forward 중간에서 고른다(비용 감소, 같은 총비용에서 K를 늘릴 여지). (2) 시간 앙상블: 앞 epoch의 같은 이미지 점수를 재사용해 선택 비용을 없앤다. 순서: 먼저 analysis로 p_true·TCKD를 rollout과 같은 K에서 비교하고, 규칙을 미리 커밋한 뒤 제출한다. student 정확도와 KL 순위의 상관은 CUB 기준선(rollout·maskedkd·random 등 @K)이 생긴 뒤에 잰다.
  - [claude] 라운드 63 갱신 (H-C2a 분석, 결과 전·제출 전에 등록): analysis/cub_tckd_scores.py. 후보 점수는 "class-specific patch evidence"다. student 마지막 블록의 patch 토큰을 student 자신의 final norm·분류 head에 넣고, student top-1 클래스의 logit을 점수로 쓴다(patch 토큰 logit lens, ViT판 CAM; Zhou et al. CVPR 2016, Chefer et al. CVPR 2021). 라벨·teacher·위치를 쓰지 않는다. 추가 비용은 이미지당 [196×192]·[192×200] 곱 하나(약 0.015 GFLOPs)로, student 7.64에 비해 작다. 비교 점수: patchcam(이것만), rollcam(log rollout + log softmax_j(cam), 두 분포의 곱), roll6(블록 6~11만의 rollout, 위 후보 (1)), maskedkd, rollout, random. K=98·49·25·12. student는 CUB final s1 run 전부(지금 ce·full@196·maskedkd@98).
    - 근거: H-C1a에서 CUB 손실의 대부분이 TCKD(정답 확률)였다. patch가 student의 예측 클래스를 얼마나 지지하는지는 정답 확률과 직접 연결된다. rollout은 클래스와 무관한 "어디를 보는가"다.
    - 예측 P1 (주): K=25와 K=12에서 rollcam의 TCKD가 rollout보다 낮고, 전체 KL도 rollout 이하다. 모든 student에서.
    - 기각: P1이 어느 student에서든 K=25 또는 K=12에서 틀리면 rollcam을 방법으로 등록하지 않는다. patchcam·roll6은 기술 결과로만 적는다. patchcam만 P1 조건을 모든 student에서 만족하면 그것을 후보로 적는다(사후 대체이므로 "사후"라고 표시한다).
    - 한계: 기술 분석이다. KL·TCKD 순위가 CUB student 정확도 순위를 예측한다는 검증은 아직 없다(Waterbirds의 Spearman 0.94는 WGA에 관한 것이다). P1이 맞아도 방법 주장은 CUB screen에서 rollout·maskedkd와 같은 총비용 비교로만 한다.
  - [claude] 라운드 65 H-C2a 판정 (analysis_cub_tckd_scores_1031_095253_a64e , student 3개: ce0_s1·full196_s1·maskedkd98_s1, val 600장): **P1은 틀렸다.** rollcam의 TCKD는 모든 student에서 K=25·12 모두 rollout보다 훨씬 높다(full196_s1 K=25 3.69 대 0.56, K=12 4.18 대 2.06). patchcam은 random보다도 나쁘다(full196_s1 K=25 KL 4.83, random 3.03, rollout 0.89). 그래서 규칙대로 rollcam·patchcam을 방법으로 등록하지 않는다. 해석(추정): DeiT-Ti 마지막 블록의 patch 토큰에 분류 head를 바로 씌우면, 그 logit은 teacher가 쓰는 물체 patch를 가리키지 않는다. patch 토큰 표현이 CLS의 head와 같은 공간에 있지 않을 수 있다.
    - 기술 결과(사전 등록한 점수가 아님): roll6(블록 6~11만의 rollout)은 student 3개 모두에서 K=49·25·12의 TCKD·KL이 rollout보다 낮다. 예: full196_s1 K=12 TCKD 1.67 대 2.06, KL 2.22 대 2.68, p_true 0.289 대 0.233. maskedkd98_s1 K=12 KL 2.25 대 2.59, ce0_s1 K=12 KL 2.25 대 2.70. K=98에서는 차이가 0.01 이하다. H-C1a 기준으로 CUB에서 center@25 KL은 1.38, center@12는 2.86(full196_s1)이다. 즉 rollout과 roll6 모두 center보다 낫다. 사후 결과라서 아래 H-C2b로 따로 확인한다.
- **H-C2b (라운드 65, 결과 전·제출 전에 등록): rollout은 어느 블록부터 곱해야 하는가. 학습 중 student에서도 같은가. 이득이 가운데 편향에서 오는가.** analysis/cub_rollout_depth.py.
  - 설계: rollout 시작 블록 s ∈ {0, 3, 6, 8, 9, 10, 11}(s=0이 harness rollout, s=11은 maskedkd와 순위가 같아서 확인용)을 K=98·49·25·12에서 잰다. student는 CUB final s1 run 전부(ce·full@196·maskedkd@98)의 최종 가중치와 e25·e50 snapshot이다. 선택기는 학습 중인 student에서 쓰이므로 snapshot도 본다. 위치 점검(타당성 요건): 점수마다 남긴 토큰이 중심에서 떨어진 평균 거리(0=중심, 1=모서리)와 center@K와 겹치는 비율을 잰다. center@K·random@K 충실도를 기준으로 넣는다. 점수에는 라벨·teacher·위치가 들어가지 않는다. 시작 블록 사이의 비용 차이는 이미지당 0.2 GFLOPs 미만이다.
  - 예측 P1 (주): K=25와 K=12에서 roll6의 TCKD가 roll0보다 낮고 KL도 roll0 이하다. 최종 가중치 3개와 snapshot 6개, 모든 student에서 그렇다.
  - 예측 P2 (위치): K=25와 K=12에서 roll6의 KL이 center@K보다 낮다. 최종 가중치 3개 모두에서 그렇다. centre_overlap(roll6 − roll0)은 기술값으로 적는다. 이 값이 +0.10을 넘으면 P1·P2가 맞아도 "가운데로 더 쏠린 결과일 수 있음"으로 적는다.
  - 기각·다음: P1이 snapshot을 포함해 어느 student에서든 틀리면 늦은 블록 rollout을 방법으로 쓰지 않는다. P1·P2가 맞으면, 최종 가중치 3개의 K=25·12 TCKD 평균이 가장 낮은 시작 블록 s*를 고른다. 그리고 rollout(s*)를 selectors/에 쓰고 CUB에서 rollout·maskedkd와 같은 총비용 비교를 사전 등록한다. 이것은 harness가 CUB에서 에이전트 방법을 받을 때에만 낸다(지금 되돌아간 harness에서는 CUB가 2단계라 막혀 있다).
  - 한계: 기술 분석이다. TCKD·KL이 CUB student 정확도를 예측하는지는 아직 검증되지 않았다. 늦은 블록 rollout은 rollout의 변형이므로, 노벨티는 작다. 이 분석의 가치는 같은 비용에서 어느 신호가 teacher soft label을 지키는지 근거를 쌓는 데 있다.
  - [claude] 라운드 65 H-C2b 판정 (analysis_cub_rollout_depth_1031_100143_a4b6 , 최종 가중치 3개 + e25·e50 snapshot 6개, val 600장): **P1은 맞았다.** K=25와 K=12에서 roll6의 TCKD는 roll0보다 낮고 KL도 roll0 이하다. 9개 student 모두 그렇다. **P2도 맞았다.** 최종 가중치 3개 모두에서 roll6의 KL이 center@K보다 낮다(full196_s1 K=25 0.810 대 1.379, K=12 2.225 대 2.860). centre_overlap(roll6 − roll0)은 9개 모두 −0.026~−0.067이다. 즉 roll6은 rollout보다 오히려 덜 가운데로 쏠린다. 위치 편향으로 설명되지 않는다. 확인용 s=11은 maskedkd와 값이 같다(full196_s1 K=25 KL 1.8228). 시작 블록 sweep은 U자 모양이다. 최종 가중치 3개의 K=25·12 TCKD 평균은 roll0 1.309, roll3 1.126, **roll6 1.092**, roll8, roll9, roll10, roll11 순서로 커진다(full196_s1 K=12 기준 1.88, 1.98, 2.25, 3.05). 규칙에 따라 s*=6이다. 해석(추정): 앞쪽 블록은 위치 섞기만 더하고, 마지막 블록만 쓰면 너무 뾰족하다.
- **H-C2c (라운드 66, 결과 전·제출 전에 등록): roll6의 TCKD 우위가 모든 주장 기준(CLAIM_REFS)에 대해, 짝지은 신뢰구간으로도 성립하는가.** analysis/cub_lateroll_refs.py.
  - 이유: H-C2b는 roll6을 roll0·center·random과 평균값으로만 비교했다. 주장 규칙은 grid·tome도 이겨야 한다. H-C3을 평균값 하나만 보고 내지 않으려고 이 분석을 한다.
  - 설계: CUB final s1 student 3개(ce·full@196·maskedkd@98), K=49·25·12. 이미지별 KL·TCKD를 잰다. 점수는 roll6, roll0(= harness rollout), roll11(= maskedkd 순위)이다. student와 무관한 기준은 grid·center·random·tome·oracle이다. tome은 harness/tome.py를 그대로 옮겼고, r은 r_for_budget 규칙으로 정한다. oracle은 teacher 자신의 마지막 블록 CLS attention이며, 상한 참고용이라 주장에 쓰지 않는다. 짝지은 bootstrap(val 이미지 600장, 2000회, 고정 index)으로 TCKD(roll6) − TCKD(기준)과 KL 차이의 평균 95% CI를 구한다. 기술값으로 headroom_closed = (roll0 − roll6)/(roll0 − oracle)도 적는다. H-C1a에서 oracle이 rollout보다 KL이 높았으므로, 이 값은 음수이거나 1을 넘을 수 있다.
  - 예측 P1 (주): K=25와 K=12에서, 3개 student 모두, roll6 − 기준의 TCKD 차이 CI 상한이 0보다 작다. 기준은 roll0·roll11·grid·center·random이다.
  - 예측 P2 (tome, 약한 예측): K=49에서 roll6의 TCKD CI 상한 < 0 (tome 대비), 3개 student 모두. K=25·12에서 tome은 같은 K의 masked teacher보다 계산을 더 받는다(cost_ratio > 1, summary에 기록). 그래서 예측은 하지 않고 값만 적는다.
  - 읽기: P1이 맞으면 H-C3의 근거에 "모든 위치 규칙·rollout·maskedkd보다 짝지은 CI로 낫다"를 더한다. P1이 어느 기준에서든 틀리면, 그 기준을 H-C3의 주된 위험으로 적는다. K=25에서 틀리면 H-C3의 K=25 예측을 철회한다. tome이 roll6보다 TCKD가 낮으면 같은 총비용 비교에서 tome이 가장 강한 상대라고 적는다.
  - 한계: 기술 분석이다. seed 1 student만 쓴다. CUB student 정확도와 TCKD의 관계는 여전히 검증되지 않았다.
  - [claude] 라운드 66 정정 (v2 제출 전): v1(analysis_cub_lateroll_refs_1031_101304_5f82 )의 CI와 verdict는 무효다. bootstrap index를 sin hash로 뽑았는데 고르게 퍼지지 않았다. 그래서 CI가 평균을 포함하지 않았다(예: ce0_s1 K=49 center, 평균 −0.204인데 CI [0.209, 0.296]). 평균값(means.json)은 bootstrap과 무관해서 유효하다. v2는 seed를 고정한 numpy 생성기로 index를 뽑고, CI가 평균을 포함하는지 assert로 확인한다. 예측·읽기 규칙은 바꾸지 않는다.
- **H-C3 (라운드 65 등록, 제출은 harness가 CUB에서 에이전트 방법을 받을 때): lateroll(블록 6~11 rollout, selectors/lateroll.py, parent rollout)이 CUB에서 같은 K의 rollout·maskedkd보다 평균 정확도가 높은가.**
  - 근거: H-C2b. 같은 K라면 teacher 비용은 같다. 선택 비용 차이는 이미지당 0.2 GFLOPs 미만이다. lateroll은 explicit-softmax 경로를 블록 6개에서만 써서 rollout보다 약간 싸다(측정값은 run의 GFLOPs로 확인).
  - 제출 계획: CUB에서 lateroll@49·@25를 harness가 허용하는 탐색 tier/seed(QUEUE 17:30 4번: screen seed 1·2·3 금지, 잡음 seed 4·5)로 낸다. 같은 seed의 rollout@K·maskedkd@K와 짝으로 비교한다.
  - 예측: K=25에서 lateroll − rollout 평균 정확도 차이 > 0, 그리고 K=49에서 ≥ −0.3. 시드 평균으로 본다. 판정 규칙: K=25에서 시드 평균 차이가 PROMOTE 문턱(max(0.5, 2σ)) 이상이면 final(seed 1·2·3)로 올린다. 0 이하면 기각하고 "teacher soft label 충실도 ≠ student 정확도"로 기록한다(이것도 중요한 결과다).
  - 지금 막힌 이유: 되돌아간 harness는 CUB를 2단계로 두므로(config.STAGE2 = "cub"), 에이전트 방법은 Waterbirds final + 감사 PASS 뒤에만 CUB에 낼 수 있다. 새 Waterbirds 실험은 QUEUE 18:30 5번이 막는다. 사람이 relock하면 바로 낸다.
