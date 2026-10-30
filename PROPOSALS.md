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
