# 사람 지시 (매 라운드 최우선)


## 2026-10-31 17:30 KST — 타당성 있는 탐색으로 전환 (최우선)
사람 판단: 지금 최선인 suffdisc 계열(suffdisc, suffstride, attnwin)은 `center`를 부모로 하고, 가운데에서 최대 몇 칸만 움직이는 원판을 고른다. Waterbirds·CUB는 새가 대체로 가운데 있어서, 이 규칙은 메커니즘이 아니라 데이터셋의 위치 편향을 이용한다. 같은 총비용에서 center@49가 suffdisc@12를 이기는 것(H-W32)도 같은 이유로 보인다. 이런 방법은 일반성이 없고 노벨티가 없다.

1. AGENT_PROMPT의 새 절 "타당성 요건"과 AUDIT_PROMPT 16·17항을 먼저 읽어라 (이번 커밋에서 추가됨).
2. suffdisc 계열은 주장 대상에서 뺀다. 진행 중인 suffdisc·suffstride job은 끝까지 돌게 두고 기록만 한다. RESULTS.md "분석에서 배운 것"에 남길 것: (a) MaskedKD의 마지막 층 attention은 토큰을 줄이면 최악 그룹이 크게 무너진다 (WGA −40%p 수준), (b) 가린 teacher의 출력 충실도(KL)가 student WGA를 잘 예측한다 (analysis_proxy_validation, Spearman 0.94), (c) 위치를 고정한 원판 규칙이 이 데이터에서 강한 이유는 위치 편향이다.
3. 위 (a)(b)는 근거로 계속 써도 된다. 다음 방법은 위치 사전지식 없이 토큰별 점수로 만들고, 그 점수가 왜 teachability를 반영하는지 분석으로 먼저 보여라. 무엇을 할지는 네가 정한다.
4. harness 변경 (이번 커밋): 감사 증거에 전체 job 이력(모든 상태)이 들어간다 → STUCK이던 감사 요청을 다시 낼 수 있다(단, suffdisc 계열 주장은 2번에 따라 내지 마라). screen에서 seed 1·2·3은 이제 금지다(final과 같은 run이 되므로). 잡음 seed는 4·5다. CUB 관문은 "MaskedKD@98 ≥ Full − 0.5, 시드 평균"으로 바뀌어 2단계가 열릴 수 있다.

## 2026-10-31 18:30 KST — 계획 재정비: 탐색은 CUB, 판정은 평균 정확도 + FLOPs (위 17:30보다 우선)
사람 판단: 이 연구의 목표는 강건성이 아니라 **KD의 FLOPs 효율**이다. Waterbirds를 쓴 이유는 KD > CE를 찾기 쉬워서였을 뿐이다. 그런데 주 지표에 WGA가 들어가 있어서 연구가 "지름길 편향" 쪽으로 흘렀다. 또 2클래스 데이터는 teacher soft label에 클래스 관계 정보가 거의 없어서 KD 연구로 약하다.

이번 커밋으로 바뀐 것 (AGENT_PROMPT·AUDIT_PROMPT 갱신됨, 먼저 읽어라):
1. **1단계(탐색) = CUB-200-2011, 2단계(확인) = Waterbirds.** 서버가 CUB에서 student lr sweep → CUB 관문 → CUB 기준선을 자동으로 돌린다. 그동안 GPU가 차 있으니, 기준선이 나오기 전에는 분석과 설계에 집중해라.
2. **주 지표 = 평균 정확도(허용 오차 = KD 이득의 절반, 최대 1%p) + 그때의 학습 1스텝 총 GFLOPs.** WGA는 진단값으로만 표에 남는다. 주장은 **시드 3개 이상 평균**으로만 한다 (Primary metric의 claim-grade 줄).
3. **위 17:30 지시 중 유지:** 위치 사전지식 금지, 토큰별 점수, 근거 먼저, 사전 등록, 같은 총비용 비교. suffdisc 계열은 주장 대상이 아니다.
4. **위 17:30 지시 중 바뀐 것:** 근거로 쓰라고 한 (a)(b)는 WGA에 관한 것이라 주 질문의 근거가 아니다. 다음 방법의 근거는 "가린 teacher가 student에게 주는 KD 신호(soft label)를 얼마나 보존하는가"처럼 **평균 정확도와 FLOPs에 연결되는 것**이어야 한다. 무엇을 분석할지는 네가 정한다.
5. Waterbirds에서 끝난 run은 2단계 표에 그대로 남는다. 새 Waterbirds 실험은 2단계 규칙(관문 + CUB final + 감사 PASS) 뒤에만 낼 수 있다.

## 2026-10-31 20:40 KST — 방향 제안: multi-view 이론 + 사람의 이전 실험(TrustMask) (위 18:30 규칙 안에서)
사람 판단: 성능 비교만으로는 노벨티가 약하다. 이론에서 예측을 끌어내고 그것을 메커니즘 수준에서 시험하는 연구여야 한다. 방법은 CUB 전용이 아니라 어떤 데이터셋에서도 같은 규칙으로 돌아야 한다. 아래는 방향이지 지시가 아니다. 근거·사전 등록·같은 총비용 비교 규칙은 그대로다.

**1. 이론: Allen-Zhu & Li, "Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning" (ICLR 2023, arXiv 2012.09816).** 클래스마다 여러 구별 특징("뷰")이 있고, KD가 효과 있는 이유는 teacher soft label이 이미지에 어떤 뷰가 얼마나 보이는지 알려 주기 때문이라는 multi-view 가설. 이것을 teacher 토큰 선택에 옮기면 예측이 나온다:
- 토큰을 가리면 teacher가 보고하는 뷰가 지워진다 → 정답 확신(TCKD)이 떨어진다. 이것은 H-C1a의 "손실은 주로 TCKD"와 맞는다.
- 그러면 좋은 선택은 "가장 두드러진 K개"(maskedkd, rollout)가 아니라 **"증거가 있는 토큰 안에서 서로 다른 뷰를 덮는 K개"**여야 한다.
- 주의: conceptcov(H-W23)는 배경 개념까지 똑같이 덮어서 Waterbirds K=12에서 졌다. 이론은 "구별 특징"의 커버리지를 말한다. 배경을 같은 몫으로 덮으라는 뜻이 아니다. 이 차이가 다음 설계의 핵심이다.

**2. 사람의 이전 실험 (TrustMask, Waterbirds, 이 harness 밖에서 한 것이라 근거 참고용일 뿐 claim 아님).**
- 남긴 비율 0.15에서 MaskedKD는 random보다도 출력 충실도가 낮았고 (87.5 vs 89.2, rollout 93.5), student 성능도 무너졌다.
- "teacher가 무시하는 토큰을 rollout 선택에서 빼기"가 rollout을 같은 seed끼리 꾸준히 이겼다 (0.15에서 clean +0.61, 5/5 seed). 반대로 "teacher는 보지만 student가 놓친 토큰을 일부러 넣기"(gap)는 크게 졌다 (−5~−6).
- 해석: 남길 토큰의 질(증거가 있는가)이 다양성보다 먼저다. 그리고 student attention을 얼마나 믿을지는 student가 pretrain됐는지에 따라 달라진다.
- 여기서는 teacher 신호를 쓸 수 없다 (selector_api: student 쪽만). 그러니 "teacher가 무시하는 토큰"을 **student 계산으로 근사**해야 한다. 후보: 고노름 register성 토큰 (Darcet et al. ICLR 2024; H-W4/H-W5에서 이미 의심한 것), 예측 기여가 낮은 토큰.

**3. 시험할 만한 가설 (고르는 것은 네가 한다):**
- (a) 증거 필터 + 뷰 커버리지: rollout 등으로 증거가 있는 토큰을 먼저 거르고, 그 안에서 student 특징의 개념 커버리지로 K개를 고른다. 부모 = 필터만 쓴 쪽 (parent check로 커버리지가 뭔가를 더하는지 보인다).
- (b) 이미지별 예산: selector_api는 배치 평균 ≤ K만 지키면 이미지마다 다른 개수를 허용한다. 증거가 몇 토큰에 몰린 이미지는 적게, 퍼진 이미지는 많이 남긴다. 사람이 해 본 "배치를 집중도로 3등분해 k×0.67 / k / k×1.33"이 출발점이다.
- (c) 위 둘의 근거 분석을 먼저: 같은 K에서 "증거 안의 커버리지" 마스크가 "상위 K" 마스크보다 TCKD를 더 지키는지 analysis로 보인 다음에 학습한다.

**4. 메커니즘 검증 (분석용, 방법에는 쓰지 않음):** CUB에는 새 부위 15개의 위치 주석이 있다 (원본 압축의 `CUB_200_2011/parts/part_locs.txt`). "고른 토큰이 실제로 부위를 더 골고루 덮고, 그것이 TCKD 보존·student 정확도와 함께 움직이는가"를 보이면 multi-view 예측을 처음으로 시험한 것이 된다. 지금 데이터 준비는 부위 주석을 저장하지 않는다. 분석에 필요하면 PROGRESS에 "NEEDS-HUMAN: CUB part_locs"라고 적어라. harness는 사람이 고친다.

**5. 일반성:** 방법은 student 내부 신호만 쓰므로 데이터셋과 무관해야 한다. CUB는 메커니즘을 보이는 곳, Waterbirds는 고정한 방법을 다시 확인하는 곳이다. 두 데이터셋 모두 물체가 가운데 있다는 한계는 그대로 적는다.
