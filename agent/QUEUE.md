# 사람 지시 (매 라운드 최우선)


## 2026-10-31 17:30 KST — 타당성 있는 탐색으로 전환 (최우선)
사람 판단: 지금 최선인 suffdisc 계열(suffdisc, suffstride, attnwin)은 `center`를 부모로 하고, 가운데에서 최대 몇 칸만 움직이는 원판을 고른다. Waterbirds·CUB는 새가 대체로 가운데 있어서, 이 규칙은 메커니즘이 아니라 데이터셋의 위치 편향을 이용한다. 같은 총비용에서 center@49가 suffdisc@12를 이기는 것(H-W32)도 같은 이유로 보인다. 이런 방법은 일반성이 없고 노벨티가 없다.

1. AGENT_PROMPT의 새 절 "타당성 요건"과 AUDIT_PROMPT 16·17항을 먼저 읽어라 (이번 커밋에서 추가됨).
2. suffdisc 계열은 주장 대상에서 뺀다. 진행 중인 suffdisc·suffstride job은 끝까지 돌게 두고 기록만 한다. RESULTS.md "분석에서 배운 것"에 남길 것: (a) MaskedKD의 마지막 층 attention은 토큰을 줄이면 최악 그룹이 크게 무너진다 (WGA −40%p 수준), (b) 가린 teacher의 출력 충실도(KL)가 student WGA를 잘 예측한다 (analysis_proxy_validation, Spearman 0.94), (c) 위치를 고정한 원판 규칙이 이 데이터에서 강한 이유는 위치 편향이다.
3. 위 (a)(b)는 근거로 계속 써도 된다. 다음 방법은 위치 사전지식 없이 토큰별 점수로 만들고, 그 점수가 왜 teachability를 반영하는지 분석으로 먼저 보여라. 무엇을 할지는 네가 정한다.
4. harness 변경 (이번 커밋): 감사 증거에 전체 job 이력(모든 상태)이 들어간다 → STUCK이던 감사 요청을 다시 낼 수 있다(단, suffdisc 계열 주장은 2번에 따라 내지 마라). screen에서 seed 1·2·3은 이제 금지다(final과 같은 run이 되므로). 잡음 seed는 4·5다. CUB 관문은 "MaskedKD@98 ≥ Full − 0.5, 시드 평균"으로 바뀌어 2단계가 열릴 수 있다.
