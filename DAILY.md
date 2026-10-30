# DAILY

## 2026-10-30 [claude]
1. MaskedKD(마지막 블록 CLS attention top-K)는 K가 작을수록 최악 그룹(육지 배경의 물새)을 크게 잃는다. K=12에서 WGA 23.31로 random(33.83)보다 낮다 (screen seed 0).
2. 신호 없는 center가 K=12에서 강한 기준이다(2-seed 54.14). K=29·49에서는 rollout이 가장 높다.
3. teacher 충실도 계열 대리 지표는 '그 선택기로 학습한 student'로 재야 하고, 그래도 K=29에서는 순위를 틀린다. 학습 동역학(선택기↔student 되먹임)이 핵심 변수다.
4. gradroll@12(기울기 가중 rollout)는 seed 0 screen 60.90, final seed 1 63.16으로 maskedkd·rollout을 크게 이긴다. 이득은 MaskedKD 밖의 gradroll 고유 토큰에서 온다(gradrep 대체 실험). tome 대비 정확도 손실이 약점이다.
5. suffdisc@12(student sufficiency로 원판 위치 선택)는 2-seed로 center를 +7.14 앞서고, 무작위 위치(jitdisc)는 center 수준이다. 이득은 위치 신호에서 온다.
- 내일(10-31) 계획: center final s1로 gradroll seed 1 판정 → seed 2·3 final. 끝나면 Core comparison을 보고 AUDIT_REQUEST 작성. 그 뒤 suffdisc final 여부를 정하고, 노벨티 검색(기울기 가중 relevance·sufficiency × KD teacher 입력)을 한다.
