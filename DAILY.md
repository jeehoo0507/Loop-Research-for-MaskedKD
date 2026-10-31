# DAILY

## 2026-10-31 [claude]
1. 탐색을 CUB로 옮기고 주 지표를 평균 정확도 + GFLOPs로 바꿨다(QUEUE 18:30). suffdisc 계열은 위치 편향이라 주장 대상에서 뺐다.
2. CUB screen seed 0·4·5 평균에서 98보다 작은 K로 허용 오차(문턱 81.50) 안에 든 것은 lateroll@49(82.11) 하나다. rollout@49 81.28, maskedkd@49 80.78은 밖이다. Parent check 실패라 승격하지 않았다.
3. 같은 K 안에서 가린 teacher TCKD는 student 정확도 순위와 맞는다(H-C4). 하지만 토큰 한계 기여, 특징 새로움, 이미지별 K 배분, 확신-오답, sink 제외, 고정 성분 설명(H-C5~C13)은 모두 기각됐다.
4. 작은 K(12)에서는 모든 선택기가 CE보다 낮다. 가린 teacher가 student를 해친다.
5. 20:21 UTC부터 GPU 스케줄러가 꺼져 H-C14(teacher 지도 거르기)와 기준선이 돌지 못했다.
- 내일(11-01) 계획: 스케줄러가 돌아오면 H-C14 판독 → trustfilt 1개 → H-C15(지도 번짐·뭉침) 제출. H-C15 P3가 맞으면 이웃 억제 선택기를 사전 등록하고 노벨티(다양성 기반 가지치기 문헌)를 확인한다.

## 2026-10-30 [claude]
1. MaskedKD(마지막 블록 CLS attention top-K)는 K가 작을수록 최악 그룹(육지 배경의 물새)을 크게 잃는다. K=12에서 WGA 23.31로 random(33.83)보다 낮다 (screen seed 0).
2. 신호 없는 center가 K=12에서 강한 기준이다(2-seed 54.14). K=29·49에서는 rollout이 가장 높다.
3. teacher 충실도 계열 대리 지표는 '그 선택기로 학습한 student'로 재야 하고, 그래도 K=29에서는 순위를 틀린다. 학습 동역학(선택기↔student 되먹임)이 핵심 변수다.
4. gradroll@12(기울기 가중 rollout)는 seed 0 screen 60.90, final seed 1 63.16으로 maskedkd·rollout을 크게 이긴다. 이득은 MaskedKD 밖의 gradroll 고유 토큰에서 온다(gradrep 대체 실험). tome 대비 정확도 손실이 약점이다.
5. suffdisc@12(student sufficiency로 원판 위치 선택)는 2-seed로 center를 +7.14 앞서고, 무작위 위치(jitdisc)는 center 수준이다. 이득은 위치 신호에서 온다.
- 내일(10-31) 계획: center final s1로 gradroll seed 1 판정 → seed 2·3 final. 끝나면 Core comparison을 보고 AUDIT_REQUEST 작성. 그 뒤 suffdisc final 여부를 정하고, 노벨티 검색(기울기 가중 relevance·sufficiency × KD teacher 입력)을 한다.
