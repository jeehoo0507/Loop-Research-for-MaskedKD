# 감사 요청 (라운드 281, [claude]) — 게이트 1 (H-A37, 사전 등록 커밋 12c8069)

주장 1 (P2): Waterbirds final K=29 seed 1–3: trustmask_v1 val WGA 62.40 vs maskedkd 38.59, 같은 seed 짝 차이 +23.81 [+22.73, +24.88] (95% t CI, df 2), 총 13.35 vs 12.99 GFLOPs/img (비율 1.028).
주장 2 (P1, 음성 결과): 같은 설정에서 trustmask_v1 − teacher_topk val WGA = −2.01 [−6.32, +2.31]. 평균이 0 이하이므로 사전 등록대로 "teacher가 고르는 뻔한 수정도 된다, 노벨티 없음"으로 적는다. v1이 teacher_topk보다 낫다고 주장하지 않는다.

## run id (10개)
- trustmask_v1: waterbirds_final_trustmask_v1_k29_s1_1103_170855_2e3c , waterbirds_final_trustmask_v1_k29_s2_1103_170951_279c , waterbirds_final_trustmask_v1_k29_s3_1103_170956_fdca
- maskedkd: waterbirds_final_maskedkd_k29_s1_1031_201502_4be8 , waterbirds_final_maskedkd_k29_s2_1103_170931_92bd , waterbirds_final_maskedkd_k29_s3_1103_170935_907a
- teacher_topk: waterbirds_final_teacher_topk_k29_s1_1103_170914_f015 , waterbirds_final_teacher_topk_k29_s2_1103_170918_4535 , waterbirds_final_teacher_topk_k29_s3_1103_170923_d12a

## 숫자 (metrics.json `val_wga_last`, 손 계산)
| seed | v1 | maskedkd | teacher_topk | v1 − maskedkd | v1 − teacher_topk |
|---|---|---|---|---|---|
| 1 | 61.65 | 37.59 | 64.66 | +24.06 | −3.01 |
| 2 | 60.90 | 37.59 | 63.91 | +23.31 | −3.01 |
| 3 | 64.66 | 40.60 | 64.66 | +24.06 | 0.00 |
| 평균 | 62.40 | 38.59 | 64.41 | +23.81 (sd 0.433) | −2.01 (sd 1.738) |

CI = 평균 ± 4.303 · sd/√3.

## harness.compare 출력 (clean val acc; compare는 WGA를 내지 않는다)
```
v1 vs maskedkd:     val acc A-B = +2.86 %p, 95% CI [+2.27, +3.46] (n=3); GFLOPs/img A 13.35 B 12.99 ratio 1.028
v1 vs teacher_topk: val acc A-B = +0.89 %p, 95% CI [-0.29, +2.07] (n=3); GFLOPs/img A 13.35 B 13.36 ratio 0.999
```

## 근거 분석
없음. 게이트 1은 사전 등록된 판정이다. 메커니즘 분석(QUEUE 01:40 "규칙"의 epoch 스냅샷 분석)은 GPU가 빈 뒤에 한다. 이 요청은 메커니즘을 주장하지 않는다.

## 노벨티
P1이 거짓이라 노벨티 근거는 없다. MaskedKD(Son et al., ECCV 2024)의 student attention 선택이 Waterbirds 소수 그룹에서 무너진다는 것만 주장한다. attention rollout(Abnar & Zuidema 2020, 기록 v1 − rollout +0.25 [−3.65, +4.14])이나 teacher 지도 top-K로 고르면 같은 수준으로 회복된다.

## caveats
- 주 지표는 QUEUE 11-04 01:40 KST 게이트 1 규칙대로 WGA다(이전 AGENT_PROMPT의 평균 정확도 주 지표와 다르다). val split이다(Waterbirds val의 최소 그룹 크기로 WGA 단위가 약 0.75 %p).
- 총비용이 같지 않다: v1은 teacher 캐시 상각과 selector가 포함되어 maskedkd보다 2.8% 크다.
- maskedkd seed 1과 rollout seed 1은 10-31 run이다(같은 seed, 이전 digest일 수 있다; 묶음의 past locks 확인 부탁).
- test_eval 미사용: harness/test_eval.py 6행 "Writes final/test_<timestamp>.json. Never writes into runs/." 36행이 그 파일을 쓴다. `ls final` → "No such file or directory". `git log --all --oneline -- final` 출력 없음. run 디렉터리에는 log.txt·metrics.json·stdout.txt·student*.pt만 있다.
- 탐색 노력: Waterbirds에서 v1은 sel_args 바꾸지 않음(Aircraft에서 정한 그대로). 결과를 보고 seed·K·방법을 더하지 않았다.
