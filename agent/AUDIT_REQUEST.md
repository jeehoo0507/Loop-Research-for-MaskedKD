# AUDIT_REQUEST — [claude] 2026-11-03 라운드 304 (P1 재요청: 사람이 증거 묶음을 고침, 064cef4)

## 주장 (1줄, 음성 결과)
P1 — Waterbirds final K=29, seed 1·2·3: trustmask_v1 − teacher_topk val WGA 같은 seed 짝 차이 = −2.01 [−6.32, +2.31] (95% t CI, df 2). v1의 teacher_topk 대비 우위는 보이지 않았다. 차이는 불확실하다: teacher_topk가 v1과 같다는 것도, 낫다는 것도 주장하지 않는다.

## 지난 감사에 대한 답
- 19:15 NEEDS-MORE의 이유(묶음 metrics가 잘려 run별 selector_sha가 감사자에게 안 보임)는 사람이 harness/evidence.py를 고쳐(064cef4) 해결했다: 묶음의 run별 요약에 selector_sha(metrics와 job 스냅샷), WGA, FLOPs가 잘림 앞에 들어간다. 아래 표는 그 요약과 맞춰 보기 위한 것이다.
- 감사 기준 (QUEUE 2026-11-04 08:50 KST, 사람): 게이트 1의 주 지표는 WGA, 사전 등록 P1·P2와 사후 관찰(v0)을 구분한다. 이 요청은 등록 P1 하나만이다. 사후 관찰 v0는 주장하지 않는다.

## run id (6개)
| run | val WGA | val acc | selector_sha |
|---|---|---|---|
| waterbirds_final_trustmask_v1_k29_s1_1103_170855_2e3c | 61.65 | 90.24 | 8186a44460ef2bea84605cb8efb46ec07611a59236d995019d79ed28a2520c8e |
| waterbirds_final_trustmask_v1_k29_s2_1103_170951_279c | 60.90 | 90.24 | 8186a44460ef2bea84605cb8efb46ec07611a59236d995019d79ed28a2520c8e |
| waterbirds_final_trustmask_v1_k29_s3_1103_170956_fdca | 64.66 | 91.33 | 8186a44460ef2bea84605cb8efb46ec07611a59236d995019d79ed28a2520c8e |
| waterbirds_final_teacher_topk_k29_s1_1103_170914_f015 | 64.66 | 89.74 | builtin |
| waterbirds_final_teacher_topk_k29_s2_1103_170918_4535 | 63.91 | 89.49 | builtin |
| waterbirds_final_teacher_topk_k29_s3_1103_170923_d12a | 64.66 | 89.91 | builtin |

짝 차이 (v1 − teacher_topk, val_wga_last): −3.01, −3.01, 0.00 → 평균 −2.01, sd 1.738, se 1.003, t(0.975, 2) = 4.303 → [−6.32, +2.31].
clean acc (harness.compare): v1 − teacher_topk +0.89 [−0.29, +2.07] (차이 불확실), 총 GFLOPs/img 13.35 vs 13.36 (비율 0.999).

## 근거 분석
없음. 사전 등록 판정(PROPOSALS H-A37, 등록 커밋 12c8069)을 그대로 한 번 적용한 결과다. 메커니즘을 주장하지 않는다.

## 노벨티
v1 우위가 보이지 않으므로 노벨티 근거 없음. 가장 가까운 선행: MaskedKD (Son et al., ECCV 2024), attention rollout (Abnar & Zuidema 2020).

## caveats
- 주 지표는 QUEUE 게이트 1 규칙의 WGA다(AUDIT_PROMPT의 평균 정확도 주 지표와 다르다; 사람이 08:50 KST에 게이트 판정의 감사 기준을 WGA로 정했다). 등록 문구는 "test WGA"였으나 val WGA로 판정했다(harness.test_eval은 연구자가 쓰지 않는다).
- 결과를 보고 seed·K·방법을 더하지 않았다. v1의 sel_args는 Aircraft에서 정한 그대로다.
