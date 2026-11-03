# AUDIT_REQUEST — [claude] 2026-11-03 라운드 282 (11-03 18:49 감사 P1 NEEDS-MORE 재요청)

## 주장 (1줄, 음성 결과)
P1 — Waterbirds final K=29, seed 1·2·3: trustmask_v1 − teacher_topk val WGA 같은 seed 짝 차이 = −2.01 [−6.32, +2.31] (95% t CI, df 2). 우위도 노벨티도 주장하지 않는다: teacher 지도 top-K로 고르는 뻔한 수정이 v1과 같거나 낫다.

## 지난 감사에 대한 답
- P1 NEEDS-MORE("truncated metrics omit the per-run selector_sha"): 아래에 run마다 selector_sha를 적었다. metrics.json의 selector_sha와 queue/done/<id>.json의 selector_sha가 같다(Grep으로 확인).
- P2는 이번에 다시 내지 않는다. CI 상한 반올림 오류(+24.88 → +24.89)는 RESULTS.md·PROPOSALS에서 고쳤다. "WGA 단독은 이 연구 질문의 주장이 아니다"(AUDIT_PROMPT 주 지표 문장)는 QUEUE 11-04 01:40 KST의 새 질문(주 지표 WGA)과 충돌하므로 사람 결정을 기다린다.

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
clean acc (harness.compare): v1 − teacher_topk +0.89 [−0.29, +2.07], 총 GFLOPs/img 13.35 vs 13.36 (비율 0.999).

## 근거 분석
없음. 사전 등록 판정(PROPOSALS H-A37, 등록 커밋 12c8069)을 그대로 한 번 적용한 결과다. 메커니즘을 주장하지 않는다.

## 노벨티
P1이 거짓이므로 노벨티 근거 없음. 가장 가까운 선행: MaskedKD (Son et al., ECCV 2024), attention rollout (Abnar & Zuidema 2020).

## caveats
- 주 지표는 QUEUE 11-04 01:40 KST 게이트 1 규칙의 WGA다. 등록 문구는 "test WGA"였으나 val WGA로 판정했다(harness.test_eval은 연구자가 쓰지 않는다). `ls final` 없음.
- 결과를 보고 seed·K·방법을 더하지 않았다. v1의 sel_args는 Aircraft에서 정한 그대로다.
