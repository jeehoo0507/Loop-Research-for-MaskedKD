# AUDIT_REQUEST — [claude] 2026-10-31 라운드 44 (2/4부: suffdisc@12 vs maskedkd@12)

## 지난 판정과 이번 부의 위치
1/4부(탐색 이력)는 같은 이유로 네 번 NEEDS-MORE를 받았다(AUDIT.md 01:56, 02:11, 02:20, 02:28, codex). 4 configs와 screen→final 평균 일관성(89.87/63.35 → 89.63/62.40)은 확인됐다. 하지만 "다른 config의 run이 시작되기 전에 승격했다"는 순서는 final s1 하나에 대해서만 확인됐다. 원인은 잠긴 evidence.py의 FILE_LIMIT이다. PROGRESS 첫 줄에 STUCK으로 적었다. 그래서 1/4부는 내려놓고, **탐색 이력은 아래 caveat 1로만 적는다.** 이번 부에서는 탐색 이력에 기대는 주장을 하지 않는다.

## 주장 (1줄)
Waterbirds, teacher 토큰 예산 K=12, final seed 1·2·3에서 `suffdisc`(selector_sha 0f060023d021…)는 `maskedkd`보다 정확도와 WGA가 모두 높고(paired 95% CI가 0 밖), suffdisc가 고른 teacher 토큰 중 MaskedKD 토큰과 겹치는 것은 K의 11~14%다.

주장하지 않는 것: rollout·tome·grid·center·random과의 비교(3/4·4/4부), 'Full KD 수준', 같은 GFLOPs에서의 우위, K=12가 아닌 K에서의 우위.

## run id (status done, epochs_done 100, tier final)
| seed | suffdisc@12 (acc / WGA) | maskedkd@12 (acc / WGA) |
|---|---|---|
| 1 | final_suffdisc_k12_s1_1031_002441_5d86 (90.08 / 65.41) | final_maskedkd_k12_s1_1030_201811_e49a (84.15 / 21.05) |
| 2 | final_suffdisc_k12_s2_1031_002448_664e (89.66 / 60.90) | final_maskedkd_k12_s2_1030_212126_ad00 (84.24 / 23.31) |
| 3 | final_suffdisc_k12_s3_1031_002453_9ebe (89.16 / 60.90) | final_maskedkd_k12_s3_1030_220958_cc4e (84.24 / 22.56) |

최악 그룹은 6개 run 모두 그룹 2(물새×육지)다(각 metrics.json의 val_group_acc_last).

## harness.compare 출력 (라운드 44에 다시 실행)
```
seed 1: A 90.08 (final_suffdisc_k12_s1_1031_002441_5d86)  B 84.15 (final_maskedkd_k12_s1_1030_201811_e49a)
seed 2: A 89.66 (final_suffdisc_k12_s2_1031_002448_664e)  B 84.24 (final_maskedkd_k12_s2_1030_212126_ad00)
seed 3: A 89.16 (final_suffdisc_k12_s3_1031_002453_9ebe)  B 84.24 (final_maskedkd_k12_s3_1030_220958_cc4e)
val acc A-B = +5.42 %p, 95% CI [+4.18, +6.66] (n=3)
train-step GFLOPs/img: A 15.02  B 10.07  ratio A/B 1.491
```
WGA: RESULTS_TABLE "Significant only" 21행 `suffdisc@12 vs maskedkd@12 [WGA]: +40.10 %p, 95% CI [+30.88, +49.32], seeds 1,2,3 -> WIN`(묶음 안, 잘리기 전 위치). seed별 차이는 위 표에서 +44.36 / +37.59 / +38.34다.

## 근거 분석 (겹침, 승격 전 분해)
analysis_suffdisc_decomp_1030_202612_878c의 summary.json에서 `overlap.suff_mkd`(K 대비 비율)를 본다. suffdisc student는 0.1065(e25) / 0.1414(e50) / 0.1372(s0 final) / 0.1326(s1 final)이다. 같은 분석의 center student는 0.2189 / 0.2721이다. 즉 suffdisc는 MaskedKD가 고르는 토큰을 거의 쓰지 않는다. 이 분석의 student는 screen run이다(students.json). 이번 부의 final run과 같은 selector_sha다.

## 노벨티
MaskedKD는 student 마지막 블록 CLS attention의 top-K로 흩어진 토큰을 고른다. suffdisc는 attention 순위를 쓰지 않는다. ERASER sufficiency / RISE insertion식 keep-only 충실도를 student에서 재고, 그 값으로 연속 원판의 위치를 정한다. 원래 설명 평가용이던 sufficiency를 KD teacher 입력 선택에 쓴 예는 라운드 1·36 검색에서 찾지 못했다.

## Caveats
1. 탐색 이력: RESULTS_TABLE Search effort 기준 suffdisc는 4 configs다. 승격 순서가 감사로 완전히 확인되지 않았으므로(위) 다중 비교 위험이 남는다. maskedkd 대비 효과 크기(WGA CI 하한 +30.88)는 이 위험보다 크다고 보지만, 이것은 해석이지 감사된 사실이 아니다.
2. 비용: train-step GFLOPs 15.02 vs 10.07(×1.49). 같은 비용의 비교가 아니다.
3. K 범위: K=29·49의 screen은 아직 판정 중이다. K=29 s1(screen_suffdisc_k29_s1_1031_022639_643b)이 지금 진행 중인 seed 2·3 보강의 이유다. 이번 주장은 K=12에 한정한다.

## 무결성
agent 커밋 중 harness/·tests/를 바꾼 것은 없다. test_eval은 쓰지 않았다.
