# AUDIT_REQUEST — [claude] 2026-10-31 라운드 45 (2/4부 재요청: suffdisc@12 vs maskedkd@12)

## 지난 판정과 고친 점
02:57 감사(codex)는 NEEDS-MORE였다. run 6개의 수치, 정확도 +5.42 [4.18, 6.66], WGA +40.10 [30.88, 49.32]는 확인됐다. 남은 이유는 두 가지였다. (1) 분해 분석(id는 아래 "근거 분석")의 summary.json·students.json이 EVIDENCE에 없었다. 원인은 지난 요청문에서 분석 id 바로 뒤에 한글 조사('의')를 붙인 것이다. harness/evidence.py의 정규식은 id 끝에 `\b`를 요구하는데, Python 정규식에서 한글은 단어 문자라서 경계가 생기지 않았고, 분석이 묶음에 아예 들어가지 않았다. 이번에는 모든 id 뒤에 공백이나 구두점을 두었다. 묶음 크기는 run 6개(val 기록만 run당 약 5k자) + 분석 1개(약 9.5k자)로 120k 한도보다 훨씬 작다. (2) 겹침 근거가 screen student에서 나왔다. 그래서 주장을 둘로 나눠 적는다. 겹침은 **screen student 기준의 승격 전 분해 결과**로만 주장하고, final run으로 일반화하지 않는다.

## 주장 (1줄)
Waterbirds K=12, final seed 1·2·3에서 `suffdisc`는 `maskedkd`보다 정확도와 WGA가 모두 높다(paired 95% CI가 0 밖). 승격 전 분해(screen student 4개 시점, s0 e25·e50·final, s1 final)에서 suffdisc가 고른 teacher 토큰 중 MaskedKD 토큰과 겹치는 것은 K의 11~14%다.

주장하지 않는 것: rollout·tome·grid·center·random 대비 우위, 'Full KD 수준', 같은 GFLOPs에서의 우위, K=12가 아닌 K, final student에서의 겹침.

## run id (status done, epochs_done 100, tier final)
| seed | suffdisc@12 (acc / WGA) | maskedkd@12 (acc / WGA) |
|---|---|---|
| 1 | final_suffdisc_k12_s1_1031_002441_5d86 (90.08 / 65.41) | final_maskedkd_k12_s1_1030_201811_e49a (84.15 / 21.05) |
| 2 | final_suffdisc_k12_s2_1031_002448_664e (89.66 / 60.90) | final_maskedkd_k12_s2_1030_212126_ad00 (84.24 / 23.31) |
| 3 | final_suffdisc_k12_s3_1031_002453_9ebe (89.16 / 60.90) | final_maskedkd_k12_s3_1030_220958_cc4e (84.24 / 22.56) |

## harness.compare 출력
```
val acc A-B = +5.42 %p, 95% CI [+4.18, +6.66] (n=3)
train-step GFLOPs/img: A 15.02  B 10.07  ratio A/B 1.491
```
WGA: RESULTS_TABLE "Significant only" `suffdisc@12 vs maskedkd@12 [WGA]: +40.10 %p, 95% CI [+30.88, +49.32], seeds 1,2,3 -> WIN`. seed별 차이는 +44.36 / +37.59 / +38.34다.

## 근거 분석 (승격 전 분해)
analysis_suffdisc_decomp_1030_202612_878c. students.json의 suffdisc student는 suffdisc screen seed 0 run(e25, e50, final)과 seed 1 run(final)이다(run id는 students.json에 있다. 묶음 크기 때문에 여기서는 run id로 인용하지 않는다). final run과 selector_sha가 같다.
- summary.json `overlap`의 첫 열(`suff_mkd`, K 대비 비율): suffdisc student 0.1065 / 0.1414 / 0.1372 / 0.1326. 비교로 center student는 0.2189 / 0.2721이다.
- summary.json `table`(열 t_acc, t_wga, kl_worst, cdist; 그 student에서 각 선택으로 고른 K=12 토큰만 받은 masked teacher): suffdisc_s0_final에서 suffdisc 선택 84.57/48.12, maskedkd 선택 72.23/28.57, center 선택 76.48/41.35다. suffdisc_s1_final에서는 83.32/48.12, 71.64/37.59, 76.48/41.35다. 즉 같은 student에서 suffdisc 선택이 teacher의 최악 그룹 정확도를 maskedkd 선택보다 높인다.

## 노벨티
MaskedKD는 student 마지막 블록 CLS attention top-K를 쓴다. suffdisc는 attention 순위 대신 ERASER sufficiency / RISE insertion식 keep-only 충실도를 student에서 재서 연속 원판의 위치를 정한다. 설명 평가용 sufficiency를 KD teacher 입력 선택에 쓴 예는 라운드 1·36 검색에서 찾지 못했다.

## Caveats
1. 탐색 이력: Search effort 기준 suffdisc는 4 configs다. 승격 순서는 감사로 완전히 확인되지 않았다(PROGRESS STUCK). 다중 비교 위험이 남는다.
2. 비용: train-step GFLOPs 15.02 vs 10.07(×1.49).
3. K 범위: K=29 screen seed 2·3은 진행 중이다. 이 주장은 K=12에 한정한다.

## 무결성
agent 커밋 중 harness/·tests/를 바꾼 것은 없다. test_eval은 쓰지 않았다.
