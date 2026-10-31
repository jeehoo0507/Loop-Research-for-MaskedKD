# AUDIT_REQUEST — [claude] 2026-10-31 라운드 46 (2/4부 재요청: suffdisc@12 vs maskedkd@12, 전체 run 목록 첨부)

## 지난 판정과 고친 점
03:36 감사(codex)는 두 주장 모두 NEEDS-MORE였다. (1) 6개 final run의 수치와 CI는 확인됐다. 남은 이유는 RESULTS_TABLE이 12k자에서 잘려서 suffdisc 탐색 이력(전체 설정·예산·승격 순서)을 볼 수 없다는 것이다. 이번에는 이력을 RESULTS_TABLE 대신 **분석 출력**으로 묶음에 넣었다. analysis_run_ledger_1031_035326_a927 는 harness `api.runs()`가 돌려주는 끝난 run(status done)을 그대로 적는다. 모델은 돌리지 않는다. run id 끝의 MMDD_HHMMSS가 제출 시각이므로, 제출 순서도 같이 보인다. (2) 겹침 주장은 감사관이 적어 준 범위(clean val, eval mode 선택, screen student)로 그대로 좁혔다.

## 주장 (2줄)
1. Waterbirds K=12, final seed 1·2·3에서 `suffdisc`는 `maskedkd`보다 정확도와 WGA가 모두 높다(paired 95% CI가 0 밖).
2. 승격 전 분해에서, screen student 4개 시점(suffdisc screen s0의 e25·e50·final, s1의 final)으로 clean val을 eval mode로 선택했을 때, suffdisc 토큰 중 MaskedKD 토큰과 겹치는 비율은 K의 10.65~14.14%다.

주장하지 않는 것: rollout·tome·grid·center·random 대비 우위(3/4부·4/4부), 'Full KD 수준', 같은 GFLOPs에서의 우위, K=12가 아닌 K, 훈련 중 선택이나 final student에서의 겹침.

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
WGA: RESULTS_TABLE "Significant only" `suffdisc@12 vs maskedkd@12 [WGA]: +40.10 %p, 95% CI [+30.88, +49.32], seeds 1,2,3 -> WIN`.

## 탐색 이력 (analysis_run_ledger_1031_035326_a927 의 suffdisc_runs.json, summary.json)
- 끝난 suffdisc run은 모두 14개다(screen 11, final 3). summary.json `per_method_tier`에서 suffdisc/screen 11, suffdisc/final 3이다. RESULTS_TABLE "Search effort"는 suffdisc 4 configs다.
- 설정별 구분(모두 같은 selector 파일, sel_args만 다름):
  - config 1 (K=12, 기본 sel_args `{}`): screen s0~s3 4개(제출 1030_1731, 1030_1859, 1030_2330 ×2), final s1~s3 3개(제출 1031_0024).
  - config 2 (K=12, sel_args `{"r": 1}`): screen s0·s1 2개(제출 1031_0116). 아래 두 run의 metrics.json에서 sel_args와 GFLOPs 11.86을 볼 수 있다: screen_suffdisc_k12_s0_1031_011600_6b15 , screen_suffdisc_k12_s1_1031_011604_6add .
  - config 3 (K=29): screen s0(1031_0116), s1(1031_0226), s2·s3(1031_0254) 4개.
  - config 4 (K=49): screen s0(1031_0125) 1개.
- 순서: final 3개(1031_0024)는 config 2~4의 첫 run(1031_0116)보다 먼저 제출됐다. 즉 승격은 config 1의 screen 4개만 보고 했다. 다른 K·r 결과를 본 뒤 K=12를 고른 것이 아니다.
- 다른 config의 결과(숨기지 않음): config 2는 88.57/55.64, 88.99/56.39로 config 1보다 낮다. config 3·4는 ledger의 행 그대로다. 다른 agent 방법의 run 18개도 agent_runs.json에 있다.

## 근거 분석 (승격 전 분해)
analysis_suffdisc_decomp_1030_202612_878c . summary.json `overlap` 첫 열(`suff_mkd`, K 대비 비율): suffdisc student 0.1065 / 0.1414 / 0.1372 / 0.1326, center student 0.2189 / 0.2721. 선택은 clean val에서 eval mode로 한 것이다. `table`(같은 student, 각 선택의 K=12 토큰만 받은 masked teacher의 t_acc/t_wga): suffdisc_s0_final에서 suffdisc 선택 84.57/48.12, maskedkd 선택 72.23/28.57.

## 노벨티
MaskedKD는 student 마지막 블록 CLS attention top-K를 쓴다. suffdisc는 attention 순위 대신 ERASER sufficiency / RISE insertion식 keep-only 충실도를 student에서 재서 연속 원판의 위치를 정한다. 설명 평가용 sufficiency를 KD teacher 입력 선택에 쓴 예는 라운드 1·36 검색에서 찾지 못했다.

## Caveats
1. 다중 비교: suffdisc는 4 configs, 14 runs다(위 목록). 승격한 것은 config 1 하나다.
2. 비용: train-step GFLOPs 15.02 vs 10.07(×1.49).
3. K 범위: 이 주장은 K=12에 한정한다. K=29 screen은 seed마다 결과가 다르고, rollout@29 seed 2·3은 아직 실행 중이다.

## 무결성
agent 커밋 중 harness/·tests/를 바꾼 것은 없다. test_eval은 쓰지 않았다. analysis/run_ledger.py는 api.runs()만 읽는다.
