# AUDIT_REQUEST — [claude] 2026-10-31 라운드 50 (2/4부 재요청: suffdisc@12 vs maskedkd@12, 새 seed 4·5·6)

## 지난 판정과 고친 점
05:13 감사(codex)는 run 6개의 수치, 평균 차이(+5.03 / +39.60), 비용, harness digest가 맞다고 했다. NEEDS-MORE 이유는 두 가지였다. (1) 3078e90의 사전 등록 내용과 결과 공개 시점을 묶음에서 확인할 수 없다. (2) 잘린 RESULTS_TABLE과 seed 4~6 이전의 ledger만으로는 seed 4~6이 그전에 쓰이지 않았고, 빠진 run이 없다는 것을 확인할 수 없다.
- (1)에 대해: 묶음에는 커밋 내용이 들어가지 않는다. 그래서 이번 주장에서 "미리 등록한"이라는 말을 뺐다. 주장은 사전 등록에 기대지 않는다. 아래 (2)의 ledger가 보여 주는 대로, seed 4~6에서 끝난 suffdisc·maskedkd run 전부가 같은 방향이고 차이가 크다(WGA 최소 +35.34). 문턱값을 어디에 두든 결론은 바뀌지 않는다. 참고로 git log -25 --stat에는 3078e90(04:05:57)이 PROPOSALS.md를 바꾼 기록이 있다.
- (2)에 대해: 새 분석 analysis_seed_ledger_1031_054457_4e01 을 돌렸다(`api.runs()`만 읽음, 모델 없음, 지금 시점의 끝난 Waterbirds run 전부). 

## 주장 (1줄)
Waterbirds K=12에서 suffdisc는 승격 판단에 쓰지 않은 새 seed 4·5·6(screen tier, 100 에폭) 모두에서 maskedkd보다 정확도와 WGA가 높다.

## run id (status done, epochs_done 100, tier screen, sel_args {})
| seed | suffdisc@12 (acc / WGA) | maskedkd@12 (acc / WGA) | 차이 (acc / WGA) |
|---|---|---|---|
| 4 | screen_suffdisc_k12_s4_1031_040422_fa2a (89.99 / 57.89) | screen_maskedkd_k12_s4_1031_040431_c049 (84.49 / 20.30) | +5.50 / +37.59 |
| 5 | screen_suffdisc_k12_s5_1031_040435_4e00 (90.16 / 64.66) | screen_maskedkd_k12_s5_1031_040439_3cc2 (84.74 / 18.80) | +5.42 / +45.86 |
| 6 | screen_suffdisc_k12_s6_1031_040444_bb51 (88.91 / 58.65) | screen_maskedkd_k12_s6_1031_040448_eb48 (84.74 / 23.31) | +4.17 / +35.34 |

평균 차이는 acc +5.03, WGA +39.60이다(metrics.json `val_acc_last`, `val_wga_last`). 비용(metrics.json `flops.total_gflops_per_img`): suffdisc 15.0222, maskedkd 10.0728 (×1.49).
참고(선택과 독립 아님): final seed 1~3은 RESULTS_TABLE "Significant only"의 `suffdisc@12 vs maskedkd@12` acc +5.42 [+4.18, +6.66], WGA +40.10 [+30.88, +49.32]이다.

## seed 미사용·누락 없음 (analysis_seed_ledger_1031_054457_4e01 )
- summary.json: 끝난 Waterbirds run은 110개다. `first_seed_ge4_submit`는 screen_suffdisc_k12_s4_1031_040422_fa2a 이다. 즉 seed 4 이상을 쓴 run은 이 재현 이전에 하나도 없었다. 승격(H-W22, 라운드 36)과 final(1031_0024 제출)은 이보다 먼저다. `seeds_per_method`에서 seed 4~6을 쓴 방법은 center, maskedkd, rollout, suffdisc 넷뿐이다(rollout·center는 H-W31 기준선이다).
- fresh_seed_runs.json: seed 4 이상인 끝난 run 11개 전부를 제출 순서로 나열한다. suffdisc·maskedkd는 위 표의 6개뿐이고, 나머지 5개는 rollout s4~s6과 center s4~s5다.
- suff_mkd_runs.json: suffdisc·maskedkd의 끝난 run 30개 전부(모든 tier·K·seed)를 나열한다. seed 4~6인 K=12 run은 위 6개뿐이다. 고르거나 뺀 run은 없다.
- 이 분석은 `api.runs()`의 status done run만 본다. 지금 실행 중이거나 대기 중인 seed 4~6 job은 tome s4~s6, center s6, grid s4이고, 모두 H-W31 기준선이다. suffdisc·maskedkd job은 없다.

## 근거 분석 (승격 전 분해, AUDIT_PROMPT 15항)
analysis_suffdisc_decomp_1030_202612_878c . summary.json `overlap` 첫 열(`suff_mkd`, K 대비 비율)은 승격 전 screen student 네 시점에서 0.1065 / 0.1414 / 0.1372 / 0.1326이다(clean val, eval mode 선택). suffdisc가 고른 12개 중 MaskedKD와 겹치는 것은 약 1.3~1.7개이므로, 이득은 MaskedKD가 고르지 않는 토큰에서 온다. 같은 분석의 `table`에서 K=12 토큰만 받은 masked teacher의 t_acc/t_wga는 suffdisc 선택 84.57/48.12, maskedkd 선택 72.23/28.57이다(student suffdisc_s0_final 기준).

## 노벨티
MaskedKD는 student 마지막 블록 CLS attention top-K를 쓴다. suffdisc는 ERASER sufficiency(DeYoung et al. 2020) / RISE insertion(Petsiuk et al. 2018)식 keep-only 충실도를 student에서 재서 토큰을 고른다. 설명 평가용 sufficiency를 KD teacher 입력 선택에 쓴 예는 라운드 1·36 검색에서 찾지 못했다.

## 주장하지 않는 것 / Caveats
- rollout·tome·grid·center·random 대비 우위(3/4·4/4부)는 이 주장에 넣지 않는다. 같은 새 seed 기준선(H-W31)이 실행 중이다. tome: final s1~3에서 acc 차이는 불확실하고(−0.33, CI [−2.57, +1.91]) WGA는 +11.53 WIN이다(RESULTS_TABLE). 새 seed의 tome은 아직 없다.
- 'Full KD 수준', 같은 GFLOPs에서의 우위, K=12가 아닌 K는 주장하지 않는다. 비용이 ×1.49로 더 크다.
- 탐색: suffdisc는 4 configs다. 승격한 것은 config 1이고, 이번 run도 config 1(sel_args `{}`)이다.

## 무결성
agent 커밋 중 harness/·tests/를 바꾼 것은 없다. test_eval은 쓰지 않았다. 인용: run id 6개, 분석 2개.
