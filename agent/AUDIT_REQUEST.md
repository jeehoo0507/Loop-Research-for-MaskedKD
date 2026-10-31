# AUDIT_REQUEST — [claude] 2026-10-31 라운드 49 (2/4부 재요청: suffdisc@12 vs maskedkd@12, 선택과 독립인 새 seed)

## 지난 판정과 고친 점
03:57 감사(codex)는 이 주장의 수치와 CI는 맞다고 했다. 다만 final seed 1~3이 승격 전 screen에서 이미 쓰였으므로 "선택과 독립인 final randomness" 증거가 없다는 이유로 NEEDS-MORE를 줬다. harness는 tier와 상관없이 같은 seed를 쓴다. 그래서 이번에는 **한 번도 쓰지 않은 seed 4·5·6**에서 같은 설정(sel_args `{}`, K=12)을 다시 돌렸다. 읽는 규칙은 PROPOSALS.md H-W30에 적어 커밋 3078e90(04:05:57)으로 올렸다. 이 커밋은 job 제출(04:04:22~04:04:48) 직후이고, 첫 결과(maskedkd s4, 04:33:23 종료, log.txt)보다 약 27분 먼저이고, 첫 suffdisc 결과(s4, 04:53:42)보다 먼저다. 규칙: 3 seed 모두 suffdisc − maskedkd의 acc와 WGA가 양수이고, WGA 평균이 +2.45(RESULTS_TABLE Noise의 WGA threshold) 이상이면 재현이다.
analysis_run_ledger_1031_035326_a927 (04:04 제출보다 먼저 끝남)의 suffdisc_runs.json에는 seed 0~3 run만 있다. 따라서 seed 4~6은 승격 판단에 쓰이지 않았다.

## 주장 (1줄)
Waterbirds K=12에서 suffdisc는 승격 뒤 미리 등록한 새 seed 4·5·6(screen tier, 100 에폭) 모두에서 maskedkd보다 정확도와 WGA가 높다.

## run id (status done, epochs_done 100, tier screen, sel_args {})
| seed | suffdisc@12 (acc / WGA) | maskedkd@12 (acc / WGA) | 차이 (acc / WGA) |
|---|---|---|---|
| 4 | screen_suffdisc_k12_s4_1031_040422_fa2a (89.99 / 57.89) | screen_maskedkd_k12_s4_1031_040431_c049 (84.49 / 20.30) | +5.50 / +37.59 |
| 5 | screen_suffdisc_k12_s5_1031_040435_4e00 (90.16 / 64.66) | screen_maskedkd_k12_s5_1031_040439_3cc2 (84.74 / 18.80) | +5.42 / +45.86 |
| 6 | screen_suffdisc_k12_s6_1031_040444_bb51 (88.91 / 58.65) | screen_maskedkd_k12_s6_1031_040448_eb48 (84.74 / 23.31) | +4.17 / +35.34 |

평균 차이는 acc +5.03, WGA +39.60이다(metrics.json `val_acc_last`, `val_wga_last`에서 계산). 6개 run은 모두 seed 4~6에서 끝난 suffdisc/maskedkd K=12 run 전부다. 고르거나 뺀 것은 없다.
harness.compare는 seed를 고를 수 없다. screen tier에는 seed 0~3 run도 섞이기 때문에 compare 출력은 붙이지 않는다. 비용(metrics.json `flops.total_gflops_per_img`): suffdisc 15.0222, maskedkd 10.0728 (×1.49).
참고(선택과 독립 아님): final seed 1~3 결과는 RESULTS_TABLE "Significant only"의 `suffdisc@12 vs maskedkd@12` acc +5.42 [+4.18, +6.66], WGA +40.10 [+30.88, +49.32]이다. 새 seed의 크기도 이와 비슷하다.

## 근거 분석 (승격 전 분해, AUDIT_PROMPT 15항)
analysis_suffdisc_decomp_1030_202612_878c . summary.json `overlap` 첫 열(`suff_mkd`, K 대비 비율)은 승격 전 screen student 네 시점에서 0.1065 / 0.1414 / 0.1372 / 0.1326이다(clean val, eval mode 선택). 즉 suffdisc가 고른 12개 토큰 중 MaskedKD와 겹치는 것은 약 1.3~1.7개다. 이득은 MaskedKD가 고르지 않는 토큰에서 온다. 같은 분석의 `table`에서 K=12 토큰만 받은 masked teacher의 t_acc/t_wga는 suffdisc 선택 84.57/48.12, maskedkd 선택 72.23/28.57이다(student suffdisc_s0_final 기준). suffdisc 토큰만으로 teacher가 더 충실한 예측을 낸다는 뜻이다.

## 노벨티
MaskedKD는 student 마지막 블록 CLS attention top-K를 쓴다. suffdisc는 attention 순위 대신 ERASER sufficiency / RISE insertion식 keep-only 충실도를 student에서 재서 연속 원판의 위치를 정한다. 가장 가까운 선행 기법은 ERASER(DeYoung et al. 2020)의 sufficiency와 RISE(Petsiuk et al. 2018)다. 설명 평가용 sufficiency를 KD teacher 입력 선택에 쓴 예는 라운드 1·36 검색에서 찾지 못했다.

## 주장하지 않는 것 / Caveats
- rollout·tome·grid·center·random 대비 우위(3/4·4/4부)는 이 주장에 넣지 않는다. 같은 새 seed의 기준선(H-W31)은 실행 중이다. tome에 대해 적어 두면, final s1~3에서 acc는 차이가 없고(−0.33, CI [−2.57, +1.91]) WGA는 +11.53 WIN이다(RESULTS_TABLE).
- 'Full KD 수준', 같은 GFLOPs에서의 우위, K=12가 아닌 K는 주장하지 않는다.
- 탐색: suffdisc는 4 configs다(ledger 참조). 승격한 것은 config 1이며, 이번 재현도 config 1이다.
- 비용이 ×1.49로 더 크다.

## 무결성
agent 커밋 중 harness/·tests/를 바꾼 것은 없다. test_eval은 쓰지 않았다. 이번 요청은 run id 6개와 분석 2개를 인용한다.
