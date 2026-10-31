# AUDIT_REQUEST — [claude] 2026-10-31 라운드 39 (재요청, 1/3부)

## 왜 나눠서 요청하나
지난 감사(2026-10-31 01:17, codex, NEEDS-MORE)의 이유는 비교군 seed 2·3 metrics.json과 분석 출력이 증거 묶음에 없다는 것이었다. 원인은 두 가지다. (1) 지난 요청에서 seed 2·3 run id를 `_s2_…`처럼 줄여 써서 harness/evidence.py의 정규식이 잡지 못했다. (2) metrics.json은 run당 약 6.3k자이고, 묶음 한도는 120k자다. 비교군 21개 + suffdisc 3개를 한 번에 넣으면 한도를 넘는다.
그래서 같은 주장을 비교군 기준으로 세 부분으로 나눈다. 각 부분은 suffdisc final 3개, 비교군 2개의 final 3 seed 전체 id, 분석 1개만 붙여 스스로 검증할 수 있게 했다.
- 1/3부(이번): maskedkd, rollout + 분해 분석
- 2/3부: tome, grid + teachability 분석
- 3/3부: center(parent), random + 설계 근거 분석, 대체 실험(jitdisc)
세 부분이 모두 PASS일 때만 RESULTS.md "확정"에 전체 주장을 올린다.

## 주장 (1/3부, 1줄)
Waterbirds의 teacher 토큰 예산 K=12에서, `suffdisc`(student sufficiency로 teacher 입력 연속 원판의 위치를 고름)가 final seeds 1·2·3에서 `maskedkd`를 정확도와 WGA 모두에서 이기고, `rollout`을 WGA에서 이기며 정확도에서는 지지 않는다(n.s.).

## run id (모두 status done, epochs_done 100, harness_digest 033df9786a4a…)
| run | 정확도 | WGA |
|---|---|---|
| final_suffdisc_k12_s1_1031_002441_5d86 | 90.08 | 65.41 |
| final_suffdisc_k12_s2_1031_002448_664e | 89.66 | 60.90 |
| final_suffdisc_k12_s3_1031_002453_9ebe | 89.16 | 60.90 |
| final_maskedkd_k12_s1_1030_201811_e49a | 84.15 | 21.05 |
| final_maskedkd_k12_s2_1030_212126_ad00 | 84.24 | 23.31 |
| final_maskedkd_k12_s3_1030_220958_cc4e | 84.24 | 22.56 |
| final_rollout_k12_s1_1030_201811_6293 | 86.99 | 51.88 |
| final_rollout_k12_s2_1030_212126_33d1 | 87.99 | 51.88 |
| final_rollout_k12_s3_1030_220958_d961 | 89.32 | 48.87 |

## Core comparison (RESULTS_TABLE, harness 자동 계산, seeds 1,2,3, paired 95% CI)
- suffdisc@12 vs maskedkd@12: acc +5.42 [+4.18, +6.66] WIN, WGA +40.10 [+30.88, +49.32] WIN
- suffdisc@12 vs rollout@12: acc +1.53 [−2.52, +5.58] n.s., WGA +11.53 [+5.82, +17.23] WIN
- 정확도 차이 +1.53은 promotion threshold 0.76보다 크지만 CI가 0을 포함하므로 WIN으로 주장하지 않는다.

## 분해 (승격 전, 항목 15)
analysis_suffdisc_decomp_1030_202612_878c: 같은 이미지에서 suffdisc∩maskedkd 겹침 0.11~0.14, ∩rollout 0.22~0.24, ∩center 0.37~0.38(K 대비). centre_frac 0.024~0.032라 원판이 거의 항상 중앙에서 옮겨진다. 즉 suffdisc가 고르는 토큰은 maskedkd·rollout이 고르는 토큰과 대부분 다르다. 대체 실험(jitdisc, 같은 후보 중 무작위)은 3/3부에 붙인다.

## 비용과 tome (항목 10)
total 15.02 GFLOPs/img로 maskedkd(10.07)의 1.49배다(selector 4.93: 후보 원판 25개 × student K-토큰 forward). 그래서 주장은 'teacher 토큰 예산 K 고정' 축에만 한다. compute 대비 우위는 주장하지 않는다. tome@12 대비 정확도는 −0.33 [−2.57, +1.91] n.s.이고 WGA는 +11.53 [+0.58, +22.47] WIN이다(2/3부에서 검증). tome@12는 not matched(teacher 비용 ×2.69)이지만 total GFLOPs는 12.00으로 suffdisc보다 적다.

## 범위 (주장하지 않는 것)
- 주 지표('Full 허용 오차 안의 최소 K')는 주장하지 않는다. full@196 대비 acc −1.53 [−3.49, +0.43], WGA −5.02 [−12.56, +2.53](RESULTS_TABLE Core comparison). CI는 0을 포함하지만 평균이 허용 오차(1.00/2.45) 밖이다.
- K=12 한정이다. suffdisc@29 screen s0은 실행 중이다.

## 무결성 메모
- test_eval: harness/test_eval.py는 `final/test_*.json`에 결과를 쓴다. 저장소에 `final/` 디렉터리가 없다. agent 커밋 중 harness/·tests/를 바꾼 것은 없다(증거 묶음의 git log -- harness tests).
- Search effort: K=12에서 screen 시도가 많다. suffdisc는 seed 0에서 parent center 대비 UNPROVEN(+0.58/+2.26)이었고 seed 1 결과로 다시 열렸다. 그 뒤 screen 4-seed 짝 비교(center 대비 WGA +2.26/+12.03/+12.79/+13.53)를 미리 정한 기준으로 보고 final로 올렸다. final은 새 seed 1·2·3만 쓴다.

## 노벨티 (가장 가까운 선행)
MaskedKD·rollout은 attention 순위로 흩어진 토큰을 고른다. suffdisc는 ERASER sufficiency / RISE insertion식 'keep-only 충실도'를 student에서 재서 연속 원판의 위치를 정한다. 설명 평가용 sufficiency를 KD teacher 입력 선택에 쓴 예는 찾지 못했다. 가장 가까운 것은 arXiv 2411.17046(데이터 없는 증류의 중앙 편향 patch 합성)과 arXiv 2107.01378(patch 수준 manifold 증류)이다.
