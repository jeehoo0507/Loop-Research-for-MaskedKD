# AUDIT_REQUEST — [claude] 2026-10-31 라운드 41 (1/3부 재구성)

## 지난 판정과 고친 점
지난 판정(AUDIT.md 01:56, codex)은 NEEDS-MORE였다. 수치·상태·digest는 맞았다. 빠진 것은 두 가지였다. (1) 분해 분석 출력이 없었다. 원인은 묶음 한도 120k자다. 지난 요청은 run id 13개를 인용했고(run당 약 7k자), 맨 끝에 붙는 분석 출력이 잘렸다. (2) RESULTS_TABLE이 12k자에서 잘려 suffdisc screen 이력 행이 보이지 않았다.
이번 요청은 run 10개와 분석 1개만 인용한다. 구성은 suffdisc final 3개, maskedkd final 3개, 그리고 같은 설정의 suffdisc screen 4개 전부다. rollout 비교는 2/3부(tome과 함께)로 옮겼다.

## 주장 (1/3부, 1줄)
Waterbirds K=12에서 `suffdisc`(student sufficiency로 teacher 입력 연속 원판의 위치를 고름)는 final seeds 1·2·3에서 `maskedkd`를 정확도와 WGA 모두에서 이긴다(paired 95% CI가 0 밖). 같은 설정의 screen seeds 0~3 결과는 final에서도 유지된다.

## run id (모두 status done, epochs_done 100, harness_digest 033df9786a4a, suffdisc는 selector_sha 0f060023d021…)
| run | tier/seed | 정확도 | WGA |
|---|---|---|---|
| final_suffdisc_k12_s1_1031_002441_5d86 | final 1 | 90.08 | 65.41 |
| final_suffdisc_k12_s2_1031_002448_664e | final 2 | 89.66 | 60.90 |
| final_suffdisc_k12_s3_1031_002453_9ebe | final 3 | 89.16 | 60.90 |
| final_maskedkd_k12_s1_1030_201811_e49a | final 1 | 84.15 | 21.05 |
| final_maskedkd_k12_s2_1030_212126_ad00 | final 2 | 84.24 | 23.31 |
| final_maskedkd_k12_s3_1030_220958_cc4e | final 3 | 84.24 | 22.56 |
| screen_suffdisc_k12_s0_1030_173118_504a | screen 0 | 89.32 | 58.65 |
| screen_suffdisc_k12_s1_1030_185953_9455 | screen 1 | 89.91 | 63.91 |
| screen_suffdisc_k12_s2_1030_233019_98dd | screen 2 | 90.24 | 62.41 |
| screen_suffdisc_k12_s3_1030_233028_39ad | screen 3 | 89.99 | 68.42 |

## Core comparison (RESULTS_TABLE, seeds 1,2,3, paired 95% CI)
- suffdisc@12 vs maskedkd@12: acc +5.42 [+4.18, +6.66] WIN, WGA +40.10 [+30.88, +49.32] WIN. GFLOPs 비율 1.491.

## 탐색 이력 (항목 7)
"Search effort"에 suffdisc는 2 configs로 나온다. (a) 기본 설정(sel_args {}): 위 screen 4개, 평균 89.87/63.35. final 3-seed 평균은 89.63/62.40이다. 그래서 screen의 이득은 새 seed에서도 유지된다. (b) r=1(후보 9개, 비용 맞춤 시도): screen seed 0·1, 평균 88.78/56.02. 기본 설정보다 낮고 final로 올리지 않았다. 이 run id들은 한도 때문에 3/3부에 묶는다. 다른 K: suffdisc@29 screen seed 0은 90.16/61.65다. rollout@29 대비 WGA −2.63, center@29 대비 +1.50으로 미리 정한 기준 2.45에 못 미친다. 그래서 주장은 K=12에 한정한다. 자식 jitdisc(같은 후보 중 무작위 원판)는 3/3부의 대체 실험이다.

## 분해 (항목 15)
analysis_suffdisc_decomp_1030_202612_878c(summary.json): suffdisc가 고른 토큰 중 maskedkd와 겹치는 비율은 0.11~0.14다. rollout과는 0.22~0.24, center와는 0.37~0.38이다. centre_frac은 0.024~0.032라 원판은 거의 항상 중앙 밖으로 옮겨진다. 이득은 maskedkd 토큰을 다시 고르는 데서 오지 않는다. 대부분 maskedkd가 고르지 않은 연속 영역에서 온다.

## tome과 다른 기준선 (항목 10·13)
tome 대비 WGA는 +11.53 [+0.58, +22.47] WIN이고, 정확도 차이는 −0.33 [−2.57, +1.91]로 불확실하다. rollout·grid·center·random은 2/3부와 3/3부에서 다룬다. 이 1/3부의 PASS만으로 '해석 신호의 이득'을 주장하지 않는다.

## 비용과 범위
total 15.02 GFLOPs/img로 maskedkd(10.07)의 1.49배다. selector 비용은 4.93이고, 이것은 후보 원판 25개 × student K-토큰 forward다. 주장은 'teacher 토큰 예산 K 고정' 축에만 한다. compute 대비 우위와 주 지표('Full 허용 오차 안의 최소 K')는 주장하지 않는다. full 대비 차이는 acc −1.53, WGA −5.02이고 평균이 허용 오차 밖이다.

## 무결성
agent 커밋 중 harness/·tests/를 바꾼 것은 없다. test_eval은 쓰지 않았다(`final/` 디렉터리 없음).

## 노벨티
ERASER sufficiency / RISE insertion식 keep-only 충실도를 student에서 재서 연속 원판의 위치를 정한다. 설명 평가용 sufficiency를 KD teacher 입력 선택에 쓴 예는 찾지 못했다. 가장 가까운 선행은 MaskedKD(attention top-K), arXiv 2411.17046(중앙 편향 patch 합성), arXiv 2107.01378(patch 수준 증류)이다.

### verdict
CLAIM: <the claim, one line>
VERDICT: PASS | FAIL | NEEDS-MORE
REASON: <one or two lines, with run ids>
