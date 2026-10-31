# AUDIT_REQUEST — [claude] 2026-10-31 라운드 43 (1/4부 재요청: 주장을 config 단위로 좁힘)

## 지난 판정과 고친 점
AUDIT.md 02:20(codex)은 NEEDS-MORE였다. run 11개의 상태·epochs·digest·SHA·수치, final/screen 평균, run id 시각 순서는 모두 확인됐다. 남은 문제는 하나다. "screen 8개가 전부"를 확인하려면 전체 run 목록이 필요한데, RESULTS_TABLE이 12k자에서 잘려 suffdisc 행(All finished runs 표의 뒤쪽)이 보이지 않는다.
이 잘림은 잠긴 harness/evidence.py(FILE_LIMIT 12,000)에서 생기고, 표가 18k자를 넘어서 나는 고칠 수 없다. 그래서 **"run 8개가 전부"라는 문장을 주장에서 뺐다.** 남긴 주장은 묶음 안에서 확인할 수 있는 것뿐이다. 탐색량은 harness가 직접 세는 RESULTS_TABLE "Search effort (screen)" 절(묶음 안, 잘리기 전)의 config 수로만 말한다.

## 주장 (1줄)
`suffdisc`(selector_sha 0f060023d021…)의 Waterbirds 탐색은 Search effort 기준 4 configs이고, final로 올린 것은 그중 K=12 기본 설정 하나이며, 이 결정은 다른 3 config의 run이 시작되기 전에 내렸다. final 3-seed 평균(89.63/62.40)은 같은 설정의 screen 4-seed 평균(89.87/63.35)과 noise threshold 안에서 일관된다.

주장하지 않는 것: config 안의 run 개수가 아래 표가 전부라는 것(묶음으로 확인 불가), 다른 방법과의 비교(2/4~4/4부).

## 4 configs와 각 config의 run (status done, epochs_done 100, harness_digest 033df9786a4a)
| config | run | 정확도 | WGA | GFLOPs |
|---|---|---|---|---|
| 1. K=12, sel_args {} | screen_suffdisc_k12_s0_1030_173118_504a | 89.32 | 58.65 | 15.02 |
| 1. | screen_suffdisc_k12_s1_1030_185953_9455 | 89.91 | 63.91 | 15.02 |
| 1. | screen_suffdisc_k12_s2_1030_233019_98dd | 90.24 | 62.41 | 15.02 |
| 1. | screen_suffdisc_k12_s3_1030_233028_39ad | 89.99 | 68.42 | 15.02 |
| 2. K=12, sel_args {"r": 1} | screen_suffdisc_k12_s0_1031_011600_6b15 | 88.57 | 55.64 | 11.86 |
| 2. | screen_suffdisc_k12_s1_1031_011604_6add | 88.99 | 56.39 | 11.86 |
| 3. K=29, sel_args {} | screen_suffdisc_k29_s0_1031_011608_3604 | 90.16 | 61.65 | 22.62 |
| 4. K=49, sel_args {} | screen_suffdisc_k49_s0_1031_012539_2b13 | 90.24 | 60.90 | 31.75 |
| final (config 1) | final_suffdisc_k12_s1_1031_002441_5d86 | 90.08 | 65.41 | 15.02 |
| final (config 1) | final_suffdisc_k12_s2_1031_002448_664e | 89.66 | 60.90 | 15.02 |
| final (config 1) | final_suffdisc_k12_s3_1031_002453_9ebe | 89.16 | 60.90 | 15.02 |

Search effort 절의 "suffdisc: 4 configs"가 위 config 1~4와 맞는다. 각 metrics.json의 keep·sel_args로 config를 확인할 수 있다.
알림: 이 요청을 쓴 뒤 config 3의 seed 1(screen_suffdisc_k29_s1_1031_022639_643b)을 제출했다. 'K=12 한정' caveat을 2 seed로 확인하려는 것이다. 새 config가 아니라서 config 수는 그대로 4다. 아직 끝나지 않아 위 표와 주장에는 넣지 않았다.

## 시간 순서 (run id의 타임스탬프)
config 1 screen 시작(10-30 17:31) → config 1 final 제출(10-31 00:24) → config 2·3 제출(01:16) → config 4 제출(01:25). config 2~4의 첫 run 시각이 모두 final 제출보다 늦다. 따라서 final로 올릴 설정을 고를 때 config 2~4의 결과는 없었다.

## 다른 config의 결과 (숨기지 않음)
- config 2(r=1, 비용 맞춤): 평균 88.78/56.02. config 1보다 낮다.
- config 3·4: RESULTS_TABLE Parent check(묶음 안) 기준 같은 seed의 center 대비 K=29 acc +1.17·WGA −0.76, K=49 acc +0.08·WGA +0.75. WGA 이득은 K=12에서만 보인다. 그래서 이후 모든 부의 주장은 K=12에 한정한다.
- config 1의 Parent check(seed 0만)는 center 대비 UNPROVEN(acc −0.17, WGA −0.75)이다. 4-seed 짝 비교와 final 비교는 4/4부에서 다룬다.

## screen → final 일관성
config 1 screen 4-seed 평균 89.87/63.35, final 3-seed 평균 89.63/62.40(Core comparison/표 final 행). 차이 −0.24/−0.95는 noise threshold(0.76/2.45, Noise 줄) 안이다. 위 표의 run 값으로 직접 다시 계산할 수 있다.

## 무결성
agent 커밋 중 harness/·tests/를 바꾼 것은 없다. test_eval은 쓰지 않았다.
