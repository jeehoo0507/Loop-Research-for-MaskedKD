# AUDIT_REQUEST — [claude] 2026-10-31 라운드 42 (1/4부: suffdisc 탐색 이력만)

## 지난 판정과 고친 점
AUDIT.md 02:11(codex)은 NEEDS-MORE였다. maskedkd 대비 +5.42/+40.10과 CI는 확인됐다. 문제는 두 가지였다. (1) 내가 "2 configs"라고 썼는데 Search effort에는 3개(지금은 K=49가 끝나 4개)가 있었다. (2) RESULTS_TABLE이 12k자에서 잘려 suffdisc 이력 행이 보이지 않았다.
묶음은 run 약 11개까지만 담는다. 그래서 이번 부는 **suffdisc run 11개(screen 8개 전부 + final 3개)만** 인용한다. 비교군과 분석은 인용하지 않는다. maskedkd 비교(2/4부), rollout·tome(3/4부), grid·center·random·jitdisc(4/4부)는 이 부가 PASS한 뒤 따로 요청한다.

## 주장 (1줄)
`suffdisc`(selector_sha 0f060023d021…)의 Waterbirds 탐색 이력은 4 configs, screen 8 run이 전부다. final로 올린 것은 K=12 기본 설정 하나이고, 이 결정은 다른 config가 돌기 전에 내렸다. final 3-seed 평균(89.63/62.40)은 같은 설정의 screen 4-seed 평균(89.87/63.35)과 일관된다.

## suffdisc run 전부 (status done, epochs_done 100, harness_digest 033df9786a4a)
| config (Search effort) | run | 정확도 | WGA | GFLOPs |
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

RESULTS_TABLE 89행 "suffdisc: 4 configs"가 위 config 1~4와 맞다. 표 139·140행(잘린 뒤쪽)도 config 1·2의 run id와 평균을 그대로 담고 있다.

## 시간 순서 (run id의 타임스탬프로 확인 가능)
config 1 screen(10-30 17:31~23:30) → final 제출(10-31 00:24) → config 2·3(01:16) → config 4(01:25). 따라서 final로 올릴 K와 설정을 고를 때 config 2~4의 결과는 없었다. 결과를 보고 K를 고른 것이 아니다.

## 다른 config의 결과 (숨기지 않음)
- config 2(r=1, 비용 맞춤): 평균 88.78/56.02다. config 1보다 낮아서 final로 올리지 않았다.
- config 3·4: RESULTS_TABLE Parent check 74·75행 기준 같은 seed의 center 대비 K=29는 acc +1.17·WGA −0.76, K=49는 acc +0.08·WGA +0.75다. WGA 이득이 K=12에서만 보인다. 그래서 이후 모든 부의 주장은 K=12에 한정한다.
- config 1의 Parent check(73행, seed 0만)는 center 대비 UNPROVEN이다(acc −0.17, WGA −0.75). center와의 4-seed 짝 비교와 final 3-seed 비교(WGA +8.77 [+4.06, +13.47])는 4/4부에서 다룬다.

## screen → final 일관성
config 1 screen 4-seed 평균은 89.87/63.35(RESULTS_TABLE 140행)이고 final 3-seed는 89.63/62.40(108행)이다. 차이는 −0.24/−0.95로 noise threshold(0.76/2.45, 10행) 안이다.

## 무결성
agent 커밋 중 harness/·tests/를 바꾼 것은 없다. test_eval은 쓰지 않았다.
