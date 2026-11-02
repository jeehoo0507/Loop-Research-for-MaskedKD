# AUDIT_REQUEST — [claude] 2026-11-02 라운드 179 (CUB lateroll@29 final, 사람 지시 QUEUE 2026-11-02 08:50 2번)

사람 지시: stage 2(FGVC-Aircraft)에 lateroll을 올리려면 AUDIT.md에 PASS가 필요하다(`submit.stage2_rule`). 그래서 CUB final 결과를 감사에 낸다.
묶음 크기(180k) 때문에 run id는 6개(lateroll·maskedkd)만 적는다. rollout·tome·grid·center·random·full 비교는 RESULTS_TABLE "Significant only" 줄(묶음 첫 12k 안)과 아래 compare 출력의 숫자로 본다.

## 주장 (1줄)
CUB-200-2011 K=29(이미지당 teacher 토큰 29개, 학습 1스텝 총 12.99 GFLOPs/img)에서 lateroll(student 블록 6~11만의 attention rollout top-K)은 final seed 1·2·3 평균 정확도 81.39로, 같은 K·같은 seed의 maskedkd(80.22)·rollout·tome·grid·center·random보다 높고 짝지은 95% CI가 모두 0을 포함하지 않는다. 단 maskedkd·rollout·tome 대비 차이(+1.17·+0.94·+1.61)는 promotion threshold 1.92보다 작고, Full KD(42.77 GFLOPs/img) 대비 차이(−0.33)는 불확실하다.

## run id (status done, tier final, 100 에폭, dataset cub, sel_args {})
| seed | lateroll@29 | maskedkd@29 | 차이 |
|---|---|---|---|
| 1 | final_lateroll_k29_s1_1101_104303_ef38 (82.33) | final_maskedkd_k29_s1_1101_104303_b676 (81.33) | +1.00 |
| 2 | final_lateroll_k29_s2_1101_125958_6470 (80.33) | final_maskedkd_k29_s2_1101_125958_5b02 (79.17) | +1.17 |
| 3 | final_lateroll_k29_s3_1101_140937_ec98 (81.50) | final_maskedkd_k29_s3_1101_140937_b00f (80.17) | +1.33 |

값은 metrics.json `val_acc_last`(소수 둘째 자리 반올림)다. 비용(metrics.json `flops`): lateroll total 12.99, selector 0.0765, teacher 5.3608 GFLOPs/img. maskedkd total 12.99.

## compare 출력 (`python -m harness.compare --a lateroll:29 --b <B> --tier final --dataset cub`, seed 1·2·3, 짝지은 95% CI)
| B | B 값 (s1 / s2 / s3) | A − B | 95% CI | GFLOPs A/B |
|---|---|---|---|---|
| maskedkd:29 | 81.33 / 79.17 / 80.17 | +1.17 | [+0.75, +1.58] | 1.000 |
| rollout:29 (META parent) | 81.33 / 79.33 / 80.67 | +0.94 | [+0.71, +1.18] | 1.000 |
| tome:29 | 80.67 / 78.67 / 80.00 | +1.61 | [+1.37, +1.85] | 1.083 (tome 12.00) |
| grid:29 | 75.50 / 75.83 / 74.67 | +6.06 | [+2.71, +9.40] | 0.999 |
| center:29 | 78.33 / 77.17 / 79.33 | +3.11 | [+0.83, +5.39] | 0.999 |
| random:29 | 74.67 / 75.00 / 75.17 | +6.44 | [+3.54, +9.35] | 0.999 |
| full:196 | 82.83 / 79.83 / 82.50 | −0.33 | [−2.23, +1.56] | 0.304 |

같은 줄이 RESULTS_TABLE "Significant only"(14~29행)에 있다. `ce:0` final은 seed 1 하나뿐이라 compare가 돌지 않는다(need >=2 shared seeds). 그래서 final에서 주 지표(Full KD 허용 오차)는 판정하지 않는다.

## 근거 분석 (방법보다 먼저, AUDIT_PROMPT 12·16항)
analysis_cub_rollout_depth_1031_100143_a4b6 (H-C2b). rollout 시작 블록 0/3/6/8/9/10/11을 student 9개(최종 3개 + epoch 25·50 snapshot 6개), val 600장에서 비교했다. K=25·12에서 블록 6부터의 rollout이 가린 teacher의 TCKD(soft label의 목표 클래스 확률 보존)를 가장 잘 지켰다(최종 3개 평균 roll0 1.309, roll3 1.126, roll6 1.092, 늦출수록 다시 커짐). 9개 student 모두 roll6 TCKD가 roll0보다 낮고, centre_overlap(roll6 − roll0)은 −0.026~−0.067로 roll6이 덜 가운데로 쏠린다(verdict.json, summary.json). 순서: 사전 등록 커밋 79fd73c(10:01:37 UTC) → 분석 analysis_cub_rollout_depth_1031_100143_a4b6 → selectors/lateroll.py 추가 커밋 95bab8f(10:04:52 UTC) → 첫 lateroll run(1031_130339). 근거가 방법보다 먼저다.

## 분해 (AUDIT_PROMPT 15항)
analysis_cub_lateroll_decomp_1101_070348_b793 (H-C18, 사전 등록 뒤 실행). lateroll K=29 screen student 3개(seed 0·4·5), val 600장, eval mode, 이미지 bootstrap 95% CI.
- MaskedKD와 겹침 |L∩M|/K = 0.413·0.424·0.481. MaskedKD와 겹치지 않는 lateroll 토큰을 lateroll 밖 random 토큰으로 바꾸면 가린 teacher TCKD가 +0.521 [0.418, 0.624]·+0.481 [0.386, 0.579]·+0.309 [0.235, 0.389] 나빠진다. maskedkd 대비 이득은 겹치지 않는 토큰에서 온다.
- rollout과 겹침 |L∩R|/K = 0.738·0.735·0.738. R − L TCKD는 +0.080·+0.074·+0.066(CI > 0)이지만, rollout과 겹치지 않는 lateroll 토큰을 random으로 바꾼 차이는 +0.030 [−0.019, 0.081]·+0.033 [−0.017, 0.085]·+0.015 [−0.032, 0.063]로 불확실하다. 그래서 rollout 대비 이득의 토큰 수준 메커니즘은 주장하지 않는다.
- 공간: 가운데 거리 L − R = +0.053·+0.095·+0.058 패치(CI > 0). lateroll은 rollout보다 덜 가운데로 쏠린다. center@29 대비 정확도는 위 표에서 +3.11 [+0.83, +5.39]다.

## 노벨티 (약함)
가장 가까운 선행: attention rollout(Abnar & Zuidema, ACL 2020, harness `rollout` 기준선)과 MaskedKD(Son et al., ECCV 2024, student 마지막 블록 CLS attention). lateroll은 rollout을 student 뒤쪽 블록 6~11에서만 곱한다. Chefer et al.(CVPR 2021)도 뒤쪽 층의 relevance를 강조한다. 새 기법이라기보다 rollout 시작 블록을 KD teacher 입력 선택에 맞춰 분석(H-C2b)으로 정한 것이다. novelty가 약하다는 판단이면 그렇게 적어 주기 바란다.

## 주장하지 않는 것 / caveats
- Parent check(screen seed 0, RESULTS_TABLE): lateroll@29 vs rollout +0.50 < 1.92 → UNPROVEN. final seed 1·2·3에서는 +0.94 [+0.71, +1.18]로 CI가 0 밖이지만 threshold 1.92보다 작다. maskedkd(+1.17)·tome(+1.61) 대비도 threshold보다 작다. 이 크기를 "의미 있는 이득"이라고 주장하지 않는다. 주장은 "같은 seed 짝 비교에서 모두 양수이고 CI가 0을 포함하지 않는다"까지다.
- FLOPs 효율 주장(같은 총비용의 더 큰 K 기준선 대비)은 하지 않는다. 참고(screen seed 0·4·5, 숫자만): lateroll@29 82.00 (12.99) vs maskedkd@49 80.78 (16.44), rollout@49 81.28 (16.45). RESULTS_TABLE의 screen 행이다.
- Search effort: lateroll 5 configs (K = 12·20·29·49·98, sel_args 모두 {}). final은 K=29 하나만 냈다(사람 지시로 고정 K 기준선으로 final을 냄, QUEUE 2026-11-01). 다른 K의 screen 평균: K=12 78.56, K=49 82.11, K=98 82.50 (RESULTS_TABLE). K=12에서는 CE보다 낮다.
- 비교군: maskedkd·rollout·tome·grid·center·random·full은 harness가 같은 seed로 자동으로 냈다(final_refs). final seed 1·2·3은 screen seed 0·4·5와 겹치지 않는다.
- CUB 하나의 결과다. stage 2(Aircraft) 이전 검증은 PASS 뒤에 사람 지시대로 한다.

## 무결성
git log -- harness tests의 최근 9개는 사람(cjh) 커밋이다. 10번째 c5753fd(2026-10-31 round 55 autosave)는 루프 autosave가 사람의 잠금 커밋을 되돌린 사건이고, 사람이 42906db로 복구했다(인용한 run은 모두 2026-11-01 이후 제출, harness_digest는 metrics.json에서 확인). harness_digest: 지금 잠금은 1f0373dc8ed0(사람 커밋 8f67ca5 뒤, `--check` OK)이다. 인용 run의 digest는 seed 1·2가 3b594cc1646f, seed 3이 8ac8a710a4ae로 지금 값과 다르다. 그 사이 harness를 바꾼 것은 사람 커밋 faeb3cd(무작위 전부/없음 기준선, scheduler 좀비 수거)와 8f67ca5(Aircraft 데이터·stage2 부팅)뿐이다. 같은 seed의 lateroll과 maskedkd는 seed마다 같은 digest를 쓴다(짝 비교는 같은 harness 안에서 이뤄짐). test_eval은 쓰지 않았다. lateroll selector_sha 69541b27aca9. 인용: run id 6개, 분석 2개.
