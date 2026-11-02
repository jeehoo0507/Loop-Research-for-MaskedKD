# AUDIT_REQUEST — [claude] 2026-11-02 라운드 180 (CUB lateroll@29 final, 03:42 NEEDS-MORE에 대한 응답)

사람 지시(QUEUE 2026-11-02 08:50 2번): stage 2(FGVC-Aircraft)에 lateroll을 올리려면 AUDIT.md에 PASS가 필요하다.
03:42 감사의 NEEDS-MORE 두 가지에 답한다. (1) "다른 기준선 metrics가 없다" → 주장을 maskedkd 비교 하나로 좁혔다. 인용 run은 그 6개뿐이고, 모두 묶음에 들어간다. (2) "digest가 지금 잠금과 다르고 과거 잠금 확인이 없다" → 아래 "무결성" 절에 run별 digest·git 커밋, 그 뒤 사람 커밋이 바꾼 파일을 적었다.

## 주장 (1줄)
CUB-200-2011 K=29(학습 1스텝 총 12.99 GFLOPs/img, 두 방법 같음)에서 lateroll(student 블록 6~11만의 attention rollout top-K)은 final seed 1·2·3 평균 81.39로 같은 seed의 maskedkd@29(80.22)보다 +1.17 [+0.75, +1.58](짝지은 95% CI) 높다. 이 차이는 promotion threshold 1.92보다 작아 "의미 있는 이득"이라고 주장하지 않는다.

## run id (status done, tier final, 100 에폭, dataset cub, sel_args {})
| seed | lateroll@29 | maskedkd@29 | 차이 |
|---|---|---|---|
| 1 | final_lateroll_k29_s1_1101_104303_ef38 (82.33) | final_maskedkd_k29_s1_1101_104303_b676 (81.33) | +1.00 |
| 2 | final_lateroll_k29_s2_1101_125958_6470 (80.33) | final_maskedkd_k29_s2_1101_125958_5b02 (79.17) | +1.17 |
| 3 | final_lateroll_k29_s3_1101_140937_ec98 (81.50) | final_maskedkd_k29_s3_1101_140937_b00f (80.17) | +1.33 |

값은 metrics.json `val_acc_last`(소수 둘째 자리 반올림)다. 비용(metrics.json `flops`): lateroll total 12.99, selector 0.0765, teacher 5.3608 GFLOPs/img. maskedkd total 12.99. compare(`python -m harness.compare --a lateroll:29 --b maskedkd:29 --tier final --dataset cub`): +1.17 [+0.75, +1.58], GFLOPs 비 1.000. 같은 숫자가 RESULTS_TABLE "Significant only"에 있다.

## 맥락 (주장 아님, 숫자만)
같은 seed·K=29의 다른 기준선 final 평균은 RESULTS_TABLE의 표에 있다(rollout 80.44, tome 79.78, grid 75.33, center 78.28, random 74.94, Full KD@196 81.72). tome 대비 차이는 +1.61로 maskedkd 대비보다 크다(AUDIT_PROMPT 10항). 이 기준선들과의 비교는 이번 주장에 넣지 않는다.

## 근거 분석 (방법보다 먼저, AUDIT_PROMPT 12·16항)
analysis_cub_rollout_depth_1031_100143_a4b6 (H-C2b). rollout 시작 블록 0/3/6/8/9/10/11을 student 9개(최종 3개 + epoch 25·50 snapshot 6개), val 600장에서 비교했다. K=25·12에서 블록 6부터의 rollout이 가린 teacher의 TCKD(soft label의 목표 클래스 확률 보존)를 가장 잘 지켰다(최종 3개 평균 roll0 1.309, roll3 1.126, roll6 1.092). 순서: 분석 → selectors/lateroll.py 추가 → 첫 lateroll run(1031_130339).

## 분해 (AUDIT_PROMPT 15항)
analysis_cub_lateroll_decomp_1101_070348_b793 (H-C18). lateroll K=29 screen student 3개(seed 0·4·5), val 600장. MaskedKD와 겹침 |L∩M|/K = 0.413·0.424·0.481. MaskedKD와 겹치지 않는 lateroll 토큰을 lateroll 밖 random 토큰으로 바꾸면 가린 teacher TCKD가 +0.521 [0.418, 0.624]·+0.481 [0.386, 0.579]·+0.309 [0.235, 0.389] 나빠진다(summary.json). maskedkd 대비 이득은 겹치지 않는 토큰에서 온다.

## 노벨티 (약함)
가장 가까운 선행: attention rollout(Abnar & Zuidema, ACL 2020; harness `rollout`), MaskedKD(Son et al., ECCV 2024), 뒤쪽 층 relevance(Chefer et al., CVPR 2021). lateroll은 rollout을 뒤쪽 블록 6~11에서만 곱한다. 새 기법이라기보다 rollout 시작 블록을 분석(H-C2b)으로 정한 것이다.

## 주장하지 않는 것 / caveats
- Parent check(RESULTS_TABLE screen): lateroll@29 vs rollout +0.50 < 1.92 → UNPROVEN. maskedkd 대비 +1.17도 threshold보다 작다.
- FLOPs 효율 주장(같은 총비용의 더 큰 K 대비)은 하지 않는다. lateroll@49 final은 진행 중이다(H-C31, 결과 전).
- Search effort: lateroll 5 configs (K = 12·20·29·49·98, sel_args 모두 {}). final은 K=29만 끝났다. 다른 K screen 평균: K=12 78.56, K=49 82.11, K=98 82.50.
- final seed 1·2·3은 screen seed 0·4·5와 겹치지 않는다. test_eval은 쓰지 않았다. CUB 하나의 결과다.

## 무결성 (03:42 NEEDS-MORE 2번에 대한 답)
지금 잠금은 1f0373dc8ed0이다(묶음의 `harness.lock`). 인용 run 6개는 그 전 잠금에서 돌았다. 짝마다 두 방법이 같은 digest다:
| run | metrics.json harness_digest | metrics.json git (run 시작 때 HEAD) |
|---|---|---|
| lateroll s1, maskedkd s1 | 3b594cc1646f | 53616be (둘 다) |
| lateroll s2 / maskedkd s2 | 3b594cc1646f | 63ddfd6 / 34ec731 |
| lateroll s3 / maskedkd s3 | 8ac8a710a4ae | faeb3cd / cc12489 |

- 잠금 대상 파일(harness/, tests/, agent/loop.sh·relock.sh·*_PROMPT.md, requirements.txt, state/*.json)의 `git diff --stat`: 53616be·63ddfd6·34ec731은 사람 커밋 97b3249(cjh, 2026-11-01 19:16 KST "Per-image teacher token budgets…")와 차이 없음. cc12489는 사람 커밋 faeb3cd(cjh, 23:40 KST "Random all-or-nothing baseline, scheduler reaps…")와 차이 없음. 즉 3b594cc1646f = 97b3249의 잠금 파일, 8ac8a710a4ae = faeb3cd의 잠금 파일이다(agent/logs/sched.log 461행: 97b3249 반영 뒤 "harness digest 3b594cc1646f").
- 그 뒤 지금(8f67ca5, cjh)까지 잠금 파일을 바꾼 커밋은 사람의 faeb3cd와 8f67ca5뿐이다(묶음의 `git log -- harness tests`). 97b3249 → HEAD 사이에 바뀐 파일: harness/baselines.py(새 기준선 rand_dyn 클래스 추가만), config.py(aircraft 데이터셋·STAGE2·STAGE2_MAX_RUNNING·ALLOC_REFS 목록), prepare_data.py(aircraft 준비), report.py(이전 check 데이터셋 표 절), scheduler.py(좀비 job 수거), tests/test_harness.py, agent/loop.sh(stage2_boot 시작). train.py·models.py·data.py·selector_api.py·compare.py·teacher_cache.py는 바뀌지 않았다. CUB 학습 경로와 maskedkd·lateroll이 쓰는 코드는 그대로다.
- 묶음은 과거 잠금 digest를 다시 계산할 수단을 주지 않는다(evidence.py는 지금 digest만 낸다). 이 절은 git 기록으로 확인할 수 있는 범위다. 이것으로 3항이 충족되지 않는다고 판단하면 FAIL이 아니라 그 이유를 적어 주기 바란다(사람에게 하네스 쪽 해결을 요청한다).
- agent/INVALID.md에 이 run들을 덮는 tamper 기록은 없다. 사람 커밋 외 harness/tests 변경은 c5753fd(2026-10-31 round 55 autosave, 사람이 42906db로 복구)뿐이고 인용 run은 모두 2026-11-01 이후다. lateroll selector_sha 69541b27aca9.
