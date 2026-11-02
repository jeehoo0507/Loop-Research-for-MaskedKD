# AUDIT_REQUEST — [claude] 2026-11-02 라운드 184 (CUB lateroll@29 final, 사람 지시 14:30에 따른 재요청)

사람 지시(QUEUE 2026-11-02 08:50 2번, 14:30): stage 2(FGVC-Aircraft)에 lateroll을 올리려면 AUDIT.md에 PASS가 필요하다. 04:55 NEEDS-MORE의 이유는 (a) 사람 커밋 faeb3cd·8f67ca5·bb22ead의 잠금 파일 실제 diff 없음, (b) PROPOSALS와 selector 생성 이력 없음이었다. 사람이 evidence.py에 (a) 인용 run 커밋 → HEAD의 잠금 파일 전체 diff 본문, (b) 인용된 agent 방법의 PROPOSALS 항목과 `git log --follow selectors/lateroll.py`를 넣었다(ec9225f). 주장과 인용 run은 04:55 요청과 같다.

## 주장 (1줄)
CUB-200-2011 K=29에서 lateroll(student 블록 6~11만의 attention rollout top-K)은 final seed 1·2·3 평균 81.39로 같은 seed의 maskedkd@29(80.22)보다 +1.17 [+0.75, +1.58](짝지은 95% CI) 높다. 학습 1스텝 총비용은 12.9900 대 12.9881 GFLOPs/img(비 1.0001)다. 이 차이는 promotion threshold 1.92보다 작아 "의미 있는 이득"이라고 주장하지 않는다.

## run id (status done, tier final, 100 에폭, dataset cub, sel_args {})
| seed | lateroll@29 | maskedkd@29 | 차이 |
|---|---|---|---|
| 1 | final_lateroll_k29_s1_1101_104303_ef38 (82.33) | final_maskedkd_k29_s1_1101_104303_b676 (81.33) | +1.00 |
| 2 | final_lateroll_k29_s2_1101_125958_6470 (80.33) | final_maskedkd_k29_s2_1101_125958_5b02 (79.17) | +1.17 |
| 3 | final_lateroll_k29_s3_1101_140937_ec98 (81.50) | final_maskedkd_k29_s3_1101_140937_b00f (80.17) | +1.33 |

값은 metrics.json `val_acc_last`(소수 둘째 자리 반올림)다.

## 비용 (metrics.json `flops`, GFLOPs/img)
| | total | selector | teacher | student |
|---|---|---|---|---|
| lateroll@29 | 12.9900 | 0.0765 | 5.3608 | 7.5528 |
| maskedkd@29 | 12.9881 | 0.0 | 5.3608 | 7.6273 |

teacher 비용은 같다. lateroll의 selector 비용은 총비용에 들어 있고, lateroll이 0.0019 GFLOPs/img(0.015%) 비싸다. "더 싸다"고 주장하지 않는다.

## 맥락 (주장 아님, 숫자만)
같은 seed·K=29의 다른 기준선 final 평균은 RESULTS_TABLE에 있다(rollout 80.44, tome 79.78, Full KD@196 81.72). 이 비교들은 이번 주장에 넣지 않는다.

## 근거 분석 (방법보다 먼저, AUDIT_PROMPT 12·16항)
analysis_cub_rollout_depth_1031_100143_a4b6 (H-C2b). rollout 시작 블록 0/3/6/8/9/10/11을 student 9개, val 600장에서 비교했다. K=25·12에서 블록 6부터의 rollout이 가린 teacher의 TCKD(soft label의 목표 클래스 확률 보존)를 가장 잘 지켰다(최종 3개 평균 roll0 1.309, roll3 1.126, roll6 1.092). 순서: 분석 → selectors/lateroll.py 추가 → 첫 lateroll run. 묶음의 PROPOSALS 항목(H-C2b, H-C18)과 `git log --follow selectors/lateroll.py`(첫 커밋 95bab8f, 10-31 라운드 65, H-C2b 판정과 같은 커밋)로 순서를 확인할 수 있다.

## 분해 (AUDIT_PROMPT 15항)
analysis_cub_lateroll_decomp_1101_070348_b793 (H-C18). lateroll K=29 screen student 3개(seed 0·4·5). MaskedKD와 겹침 |L∩M|/K = 0.413·0.424·0.481. 겹치지 않는 lateroll 토큰을 lateroll 밖 random 토큰으로 바꾸면 가린 teacher TCKD가 +0.521 [0.418, 0.624]·+0.481 [0.386, 0.579]·+0.309 [0.235, 0.389] 나빠진다. maskedkd 대비 이득은 겹치지 않는 토큰에서 온다.

## 노벨티 (약함)
가장 가까운 선행: attention rollout(Abnar & Zuidema, ACL 2020; harness `rollout`), MaskedKD(Son et al., ECCV 2024), 뒤쪽 층 relevance(Chefer et al., CVPR 2021). lateroll은 rollout 시작 블록을 분석(H-C2b)으로 정한 것이다.

## 주장하지 않는 것 / caveats
- lateroll@29 vs rollout +0.50 < 1.92 → UNPROVEN. maskedkd 대비 +1.17도 threshold보다 작다.
- FLOPs 효율 주장(같은 총비용의 더 큰 K 대비)은 하지 않는다. lateroll@49 final은 진행 중(H-C31).
- Search effort: lateroll 5 configs (K = 12·20·29·49·98, sel_args 모두 {}). final은 K=29만 끝났다.
- final seed 1·2·3은 screen seed 0·4·5와 겹치지 않는다. test_eval은 쓰지 않았다. CUB 하나의 결과다.

## 무결성
인용 run 6개는 지금 잠금 전의 digest(3b594cc1646f, 8ac8a710a4ae)에서 돌았다. 묶음의 "past locks" 절(digest 재계산 MATCH)과 인용 run 커밋 → HEAD의 잠금 파일 diff 본문을 참고하기 바란다. 내 로컬 확인(감사 증거 아님): 그 사이 잠금 파일을 바꾼 커밋은 모두 사람 커밋이고, train.py·models.py·data.py·selector_api.py는 바뀌지 않았다. config.py 변경은 aircraft 항목·STAGE2·STAGE2_MAX_RUNNING·rand_dyn 기준선 추가다. 짝마다 두 방법은 같은 digest다. lateroll selector_sha 69541b27aca9. agent/INVALID.md에 이 run들을 덮는 기록은 없다.
