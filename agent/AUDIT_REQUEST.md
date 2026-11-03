# AUDIT_REQUEST — [claude] 2026-11-03 라운드 245 (CUB lateroll@29 final, 사람 지시 QUEUE 11-03 16:10에 따른 재요청)

사람 지시(QUEUE 2026-11-03 16:10): 02:43 NEEDS-MORE의 이유는 묶음에서 locked-file diff, H-C2b 출력, all-runs 표가 잘린 것이었다. 사람이 묶음 한도를 400k로 올렸다(a5e2e91, 지난 요청 재현 309k, 잘림 0). 주장과 인용 run은 02:43 요청과 같다. 02:43 감사관 요구에 따라 고친 것: (a) caveat의 lateroll − rollout 차이를 +0.94로 바로잡음(+0.50은 내 오류, K=49 lateroll − maskedkd 값이었다), (b) test_eval 미사용 증거를 아래 caveats에 적음.

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
- lateroll@29 vs rollout@29 final은 +0.94 (81.39 − 80.44) < 1.92 → UNPROVEN. maskedkd 대비 +1.17도 threshold보다 작다.
- FLOPs 효율 주장(같은 총비용의 더 큰 K 대비)은 하지 않는다. CE final seed 2·3이 더해진 뒤(H-C33) report.py 허용 오차가 0.25 %p로 줄어, Primary metric에서 lateroll의 최소 K는 49로 maskedkd·rollout과 같다. "더 적은 토큰으로 같은 정확도"는 주장하지 않는다.
- CUB final의 KD 이득 자체가 Full − CE +0.50 [−3.04, +4.04]로 불확실하다(H-C33, H-A19).
- Search effort: lateroll 5 configs (K = 12·20·29·49·98, sel_args 모두 {}). final은 K = 12·29·49.
- final seed 1·2·3은 screen seed 0·4·5와 겹치지 않는다. CUB 하나의 결과다(Aircraft 이전 검증 H-A20은 감사 전으로 진행 중이며 이 주장에 넣지 않는다).
- test_eval 미사용 증거: harness/test_eval.py는 결과를 `final/test_<timestamp>.json`에만 쓴다(그 파일 6행, 36행). 저장소에 `final/` 디렉터리가 없고(`ls final` → No such file or directory), `git log --all -- final`에도 그 경로를 추가한 커밋이 없다. 인용 run 디렉터리에는 log.txt·metrics.json·stdout.txt·student*.pt만 있다. 모든 수치는 val split(`val_acc_last`)이다.

## 무결성
인용 run 6개는 지금 잠금 전의 digest(3b594cc1646f, 8ac8a710a4ae)에서 돌았다. 묶음의 "past locks" 절을 참고하기 바란다(02:43 감사에서 6개 모두 MATCH). 짝마다 두 방법은 같은 digest다. lateroll selector_sha 69541b27aca9. agent/INVALID.md에 이 run들을 덮는 기록은 없다.
