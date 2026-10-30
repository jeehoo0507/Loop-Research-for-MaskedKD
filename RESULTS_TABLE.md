# RESULTS_TABLE (written by harness/report.py; do not edit by hand)

*img/s: training throughput averaged over epochs; depends on how many jobs shared the GPU.

# Stage 1: CIFAR-100

Gate (screen, seed 0): Full KD 24.38 vs CE 21.10 (need +1.0): PASS; MaskedKD@98 24.74 vs Full 24.38 (need within 0.5): PASS

Noise: seed sigma: not measured yet (run harness.submit --baselines)

## Significant only (final tier, paired 95% CI excludes 0; wins AND losses)
- none yet

## Core comparison (final tier: every agent method vs maskedkd, tome, grid, center, random, full, parent)
- no agent method has >= 2 final seeds yet

## Parent check (screen seed 0: a child must beat its META parent by the promotion threshold)
- no child selectors yet

## Primary metric (analysis-only methods excluded: oracle)
### screen: Full KD val 24.38; smallest mean kept tokens within 1.0 %p
- maskedkd : 98.0

## Where token choice matters (maskedkd - random, screen seed 0)
- not measured yet

## Search effort (screen)
Total agent screen configs: 0

## All finished runs
| tier | method | sel_args | keep | mean kept | seeds | val acc (mean ± sd) | total GFLOPs/img | teacher | selector | img/s* | run ids |
|---|---|---|---|---|---|---|---|---|---|---|---|
| screen | ce |  | 0 | None | 0 | 21.10 ± 0.00 | 7.64 | 0.00 | 0.00 | 667 | screen_ce_k0_s0_1029_233749_6609 |
| screen | full |  | 196 | 196.0 | 0 | 24.38 ± 0.00 | 16.84 | 9.20 | 0.00 | 488 | screen_full_k196_s0_1029_233749_5549 |
| screen | maskedkd |  | 98 | 98.0 | 0 | 24.74 ± 0.00 | 12.13 | 4.50 | 0.00 | 493 | screen_maskedkd_k98_s0_1029_233749_7ad0 |

# Stage 2: ImageNet-100 (fixed methods from stage 1; 'stage2' column in metrics.json says untouched/tuned)

Gate (final, seed 1): gate incomplete: ce=None full=None maskedkd98=None

## Core comparison (final)
- none yet

## Primary metric

## All finished runs
- none yet
