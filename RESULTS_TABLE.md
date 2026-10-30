# RESULTS_TABLE (written by harness/report.py; do not edit by hand)

*img/s: training throughput averaged over epochs; depends on how many jobs shared the GPU.
Worst-group acc: Waterbirds groups (y x place) on the val split; never used in training.

# Stage 1: waterbirds (search)

Gate: [waterbirds screen, 100 epochs, mean of seeds [0, 1, 2]] Full KD 90.83 vs CE 89.32 (need +1.0), WGA 66.67 vs 66.42 (or need +2.0): PASS; MaskedKD@98 90.69 vs Full 90.83 (need within 0.5): PASS

Noise: acc seed sigma 0.38 %p (pooled, dof 4) -> threshold 0.76 %p; WGA seed sigma 1.23 %p (pooled, dof 4) -> threshold 2.45 %p

## Significant only (final tier, paired 95% CI excludes 0; wins AND losses)
- none yet

## Core comparison (final tier: every agent method vs maskedkd, rollout, tome, grid, center, random, full, parent)
- no agent method has >= 2 final seeds yet

## Parent check (screen seed 0: a child must beat its META parent by the promotion threshold)
- no child selectors yet

## Primary metric (analysis-only methods excluded: oracle)
### screen: Full KD val 90.83; KD gain +1.50, tolerance 0.75 %p; WGA 66.67, WGA tolerance 2.45 %p; smallest mean kept tokens within it
- maskedkd : 98.0

## Where token choice matters (screen seed 0, each baseline vs Full KD)
- K=49: maskedkd 89.24 / WGA 51.9; rollout 89.66 / WGA 64.7; random 85.40 / WGA 36.1; grid 84.74 / WGA 35.3; center 90.16 / WGA 60.1; tome 89.66 / WGA 51.1 (Full KD 90.66 / WGA 65.4)

## Search effort (screen)
Total agent screen configs: 0

## All finished runs
| tier | method | sel_args | keep | mean kept | seeds | val acc (mean ± sd) | worst-group acc | total GFLOPs/img | teacher | selector | img/s* | run ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| screen | ce |  | 0 | None | 0,1,2 | 89.32 ± 0.71 | 66.42 ± 1.89 | 7.64 | 0.00 | 0.00 | 513 | screen_ce_k0_s0_1030_123806_4f57 screen_ce_k0_s2_1030_130545_295c screen_ce_k0_s1_1030_130545_c89b |
| screen | center |  | 49 | 49.0 | 0 | 90.16 ± 0.00 | 60.15 ± 0.00 | 16.46 | 8.82 | 0.00 | 332 | screen_center_k49_s0_1030_135953_4476 |
| screen | full |  | 196 | 196.0 | 0,1,2 | 90.83 ± 0.17 | 66.67 ± 1.57 | 42.77 | 35.13 | 0.00 | 273 | screen_full_k196_s0_1030_123806_2c89 screen_full_k196_s1_1030_130545_2b61 screen_full_k196_s2_1030_130545_5a4a |
| screen | grid |  | 49 | 49.0 | 0 | 84.74 ± 0.00 | 35.34 ± 0.00 | 16.46 | 8.82 | 0.00 | 337 | screen_grid_k49_s0_1030_135953_19b7 |
| screen | maskedkd |  | 29 | 29.0 | 0 | 87.66 ± 0.00 | 43.61 ± 0.00 | 12.99 | 5.36 | 0.00 | 357 | screen_maskedkd_k29_s0_1030_135953_764a |
| screen | maskedkd |  | 49 | 49.0 | 0 | 89.24 ± 0.00 | 51.88 ± 0.00 | 16.44 | 8.82 | 0.00 | 328 | screen_maskedkd_k49_s0_1030_135953_3e16 |
| screen | maskedkd |  | 98 | 98.0 | 0,1,2 | 90.69 ± 0.51 | 64.66 ± 0.75 | 25.04 | 17.41 | 0.00 | 313 | screen_maskedkd_k98_s2_1030_130545_f22a screen_maskedkd_k98_s1_1030_130545_ce0a screen_maskedkd_k98_s0_1030_123806_3217 |
| screen | random |  | 49 | 49.0 | 0 | 85.40 ± 0.00 | 36.09 ± 0.00 | 16.46 | 8.82 | 0.00 | 333 | screen_random_k49_s0_1030_135953_14e9 |
| screen | rollout |  | 49 | 49.0 | 0 | 89.66 ± 0.00 | 64.66 ± 0.00 | 16.45 | 8.82 | 0.17 | 274 | screen_rollout_k49_s0_1030_135953_a1a9 |
| screen | tome |  | 49 | 49.0 | 0 | 89.66 ± 0.00 | 51.13 ± 0.00 | 14.50 | 6.86 | 0.00 | 346 | screen_tome_k49_s0_1030_135953_1e32 |

# Stage 2: cub (fixed methods from stage 1; 'stage2' in metrics.json says untouched/tuned)

Gate: gate incomplete: ce seeds [], full seeds [], maskedkd seeds []

Noise: acc seed sigma: not measured yet (run harness.submit --baselines)

## Significant only (final tier, paired 95% CI excludes 0; wins AND losses)
- none yet

## Core comparison (final tier: every agent method vs maskedkd, rollout, tome, grid, center, random, full, parent)
- no agent method has >= 2 final seeds yet

## Parent check (screen seed 0: a child must beat its META parent by the promotion threshold)
- no child selectors yet

## Primary metric (analysis-only methods excluded: oracle)

## Where token choice matters (screen seed 0, each baseline vs Full KD)
- not measured yet

## Search effort (screen)
Total agent screen configs: 0

## All finished runs
- none yet

