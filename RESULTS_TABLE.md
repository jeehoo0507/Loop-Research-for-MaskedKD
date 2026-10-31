# RESULTS_TABLE (written by harness/report.py; do not edit by hand)

*img/s: training throughput averaged over epochs; depends on how many jobs shared the GPU.
Worst-group acc (Waterbirds groups y x place, val split): a diagnostic column only, never used in training or
in any decision here. Decisions use mean accuracy and measured train-step GFLOPs.

# Stage 1: cub (search)

Gate: [cub screen, 100 epochs, mean of seeds [0, 4, 5]] Full KD 82.17 vs CE 80.83 (need +1.0): PASS; MaskedKD@98 82.56 vs Full 82.17 (need >= Full - 0.5): PASS

Noise: acc seed sigma 1.35 %p (pooled, dof 4) -> threshold 2.69 %p

## Significant only (final tier, paired 95% CI excludes 0; wins AND losses)
- none yet

## Core comparison (final tier: every agent method vs maskedkd, rollout, tome, grid, center, random, full, parent)
- no agent method has >= 2 final seeds yet

## Parent check (screen seed 0: a child must beat its META parent by the promotion threshold)
- lateroll@12 vs parent rollout: acc +0.00 %p (threshold 2.69) -> added part UNPROVEN
- lateroll@29 vs parent rollout: acc +0.50 %p (threshold 2.69) -> added part UNPROVEN
- lateroll@49 vs parent rollout: acc +1.00 %p (threshold 2.69) -> added part UNPROVEN
- sinkfree@12 vs parent maskedkd: acc +1.50 %p (threshold 2.69) -> added part UNPROVEN
- sinkfree@29 vs parent maskedkd: acc -0.33 %p (threshold 2.69) -> added part UNPROVEN
- sinkfree@49 vs parent maskedkd: acc +0.33 %p (threshold 2.69) -> added part UNPROVEN

## Primary metric (analysis-only methods excluded: oracle)
### final: Full KD val 82.83 (seeds 1); KD gain +1.83, tolerance 0.92 %p; smallest mean kept tokens within it
- claim-grade (>= 3 seeds): none
- screening only (fewer seeds):
  - maskedkd : 98.0 (1 seeds)
### screen: Full KD val 82.17 (seeds 0,4,5); KD gain +1.33, tolerance 0.67 %p; smallest mean kept tokens within it
- claim-grade (>= 3 seeds):
  - maskedkd : 98.0 (3 seeds)
- screening only (fewer seeds):
  - lateroll@69541b : 49.0 (1 seeds)

## Where token choice matters (screen seed 0, each baseline vs Full KD)
- K=49: maskedkd 80.50; rollout 81.67; random 78.83; grid 77.33; center 80.17; tome 79.00 (Full KD 82.17)
- K=29: maskedkd 79.17; rollout 80.83; random 74.00; grid 74.50; center 79.00; tome 79.67 (Full KD 82.17)
- K=12: maskedkd 76.00; rollout 78.33; random 75.33; grid 74.33; center 76.00; tome 78.17 (Full KD 82.17)

## Search effort (screen)
Total agent screen configs: 6
- sinkfree: 3 configs
- lateroll: 3 configs

## All finished runs
| tier | method | sel_args | keep | mean kept | seeds | val acc (mean ± sd) | worst-group acc | total GFLOPs/img | teacher | selector | img/s* | run ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final | ce |  | 0 | None | 1 | 81.00 ± 0.00 | - | 7.64 | 0.00 | 0.00 | 511 | cub_final_ce_k0_s1_1030_153055_1256 |
| final | full |  | 196 | 196.0 | 1 | 82.83 ± 0.00 | - | 42.77 | 35.13 | 0.00 | 294 | cub_final_full_k196_s1_1030_153055_d87b |
| final | maskedkd |  | 12 | 12.0 | 2,3 | 78.00 ± 0.00 | - | 10.07 | 2.45 | 0.00 | 880 | cub_final_maskedkd_k12_s2_1031_103004_2d8c cub_final_maskedkd_k12_s3_1031_103017_3c57 |
| final | maskedkd |  | 98 | 98.0 | 1 | 83.50 ± 0.00 | - | 25.04 | 17.41 | 0.00 | 298 | cub_final_maskedkd_k98_s1_1030_153055_cdbc |
| final | random |  | 12 | 12.0 | 2 | 76.00 ± 0.00 | - | 10.09 | 2.45 | 0.00 | 1258 | cub_final_random_k12_s2_1031_103009_1917 |
| final | rollout |  | 12 | 12.0 | 2,3 | 78.33 ± 0.94 | - | 10.08 | 2.45 | 0.17 | 984 | cub_final_rollout_k12_s3_1031_103013_2dcf cub_final_rollout_k12_s2_1031_102956_fadd |
| screen | ce |  | 0 | None | 0,4,5 | 80.83 ± 1.96 | - | 7.64 | 0.00 | 0.00 | 1482 | screen_ce_k0_s4_1031_114911_08b9 screen_ce_k0_s5_1031_114911_5fdb screen_ce_k0_s0_1031_111002_24e1 |
| screen | center |  | 12 | 12.0 | 0 | 76.00 ± 0.00 | - | 10.09 | 2.45 | 0.00 | 1264 | screen_center_k12_s0_1031_130151_50b4 |
| screen | center |  | 29 | 29.0 | 0 | 79.00 ± 0.00 | - | 13.00 | 5.36 | 0.00 | 1181 | screen_center_k29_s0_1031_130150_cbde |
| screen | center |  | 49 | 49.0 | 0 | 80.17 ± 0.00 | - | 16.46 | 8.82 | 0.00 | 1093 | screen_center_k49_s0_1031_130150_3629 |
| screen | full |  | 196 | 196.0 | 0,4,5 | 82.17 ± 0.17 | - | 42.77 | 35.13 | 0.00 | 708 | screen_full_k196_s4_1031_114911_a837 screen_full_k196_s5_1031_114911_9c0a screen_full_k196_s0_1031_111002_00d5 |
| screen | grid |  | 12 | 12.0 | 0 | 74.33 ± 0.00 | - | 10.09 | 2.45 | 0.00 | 1266 | screen_grid_k12_s0_1031_130151_33bd |
| screen | grid |  | 29 | 29.0 | 0 | 74.50 ± 0.00 | - | 13.00 | 5.36 | 0.00 | 1183 | screen_grid_k29_s0_1031_130150_df6c |
| screen | grid |  | 49 | 49.0 | 0 | 77.33 ± 0.00 | - | 16.46 | 8.82 | 0.00 | 1094 | screen_grid_k49_s0_1031_130150_702e |
| screen | lateroll@69541b |  | 12 | 12.0 | 0 | 78.33 ± 0.00 | - | 10.08 | 2.45 | 0.08 | 1108 | screen_lateroll_k12_s0_1031_130339_f599 |
| screen | lateroll@69541b |  | 29 | 29.0 | 0 | 81.33 ± 0.00 | - | 12.99 | 5.36 | 0.08 | 1043 | screen_lateroll_k29_s0_1031_130348_fd19 |
| screen | lateroll@69541b |  | 49 | 49.0 | 0 | 82.67 ± 0.00 | - | 16.45 | 8.82 | 0.08 | 974 | screen_lateroll_k49_s0_1031_130352_adc5 |
| screen | maskedkd |  | 12 | 12.0 | 0 | 76.00 ± 0.00 | - | 10.07 | 2.45 | 0.00 | 1240 | screen_maskedkd_k12_s0_1031_130150_a887 |
| screen | maskedkd |  | 29 | 29.0 | 0,4 | 79.42 ± 0.35 | - | 12.99 | 5.36 | 0.00 | 1161 | screen_maskedkd_k29_s4_1031_183523_7354 screen_maskedkd_k29_s0_1031_130150_1393 |
| screen | maskedkd |  | 49 | 49.0 | 0,4,5 | 80.78 ± 0.48 | - | 16.44 | 8.82 | 0.00 | 1035 | screen_maskedkd_k49_s5_1031_172301_3f83 screen_maskedkd_k49_s0_1031_130150_e26e screen_maskedkd_k49_s4_1031_172257_73af |
| screen | maskedkd |  | 98 | 98.0 | 0,4,5 | 82.56 ± 1.90 | - | 25.04 | 17.41 | 0.00 | 916 | screen_maskedkd_k98_s0_1031_111002_5715 screen_maskedkd_k98_s4_1031_114911_7153 screen_maskedkd_k98_s5_1031_114911_80fa |
| screen | oracle |  | 12 | 12.0 | 0 | 76.33 ± 0.00 | - | 45.21 | 2.45 | 35.13 | 638 | screen_oracle_k12_s0_1031_130151_6802 |
| screen | random |  | 12 | 12.0 | 0 | 75.33 ± 0.00 | - | 10.09 | 2.45 | 0.00 | 1266 | screen_random_k12_s0_1031_130151_7444 |
| screen | random |  | 29 | 29.0 | 0 | 74.00 ± 0.00 | - | 13.00 | 5.36 | 0.00 | 1184 | screen_random_k29_s0_1031_130150_a433 |
| screen | random |  | 49 | 49.0 | 0 | 78.83 ± 0.00 | - | 16.46 | 8.82 | 0.00 | 1094 | screen_random_k49_s0_1031_130150_91ee |
| screen | rollout |  | 12 | 12.0 | 0 | 78.33 ± 0.00 | - | 10.08 | 2.45 | 0.17 | 983 | screen_rollout_k12_s0_1031_130151_9b9e |
| screen | rollout |  | 29 | 29.0 | 0,4,5 | 80.94 ± 0.35 | - | 12.99 | 5.36 | 0.17 | 931 | screen_rollout_k29_s4_1031_183515_ff99 screen_rollout_k29_s0_1031_130150_5726 screen_rollout_k29_s5_1031_183519_47eb |
| screen | rollout |  | 49 | 49.0 | 0,4,5 | 81.28 ± 0.35 | - | 16.45 | 8.82 | 0.17 | 878 | screen_rollout_k49_s4_1031_172246_23ee screen_rollout_k49_s0_1031_130150_674d screen_rollout_k49_s5_1031_172253_99b6 |
| screen | sinkfree@60bc72 |  | 12 | 12.0 | 0 | 77.50 ± 0.00 | - | 10.07 | 2.45 | 0.00 | 1240 | screen_sinkfree_k12_s0_1031_180704_f0b3 |
| screen | sinkfree@60bc72 |  | 29 | 29.0 | 0 | 78.83 ± 0.00 | - | 12.99 | 5.36 | 0.00 | 1161 | screen_sinkfree_k29_s0_1031_180708_1cb6 |
| screen | sinkfree@60bc72 |  | 49 | 49.0 | 0 | 80.83 ± 0.00 | - | 16.44 | 8.82 | 0.00 | 1076 | screen_sinkfree_k49_s0_1031_180713_dba8 |
| screen | tome (teacher cost x2.69 of budget: not matched) |  | 12 | 12.0 | 0 | 78.17 ± 0.00 | - | 12.00 | 4.36 | 0.00 | 1183 | screen_tome_k12_s0_1031_130151_473d |
| screen | tome (teacher cost x1.16 of budget: not matched) |  | 29 | 29.0 | 0 | 79.67 ± 0.00 | - | 12.00 | 4.36 | 0.00 | 1185 | screen_tome_k29_s0_1031_130150_b286 |
| screen | tome |  | 49 | 49.0 | 0 | 79.00 ± 0.00 | - | 14.50 | 6.86 | 0.00 | 1110 | screen_tome_k49_s0_1031_130150_83af |

# Stage 2: waterbirds (fixed methods from stage 1; 'stage2' in metrics.json says untouched/tuned)

Gate: [waterbirds final, 100 epochs, mean of seeds [1]] Full KD 91.33 vs CE 90.08 (need +1.0): PASS; MaskedKD@98 90.33 vs Full 91.33 (need >= Full - 0.5): FAIL

Noise: acc seed sigma 0.44 %p (pooled, dof 7) -> threshold 0.88 %p (diagnostic: WGA seed sigma 2.78 %p (pooled, dof 7) -> threshold 5.56 %p)

## Significant only (final tier, paired 95% CI excludes 0; wins AND losses)
- gradroll@12 vs maskedkd@12 [acc]: +3.78 %p, 95% CI [+2.74, +4.82], seeds 1,2,3, GFLOPs ratio 1.272 -> WIN
- gradroll@12 vs grid@12 [acc]: +4.89 %p, 95% CI [+3.42, +6.36], seeds 1,2,3, GFLOPs ratio 1.270 -> WIN
- gradroll@12 vs random@12 [acc]: +4.14 %p, 95% CI [+1.03, +7.25], seeds 1,2,3, GFLOPs ratio 1.270 -> WIN
- gradroll@12 vs full@196 [acc]: -3.17 %p, 95% CI [-5.48, -0.86], seeds 1,2,3, GFLOPs ratio 0.300 -> LOSS
- suffdisc@12 vs maskedkd@12 [acc]: +5.42 %p, 95% CI [+4.18, +6.66], seeds 1,2,3, GFLOPs ratio 1.491 -> WIN
- suffdisc@12 vs grid@12 [acc]: +6.53 %p, 95% CI [+5.51, +7.56], seeds 1,2,3, GFLOPs ratio 1.489 -> WIN
- suffdisc@12 vs random@12 [acc]: +5.78 %p, 95% CI [+3.43, +8.14], seeds 1,2,3, GFLOPs ratio 1.489 -> WIN

## Core comparison (final tier: every agent method vs maskedkd, rollout, tome, grid, center, random, full, parent)
- gradroll@12 vs maskedkd@12 [acc]: +3.78 %p, 95% CI [+2.74, +4.82], seeds 1,2,3, GFLOPs ratio 1.272 -> WIN
- gradroll@12 vs rollout@12 [acc]: -0.11 %p, 95% CI [-3.71, +3.49], seeds 1,2,3, GFLOPs ratio 1.271 -> no difference
- gradroll@12 vs tome@12 [acc]: -1.97 %p, 95% CI [-4.05, +0.10], seeds 1,2,3, GFLOPs ratio 1.068 -> no difference
- gradroll@12 vs grid@12 [acc]: +4.89 %p, 95% CI [+3.42, +6.36], seeds 1,2,3, GFLOPs ratio 1.270 -> WIN
- gradroll@12 vs center@12 [acc]: -0.70 %p, 95% CI [-1.53, +0.14], seeds 1,2,3, GFLOPs ratio 1.270 -> no difference
- gradroll@12 vs random@12 [acc]: +4.14 %p, 95% CI [+1.03, +7.25], seeds 1,2,3, GFLOPs ratio 1.270 -> WIN
- gradroll@12 vs full@196 [acc]: -3.17 %p, 95% CI [-5.48, -0.86], seeds 1,2,3, GFLOPs ratio 0.300 -> LOSS
- gradroll@12 vs rollout@12 [acc]: -0.11 %p, 95% CI [-3.71, +3.49], seeds 1,2,3, GFLOPs ratio 1.271 -> no difference
- suffdisc@12 vs maskedkd@12 [acc]: +5.42 %p, 95% CI [+4.18, +6.66], seeds 1,2,3, GFLOPs ratio 1.491 -> WIN
- suffdisc@12 vs rollout@12 [acc]: +1.53 %p, 95% CI [-2.52, +5.58], seeds 1,2,3, GFLOPs ratio 1.491 -> no difference
- suffdisc@12 vs tome@12 [acc]: -0.33 %p, 95% CI [-2.57, +1.91], seeds 1,2,3, GFLOPs ratio 1.252 -> no difference
- suffdisc@12 vs grid@12 [acc]: +6.53 %p, 95% CI [+5.51, +7.56], seeds 1,2,3, GFLOPs ratio 1.489 -> WIN
- suffdisc@12 vs center@12 [acc]: +0.95 %p, 95% CI [-0.79, +2.68], seeds 1,2,3, GFLOPs ratio 1.489 -> no difference
- suffdisc@12 vs random@12 [acc]: +5.78 %p, 95% CI [+3.43, +8.14], seeds 1,2,3, GFLOPs ratio 1.489 -> WIN
- suffdisc@12 vs full@196 [acc]: -1.53 %p, 95% CI [-3.49, +0.43], seeds 1,2,3, GFLOPs ratio 0.351 -> no difference
- suffdisc@12 vs center@12 [acc]: +0.95 %p, 95% CI [-0.79, +2.68], seeds 1,2,3, GFLOPs ratio 1.489 -> no difference

## Parent check (screen seed 0: a child must beat its META parent by the promotion threshold)
- adaroll@49 vs parent rollout: acc +0.42 %p (threshold 0.88) -> added part UNPROVEN
- gradrep@12 vs parent gradroll: acc -0.17 %p (threshold 0.88) -> added part UNPROVEN
- gradroll@12 vs parent rollout: acc +0.92 %p (threshold 0.88) -> added part helps
- gradroll@29 vs parent rollout: acc -0.33 %p (threshold 0.88) -> added part UNPROVEN
- jitdisc@12 vs parent suffdisc: acc -0.42 %p (threshold 0.88) -> added part UNPROVEN
- priorroll@12 vs parent rollout: acc +1.83 %p (threshold 0.88) -> added part helps
- priorroll@29 vs parent rollout: acc +0.00 %p (threshold 0.88) -> added part UNPROVEN
- suffdisc@12 vs parent center: acc -0.17 %p (threshold 0.88) -> added part UNPROVEN
- suffdisc@29 vs parent center: acc +1.17 %p (threshold 0.88) -> added part helps
- suffdisc@49 vs parent center: acc +0.08 %p (threshold 0.88) -> added part UNPROVEN
- suffstride@12 vs parent suffdisc: acc +0.67 %p (threshold 0.88) -> added part UNPROVEN

## Primary metric (analysis-only methods excluded: oracle)
### final: Full KD val 91.16 (seeds 1,2,3); KD gain +1.08, tolerance 0.54 %p; smallest mean kept tokens within it
- claim-grade (>= 3 seeds): none
- screening only (fewer seeds): none
### screen: Full KD val 90.74 (seeds 0,1,2,4,5,6); KD gain +1.42, tolerance 0.71 %p; smallest mean kept tokens within it
- claim-grade (>= 3 seeds):
  - suffdisc@0f0600 : 29.0 (7 seeds)
  - center : 49.0 (4 seeds)
  - maskedkd : 98.0 (3 seeds)
- screening only (fewer seeds):
  - adaroll@069f50 : 48.5 (1 seeds)
  - suffdisc@0f0600 : 49.0 (1 seeds)

## Where token choice matters (screen seed 0, each baseline vs Full KD)
- K=49: maskedkd 89.24 / WGA 51.9; rollout 89.66 / WGA 64.7; random 85.40 / WGA 36.1; grid 84.74 / WGA 35.3; center 90.16 / WGA 60.1; tome 89.66 / WGA 51.1 (Full KD 90.66 / WGA 65.4)
- K=29: maskedkd 87.66 / WGA 43.6; rollout 89.82 / WGA 65.4; random 82.49 / WGA 27.8; grid 81.65 / WGA 36.8; center 88.99 / WGA 62.4; tome 89.66 / WGA 51.1 (Full KD 90.66 / WGA 65.4)
- K=12: maskedkd 85.15 / WGA 23.3; rollout 87.24 / WGA 45.1; random 83.49 / WGA 33.8; grid 83.90 / WGA 35.3; center 88.74 / WGA 56.4; tome 88.99 / WGA 48.1 (Full KD 90.66 / WGA 65.4)

## Search effort (screen)
Total agent screen configs: 15
- suffdisc: 4 configs
- priorroll: 2 configs
- midattn: 2 configs
- gradroll: 2 configs
- jitdisc: 1 configs
- gradrep: 1 configs
- adaroll: 1 configs
- suffstride: 1 configs
- conceptcov: 1 configs

## All finished runs
| tier | method | sel_args | keep | mean kept | seeds | val acc (mean ± sd) | worst-group acc | total GFLOPs/img | teacher | selector | img/s* | run ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final | ce |  | 0 | None | 1 | 90.08 ± 0.00 | 68.42 ± 0.00 | 7.64 | 0.00 | 0.00 | 488 | waterbirds_final_ce_k0_s1_1031_130203_4b89 |
| final | center |  | 12 | 12.0 | 1,2,3 | 88.69 ± 0.43 | 53.64 ± 1.57 | 10.09 | 2.45 | 0.00 | 304 | final_center_k12_s1_1030_201811_341b final_center_k12_s3_1030_220958_413c final_center_k12_s2_1030_212126_5486 |
| final | full |  | 196 | 196.0 | 1,2,3 | 91.16 ± 0.52 | 67.42 ± 0.43 | 42.77 | 35.13 | 0.00 | 196 | final_full_k196_s1_1030_201811_b764 final_full_k196_s2_1030_212126_a1dc final_full_k196_s3_1030_220958_8d7e |
| final | gradroll@4f3962 |  | 12 | 12.0 | 1,2,3 | 87.99 ± 0.42 | 59.15 ± 6.31 | 12.81 | 2.45 | 2.90 | 195 | final_gradroll_k12_s3_1030_220958_176b final_gradroll_k12_s1_1030_201811_faf0 final_gradroll_k12_s2_1030_212126_da50 |
| final | grid |  | 12 | 12.0 | 1,2,3 | 83.10 ± 0.84 | 35.34 ± 0.00 | 10.09 | 2.45 | 0.00 | 301 | final_grid_k12_s2_1030_212126_9336 final_grid_k12_s3_1030_220958_14eb final_grid_k12_s1_1030_201811_dc71 |
| final | maskedkd |  | 12 | 12.0 | 1,2,3 | 84.21 ± 0.05 | 22.31 ± 1.15 | 10.07 | 2.45 | 0.00 | 280 | final_maskedkd_k12_s1_1030_201811_e49a final_maskedkd_k12_s2_1030_212126_ad00 final_maskedkd_k12_s3_1030_220958_cc4e |
| final | maskedkd |  | 98 | 98.0 | 1 | 90.33 ± 0.00 | 66.92 ± 0.00 | 25.04 | 17.41 | 0.00 | 387 | waterbirds_final_maskedkd_k98_s1_1031_130203_5bef |
| final | random |  | 12 | 12.0 | 1,2,3 | 83.85 ± 0.91 | 37.34 ± 1.89 | 10.09 | 2.45 | 0.00 | 320 | final_random_k12_s2_1030_212126_b5bc final_random_k12_s1_1030_201811_19bf final_random_k12_s3_1030_220958_e1db |
| final | rollout |  | 12 | 12.0 | 1,2,3 | 88.10 ± 1.17 | 50.88 ± 1.74 | 10.08 | 2.45 | 0.17 | 237 | final_rollout_k12_s1_1030_201811_6293 final_rollout_k12_s3_1030_220958_d961 final_rollout_k12_s2_1030_212126_33d1 |
| final | suffdisc@0f0600 |  | 12 | 12.0 | 1,2,3 | 89.63 ± 0.46 | 62.40 ± 2.60 | 15.02 | 2.45 | 4.93 | 192 | final_suffdisc_k12_s3_1031_002453_9ebe final_suffdisc_k12_s1_1031_002441_5d86 final_suffdisc_k12_s2_1031_002448_664e |
| final | tome (teacher cost x2.69 of budget: not matched) |  | 12 | 12.0 | 1,2,3 | 89.96 ± 0.46 | 50.88 ± 1.89 | 12.00 | 4.36 | 0.00 | 292 | final_tome_k12_s2_1030_212126_9500 final_tome_k12_s3_1030_220958_918a final_tome_k12_s1_1030_201811_9fb4 |
| screen | adaroll@069f50 |  | 49 | 48.5 | 0 | 90.08 ± 0.00 | 63.91 ± 0.00 | 16.36 | 8.73 | 0.17 | 206 | screen_adaroll_k49_s0_1030_192757_7edc |
| screen | ce |  | 0 | None | 0,1,2 | 89.32 ± 0.71 | 66.42 ± 1.89 | 7.64 | 0.00 | 0.00 | 513 | screen_ce_k0_s0_1030_123806_4f57 screen_ce_k0_s2_1030_130545_295c screen_ce_k0_s1_1030_130545_c89b |
| screen | center |  | 12 | 12.0 | 0,1,2,3,4,5,6 | 88.54 ± 0.42 | 54.03 ± 2.47 | 10.09 | 2.45 | 0.00 | 308 | screen_center_k12_s0_1030_135953_c64e screen_center_k12_s6_1031_051116_3b33 screen_center_k12_s3_1030_233032_bf21 screen_center_k12_s5_1031_051111_594a screen_center_k12_s4_1031_051107_b568 screen_center_k12_s1_1030_185949_1766 screen_center_k12_s2_1030_233024_3dcf |
| screen | center |  | 29 | 29.0 | 0,1,2,3 | 89.37 ± 0.42 | 60.71 ± 2.16 | 13.00 | 5.36 | 0.00 | 282 | screen_center_k29_s2_1031_025420_4cff screen_center_k29_s1_1030_182732_85c3 screen_center_k29_s0_1030_135953_2878 screen_center_k29_s3_1031_025424_fc02 |
| screen | center |  | 49 | 49.0 | 0,4,5,6 | 90.62 ± 0.36 | 64.28 ± 4.18 | 16.46 | 8.82 | 0.00 | 275 | screen_center_k49_s0_1030_135953_4476 screen_center_k49_s4_1031_064043_463c screen_center_k49_s6_1031_064052_a909 screen_center_k49_s5_1031_064047_c731 |
| screen | conceptcov@5bd86b |  | 12 | 12.0 | 0 | 81.82 ± 0.00 | 27.07 ± 0.00 | 10.12 | 2.45 | 0.03 | 394 | screen_conceptcov_k12_s0_1031_000621_b88e |
| screen | full |  | 196 | 196.0 | 0,1,2,4,5,6 | 90.74 ± 0.41 | 66.54 ± 3.25 | 42.77 | 35.13 | 0.00 | 228 | screen_full_k196_s5_1031_073620_9291 screen_full_k196_s6_1031_073628_b37a screen_full_k196_s0_1030_123806_2c89 screen_full_k196_s4_1031_071437_e327 screen_full_k196_s1_1030_130545_2b61 screen_full_k196_s2_1030_130545_5a4a |
| screen | gradrep@fce177 |  | 12 | 12.0 | 0 | 87.99 ± 0.00 | 54.89 ± 0.00 | 12.81 | 2.45 | 2.90 | 233 | screen_gradrep_k12_s0_1030_193817_c83d |
| screen | gradroll@4f3962 |  | 12 | 12.0 | 0 | 88.16 ± 0.00 | 60.90 ± 0.00 | 12.81 | 2.45 | 2.90 | 204 | screen_gradroll_k12_s0_1030_184259_e2cd |
| screen | gradroll@4f3962 |  | 29 | 29.0 | 0 | 89.49 ± 0.00 | 63.91 ± 0.00 | 15.73 | 5.36 | 2.90 | 197 | screen_gradroll_k29_s0_1030_184255_9017 |
| screen | grid |  | 12 | 12.0 | 0,4,5,6 | 84.38 ± 0.52 | 35.72 ± 0.97 | 10.09 | 2.45 | 0.00 | 311 | screen_grid_k12_s6_1031_060900_c468 screen_grid_k12_s4_1031_054509_b45f screen_grid_k12_s5_1031_060856_ae60 screen_grid_k12_s0_1030_135953_1e84 |
| screen | grid |  | 29 | 29.0 | 0 | 81.65 ± 0.00 | 36.84 ± 0.00 | 13.00 | 5.36 | 0.00 | 283 | screen_grid_k29_s0_1030_135953_a69e |
| screen | grid |  | 49 | 49.0 | 0 | 84.74 ± 0.00 | 35.34 ± 0.00 | 16.46 | 8.82 | 0.00 | 337 | screen_grid_k49_s0_1030_135953_19b7 |
| screen | jitdisc@8a1e65 |  | 12 | 12.0 | 0,1,2,3 | 88.74 ± 0.54 | 54.32 ± 1.67 | 10.09 | 2.45 | 0.00 | 481 | screen_jitdisc_k12_s2_1031_020851_1187 screen_jitdisc_k12_s0_1030_195912_ead1 screen_jitdisc_k12_s1_1030_195921_48c7 screen_jitdisc_k12_s3_1031_020901_8cd5 |
| screen | maskedkd |  | 12 | 12.0 | 0,4,5,6 | 84.78 ± 0.28 | 21.43 ± 2.26 | 10.07 | 2.45 | 0.00 | 315 | screen_maskedkd_k12_s6_1031_040448_eb48 screen_maskedkd_k12_s5_1031_040439_3cc2 screen_maskedkd_k12_s0_1030_135953_ba8e screen_maskedkd_k12_s4_1031_040431_c049 |
| screen | maskedkd |  | 29 | 29.0 | 0 | 87.66 ± 0.00 | 43.61 ± 0.00 | 12.99 | 5.36 | 0.00 | 357 | screen_maskedkd_k29_s0_1030_135953_764a |
| screen | maskedkd |  | 49 | 49.0 | 0,4,5,6 | 88.89 ± 0.60 | 50.56 ± 1.78 | 16.44 | 8.82 | 0.00 | 281 | screen_maskedkd_k49_s4_1031_064030_4c7b screen_maskedkd_k49_s6_1031_064039_1864 screen_maskedkd_k49_s5_1031_064034_443d screen_maskedkd_k49_s0_1030_135953_3e16 |
| screen | maskedkd |  | 98 | 98.0 | 0,1,2 | 90.69 ± 0.51 | 64.66 ± 0.75 | 25.04 | 17.41 | 0.00 | 313 | screen_maskedkd_k98_s2_1030_130545_f22a screen_maskedkd_k98_s1_1030_130545_ce0a screen_maskedkd_k98_s0_1030_123806_3217 |
| screen | midattn@7283f1 |  | 12 | 12.0 | 0 | 87.66 ± 0.00 | 30.08 ± 0.00 | 10.07 | 2.45 | 0.00 | 422 | screen_midattn_k12_s0_1030_170938_9fc6 |
| screen | midattn@7283f1 |  | 49 | 49.0 | 0 | 89.91 ± 0.00 | 57.89 ± 0.00 | 16.44 | 8.82 | 0.00 | 272 | screen_midattn_k49_s0_1030_153152_3b79 |
| screen | oracle |  | 12 | 12.0 | 0 | 87.49 ± 0.00 | 50.38 ± 0.00 | 45.21 | 2.45 | 35.13 | 248 | screen_oracle_k12_s0_1030_135953_d16c |
| screen | priorroll@6e20f5 |  | 12 | 12.0 | 0,1 | 88.91 ± 0.24 | 54.14 ± 1.07 | 10.08 | 2.45 | 0.17 | 272 | screen_priorroll_k12_s0_1030_175140_405f screen_priorroll_k12_s1_1030_185957_7cf9 |
| screen | priorroll@6e20f5 |  | 29 | 29.0 | 0,1 | 89.62 ± 0.29 | 60.52 ± 0.53 | 12.99 | 5.36 | 0.17 | 338 | screen_priorroll_k29_s0_1030_175144_018d screen_priorroll_k29_s1_1030_182720_dbfd |
| screen | random |  | 12 | 12.0 | 0,4,5,6 | 84.11 ± 1.17 | 33.27 ± 1.67 | 10.09 | 2.45 | 0.00 | 320 | screen_random_k12_s0_1030_135953_bc03 screen_random_k12_s4_1031_060904_b8db screen_random_k12_s5_1031_061741_3967 screen_random_k12_s6_1031_061745_8dca |
| screen | random |  | 29 | 29.0 | 0 | 82.49 ± 0.00 | 27.82 ± 0.00 | 13.00 | 5.36 | 0.00 | 285 | screen_random_k29_s0_1030_135953_8780 |
| screen | random |  | 49 | 49.0 | 0 | 85.40 ± 0.00 | 36.09 ± 0.00 | 16.46 | 8.82 | 0.00 | 333 | screen_random_k49_s0_1030_135953_14e9 |
| screen | rollout |  | 12 | 12.0 | 0,4,5,6 | 87.78 ± 0.72 | 48.69 ± 4.32 | 10.08 | 2.45 | 0.17 | 284 | screen_rollout_k12_s5_1031_051059_49f9 screen_rollout_k12_s0_1030_135953_489a screen_rollout_k12_s6_1031_051103_40bf screen_rollout_k12_s4_1031_051050_86f8 |
| screen | rollout |  | 29 | 29.0 | 0,1,2,3 | 89.20 ± 0.46 | 63.16 ± 1.84 | 12.99 | 5.36 | 0.17 | 282 | screen_rollout_k29_s3_1031_025433_3854 screen_rollout_k29_s0_1030_135953_7320 screen_rollout_k29_s1_1030_182725_2086 screen_rollout_k29_s2_1031_025429_73a1 |
| screen | rollout |  | 49 | 49.0 | 0 | 89.66 ± 0.00 | 64.66 ± 0.00 | 16.45 | 8.82 | 0.17 | 274 | screen_rollout_k49_s0_1030_135953_a1a9 |
| screen | suffdisc@0f0600 | {"r": 1} | 12 | 12.0 | 0,1 | 88.78 ± 0.29 | 56.02 ± 0.53 | 11.86 | 2.45 | 1.78 | 260 | screen_suffdisc_k12_s1_1031_011604_6add screen_suffdisc_k12_s0_1031_011600_6b15 |
| screen | suffdisc@0f0600 |  | 12 | 12.0 | 0,1,2,3,4,5,6 | 89.79 ± 0.49 | 62.08 ± 3.90 | 15.02 | 2.45 | 4.93 | 187 | screen_suffdisc_k12_s2_1030_233019_98dd screen_suffdisc_k12_s6_1031_040444_bb51 screen_suffdisc_k12_s3_1030_233028_39ad screen_suffdisc_k12_s5_1031_040435_4e00 screen_suffdisc_k12_s0_1030_173118_504a screen_suffdisc_k12_s1_1030_185953_9455 screen_suffdisc_k12_s4_1031_040422_fa2a |
| screen | suffdisc@0f0600 |  | 29 | 29.0 | 0,1,2,3,4,5,6 | 90.52 ± 0.63 | 64.66 ± 3.33 | 22.62 | 5.36 | 9.62 | 179 | screen_suffdisc_k29_s2_1031_025411_5cac screen_suffdisc_k29_s4_1031_071433_b562 screen_suffdisc_k29_s0_1031_011608_3604 screen_suffdisc_k29_s6_1031_073624_f04c screen_suffdisc_k29_s5_1031_071442_6492 screen_suffdisc_k29_s3_1031_025416_358a screen_suffdisc_k29_s1_1031_022639_643b |
| screen | suffdisc@0f0600 |  | 49 | 49.0 | 0 | 90.24 ± 0.00 | 60.90 ± 0.00 | 31.75 | 8.82 | 15.29 | 201 | screen_suffdisc_k49_s0_1031_012539_2b13 |
| screen | suffstride@6afabc |  | 12 | 12.0 | 0 | 89.24 ± 0.00 | 60.90 ± 0.00 | 11.86 | 2.45 | 1.78 | 260 | screen_suffstride_k12_s0_1031_074452_d5ef |
| screen | tome (teacher cost x2.69 of budget: not matched) |  | 12 | 12.0 | 0,4,5,6 | 89.57 ± 0.69 | 49.44 ± 1.67 | 12.00 | 4.36 | 0.00 | 311 | screen_tome_k12_s0_1030_135953_84af screen_tome_k12_s4_1031_051120_5c90 screen_tome_k12_s6_1031_054505_d130 screen_tome_k12_s5_1031_051124_f881 |
| screen | tome (teacher cost x1.16 of budget: not matched) |  | 29 | 29.0 | 0 | 89.66 ± 0.00 | 51.13 ± 0.00 | 12.00 | 4.36 | 0.00 | 297 | screen_tome_k29_s0_1030_135953_88a4 |
| screen | tome |  | 49 | 49.0 | 0 | 89.66 ± 0.00 | 51.13 ± 0.00 | 14.50 | 6.86 | 0.00 | 346 | screen_tome_k49_s0_1030_135953_1e32 |

