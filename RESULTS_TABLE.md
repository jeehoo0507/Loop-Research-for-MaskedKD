# RESULTS_TABLE (written by harness/report.py; do not edit by hand)

*img/s: training throughput averaged over epochs; depends on how many jobs shared the GPU.
Worst-group acc (Waterbirds groups y x place, val split): a diagnostic column only, never used in training or
in any decision here. Decisions use mean accuracy and measured train-step GFLOPs.

# Stage 1: cub (search)

Gate: [cub screen, 100 epochs, mean of seeds [0, 4, 5, 6, 7]] Full KD 82.17 vs CE 80.80 (need +1.0): PASS; MaskedKD@98 82.47 vs Full 82.17 (need >= Full - 0.5): PASS

Noise: acc seed sigma 0.96 %p (pooled, dof 8) -> threshold 1.92 %p

## Significant only (final tier, paired 95% CI excludes 0; wins AND losses)
- lateroll@12 vs tome@12 [acc]: -1.06 %p, 95% CI [-1.53, -0.58], seeds 1,2,3, GFLOPs ratio 0.840 -> LOSS
- lateroll@12 vs grid@12 [acc]: +2.72 %p, 95% CI [+0.86, +4.59], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@12 vs center@12 [acc]: +2.61 %p, 95% CI [+1.98, +3.24], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@12 vs random@12 [acc]: +2.11 %p, 95% CI [+0.85, +3.38], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@29 vs maskedkd@29 [acc]: +1.17 %p, 95% CI [+0.75, +1.58], seeds 1,2,3, GFLOPs ratio 1.000 -> WIN
- lateroll@29 vs rollout@29 [acc]: +0.94 %p, 95% CI [+0.71, +1.18], seeds 1,2,3, GFLOPs ratio 1.000 -> WIN
- lateroll@29 vs tome@29 [acc]: +1.61 %p, 95% CI [+1.37, +1.85], seeds 1,2,3, GFLOPs ratio 1.083 -> WIN
- lateroll@29 vs grid@29 [acc]: +6.06 %p, 95% CI [+2.71, +9.40], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@29 vs center@29 [acc]: +3.11 %p, 95% CI [+0.83, +5.39], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@29 vs random@29 [acc]: +6.44 %p, 95% CI [+3.54, +9.35], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@29 vs rollout@29 [acc]: +0.94 %p, 95% CI [+0.71, +1.18], seeds 1,2,3, GFLOPs ratio 1.000 -> WIN
- lateroll@49 vs tome@49 [acc]: +2.50 %p, 95% CI [+0.31, +4.69], seeds 1,2,3, GFLOPs ratio 1.134 -> WIN
- lateroll@49 vs grid@49 [acc]: +4.11 %p, 95% CI [+3.25, +4.97], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@49 vs random@49 [acc]: +3.83 %p, 95% CI [+2.40, +5.27], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- trustfilt@29 vs grid@29 [acc]: +5.56 %p, 95% CI [+3.51, +7.60], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustfilt@29 vs center@29 [acc]: +2.61 %p, 95% CI [+0.26, +4.97], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustfilt@29 vs random@29 [acc]: +5.94 %p, 95% CI [+5.71, +6.18], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v1@29 vs grid@29 [acc]: +5.94 %p, 95% CI [+4.08, +7.81], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v1@29 vs center@29 [acc]: +3.00 %p, 95% CI [+0.69, +5.31], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v1@29 vs random@29 [acc]: +6.33 %p, 95% CI [+5.24, +7.43], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v2@29 vs grid@29 [acc]: +5.83 %p, 95% CI [+2.04, +9.63], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v2@29 vs center@29 [acc]: +2.89 %p, 95% CI [+2.26, +3.52], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v2@29 vs random@29 [acc]: +6.22 %p, 95% CI [+4.14, +8.31], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v2r@29 vs grid@29 [acc]: +5.94 %p, 95% CI [+0.67, +11.22], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v2r@29 vs center@29 [acc]: +3.00 %p, 95% CI [+0.41, +5.59], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v2r@29 vs random@29 [acc]: +6.33 %p, 95% CI [+1.85, +10.81], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v2r@29 vs full@196 [acc]: -0.44 %p, 95% CI [-0.68, -0.21], seeds 1,2,3, GFLOPs ratio 0.312 -> LOSS

## Core comparison (final tier: every agent method vs maskedkd, rollout, tome, grid, center, random, full, parent)
- lateroll@12 vs maskedkd@12 [acc]: +0.50 %p, 95% CI [-0.60, +1.60], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- lateroll@12 vs rollout@12 [acc]: +0.28 %p, 95% CI [-1.68, +2.23], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- lateroll@12 vs tome@12 [acc]: -1.06 %p, 95% CI [-1.53, -0.58], seeds 1,2,3, GFLOPs ratio 0.840 -> LOSS
- lateroll@12 vs grid@12 [acc]: +2.72 %p, 95% CI [+0.86, +4.59], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@12 vs center@12 [acc]: +2.61 %p, 95% CI [+1.98, +3.24], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@12 vs random@12 [acc]: +2.11 %p, 95% CI [+0.85, +3.38], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@12 vs full@196 [acc]: -2.94 %p, 95% CI [-6.79, +0.90], seeds 1,2,3, GFLOPs ratio 0.236 -> no difference
- lateroll@12 vs rollout@12 [acc]: +0.28 %p, 95% CI [-1.68, +2.23], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- lateroll@29 vs maskedkd@29 [acc]: +1.17 %p, 95% CI [+0.75, +1.58], seeds 1,2,3, GFLOPs ratio 1.000 -> WIN
- lateroll@29 vs rollout@29 [acc]: +0.94 %p, 95% CI [+0.71, +1.18], seeds 1,2,3, GFLOPs ratio 1.000 -> WIN
- lateroll@29 vs tome@29 [acc]: +1.61 %p, 95% CI [+1.37, +1.85], seeds 1,2,3, GFLOPs ratio 1.083 -> WIN
- lateroll@29 vs grid@29 [acc]: +6.06 %p, 95% CI [+2.71, +9.40], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@29 vs center@29 [acc]: +3.11 %p, 95% CI [+0.83, +5.39], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@29 vs random@29 [acc]: +6.44 %p, 95% CI [+3.54, +9.35], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@29 vs full@196 [acc]: -0.33 %p, 95% CI [-2.23, +1.56], seeds 1,2,3, GFLOPs ratio 0.304 -> no difference
- lateroll@29 vs rollout@29 [acc]: +0.94 %p, 95% CI [+0.71, +1.18], seeds 1,2,3, GFLOPs ratio 1.000 -> WIN
- lateroll@49 vs maskedkd@49 [acc]: +0.50 %p, 95% CI [-2.73, +3.73], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- lateroll@49 vs rollout@49 [acc]: +0.06 %p, 95% CI [-3.05, +3.16], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- lateroll@49 vs tome@49 [acc]: +2.50 %p, 95% CI [+0.31, +4.69], seeds 1,2,3, GFLOPs ratio 1.134 -> WIN
- lateroll@49 vs grid@49 [acc]: +4.11 %p, 95% CI [+3.25, +4.97], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@49 vs center@49 [acc]: +2.50 %p, 95% CI [-0.02, +5.02], seeds 1,2,3, GFLOPs ratio 0.999 -> no difference
- lateroll@49 vs random@49 [acc]: +3.83 %p, 95% CI [+2.40, +5.27], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@49 vs full@196 [acc]: +0.44 %p, 95% CI [-4.10, +4.99], seeds 1,2,3, GFLOPs ratio 0.385 -> no difference
- lateroll@49 vs rollout@49 [acc]: +0.06 %p, 95% CI [-3.05, +3.16], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- trustfilt@29 vs maskedkd@29 [acc]: +0.67 %p, 95% CI [-2.32, +3.65], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustfilt@29 vs rollout@29 [acc]: +0.44 %p, 95% CI [-2.25, +3.14], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustfilt@29 vs tome@29 [acc]: +1.11 %p, 95% CI [-1.58, +3.81], seeds 1,2,3, GFLOPs ratio 1.112 -> no difference
- trustfilt@29 vs grid@29 [acc]: +5.56 %p, 95% CI [+3.51, +7.60], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustfilt@29 vs center@29 [acc]: +2.61 %p, 95% CI [+0.26, +4.97], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustfilt@29 vs random@29 [acc]: +5.94 %p, 95% CI [+5.71, +6.18], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustfilt@29 vs full@196 [acc]: -0.83 %p, 95% CI [-4.91, +3.24], seeds 1,2,3, GFLOPs ratio 0.312 -> no difference
- trustfilt@29 vs lateroll@29 (parent) [acc]: -0.50 %p, 95% CI [-3.22, +2.22], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustmask_v1@29 vs maskedkd@29 [acc]: +1.06 %p, 95% CI [-1.03, +3.14], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustmask_v1@29 vs rollout@29 [acc]: +0.83 %p, 95% CI [-1.06, +2.73], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustmask_v1@29 vs tome@29 [acc]: +1.50 %p, 95% CI [-0.40, +3.40], seeds 1,2,3, GFLOPs ratio 1.112 -> no difference
- trustmask_v1@29 vs grid@29 [acc]: +5.94 %p, 95% CI [+4.08, +7.81], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v1@29 vs center@29 [acc]: +3.00 %p, 95% CI [+0.69, +5.31], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v1@29 vs random@29 [acc]: +6.33 %p, 95% CI [+5.24, +7.43], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v1@29 vs full@196 [acc]: -0.44 %p, 95% CI [-3.92, +3.03], seeds 1,2,3, GFLOPs ratio 0.312 -> no difference
- trustmask_v1@29 vs trustfilt@29 (parent) [acc]: +0.39 %p, 95% CI [-0.57, +1.35], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- trustmask_v2@29 vs maskedkd@29 [acc]: +0.94 %p, 95% CI [-1.99, +3.88], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustmask_v2@29 vs rollout@29 [acc]: +0.72 %p, 95% CI [-1.63, +3.08], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustmask_v2@29 vs tome@29 [acc]: +1.39 %p, 95% CI [-0.97, +3.74], seeds 1,2,3, GFLOPs ratio 1.112 -> no difference
- trustmask_v2@29 vs grid@29 [acc]: +5.83 %p, 95% CI [+2.04, +9.63], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v2@29 vs center@29 [acc]: +2.89 %p, 95% CI [+2.26, +3.52], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v2@29 vs random@29 [acc]: +6.22 %p, 95% CI [+4.14, +8.31], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v2@29 vs full@196 [acc]: -0.56 %p, 95% CI [-3.49, +2.38], seeds 1,2,3, GFLOPs ratio 0.312 -> no difference
- trustmask_v2@29 vs trustmask_v1@29 (parent) [acc]: -0.11 %p, 95% CI [-2.15, +1.93], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- trustmask_v2r@29 vs maskedkd@29 [acc]: +1.06 %p, 95% CI [-1.03, +3.14], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustmask_v2r@29 vs rollout@29 [acc]: +0.83 %p, 95% CI [-0.97, +2.64], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustmask_v2r@29 vs tome@29 [acc]: +1.50 %p, 95% CI [-0.30, +3.30], seeds 1,2,3, GFLOPs ratio 1.112 -> no difference
- trustmask_v2r@29 vs grid@29 [acc]: +5.94 %p, 95% CI [+0.67, +11.22], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v2r@29 vs center@29 [acc]: +3.00 %p, 95% CI [+0.41, +5.59], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v2r@29 vs random@29 [acc]: +6.33 %p, 95% CI [+1.85, +10.81], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v2r@29 vs full@196 [acc]: -0.44 %p, 95% CI [-0.68, -0.21], seeds 1,2,3, GFLOPs ratio 0.312 -> LOSS
- trustmask_v2r@29 vs trustmask_v1@29 (parent) [acc]: +0.00 %p, 95% CI [-3.61, +3.61], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference

## Parent check (screen seed 0: a child must beat its META parent by the promotion threshold)
- align_dyn@12 vs parent rand_dyn: acc +0.67 %p (threshold 1.92) -> added part UNPROVEN
- kdhist_dyn@12 vs parent rand_dyn: acc +0.50 %p (threshold 1.92) -> added part UNPROVEN
- lateroll@12 vs parent rollout: acc +0.00 %p (threshold 1.92) -> added part UNPROVEN
- lateroll@20 vs parent rollout: acc +2.17 %p (threshold 1.92) -> added part helps
- lateroll@29 vs parent rollout: acc +0.50 %p (threshold 1.92) -> added part UNPROVEN
- lateroll@49 vs parent rollout: acc +1.00 %p (threshold 1.92) -> added part UNPROVEN
- lateroll@98 vs parent rollout: acc +1.00 %p (threshold 1.92) -> added part UNPROVEN
- mix_dyn@29 vs parent rand_dyn: acc -1.67 %p (threshold 1.92) -> added part UNPROVEN
- mix_dyn@98 vs parent rand_dyn: acc +0.67 %p (threshold 1.92) -> added part UNPROVEN
- mk_halfkd@98 vs parent maskedkd: acc +0.33 %p (threshold 1.92) -> added part UNPROVEN
- sinkfree@12 vs parent maskedkd: acc +1.50 %p (threshold 1.92) -> added part UNPROVEN
- sinkfree@29 vs parent maskedkd: acc -0.33 %p (threshold 1.92) -> added part UNPROVEN
- sinkfree@49 vs parent maskedkd: acc +0.33 %p (threshold 1.92) -> added part UNPROVEN
- trustfilt@29 vs parent lateroll: acc +0.50 %p (threshold 1.92) -> added part UNPROVEN

## Primary metric (analysis-only methods excluded: oracle)
### final: Full KD val 81.72 (seeds 1,2,3); KD gain +0.50, tolerance 0.25 %p; smallest mean kept tokens within it
- claim-grade (>= 3 seeds):
  - lateroll@69541b : 49.0 (3 seeds)
  - maskedkd : 49.0 (3 seeds)
  - rollout : 49.0 (3 seeds)
- screening only (fewer seeds):
  - maskedkd : 98.0 (1 seeds)
### screen: Full KD val 82.17 (seeds 0,4,5,6,7); KD gain +1.37, tolerance 0.68 %p; smallest mean kept tokens within it
- claim-grade (>= 3 seeds):
  - lateroll@69541b : 29.0 (3 seeds)
  - maskedkd : 98.0 (5 seeds)
  - mk_halfkd@1b2cf8 : 98.0 (3 seeds)
- screening only (fewer seeds):
  - lateroll@69541b : 20.0 (1 seeds)

## Where token choice matters (screen seed 0, each baseline vs Full KD)
- K=49: maskedkd 80.50; rollout 81.67; random 78.83; grid 77.33; center 80.17; tome 79.00 (Full KD 82.17)
- K=29: maskedkd 79.17; rollout 80.83; random 74.00; grid 74.50; center 79.00; tome 79.67 (Full KD 82.17)
- K=20: maskedkd 78.00; rollout 79.83; random 73.33 (Full KD 82.17)
- K=12: maskedkd 76.00; rollout 78.33; random 75.33; grid 74.33; center 76.00; tome 78.17 (Full KD 82.17)

## Search effort (screen)
Total agent screen configs: 14
- lateroll: 5 configs
- sinkfree: 3 configs
- mix_dyn: 2 configs
- align_dyn: 1 configs
- trustfilt: 1 configs
- mk_halfkd: 1 configs
- kdhist_dyn: 1 configs

## All finished runs
| tier | method | sel_args | keep | mean kept | seeds | val acc (mean ± sd) | worst-group acc | total GFLOPs/img | teacher | selector | img/s* | run ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final | ce |  | 0 | None | 1,2,3 | 81.22 ± 0.54 | - | 7.64 | 0.00 | 0.00 | 732 | final_ce_k0_s3_1103_013646_bb9a final_ce_k0_s2_1103_013642_841b cub_final_ce_k0_s1_1030_153055_1256 |
| final | center |  | 12 | 12.0 | 1,2,3 | 76.17 ± 0.17 | - | 10.09 | 2.45 | 0.00 | 313 | final_center_k12_s2_1102_232847_3d41 final_center_k12_s1_1102_225104_539f final_center_k12_s3_1102_235618_722f |
| final | center |  | 29 | 29.0 | 1,2,3 | 78.28 ± 1.08 | - | 13.00 | 5.36 | 0.00 | 279 | final_center_k29_s3_1101_140937_62e7 final_center_k29_s1_1101_104303_c1f7 final_center_k29_s2_1101_125958_d08b |
| final | center |  | 49 | 49.0 | 1,2,3 | 79.67 ± 0.88 | - | 16.46 | 8.82 | 0.00 | 340 | final_center_k49_s2_1102_043001_d845 final_center_k49_s1_1102_034050_dbd2 final_center_k49_s3_1102_153530_7a3d |
| final | full |  | 196 | 196.0 | 1,2,3 | 81.72 ± 1.64 | - | 42.77 | 35.13 | 0.00 | 227 | final_full_k196_s2_1101_125958_cc04 final_full_k196_s3_1101_140937_e8fa cub_final_full_k196_s1_1030_153055_d87b |
| final | grid |  | 12 | 12.0 | 1,2,3 | 76.06 ± 0.84 | - | 10.09 | 2.45 | 0.00 | 304 | final_grid_k12_s3_1102_235618_ee11 final_grid_k12_s1_1102_225104_b379 final_grid_k12_s2_1102_232847_19e5 |
| final | grid |  | 29 | 29.0 | 1,2,3 | 75.33 ± 0.60 | - | 13.00 | 5.36 | 0.00 | 281 | final_grid_k29_s1_1101_104303_0ce5 final_grid_k29_s2_1101_125958_44d7 final_grid_k29_s3_1101_140937_8b9f |
| final | grid |  | 49 | 49.0 | 1,2,3 | 78.06 ± 0.38 | - | 16.46 | 8.82 | 0.00 | 313 | final_grid_k49_s2_1102_043001_e20e final_grid_k49_s3_1102_153530_d0e5 final_grid_k49_s1_1102_034050_f172 |
| final | lateroll@69541b |  | 12 | 12.0 | 1,2,3 | 78.78 ± 0.10 | - | 10.08 | 2.45 | 0.08 | 310 | final_lateroll_k12_s3_1102_235618_322a final_lateroll_k12_s2_1102_232847_70cb final_lateroll_k12_s1_1102_225104_b47e |
| final | lateroll@69541b |  | 29 | 29.0 | 1,2,3 | 81.39 ± 1.00 | - | 12.99 | 5.36 | 0.08 | 284 | final_lateroll_k29_s2_1101_125958_6470 final_lateroll_k29_s1_1101_104303_ef38 final_lateroll_k29_s3_1101_140937_ec98 |
| final | lateroll@69541b |  | 49 | 49.0 | 1,2,3 | 82.17 ± 0.29 | - | 16.45 | 8.82 | 0.08 | 294 | final_lateroll_k49_s2_1102_043001_b758 final_lateroll_k49_s1_1102_034050_9f5a final_lateroll_k49_s3_1102_153530_c96e |
| final | maskedkd |  | 12 | 12.0 | 1,2,3 | 78.28 ± 0.48 | - | 10.07 | 2.45 | 0.00 | 690 | cub_final_maskedkd_k12_s2_1031_103004_2d8c cub_final_maskedkd_k12_s3_1031_103017_3c57 final_maskedkd_k12_s1_1102_225104_7267 |
| final | maskedkd |  | 29 | 29.0 | 1,2,3 | 80.22 ± 1.08 | - | 12.99 | 5.36 | 0.00 | 278 | final_maskedkd_k29_s3_1101_140937_b00f final_maskedkd_k29_s2_1101_125958_5b02 final_maskedkd_k29_s1_1101_104303_b676 |
| final | maskedkd |  | 49 | 49.0 | 1,2,3 | 81.67 ± 1.17 | - | 16.44 | 8.82 | 0.00 | 258 | final_maskedkd_k49_s3_1102_153530_68ef final_maskedkd_k49_s2_1102_043001_cd83 final_maskedkd_k49_s1_1102_034050_d0b8 |
| final | maskedkd |  | 98 | 98.0 | 1 | 83.50 ± 0.00 | - | 25.04 | 17.41 | 0.00 | 298 | cub_final_maskedkd_k98_s1_1030_153055_cdbc |
| final | rand_alloc |  | 29 | 29.0 | 1,2,3 | 80.56 ± 0.69 | - | 12.99 | 5.37 | 0.00 | 129 | final_rand_alloc_k29_s3_1103_100031_f4ab final_rand_alloc_k29_s2_1103_100027_fe35 final_rand_alloc_k29_s1_1103_100019_837a |
| final | random |  | 12 | 12.0 | 1,2,3 | 76.67 ± 0.60 | - | 10.09 | 2.45 | 0.00 | 734 | cub_final_random_k12_s2_1031_103009_1917 final_random_k12_s3_1102_235618_869b final_random_k12_s1_1102_225104_c1a9 |
| final | random |  | 29 | 29.0 | 1,2,3 | 74.94 ± 0.25 | - | 13.00 | 5.36 | 0.00 | 279 | final_random_k29_s3_1101_140937_24cb final_random_k29_s1_1101_104303_9664 final_random_k29_s2_1101_125958_9495 |
| final | random |  | 49 | 49.0 | 1,2,3 | 78.33 ± 0.87 | - | 16.46 | 8.82 | 0.00 | 434 | final_random_k49_s1_1102_034050_5828 final_random_k49_s3_1102_153530_ccb4 final_random_k49_s2_1102_043001_3091 |
| final | rollout |  | 12 | 12.0 | 1,2,3 | 78.50 ± 0.73 | - | 10.08 | 2.45 | 0.17 | 739 | cub_final_rollout_k12_s3_1031_103013_2dcf cub_final_rollout_k12_s2_1031_102956_fadd final_rollout_k12_s1_1102_225104_31a8 |
| final | rollout |  | 29 | 29.0 | 1,2,3 | 80.44 ± 1.02 | - | 12.99 | 5.36 | 0.17 | 234 | final_rollout_k29_s1_1101_104303_3861 final_rollout_k29_s2_1101_125958_2ad2 final_rollout_k29_s3_1101_140937_8573 |
| final | rollout |  | 49 | 49.0 | 1,2,3 | 82.11 ± 1.13 | - | 16.45 | 8.82 | 0.17 | 230 | final_rollout_k49_s1_1102_034050_c1e4 final_rollout_k49_s2_1102_043001_2aed final_rollout_k49_s3_1102_153530_4395 |
| final | tome (teacher cost x2.69 of budget: not matched) |  | 12 | 12.0 | 1,2,3 | 79.83 ± 0.29 | - | 12.00 | 4.36 | 0.00 | 305 | final_tome_k12_s1_1102_225104_3037 final_tome_k12_s3_1102_235618_333a final_tome_k12_s2_1102_232847_341f |
| final | tome (teacher cost x1.16 of budget: not matched) |  | 29 | 29.0 | 1,2,3 | 79.78 ± 1.02 | - | 12.00 | 4.36 | 0.00 | 292 | final_tome_k29_s1_1101_104303_ff82 final_tome_k29_s3_1101_140937_cdbb final_tome_k29_s2_1101_125958_ab06 |
| final | tome |  | 49 | 49.0 | 1,2,3 | 79.67 ± 0.83 | - | 14.50 | 6.86 | 0.00 | 272 | final_tome_k49_s2_1102_043001_61ab final_tome_k49_s3_1102_153530_f477 final_tome_k49_s1_1102_034050_9b6d |
| final | trustfilt@27da28 |  | 29 | 29.0 | 1,2,3 | 80.89 ± 0.25 | - | 13.34 | 5.36 | 0.43 | 256 | final_trustfilt_k29_s2_1103_095842_42fd final_trustfilt_k29_s3_1103_095847_a7af final_trustfilt_k29_s1_1103_095835_0073 |
| final | trustmask_v1@8186a4 |  | 29 | 29.0 | 1,2,3 | 81.28 ± 0.25 | - | 13.34 | 5.36 | 0.43 | 246 | final_trustmask_v1_k29_s2_1103_095842_a161 final_trustmask_v1_k29_s3_1103_095847_3c94 final_trustmask_v1_k29_s1_1103_095834_2518 |
| final | trustmask_v2@b2c3bb |  | 29 | 29.0 | 1,2,3 | 81.17 ± 0.93 | - | 13.34 | 5.36 | 0.43 | 240 | final_trustmask_v2_k29_s1_1103_095927_e7db final_trustmask_v2_k29_s3_1103_095936_ddb7 final_trustmask_v2_k29_s2_1103_095932_58e7 |
| final | trustmask_v2r@ef2d38 |  | 29 | 29.0 | 1,2,3 | 81.28 ± 1.70 | - | 13.34 | 5.36 | 0.43 | 419 | final_trustmask_v2r_k29_s3_1103_095950_3bf0 final_trustmask_v2r_k29_s2_1103_095946_94d8 final_trustmask_v2r_k29_s1_1103_095941_713b |
| screen | align_dyn@a5a498 |  | 12 | 9.2 | 0,4,5 | 80.50 ± 0.33 | - | 9.63 | 1.65 | 0.52 | 353 | screen_align_dyn_k12_s4_1101_174735_5d50 screen_align_dyn_k12_s0_1101_174726_2df8 screen_align_dyn_k12_s5_1101_174743_56e0 |
| screen | ce |  | 0 | None | 0,4,5,6,7 | 80.80 ± 1.45 | - | 7.64 | 0.00 | 0.00 | 1041 | screen_ce_k0_s4_1031_114911_08b9 screen_ce_k0_s5_1031_114911_5fdb screen_ce_k0_s0_1031_111002_24e1 screen_ce_k0_s7_1101_230304_50d4 screen_ce_k0_s6_1101_230256_cb6e |
| screen | center |  | 12 | 12.0 | 0 | 76.00 ± 0.00 | - | 10.09 | 2.45 | 0.00 | 1264 | screen_center_k12_s0_1031_130151_50b4 |
| screen | center |  | 29 | 29.0 | 0,4,5 | 79.61 ± 0.67 | - | 13.00 | 5.36 | 0.00 | 639 | screen_center_k29_s4_1101_074021_b6e4 screen_center_k29_s5_1101_074025_d5c6 screen_center_k29_s0_1031_130150_cbde |
| screen | center |  | 49 | 49.0 | 0,4,5 | 80.44 ± 0.35 | - | 16.46 | 8.82 | 0.00 | 588 | screen_center_k49_s5_1101_085152_c3b1 screen_center_k49_s0_1031_130150_3629 screen_center_k49_s4_1101_085148_3a5e |
| screen | dynkd |  | 12 | 10.7 | 0,4,5 | 80.78 ± 0.86 | - | 9.56 | 1.92 | 0.00 | 335 | screen_dynkd_k12_s4_1101_163733_a713 screen_dynkd_k12_s5_1101_163746_4a1a screen_dynkd_k12_s0_1101_163718_78af |
| screen | dynkd |  | 29 | 27.6 | 0,4,5 | 80.78 ± 0.51 | - | 12.58 | 4.94 | 0.00 | 333 | screen_dynkd_k29_s4_1101_145529_d102 screen_dynkd_k29_s0_1101_145520_48c3 screen_dynkd_k29_s5_1101_150155_f994 |
| screen | full |  | 196 | 196.0 | 0,4,5,6,7 | 82.17 ± 0.17 | - | 42.77 | 35.13 | 0.00 | 532 | screen_full_k196_s4_1031_114911_a837 screen_full_k196_s5_1031_114911_9c0a screen_full_k196_s7_1101_230313_4eff screen_full_k196_s0_1031_111002_00d5 screen_full_k196_s6_1101_230309_8730 |
| screen | grid |  | 12 | 12.0 | 0 | 74.33 ± 0.00 | - | 10.09 | 2.45 | 0.00 | 1266 | screen_grid_k12_s0_1031_130151_33bd |
| screen | grid |  | 29 | 29.0 | 0 | 74.50 ± 0.00 | - | 13.00 | 5.36 | 0.00 | 1183 | screen_grid_k29_s0_1031_130150_df6c |
| screen | grid |  | 49 | 49.0 | 0 | 77.33 ± 0.00 | - | 16.46 | 8.82 | 0.00 | 1094 | screen_grid_k49_s0_1031_130150_702e |
| screen | kdhist_dyn@cca0d3 |  | 12 | 10.7 | 0,4,5 | 80.22 ± 0.54 | - | 9.56 | 1.92 | 0.00 | 399 | screen_kdhist_dyn_k12_s0_1101_174730_0fb5 screen_kdhist_dyn_k12_s4_1101_174739_13a9 screen_kdhist_dyn_k12_s5_1101_174748_53c6 |
| screen | lateroll@69541b |  | 12 | 12.0 | 0,4,5 | 78.56 ± 0.25 | - | 10.08 | 2.45 | 0.08 | 547 | screen_lateroll_k12_s5_1101_171055_ddbf screen_lateroll_k12_s4_1101_171051_5464 screen_lateroll_k12_s0_1031_130339_f599 |
| screen | lateroll@69541b |  | 20 | 20.0 | 0 | 82.00 ± 0.00 | - | 11.44 | 3.81 | 0.08 | 377 | screen_lateroll_k20_s0_1101_100242_f95e |
| screen | lateroll@69541b |  | 29 | 29.0 | 0,4,5 | 82.00 ± 0.58 | - | 12.99 | 5.36 | 0.08 | 518 | screen_lateroll_k29_s4_1101_055810_8619 screen_lateroll_k29_s5_1101_055814_a9a4 screen_lateroll_k29_s0_1031_130348_fd19 |
| screen | lateroll@69541b |  | 49 | 49.0 | 0,4,5 | 82.11 ± 0.51 | - | 16.45 | 8.82 | 0.08 | 974 | screen_lateroll_k49_s4_1031_184753_4b41 screen_lateroll_k49_s0_1031_130352_adc5 screen_lateroll_k49_s5_1031_184758_5d9f |
| screen | lateroll@69541b |  | 98 | 98.0 | 0,4,5 | 82.50 ± 0.33 | - | 25.04 | 17.41 | 0.08 | 279 | screen_lateroll_k98_s0_1102_020138_8a4c screen_lateroll_k98_s4_1102_020142_9191 screen_lateroll_k98_s5_1102_020146_a022 |
| screen | maskedkd |  | 12 | 12.0 | 0,4,5 | 76.72 ± 0.63 | - | 10.07 | 2.45 | 0.00 | 622 | screen_maskedkd_k12_s5_1101_163741_b2e9 screen_maskedkd_k12_s4_1101_163726_3dc8 screen_maskedkd_k12_s0_1031_130150_a887 |
| screen | maskedkd |  | 20 | 20.0 | 0 | 78.00 ± 0.00 | - | 11.44 | 3.81 | 0.00 | 312 | screen_maskedkd_k20_s0_1101_100251_336a |
| screen | maskedkd |  | 29 | 29.0 | 0,4,5,6,7 | 80.23 ± 0.95 | - | 12.99 | 5.36 | 0.00 | 814 | screen_maskedkd_k29_s4_1031_183523_7354 screen_maskedkd_k29_s5_1031_183528_bfe5 screen_maskedkd_k29_s6_1101_215100_14b0 screen_maskedkd_k29_s7_1101_215118_b650 screen_maskedkd_k29_s0_1031_130150_1393 |
| screen | maskedkd |  | 49 | 49.0 | 0,4,5 | 80.78 ± 0.48 | - | 16.44 | 8.82 | 0.00 | 1035 | screen_maskedkd_k49_s5_1031_172301_3f83 screen_maskedkd_k49_s0_1031_130150_e26e screen_maskedkd_k49_s4_1031_172257_73af |
| screen | maskedkd |  | 98 | 98.0 | 0,4,5,6,7 | 82.47 ± 1.35 | - | 25.04 | 17.41 | 0.00 | 660 | screen_maskedkd_k98_s6_1101_215109_4112 screen_maskedkd_k98_s0_1031_111002_5715 screen_maskedkd_k98_s4_1031_114911_7153 screen_maskedkd_k98_s5_1031_114911_80fa screen_maskedkd_k98_s7_1101_215126_017f |
| screen | mix_dyn@3b23e6 |  | 29 | 29.0 | 0,4,5 | 78.61 ± 0.59 | - | 13.07 | 5.44 | 0.00 | 303 | screen_mix_dyn_k29_s5_1102_002401_a7c4 screen_mix_dyn_k29_s4_1102_002357_d2a8 screen_mix_dyn_k29_s0_1102_002353_9d45 |
| screen | mix_dyn@3b23e6 |  | 98 | 98.0 | 0,4,5 | 81.28 ± 0.75 | - | 25.16 | 17.53 | 0.00 | 341 | screen_mix_dyn_k98_s5_1102_002414_ad9e screen_mix_dyn_k98_s4_1102_002410_a8ef screen_mix_dyn_k98_s0_1102_002405_3354 |
| screen | mk_halfkd@1b2cf8 |  | 98 | 98.0 | 0,4,5 | 81.83 ± 0.73 | - | 25.04 | 17.41 | 0.00 | 335 | screen_mk_halfkd_k98_s5_1101_204003_05ab screen_mk_halfkd_k98_s0_1101_203954_2f5c screen_mk_halfkd_k98_s4_1101_203959_bb22 |
| screen | oracle |  | 12 | 12.0 | 0 | 76.33 ± 0.00 | - | 45.21 | 2.45 | 35.13 | 638 | screen_oracle_k12_s0_1031_130151_6802 |
| screen | rand_dyn |  | 12 | 10.7 | 0,4,5 | 80.89 ± 1.23 | - | 9.56 | 1.92 | 0.00 | 376 | screen_rand_dyn_k12_s4_1101_163722_152f screen_rand_dyn_k12_s5_1101_163737_06e0 screen_rand_dyn_k12_s0_1101_163714_2f8d |
| screen | rand_dyn |  | 29 | 27.6 | 0,4,5,6,7 | 81.23 ± 0.63 | - | 12.58 | 4.94 | 0.00 | 379 | screen_rand_dyn_k29_s7_1101_215113_d43f screen_rand_dyn_k29_s4_1101_145534_d590 screen_rand_dyn_k29_s0_1101_145525_79b0 screen_rand_dyn_k29_s6_1101_215056_3fe9 screen_rand_dyn_k29_s5_1101_151255_4296 |
| screen | rand_dyn |  | 49 | 49.0 | 0,4,5 | 80.89 ± 0.77 | - | 16.42 | 8.78 | 0.00 | 313 | screen_rand_dyn_k49_s5_1101_190526_0e9d screen_rand_dyn_k49_s0_1101_190515_ccab screen_rand_dyn_k49_s4_1101_190522_983e |
| screen | rand_dyn |  | 98 | 98.0 | 0,4,5,6,7 | 81.40 ± 0.45 | - | 25.21 | 17.56 | 0.00 | 333 | screen_rand_dyn_k98_s4_1101_190535_b5c8 screen_rand_dyn_k98_s7_1101_215122_b95c screen_rand_dyn_k98_s0_1101_190530_0832 screen_rand_dyn_k98_s6_1101_215105_a208 screen_rand_dyn_k98_s5_1101_190539_cd98 |
| screen | random |  | 12 | 12.0 | 0 | 75.33 ± 0.00 | - | 10.09 | 2.45 | 0.00 | 1266 | screen_random_k12_s0_1031_130151_7444 |
| screen | random |  | 20 | 20.0 | 0 | 73.33 ± 0.00 | - | 11.46 | 3.81 | 0.00 | 320 | screen_random_k20_s0_1101_100259_6430 |
| screen | random |  | 29 | 29.0 | 0 | 74.00 ± 0.00 | - | 13.00 | 5.36 | 0.00 | 1184 | screen_random_k29_s0_1031_130150_a433 |
| screen | random |  | 49 | 49.0 | 0 | 78.83 ± 0.00 | - | 16.46 | 8.82 | 0.00 | 1094 | screen_random_k49_s0_1031_130150_91ee |
| screen | rollout |  | 12 | 12.0 | 0,4,5 | 78.11 ± 0.38 | - | 10.08 | 2.45 | 0.17 | 490 | screen_rollout_k12_s4_1101_171039_2114 screen_rollout_k12_s0_1031_130151_9b9e screen_rollout_k12_s5_1101_171046_f2c5 |
| screen | rollout |  | 20 | 20.0 | 0 | 79.83 ± 0.00 | - | 11.45 | 3.81 | 0.17 | 353 | screen_rollout_k20_s0_1101_100255_f59b |
| screen | rollout |  | 29 | 29.0 | 0,4,5 | 80.94 ± 0.35 | - | 12.99 | 5.36 | 0.17 | 931 | screen_rollout_k29_s4_1031_183515_ff99 screen_rollout_k29_s0_1031_130150_5726 screen_rollout_k29_s5_1031_183519_47eb |
| screen | rollout |  | 49 | 49.0 | 0,4,5 | 81.28 ± 0.35 | - | 16.45 | 8.82 | 0.17 | 878 | screen_rollout_k49_s4_1031_172246_23ee screen_rollout_k49_s0_1031_130150_674d screen_rollout_k49_s5_1031_172253_99b6 |
| screen | rollout |  | 98 | 98.0 | 0,4,5 | 81.39 ± 0.54 | - | 25.04 | 17.41 | 0.17 | 259 | screen_rollout_k98_s4_1102_020129_44d7 screen_rollout_k98_s5_1102_020133_2625 screen_rollout_k98_s0_1102_020122_4467 |
| screen | sinkfree@60bc72 |  | 12 | 12.0 | 0 | 77.50 ± 0.00 | - | 10.07 | 2.45 | 0.00 | 1240 | screen_sinkfree_k12_s0_1031_180704_f0b3 |
| screen | sinkfree@60bc72 |  | 29 | 29.0 | 0 | 78.83 ± 0.00 | - | 12.99 | 5.36 | 0.00 | 1161 | screen_sinkfree_k29_s0_1031_180708_1cb6 |
| screen | sinkfree@60bc72 |  | 49 | 49.0 | 0 | 80.83 ± 0.00 | - | 16.44 | 8.82 | 0.00 | 1076 | screen_sinkfree_k49_s0_1031_180713_dba8 |
| screen | teacher_topk |  | 29 | 29.0 | 0,4,5 | 80.00 ± 1.01 | - | 13.36 | 5.36 | 0.35 | 436 | screen_teacher_topk_k29_s4_1101_055819_fa5a screen_teacher_topk_k29_s0_1031_202120_8e2b screen_teacher_topk_k29_s5_1101_055826_eae2 |
| screen | tome (teacher cost x2.69 of budget: not matched) |  | 12 | 12.0 | 0 | 78.17 ± 0.00 | - | 12.00 | 4.36 | 0.00 | 1183 | screen_tome_k12_s0_1031_130151_473d |
| screen | tome (teacher cost x1.16 of budget: not matched) |  | 29 | 29.0 | 0,4,5 | 80.39 ± 0.67 | - | 12.00 | 4.36 | 0.00 | 617 | screen_tome_k29_s0_1031_130150_b286 screen_tome_k29_s4_1101_074029_5765 screen_tome_k29_s5_1101_074033_0da0 |
| screen | tome |  | 49 | 49.0 | 0,4,5 | 79.83 ± 0.73 | - | 14.50 | 6.86 | 0.00 | 574 | screen_tome_k49_s0_1031_130150_83af screen_tome_k49_s4_1101_085156_6f20 screen_tome_k49_s5_1101_085201_e36a |
| screen | trustfilt@27da28 |  | 29 | 29.0 | 0,4,5 | 81.33 ± 0.50 | - | 13.34 | 5.36 | 0.43 | 372 | screen_trustfilt_k29_s5_1101_054141_7ad5 screen_trustfilt_k29_s4_1101_054136_636e screen_trustfilt_k29_s0_1101_052028_1577 |

# Stage 2: aircraft (fixed methods from stage 1; 'stage2' in metrics.json says untouched/tuned)

Gate: [aircraft final, 100 epochs, mean of seeds [1, 2, 3]] Full KD 68.05 vs CE 66.43 (need +1.0): PASS; MaskedKD@98 67.43 vs Full 68.05 (need >= Full - 0.5): FAIL

Noise: acc seed sigma: not measured yet (run harness.submit --baselines)

## Significant only (final tier, paired 95% CI excludes 0; wins AND losses)
- trustfilt@12 vs full@196 [acc]: -4.14 %p, 95% CI [-6.36, -1.93], seeds 1,2,3, GFLOPs ratio 0.244 -> LOSS
- trustfilt@29 vs random@29 [acc]: +2.33 %p, 95% CI [+0.86, +3.81], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustfilt@29 vs full@196 [acc]: -2.48 %p, 95% CI [-3.82, -1.13], seeds 1,2,3, GFLOPs ratio 0.312 -> LOSS
- trustmask_v1@12 vs random@12 [acc]: +1.00 %p, 95% CI [+0.65, +1.35], seeds 1,2,3, GFLOPs ratio 1.034 -> WIN
- trustmask_v1@12 vs full@196 [acc]: -3.90 %p, 95% CI [-5.34, -2.47], seeds 1,2,3, GFLOPs ratio 0.244 -> LOSS
- trustmask_v1@29 vs random@29 [acc]: +3.81 %p, 95% CI [+0.44, +7.18], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN

## Core comparison (final tier: every agent method vs maskedkd, rollout, tome, grid, center, random, full, parent)
- lateroll@29 vs maskedkd@29 [acc]: +1.00 %p, 95% CI [-3.97, +5.97], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- lateroll@29 vs rollout@29 [acc]: -0.14 %p, 95% CI [-3.53, +3.24], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- lateroll@29 vs tome@29 [acc]: +0.10 %p, 95% CI [-3.81, +4.00], seeds 1,2,3, GFLOPs ratio 1.083 -> no difference
- lateroll@29 vs grid@29 [acc]: missing (needs >= 2 shared seeds)
- lateroll@29 vs center@29 [acc]: missing (needs >= 2 shared seeds)
- lateroll@29 vs random@29 [acc]: +3.05 %p, 95% CI [-1.33, +7.43], seeds 1,2,3, GFLOPs ratio 0.999 -> no difference
- lateroll@29 vs full@196 [acc]: -1.76 %p, 95% CI [-4.86, +1.34], seeds 1,2,3, GFLOPs ratio 0.304 -> no difference
- lateroll@29 vs rollout@29 [acc]: -0.14 %p, 95% CI [-3.53, +3.24], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- trustfilt@12 vs maskedkd@12 [acc]: +1.29 %p, 95% CI [-1.97, +4.54], seeds 1,2,3, GFLOPs ratio 1.036 -> no difference
- trustfilt@12 vs rollout@12 [acc]: +1.29 %p, 95% CI [-1.75, +4.32], seeds 1,2,3, GFLOPs ratio 1.035 -> no difference
- trustfilt@12 vs tome@12 [acc]: -1.52 %p, 95% CI [-5.96, +2.91], seeds 1,2,3, GFLOPs ratio 0.869 -> no difference
- trustfilt@12 vs grid@12 [acc]: missing (needs >= 2 shared seeds)
- trustfilt@12 vs center@12 [acc]: missing (needs >= 2 shared seeds)
- trustfilt@12 vs random@12 [acc]: +0.76 %p, 95% CI [-0.13, +1.66], seeds 1,2,3, GFLOPs ratio 1.034 -> no difference
- trustfilt@12 vs full@196 [acc]: -4.14 %p, 95% CI [-6.36, -1.93], seeds 1,2,3, GFLOPs ratio 0.244 -> LOSS
- trustfilt@12 vs lateroll@12 (parent) [acc]: missing (needs >= 2 shared seeds)
- trustfilt@29 vs maskedkd@29 [acc]: +0.29 %p, 95% CI [-1.87, +2.44], seeds 1,2,3, GFLOPs ratio 1.028 -> no difference
- trustfilt@29 vs rollout@29 [acc]: -0.86 %p, 95% CI [-1.92, +0.21], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustfilt@29 vs tome@29 [acc]: -0.62 %p, 95% CI [-2.98, +1.74], seeds 1,2,3, GFLOPs ratio 1.112 -> no difference
- trustfilt@29 vs grid@29 [acc]: missing (needs >= 2 shared seeds)
- trustfilt@29 vs center@29 [acc]: missing (needs >= 2 shared seeds)
- trustfilt@29 vs random@29 [acc]: +2.33 %p, 95% CI [+0.86, +3.81], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustfilt@29 vs full@196 [acc]: -2.48 %p, 95% CI [-3.82, -1.13], seeds 1,2,3, GFLOPs ratio 0.312 -> LOSS
- trustfilt@29 vs lateroll@29 (parent) [acc]: -0.71 %p, 95% CI [-3.62, +2.19], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustmask_v1@12 vs maskedkd@12 [acc]: +1.52 %p, 95% CI [-0.97, +4.02], seeds 1,2,3, GFLOPs ratio 1.036 -> no difference
- trustmask_v1@12 vs rollout@12 [acc]: +1.52 %p, 95% CI [-0.84, +3.89], seeds 1,2,3, GFLOPs ratio 1.035 -> no difference
- trustmask_v1@12 vs tome@12 [acc]: -1.29 %p, 95% CI [-4.67, +2.10], seeds 1,2,3, GFLOPs ratio 0.869 -> no difference
- trustmask_v1@12 vs grid@12 [acc]: missing (needs >= 2 shared seeds)
- trustmask_v1@12 vs center@12 [acc]: missing (needs >= 2 shared seeds)
- trustmask_v1@12 vs random@12 [acc]: +1.00 %p, 95% CI [+0.65, +1.35], seeds 1,2,3, GFLOPs ratio 1.034 -> WIN
- trustmask_v1@12 vs full@196 [acc]: -3.90 %p, 95% CI [-5.34, -2.47], seeds 1,2,3, GFLOPs ratio 0.244 -> LOSS
- trustmask_v1@12 vs trustfilt@12 (parent) [acc]: +0.24 %p, 95% CI [-0.85, +1.32], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- trustmask_v1@29 vs maskedkd@29 [acc]: +1.76 %p, 95% CI [-2.50, +6.03], seeds 1,2,3, GFLOPs ratio 1.028 -> no difference
- trustmask_v1@29 vs rollout@29 [acc]: +0.62 %p, 95% CI [-1.17, +2.41], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustmask_v1@29 vs tome@29 [acc]: +0.86 %p, 95% CI [-0.08, +1.80], seeds 1,2,3, GFLOPs ratio 1.112 -> no difference
- trustmask_v1@29 vs grid@29 [acc]: missing (needs >= 2 shared seeds)
- trustmask_v1@29 vs center@29 [acc]: missing (needs >= 2 shared seeds)
- trustmask_v1@29 vs random@29 [acc]: +3.81 %p, 95% CI [+0.44, +7.18], seeds 1,2,3, GFLOPs ratio 1.026 -> WIN
- trustmask_v1@29 vs full@196 [acc]: -1.00 %p, 95% CI [-2.28, +0.28], seeds 1,2,3, GFLOPs ratio 0.312 -> no difference
- trustmask_v1@29 vs trustfilt@29 (parent) [acc]: +1.48 %p, 95% CI [-1.09, +4.04], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- trustmask_v2@29 vs maskedkd@29 [acc]: +1.29 %p, 95% CI [-24.13, +26.70], seeds 1,2, GFLOPs ratio 1.028 -> no difference
- trustmask_v2@29 vs rollout@29 [acc]: +0.14 %p, 95% CI [-12.56, +12.85], seeds 1,2, GFLOPs ratio 1.027 -> no difference
- trustmask_v2@29 vs tome@29 [acc]: +0.57 %p, 95% CI [-6.69, +7.83], seeds 1,2, GFLOPs ratio 1.112 -> no difference
- trustmask_v2@29 vs grid@29 [acc]: missing (needs >= 2 shared seeds)
- trustmask_v2@29 vs center@29 [acc]: missing (needs >= 2 shared seeds)
- trustmask_v2@29 vs random@29 [acc]: +3.36 %p, 95% CI [-17.52, +24.23], seeds 1,2, GFLOPs ratio 1.027 -> no difference
- trustmask_v2@29 vs full@196 [acc]: -1.50 %p, 95% CI [-11.48, +8.48], seeds 1,2, GFLOPs ratio 0.312 -> no difference
- trustmask_v2@29 vs trustmask_v1@29 (parent) [acc]: -0.43 %p, 95% CI [-4.06, +3.20], seeds 1,2, GFLOPs ratio 1.000 -> no difference

## Parent check (screen seed 0: a child must beat its META parent by the promotion threshold)
- no child selectors yet

## Primary metric (analysis-only methods excluded: oracle)
### final: Full KD val 68.05 (seeds 1,2,3); KD gain +1.62, tolerance 0.81 %p; smallest mean kept tokens within it
- claim-grade (>= 3 seeds):
  - rollout : 49.0 (3 seeds)
  - maskedkd : 98.0 (3 seeds)
- screening only (fewer seeds): none

## Where token choice matters (screen seed 0, each baseline vs Full KD)
- not measured yet

## Search effort (screen)
Total agent screen configs: 0

## All finished runs
| tier | method | sel_args | keep | mean kept | seeds | val acc (mean ± sd) | worst-group acc | total GFLOPs/img | teacher | selector | img/s* | run ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final | ce |  | 0 | None | 1,2,3 | 66.43 ± 0.43 | - | 7.64 | 0.00 | 0.00 | 481 | aircraft_final_ce_k0_s2_1102_132227_80b7 aircraft_final_ce_k0_s3_1102_132227_55cb aircraft_final_ce_k0_s1_1102_123044_6d5f |
| final | ent_weight |  | 29 | 29.0 | 1,2,3 | 66.48 ± 1.47 | - | 12.99 | 5.36 | 0.00 | 294 | aircraft_final_ent_weight_k29_s2_1103_083831_e884 aircraft_final_ent_weight_k29_s1_1103_083827_9ace aircraft_final_ent_weight_k29_s3_1103_083836_6aea |
| final | full |  | 196 | 196.0 | 1,2,3 | 68.05 ± 0.54 | - | 42.77 | 35.13 | 0.00 | 252 | aircraft_final_full_k196_s1_1102_123044_ed5d aircraft_final_full_k196_s3_1102_132227_69cc aircraft_final_full_k196_s2_1102_132227_123b |
| final | lateroll@69541b |  | 29 | 29.0 | 1,2,3 | 66.29 ± 1.78 | - | 12.99 | 5.36 | 0.08 | 380 | aircraft_final_lateroll_k29_s1_1103_065756_f0d9 aircraft_final_lateroll_k29_s3_1103_065805_2130 aircraft_final_lateroll_k29_s2_1103_065801_38e1 |
| final | maskedkd |  | 12 | 12.0 | 1,2,3 | 62.62 ± 0.33 | - | 10.07 | 2.45 | 0.00 | 455 | aircraft_final_maskedkd_k12_s1_1102_132227_f01c aircraft_final_maskedkd_k12_s3_1103_024931_458c aircraft_final_maskedkd_k12_s2_1103_024925_555b |
| final | maskedkd |  | 29 | 29.0 | 1,2,3 | 65.29 ± 1.25 | - | 12.99 | 5.36 | 0.00 | 384 | aircraft_final_maskedkd_k29_s2_1102_180321_52e2 aircraft_final_maskedkd_k29_s3_1102_180336_347e aircraft_final_maskedkd_k29_s1_1102_132227_6542 |
| final | maskedkd |  | 49 | 49.0 | 1,2,3 | 66.95 ± 1.76 | - | 16.44 | 8.82 | 0.00 | 398 | aircraft_final_maskedkd_k49_s3_1102_193521_573d aircraft_final_maskedkd_k49_s1_1102_193509_429e aircraft_final_maskedkd_k49_s2_1102_193516_9d71 |
| final | maskedkd |  | 98 | 98.0 | 1,2,3 | 67.43 ± 0.62 | - | 25.04 | 17.41 | 0.00 | 398 | aircraft_final_maskedkd_k98_s2_1102_132227_2f10 aircraft_final_maskedkd_k98_s1_1102_123044_bd9b aircraft_final_maskedkd_k98_s3_1102_132227_c979 |
| final | oracle |  | 12 | 12.0 | 1 | 63.71 ± 0.00 | - | 45.21 | 2.45 | 35.13 | 172 | aircraft_final_oracle_k12_s1_1102_132227_86cd |
| final | random |  | 12 | 12.0 | 1,2,3 | 63.14 ± 1.17 | - | 10.09 | 2.45 | 0.00 | 401 | aircraft_final_random_k12_s3_1103_024939_6b93 aircraft_final_random_k12_s2_1103_024935_e173 aircraft_final_random_k12_s1_1102_132227_c818 |
| final | random |  | 29 | 29.0 | 1,2,3 | 63.24 ± 0.91 | - | 13.00 | 5.36 | 0.00 | 543 | aircraft_final_random_k29_s2_1103_042255_db43 aircraft_final_random_k29_s3_1103_042259_2272 aircraft_final_random_k29_s1_1102_132227_7454 |
| final | random |  | 49 | 49.0 | 1,2,3 | 65.52 ± 1.07 | - | 16.46 | 8.82 | 0.00 | 373 | aircraft_final_random_k49_s3_1103_051949_322a aircraft_final_random_k49_s2_1103_051945_0e0b aircraft_final_random_k49_s1_1103_051941_6b71 |
| final | rollout |  | 12 | 12.0 | 1,2,3 | 62.62 ± 0.46 | - | 10.08 | 2.45 | 0.17 | 358 | aircraft_final_rollout_k12_s3_1103_024948_a6c2 aircraft_final_rollout_k12_s2_1103_024944_52f8 aircraft_final_rollout_k12_s1_1102_132227_d6d1 |
| final | rollout |  | 29 | 29.0 | 1,2,3 | 66.43 ± 0.52 | - | 12.99 | 5.36 | 0.17 | 335 | aircraft_final_rollout_k29_s1_1102_132227_c3d7 aircraft_final_rollout_k29_s3_1102_180341_8606 aircraft_final_rollout_k29_s2_1102_180325_20b6 |
| final | rollout |  | 49 | 49.0 | 1,2,3 | 68.14 ± 0.87 | - | 16.45 | 8.82 | 0.17 | 296 | aircraft_final_rollout_k49_s1_1103_051925_1dd0 aircraft_final_rollout_k49_s3_1103_051937_47c7 aircraft_final_rollout_k49_s2_1103_051932_f37d |
| final | teacher_topk |  | 29 | 29.0 | 1,2,3 | 65.48 ± 0.86 | - | 13.36 | 5.36 | 0.36 | 401 | aircraft_final_teacher_topk_k29_s3_1103_074603_470b aircraft_final_teacher_topk_k29_s2_1103_074559_69a5 aircraft_final_teacher_topk_k29_s1_1103_074554_cd8b |
| final | tome (teacher cost x2.69 of budget: not matched) |  | 12 | 12.0 | 1,2,3 | 65.43 ± 0.52 | - | 12.00 | 4.36 | 0.00 | 395 | aircraft_final_tome_k12_s2_1103_093020_84da aircraft_final_tome_k12_s3_1103_093025_0e5e aircraft_final_tome_k12_s1_1102_132227_991f |
| final | tome (teacher cost x1.16 of budget: not matched) |  | 29 | 29.0 | 1,2,3 | 66.19 ± 0.33 | - | 12.00 | 4.36 | 0.00 | 398 | aircraft_final_tome_k29_s1_1102_132227_ccc7 aircraft_final_tome_k29_s2_1102_180330_3c69 aircraft_final_tome_k29_s3_1102_180345_552f |
| final | trustfilt@27da28 |  | 12 | 12.0 | 1,2,3 | 63.90 ± 1.41 | - | 10.43 | 2.45 | 0.43 | 322 | aircraft_final_trustfilt_k12_s3_1103_093025_5a85 aircraft_final_trustfilt_k12_s1_1103_093013_2336 aircraft_final_trustfilt_k12_s2_1103_093020_1091 |
| final | trustfilt@27da28 |  | 29 | 29.0 | 1,2,3 | 65.57 ± 0.94 | - | 13.35 | 5.36 | 0.43 | 278 | aircraft_final_trustfilt_k29_s2_1103_074535_db76 aircraft_final_trustfilt_k29_s3_1103_074540_2203 aircraft_final_trustfilt_k29_s1_1103_074527_357e |
| final | trustmask_v1@8186a4 |  | 12 | 12.0 | 1,2,3 | 64.14 ± 1.03 | - | 10.43 | 2.45 | 0.43 | 277 | aircraft_final_trustmask_v1_k12_s2_1103_093020_777f aircraft_final_trustmask_v1_k12_s1_1103_093013_330f aircraft_final_trustmask_v1_k12_s3_1103_093025_7076 |
| final | trustmask_v1@8186a4 |  | 29 | 29.0 | 1,2,3 | 67.05 ± 0.70 | - | 13.35 | 5.36 | 0.43 | 252 | aircraft_final_trustmask_v1_k29_s3_1103_083822_5bcd aircraft_final_trustmask_v1_k29_s2_1103_083818_f341 aircraft_final_trustmask_v1_k29_s1_1103_083809_92a4 |
| final | trustmask_v2@b2c3bb |  | 29 | 29.0 | 1,2 | 66.86 ± 1.21 | - | 13.35 | 5.36 | 0.43 | 319 | aircraft_final_trustmask_v2_k29_s1_1103_095851_4cd0 aircraft_final_trustmask_v2_k29_s2_1103_095900_c4a8 |

# Earlier check: waterbirds (no new runs; kept for the record)

Gate: [waterbirds final, 100 epochs, mean of seeds [1, 2, 3]] Full KD 91.16 vs CE 89.80 (need +1.0): PASS; MaskedKD@98 90.46 vs Full 91.16 (need >= Full - 0.5): FAIL

Noise: acc seed sigma 0.44 %p (pooled, dof 7) -> threshold 0.88 %p (diagnostic: WGA seed sigma 2.78 %p (pooled, dof 7) -> threshold 5.56 %p)

## Significant only (final tier, paired 95% CI excludes 0; wins AND losses)
- gradroll@12 vs maskedkd@12 [acc]: +3.78 %p, 95% CI [+2.74, +4.82], seeds 1,2,3, GFLOPs ratio 1.272 -> WIN
- gradroll@12 vs grid@12 [acc]: +4.89 %p, 95% CI [+3.42, +6.36], seeds 1,2,3, GFLOPs ratio 1.270 -> WIN
- gradroll@12 vs random@12 [acc]: +4.14 %p, 95% CI [+1.03, +7.25], seeds 1,2,3, GFLOPs ratio 1.270 -> WIN
- gradroll@12 vs full@196 [acc]: -3.17 %p, 95% CI [-5.48, -0.86], seeds 1,2,3, GFLOPs ratio 0.300 -> LOSS
- lateroll@12 vs maskedkd@12 [acc]: +4.14 %p, 95% CI [+3.16, +5.12], seeds 1,2,3, GFLOPs ratio 1.000 -> WIN
- lateroll@12 vs tome@12 [acc]: -1.61 %p, 95% CI [-1.85, -1.37], seeds 1,2,3, GFLOPs ratio 0.840 -> LOSS
- lateroll@12 vs grid@12 [acc]: +5.25 %p, 95% CI [+2.19, +8.32], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@12 vs random@12 [acc]: +4.50 %p, 95% CI [+2.40, +6.61], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@12 vs full@196 [acc]: -2.81 %p, 95% CI [-3.65, -1.97], seeds 1,2,3, GFLOPs ratio 0.236 -> LOSS
- lateroll@29 vs maskedkd@29 [acc]: +2.97 %p, 95% CI [+1.25, +4.70], seeds 1,2,3, GFLOPs ratio 1.000 -> WIN
- lateroll@29 vs rollout@29 [acc]: +1.14 %p, 95% CI [+0.10, +2.18], seeds 1,2,3, GFLOPs ratio 1.000 -> WIN
- lateroll@29 vs rollout@29 [acc]: +1.14 %p, 95% CI [+0.10, +2.18], seeds 1,2,3, GFLOPs ratio 1.000 -> WIN
- maskedkd_frozen@29 vs full@196 [acc]: -2.03 %p, 95% CI [-2.27, -1.79], seeds 1,2,3, GFLOPs ratio 0.359 -> LOSS
- suffdisc@12 vs maskedkd@12 [acc]: +5.42 %p, 95% CI [+4.18, +6.66], seeds 1,2,3, GFLOPs ratio 1.491 -> WIN
- suffdisc@12 vs grid@12 [acc]: +6.53 %p, 95% CI [+5.51, +7.56], seeds 1,2,3, GFLOPs ratio 1.489 -> WIN
- suffdisc@12 vs random@12 [acc]: +5.78 %p, 95% CI [+3.43, +8.14], seeds 1,2,3, GFLOPs ratio 1.489 -> WIN
- trustfilt@12 vs maskedkd@12 [acc]: +5.84 %p, 95% CI [+4.74, +6.93], seeds 1,2,3, GFLOPs ratio 1.036 -> WIN
- trustfilt@12 vs grid@12 [acc]: +6.95 %p, 95% CI [+4.12, +9.78], seeds 1,2,3, GFLOPs ratio 1.034 -> WIN
- trustfilt@12 vs random@12 [acc]: +6.20 %p, 95% CI [+4.74, +7.65], seeds 1,2,3, GFLOPs ratio 1.034 -> WIN
- trustfilt@12 vs full@196 [acc]: -1.11 %p, 95% CI [-1.35, -0.87], seeds 1,2,3, GFLOPs ratio 0.244 -> LOSS
- trustfilt@12 vs lateroll@12 (parent) [acc]: +1.70 %p, 95% CI [+1.03, +2.36], seeds 1,2,3, GFLOPs ratio 1.035 -> WIN
- trustfilt@29 vs maskedkd@29 [acc]: +3.20 %p, 95% CI [+0.91, +5.48], seeds 1,2,3, GFLOPs ratio 1.028 -> WIN
- trustfilt@29 vs rollout@29 [acc]: +1.36 %p, 95% CI [+0.41, +2.32], seeds 1,2,3, GFLOPs ratio 1.027 -> WIN
- trustmask_v1@12 vs maskedkd@12 [acc]: +3.73 %p, 95% CI [+2.66, +4.79], seeds 1,2,3, GFLOPs ratio 1.036 -> WIN
- trustmask_v1@12 vs tome@12 [acc]: -2.03 %p, 95% CI [-3.36, -0.70], seeds 1,2,3, GFLOPs ratio 0.869 -> LOSS
- trustmask_v1@12 vs grid@12 [acc]: +4.84 %p, 95% CI [+2.03, +7.64], seeds 1,2,3, GFLOPs ratio 1.034 -> WIN
- trustmask_v1@12 vs center@12 [acc]: -0.75 %p, 95% CI [-1.30, -0.20], seeds 1,2,3, GFLOPs ratio 1.034 -> LOSS
- trustmask_v1@12 vs random@12 [acc]: +4.09 %p, 95% CI [+0.75, +7.42], seeds 1,2,3, GFLOPs ratio 1.034 -> WIN
- trustmask_v1@12 vs full@196 [acc]: -3.22 %p, 95% CI [-5.38, -1.07], seeds 1,2,3, GFLOPs ratio 0.244 -> LOSS
- trustmask_v1@12 vs trustfilt@12 (parent) [acc]: -2.11 %p, 95% CI [-4.04, -0.19], seeds 1,2,3, GFLOPs ratio 1.000 -> LOSS
- trustmask_v1@29 vs maskedkd@29 [acc]: +2.86 %p, 95% CI [+2.27, +3.46], seeds 1,2,3, GFLOPs ratio 1.028 -> WIN

## Core comparison (final tier: every agent method vs maskedkd, rollout, tome, grid, center, random, full, parent)
- gradroll@12 vs maskedkd@12 [acc]: +3.78 %p, 95% CI [+2.74, +4.82], seeds 1,2,3, GFLOPs ratio 1.272 -> WIN
- gradroll@12 vs rollout@12 [acc]: -0.11 %p, 95% CI [-3.71, +3.49], seeds 1,2,3, GFLOPs ratio 1.271 -> no difference
- gradroll@12 vs tome@12 [acc]: -1.97 %p, 95% CI [-4.05, +0.10], seeds 1,2,3, GFLOPs ratio 1.068 -> no difference
- gradroll@12 vs grid@12 [acc]: +4.89 %p, 95% CI [+3.42, +6.36], seeds 1,2,3, GFLOPs ratio 1.270 -> WIN
- gradroll@12 vs center@12 [acc]: -0.70 %p, 95% CI [-1.53, +0.14], seeds 1,2,3, GFLOPs ratio 1.270 -> no difference
- gradroll@12 vs random@12 [acc]: +4.14 %p, 95% CI [+1.03, +7.25], seeds 1,2,3, GFLOPs ratio 1.270 -> WIN
- gradroll@12 vs full@196 [acc]: -3.17 %p, 95% CI [-5.48, -0.86], seeds 1,2,3, GFLOPs ratio 0.300 -> LOSS
- gradroll@12 vs rollout@12 [acc]: -0.11 %p, 95% CI [-3.71, +3.49], seeds 1,2,3, GFLOPs ratio 1.271 -> no difference
- lateroll@12 vs maskedkd@12 [acc]: +4.14 %p, 95% CI [+3.16, +5.12], seeds 1,2,3, GFLOPs ratio 1.000 -> WIN
- lateroll@12 vs rollout@12 [acc]: +0.25 %p, 95% CI [-1.85, +2.35], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- lateroll@12 vs tome@12 [acc]: -1.61 %p, 95% CI [-1.85, -1.37], seeds 1,2,3, GFLOPs ratio 0.840 -> LOSS
- lateroll@12 vs grid@12 [acc]: +5.25 %p, 95% CI [+2.19, +8.32], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@12 vs center@12 [acc]: -0.33 %p, 95% CI [-2.03, +1.36], seeds 1,2,3, GFLOPs ratio 0.999 -> no difference
- lateroll@12 vs random@12 [acc]: +4.50 %p, 95% CI [+2.40, +6.61], seeds 1,2,3, GFLOPs ratio 0.999 -> WIN
- lateroll@12 vs full@196 [acc]: -2.81 %p, 95% CI [-3.65, -1.97], seeds 1,2,3, GFLOPs ratio 0.236 -> LOSS
- lateroll@12 vs rollout@12 [acc]: +0.25 %p, 95% CI [-1.85, +2.35], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- lateroll@29 vs maskedkd@29 [acc]: +2.97 %p, 95% CI [+1.25, +4.70], seeds 1,2,3, GFLOPs ratio 1.000 -> WIN
- lateroll@29 vs rollout@29 [acc]: +1.14 %p, 95% CI [+0.10, +2.18], seeds 1,2,3, GFLOPs ratio 1.000 -> WIN
- lateroll@29 vs tome@29 [acc]: +0.50 %p, 95% CI [-1.74, +2.74], seeds 1,2,3, GFLOPs ratio 1.083 -> no difference
- lateroll@29 vs grid@29 [acc]: missing (needs >= 2 shared seeds)
- lateroll@29 vs center@29 [acc]: missing (needs >= 2 shared seeds)
- lateroll@29 vs random@29 [acc]: missing (needs >= 2 shared seeds)
- lateroll@29 vs full@196 [acc]: -0.44 %p, 95% CI [-1.31, +0.42], seeds 1,2,3, GFLOPs ratio 0.304 -> no difference
- lateroll@29 vs rollout@29 [acc]: +1.14 %p, 95% CI [+0.10, +2.18], seeds 1,2,3, GFLOPs ratio 1.000 -> WIN
- maskedkd_frozen@29 vs maskedkd@29 [acc]: +1.39 %p, 95% CI [-0.07, +2.85], seeds 1,2,3, GFLOPs ratio 1.183 -> no difference
- maskedkd_frozen@29 vs rollout@29 [acc]: -0.44 %p, 95% CI [-0.92, +0.03], seeds 1,2,3, GFLOPs ratio 1.183 -> no difference
- maskedkd_frozen@29 vs tome@29 [acc]: -1.08 %p, 95% CI [-3.96, +1.79], seeds 1,2,3, GFLOPs ratio 1.280 -> no difference
- maskedkd_frozen@29 vs grid@29 [acc]: missing (needs >= 2 shared seeds)
- maskedkd_frozen@29 vs center@29 [acc]: missing (needs >= 2 shared seeds)
- maskedkd_frozen@29 vs random@29 [acc]: missing (needs >= 2 shared seeds)
- maskedkd_frozen@29 vs full@196 [acc]: -2.03 %p, 95% CI [-2.27, -1.79], seeds 1,2,3, GFLOPs ratio 0.359 -> LOSS
- maskedkd_frozen@29 vs maskedkd@29 [acc]: +1.39 %p, 95% CI [-0.07, +2.85], seeds 1,2,3, GFLOPs ratio 1.183 -> no difference
- suffdisc@12 vs maskedkd@12 [acc]: +5.42 %p, 95% CI [+4.18, +6.66], seeds 1,2,3, GFLOPs ratio 1.491 -> WIN
- suffdisc@12 vs rollout@12 [acc]: +1.53 %p, 95% CI [-2.52, +5.58], seeds 1,2,3, GFLOPs ratio 1.491 -> no difference
- suffdisc@12 vs tome@12 [acc]: -0.33 %p, 95% CI [-2.57, +1.91], seeds 1,2,3, GFLOPs ratio 1.252 -> no difference
- suffdisc@12 vs grid@12 [acc]: +6.53 %p, 95% CI [+5.51, +7.56], seeds 1,2,3, GFLOPs ratio 1.489 -> WIN
- suffdisc@12 vs center@12 [acc]: +0.95 %p, 95% CI [-0.79, +2.68], seeds 1,2,3, GFLOPs ratio 1.489 -> no difference
- suffdisc@12 vs random@12 [acc]: +5.78 %p, 95% CI [+3.43, +8.14], seeds 1,2,3, GFLOPs ratio 1.489 -> WIN
- suffdisc@12 vs full@196 [acc]: -1.53 %p, 95% CI [-3.49, +0.43], seeds 1,2,3, GFLOPs ratio 0.351 -> no difference
- suffdisc@12 vs center@12 [acc]: +0.95 %p, 95% CI [-0.79, +2.68], seeds 1,2,3, GFLOPs ratio 1.489 -> no difference
- trustfilt@12 vs maskedkd@12 [acc]: +5.84 %p, 95% CI [+4.74, +6.93], seeds 1,2,3, GFLOPs ratio 1.036 -> WIN
- trustfilt@12 vs rollout@12 [acc]: +1.95 %p, 95% CI [-0.76, +4.65], seeds 1,2,3, GFLOPs ratio 1.035 -> no difference
- trustfilt@12 vs tome@12 [acc]: +0.08 %p, 95% CI [-0.82, +0.99], seeds 1,2,3, GFLOPs ratio 0.869 -> no difference
- trustfilt@12 vs grid@12 [acc]: +6.95 %p, 95% CI [+4.12, +9.78], seeds 1,2,3, GFLOPs ratio 1.034 -> WIN
- trustfilt@12 vs center@12 [acc]: +1.36 %p, 95% CI [-0.67, +3.40], seeds 1,2,3, GFLOPs ratio 1.034 -> no difference
- trustfilt@12 vs random@12 [acc]: +6.20 %p, 95% CI [+4.74, +7.65], seeds 1,2,3, GFLOPs ratio 1.034 -> WIN
- trustfilt@12 vs full@196 [acc]: -1.11 %p, 95% CI [-1.35, -0.87], seeds 1,2,3, GFLOPs ratio 0.244 -> LOSS
- trustfilt@12 vs lateroll@12 (parent) [acc]: +1.70 %p, 95% CI [+1.03, +2.36], seeds 1,2,3, GFLOPs ratio 1.035 -> WIN
- trustfilt@29 vs maskedkd@29 [acc]: +3.20 %p, 95% CI [+0.91, +5.48], seeds 1,2,3, GFLOPs ratio 1.028 -> WIN
- trustfilt@29 vs rollout@29 [acc]: +1.36 %p, 95% CI [+0.41, +2.32], seeds 1,2,3, GFLOPs ratio 1.027 -> WIN
- trustfilt@29 vs tome@29 [acc]: +0.72 %p, 95% CI [-2.11, +3.55], seeds 1,2,3, GFLOPs ratio 1.112 -> no difference
- trustfilt@29 vs grid@29 [acc]: missing (needs >= 2 shared seeds)
- trustfilt@29 vs center@29 [acc]: missing (needs >= 2 shared seeds)
- trustfilt@29 vs random@29 [acc]: missing (needs >= 2 shared seeds)
- trustfilt@29 vs full@196 [acc]: -0.22 %p, 95% CI [-1.27, +0.82], seeds 1,2,3, GFLOPs ratio 0.312 -> no difference
- trustfilt@29 vs lateroll@29 (parent) [acc]: +0.22 %p, 95% CI [-0.41, +0.86], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustmask_v1@12 vs maskedkd@12 [acc]: +3.73 %p, 95% CI [+2.66, +4.79], seeds 1,2,3, GFLOPs ratio 1.036 -> WIN
- trustmask_v1@12 vs rollout@12 [acc]: -0.17 %p, 95% CI [-2.50, +2.17], seeds 1,2,3, GFLOPs ratio 1.035 -> no difference
- trustmask_v1@12 vs tome@12 [acc]: -2.03 %p, 95% CI [-3.36, -0.70], seeds 1,2,3, GFLOPs ratio 0.869 -> LOSS
- trustmask_v1@12 vs grid@12 [acc]: +4.84 %p, 95% CI [+2.03, +7.64], seeds 1,2,3, GFLOPs ratio 1.034 -> WIN
- trustmask_v1@12 vs center@12 [acc]: -0.75 %p, 95% CI [-1.30, -0.20], seeds 1,2,3, GFLOPs ratio 1.034 -> LOSS
- trustmask_v1@12 vs random@12 [acc]: +4.09 %p, 95% CI [+0.75, +7.42], seeds 1,2,3, GFLOPs ratio 1.034 -> WIN
- trustmask_v1@12 vs full@196 [acc]: -3.22 %p, 95% CI [-5.38, -1.07], seeds 1,2,3, GFLOPs ratio 0.244 -> LOSS
- trustmask_v1@12 vs trustfilt@12 (parent) [acc]: -2.11 %p, 95% CI [-4.04, -0.19], seeds 1,2,3, GFLOPs ratio 1.000 -> LOSS
- trustmask_v1@29 vs maskedkd@29 [acc]: +2.86 %p, 95% CI [+2.27, +3.46], seeds 1,2,3, GFLOPs ratio 1.028 -> WIN
- trustmask_v1@29 vs rollout@29 [acc]: +1.03 %p, 95% CI [-0.47, +2.53], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustmask_v1@29 vs tome@29 [acc]: +0.39 %p, 95% CI [-2.11, +2.89], seeds 1,2,3, GFLOPs ratio 1.112 -> no difference
- trustmask_v1@29 vs grid@29 [acc]: missing (needs >= 2 shared seeds)
- trustmask_v1@29 vs center@29 [acc]: missing (needs >= 2 shared seeds)
- trustmask_v1@29 vs random@29 [acc]: missing (needs >= 2 shared seeds)
- trustmask_v1@29 vs full@196 [acc]: -0.56 %p, 95% CI [-1.70, +0.59], seeds 1,2,3, GFLOPs ratio 0.312 -> no difference
- trustmask_v1@29 vs trustfilt@29 (parent) [acc]: -0.33 %p, 95% CI [-2.44, +1.77], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference

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
### final: Full KD val 91.16 (seeds 1,2,3); KD gain +1.36, tolerance 0.68 %p; smallest mean kept tokens within it
- claim-grade (>= 3 seeds):
  - lateroll@69541b : 29.0 (3 seeds)
  - trustfilt@27da28 : 29.0 (3 seeds)
  - trustmask_v1@8186a4 : 29.0 (3 seeds)
- screening only (fewer seeds): none
### screen: Full KD val 90.74 (seeds 0,1,2,4,5,6); KD gain +1.42, tolerance 0.71 %p; smallest mean kept tokens within it
- claim-grade (>= 3 seeds):
  - trustfilt@27da28 : 20.0 (3 seeds)
  - lateroll@69541b : 29.0 (3 seeds)
  - suffdisc@0f0600 : 29.0 (7 seeds)
  - trustfilt_attn@8fd44d : 29.0 (3 seeds)
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
Total agent screen configs: 20
- suffdisc: 4 configs
- priorroll: 2 configs
- midattn: 2 configs
- trustfilt: 2 configs
- gradroll: 2 configs
- jitdisc: 1 configs
- trustfilt_pure: 1 configs
- trustfilt_attn: 1 configs
- lateroll: 1 configs
- gradrep: 1 configs
- adaroll: 1 configs
- suffstride: 1 configs
- conceptcov: 1 configs

## All finished runs
| tier | method | sel_args | keep | mean kept | seeds | val acc (mean ± sd) | worst-group acc | total GFLOPs/img | teacher | selector | img/s* | run ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final | ce |  | 0 | None | 1,2,3 | 89.80 ± 0.56 | 68.42 ± 0.75 | 7.64 | 0.00 | 0.00 | 653 | waterbirds_final_ce_k0_s1_1031_130203_4b89 waterbirds_final_ce_k0_s3_1031_201502_f7a9 waterbirds_final_ce_k0_s2_1031_201229_603c |
| final | center |  | 12 | 12.0 | 1,2,3 | 88.69 ± 0.43 | 53.64 ± 1.57 | 10.09 | 2.45 | 0.00 | 304 | final_center_k12_s1_1030_201811_341b final_center_k12_s3_1030_220958_413c final_center_k12_s2_1030_212126_5486 |
| final | full |  | 196 | 196.0 | 1,2,3 | 91.16 ± 0.52 | 67.42 ± 0.43 | 42.77 | 35.13 | 0.00 | 196 | final_full_k196_s1_1030_201811_b764 final_full_k196_s2_1030_212126_a1dc final_full_k196_s3_1030_220958_8d7e |
| final | gradroll@4f3962 |  | 12 | 12.0 | 1,2,3 | 87.99 ± 0.42 | 59.15 ± 6.31 | 12.81 | 2.45 | 2.90 | 195 | final_gradroll_k12_s3_1030_220958_176b final_gradroll_k12_s1_1030_201811_faf0 final_gradroll_k12_s2_1030_212126_da50 |
| final | grid |  | 12 | 12.0 | 1,2,3 | 83.10 ± 0.84 | 35.34 ± 0.00 | 10.09 | 2.45 | 0.00 | 301 | final_grid_k12_s2_1030_212126_9336 final_grid_k12_s3_1030_220958_14eb final_grid_k12_s1_1030_201811_dc71 |
| final | lateroll@69541b |  | 12 | 12.0 | 1,2,3 | 88.35 ± 0.41 | 46.37 ± 3.56 | 10.07 | 2.45 | 0.08 | 445 | waterbirds_final_lateroll_k12_s2_1104_025021_d386 waterbirds_final_lateroll_k12_s1_1104_025017_463f waterbirds_final_lateroll_k12_s3_1104_025026_0de9 |
| final | lateroll@69541b |  | 29 | 29.0 | 1,2,3 | 90.71 ± 0.29 | 63.91 ± 1.99 | 12.99 | 5.36 | 0.08 | 295 | waterbirds_final_lateroll_k29_s1_1104_025003_b396 waterbirds_final_lateroll_k29_s3_1104_025012_629b waterbirds_final_lateroll_k29_s2_1104_025008_c94e |
| final | maskedkd |  | 12 | 12.0 | 1,2,3 | 84.21 ± 0.05 | 22.31 ± 1.15 | 10.07 | 2.45 | 0.00 | 280 | final_maskedkd_k12_s1_1030_201811_e49a final_maskedkd_k12_s2_1030_212126_ad00 final_maskedkd_k12_s3_1030_220958_cc4e |
| final | maskedkd |  | 29 | 29.0 | 1,2,3 | 87.74 ± 0.55 | 38.59 ± 1.74 | 12.99 | 5.36 | 0.00 | 366 | waterbirds_final_maskedkd_k29_s2_1103_170931_92bd waterbirds_final_maskedkd_k29_s1_1031_201502_4be8 waterbirds_final_maskedkd_k29_s3_1103_170935_907a |
| final | maskedkd |  | 98 | 98.0 | 1,2,3 | 90.46 ± 0.39 | 64.16 ± 2.42 | 25.04 | 17.41 | 0.00 | 359 | waterbirds_final_maskedkd_k98_s3_1031_201502_0f31 waterbirds_final_maskedkd_k98_s1_1031_130203_5bef waterbirds_final_maskedkd_k98_s2_1031_201502_95ac |
| final | maskedkd_frozen@0f7bcc |  | 29 | 29.0 | 1,2,3 | 89.13 ± 0.59 | 61.40 ± 2.64 | 15.36 | 5.36 | 2.36 | 253 | waterbirds_final_maskedkd_frozen_k29_s2_1103_171026_2b13 waterbirds_final_maskedkd_frozen_k29_s3_1103_171031_b65e waterbirds_final_maskedkd_frozen_k29_s1_1103_171021_f885 |
| final | oracle |  | 12 | 12.0 | 1 | 88.24 ± 0.00 | 52.63 ± 0.00 | 45.21 | 2.45 | 35.13 | 640 | waterbirds_final_oracle_k12_s1_1031_201502_7b51 |
| final | random |  | 12 | 12.0 | 1,2,3 | 83.85 ± 0.91 | 37.34 ± 1.89 | 10.09 | 2.45 | 0.00 | 320 | final_random_k12_s2_1030_212126_b5bc final_random_k12_s1_1030_201811_19bf final_random_k12_s3_1030_220958_e1db |
| final | random |  | 29 | 29.0 | 1 | 81.57 ± 0.00 | 30.83 ± 0.00 | 13.00 | 5.36 | 0.00 | 1182 | waterbirds_final_random_k29_s1_1031_201502_830c |
| final | rollout |  | 12 | 12.0 | 1,2,3 | 88.10 ± 1.17 | 50.88 ± 1.74 | 10.08 | 2.45 | 0.17 | 237 | final_rollout_k12_s1_1030_201811_6293 final_rollout_k12_s3_1030_220958_d961 final_rollout_k12_s2_1030_212126_33d1 |
| final | rollout |  | 29 | 29.0 | 1,2,3 | 89.57 ± 0.65 | 62.16 ± 2.42 | 12.99 | 5.36 | 0.17 | 459 | waterbirds_final_rollout_k29_s2_1103_170940_2b72 waterbirds_final_rollout_k29_s1_1031_201502_e135 waterbirds_final_rollout_k29_s3_1103_170944_dd2c |
| final | suffdisc@0f0600 |  | 12 | 12.0 | 1,2,3 | 89.63 ± 0.46 | 62.40 ± 2.60 | 15.02 | 2.45 | 4.93 | 192 | final_suffdisc_k12_s3_1031_002453_9ebe final_suffdisc_k12_s1_1031_002441_5d86 final_suffdisc_k12_s2_1031_002448_664e |
| final | teacher_topk |  | 12 | 12.0 | 1,2,3 | 88.80 ± 0.42 | 57.64 ± 3.04 | 10.44 | 2.45 | 0.36 | 301 | waterbirds_final_teacher_topk_k12_s2_1103_171051_3651 waterbirds_final_teacher_topk_k12_s1_1103_171046_0a7d waterbirds_final_teacher_topk_k12_s3_1103_171055_5959 |
| final | teacher_topk |  | 29 | 29.0 | 1,2,3 | 89.71 ± 0.21 | 64.41 ± 0.43 | 13.36 | 5.36 | 0.36 | 294 | waterbirds_final_teacher_topk_k29_s1_1103_170914_f015 waterbirds_final_teacher_topk_k29_s3_1103_170923_d12a waterbirds_final_teacher_topk_k29_s2_1103_170918_4535 |
| final | tome (teacher cost x2.69 of budget: not matched) |  | 12 | 12.0 | 1,2,3 | 89.96 ± 0.46 | 50.88 ± 1.89 | 12.00 | 4.36 | 0.00 | 292 | final_tome_k12_s2_1030_212126_9500 final_tome_k12_s3_1030_220958_918a final_tome_k12_s1_1030_201811_9fb4 |
| final | tome (teacher cost x1.16 of budget: not matched) |  | 29 | 29.0 | 1,2,3 | 90.21 ± 0.61 | 51.38 ± 3.13 | 12.00 | 4.36 | 0.00 | 597 | waterbirds_final_tome_k29_s3_1103_234734_fbbf waterbirds_final_tome_k29_s1_1031_201502_610a waterbirds_final_tome_k29_s2_1103_234725_3539 |
| final | trustfilt@27da28 |  | 12 | 12.0 | 1,2,3 | 90.05 ± 0.43 | 63.41 ± 2.42 | 10.43 | 2.45 | 0.43 | 494 | waterbirds_final_trustfilt_k12_s3_1103_171144_fdb5 waterbirds_final_trustfilt_k12_s2_1103_171139_aea5 waterbirds_final_trustfilt_k12_s1_1103_171135_cb3a |
| final | trustfilt@27da28 |  | 29 | 29.0 | 1,2,3 | 90.94 ± 0.54 | 72.68 ± 0.87 | 13.35 | 5.36 | 0.43 | 254 | waterbirds_final_trustfilt_k29_s2_1103_171126_b33d waterbirds_final_trustfilt_k29_s3_1103_171130_db65 waterbirds_final_trustfilt_k29_s1_1103_171121_24be |
| final | trustmask_v1@8186a4 |  | 12 | 12.0 | 1,2,3 | 87.93 ± 0.47 | 44.86 ± 4.59 | 10.43 | 2.45 | 0.43 | 259 | waterbirds_final_trustmask_v1_k12_s3_1103_171109_331a waterbirds_final_trustmask_v1_k12_s1_1103_171059_cd8d waterbirds_final_trustmask_v1_k12_s2_1103_171104_150e |
| final | trustmask_v1@8186a4 |  | 29 | 29.0 | 1,2,3 | 90.60 ± 0.63 | 62.40 ± 1.99 | 13.35 | 5.36 | 0.43 | 269 | waterbirds_final_trustmask_v1_k29_s1_1103_170855_2e3c waterbirds_final_trustmask_v1_k29_s2_1103_170951_279c waterbirds_final_trustmask_v1_k29_s3_1103_170956_fdca |
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
| screen | lateroll@69541b |  | 29 | 29.0 | 4,5,6 | 90.74 ± 0.52 | 61.66 ± 1.30 | 12.99 | 5.36 | 0.08 | 396 | waterbirds_screen_lateroll_k29_s5_1104_093818_1d16 waterbirds_screen_lateroll_k29_s6_1104_093822_07aa waterbirds_screen_lateroll_k29_s4_1104_093813_f1e8 |
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
| screen | teacher_topk |  | 29 | 29.0 | 4,5,6 | 89.80 ± 0.67 | 63.41 ± 2.30 | 13.36 | 5.36 | 0.36 | 471 | waterbirds_screen_teacher_topk_k29_s5_1104_041303_2db2 waterbirds_screen_teacher_topk_k29_s6_1104_041307_d108 waterbirds_screen_teacher_topk_k29_s4_1104_041258_c453 |
| screen | tome (teacher cost x2.69 of budget: not matched) |  | 12 | 12.0 | 0,4,5,6 | 89.57 ± 0.69 | 49.44 ± 1.67 | 12.00 | 4.36 | 0.00 | 311 | screen_tome_k12_s0_1030_135953_84af screen_tome_k12_s4_1031_051120_5c90 screen_tome_k12_s6_1031_054505_d130 screen_tome_k12_s5_1031_051124_f881 |
| screen | tome (teacher cost x1.16 of budget: not matched) |  | 29 | 29.0 | 0 | 89.66 ± 0.00 | 51.13 ± 0.00 | 12.00 | 4.36 | 0.00 | 297 | screen_tome_k29_s0_1030_135953_88a4 |
| screen | tome |  | 49 | 49.0 | 0 | 89.66 ± 0.00 | 51.13 ± 0.00 | 14.50 | 6.86 | 0.00 | 346 | screen_tome_k49_s0_1030_135953_1e32 |
| screen | trustfilt@27da28 |  | 20 | 20.0 | 4,5,6 | 90.27 ± 0.13 | 67.92 ± 1.15 | 11.80 | 3.81 | 0.43 | 398 | waterbirds_screen_trustfilt_k20_s4_1104_050339_b2dd waterbirds_screen_trustfilt_k20_s5_1104_050343_9a6d waterbirds_screen_trustfilt_k20_s6_1104_050347_3cfa |
| screen | trustfilt@27da28 |  | 29 | 29.0 | 4,5,6 | 90.69 ± 0.50 | 69.42 ± 0.87 | 13.35 | 5.36 | 0.43 | 295 | waterbirds_screen_trustfilt_k29_s6_1104_041254_004f waterbirds_screen_trustfilt_k29_s4_1104_041245_3797 waterbirds_screen_trustfilt_k29_s5_1104_041250_8224 |
| screen | trustfilt_attn@8fd44d |  | 29 | 29.0 | 4,5,6 | 90.19 ± 0.46 | 65.91 ± 0.87 | 13.34 | 5.36 | 0.36 | 337 | waterbirds_screen_trustfilt_attn_k29_s4_1104_050310_43e0 waterbirds_screen_trustfilt_attn_k29_s6_1104_050321_808a waterbirds_screen_trustfilt_attn_k29_s5_1104_050317_860f |
| screen | trustfilt_pure@2e6797 |  | 29 | 11.0 | 4,5,6 | 89.88 ± 0.35 | 64.66 ± 0.75 | 10.22 | 2.24 | 0.43 | 177 | waterbirds_screen_trustfilt_pure_k29_s5_1104_050330_15c1 waterbirds_screen_trustfilt_pure_k29_s6_1104_050334_db07 waterbirds_screen_trustfilt_pure_k29_s4_1104_050326_25d1 |

# Held-out check: caltech256

Gate: [caltech256 final, 20 epochs, mean of seeds [1]] Full KD 84.35 vs CE 83.31 (need +1.0): PASS; MaskedKD@98 83.81 vs Full 84.35 (need >= Full - 0.5): FAIL

Noise: acc seed sigma: not measured yet (run harness.submit --baselines)

## Significant only (final tier, paired 95% CI excludes 0; wins AND losses)
- none yet

## Core comparison (final tier: every agent method vs maskedkd, rollout, tome, grid, center, random, full, parent)
- no agent method has >= 2 final seeds yet

## Parent check (screen seed 0: a child must beat its META parent by the promotion threshold)
- no child selectors yet

## Primary metric (analysis-only methods excluded: oracle)
### final: Full KD val 84.35 (seeds 1); KD gain +1.04, tolerance 0.52 %p; smallest mean kept tokens within it
- claim-grade (>= 3 seeds): none
- screening only (fewer seeds): none

## Where token choice matters (screen seed 0, each baseline vs Full KD)
- not measured yet

## Search effort (screen)
Total agent screen configs: 0

## All finished runs
| tier | method | sel_args | keep | mean kept | seeds | val acc (mean ± sd) | worst-group acc | total GFLOPs/img | teacher | selector | img/s* | run ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final | ce |  | 0 | None | 1 | 83.31 ± 0.00 | - | 7.64 | 0.00 | 0.00 | 512 | caltech256_final_ce_k0_s1_1103_123120_2db8 |
| final | full |  | 196 | 196.0 | 1 | 84.35 ± 0.00 | - | 42.77 | 35.13 | 0.00 | 252 | caltech256_final_full_k196_s1_1103_123120_d6de |
| final | maskedkd |  | 98 | 98.0 | 1 | 83.81 ± 0.00 | - | 25.04 | 17.41 | 0.00 | 305 | caltech256_final_maskedkd_k98_s1_1103_123120_9e43 |

# Earlier check: cub_lt (no new runs; kept for the record)

Gate: gate incomplete: ce seeds [1, 2, 3], full seeds [1, 2, 3], maskedkd seeds []

Noise: acc seed sigma: not measured yet (run harness.submit --baselines) (diagnostic: WGA seed sigma: not measured yet (run harness.submit --baselines))

## Significant only (final tier, paired 95% CI excludes 0; wins AND losses)
- lateroll@29 vs maskedkd@29 [acc]: +2.94 %p, 95% CI [+0.90, +4.99], seeds 1,2,3, GFLOPs ratio 1.000 -> WIN
- lateroll@29 vs tome@29 [acc]: +10.67 %p, 95% CI [+8.08, +13.25], seeds 1,2,3, GFLOPs ratio 1.083 -> WIN
- maskedkd_frozen@29 vs maskedkd@29 [acc]: -1.22 %p, 95% CI [-1.85, -0.59], seeds 1,2,3, GFLOPs ratio 1.183 -> LOSS
- maskedkd_frozen@29 vs rollout@29 [acc]: -3.44 %p, 95% CI [-4.71, -2.18], seeds 1,2,3, GFLOPs ratio 1.183 -> LOSS
- maskedkd_frozen@29 vs tome@29 [acc]: +6.50 %p, 95% CI [+5.40, +7.60], seeds 1,2,3, GFLOPs ratio 1.280 -> WIN
- maskedkd_frozen@29 vs full@196 [acc]: -2.72 %p, 95% CI [-4.59, -0.86], seeds 1,2,3, GFLOPs ratio 0.359 -> LOSS
- maskedkd_frozen@29 vs maskedkd@29 [acc]: -1.22 %p, 95% CI [-1.85, -0.59], seeds 1,2,3, GFLOPs ratio 1.183 -> LOSS
- trustfilt@29 vs tome@29 [acc]: +10.06 %p, 95% CI [+5.87, +14.24], seeds 1,2,3, GFLOPs ratio 1.112 -> WIN
- trustfilt@29 vs grid@29 [acc]: +8.83 %p, 95% CI [+8.83, +8.83], seeds 1,2, GFLOPs ratio 1.026 -> WIN
- trustfilt_attn@29 vs tome@29 [acc]: +8.94 %p, 95% CI [+5.42, +12.47], seeds 1,2,3, GFLOPs ratio 1.112 -> WIN
- trustmask_v1@29 vs maskedkd@29 [acc]: +3.50 %p, 95% CI [+0.60, +6.40], seeds 1,2,3, GFLOPs ratio 1.027 -> WIN
- trustmask_v1@29 vs tome@29 [acc]: +11.22 %p, 95% CI [+7.70, +14.74], seeds 1,2,3, GFLOPs ratio 1.112 -> WIN
- trustmask_v1@29 vs grid@29 [acc]: +10.17 %p, 95% CI [+1.70, +18.64], seeds 1,2, GFLOPs ratio 1.026 -> WIN

## Core comparison (final tier: every agent method vs maskedkd, rollout, tome, grid, center, random, full, parent)
- lateroll@29 vs maskedkd@29 [acc]: +2.94 %p, 95% CI [+0.90, +4.99], seeds 1,2,3, GFLOPs ratio 1.000 -> WIN
- lateroll@29 vs rollout@29 [acc]: +0.72 %p, 95% CI [-2.62, +4.07], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- lateroll@29 vs tome@29 [acc]: +10.67 %p, 95% CI [+8.08, +13.25], seeds 1,2,3, GFLOPs ratio 1.083 -> WIN
- lateroll@29 vs grid@29 [acc]: +9.50 %p, 95% CI [-5.32, +24.32], seeds 1,2, GFLOPs ratio 0.999 -> no difference
- lateroll@29 vs center@29 [acc]: missing (needs >= 2 shared seeds)
- lateroll@29 vs random@29 [acc]: missing (needs >= 2 shared seeds)
- lateroll@29 vs full@196 [acc]: +1.44 %p, 95% CI [-1.61, +4.50], seeds 1,2,3, GFLOPs ratio 0.304 -> no difference
- lateroll@29 vs rollout@29 [acc]: +0.72 %p, 95% CI [-2.62, +4.07], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- maskedkd_frozen@29 vs maskedkd@29 [acc]: -1.22 %p, 95% CI [-1.85, -0.59], seeds 1,2,3, GFLOPs ratio 1.183 -> LOSS
- maskedkd_frozen@29 vs rollout@29 [acc]: -3.44 %p, 95% CI [-4.71, -2.18], seeds 1,2,3, GFLOPs ratio 1.183 -> LOSS
- maskedkd_frozen@29 vs tome@29 [acc]: +6.50 %p, 95% CI [+5.40, +7.60], seeds 1,2,3, GFLOPs ratio 1.280 -> WIN
- maskedkd_frozen@29 vs grid@29 [acc]: +4.83 %p, 95% CI [-7.87, +17.54], seeds 1,2, GFLOPs ratio 1.182 -> no difference
- maskedkd_frozen@29 vs center@29 [acc]: missing (needs >= 2 shared seeds)
- maskedkd_frozen@29 vs random@29 [acc]: missing (needs >= 2 shared seeds)
- maskedkd_frozen@29 vs full@196 [acc]: -2.72 %p, 95% CI [-4.59, -0.86], seeds 1,2,3, GFLOPs ratio 0.359 -> LOSS
- maskedkd_frozen@29 vs maskedkd@29 [acc]: -1.22 %p, 95% CI [-1.85, -0.59], seeds 1,2,3, GFLOPs ratio 1.183 -> LOSS
- trustfilt@29 vs maskedkd@29 [acc]: +2.33 %p, 95% CI [-1.25, +5.92], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustfilt@29 vs rollout@29 [acc]: +0.11 %p, 95% CI [-4.14, +4.36], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustfilt@29 vs tome@29 [acc]: +10.06 %p, 95% CI [+5.87, +14.24], seeds 1,2,3, GFLOPs ratio 1.112 -> WIN
- trustfilt@29 vs grid@29 [acc]: +8.83 %p, 95% CI [+8.83, +8.83], seeds 1,2, GFLOPs ratio 1.026 -> WIN
- trustfilt@29 vs center@29 [acc]: missing (needs >= 2 shared seeds)
- trustfilt@29 vs random@29 [acc]: missing (needs >= 2 shared seeds)
- trustfilt@29 vs full@196 [acc]: +0.83 %p, 95% CI [-1.06, +2.73], seeds 1,2,3, GFLOPs ratio 0.312 -> no difference
- trustfilt@29 vs lateroll@29 (parent) [acc]: -0.61 %p, 95% CI [-3.52, +2.30], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustfilt_attn@29 vs maskedkd@29 [acc]: +1.22 %p, 95% CI [-1.69, +4.13], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustfilt_attn@29 vs rollout@29 [acc]: -1.00 %p, 95% CI [-5.08, +3.08], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustfilt_attn@29 vs tome@29 [acc]: +8.94 %p, 95% CI [+5.42, +12.47], seeds 1,2,3, GFLOPs ratio 1.112 -> WIN
- trustfilt_attn@29 vs grid@29 [acc]: +7.92 %p, 95% CI [-1.61, +17.45], seeds 1,2, GFLOPs ratio 1.026 -> no difference
- trustfilt_attn@29 vs center@29 [acc]: missing (needs >= 2 shared seeds)
- trustfilt_attn@29 vs random@29 [acc]: missing (needs >= 2 shared seeds)
- trustfilt_attn@29 vs full@196 [acc]: -0.28 %p, 95% CI [-3.19, +2.63], seeds 1,2,3, GFLOPs ratio 0.312 -> no difference
- trustfilt_attn@29 vs trustfilt@29 (parent) [acc]: -1.11 %p, 95% CI [-3.15, +0.93], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference
- trustmask_v1@29 vs maskedkd@29 [acc]: +3.50 %p, 95% CI [+0.60, +6.40], seeds 1,2,3, GFLOPs ratio 1.027 -> WIN
- trustmask_v1@29 vs rollout@29 [acc]: +1.28 %p, 95% CI [-2.74, +5.30], seeds 1,2,3, GFLOPs ratio 1.027 -> no difference
- trustmask_v1@29 vs tome@29 [acc]: +11.22 %p, 95% CI [+7.70, +14.74], seeds 1,2,3, GFLOPs ratio 1.112 -> WIN
- trustmask_v1@29 vs grid@29 [acc]: +10.17 %p, 95% CI [+1.70, +18.64], seeds 1,2, GFLOPs ratio 1.026 -> WIN
- trustmask_v1@29 vs center@29 [acc]: missing (needs >= 2 shared seeds)
- trustmask_v1@29 vs random@29 [acc]: missing (needs >= 2 shared seeds)
- trustmask_v1@29 vs full@196 [acc]: +2.00 %p, 95% CI [-0.72, +4.72], seeds 1,2,3, GFLOPs ratio 0.312 -> no difference
- trustmask_v1@29 vs trustfilt@29 (parent) [acc]: +1.17 %p, 95% CI [-0.64, +2.97], seeds 1,2,3, GFLOPs ratio 1.000 -> no difference

## Parent check (screen seed 0: a child must beat its META parent by the promotion threshold)
- no child selectors yet

## Primary metric (analysis-only methods excluded: oracle)
### final: Full KD val 63.83 (seeds 1,2,3); KD gain +4.28, tolerance 1.00 %p; smallest mean kept tokens within it
- claim-grade (>= 3 seeds):
  - lateroll@69541b : 29.0 (3 seeds)
  - rollout : 29.0 (3 seeds)
  - teacher_topk : 29.0 (3 seeds)
  - trustfilt@27da28 : 29.0 (3 seeds)
  - trustfilt_attn@8fd44d : 29.0 (3 seeds)
  - trustmask_v1@8186a4 : 29.0 (3 seeds)
- screening only (fewer seeds): none

## Where token choice matters (screen seed 0, each baseline vs Full KD)
- not measured yet

## Search effort (screen)
Total agent screen configs: 0

## All finished runs
| tier | method | sel_args | keep | mean kept | seeds | val acc (mean ± sd) | worst-group acc | total GFLOPs/img | teacher | selector | img/s* | run ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final | ce |  | 0 | None | 1,2,3 | 59.56 ± 1.46 | 38.22 ± 3.03 | 7.64 | 0.00 | 0.00 | 389 | cub_lt_final_ce_k0_s3_1103_234720_e74b cub_lt_final_ce_k0_s2_1103_234700_71de cub_lt_final_ce_k0_s1_1103_234642_ee32 |
| final | center |  | 29 | 29.0 | 1 | 61.83 ± 0.00 | 43.33 ± 0.00 | 13.00 | 5.36 | 0.00 | 467 | cub_lt_final_center_k29_s1_1104_084157_57ad |
| final | full |  | 196 | 196.0 | 1,2,3 | 63.83 ± 1.50 | 46.33 ± 2.73 | 42.77 | 35.13 | 0.00 | 188 | cub_lt_final_full_k196_s3_1103_234704_3d84 cub_lt_final_full_k196_s2_1103_234647_7b36 cub_lt_final_full_k196_s1_1103_234609_88d5 |
| final | grid |  | 29 | 29.0 | 1,2 | 56.25 ± 2.47 | 35.50 ± 3.54 | 13.00 | 5.36 | 0.00 | 494 | cub_lt_final_grid_k29_s1_1104_074618_a6a8 cub_lt_final_grid_k29_s2_1104_113520_5a1e |
| final | lateroll@69541b |  | 29 | 29.0 | 1,2,3 | 65.28 ± 1.00 | 49.78 ± 1.71 | 12.99 | 5.36 | 0.08 | 318 | cub_lt_final_lateroll_k29_s3_1104_084209_a3f8 cub_lt_final_lateroll_k29_s1_1104_084157_2b15 cub_lt_final_lateroll_k29_s2_1104_084204_9f09 |
| final | maskedkd |  | 29 | 29.0 | 1,2,3 | 62.33 ± 0.50 | 44.00 ± 0.00 | 12.99 | 5.36 | 0.00 | 271 | cub_lt_final_maskedkd_k29_s1_1103_234609_76d1 cub_lt_final_maskedkd_k29_s2_1103_234647_0c6c cub_lt_final_maskedkd_k29_s3_1103_234704_b28b |
| final | maskedkd_frozen@0f7bcc |  | 29 | 29.0 | 1,2,3 | 61.11 ± 0.75 | 41.78 ± 1.39 | 15.36 | 5.36 | 2.36 | 309 | cub_lt_final_maskedkd_frozen_k29_s3_1104_074627_dd2d cub_lt_final_maskedkd_frozen_k29_s2_1104_074622_647c cub_lt_final_maskedkd_frozen_k29_s1_1104_074618_8245 |
| final | random |  | 29 | 29.0 | 1 | 55.83 ± 0.00 | 36.00 ± 0.00 | 13.00 | 5.36 | 0.00 | 476 | cub_lt_final_random_k29_s1_1104_103758_b521 |
| final | rollout |  | 29 | 29.0 | 1,2,3 | 64.56 ± 0.79 | 48.78 ± 1.26 | 12.99 | 5.36 | 0.17 | 346 | cub_lt_final_rollout_k29_s3_1103_234704_0edf cub_lt_final_rollout_k29_s1_1103_234609_8382 cub_lt_final_rollout_k29_s2_1103_234647_9b4c |
| final | teacher_topk |  | 29 | 29.0 | 1,2,3 | 63.39 ± 1.51 | 46.11 ± 2.71 | 13.35 | 5.36 | 0.35 | 289 | cub_lt_final_teacher_topk_k29_s1_1103_234627_9ef0 cub_lt_final_teacher_topk_k29_s2_1103_234655_cfa0 cub_lt_final_teacher_topk_k29_s3_1103_234716_6d91 |
| final | tome (teacher cost x1.16 of budget: not matched) |  | 29 | 29.0 | 1,2,3 | 54.61 ± 0.38 | 29.33 ± 0.88 | 12.00 | 4.36 | 0.00 | 296 | cub_lt_final_tome_k29_s2_1103_234647_6467 cub_lt_final_tome_k29_s3_1103_234704_41ee cub_lt_final_tome_k29_s1_1103_234609_6b58 |
| final | trustfilt@27da28 |  | 29 | 29.0 | 1,2,3 | 64.67 ± 1.89 | 47.33 ± 2.73 | 13.34 | 5.36 | 0.43 | 247 | cub_lt_final_trustfilt_k29_s1_1103_234609_dbe3 cub_lt_final_trustfilt_k29_s3_1103_234704_8732 cub_lt_final_trustfilt_k29_s2_1103_234647_43e9 |
| final | trustfilt_attn@8fd44d |  | 29 | 29.0 | 1,2,3 | 63.56 ± 1.46 | 47.22 ± 2.84 | 13.34 | 5.36 | 0.35 | 352 | cub_lt_final_trustfilt_attn_k29_s3_1104_103809_befb cub_lt_final_trustfilt_attn_k29_s1_1104_103757_81f3 cub_lt_final_trustfilt_attn_k29_s2_1104_103804_0d6e |
| final | trustmask_v1@8186a4 |  | 29 | 29.0 | 1,2,3 | 65.83 ± 1.48 | 50.33 ± 1.85 | 13.34 | 5.36 | 0.43 | 288 | cub_lt_final_trustmask_v1_k29_s2_1103_234647_4e79 cub_lt_final_trustmask_v1_k29_s3_1103_234704_d748 cub_lt_final_trustmask_v1_k29_s1_1103_234609_10df |

