# Corrected distillation: the same teacher compute, spent on tokens or on images

Human study, 2026-11-13. The agent loop stays off. Written before any run of it exists: the readings below are fixed.

## The claim under test

MaskedKD trains the student on the full image against the teacher's answer on the kept tokens,
`KL(T(x_S) || s(x))`. That loss is smallest at the masked teacher's answer, not at the teacher's: a student equal to
the teacher is still pushed away. Its step is a **biased** estimate of the full-KD step, and a better token selector
only makes the bias smaller, never zero.

The teacher compute that masking saves can be spent another way: a cheap answer `h` for every image and the full
teacher for a random `m` of the 128 images of a step, with the difference added back with weight `128 / m`
(`harness/twophase.py`). That step is an **unbiased** estimate of the full-KD step whatever `h` is; `h` only sets
the noise. Free choices of `h`: the student's own prediction, the label. Paid choice: the teacher on a few tokens.
So the question is no longer "which tokens" but **"tokens or images: where should the teacher's compute go?"**

## Arms (`corrected/arms.py`)

Budget 29 and 49 = the teacher GFLOPs of MaskedKD at K=29 and K=49 (measured, a test checks it).

| arm | teacher sees | KD target | unbiased |
|---|---|---|---|
| MaskedKD, rollout | K tokens of every image | masked answer | no |
| images, unweighted (rand_dyn) | all tokens of 20 / 33 images | full answer, KD weight 1 | no (KD term 6x too small) |
| images | same | full answer, weight 128 / m (`cv student`) | yes |
| images + label | same | label + correction | yes |
| images + fit | same | best mix of student and label + correction | yes |
| tokens + images, masked | 12 / 29 tokens of every image + all tokens of 11 / 13 | masked answer + correction | yes |
| tokens + images, recal | same | recalibrated masked answer + correction | yes |
| tokens + images, fit (and with the rollout selector) | same | best mix of student, label, masked + correction | yes |
| channel | 25 / 45 tokens of every image + all tokens of 2 | masked answer, student read through a fitted map | no |

References: CE, Full KD. The surgical study's runs of CE, Full KD, MaskedKD and rollout at K=29 are reused.

## Readings, fixed now

Primary metric per dataset: worst-group accuracy where the dataset has groups (Waterbirds, CelebA; tail accuracy on
cub_lt), else val accuracy. Differences are paired over seeds 1-3 with a 95 % t-interval. A dataset whose KD gain
(Full KD − CE) interval includes 0 is an observation, not a claim.

- **R1, bias.** `images + fit − MaskedKD` and `tokens + images, fit − MaskedKD` at the same budget. Supported if
  the interval is above 0 on at least two datasets at budget 29. Dropped if neither is above MaskedKD on any dataset
  with a KD gain.
- **R2, tokens or images.** The sign of `tokens + images, fit − images + fit`, against what the dissection
  predicted before the grid was queued (`corrected/PREDICTIONS.md`, student = Full KD a quarter into training).
  An interval that includes 0 counts as "no difference". Reported as hits over dataset x budget cells.
- **R3, the weight.** `images − images, unweighted` > 0: the earlier "a third of the gain" result of rand_dyn was a
  missing weight, not a limit of subsampling.
- **R4, floor.** Every unbiased arm is not below CE (interval not entirely below 0); MaskedKD is, where it was.

Exploratory, not claims: channel vs MaskedKD; the rollout selector inside tokens + images; how the mix weights and
"masked explains beyond free" move during training (the table prints them from each run's own paired images).

What would make this uninteresting: `images` alone matching every other unbiased arm and Full KD. Then the method
is "full KD on a share of the batch", the masked teacher adds nothing at these budgets, and the result is a statement
about MaskedKD, not a new method. That outcome is reported as it is.

Known limits: the dissection reads clean val images, training uses Mixup/CutMix views; its prediction can miss for
that reason, which is why each run also measures the same quantities on its own training images. Unbiased
corrections were unstable in related work (weights up to 128 / m = 11.6 here); `rho` in the config shrinks the
correction if runs diverge, at the price of bias. One teacher size only; an unmasked small teacher at the same
GFLOPs is a baseline this study does not have yet.

## On the server

```bash
git pull
CUDA_VISIBLE_DEVICES= bash agent/relock.sh        # harness/ and tests/ changed; tests run on the CPU
.venv/bin/python -m corrected.autorun --dry-run  # what would be queued
tmux new -d -s corrected '.venv/bin/python -m corrected.autorun 2>&1 | tee -a corrected/autorun.log'
tail -f corrected/STATUS.md
```

Default datasets: waterbirds, cifar10c_gaussian_noise, celeba, cifar10. Per dataset the order is base runs ->
dissection (no training, minutes) -> prediction written -> grid. Budget 29 runs on seeds 1 2 3 (seed 1 first),
budget 49 on seed 1: about 40 new runs per dataset (22 arm x budget cells; runs that exist are reused). A dataset without data or teacher here is set
up after the others are done (`--no-setup` leaves it out).

```bash
.venv/bin/python -m corrected.table                          # any time: what is finished so far
cat corrected/PREDICTIONS.md
.venv/bin/python -m corrected.autorun --datasets cub_lt      # later: 200 classes, the largest KD gain (needs CUB + its teacher)
.venv/bin/python -m corrected.autorun --seeds-more 1 2 3     # later: budget 49 on all seeds
```

| file | what |
|---|---|
| `harness/twophase.py` | the loss (method `twophase`, config in `sel_args`), its online fits and diagnostics |
| `analysis/masked_teacher_dissection.py` | no training: size and make-up of MaskedKD's error, and which design should win |
| `corrected/arms.py` | the arms and the budget splits |
| `corrected/autorun.py` | base -> dissection -> prediction -> grid, unattended |
| `corrected/table.py` | results, the fixed readings, prediction vs outcome, in-training diagnostics |
