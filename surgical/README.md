# Surgical study: which student blocks should pick the teacher's tokens?

Plan and pre-set readings: `maskedkd_plan_surgical_1112.md` (kept outside this repo, section 0).
Everything here is run by a human on the GPU server. The agent loop stays off.

| file | what |
|---|---|
| `selectors/layerroll.py` | attention rollout over a chosen block set (`sel_args: {"layers": [...], "head": "mean"}`). `[11]` = MaskedKD, `0..11` = rollout, `6..11` = lateroll, bit-identical (tested) |
| `surgical/arms.py` | the arms: CE, Full KD, random, last block (MaskedKD), B1 0-2, B2 3-5, B3 6-8, B4 9-11, all, no last (0-10), no first (3-11) |
| `surgical/submit_grid.py` | queue the arms for one dataset |
| `surgical/table.py` | results from `runs/*/metrics.json` |
| `surgical/setup_server.sh` | a new or old server made ready (env, lock, scheduler); never starts the agent loop |
| `surgical/smoke.py` | server smoke test: what this server has, downloads, speed, time estimate, one end-to-end job per dataset |
| `analysis/surgical_layer_sweep.py` | offline: every block, every head and every block set, masked-teacher fidelity on val |
| `harness/` | new datasets `cifar10`, `cifar10c_gaussian_noise`, `cifar10c_defocus_blur`, `cifar10c_frost` (same images in every split) |

## On the server

New server, or one that never had this repository:

```bash
sudo apt-get update -qq && sudo apt-get install -y -qq git tmux curl
git clone https://github.com/jeehoo0507/Loop-Research-for-MaskedKD.git && cd Loop-Research-for-MaskedKD
git checkout surgical && bash surgical/setup_server.sh
```

A server that already has it: `git fetch && git checkout surgical && bash surgical/setup_server.sh`.

`setup_server.sh` checks the GPU and disk, builds the Python env if it is missing, takes the lock, starts the GPU
scheduler (never the agent loop) and prints what this server has. Pushing results back needs `gh auth login`;
cloning and running do not (the repository is public).

**The teacher checkpoints (`state/*.pt`) are not in git.** On a server set up later every teacher is fine-tuned again
(about 1-2 h each for Waterbirds / CUB), and the KD runs already in `runs/` were made with the old teachers: the
scripts here leave them out of the table and run those arms again. CE runs have no teacher and are reused. The
student lr each dataset already has in `state/recipe.json` is kept.

Smoke test first (whole GPU, nothing else running; queues nothing, writes nothing into `runs/`):

```bash
.venv/bin/python -m surgical.smoke        # waterbirds, cifar10, cifar10c_gaussian_noise, celeba
cat surgical_smoke_report.md              # downloads ok? minutes per run, hours for the grid, jobs that fit in VRAM
```

It prepares (downloads) each dataset and checks the splits, measures images/s per arm alone and with as many jobs
as fit in VRAM, and runs one 30-step job end to end per dataset. CelebA is the slow part (1.4 GB, 200k images).
Exit code 1 and a FAIL line in the report if any part failed.

New dataset (data, teacher, then the KD-gain gate before anything else):

```bash
D=cifar10c_gaussian_noise                 # also: cifar10, celeba
.venv/bin/python -m harness.prepare_data --datasets $D
.venv/bin/python -m harness.finetune_teacher --dataset $D     # needs the whole GPU: no jobs running
git add state/ && git commit -m "surgical: $D teacher" && bash agent/relock.sh    # recipe.json is locked
.venv/bin/python -m surgical.submit_grid --dataset $D --seeds 1 2 3 --only "CE" "Full KD"
.venv/bin/python -m surgical.table --datasets $D              # read the "KD gain" line
```

Offline sweep (needs the CE and Full KD students from the gate), then the grid:

```bash
.venv/bin/python -m harness.submit --analysis surgical_layer_sweep --dataset $D --by human
.venv/bin/python -m surgical.submit_grid --dataset $D --seeds 1          # 11 arms, one seed first
.venv/bin/python -m surgical.submit_grid --dataset $D --seeds 2 3        # only where the gate held
.venv/bin/python -m harness.submit --status
.venv/bin/python -m surgical.table
```

Datasets whose teacher is on this server (`surgical.smoke --env-only` says which): skip to the last block; finished
runs made with that teacher are reused. `cub_lt` distils from the `cub` teacher.
`python -m surgical.table --any-teacher` shows the old runs too (for reading only, never mixed into a claim).

## Rules fixed before any result

- A dataset carries a training claim only if Full KD − CE is above 0 with its interval (the table says so).
- `follow − avoid` and the per-arm loss (clean − corrupted, or accuracy − worst group) are the two readings; nothing
  else is added after seeing results.
- No new selector. The only variable is the block set.
