"""Fixed experiment recipe. Locked: the agent may not change anything in harness/.

Only a human changes these values, then re-locks with `bash agent/relock.sh`.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
STATE = ROOT / "state"          # teacher checkpoint, bench results (not code)
RUNS = ROOT / "runs"            # one folder per job, metrics.json written only by harness
QUEUE = ROOT / "queue"          # pending / running / done job files

IMG_SIZE = 224
RESIZE = 256                             # eval: Resize(256, bicubic) -> CenterCrop(224); train images stored at 256x256
PATCH = 16
NUM_PATCHES = (IMG_SIZE // PATCH) ** 2   # 196 spatial tokens (CLS not counted in K)

# Setting (2026-10-30, third version). CIFAR-100 did not give a usable KD > CE regime: a scratch DeiT-Ti reached
# ~24 % in the screen budget, and a pretrained one gained only +0.76 %p from KD. aim-lab-test-5 had already
# established KD > CE with this recipe (DeiT-B teacher fine-tuned on the dataset, ImageNet-pretrained DeiT-Ti
# student, 100 epochs; CUB: KD - CE = +1.62 %p over 3 seeds), and on Waterbirds MaskedKD at 15 % of the tokens cost
# 18.5 %p of worst-group accuracy against Full KD: token choice matters there. Recipe = aim-lab-test-5
# docs/maskedkd_settings.md section 6, except that augmentation runs on the GPU from stored 256x256 images.
DATASETS = ("waterbirds", "cub", "aircraft", "caltech256", "cub_lt", "celeba")
# 2026-10-31: the search moved to CUB. The aim is FLOPs-efficient KD, not robustness: Waterbirds was picked only because
# KD > CE there, but with 2 classes the teacher's soft labels carry almost no class-relation information (the thing KD
# transfers), so a token-selection result there says little about MaskedKD's multi-class setting. CUB (200 classes)
# has the larger, verified KD gain (aim-lab-test-5 +1.62 %p over 3 seeds; +1.83 here). Waterbirds is the check.
STAGE1 = "cub"               # search
STAGE2 = "aircraft"          # check on a second dataset (fixed methods, fresh seeds)
# 2026-11-02: the check moved from Waterbirds to FGVC-Aircraft. Waterbirds (2 classes, spurious background) checks
# robustness, not whether a token budget chosen on CUB transfers to another fine-grained, many-class task. Its runs stay
# valid and stay in RESULTS_TABLE ("earlier check").
NUM_CLASSES = {"waterbirds": 2, "cub": 200, "aircraft": 100, "caltech256": 256, "cub_lt": 200, "celeba": 2}
NUM_GROUPS = {"waterbirds": 4, "cub": 0, "aircraft": 0, "caltech256": 0, "cub_lt": 2, "celeba": 4}   # CelebA group = 2*blond + male   # Waterbirds group = 2*y + place; evaluation only, never used in training
DATA_URLS = {
    "waterbirds": ["https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz",
                   "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"],
    "cub": ["https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"],
    "aircraft": ["https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz",
                 # the server cannot reach robots.ox.ac.uk (2026-11-02, curl 28); a Hugging Face copy of the same
                 # archive (2,753,340,328 bytes); prepare_data checks the split sizes against EXPECTED_SIZES
                 "https://huggingface.co/datasets/xingslong/fgvc-aircraft-2013b/resolve/main/fgvc-aircraft-2013b.tar.gz"],
    "caltech256": ["https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1",
                   # same archive (1,183,006,720 bytes) on Hugging Face; the official link refused automated download
                   "https://huggingface.co/datasets/xingslong/caltech256/resolve/main/256_ObjectCategories.tar"],
    "cub_lt": ["https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"],   # the CUB archive
    # CelebA (Liu et al. 2015) as 132 parquet shards with the 40 attributes and image_id (official numbering, so the
    # official train/val/test partition is recovered from the id): the official Google Drive link refuses automated
    # downloads. Shard i is this prefix + f"train-{i:05d}-of-00132.parquet".
    "celeba": ["https://huggingface.co/datasets/huggan/CelebA-faces-with-attributes/resolve/main/data/"],
}
CELEBA_SHARDS = 132
# CelebA spurious-correlation setting (Sagawa et al. ICLR 2020): y = Blond_Hair, spurious attribute = Male, group =
# 2*y + male; the worst group is blond men. Official partition by image number: 1..162770 train, ..182637 val, rest
# test. To fit the harness (every train image in RAM per job, 4 jobs) the train split is a random
# CELEBA_TRAIN_FRACTION of the official train (correlation preserved) and val/test cap every group at CELEBA_EVAL_CAP
# images (the minority groups are kept whole, so worst-group accuracy is measured on all of them).
CELEBA_TRAIN_FRACTION, CELEBA_EVAL_CAP = 0.12, 2000
EXPECTED_SIZES = {"waterbirds": {"train": 4795, "val": 1199, "test": 5794},
                  "cub": {"train": 5394, "val": 600, "test": 5794},     # CUB val = 10 % per class of official train
                  "cub_lt": {"val": 600, "test": 5794, "partial": True},   # train = CUB train with the tail cut (size logged)
                  "celeba": {"train": 19532, "partial": True},   # round(0.12 * 162770); val/test sizes logged (group caps)
                  "aircraft": {"train": 5967, "val": 700, "test": 3333},  # Aircraft (100 variants): val = 10 % per
                  # class of official trainval (the CUB rule), official test
                  "caltech256": {"total": 29780}}  # 256 object classes (257.clutter dropped); per class 80/10/10
CUB_VAL_FRACTION, CUB_VAL_SEED = 0.10, 0     # same split rule as aim-lab-test-5 (teachers are interchangeable)
TRIM_BOTTOM = {"aircraft": 20}               # Aircraft images carry a 20-px copyright banner (removed, as is standard)
STAGE0_OUTPUTS = Path(os.environ.get("STAGE0_OUTPUTS", Path.home() / "stage0" / "outputs"))  # aim-lab-test-5 results
STAGE0_DATA = Path(os.environ.get("STAGE0_DATA", Path.home() / "stage0" / "data"))

STUDENT = "deit_tiny_patch16_224.fb_in1k"
STUDENT_PRETRAINED = True                # ImageNet-1k weights; the classifier head is new
TEACHER = "deit_base_patch16_224.fb_in1k"
TEACHER_DIM = 768                        # used by the ToMe cost model to match budgets

# Student recipe (aim-lab-test-5 = MaskedKD defaults except where noted there)
BATCH = 128
STUDENT_LR_DEFAULT = 1e-4    # used only when no lr selection exists (state/recipe.json, from aim-lab-test-5 or a sweep)
STUDENT_LRS = (5e-5, 1e-4, 3e-4)
MIN_LR_RATIO = 0.01          # min lr = lr / 100
WARMUP_LR = 1e-6
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 0.05
DROP_PATH = 0.1
SMOOTHING = 0.1          # label smoothing for the student CE term (via mixup targets)
MIXUP = 0.8
CUTMIX = 1.0
REPROB = 0.25
RAND_AUG_N = 2
RAND_AUG_M = 9
ALPHA = 0.5              # loss = (1-ALPHA)*CE + ALPHA*KD
TAU = 1.0

# Teacher fine-tune (no mixup, label smoothing 0: LS 0.1 flattened the teacher in aim-lab-test-4; best val epoch kept)
TEACHER_BATCH = 64       # DeiT-B full fine-tune fits a 24 GB GPU at this batch
TEACHER_EPOCHS = 100
TEACHER_EPOCHS_BY = {"caltech256": 20, "celeba": 8}   # same number of images seen as 100 epochs of a ~6k-image set
TEACHER_LR = 1e-4
TEACHER_WARMUP_EPOCHS = 5

RECIPE_FILE = Path(__file__).resolve().parent.parent / "state" / "recipe.json"   # chosen lrs, locked with the harness


def student_lr(dataset):
    import json
    if RECIPE_FILE.exists():
        lr = json.loads(RECIPE_FILE.read_text()).get(dataset, {}).get("student_lr")
        if lr:
            return float(lr)
    return STUDENT_LR_DEFAULT


STAGE2_MAX_RUNNING = 4     # stage-2 jobs share the GPU with the search: at most this many at once (was 1 while the
                           # CUB search filled the GPU; 2026-11-02 the CUB search is done and Aircraft is the main work)
STAGE2_PRIORITY = -10      # and they start only when nothing else is waiting
# Agent methods a human let onto stage 2 before an audit PASS (the stage-1 final on all fresh seeds is still required).
# This only admits the runs; a claim still needs an audit PASS. 2026-11-03: the lateroll@29 audit kept failing on the
# evidence bundle, not on the result, and blocked the Aircraft transfer check for a day.
STAGE2_PREAUDIT = {"lateroll": "2026-11-03 human: run unchanged on Aircraft before the audit PASS"}
# 2026-11-03 human: CUB's KD gain is uncertain (Full - CE +0.50, CI includes 0), so the TrustMask search moves to
# Aircraft (KD gain +1.62). While True, any agent method may run on stage 2; its runs are labelled "stage2_explore"
# and are search results, not a transfer check: a claim needs the unchanged method on a held-out dataset later.
STAGE2_EXPLORE = True
# 2026-11-03 human: a held-out general-object dataset for the final TrustMask candidate (not searched on). An agent
# method may run here only as a final-tier run of a selector source that already has final seeds 1-3 on both CUB and
# Aircraft (submit.holdout_rule); no screen runs. Its epochs are set so a run sees about as many images as an
# Aircraft run (~24k train images x 20 epochs vs ~6k x 100) and finishes within the day.
HOLDOUT = ("caltech256", "celeba")   # set up by the loop like stage 2 (data, teacher, gate); no baseline grid
HOLDOUT_STRICT = ("caltech256",)     # only these take agent selectors as frozen candidates (submit.holdout_rule)
BOOT_WHEN_IDLE = ("celeba",)         # the loop sets these up only while no job is queued or running (QUEUE 11-04 09:30)
# 2026-11-04 00:30 KST (human): nothing is retired. Waterbirds is the main analysis bed again (QUEUE 2026-11-04).
RETIRED = ()
# CUB long-tail (QUEUE 2026-11-04, gate 2): classes 0..99 keep every CUB train image, classes 100..199 keep
# LT_TAIL_FRACTION of theirs (same CUB val/test, same CUB teacher). Group 0 = head class, 1 = tail class (evaluation
# only): "wga" of a cub_lt run is the tail-class accuracy.
LT_TAIL_CLASSES, LT_TAIL_FRACTION = range(100, 200), 0.20
TEACHER_SHARED = {"cub_lt": "cub"}   # datasets that distil from another dataset's teacher (same classes)

# Epochs per dataset and tier: 100 everywhere (the aim-lab-test-5 recipe; a Waterbirds run takes minutes).
TIER_EPOCHS = {
    "cub": {"smoke": 1, "screen": 100, "final": 100},
    "cub_lt": {"smoke": 1, "screen": 100, "final": 100},   # gate 2 (QUEUE 2026-11-04)
    "celeba": {"smoke": 1, "screen": 30, "final": 30},     # 19.5k train images x 30 epochs ~ a Waterbirds run
    "waterbirds": {"smoke": 1, "screen": 100, "final": 100},   # earlier check
    "aircraft": {"smoke": 1, "screen": 100, "final": 100},     # stage 2: only harness jobs and audited methods
    "caltech256": {"smoke": 1, "final": 20},                   # held-out check: final tier only (see HOLDOUT)
}
TIER_MAX_STEPS = {"smoke": 30}   # smoke = a few minutes: catches CUDA/shape errors, nothing else
SMOKE_VAL_LIMIT = 500
TIER_TIME_LIMIT_H = {"smoke": 0.5, "screen": 3.0, "final": 3.0, "analysis": 1.0}
STAGE2_TIME_LIMIT_H = 6.0
SCREEN_SEED = 0
FINAL_SEEDS = (1, 2, 3)      # fresh seeds only: the screen seed never enters a final claim
NOISE_SEEDS = (4, 5)         # extra screen seeds of maskedkd@98 and full (noise); never the final seeds 1-3
BASELINE_KEEPS = (49, 29, 12)  # 25 %, 15 %, 6 % of the 196 patch tokens
CLAIM_MIN_SEEDS = 3            # a primary-metric entry or a claim needs the mean of at least this many seeds
PROMOTE_MIN = 0.5            # promotion threshold = max(PROMOTE_MIN, 2 * seed sigma) %p
SNAPSHOT_FRACS = (0.25, 0.5) # extra student snapshots (fraction of epochs) for analysis jobs

# Scheduler
MAX_JOBS = 4                 # hard cap; real concurrency also limited by measured VRAM
VRAM_MARGIN_GIB = 1.5
EVAL_EVERY = 5               # epochs between val evaluations (last epoch always evaluated)
CKPT_EVERY = 1               # epochs between resume checkpoints
FLOP_STEPS = 3               # steps measured with FlopCounterMode (after FLOP_WARMUP steps)
FLOP_WARMUP = 2
FLOP_FRACS = (0.0, 0.25, 0.5, 0.75, 0.95)   # FLOP windows over the run (a per-image budget may change with training)

# Reproduction gate (must pass before the agent loop starts)
GATE_KD_OVER_CE = 1.0        # Full KD val acc >= CE + 1.0 %p
GATE_MASKEDKD98_GAP = 0.5
GATE_KD_WGA_OVER_CE = 2.0    # Waterbirds: KD may instead show its value on worst-group accuracy (>= CE + 2 %p)
WGA_TOL_MIN = 1.0            # worst-group tolerance = max(WGA_TOL_MIN, 2 x seed sigma of WGA); groups are small    # |MaskedKD@98 - Full KD| <= 0.5 %p
# Primary metric: smallest mean teacher-token budget whose val acc >= FullKD - 1.0 %p
# A pretrained student gains little from KD (gate run 2026-10-30: Full 83.50 vs CE 82.74), so a fixed 1 %p would let
# CE itself (zero teacher tokens) almost qualify. The tolerance is therefore a share of the KD gain: a method must keep
# at least half of what the full teacher adds over CE, and never fall more than 1 %p below Full KD.
TOLERANCE = 1.0              # cap, %p
TOLERANCE_FRAC = 0.5         # tolerance = min(TOLERANCE, TOLERANCE_FRAC * (Full KD - CE)), same tier


def tolerance(full_acc, ce_acc):
    if ce_acc is None or full_acc is None:
        return TOLERANCE
    return max(0.0, min(TOLERANCE, TOLERANCE_FRAC * (full_acc - ce_acc)))
KEEP_GRID = (196, 98, 49, 25, 12)
ANALYSIS_ONLY = ("oracle",)   # methods that may run but never count toward claims or the primary metric
BASELINES = ("ce", "full", "maskedkd", "random", "grid", "center", "rollout", "tome", "oracle", "teacher_topk",
             "rand_alloc", "dynkd", "rand_dyn", "sal_ratio", "kd_sched", "ent_weight")   # harness-owned reference methods
# A per-image budget method (QUEUE 2026-11-01) must also beat each of these at the same mean budget
ALLOC_REFS = ("rand_alloc", "dynkd", "rand_dyn", "sal_ratio", "kd_sched", "ent_weight")
# A claim at budget K must beat every one of these at the same K (spatial ones: no interpretability signal at all)
CLAIM_REFS = ("maskedkd", "rollout", "tome", "grid", "center", "random")
FINAL_PRIORITY = 3           # agent final-seed jobs run before agent screen jobs (default priority 0)

# FLOP counter self-check (bench): DeiT-B forward at 197 tokens is 17.6 GMACs (DeiT paper) = 35.2 GFLOPs
TEACHER_REF_GFLOPS = 35.2
FLOP_CHECK_TOL = 0.10


def time_limit_h(dataset, tier):
    return STAGE2_TIME_LIMIT_H if dataset == STAGE2 and tier != "smoke" else TIER_TIME_LIMIT_H[tier]


# Epoch budgets and concurrency decided by agent/bootstrap.sh from the bench (locked with the harness).
EPOCHS_FILE = STATE / "epochs.json"
if EPOCHS_FILE.exists():
    import json as _json
    _d = _json.loads(EPOCHS_FILE.read_text())
    for _ds, _tiers in _d.get("epochs", {}).items():
        if _ds in TIER_EPOCHS:
            TIER_EPOCHS[_ds].update(_tiers)
    MAX_JOBS = _d.get("max_jobs", MAX_JOBS)
