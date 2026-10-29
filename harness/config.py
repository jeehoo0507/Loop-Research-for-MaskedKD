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

NUM_CLASSES = 100
IMG_SIZE = 224
PATCH = 16
NUM_PATCHES = (IMG_SIZE // PATCH) ** 2   # 196 spatial tokens (CLS not counted in K)
VAL_PER_CLASS = 50                       # 45k train / 5k val split of the CIFAR-100 train set
SPLIT_SEED = 0

STUDENT = "deit_tiny_patch16_224"        # from scratch
TEACHER = "deit_small_patch16_224.fb_in1k"

# Student recipe (DeiT defaults; MaskedKD loss)
BATCH = 128
LR = 5e-4 * BATCH / 512
MIN_LR = 1e-5
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

# Teacher fine-tune (label smoothing 0: LS 0.1 flattened the teacher in aim-lab-test-4)
TEACHER_EPOCHS = 10
TEACHER_LR = 1e-4
TEACHER_WARMUP_EPOCHS = 1

# Datasets. Stage 1 = cifar100 (search). Stage 2 = in100 (ImageNet-100, CMC 100-class subset from the
# public Hugging Face dataset clane9/imagenet-100; prepared by `python -m harness.prepare_stage2`).
DATASETS = ("cifar100", "in100")
STAGE2 = "in100"
IN100_HF_REPO = "clane9/imagenet-100"
IN100_TRAIN_RES = 256      # stored train images (RandomResizedCrop to 224 happens on the GPU)
STAGE2_MAX_RUNNING = 1     # stage-2 jobs share the GPU with the search: at most this many at once
STAGE2_PRIORITY = -10      # and they start only when nothing else is waiting

# Epoch budgets per dataset and tier. None until a human sets them from `python -m harness.bench`.
TIER_EPOCHS = {
    "cifar100": {"smoke": 1, "screen": None, "final": None},
    "in100": {"smoke": 1, "final": None},      # stage 2 has no screen tier: fixed methods, fresh seeds
}
TIER_MAX_STEPS = {"smoke": 30}   # smoke = a few minutes: catches CUDA/shape errors, nothing else
SMOKE_VAL_LIMIT = 500
TIER_TIME_LIMIT_H = {"smoke": 0.5, "screen": 1.5, "final": 6.0, "analysis": 1.0}
STAGE2_TIME_LIMIT_H = 24.0
SCREEN_SEED = 0
FINAL_SEEDS = (1, 2, 3)      # fresh seeds only: the screen seed never enters a final claim
NOISE_SEEDS = (1, 2)         # extra screen seeds of maskedkd@98 and full, to measure seed noise
BASELINE_KEEPS = (49, 25, 12)  # maskedkd and random at these budgets, submitted once by a human
PROMOTE_MIN = 0.5            # promotion threshold = max(PROMOTE_MIN, 2 * seed sigma) %p
SNAPSHOT_FRACS = (0.25, 0.5) # extra student snapshots (fraction of epochs) for analysis jobs

# Scheduler
MAX_JOBS = 4                 # hard cap; real concurrency also limited by measured VRAM
VRAM_MARGIN_GIB = 1.5
EVAL_EVERY = 5               # epochs between val evaluations (last epoch always evaluated)
CKPT_EVERY = 1               # epochs between resume checkpoints
FLOP_STEPS = 3               # steps measured with FlopCounterMode (after FLOP_WARMUP steps)
FLOP_WARMUP = 2

# Reproduction gate (must pass before the agent loop starts)
GATE_KD_OVER_CE = 1.0        # Full KD val acc >= CE + 1.0 %p
GATE_MASKEDKD98_GAP = 0.5    # |MaskedKD@98 - Full KD| <= 0.5 %p
# Primary metric: smallest mean teacher-token budget whose val acc >= FullKD - 1.0 %p
TOLERANCE = 1.0
KEEP_GRID = (196, 98, 49, 25, 12)
ANALYSIS_ONLY = ("oracle",)   # methods that may run but never count toward claims or the primary metric
BASELINES = ("ce", "full", "maskedkd", "random", "tome", "oracle")   # harness-owned reference methods

# FLOP counter self-check (bench): DeiT-S forward at 197 tokens is 4.6 GMACs (DeiT paper) = 9.2 GFLOPs
TEACHER_REF_GFLOPS = 9.2
FLOP_CHECK_TOL = 0.10


def time_limit_h(dataset, tier):
    return STAGE2_TIME_LIMIT_H if dataset == STAGE2 and tier != "smoke" else TIER_TIME_LIMIT_H[tier]


# Epoch budgets and concurrency decided by agent/bootstrap.sh from the bench (locked with the harness).
EPOCHS_FILE = STATE / "epochs.json"
if EPOCHS_FILE.exists():
    import json as _json
    _d = _json.loads(EPOCHS_FILE.read_text())
    for _ds, _tiers in _d.get("epochs", {}).items():
        TIER_EPOCHS[_ds].update(_tiers)
    MAX_JOBS = _d.get("max_jobs", MAX_JOBS)
