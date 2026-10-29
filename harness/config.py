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

# Epoch budgets per tier. None until a human sets them from `python -m harness.bench`.
TIER_EPOCHS = {
    "smoke": 1,
    "screen": None,
    "final": None,
}
TIER_TIME_LIMIT_H = {"smoke": 0.5, "screen": 1.5, "final": 6.0}
FINAL_SEEDS = (0, 1, 2)

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
