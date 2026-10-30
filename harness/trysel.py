"""Quick check of a selector before queueing it: static validation + one training step on random data.

    python -m harness.trysel --method my_rollout --keep 25 [--sel-args '{"layers": 4}']

Random-init models, batch 8, no dataset access. Prints kept tokens and FLOPs per image by segment.
Use it to catch bugs; it says nothing about accuracy.
"""
import argparse
import json
import time

import torch
from timm.loss import SoftTargetCrossEntropy
from torch.utils.flop_counter import FlopCounterMode

from . import config as C
from .data import GpuAugment
from .models import make_student, make_teacher
from .train import Segments, train_step
from .validate import load_selector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--keep", type=float, default=49)
    ap.add_argument("--sel-args", default="{}")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sel, meta, sha = load_selector(a.method, kwargs=json.loads(a.sel_args))
    s = make_student(C.STAGE1).to(device).train()
    t = make_teacher(C.STAGE1, pretrained=False).to(device).eval()
    opt = torch.optim.AdamW(s.parameters(), lr=1e-4)
    aug = GpuAugment(device, C.NUM_CLASSES[C.STAGE1])
    x, y = aug(torch.randint(0, 256, (8, 3, C.RESIZE, C.RESIZE), dtype=torch.uint8),
               torch.arange(8) % C.NUM_CLASSES[C.STAGE1])
    segs = Segments(True)
    t0 = time.time()
    with FlopCounterMode(display=False) as fc:
        loss, kept = train_step(s, t, sel, a.method, a.keep, x, y, opt, SoftTargetCrossEntropy(), 0, 0.0, segs)
    n = x.shape[0]
    print(json.dumps({"ok": True, "meta": meta, "sha": sha[:12], "loss": round(loss, 4), "mean_kept": kept,
                      "gflops_per_img": {k: round(v / n / 1e9, 3) for k, v in segs.flops.items()} |
                      {"total": round(fc.get_total_flops() / n / 1e9, 3)},
                      "step_s": round(time.time() - t0, 2), "device": str(device)}, indent=1))


if __name__ == "__main__":
    main()
