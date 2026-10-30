"""Fine-tune the teacher once (bootstrap runs this). DeiT-B ImageNet-1k -> CIFAR-100 224px, label smoothing 0.

    python -m harness.finetune_teacher

Writes state/teacher.pt and state/teacher.json (sha256, val acc). Refuses to overwrite an existing teacher.
"""
import json
import math
import time

import torch
from timm.loss import SoftTargetCrossEntropy

from . import config as C
from .data import GpuAugment, GpuLoader, load_split
from .models import make_teacher
from .train import evaluate, sha256_file


def main():
    C.STATE.mkdir(exist_ok=True)
    out = C.STATE / "teacher.pt"
    if out.exists():
        raise SystemExit(f"{out} exists; delete it by hand if you really want a new teacher")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    xtr, ytr = load_split("train")
    xva, yva = load_split("val")
    loader = GpuLoader(xtr, ytr, C.TEACHER_BATCH, device, seed=0)
    aug = GpuAugment(device, smoothing=0.0)
    model = make_teacher(pretrained=True).to(device)
    crit = SoftTargetCrossEntropy()
    opt = torch.optim.AdamW(model.parameters(), lr=C.TEACHER_LR, weight_decay=C.WEIGHT_DECAY)
    E, spe = C.TEACHER_EPOCHS, len(loader)
    hist = []
    for ep in range(E):
        t0 = time.time()
        for step, (xb, yb) in enumerate(loader):
            e = ep + step / spe
            lr = C.TEACHER_LR * (e / C.TEACHER_WARMUP_EPOCHS if e < C.TEACHER_WARMUP_EPOCHS else
                                 0.5 * (1 + math.cos(math.pi * (e - C.TEACHER_WARMUP_EPOCHS) / (E - C.TEACHER_WARMUP_EPOCHS))))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = aug(xb, yb)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        acc = evaluate(model, xva, yva, device)
        hist.append({"epoch": ep + 1, "val_acc": acc, "epoch_s": round(time.time() - t0, 1)})
        print(hist[-1], flush=True)
    torch.save(model.state_dict(), out)
    meta = {"model": C.TEACHER, "epochs": E, "lr": C.TEACHER_LR, "label_smoothing": 0.0,
            "val_acc": hist[-1]["val_acc"], "history": hist, "sha256": sha256_file(out)}
    (C.STATE / "teacher.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("val_acc", "sha256")}))


if __name__ == "__main__":
    main()
