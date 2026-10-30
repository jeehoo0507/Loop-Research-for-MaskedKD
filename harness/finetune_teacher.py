"""The teacher and the student lr for one dataset, once (bootstrap runs this). Refuses to overwrite.

    python -m harness.finetune_teacher --dataset waterbirds

1. If aim-lab-test-5 left a teacher ($STAGE0_OUTPUTS/<dataset>/teacher/seed0/best.pt, DeiT-B fine-tuned with the
   same split, label smoothing 0, no mixup), it is imported as is and checked on our val split.
   Otherwise DeiT-B ImageNet-1k is fine-tuned here with the same recipe (lr 1e-4, 100 epochs, warmup 5, label
   smoothing 0, no mixup; the best val epoch is kept).
2. The student lr is aim-lab-test-5's choice ($STAGE0_OUTPUTS/<dataset>/lr_selection.json: CE final-epoch val acc
   over 5e-5 / 1e-4 / 3e-4) when it exists, else config.STUDENT_LR_DEFAULT. Written to state/recipe.json, which is
   locked with the harness. CE and every KD method use the same lr.

Writes state/teacher[_<dataset>].pt and .json (sha256, val acc, source).
"""
import argparse
import json
import time

import torch
from timm.loss import SoftTargetCrossEntropy

from . import config as C
from .data import GpuAugment, GpuLoader, load_groups, load_split
from .models import make_teacher
from .train import evaluate, lr_at, sha256_file, teacher_paths


def write_recipe(dataset):
    sel = C.STAGE0_OUTPUTS / dataset / "lr_selection.json"
    d = json.loads(C.RECIPE_FILE.read_text()) if C.RECIPE_FILE.exists() else {}
    if sel.exists():
        s = json.loads(sel.read_text())["student"]
        d[dataset] = {"student_lr": float(s["lr"]), "source": f"aim-lab-test-5 {sel}", "scores": s.get("scores")}
    else:
        d[dataset] = {"student_lr": C.STUDENT_LR_DEFAULT, "source": "config.STUDENT_LR_DEFAULT (no aim-lab-test-5 sweep found)"}
    C.STATE.mkdir(exist_ok=True)
    C.RECIPE_FILE.write_text(json.dumps(d, indent=1) + "\n")
    print("recipe:", json.dumps(d[dataset]))


def import_stage0(dataset, out):
    src = C.STAGE0_OUTPUTS / dataset / "teacher" / "seed0" / "best.pt"
    if not src.exists():
        return None
    ck = torch.load(src, map_location="cpu", weights_only=False)
    t = make_teacher(dataset, pretrained=False)
    t.load_state_dict(ck["model"])
    torch.save(t.state_dict(), out)
    return t, {"source": f"aim-lab-test-5 {src}", "stage0_epoch": ck.get("epoch"), "stage0_val_acc": ck.get("val_acc")}


def finetune(dataset, out, device):
    torch.manual_seed(0)
    xtr, ytr = load_split("train", dataset)
    xva, yva = load_split("val", dataset)
    loader = GpuLoader(xtr, ytr, C.TEACHER_BATCH, device, seed=0)
    aug = GpuAugment(device, C.NUM_CLASSES[dataset], smoothing=0.0, mixup=False)
    model = make_teacher(dataset, pretrained=True).to(device)
    crit = SoftTargetCrossEntropy()
    opt = torch.optim.AdamW(model.parameters(), lr=C.TEACHER_LR, weight_decay=C.WEIGHT_DECAY)
    E, spe, best, hist = C.TEACHER_EPOCHS, len(loader), -1.0, []
    for ep in range(E):
        t0 = time.time()
        for step, (xb, yb) in enumerate(loader):
            for g in opt.param_groups:
                g["lr"] = lr_at(ep + step / spe, E, C.TEACHER_LR, C.TEACHER_WARMUP_EPOCHS)
            x, y = aug(xb, yb)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if (ep + 1) % 5 == 0 or ep + 1 == E:
            acc = evaluate(model, xva, yva, device)
            hist.append({"epoch": ep + 1, "val_acc": acc, "epoch_s": round(time.time() - t0, 1)})
            print(hist[-1], flush=True)
            if acc > best:
                best = acc
                torch.save(model.state_dict(), out)
    model.load_state_dict(torch.load(out, map_location="cpu"))
    return model, {"source": "fine-tuned here", "epochs": E, "lr": C.TEACHER_LR, "history": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=C.STAGE1, choices=list(C.DATASETS))
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out, meta_path = teacher_paths(a.dataset)
    if out.exists():
        raise SystemExit(f"{out} exists; delete it by hand if you really want a new teacher")
    C.STATE.mkdir(exist_ok=True)
    write_recipe(a.dataset)
    got = import_stage0(a.dataset, out)
    model, info = (got if got else finetune(a.dataset, out, device))
    model = model.to(device)
    xva, yva = load_split("val", a.dataset)
    ev = evaluate(model, xva, yva, device, load_groups("val", a.dataset))
    acc = ev["acc"] if isinstance(ev, dict) else ev
    meta = {"model": C.TEACHER, "dataset": a.dataset, "label_smoothing": 0.0, "val_acc": round(acc, 2),
            **({"val_wga": ev["wga"], "val_group_acc": ev["group_acc"]} if isinstance(ev, dict) else {}),
            **info, "sha256": sha256_file(out)}
    meta_path.write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("val_acc", "source", "sha256")}))


if __name__ == "__main__":
    main()
