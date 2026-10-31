"""Train one student job. Only harness code writes runs/<id>/metrics.json.

    python -m harness.train --job queue/running/<id>.json

The job file comes from harness/submit.py. Resumes from runs/<id>/ckpt.pt if present.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy
from torch.utils.flop_counter import FlopCounterMode

from . import config as C
from .data import GpuAugment, as_tensor, load_groups, load_split, make_loader, normalize_resize
from .lock import harness_digest
from .models import make_student, make_teacher, student_forward, teacher_forward_keepmask
from .selector_api import Context
from .baselines import oracle_keep
from .tome import r_for_budget, teacher_forward_tome
from .validate import load_selector


class HarnessViolation(Exception):
    pass


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def teacher_paths(dataset=C.STAGE1):
    tag = "" if dataset == "waterbirds" else f"_{dataset}"   # file names fixed when Waterbirds was stage 1
    return C.STATE / f"teacher{tag}.pt", C.STATE / f"teacher{tag}.json"


def load_teacher(device, dataset=C.STAGE1, allow_random=False):
    """allow_random: smoke runs before the teacher exists use a random-init teacher (same compute)."""
    ckpt, meta_path = teacher_paths(dataset)
    if os.environ.get("MASKEDKD_FAKE_DATA") or (allow_random and not meta_path.exists()):
        t = make_teacher(dataset, pretrained=False)
        return t.to(device).eval(), "fake"
    meta = json.loads(meta_path.read_text())
    sha = sha256_file(ckpt)
    if sha != meta["sha256"]:
        raise HarnessViolation(f"teacher checkpoint sha256 {sha} != recorded {meta['sha256']}")
    t = make_teacher(dataset, pretrained=False)
    t.load_state_dict(torch.load(ckpt, map_location="cpu"))
    for p in t.parameters():
        p.requires_grad_(False)
    return t.to(device).eval(), sha


def lr_at(epoch_float, epochs, lr, warmup=C.WARMUP_EPOCHS):
    """Linear warmup from WARMUP_LR, then cosine to lr * MIN_LR_RATIO (aim-lab-test-5 recipe)."""
    if epoch_float < warmup:
        return C.WARMUP_LR + (lr - C.WARMUP_LR) * epoch_float / warmup
    t = (epoch_float - warmup) / max(1e-8, epochs - warmup)
    lo = lr * C.MIN_LR_RATIO
    return lo + 0.5 * (lr - lo) * (1 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def predict(model, x, device, batch=250):
    was = model.training
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = normalize_resize(as_tensor(x[i:i + batch]).to(device))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(model(xb).argmax(1).cpu())
    model.train(was)
    return torch.cat(out)


def evaluate(model, x, y, device, groups=None):
    """Top-1 accuracy (%), or with `groups` a dict: acc, per-group acc, worst-group acc (wga)."""
    hit = (predict(model, x, device) == y.cpu()).float()
    acc = 100.0 * hit.mean().item()
    if groups is None:
        return acc
    per = {int(g): round(100.0 * hit[groups == g].mean().item(), 2) for g in groups.unique().tolist()}
    return {"acc": acc, "group_acc": per, "wga": min(per.values())}


class Segments:
    """Optional per-segment FLOP counting (used only for the measured steps)."""

    def __init__(self, active):
        self.active = active
        self.flops = {}

    @contextlib.contextmanager
    def seg(self, name):
        if not self.active:
            yield
            return
        with FlopCounterMode(display=False) as fc:
            yield
        self.flops[name] = self.flops.get(name, 0) + fc.get_total_flops()


def train_step(student, teacher, selector, method, budget, x, y, opt, criterion, epoch, progress, segs):
    """One optimizer step. Returns (loss, mean kept tokens or None)."""
    device = x.device
    needs = getattr(selector, "NEEDS", {}) if selector is not None else {}
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = student_forward(student, x, needs.get("attn", []), needs.get("tokens", []))
        base = criterion(out.logits.float(), y)
        if method == "ce":
            loss, kept = base, None
        else:
            ctx = Context(images=x, student=student, out=out, budget=float(budget),
                          num_patches=C.NUM_PATCHES, epoch=epoch, progress=progress)
            with segs.seg("selector"):
                if method == "oracle":
                    keep = oracle_keep(teacher, x, budget)
                elif method == "tome":
                    keep = torch.ones(x.shape[0], C.NUM_PATCHES, dtype=torch.bool, device=x.device)
                else:
                    keep = selector.select(ctx)
            if any(p.grad is not None for p in student.parameters()):
                raise HarnessViolation("selector left parameter gradients (called .backward())")
            keep = to_keep_mask(keep, x.shape[0])
            kept = float(budget) if method == "tome" else keep.sum(1).float().mean().item()
            if kept > budget + 1e-6:
                raise HarnessViolation(f"selector kept {kept:.2f} tokens/img > budget {budget}")
            with torch.no_grad(), segs.seg("teacher"):
                if method == "tome":   # teacher sees every token and merges inside; keep = budget it is matched to
                    t_logits = teacher_forward_tome(teacher, x, r_for_budget(budget))
                else:
                    t_logits = teacher_forward_keepmask(teacher, x, keep)
            kd = F.kl_div(F.log_softmax(out.logits.float() / C.TAU, dim=1),
                          F.softmax(t_logits.float() / C.TAU, dim=1), reduction="batchmean") * C.TAU * C.TAU
            loss = (1 - C.ALPHA) * base + C.ALPHA * kd
    loss.backward()
    opt.step()
    return loss.item(), kept


def to_keep_mask(keep, B):
    if keep.dtype == torch.bool:
        if keep.shape != (B, C.NUM_PATCHES):
            raise HarnessViolation(f"bool mask shape {tuple(keep.shape)} != {(B, C.NUM_PATCHES)}")
        return keep.detach()
    if keep.dtype in (torch.int64, torch.int32) and keep.dim() == 2 and keep.shape[0] == B:
        if keep.numel() and (keep.min() < 0 or keep.max() >= C.NUM_PATCHES):
            raise HarnessViolation("keep index out of range")
        m = torch.zeros(B, C.NUM_PATCHES, dtype=torch.bool, device=keep.device)
        return m.scatter_(1, keep.long(), True)
    raise HarnessViolation(f"selector must return bool [B,196] or long [B,K], got {keep.dtype} {tuple(keep.shape)}")


def run(job_path):
    job = json.loads(open(job_path).read())
    rid = job["id"]
    rdir = C.RUNS / rid
    rdir.mkdir(parents=True, exist_ok=True)
    log = open(rdir / "log.txt", "a")

    def say(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, file=log, flush=True)
        print(line, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(job["seed"])
    epochs = job["epochs"]
    method = job["method"]          # "ce" or a selector name (builtin or selectors/<name>.py)
    budget = job["keep"]

    selector, meta, sel_sha = (None, None, None)
    if method not in ("ce", "oracle", "tome"):
        src = job.get("selector_src")
        selector, meta, sel_sha = load_selector(method, src=src, kwargs=job.get("sel_args", {}))
        if sel_sha != job.get("selector_sha", "builtin"):
            raise HarnessViolation("selector source changed after submit")

    dataset = job.get("dataset", C.STAGE1)
    xtr, ytr = load_split("train", dataset)
    xva, yva = load_split("val", dataset)
    gva = load_groups("val", dataset)
    if job["tier"] == "smoke":
        xva, yva = xva[:C.SMOKE_VAL_LIMIT], yva[:C.SMOKE_VAL_LIMIT]
        gva = None if gva is None else gva[:C.SMOKE_VAL_LIMIT]
    loader = make_loader(xtr, ytr, C.BATCH, device, job["seed"], dataset)
    aug = GpuAugment(device, C.NUM_CLASSES[dataset])
    student = make_student(dataset).to(device)
    lr = float(job.get("lr") or C.student_lr(dataset))   # job lr: only the bootstrap lr sweep sets it
    teacher, teacher_sha = (None, None) if method == "ce" else load_teacher(device, dataset, job["tier"] == "smoke")
    criterion = SoftTargetCrossEntropy()
    decay, no_decay = [], []
    for n, p in student.named_parameters():
        (no_decay if p.ndim <= 1 or n in ("pos_embed", "cls_token") else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": C.WEIGHT_DECAY},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr)

    state = {"epoch": 0, "history": [], "flops": None, "kept": [], "train_time_s": 0.0}
    ckpt_path = rdir / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        student.load_state_dict(ck["student"])
        opt.load_state_dict(ck["opt"])
        loader.g.set_state(ck["loader_rng"])
        torch.set_rng_state(ck["cpu_rng"])
        if device.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"])
        state = ck["state"]
        say(f"resumed at epoch {state['epoch']}")

    say(f"job {rid} method={method} keep={budget} seed={job['seed']} epochs={epochs} device={device}")
    steps_per_epoch = len(loader)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_limit = C.time_limit_h(dataset, job["tier"]) * 3600
    max_steps = C.TIER_MAX_STEPS.get(job["tier"])
    for epoch in range(state["epoch"], epochs):
        t0 = time.time()
        kept_sum, kept_n, loss_sum, n_steps = 0.0, 0, 0.0, 0
        for step, (xb, yb) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            g = epoch * steps_per_epoch + step
            for pg in opt.param_groups:
                pg["lr"] = lr_at(g / steps_per_epoch, epochs, lr)
            x, y = aug(xb, yb)
            measure = state["flops"] is None and C.FLOP_WARMUP <= g < C.FLOP_WARMUP + C.FLOP_STEPS
            segs = Segments(measure)
            if measure:
                with FlopCounterMode(display=False) as fc:
                    loss, kept = train_step(student, teacher, selector, method, budget, x, y, opt, criterion,
                                            epoch, g / (epochs * steps_per_epoch), segs)
                segs.flops["total"] = segs.flops.get("total", 0) + fc.get_total_flops()
                segs.flops["images"] = x.shape[0]
                acc = state.setdefault("_flop_acc", [])
                acc.append(segs.flops)
                if len(acc) == C.FLOP_STEPS:
                    n = sum(a["images"] for a in acc)
                    tot = {k: sum(a.get(k, 0) for a in acc) / n / 1e9 for k in ("total", "selector", "teacher")}
                    tot["student"] = tot["total"] - tot["selector"] - tot["teacher"]
                    state["flops"] = {f"{k}_gflops_per_img": round(v, 4) for k, v in tot.items()}
                    state.pop("_flop_acc")
                    say(f"flops per image (train step): {state['flops']}")
            else:
                loss, kept = train_step(student, teacher, selector, method, budget, x, y, opt, criterion,
                                        epoch, g / (epochs * steps_per_epoch), segs)
            if not math.isfinite(loss):
                raise HarnessViolation(f"non-finite loss at epoch {epoch} step {step}")
            loss_sum += loss
            n_steps += 1
            if kept is not None:
                kept_sum += kept
                kept_n += 1
        dt = time.time() - t0
        state["train_time_s"] += dt
        rec = {"epoch": epoch + 1, "loss": loss_sum / max(1, n_steps), "epoch_s": round(dt, 1),
               "img_per_s": round(n_steps * C.BATCH / dt, 1)}
        if kept_n:
            rec["mean_kept"] = kept_sum / kept_n
            state["kept"].append(rec["mean_kept"])
        if (epoch + 1) % C.EVAL_EVERY == 0 or epoch + 1 == epochs:
            ev = evaluate(student, xva, yva, device, gva)
            if isinstance(ev, dict):
                rec["val_acc"], rec["val_wga"], rec["val_group_acc"] = ev["acc"], ev["wga"], ev["group_acc"]
            else:
                rec["val_acc"] = ev
        state["history"].append(rec)
        state["epoch"] = epoch + 1
        say(json.dumps(rec))
        if (epoch + 1) % C.CKPT_EVERY == 0 or epoch + 1 == epochs:
            torch.save({"student": student.state_dict(), "opt": opt.state_dict(), "state": state,
                        "loader_rng": loader.g.get_state(), "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None}, ckpt_path)
        for frac in C.SNAPSHOT_FRACS:   # snapshots for analysis jobs (e.g. how attention changes during training)
            if epoch + 1 == max(1, int(round(frac * epochs))) and epoch + 1 < epochs:
                torch.save(student.state_dict(), rdir / f"student_e{epoch + 1}.pt")
        if state["train_time_s"] > t_limit and epoch + 1 < epochs:
            say(f"time limit {t_limit/3600:.1f} h exceeded at epoch {epoch+1}; stopping")
            write_metrics(rdir, job, state, meta, sel_sha, teacher_sha, device, status="timeout")
            return 3
    write_metrics(rdir, job, state, meta, sel_sha, teacher_sha, device, status="done")
    torch.save(student.state_dict(), rdir / "student.pt")
    ckpt_path.unlink(missing_ok=True)
    say("done")
    return 0


def write_metrics(rdir, job, state, meta, sel_sha, teacher_sha, device, status):
    last = next((h for h in reversed(state["history"]) if "val_acc" in h), {})
    last_val = last.get("val_acc")
    m = {
        "id": job["id"], "status": status, "method": job["method"], "keep": job["keep"], "seed": job["seed"],
        "tier": job["tier"], "epochs": job["epochs"], "epochs_done": state["epoch"],
        "val_acc_last": last_val,
        "val_wga_last": last.get("val_wga"), "val_group_acc_last": last.get("val_group_acc"),
        "student_lr": float(job.get("lr") or C.student_lr(job.get("dataset", C.STAGE1))),
        "lr_override": job.get("lr"),
        "mean_kept": (sum(state["kept"]) / len(state["kept"])) if state["kept"] else None,
        "dataset": job.get("dataset", C.STAGE1),
        "tome_r": r_for_budget(job["keep"]) if job["method"] == "tome" else None,
        # wall-clock throughput depends on how many jobs shared the GPU; compare within the same concurrency
        "img_per_s_mean": round(sum(h["img_per_s"] for h in state["history"]) / len(state["history"]), 1)
        if state["history"] else None,
        "flops": state["flops"], "train_time_s": round(state["train_time_s"], 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None,
        "history": state["history"], "selector_meta": meta, "selector_sha": sel_sha,
        "sel_args": job.get("sel_args", {}), "teacher_sha": teacher_sha,
        "harness_digest": harness_digest(), "git": git_hash(), "submitted_by": job.get("by"),
        "note": job.get("note", ""), "analysis_only": job["method"] in C.ANALYSIS_ONLY,
    }
    (rdir / "metrics.json").write_text(json.dumps(m, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    a = ap.parse_args()
    try:
        sys.exit(run(a.job))
    except Exception as e:  # recorded so the agent sees why a job died
        job = json.loads(open(a.job).read())
        rdir = C.RUNS / job["id"]
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "error.txt").write_text(f"{type(e).__name__}: {e}\n")
        raise
