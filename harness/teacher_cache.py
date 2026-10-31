"""Cached teacher attribution: an amortized teacher signal for selectors (QUEUE 2026-10-31 21:30 KST).

Once per training image, before training, the teacher sees the whole un-augmented image (stored 256px, resized
to 224) and its attention rollout (Abnar & Zuidema 2020; same rule as the `rollout` baseline: head mean,
0.5 A + 0.5 I, rows renormalised, product over all blocks) gives a 14x14 map of where it looks. The map is kept
in state/teacher_map_<dataset>.pt and reused every epoch: harness/data.py GpuAugment warps it with the same
crop/flip/affine and mixes it with the same Mixup/CutMix as the image, so ctx.teacher_map [B,196] lines up with
the tokens the teacher will get. Logits could not be cached like this (they change with every augmentation);
a map of where the teacher looks can be moved with the crop.

Cost: one full teacher forward per training image, measured with FlopCounterMode when the cache is built, and
spread over the run: every train step of a selector that uses the cache is charged
gflops_per_image * n_train / (epochs * steps_per_epoch * batch) per image (`amortized_gflops`), inside the
selector and total GFLOPs.
"""
import os

import torch
from torch.utils.flop_counter import FlopCounterMode

from . import config as C
from .data import as_tensor, load_split, normalize_resize
from .models import student_forward

BATCH = 64


def cache_path(dataset):
    return C.STATE / f"teacher_map_{dataset}.pt"


def rollout_cls(attn):
    """attn: list of [B,H,N,N] per block -> CLS row of the rollout over the patches [B,196], rows sum to 1."""
    r = None
    for a in attn:
        a = a.float().mean(dim=1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        r = a if r is None else a @ r
    m = r[:, 0, 1:]
    return m / m.sum(dim=1, keepdim=True)


@torch.no_grad()
def build(dataset, teacher, device):
    """Teacher rollout maps [N,1,14,14] (float16) of every training image, and the measured cost per image."""
    xtr, _ = load_split("train", dataset)
    was_training = teacher.training
    teacher.eval()
    depth = len(teacher.blocks)
    g = C.IMG_SIZE // C.PATCH
    maps, gflops = [], None
    for i in range(0, len(xtr), BATCH):
        x = normalize_resize(as_tensor(xtr[i:i + BATCH])).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if gflops is None:
                with FlopCounterMode(display=False) as fc:
                    out = student_forward(teacher, x, attn_layers=range(depth))
                gflops = fc.get_total_flops() / x.shape[0] / 1e9
            else:
                out = student_forward(teacher, x, attn_layers=range(depth))
        maps.append(rollout_cls(out.attn).view(-1, 1, g, g).half().cpu())
    teacher.train(was_training)
    return torch.cat(maps), gflops


def load_or_build(dataset, teacher, teacher_sha, device):
    """Returns (maps [N,1,14,14] on device, gflops per image). Rebuilt when the teacher weights change."""
    p = cache_path(dataset)
    if p.exists():
        d = torch.load(p, map_location="cpu", weights_only=False)
        if d.get("teacher_sha") == teacher_sha and d["maps"].shape[0] == len(load_split("train", dataset)[1]):
            return d["maps"].to(device), d["gflops_per_image"]
    maps, gflops = build(dataset, teacher, device)
    C.STATE.mkdir(exist_ok=True)
    tmp = p.with_suffix(f".tmp{os.getpid()}")
    torch.save({"maps": maps, "gflops_per_image": gflops, "teacher_sha": teacher_sha,
                "rule": "teacher attention rollout, whole train image resized to 224, no augmentation"}, tmp)
    tmp.replace(p)   # two jobs building at once: the last complete file wins, never a partial one
    return maps.to(device), gflops


def amortized_gflops(gflops_per_image, n_train, epochs, steps_per_epoch, batch):
    """Cache cost charged to each image of each train step."""
    return gflops_per_image * n_train / (epochs * steps_per_epoch * batch)
