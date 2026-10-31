"""Waterbirds and CUB-200-2011 held as uint8 tensors on the GPU, with DeiT-style augmentation done on the GPU.

Earlier repos were data-loading bound (~152 img/s on an A5000), so nothing here uses a DataLoader.
harness/prepare_data.py stores train images at 256x256 (Resize(256) -> CenterCrop(256); RandomResizedCrop to 224
happens on the GPU) and val / test images exactly as evaluated (Resize(256, bicubic) -> CenterCrop(224)).
Splits: Waterbirds official train / val / test; CUB official test, official train split 90/10 per class into
train / val (aim-lab-test-5's rule). The test split is only read by harness/test_eval.py (a human, once, at the end).
Waterbirds group labels (2*y + place) are for evaluation only; training never sees them.
"""
import json
import math
import os
import queue
import threading

import numpy as np
import torch
import torch.nn.functional as F
from timm.data import Mixup
from timm.data.random_erasing import RandomErasing

from . import config as C

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
FAKE_N = {"train": 256, "val": 96, "test": 96}


def _fake(dataset, split):
    seed = {"train": 1, "val": 2, "test": 3}[split] + 10 * C.DATASETS.index(dataset)
    g = torch.Generator().manual_seed(seed)
    n, res = FAKE_N[split], (64 if split == "train" else 56)
    x = torch.randint(0, 256, (n, 3, res, res), dtype=torch.uint8, generator=g)
    y = torch.arange(n) % C.NUM_CLASSES[dataset]
    groups = (2 * y + torch.randint(0, 2, (n,), generator=g)) if C.NUM_GROUPS[dataset] else torch.full((n,), -1)
    return x, y, groups


def _arrays(dataset, split):
    if os.environ.get("MASKEDKD_FAKE_DATA"):
        return _fake(dataset, split)
    d = C.DATA_ROOT / dataset
    meta = json.loads((d / "meta.json").read_text())
    x = np.memmap(d / f"{split}_x.u8", dtype=np.uint8, mode="r", shape=tuple(meta["shapes"][split]))
    y = torch.from_numpy(np.load(d / f"{split}_y.npy")).long()
    g = torch.from_numpy(np.load(d / f"{split}_g.npy")).long()
    return torch.from_numpy(np.ascontiguousarray(x)), y, g


def load_split(split: str, dataset: str = C.STAGE1):
    """split in {'train', 'val', 'test'} -> (uint8 images [N,3,H,W], int64 labels)."""
    x, y, _ = _arrays(dataset, split)
    return x, y


def load_groups(split: str, dataset: str = C.STAGE1):
    """Waterbirds group of every image (2*y + place), or None for datasets without groups. Evaluation only."""
    if not C.NUM_GROUPS[dataset]:
        return None
    return _arrays(dataset, split)[2]


def as_tensor(x):
    """uint8 images as a torch tensor (memmap slices are copied)."""
    return torch.from_numpy(np.ascontiguousarray(x)) if isinstance(x, np.ndarray) else x


def normalize_resize(x_uint8):
    """Eval transform: [0,255] uint8 (stored at 224 already; fake test data is resized) -> normalized float 224px."""
    x = as_tensor(x_uint8).float() / 255.0
    x = F.interpolate(x, size=C.IMG_SIZE, mode="bilinear", align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


class GpuAugment:
    """RandomResizedCrop + flip + simplified RandAugment + normalize + RandomErasing + Mixup/CutMix.

    Geometric RandAugment ops (rotate, shear, translate) are folded into the same affine warp as the
    crop, so every image is resampled once (32px -> 224px). Color ops act on the 32px image.
    This is a simplified RandAugment (no equalize / sharpness); it is the same for every method.
    """

    COLOR = ("identity", "autocontrast", "brightness", "contrast", "color", "solarize", "posterize")
    GEOM = ("rotate", "shear_x", "shear_y", "translate_x", "translate_y")

    def __init__(self, device, num_classes, smoothing=C.SMOOTHING, mixup=True, erase=True):
        self.device = device
        self.num_classes = num_classes
        self.ops = self.COLOR + self.GEOM
        self.mixup = Mixup(mixup_alpha=C.MIXUP, cutmix_alpha=C.CUTMIX, label_smoothing=smoothing,
                           num_classes=num_classes) if mixup else None
        self.erase = RandomErasing(probability=C.REPROB, mode="pixel", device=str(device)) if erase else None
        self.smoothing = smoothing

    def _mag(self, n):
        # magnitude M with std 0.5, clipped to [0, 10], as fraction of max
        m = (C.RAND_AUG_M + 0.5 * torch.randn(n, device=self.device)).clamp(0, 10) / 10.0
        return m

    def __call__(self, x_uint8, y, extra=None):
        """extra: optional float map [B,1,h,w] over the same (un-augmented) images, e.g. a cached teacher
        attribution. It gets the same crop/flip/affine warp, and the same Mixup/CutMix (as a 4th channel, which
        leaves the image math and the random draws unchanged); returned as patch-grid means [B,196] after (x, y)."""
        B = x_uint8.shape[0]
        dev = self.device
        x = x_uint8.to(dev, non_blocking=True).float() / 255.0
        y = y.to(dev, non_blocking=True)

        rot = torch.zeros(B, device=dev)
        shx = torch.zeros(B, device=dev)
        shy = torch.zeros(B, device=dev)
        tx = torch.zeros(B, device=dev)
        ty = torch.zeros(B, device=dev)
        for _ in range(C.RAND_AUG_N):
            op = torch.randint(len(self.ops), (B,), device=dev)
            m = self._mag(B)
            sign = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
            for i, name in enumerate(self.ops):
                sel = (op == i)
                if not sel.any():
                    continue
                s4 = sel.view(B, 1, 1, 1)
                if name == "autocontrast":
                    lo = x.amin(dim=(2, 3), keepdim=True)
                    hi = x.amax(dim=(2, 3), keepdim=True)
                    x = torch.where(s4, (x - lo) / (hi - lo).clamp_min(1e-3), x)
                elif name == "brightness":
                    f = (1 + sign * m * 0.9).view(B, 1, 1, 1)
                    x = torch.where(s4, (x * f).clamp(0, 1), x)
                elif name == "contrast":
                    f = (1 + sign * m * 0.9).view(B, 1, 1, 1)
                    mean = x.mean(dim=(1, 2, 3), keepdim=True)
                    x = torch.where(s4, ((x - mean) * f + mean).clamp(0, 1), x)
                elif name == "color":
                    f = (1 + sign * m * 0.9).view(B, 1, 1, 1)
                    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
                    x = torch.where(s4, ((x - gray) * f + gray).clamp(0, 1), x)
                elif name == "solarize":
                    thr = (1 - m).view(B, 1, 1, 1)
                    x = torch.where(s4 & (x >= thr), 1 - x, x)
                elif name == "posterize":
                    bits = (8 - (m * 4).round()).view(B, 1, 1, 1)
                    q = 2 ** (8 - bits)
                    x = torch.where(s4, torch.floor(x * 255 / q) * q / 255, x)
                elif name == "rotate":
                    rot = torch.where(sel, sign * m * math.radians(30), rot)
                elif name == "shear_x":
                    shx = torch.where(sel, sign * m * 0.3, shx)
                elif name == "shear_y":
                    shy = torch.where(sel, sign * m * 0.3, shy)
                elif name == "translate_x":
                    tx = torch.where(sel, sign * m * 0.45 * 2, tx)   # fraction of width, in [-1,1] coords
                elif name == "translate_y":
                    ty = torch.where(sel, sign * m * 0.45 * 2, ty)

        # RandomResizedCrop(scale=(0.08,1), ratio=(3/4,4/3)) as an affine map in normalized coords
        area = torch.empty(B, device=dev).uniform_(0.08, 1.0)
        logr = torch.empty(B, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3))
        r = torch.exp(logr)
        w = torch.sqrt(area * r).clamp(max=1.0)
        h = torch.sqrt(area / r).clamp(max=1.0)
        cx = (torch.rand(B, device=dev) * 2 - 1) * (1 - w)
        cy = (torch.rand(B, device=dev) * 2 - 1) * (1 - h)
        flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)

        crop = torch.zeros(B, 3, 3, device=dev)
        crop[:, 0, 0] = w * flip
        crop[:, 0, 2] = cx
        crop[:, 1, 1] = h
        crop[:, 1, 2] = cy
        crop[:, 2, 2] = 1
        geo = torch.zeros(B, 3, 3, device=dev)
        c, s = torch.cos(rot), torch.sin(rot)
        geo[:, 0, 0] = c
        geo[:, 0, 1] = -s + shx
        geo[:, 1, 0] = s + shy
        geo[:, 1, 1] = c
        geo[:, 0, 2] = tx
        geo[:, 1, 2] = ty
        geo[:, 2, 2] = 1
        theta = (geo @ crop)[:, :2]
        grid = F.affine_grid(theta, (B, 3, C.IMG_SIZE, C.IMG_SIZE), align_corners=False)
        # padding fill: mid gray (timm RandAugment fills with (128,128,128))
        x = F.grid_sample(x - 0.5, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + 0.5

        x = (x - MEAN.to(dev)) / STD.to(dev)
        if self.erase is not None:
            x = self.erase(x)
        if extra is not None:   # same warp; outside the source image the map is 0
            e = F.grid_sample(extra.to(dev).float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            x = torch.cat([x, e], dim=1)
        if self.mixup is not None:
            if B % 2 == 1:
                x, y = x[:-1], y[:-1]
            x, y = self.mixup(x, y)
        else:
            y = F.one_hot(y, self.num_classes).float() * (1 - self.smoothing) + self.smoothing / self.num_classes
        if extra is None:
            return x, y
        g = C.IMG_SIZE // C.PATCH
        tokens = F.adaptive_avg_pool2d(x[:, 3:4], g).flatten(1)
        return x[:, :3].contiguous(), y, tokens


class GpuLoader:
    """Shuffled minibatches from uint8 tensors kept on `device`. drop_last for training."""

    def __init__(self, x, y, batch, device, shuffle=True, drop_last=True, seed=0):
        self.x = x.to(device)
        self.y = y.to(device)
        self.batch = batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.g = torch.Generator(device="cpu").manual_seed(seed)

    def __len__(self):
        n = len(self.y)
        return n // self.batch if self.drop_last else math.ceil(n / self.batch)

    def __iter__(self):
        n = len(self.y)
        perm = torch.randperm(n, generator=self.g).to(self.x.device) if self.shuffle else torch.arange(n, device=self.x.device)
        for i in range(len(self)):
            idx = perm[i * self.batch:(i + 1) * self.batch]
            self.last_idx = idx            # dataset rows of the batch just yielded (cached teacher maps)
            yield self.x[idx], self.y[idx]


class DiskLoader:
    """Shuffled minibatches read from a uint8 memmap by a background thread, pinned and copied to `device`.
    Used for ImageNet-100 (too large to keep on the GPU). Same interface as GpuLoader."""

    def __init__(self, x, y, batch, device, shuffle=True, drop_last=True, seed=0, prefetch=4):
        self.x, self.y = x, y
        self.batch, self.device, self.shuffle, self.drop_last, self.prefetch = batch, device, shuffle, drop_last, prefetch
        self.g = torch.Generator(device="cpu").manual_seed(seed)

    def __len__(self):
        n = len(self.y)
        return n // self.batch if self.drop_last else math.ceil(n / self.batch)

    def __iter__(self):
        n = len(self.y)
        perm = torch.randperm(n, generator=self.g) if self.shuffle else torch.arange(n)
        q = queue.Queue(maxsize=self.prefetch)

        def work():
            for i in range(len(self)):
                idx = perm[i * self.batch:(i + 1) * self.batch].sort().values.numpy()   # sorted: disk locality
                xb = torch.from_numpy(np.ascontiguousarray(self.x[idx]))
                if self.device.type == "cuda":
                    xb = xb.pin_memory()
                q.put((xb, self.y[idx], idx))
            q.put(None)

        threading.Thread(target=work, daemon=True).start()
        while (item := q.get()) is not None:
            xb, yb, idx = item
            self.last_idx = torch.from_numpy(idx)
            yield xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)


def make_loader(x, y, batch, device, seed, dataset):
    if isinstance(x, np.ndarray):
        return DiskLoader(x, y, batch, device, seed=seed)
    return GpuLoader(x, y, batch, device, seed=seed)
