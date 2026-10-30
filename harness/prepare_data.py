"""Download Waterbirds and CUB-200-2011 and store them as uint8 arrays (bootstrap runs this once).

    python -m harness.prepare_data [--datasets waterbirds cub] [--workers 8]

Sources: the official archives (config.DATA_URLS). If aim-lab-test-5 already extracted them under $STAGE0_DATA
(~/stage0/data/<dataset>/...), those folders are used and nothing is downloaded.
Splits (same as aim-lab-test-5, so its teachers can be reused):
  Waterbirds: metadata.csv split column (train 4795 / val 1199 / test 5794); group = 2*y + place.
  CUB: official test 5794; official train split per class 90 / 10 into train 5394 / val 600 (random.Random(0)).
Stored per dataset in $DATA_ROOT/<dataset>/: {split}_x.u8 (memmap uint8 [N,3,H,W]), {split}_y.npy, {split}_g.npy,
meta.json. Train images: Resize(256, bicubic) -> CenterCrop(256) (RandomResizedCrop to 224 runs on the GPU).
Val / test: Resize(256, bicubic) -> CenterCrop(224), exactly the evaluation view.
"""
import argparse
import csv
import json
import random
import subprocess
import tarfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import config as C

FOLDER = {"waterbirds": "waterbird_complete95_forest2water2", "cub": "CUB_200_2011"}
ARCHIVE = {"waterbirds": "waterbird_complete95_forest2water2.tar.gz", "cub": "CUB_200_2011.tgz"}


def _load(args):
    path, crop = args
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = C.RESIZE / min(w, h)
        im = im.resize((max(C.RESIZE, round(w * s)), max(C.RESIZE, round(h * s))), Image.BICUBIC)
        w, h = im.size
        l, t = (w - crop) // 2, (h - crop) // 2
        return np.asarray(im.crop((l, t, l + crop, t + crop)), dtype=np.uint8).transpose(2, 0, 1)


def _write(out, split, items, crop, workers):
    arr = np.memmap(out / f"{split}_x.u8", dtype=np.uint8, mode="w+", shape=(len(items), 3, crop, crop))
    with Pool(workers) as pool:
        for i, img in enumerate(pool.imap(_load, [(p, crop) for p, _, _ in items], chunksize=32)):
            arr[i] = img
    arr.flush()
    np.save(out / f"{split}_y.npy", np.array([y for _, y, _ in items], dtype=np.int64))
    np.save(out / f"{split}_g.npy", np.array([g for _, _, g in items], dtype=np.int64))
    return [len(items), 3, crop, crop]


def _source(dataset):
    """Extracted dataset folder: aim-lab-test-5's if present, else download + extract into $DATA_ROOT/raw."""
    for root in (C.STAGE0_DATA / dataset, C.DATA_ROOT / "raw"):
        if (root / FOLDER[dataset]).is_dir():
            return root / FOLDER[dataset]
    raw = C.DATA_ROOT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    arc = raw / ARCHIVE[dataset]
    for url in C.DATA_URLS[dataset]:
        if arc.exists() and arc.stat().st_size > 1 << 20:
            break
        print(f"downloading {url}", flush=True)
        subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "-C", "-", "-A", "Mozilla/5.0", "-o", str(arc), url])
    if not arc.exists() or arc.stat().st_size < 1 << 20:
        raise SystemExit(f"could not download {ARCHIVE[dataset]}; put it at {arc} by hand")
    with tarfile.open(arc) as t:
        t.extractall(raw, filter="data")
    return raw / FOLDER[dataset]


def splits_waterbirds(root):
    out = {"train": [], "val": [], "test": []}
    names = {0: "train", 1: "val", 2: "test"}
    with open(root / "metadata.csv") as f:
        for row in csv.DictReader(f):
            y, place = int(row["y"]), int(row["place"])
            out[names[int(row["split"])]].append((str(root / row["img_filename"]), y, 2 * y + place))
    return out


def splits_cub(root):
    rd = lambda n: [line.split() for line in open(root / n) if line.strip()]   # noqa: E731
    images = {int(i): p for i, p in rd("images.txt")}
    labels = {int(i): int(c) - 1 for i, c in rd("image_class_labels.txt")}
    is_train = {int(i): int(t) == 1 for i, t in rd("train_test_split.txt")}
    train_full = [i for i in sorted(images) if is_train[i]]
    by_class = defaultdict(list)
    for i in train_full:
        by_class[labels[i]].append(i)
    rng, val = random.Random(C.CUB_VAL_SEED), []
    for c in sorted(by_class):
        ids = sorted(by_class[c])
        val += rng.sample(ids, max(1, round(C.CUB_VAL_FRACTION * len(ids))))
    vs = set(val)
    item = lambda i: (str(root / "images" / images[i]), labels[i], -1)   # noqa: E731
    return {"train": [item(i) for i in train_full if i not in vs], "val": [item(i) for i in sorted(val)],
            "test": [item(i) for i in sorted(images) if not is_train[i]]}


def prepare(dataset, workers):
    out = C.DATA_ROOT / dataset
    if (out / "meta.json").exists():
        print(f"{dataset}: already prepared ({out})")
        return
    root = _source(dataset)
    sp = (splits_waterbirds if dataset == "waterbirds" else splits_cub)(root)
    sizes = {k: len(v) for k, v in sp.items()}
    if sizes != C.EXPECTED_SIZES[dataset]:
        raise SystemExit(f"{dataset} split sizes {sizes} != expected {C.EXPECTED_SIZES[dataset]}")
    out.mkdir(parents=True, exist_ok=True)
    shapes = {s: _write(out, s, sp[s], C.RESIZE if s == "train" else C.IMG_SIZE, workers) for s in ("train", "val", "test")}
    (out / "meta.json").write_text(json.dumps({"dataset": dataset, "source": str(root), "shapes": shapes,
                                               "sizes": sizes}, indent=1))
    print(f"{dataset}: {sizes} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS), choices=list(C.DATASETS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for d in a.datasets:
        prepare(d, a.workers)
