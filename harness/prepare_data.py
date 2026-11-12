"""Download Waterbirds, CUB-200-2011 and FGVC-Aircraft and store them as uint8 arrays (bootstrap runs this once).

    python -m harness.prepare_data [--datasets waterbirds cub aircraft] [--workers 8]

Sources: the official archives (config.DATA_URLS). If aim-lab-test-5 already extracted them under $STAGE0_DATA
(~/stage0/data/<dataset>/...), those folders are used and nothing is downloaded.
Splits (same as aim-lab-test-5, so its teachers can be reused):
  Waterbirds: metadata.csv split column (train 4795 / val 1199 / test 5794); group = 2*y + place.
  CUB: official test 5794; official train split per class 90 / 10 into train 5394 / val 600 (random.Random(0)).
  Aircraft (2013b, 100 variants): official test 3333; official trainval split per class 90 / 10 (the CUB rule) into
  train 5967 / val 700. The 20-px copyright banner at the bottom of every image is cut off first.
Stored per dataset in $DATA_ROOT/<dataset>/: {split}_x.u8 (memmap uint8 [N,3,H,W]), {split}_y.npy, {split}_g.npy,
meta.json. Train images: Resize(256, bicubic) -> CenterCrop(256) (RandomResizedCrop to 224 runs on the GPU).
Val / test: Resize(256, bicubic) -> CenterCrop(224), exactly the evaluation view.
CIFAR-10 / CIFAR-10-C (config.CIFAR_SETS) come as arrays, not image files: stored at 32x32 as they are; every
split of every CIFAR set holds the same test-set images (see cifar_arrays).
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

FOLDER = {"waterbirds": "waterbird_complete95_forest2water2", "cub": "CUB_200_2011", "aircraft": "fgvc-aircraft-2013b",
          "caltech256": "256_ObjectCategories", "cub_lt": "CUB_200_2011", "celeba": "celeba"}
ARCHIVE = {"waterbirds": "waterbird_complete95_forest2water2.tar.gz", "cub": "CUB_200_2011.tgz",
           "aircraft": "fgvc-aircraft-2013b.tar.gz", "caltech256": "256_ObjectCategories.tar", "cub_lt": "CUB_200_2011.tgz"}


def _load(args):
    path, crop, trim = args
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if trim:
            im = im.crop((0, 0, w, h - trim))
            h -= trim
        s = C.RESIZE / min(w, h)
        im = im.resize((max(C.RESIZE, round(w * s)), max(C.RESIZE, round(h * s))), Image.BICUBIC)
        w, h = im.size
        l, t = (w - crop) // 2, (h - crop) // 2
        return np.asarray(im.crop((l, t, l + crop, t + crop)), dtype=np.uint8).transpose(2, 0, 1)


def _write(out, split, items, crop, workers, trim=0):
    arr = np.memmap(out / f"{split}_x.u8", dtype=np.uint8, mode="w+", shape=(len(items), 3, crop, crop))
    with Pool(workers) as pool:
        for i, img in enumerate(pool.imap(_load, [(p, crop, trim) for p, _, _ in items], chunksize=32)):
            arr[i] = img
    arr.flush()
    np.save(out / f"{split}_y.npy", np.array([y for _, y, _ in items], dtype=np.int64))
    np.save(out / f"{split}_g.npy", np.array([g for _, _, g in items], dtype=np.int64))
    return [len(items), 3, crop, crop]


def _source_celeba(raw):
    """CelebA from the parquet shards (config.DATA_URLS): images written as raw/celeba/img/<image_id> and the two
    attributes this harness uses as raw/celeba/attrs.csv (image_id, blond, male; -1/1 as in the official file)."""
    import pyarrow.parquet as pq
    out, shards = raw / FOLDER["celeba"], raw / "celeba_parquet"
    (out / "img").mkdir(parents=True, exist_ok=True)
    shards.mkdir(parents=True, exist_ok=True)
    base = C.DATA_URLS["celeba"][0]
    rows = []
    for i in range(C.CELEBA_SHARDS):
        name = f"train-{i:05d}-of-{C.CELEBA_SHARDS:05d}.parquet"
        f = shards / name
        if not f.exists() or f.stat().st_size < 1 << 16:
            print(f"downloading {name}", flush=True)
            rc = subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "--connect-timeout", "30", "-A",
                                 "Mozilla/5.0", "-o", str(f), base + name]).returncode
            if rc:
                print(f"curl exit code {rc} for {base + name}", flush=True)
        if not f.exists() or f.stat().st_size < 1 << 16:
            raise SystemExit(f"could not download {name}; put the shards in {shards} by hand")
        # 2026-11-04 the first run died here without a message (stage2_boot.log ends at "downloading train-00000"):
        # read in small batches (never a whole shard as Python objects) and print before reading; main() turns on
        # faulthandler so a crash inside pyarrow leaves a traceback
        print(f"reading {name} ({f.stat().st_size} bytes)", flush=True)
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["image", "image_id", "Blond_Hair", "Male"]):
            for r in batch.to_pylist():
                img = out / "img" / r["image_id"]
                if not img.exists():
                    img.write_bytes(r["image"]["bytes"])
                rows.append((r["image_id"], int(r["Blond_Hair"]), int(r["Male"])))
    with open(out / "attrs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "blond", "male"])
        w.writerows(sorted(rows))
    return out


def _source(dataset):
    """Extracted dataset folder: aim-lab-test-5's if present, else download + extract into $DATA_ROOT/raw."""
    for root in (C.STAGE0_DATA / dataset, C.DATA_ROOT / "raw"):
        if (root / FOLDER[dataset]).is_dir() and (dataset != "celeba" or (root / FOLDER[dataset] / "attrs.csv").is_file()):
            return root / FOLDER[dataset]
    if dataset == "celeba":
        return _source_celeba(C.DATA_ROOT / "raw")
    raw = C.DATA_ROOT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    arc = raw / ARCHIVE[dataset]
    for url in C.DATA_URLS[dataset]:
        if arc.exists() and arc.stat().st_size > 1 << 20:
            break
        print(f"downloading {url}", flush=True)
        subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "--connect-timeout", "30", "-C", "-", "-A", "Mozilla/5.0",
                        "-o", str(arc), url])   # connect timeout: an unreachable host should not hold up the next URL
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


def _split_per_class(ids, label):
    """90 / 10 per class with random.Random(CUB_VAL_SEED): the CUB rule. Returns (train ids, val ids)."""
    by_class = defaultdict(list)
    for i in ids:
        by_class[label[i]].append(i)
    rng, val = random.Random(C.CUB_VAL_SEED), []
    for c in sorted(by_class):
        cs = sorted(by_class[c])
        val += rng.sample(cs, max(1, round(C.CUB_VAL_FRACTION * len(cs))))
    vs = set(val)
    return [i for i in ids if i not in vs], sorted(val)


def splits_aircraft(root):
    d = root / "data"
    variants = [v.strip() for v in open(d / "variants.txt") if v.strip()]
    cls = {v: k for k, v in enumerate(variants)}

    def read(split):   # lines "<image id> <variant name, may contain spaces>"
        return [tuple(line.rstrip("\n").split(" ", 1)) for line in open(d / f"images_variant_{split}.txt") if line.strip()]
    trainval, test = read("trainval"), read("test")
    label = {i: cls[v] for i, v in trainval + test}
    train, val = _split_per_class(sorted(i for i, _ in trainval), label)
    item = lambda i: (str(d / "images" / f"{i}.jpg"), label[i], -1)   # noqa: E731
    return {"train": [item(i) for i in train], "val": [item(i) for i in val], "test": [item(i) for i, _ in test]}


def splits_caltech256(root):
    """256 object classes (folders 001..256; 257.clutter is not a class), per class 80 / 10 / 10 train / val / test
    with random.Random(CUB_VAL_SEED) over the sorted file names."""
    out = {"train": [], "val": [], "test": []}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        c = int(d.name.split(".", 1)[0]) - 1
        if c >= 256:
            continue
        files = sorted(str(f) for f in d.iterdir() if f.suffix.lower() == ".jpg")
        random.Random(C.CUB_VAL_SEED + c).shuffle(files)
        n_val = n_test = max(1, round(0.1 * len(files)))
        for i, f in enumerate(files):
            out["test" if i < n_test else "val" if i < n_test + n_val else "train"].append((f, c, -1))
    return out


def splits_cub_lt(root):
    """CUB long-tail: the CUB split with classes C.LT_TAIL_CLASSES cut to C.LT_TAIL_FRACTION of their train images
    (random.Random(C.CUB_VAL_SEED + 1000 + class), at least 1). val/test unchanged. Group = 1 for a tail class."""
    sp = splits_cub(root)
    tail = set(C.LT_TAIL_CLASSES)
    by_class = defaultdict(list)
    for item in sp["train"]:
        by_class[item[1]].append(item)
    train = []
    for c in sorted(by_class):
        items = by_class[c]
        if c in tail:
            rng = random.Random(C.CUB_VAL_SEED + 1000 + c)
            items = rng.sample(items, max(1, round(C.LT_TAIL_FRACTION * len(items))))
        train += items
    grp = lambda item: (item[0], item[1], int(item[1] in tail))   # noqa: E731
    return {"train": [grp(i) for i in train], "val": [grp(i) for i in sp["val"]], "test": [grp(i) for i in sp["test"]]}


def splits_celeba(root):
    """CelebA (config: CELEBA_*): y = blond, group = 2*y + male. Official partition by image number; train = random
    CELEBA_TRAIN_FRACTION of official train (random.Random(CUB_VAL_SEED)); val/test keep at most CELEBA_EVAL_CAP
    images per group (random.Random(CUB_VAL_SEED + 1 + group), minority groups whole)."""
    parts = {"train": [], "val": [], "test": []}
    with open(root / "attrs.csv") as f:
        for row in csv.DictReader(f):
            n = int(row["image_id"].split(".")[0])
            y, a = int(row["blond"] == "1"), int(row["male"] == "1")
            split = "train" if n <= 162770 else ("val" if n <= 182637 else "test")
            parts[split].append((str(root / "img" / row["image_id"]), y, 2 * y + a))
    out = {}
    tr = sorted(parts["train"])
    out["train"] = sorted(random.Random(C.CUB_VAL_SEED).sample(tr, round(C.CELEBA_TRAIN_FRACTION * len(tr))))
    for split in ("val", "test"):
        kept = []
        for g in range(4):
            items = sorted(it for it in parts[split] if it[2] == g)
            if len(items) > C.CELEBA_EVAL_CAP:
                items = sorted(random.Random(C.CUB_VAL_SEED + 1 + g).sample(items, C.CELEBA_EVAL_CAP))
            kept += items
        out[split] = kept
    return out


def _fetch(urls, arc, min_size=1 << 20):
    """Download `arc` from the first URL that works (resumable); a file already there is kept."""
    arc.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        if arc.exists() and arc.stat().st_size >= min_size:
            break
        print(f"downloading {url}", flush=True)
        rc = subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "--connect-timeout", "30", "-C", "-", "-A",
                             "Mozilla/5.0", "-o", str(arc), url]).returncode
        if rc:
            print(f"curl exit code {rc} for {url}", flush=True)
    if not arc.exists() or arc.stat().st_size < min_size:
        raise SystemExit(f"could not download {arc.name}; put it at {arc} by hand")
    return arc


def _extract(arc, wanted, raw):
    """Extract only the members named in `wanted` (paths inside the archive, with or without a leading './')."""
    left = set(wanted)
    with tarfile.open(arc) as t:
        for m in t:
            name = m.name[2:] if m.name.startswith("./") else m.name
            if name in left:
                m.name = name
                t.extract(m, raw, filter="data")
                left.discard(name)
            if not left:
                break
    if left:
        raise SystemExit(f"{arc.name} has no member {sorted(left)}")


def _cifar_clean(raw):
    """CIFAR-10 test set from the binary archive: 10,000 records of 1 label byte + 3072 pixel bytes (R, G, B planes,
    row-major). Returns (uint8 [10000,3,32,32], int64 [10000])."""
    f = raw / "cifar-10-batches-bin" / "test_batch.bin"
    if not f.exists():
        _extract(_fetch(C.CIFAR_URLS["clean"], raw / "cifar-10-binary.tar.gz"), ["cifar-10-batches-bin/test_batch.bin"], raw)
    rec = np.fromfile(f, dtype=np.uint8).reshape(-1, 3073)
    return np.ascontiguousarray(rec[:, 1:].reshape(-1, 3, 32, 32)), rec[:, 0].astype(np.int64)


def _cifar_c(raw, corruption):
    """One corruption of CIFAR-10-C: <corruption>.npy is uint8 [5 * 10000, 32, 32, 3], severity 1..5 in blocks of
    10,000, each block in CIFAR-10 test-set order; labels.npy repeats the test labels 5 times.
    Returns (uint8 [10000,3,32,32] at config.CIFAR_C_SEVERITY, int64 [10000])."""
    d = raw / "CIFAR-10-C"
    need = [f"CIFAR-10-C/{corruption}.npy", "CIFAR-10-C/labels.npy"]
    missing = [n for n in need if not (raw / n).exists()]
    if missing:   # only the files this corruption needs (the archive holds 19 corruptions, 2.9 GB)
        _extract(_fetch(C.CIFAR_URLS["c"], raw / "CIFAR-10-C.tar"), missing, raw)
    x, y = np.load(d / f"{corruption}.npy", mmap_mode="r"), np.load(d / "labels.npy")
    n = len(y) // 5
    lo = (C.CIFAR_C_SEVERITY - 1) * n
    return np.ascontiguousarray(x[lo:lo + n].transpose(0, 3, 1, 2)), y[lo:lo + n].astype(np.int64)


def cifar_split(labels):
    """Row indices {train, val, test} of the test-set images: per class, shuffled with random.Random(CUB_VAL_SEED +
    class), the first CIFAR_SPLIT[2] go to test, the next CIFAR_SPLIT[1] to val, the rest to train (the Caltech-256
    rule). It depends on the labels only, so the clean set and every corruption get the same rows."""
    by_class = defaultdict(list)
    for i, c in enumerate(labels.tolist()):
        by_class[c].append(i)
    out = {"train": [], "val": [], "test": []}
    for c in sorted(by_class):
        ids = by_class[c]
        random.Random(C.CUB_VAL_SEED + c).shuffle(ids)
        n_test, n_val = round(C.CIFAR_SPLIT[2] * len(ids)), round(C.CIFAR_SPLIT[1] * len(ids))
        out["test"] += ids[:n_test]
        out["val"] += ids[n_test:n_test + n_val]
        out["train"] += ids[n_test + n_val:]
    return {k: sorted(v) for k, v in out.items()}


def cifar_arrays(dataset, raw):
    """{split: (uint8 [N,3,32,32], int64 [N])} for a config.CIFAR_SETS dataset. A corrupted set must carry the clean
    test labels in the same order, or its images are not the paired versions of the clean ones: checked here."""
    x, y = _cifar_clean(raw)
    if dataset in C.CIFAR_C:
        x, yc = _cifar_c(raw, C.CIFAR_C[dataset])
        if not np.array_equal(y, yc):
            raise SystemExit(f"{dataset}: CIFAR-10-C labels are not the CIFAR-10 test labels in order (pairing broken)")
    return {split: (x[rows], y[rows]) for split, rows in cifar_split(y).items()}


def _write_arrays(out, split, x, y):
    arr = np.memmap(out / f"{split}_x.u8", dtype=np.uint8, mode="w+", shape=x.shape)
    arr[:] = x
    arr.flush()
    np.save(out / f"{split}_y.npy", y)
    np.save(out / f"{split}_g.npy", np.full(len(y), -1, dtype=np.int64))
    return list(x.shape)


SPLITS = {"waterbirds": splits_waterbirds, "cub": splits_cub, "aircraft": splits_aircraft, "caltech256": splits_caltech256,
          "cub_lt": splits_cub_lt, "celeba": splits_celeba}


def prepare(dataset, workers):
    out = C.DATA_ROOT / dataset
    if (out / "meta.json").exists():
        print(f"{dataset}: already prepared ({out})")
        return
    if dataset in C.CIFAR_SETS:
        raw = C.DATA_ROOT / "raw"
        sp = cifar_arrays(dataset, raw)
        sizes = {k: len(v[1]) for k, v in sp.items()}
        if sizes != C.EXPECTED_SIZES[dataset]:
            raise SystemExit(f"{dataset} split sizes {sizes} != expected {C.EXPECTED_SIZES[dataset]}")
        out.mkdir(parents=True, exist_ok=True)
        shapes = {s: _write_arrays(out, s, *sp[s]) for s in ("train", "val", "test")}
        (out / "meta.json").write_text(json.dumps({"dataset": dataset, "source": str(raw), "shapes": shapes,
                                                   "sizes": sizes, "trim_bottom": 0}, indent=1))
        print(f"{dataset}: {sizes} -> {out}")
        return
    root = _source(dataset)
    sp = SPLITS[dataset](root)
    sizes = {k: len(v) for k, v in sp.items()}
    exp = dict(C.EXPECTED_SIZES[dataset])
    if exp.pop("partial", False):   # only the listed splits are checked (a derived train split logs its size)
        sizes_checked = {k: v for k, v in sizes.items() if k in exp}
    else:
        sizes_checked = sizes
    if ("total" in exp and sum(sizes.values()) != exp["total"]) or ("total" not in exp and sizes_checked != exp):
        raise SystemExit(f"{dataset} split sizes {sizes} != expected {C.EXPECTED_SIZES[dataset]}")
    out.mkdir(parents=True, exist_ok=True)
    trim = C.TRIM_BOTTOM.get(dataset, 0)
    shapes = {s: _write(out, s, sp[s], C.RESIZE if s == "train" else C.IMG_SIZE, workers, trim)
              for s in ("train", "val", "test")}
    (out / "meta.json").write_text(json.dumps({"dataset": dataset, "source": str(root), "shapes": shapes,
                                               "sizes": sizes, "trim_bottom": trim}, indent=1))
    print(f"{dataset}: {sizes} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS), choices=list(C.DATASETS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import faulthandler
    faulthandler.enable()   # a crash in native code (pyarrow, PIL) prints a Python traceback instead of nothing
    for d in a.datasets:
        prepare(d, a.workers)
