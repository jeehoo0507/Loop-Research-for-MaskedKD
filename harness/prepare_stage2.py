"""Prepare stage 2 (ImageNet-100) from scratch. Human runs this once, while stage 1 is searching.

    python -m harness.prepare_stage2            # download + convert (~8.4 GB download, ~25 GB arrays)
    python -m harness.prepare_stage2 --teacher  # build the ImageNet-100 teacher and check its val accuracy

Data: Hugging Face dataset clane9/imagenet-100 (public, ungated; the CMC ImageNet-100 class subset of
ILSVRC-2012, so ImageNet's research-use terms apply to the images).
Splits: train = official train minus a held-out val (VAL_PER_CLASS per class, seeded); test = official
validation (5,000 images), used only by harness.test_eval.
Arrays: train uint8 [N,3,256,256] (shorter side 256, center crop; RandomResizedCrop to 224 happens on the
GPU), val/test uint8 [N,3,224,224] (shorter side 256, center crop 224, the usual eval transform).

Teacher: DeiT-S ImageNet-1k weights with the classifier rows of these 100 classes (no fine-tuning; class
names are matched to timm's ImageNet-1k descriptions, all 100 must match uniquely).
"""
import argparse
import io
import json
from multiprocessing import Pool

import numpy as np
import torch

from . import config as C

OUT = C.DATA_ROOT / "in100"


def _decode(args):
    blob, size, crop = args
    from PIL import Image
    im = Image.open(io.BytesIO(blob)).convert("RGB")
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - crop) // 2, (h - crop) // 2
    return np.asarray(im.crop((l, t, l + crop, t + crop)), dtype=np.uint8).transpose(2, 0, 1)


def _write(name, blobs, labels, size, crop, workers):
    path = OUT / f"{name}_x.u8"
    x = np.memmap(path, dtype=np.uint8, mode="w+", shape=(len(blobs), 3, crop, crop))
    with Pool(workers) as pool:
        for i, arr in enumerate(pool.imap(_decode, ((b, size, crop) for b in blobs), chunksize=64)):
            x[i] = arr
            if i % 10000 == 0:
                print(f"  {name}: {i}/{len(blobs)}", flush=True)
    x.flush()
    np.save(OUT / f"{name}_y.npy", np.asarray(labels, dtype=np.int64))
    return [len(blobs), 3, crop, crop]


def prepare(workers):
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "meta.json").exists():
        raise SystemExit(f"{OUT}/meta.json exists; delete {OUT} by hand to rebuild")
    raw = snapshot_download(C.IN100_HF_REPO, repo_type="dataset", local_dir=str(OUT / "raw"),
                            allow_patterns=["data/*.parquet"])
    from pathlib import Path

    def read(split):
        blobs, labels = [], []
        for f in sorted(Path(raw, "data").glob(f"{split}-*.parquet")):
            t = pq.read_table(f, columns=["image", "label"]).to_pydict()
            blobs += [im["bytes"] for im in t["image"]]
            labels += t["label"]
        return blobs, labels

    tr_b, tr_y = read("train")
    te_b, te_y = read("validation")
    y = torch.tensor(tr_y)
    g = torch.Generator().manual_seed(C.SPLIT_SEED)
    val_idx = torch.cat([(y == c).nonzero().flatten()[torch.randperm(int((y == c).sum()), generator=g)[:C.VAL_PER_CLASS]]
                         for c in range(C.NUM_CLASSES)]).sort().values.tolist()
    vset = set(val_idx)
    train_idx = [i for i in range(len(tr_y)) if i not in vset]
    print(f"train {len(train_idx)}, val {len(val_idx)}, test {len(te_y)}", flush=True)
    shapes = {
        "train": _write("train", [tr_b[i] for i in train_idx], [tr_y[i] for i in train_idx], C.IN100_TRAIN_RES, C.IN100_TRAIN_RES, workers),
        "val": _write("val", [tr_b[i] for i in val_idx], [tr_y[i] for i in val_idx], 256, C.IMG_SIZE, workers),
        "test": _write("test", te_b, te_y, 256, C.IMG_SIZE, workers),
    }
    (OUT / "meta.json").write_text(json.dumps({"source": C.IN100_HF_REPO, "shapes": shapes,
                                               "class_names": class_names(raw)}, indent=1))
    print("done", shapes)


def class_names(raw):
    """Label names in label order, from the Hugging Face features stored in the parquet schema metadata."""
    import pyarrow.parquet as pq
    from pathlib import Path
    first = sorted(Path(raw, "data").glob("train-*.parquet"))[0]
    hf = json.loads(pq.read_schema(first).metadata[b"huggingface"])
    names = hf["info"]["features"]["label"]["names"]
    assert len(names) == C.NUM_CLASSES, len(names)
    return names


def build_teacher():
    import timm
    from timm.data import ImageNetInfo
    from .data import load_split, normalize_resize, as_tensor
    from .models import make_teacher
    from .train import evaluate, sha256_file, teacher_paths
    meta = json.loads((OUT / "meta.json").read_text())
    info = ImageNetInfo()
    desc = [info.index_to_description(i) for i in range(1000)]
    idx = []
    for n in meta["class_names"]:
        hits = [i for i, d in enumerate(desc) if d == n]
        if len(hits) != 1:
            raise SystemExit(f"class name {n!r} matches {len(hits)} ImageNet-1k classes")
        idx.append(hits[0])
    full = timm.create_model(C.TEACHER, pretrained=True)        # 1000-way head
    t = make_teacher(pretrained=False)
    sd = full.state_dict()
    sd["head.weight"], sd["head.bias"] = sd["head.weight"][idx].clone(), sd["head.bias"][idx].clone()
    t.load_state_dict(sd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = t.to(device).eval()
    xva, yva = load_split("val", C.STAGE2)
    acc = evaluate(t, xva, yva, device)
    ckpt, meta_path = teacher_paths(C.STAGE2)
    C.STATE.mkdir(exist_ok=True)
    torch.save(t.state_dict(), ckpt)
    meta_path.write_text(json.dumps({"model": C.TEACHER, "source": "ImageNet-1k head rows, no fine-tuning",
                                     "imagenet_indices": idx, "val_acc": acc, "sha256": sha256_file(ckpt)}, indent=1))
    print(json.dumps({"val_acc": acc}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    build_teacher() if a.teacher else prepare(a.workers)
