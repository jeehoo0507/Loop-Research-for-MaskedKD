"""Tokens or images? One seed, six runs, on Imagenette (A5000 or Colab GPU).

    python quick_train.py                      # all six runs, seed 1, 50 epochs (about 1-1.5 h on an A5000)
    python quick_train.py --runs ce mkd50 img50 --epochs 30
    python quick_train.py --data /path/dir     # where Imagenette is downloaded and cached (default ./imagenette_cache)

Teacher: DeiT-S, ImageNet-1k pretrained, frozen, its 10 Imagenette logits (no fine-tuning needed: 86 % on this val).
Student: DeiT-Ti from scratch (KD matters a lot there). Loss = CE(student, label) + KD, KD = KL(teacher || student),
tau 1, weight 1 (MaskedKD's default). Same augmentation for every run (random resized crop + flip, on the GPU).

  ce      no teacher
  full    teacher on every image, all tokens                       teacher cost 100 %
  mkd50   MaskedKD: teacher on every image, top 50 % of the
          patches by the student's last-block CLS attention         ~50 %
  img50   no masking: teacher on a random 50 % of each batch, all
          tokens; KD averaged over those images (unbiased for full)  50 %
  mkd25 / img25   the same at 25 %

Reading (decided before the runs):
  KD gain   full - ce. If it is small (< 2 points), this dataset cannot tell the arms apart.
  main      img50 vs mkd50 and img25 vs mkd25. Differences under ~1 point with one seed are not a result.
            img >= mkd at both budgets  -> "spend teacher compute on fewer images, not fewer tokens"
            mkd > img at 25 % only      -> masking pays exactly where the no-training check said it would not:
                                           the curriculum / regularisation effect is real, look into it
            mkd > img at both           -> masking has a benefit beyond the information in the target
"""
import argparse
import json
import math
import os
import tarfile
import time
import urllib.request

import numpy as np
import timm
import torch
import torch.nn.functional as F

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
RUNS = {"ce": ("ce", 0.0), "full": ("full", 1.0), "mkd50": ("mask", 0.5), "img50": ("img", 0.5),
        "mkd25": ("mask", 0.25), "img25": ("img", 0.25)}
FAKE = bool(os.environ.get("QUICK_FAKE"))   # random images and weights, a few steps: only checks that the code runs

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--data", default="imagenette_cache")
ap.add_argument("--out", default="quick_train_out")
args = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.out, exist_ok=True)


# ---------------------------------------------------------------- data, decoded once and cached as uint8 tensors

def prepare():
    cache = os.path.join(args.data, "tensors.pt")
    if FAKE:
        g = torch.Generator().manual_seed(0)
        return (torch.randint(0, 256, (256, 3, 256, 256), dtype=torch.uint8, generator=g), torch.arange(256) % 10,
                torch.randint(0, 256, (64, 3, 224, 224), dtype=torch.uint8, generator=g), torch.arange(64) % 10,
                list(range(10)))
    if os.path.exists(cache):
        return torch.load(cache)
    from PIL import Image
    from timm.data import ImageNetInfo
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    os.makedirs(args.data, exist_ok=True)
    root = os.path.join(args.data, "imagenette2-320")
    if not os.path.isdir(root):
        tgz = os.path.join(args.data, "imagenette2-320.tgz")
        print("downloading Imagenette (330 MB)", flush=True)
        urllib.request.urlretrieve(URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(args.data)
    classes = sorted(os.listdir(os.path.join(root, "train")))
    wnids = ImageNetInfo().label_names()
    tidx = [wnids.index(c) for c in classes]

    class Folder(Dataset):
        def __init__(self, split, tf):
            self.items = [(os.path.join(root, split, c, f), i) for i, c in enumerate(classes)
                          for f in sorted(os.listdir(os.path.join(root, split, c)))]
            self.tf = tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            p, y = self.items[i]
            im = self.tf(Image.open(p).convert("RGB"))
            return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1), y

    def load(split, size):
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size)])
        xs, ys = [], []
        for x, y in DataLoader(Folder(split, tf), batch_size=256, num_workers=min(8, os.cpu_count() or 2)):
            xs.append(x)
            ys.append(y)
        return torch.cat(xs), torch.cat(ys)

    print("decoding images once (a few minutes)", flush=True)
    xtr, ytr = load("train", 256)    # train kept at 256x256: random resized crop to 224 happens on the GPU
    xva, yva = load("val", 224)      # val exactly as evaluated: resize 256, centre crop 224
    out = (xtr, ytr, xva, yva, tidx)
    torch.save(out, cache)
    return out


def augment(xb):
    """Random resized crop (scale 0.08-1, ratio 3/4-4/3) to 224 and horizontal flip, as one affine warp, normalised."""
    B = xb.shape[0]
    x = xb.float() / 255
    area = torch.empty(B, device=x.device).uniform_(0.08, 1.0)
    r = torch.exp(torch.empty(B, device=x.device).uniform_(math.log(3 / 4), math.log(4 / 3)))
    w, h = torch.sqrt(area * r).clamp(max=1), torch.sqrt(area / r).clamp(max=1)
    cx, cy = (torch.rand(B, device=x.device) * 2 - 1) * (1 - w), (torch.rand(B, device=x.device) * 2 - 1) * (1 - h)
    flip = torch.where(torch.rand(B, device=x.device) < 0.5, -1.0, 1.0)
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = w * flip, cx, h, cy
    grid = F.affine_grid(theta, (B, 3, 224, 224), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def normalise(xb):
    x = xb.float() / 255
    if x.shape[-1] != 224:
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


# ---------------------------------------------------------------- models

def fwd_tokens(model, img, keep_idx):
    """ViT on the kept patch tokens only (MaskedKD's masked teacher forward)."""
    x = model.patch_embed(img) + model.pos_embed[:, 1:]
    x = torch.gather(x, 1, keep_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    cls = (model.cls_token + model.pos_embed[:, :1]).expand(x.shape[0], -1, -1)
    return model.forward_head(model.norm(model.blocks(torch.cat([cls, x], 1))))


@torch.no_grad()
def evaluate(model, xva, yva, tidx=None, keep=None, student=None):
    """Top-1 accuracy (%). With keep: the teacher on the top-`keep` patches by `student`'s attention."""
    model.eval()
    hit = 0
    for i in range(0, len(yva), 250):
        x = normalise(xva[i:i + 250].to(dev))
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            if keep is None:
                z = model(x)
            else:
                student.eval()
                student(x)
                a = student._attn[:, :, 0, 1:].float().mean(1)
                z = fwd_tokens(model, x, a.topk(round(196 * keep), dim=1).indices)
        z = z.float() if tidx is None else z.float()[:, tidx]
        hit += (z.argmax(1).cpu() == yva[i:i + 250]).sum().item()
    return 100 * hit / len(yva)


def make_student():
    s = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=10, drop_path_rate=0.1).to(dev)
    last = s.blocks[-1].attn
    last.fused_attn = False                                   # explicit softmax, so the attention can be read
    last.attn_drop.register_forward_hook(lambda m, i, o: setattr(s, "_attn", o))
    return s


def train(name, teacher, tidx, data):
    kind, frac = RUNS[name]
    xtr, ytr, xva, yva = (t.to(dev) for t in data)
    torch.manual_seed(args.seed)
    student = make_student()
    decay = [p for n, p in student.named_parameters() if p.ndim > 1 and n not in ("pos_embed", "cls_token")]
    rest = [p for n, p in student.named_parameters() if not (p.ndim > 1 and n not in ("pos_embed", "cls_token"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": rest, "weight_decay": 0.0}],
                            lr=args.lr * args.bs / 512)
    steps_ep = len(ytr) // args.bs if not FAKE else 2
    total, warm = args.epochs * steps_ep, steps_ep * min(5, max(1, args.epochs // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    tidx_t = torch.tensor(tidx, device=dev)
    nkeep = round(196 * frac)
    tokens, seen, hist, t0 = 0.0, 0, [], time.time()
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    for ep in range(args.epochs):
        student.train()
        perm = torch.randperm(len(ytr), generator=gen).to(dev)
        loss_sum = 0.0
        for st in range(steps_ep):
            idx = perm[st * args.bs:(st + 1) * args.bs]
            x, y = augment(xtr[idx]), ytr[idx]
            B = x.shape[0]
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out = student(x)
            logits = out.float()
            loss = F.cross_entropy(logits, y)
            if kind != "ce":
                with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    if kind == "full":
                        rows, zt = None, teacher(x)
                        tokens += B * 197
                    elif kind == "mask":
                        a = student._attn[:, :, 0, 1:].float().mean(1).detach()
                        rows, zt = None, fwd_tokens(teacher, x, a.topk(nkeep, dim=1).indices)
                        tokens += B * (nkeep + 1)
                    else:   # img: a random share of the batch, all tokens
                        m = max(1, round(frac * B))
                        rows = torch.randperm(B, device=dev)[:m]
                        zt = teacher(x[rows])
                        tokens += m * 197
                pt = F.softmax(zt.float()[:, tidx_t], -1)
                ls = F.log_softmax(logits if rows is None else logits[rows], -1)
                loss = loss + (pt * (pt.clamp_min(1e-12).log() - ls)).sum(1).mean()
            seen += B
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item()
        rec = {"ep": ep + 1, "loss": round(loss_sum / steps_ep, 4), "min": round((time.time() - t0) / 60, 1)}
        if (ep + 1) % 10 == 0 or ep + 1 == args.epochs:
            rec["acc"] = round(evaluate(student, xva, yva), 2)
        hist.append(rec)
        print(f"[{name}] {json.dumps(rec)}", flush=True)
    res = {"run": name, "acc": hist[-1]["acc"], "teacher_cost": round(tokens / (max(seen, 1) * 197), 3),
           "minutes": hist[-1]["min"], "seed": args.seed, "epochs": args.epochs, "history": hist}
    with open(os.path.join(args.out, f"{name}_s{args.seed}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main():
    xtr, ytr, xva, yva, tidx = prepare()
    print(f"train {len(ytr)}, val {len(yva)}, device {dev}", flush=True)
    teacher = timm.create_model("deit_small_patch16_224", pretrained=not FAKE).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    t_acc = evaluate(teacher, xva, yva, tidx)
    print(f"teacher val acc (all tokens) {t_acc:.1f}", flush=True)
    if FAKE:
        args.epochs = 1
    results = {}
    for name in args.runs:
        done = os.path.join(args.out, f"{name}_s{args.seed}.json")
        if os.path.exists(done) and not FAKE:
            results[name] = json.load(open(done))
            print(f"[{name}] already done: {results[name]['acc']}", flush=True)
            continue
        results[name] = train(name, teacher, tidx, (xtr, ytr, xva, yva))
    print(f"\n=== seed {args.seed}, {args.epochs} epochs, teacher {t_acc:.1f}")
    print(f"{'run':8s} {'val acc':>8s} {'teacher cost':>13s} {'minutes':>8s}")
    for name in RUNS:
        if name in results:
            r = results[name]
            print(f"{name:8s} {r['acc']:8.2f} {r['teacher_cost']:13.3f} {r['minutes']:8.1f}")
    g = lambda n: results[n]["acc"] if n in results else None   # noqa: E731
    if g("full") is not None and g("ce") is not None:
        print(f"KD gain (full - ce): {g('full') - g('ce'):+.2f}")
    for b in ("50", "25"):
        if g("img" + b) is not None and g("mkd" + b) is not None:
            print(f"img{b} - mkd{b}: {g('img' + b) - g('mkd' + b):+.2f}")


if __name__ == "__main__":
    main()
