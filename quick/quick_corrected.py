"""Corrected MaskedKD vs MaskedKD vs fewer images, all at the teacher cost of MaskedKD 35 % (Colab or any GPU).

    !pip -q install timm
    !python quick_corrected.py                         # 7 runs, seed 1, 50 epochs
    !python quick_corrected.py --runs corr ours --epochs 30

Imagenette. Teacher DeiT-S (ImageNet-1k pretrained, frozen, its 10 Imagenette logits). Student DeiT-Ti from scratch.
Loss = CE(student, label) + KD, tau 1, weight 1. Same GPU augmentation for every run.

The method (the user's plan, completed after the no-training check):
  every image   teacher on the top 25 % patches by the student's last-block CLS attention (as MaskedKD)
  ~10 % images  the full teacher as well; the KD target there is corrected:
                    q = h + rho * (T_full - h) / pi          (h = cheap answer, pi = chance the image was drawn)
                which is unbiased for full KD when rho = 1 (prediction-powered / doubly robust estimators)
  placement     pi from a free student-side signal: |T_mask - student|^2 (spearman 0.7-0.8 with the real damage
                in the check), pi ~ sqrt(score) mixed half and half with uniform, floor 0.02, mean 10 %
  cheap answer  h = non-negative mix of (student prediction, label, masked teacher), weights fitted online by
                least squares on the drawn images of earlier steps (the check: student + label explain most of
                T_full, the masked answer alone does not)
  ramp          rho goes 0 -> 1 over the first 30 % of training: MaskedKD's easy early target, full KD later
  dropped       the temperature recalibration (removed 0 % in the check) and "direction-only" targets

Runs (teacher cost printed for each, all near 0.35 except ce and full):
  ce       no teacher                        full    teacher on every image, all tokens (upper reference)
  mkd35    MaskedKD, 35 % of the patches     img35   no masking, full teacher on a random 35 % of each batch
  corr     25 % masked + 10 % full, uniform placement, h = masked teacher, rho = 1     (the plan as first written)
  corr_s   corr + student placement
  ours     corr_s + fitted mix + ramp                                                    (the completed method)

Reading, fixed before the runs (one seed: a difference under ~1 point is not a result):
  corr - mkd35     does correcting the masked teacher help at all
  corr_s - corr    does the student know where to correct
  ours - corr_s    do the mix and the ramp help
  ours - img35     does masking + correction beat simply using fewer images (the no-training check said no)
  full - ce        KD gain on this data; under 2 points nothing above can be read
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
RUNS = {
    "ce": dict(kind="ce"),
    "full": dict(kind="full"),
    "mkd35": dict(kind="mask", keep=0.35),
    "img35": dict(kind="img", frac=0.35),
    "corr": dict(kind="corr", keep=0.25, p=0.10, place="uniform", cheap="masked", ramp=0.0),
    "corr_s": dict(kind="corr", keep=0.25, p=0.10, place="student", cheap="masked", ramp=0.0),
    "ours": dict(kind="corr", keep=0.25, p=0.10, place="student", cheap="mix", ramp=0.3),
    "mkd25": dict(kind="mask", keep=0.25),   # not in the default list: how bad the masked part alone is
}
DEFAULT = ["ce", "full", "mkd35", "img35", "corr", "corr_s", "ours"]
PI_FLOOR, PI_MIX, EMA = 0.02, 0.5, 0.99
FAKE = bool(os.environ.get("QUICK_FAKE"))   # random images and weights, a few steps: only checks that the code runs

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=DEFAULT, choices=list(RUNS))
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--data", default="imagenette_cache")
ap.add_argument("--out", default="quick_corrected_out")
args = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.out, exist_ok=True)
# bf16 where the GPU has it (A100, L4, G4, A5000); fp16 with loss scaling otherwise (T4, V100)
BF16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
AMP = dict(device_type=dev.type, dtype=torch.bfloat16 if BF16 else torch.float16, enabled=dev.type == "cuda")


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
    xtr, ytr = load("train", 256)
    xva, yva = load("val", 224)
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


# ---------------------------------------------------------------- models and the corrected target

def fwd_tokens(model, img, keep_idx):
    """ViT on the kept patch tokens only (MaskedKD's masked teacher forward)."""
    x = model.patch_embed(img) + model.pos_embed[:, 1:]
    x = torch.gather(x, 1, keep_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    cls = (model.cls_token + model.pos_embed[:, :1]).expand(x.shape[0], -1, -1)
    return model.forward_head(model.norm(model.blocks(torch.cat([cls, x], 1))))


@torch.no_grad()
def evaluate(model, xva, yva, tidx=None):
    model.eval()
    hit = 0
    for i in range(0, len(yva), 250):
        with torch.autocast(**AMP):
            z = model(normalise(xva[i:i + 250].to(dev))).float()
        z = z if tidx is None else z[:, tidx]
        hit += (z.argmax(1).cpu() == yva[i:i + 250].cpu()).sum().item()
    return 100 * hit / len(yva)


def make_student():
    s = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=10, drop_path_rate=0.1).to(dev)
    last = s.blocks[-1].attn
    last.fused_attn = False                                   # explicit softmax, so the attention can be read
    last.attn_drop.register_forward_hook(lambda m, i, o: setattr(s, "_attn", o))
    return s


def nnls(G):
    """Non-negative weights w on the 3 regressors minimising |t - sum w_j r_j|^2, from the Gram matrix of (r, t)."""
    G = G.double().cpu()
    n = G.shape[0] - 1
    best_w, best = torch.zeros(n, dtype=torch.float64), float(G[n, n])
    for mask in range(1, 2 ** n):
        A = [j for j in range(n) if mask >> j & 1]
        M = G[A][:, A]
        w = torch.linalg.solve(M + 1e-6 * M.diagonal().mean().clamp_min(1e-12) * torch.eye(len(A), dtype=torch.float64),
                               G[A, n])
        if (w < 0).any():
            continue
        val = float(G[n, n] - 2 * (w * G[A, n]).sum() + w @ M @ w)
        if val < best:
            best, best_w = val, torch.zeros(n, dtype=torch.float64)
            best_w[A] = w
    return best_w.float()


def place(score, pbar):
    """Draw probabilities with mean pbar: sqrt(score) mixed half and half with uniform, floor PI_FLOOR."""
    s = score.clamp_min(0).sqrt()
    pi = pbar * (PI_MIX + (1 - PI_MIX) * s / s.mean().clamp_min(1e-12))
    for _ in range(4):
        pi = pi.clamp(PI_FLOOR, 1.0)
        pi = pi * pbar / pi.mean()
    return pi.clamp(PI_FLOOR, 1.0)


def train(name, teacher, tidx, data):
    cfg = RUNS[name]
    kind = cfg["kind"]
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
    nkeep = round(196 * cfg.get("keep", 0))
    gram, w = None, torch.tensor([0.0, 0.0, 1.0], device=dev)   # mix weights (student, label, masked); start = masked
    tokens, seen, hist, t0, step = 0.0, 0, [], time.time(), 0
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda" and not BF16)
    for ep in range(args.epochs):
        student.train()
        perm = torch.randperm(len(ytr), generator=gen).to(dev)
        loss_sum, diag = 0.0, {"drawn": 0.0, "res_h": 0.0, "res_mask": 0.0, "pairs": 0}
        for st in range(steps_ep):
            idx = perm[st * args.bs:(st + 1) * args.bs]
            x, y = augment(xtr[idx]), ytr[idx]
            B = x.shape[0]
            with torch.autocast(**AMP):
                logits = student(x).float()
            loss = F.cross_entropy(logits, y)
            if kind in ("full", "mask", "img"):
                with torch.no_grad(), torch.autocast(**AMP):
                    if kind == "full":
                        rows, zt = None, teacher(x)
                        tokens += B * 197
                    elif kind == "mask":
                        a = student._attn[:, :, 0, 1:].float().mean(1)
                        rows, zt = None, fwd_tokens(teacher, x, a.topk(nkeep, dim=1).indices)
                        tokens += B * (nkeep + 1)
                    else:
                        m = max(1, round(cfg["frac"] * B))
                        rows = torch.randperm(B, device=dev)[:m]
                        zt = teacher(x[rows])
                        tokens += m * 197
                pt = F.softmax(zt.float()[:, tidx_t], -1)
                ls = F.log_softmax(logits if rows is None else logits[rows], -1)
                loss = loss + (pt * (pt.clamp_min(1e-12).log() - ls)).sum(1).mean()
            elif kind == "corr":
                with torch.no_grad():
                    a = student._attn[:, :, 0, 1:].float().mean(1)
                    with torch.autocast(**AMP):
                        zm = fwd_tokens(teacher, x, a.topk(nkeep, dim=1).indices)
                    tokens += B * (nkeep + 1)
                    qm = F.softmax(zm.float()[:, tidx_t], -1)
                    ps = F.softmax(logits.detach(), -1)
                    yh = F.one_hot(y, 10).float()
                    h = qm if cfg["cheap"] == "masked" else w[0] * ps + w[1] * yh + w[2] * qm
                    if cfg["place"] == "uniform":
                        pi = torch.full((B,), cfg["p"], device=dev)
                    else:
                        pi = place(((qm - ps) ** 2).sum(1), cfg["p"])
                    b = torch.bernoulli(pi).bool()
                    prog = step / max(1, total)
                    rho = min(1.0, prog / cfg["ramp"]) if cfg["ramp"] > 0 else 1.0
                    q = h.clone()
                    if b.any():
                        with torch.autocast(**AMP):
                            zf = teacher(x[b])
                        tokens += int(b.sum()) * 197
                        pf = F.softmax(zf.float()[:, tidx_t], -1)
                        q[b] = h[b] + rho * (pf - h[b]) / pi[b].unsqueeze(1)
                        R = torch.stack([ps[b], yh[b], qm[b], pf], 1)                 # [m, 4, C]
                        G = torch.einsum("mic,mjc->ij", R, R) / R.shape[0]
                        gram = G if gram is None else EMA * gram + (1 - EMA) * G
                        if cfg["cheap"] == "mix":
                            w = nnls(gram).to(dev)                                    # used from the next step on
                        diag["drawn"] += float(b.float().mean())
                        diag["res_h"] += float(((pf - h[b]) ** 2).sum(1).sum())
                        diag["res_mask"] += float(((pf - qm[b]) ** 2).sum(1).sum())
                        diag["pairs"] += int(b.sum())
                # gradient in the logits is softmax - q, unbiased for full KD when rho = 1 (q may leave the simplex)
                loss = loss + (torch.logsumexp(logits, 1) - (q * logits).sum(1)).mean()
            seen += B
            step += 1
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item()
        rec = {"ep": ep + 1, "loss": round(loss_sum / steps_ep, 4), "min": round((time.time() - t0) / 60, 1)}
        if kind == "corr" and diag["pairs"]:
            rec.update(drawn=round(diag["drawn"] / steps_ep, 3), res_h=round(diag["res_h"] / diag["pairs"], 4),
                       res_mask=round(diag["res_mask"] / diag["pairs"], 4), w=[round(float(v), 2) for v in w])
        if (ep + 1) % 10 == 0 or ep + 1 == args.epochs:
            rec["acc"] = round(evaluate(student, xva, yva), 2)
        hist.append(rec)
        print(f"[{name}] {json.dumps(rec)}", flush=True)
    res = {"run": name, "cfg": cfg, "acc": hist[-1]["acc"], "teacher_cost": round(tokens / (max(seen, 1) * 197), 3),
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
    for a, b, what in (("full", "ce", "KD gain"), ("corr", "mkd35", "correction helps"),
                       ("corr_s", "corr", "student placement helps"), ("ours", "corr_s", "mix + ramp help"),
                       ("ours", "img35", "masking + correction beats fewer images"), ("ours", "mkd35", "ours vs MaskedKD")):
        if g(a) is not None and g(b) is not None:
            print(f"{a + ' - ' + b:16s} {g(a) - g(b):+6.2f}   ({what})")


if __name__ == "__main__":
    main()
