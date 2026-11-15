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

URLS = {d: f"https://s3.amazonaws.com/fast-ai-imageclas/{d}2-320.tgz" for d in ("imagenette", "imagewoof")}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
RUNS = {
    "ce": dict(kind="ce"),
    "full": dict(kind="full"),
    "mkd35": dict(kind="mask", keep=0.35),
    "img35": dict(kind="img", frac=0.35),
    "corr": dict(kind="corr", keep=0.25, p=0.10, place="uniform", cheap="masked", ramp=0.0),
    "corr_s": dict(kind="corr", keep=0.25, p=0.10, place="student", cheap="masked", ramp=0.0),
    "ours": dict(kind="corr", keep=0.25, p=0.10, place="student", cheap=("s", "y", "m"), ramp=0.3),
    "mkd25": dict(kind="mask", keep=0.25),   # not in the default list: how bad the masked part alone is
    # the plan as first proposed, parts that had not been run (2026-11-16):
    #   dir     the correction target is the full teacher's ranking at the masked teacher's entropy ("direction
    #           only"): where the two agree on the ranking the correction is ~0, so it adds far less noise
    #   dir_s   dir + student placement
    #   repl_s  the student picks the 10 % it disagrees with the masked teacher most; those images get the full
    #           answer as it is (no 1/pi weight): biased, but no noise blow-up
    #   ours_h  ours with a learned cheap answer: softmax(log T_mask + head(T_mask, student, label)), the head
    #           (zero at start = MaskedKD) trained on the drawn pairs to predict the full answer
    "dir": dict(kind="corr", keep=0.25, p=0.10, place="uniform", cheap="masked", ramp=0.0, target="dir"),
    "dir_s": dict(kind="corr", keep=0.25, p=0.10, place="student", cheap="masked", ramp=0.0, target="dir"),
    "repl_s": dict(kind="corr", keep=0.25, p=0.10, place="topk", cheap="masked", ramp=0.0),
    "ours_h": dict(kind="corr", keep=0.25, p=0.10, place="student", cheap="head", ramp=0.3),
    # --- three methods built on MaskedKD (2026-11-16). cheap = tuple of regressors of the fitted mix:
    #     s student, y label, m masked teacher (this view), c cache of this image's last full-teacher answer
    # A. same view: the KD term compares teacher and student on the SAME kept tokens (the target is then exactly
    #    the teacher's answer for the student's input: no mismatch, no extra teacher compute); CE on the full image
    "mkd50": dict(kind="mask", keep=0.50),   # MaskedKD at about the step cost of sv25 (the student's extra view counted)
    "sv25": dict(kind="sv", keep=0.25),
    "sv35": dict(kind="sv", keep=0.35),
    "sv15": dict(kind="sv", keep=0.15),
    "sv25_rand": dict(kind="sv", keep=0.25, select="random"),      # does student-guided selection still matter
    "sv25_both": dict(kind="sv", keep=0.25, both=True),            # + MaskedKD's own full-view term, half each
    # B. space and time: the teacher's answer is redundant over tokens (masking) and over epochs (the same image
    #    comes back). The cache is a second cheap answer; a few fresh full passes keep both honest.
    "ours_c": dict(kind="corr", keep=0.25, p=0.10, place="student", cheap=("s", "y", "m", "c"), ramp=0.3),
    "img_c": dict(kind="corr", keep=0.0, p=0.35, place="uniform", cheap=("s", "y", "c"), ramp=0.3),      # time only
    "img_c10": dict(kind="corr", keep=0.0, p=0.10, place="uniform", cheap=("s", "y", "c"), ramp=0.3),    # 10 % cost
    "cache10": dict(kind="corr", keep=0.0, p=0.10, place="uniform", cheap=("c",), fit=False, ipw=False, ramp=0.0),
    "img10": dict(kind="img", frac=0.10),                          # 10 % cost without a cache
    # C. a head for masked inputs: the frozen teacher's own classifier was fitted to full-image features; a linear
    #    head on its MASKED feature, trained online on the ~10 % paired full passes, predicts the full answer
    "mkd_f": dict(kind="corr", keep=0.25, p=0.10, place="uniform", cheap="feat", ipw=False, ramp=0.0),
    "ours_f": dict(kind="corr", keep=0.25, p=0.10, place="student", cheap="feat", ramp=0.3),
}
DEFAULT = ["ce", "full", "mkd35", "img35", "corr", "corr_s", "ours"]
PI_FLOOR, PI_MIX, EMA = 0.02, 0.5, 0.99
FAKE = bool(os.environ.get("QUICK_FAKE"))   # random images and weights, a few steps: only checks that the code runs

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=DEFAULT, choices=list(RUNS))
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--seeds", type=int, nargs="+", default=[1])
ap.add_argument("--dataset", default="imagenette", choices=list(URLS),
                help="imagewoof: 10 dog breeds, the teacher is less sure, so its soft answer carries more")
ap.add_argument("--tau", type=float, default=1.0, help="KD temperature (MaskedKD: 1; classic KD: 4)")
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--data", default=None, help="default: <dataset>_cache")
ap.add_argument("--out", default="quick_corrected_out")
ap.add_argument("--summary", action="store_true", help="only print the table from the result files that exist")
args = ap.parse_args()
args.data = args.data or f"{args.dataset}_cache"
TAG = f"{args.dataset}_t{args.tau:g}" + ("" if args.epochs == 50 else f"_e{args.epochs}")
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
    root = os.path.join(args.data, f"{args.dataset}2-320")
    if not os.path.isdir(root):
        tgz = os.path.join(args.data, f"{args.dataset}2-320.tgz")
        print(f"downloading {args.dataset} (about 330 MB)", flush=True)
        urllib.request.urlretrieve(URLS[args.dataset], tgz)
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

def fwd_tokens(model, img, keep_idx, want_feat=False):
    """ViT on the kept patch tokens only (MaskedKD's masked teacher forward). want_feat: also the pre-head feature."""
    x = model.patch_embed(img) + model.pos_embed[:, 1:]
    x = torch.gather(x, 1, keep_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    cls = (model.cls_token + model.pos_embed[:, :1]).expand(x.shape[0], -1, -1)
    f = model.forward_head(model.norm(model.blocks(torch.cat([cls, x], 1))), pre_logits=True)
    return (model.head(f), f) if want_feat else model.head(f)


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


def ent(p):
    return -(p * p.clamp_min(1e-12).log()).sum(-1)


def match_entropy(u, h_target, iters=30):
    """softmax(u / t), t per row such that its entropy equals h_target (bisection on log t): the ranking of u with
    the confidence of another distribution."""
    lo, hi = torch.full_like(h_target, math.log(0.05)), torch.full_like(h_target, math.log(20.0))
    for _ in range(iters):
        mid = (lo + hi) / 2
        low = ent(F.softmax(u / mid.exp().unsqueeze(-1), -1)) < h_target
        lo, hi = torch.where(low, mid, lo), torch.where(low, hi, mid)
    return F.softmax(u / ((lo + hi) / 2).exp().unsqueeze(-1), -1)


def place(score, pbar):
    """Draw probabilities with mean pbar: sqrt(score) mixed half and half with uniform, floor PI_FLOOR."""
    s = score.clamp_min(0).sqrt()
    pi = pbar * (PI_MIX + (1 - PI_MIX) * s / s.mean().clamp_min(1e-12))
    for _ in range(4):
        pi = pi.clamp(PI_FLOOR, 1.0)
        pi = pi * pbar / pi.mean()
    return pi.clamp(PI_FLOOR, 1.0)


def train(name, teacher, tidx, data, seed):
    cfg = RUNS[name]
    kind = cfg["kind"]
    xtr, ytr, xva, yva = (t.to(dev) for t in data)
    torch.manual_seed(seed)
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
    cheap = cfg.get("cheap")
    gram, w = None, None
    if isinstance(cheap, tuple):   # start: the masked answer if there is one, else the label, else the cache
        w = torch.zeros(len(cheap), device=dev)
        w[cheap.index(next(n for n in ("m", "y", "c") if n in cheap))] = 1.0
    if cheap == "feat":            # linear on (masked teacher feature, log T_mask, log student, label)
        head = torch.nn.Linear(teacher.num_features + 30, 10).to(dev)
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    else:
        head = torch.nn.Sequential(torch.nn.Linear(30, 64), torch.nn.GELU(), torch.nn.Linear(64, 10)).to(dev)
        torch.nn.init.zeros_(head[2].weight)
        torch.nn.init.zeros_(head[2].bias)                     # zero correction at the start: h = masked answer
    head_opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    use_cache = isinstance(cheap, tuple) and "c" in cheap
    cache = torch.zeros(len(ytr), 10, device=dev)               # last full-teacher answer of each training image
    has = torch.zeros(len(ytr), dtype=torch.bool, device=dev)
    s_extra = 0.0
    tokens, seen, hist, t0, step = 0.0, 0, [], time.time(), 0
    gen = torch.Generator(device="cpu").manual_seed(seed)
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
                pt = F.softmax(zt.float()[:, tidx_t] / args.tau, -1)
                ls = F.log_softmax((logits if rows is None else logits[rows]) / args.tau, -1)
                loss = loss + args.tau ** 2 * (pt * (pt.clamp_min(1e-12).log() - ls)).sum(1).mean()
            elif kind == "corr":
                pair = None
                with torch.no_grad():
                    ps = F.softmax(logits.detach() / args.tau, -1)
                    yh = F.one_hot(y, 10).float()
                    regs, qm = {"s": ps, "y": yh}, None
                    if nkeep:
                        a = student._attn[:, :, 0, 1:].float().mean(1)
                        with torch.autocast(**AMP):
                            zm, fm = fwd_tokens(teacher, x, a.topk(nkeep, dim=1).indices, want_feat=True)
                        tokens += B * (nkeep + 1)
                        qm = F.softmax(zm.float()[:, tidx_t] / args.tau, -1)
                        regs["m"] = qm
                    if use_cache:
                        regs["c"] = torch.where(has[idx].unsqueeze(1), cache[idx], yh)   # no answer yet: the label
                    if cheap == "masked":
                        h = qm
                    elif isinstance(cheap, tuple):
                        h = sum(w[k] * regs[n] for k, n in enumerate(cheap))
                    else:
                        feats = torch.cat(([fm.float()] if cheap == "feat" else [])
                                          + [qm.clamp_min(1e-6).log(), ps.clamp_min(1e-6).log(), yh], 1)
                        h = F.softmax(feats[:, -30:-20] + head(feats), -1)
                    score = ((regs.get("m", regs.get("c", yh)) - ps) ** 2).sum(1)
                    if cfg["place"] == "topk":   # the images the signal ranks highest, taken as they are
                        b = torch.zeros(B, dtype=torch.bool, device=dev)
                        b[score.topk(max(1, round(cfg["p"] * B))).indices] = True
                        pi = torch.ones(B, device=dev)
                    else:
                        pi = torch.full((B,), cfg["p"], device=dev) if cfg["place"] == "uniform" else place(score, cfg["p"])
                        b = torch.bernoulli(pi).bool()
                        if not cfg.get("ipw", True):   # the drawn images get the full answer as it is
                            pi = torch.ones(B, device=dev)
                    prog = step / max(1, total)
                    rho = min(1.0, prog / cfg["ramp"]) if cfg["ramp"] > 0 else 1.0
                    q = h.clone()
                    if b.any():
                        with torch.autocast(**AMP):
                            zf = teacher(x[b])
                        tokens += int(b.sum()) * 197
                        uf = zf.float()[:, tidx_t] / args.tau
                        pf = F.softmax(uf, -1)
                        tgt = match_entropy(uf, ent(qm[b])) if cfg.get("target") == "dir" else pf
                        q[b] = h[b] + rho * (tgt - h[b]) / pi[b].unsqueeze(1)
                        if isinstance(cheap, tuple) and cfg.get("fit", True):
                            R = torch.stack([regs[n][b] for n in cheap] + [pf], 1)   # [m, n+1, C]
                            G = torch.einsum("mic,mjc->ij", R, R) / R.shape[0]
                            gram = G if gram is None else EMA * gram + (1 - EMA) * G
                            w = nnls(gram).to(dev)                                    # used from the next step on
                        if use_cache:
                            cache[idx[b]] = pf
                            has[idx[b]] = True
                        if cheap in ("head", "feat"):
                            pair = (feats[b], pf)
                        diag["drawn"] += float(b.float().mean())
                        diag["res_h"] += float(((pf - h[b]) ** 2).sum(1).sum())
                        diag["res_mask"] += float(((pf - (qm if qm is not None else yh)[b]) ** 2).sum(1).sum())
                        diag["pairs"] += int(b.sum())
                if pair is not None:   # after this step's target was built
                    fb, pfb = pair
                    hl = -(pfb * F.log_softmax(fb[:, -30:-20] + head(fb), -1)).sum(1).mean()
                    head_opt.zero_grad(set_to_none=True)
                    hl.backward()
                    head_opt.step()
                # gradient in the logits is softmax - q, unbiased for full KD when rho = 1 (q may leave the simplex)
                u = logits / args.tau
                loss = loss + args.tau ** 2 * (torch.logsumexp(u, 1) - (q * u).sum(1)).mean()
            elif kind == "sv":
                with torch.no_grad():
                    if cfg.get("select") == "random":
                        keep_idx = torch.rand(B, 196, device=dev).topk(nkeep, dim=1).indices
                    else:
                        keep_idx = student._attn[:, :, 0, 1:].float().mean(1).topk(nkeep, dim=1).indices
                    with torch.autocast(**AMP):
                        zt = fwd_tokens(teacher, x, keep_idx)
                    tokens += B * (nkeep + 1)
                    pt = F.softmax(zt.float()[:, tidx_t] / args.tau, -1)
                with torch.autocast(**AMP):
                    zs = fwd_tokens(student, x, keep_idx).float()      # the student on the same tokens, with gradient
                s_extra += B * (nkeep + 1)
                kd = (pt * (pt.clamp_min(1e-12).log() - F.log_softmax(zs / args.tau, -1))).sum(1).mean()
                if cfg.get("both"):   # MaskedKD's term on the full view as well, half each
                    kd = 0.5 * kd + 0.5 * (pt * (pt.clamp_min(1e-12).log() - F.log_softmax(logits / args.tau, -1))).sum(1).mean()
                loss = loss + args.tau ** 2 * kd
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
                       res_mask=round(diag["res_mask"] / diag["pairs"], 4))
            if w is not None:
                rec["w"] = dict(zip(cheap, [round(float(v), 2) for v in w]))
        if (ep + 1) % 10 == 0 or ep + 1 == args.epochs:
            rec["acc"] = round(evaluate(student, xva, yva), 2)
        hist.append(rec)
        print(f"[{name}] {json.dumps(rec)}", flush=True)
    t_cost, s_cost = tokens / (max(seen, 1) * 197), s_extra / (max(seen, 1) * 197)
    # one training step per image, estimated: student forward + backward (3 x 1.3 GFLOPs, DeiT-Ti) on the full image
    # and on any extra masked view, plus the teacher's forwards (4.6 GFLOPs, DeiT-S, linear in the tokens)
    res = {"run": name, "cfg": cfg, "acc": hist[-1]["acc"], "teacher_cost": round(t_cost, 3),
           "student_extra": round(s_cost, 3), "step_gflops": round(3.9 * (1 + s_cost) + 4.6 * t_cost, 2),
           "minutes": hist[-1]["min"], "seed": seed, "dataset": args.dataset, "tau": args.tau, "epochs": args.epochs,
           "history": hist}
    with open(os.path.join(args.out, f"{name}_{TAG}_s{seed}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main():
    if args.summary:
        return summarise({s: {n: json.load(open(f)) for n in args.runs
                              if os.path.exists(f := os.path.join(args.out, f"{n}_{TAG}_s{s}.json"))}
                          for s in args.seeds}, float("nan"))
    xtr, ytr, xva, yva, tidx = prepare()
    print(f"train {len(ytr)}, val {len(yva)}, device {dev}", flush=True)
    teacher = timm.create_model("deit_small_patch16_224", pretrained=not FAKE).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    t_acc = evaluate(teacher, xva, yva, tidx)
    print(f"teacher val acc (all tokens) {t_acc:.1f}", flush=True)
    if FAKE:
        args.epochs = 1
    results = {}   # seed -> name -> result
    for seed in args.seeds:
        results[seed] = {}
        for name in args.runs:
            done = os.path.join(args.out, f"{name}_{TAG}_s{seed}.json")
            if os.path.exists(done) and not FAKE:
                results[seed][name] = json.load(open(done))
                print(f"[{name} s{seed}] already done: {results[seed][name]['acc']}", flush=True)
                continue
            print(f"--- {name}, seed {seed}", flush=True)
            results[seed][name] = train(name, teacher, tidx, (xtr, ytr, xva, yva), seed)
    summarise(results, t_acc)


def summarise(results, t_acc):
    seeds = args.seeds
    sd = lambda v: (sum((a - sum(v) / len(v)) ** 2 for a in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0   # noqa: E731
    print(f"\n=== {args.dataset}, tau {args.tau:g}, seeds {seeds}, {args.epochs} epochs, teacher {t_acc:.1f}")
    print(f"{'run':10s} {'val acc (mean +- sd)':>22s} {'per seed':>24s} {'teacher cost':>13s} {'step GFLOPs':>12s}")
    for name in RUNS:
        accs = [results[s][name]["acc"] for s in seeds if name in results[s]]
        if accs:
            r = results[next(s for s in seeds if name in results[s])][name]
            print(f"{name:10s} {sum(accs) / len(accs):14.2f} +- {sd(accs):4.2f} {str(accs):>24s} "
                  f"{r['teacher_cost']:13.3f} {r.get('step_gflops', float('nan')):12.2f}")
    print("\npaired differences (same seed), mean and per seed:")
    for a, b, what in (("full", "ce", "KD gain: under 2 points nothing below can be read"),
                       ("corr", "mkd35", "correction helps"), ("corr_s", "corr", "student placement helps"),
                       ("ours", "corr_s", "mix + ramp help"), ("ours", "img35", "masking + correction beats fewer images"),
                       ("img35", "mkd35", "fewer images vs fewer tokens"), ("ours", "mkd35", "ours vs MaskedKD"),
                       ("mkd35", "mkd25", "what 10 % more tokens buy MaskedKD"),
                       ("dir", "mkd35", "direction-only correction vs MaskedKD"), ("dir_s", "dir", "student placement, direction-only"),
                       ("repl_s", "mkd35", "student-routed full answers vs MaskedKD"), ("repl_s", "mkd25", "what the 10 % routed full passes add"),
                       ("ours_h", "ours", "learned cheap answer vs fitted mix"), ("ours_h", "mkd35", "ours_h vs MaskedKD"),
                       ("ours_h", "img35", "ours_h vs fewer images"),
                       ("sv25", "mkd25", "A same view vs MaskedKD, same teacher tokens"), ("sv25", "mkd35", "A vs MaskedKD, similar step cost"),
                       ("sv25", "full", "A vs full KD"), ("sv25_rand", "sv25", "A: random instead of student-chosen tokens"),
                       ("sv25_both", "sv25", "A: adding MaskedKD's full-view term"), ("sv15", "sv25", "A: fewer tokens"),
                       ("ours_c", "ours", "B: what the cache adds to masked + correction"), ("ours_c", "mkd35", "B space + time vs MaskedKD"),
                       ("img_c", "img35", "B: what the cache adds without masking"), ("ours_c", "img_c", "B: masking + cache vs cache alone"),
                       ("img_c10", "img10", "B at 10 % cost: cache + correction vs nothing"), ("img_c10", "cache10", "B at 10 %: correction vs stale cache"),
                       ("img_c10", "full", "B at 10 % cost vs full KD"),
                       ("mkd_f", "mkd35", "C masked-input head vs MaskedKD"), ("mkd_f", "mkd25", "C: what the head adds to the same tokens"),
                       ("ours_f", "mkd_f", "C: + unbiased correction")):
        d = [results[s][a]["acc"] - results[s][b]["acc"] for s in seeds if a in results[s] and b in results[s]]
        if d:
            print(f"{a + ' - ' + b:16s} {sum(d) / len(d):+6.2f}  {[round(x, 2) for x in d]}   ({what})")


if __name__ == "__main__":
    main()
