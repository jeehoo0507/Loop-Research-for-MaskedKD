"""Quick check, no training (Colab, GPU runtime, about 5-10 minutes on a T4).

    !pip -q install timm
    then paste this file into one cell and run it.

Question: do the assumptions behind "correct the masked teacher with a few full passes, placed by the student" hold?
Pretrained DeiT-S (teacher) and DeiT-Ti (student) from timm, 1000-way outputs, on the val images of Imagenette
(10 easy classes) and Imagewoof (10 dog breeds, hard). The student's last-block CLS attention keeps the top r of the
patches (MaskedKD); the teacher answers on those and on the full image. Per keep ratio r:

  B1 bias        KL(full || masked), share of images whose top class flips, teacher accuracy full vs masked
  B2 tail        share of the total damage |T_full - T_mask|^2 carried by the worst 10 % of the images
  B3 predictable Spearman with the damage and AUROC for a flip, of three student-side signals:
                   drop_attn  attention mass on the removed patches (free)
                   disagree   |T_mask - student|^2 (free)
                   resp       KL between the student's answers with and without the removed patches (one extra
                              small student forward)
                 persistence: correlation of the damage between the image and its mirror image
                 pay-off: noise of the correction when the full passes are placed by a signal, relative to placing
                 them uniformly (below 1 = better; "oracle" = placed by the true damage)
  B4 shared      share of KL(full || masked) one temperature removes (fitted on half, scored on the other half)
  B7 tokens or images, at the teacher cost of MaskedKD at r + 0.10:
                 what the masked answer explains of T_full beyond the label (a student that knows nothing yet) and
                 beyond label + student (a trained student), against what it must explain for "r tokens for all +
                 full teacher for 10 %" to add less noise than "full teacher for r + 10 % of the images, no masking"
Caveats: trained student and clean centre crops, so this is the late-training, easy-view end; the teacher may have
seen these images in training. It can rule the idea out cheaply; it cannot prove it.
"""
import os
import tarfile
import urllib.request

import numpy as np
import timm
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

FAKE = bool(os.environ.get("QUICK_FAKE"))          # random images and weights: only checks that the code runs
KEEPS = (0.50, 0.35, 0.25, 0.15)
P_FULL = 0.10                                       # share of images that get the full teacher in the corrected design
N_PER_SET = 64 if FAKE else 2000
BATCH = 16 if FAKE else 100
SETS = {"imagenette": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
        "imagewoof": "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz"}
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def load(name):
    """(images uint8-normalised float [N,3,224,224] on CPU, ImageNet-1k labels [N])."""
    if FAKE:
        return torch.randn(N_PER_SET, 3, 224, 224), torch.randint(0, 1000, (N_PER_SET,))
    from timm.data import ImageNetInfo
    from torchvision import datasets, transforms
    root = f"/content/{name}"
    if not os.path.exists(root):
        os.makedirs(root)
        tgz = root + ".tgz"
        urllib.request.urlretrieve(SETS[name], tgz)
        with tarfile.open(tgz) as t:
            t.extractall(root)
    val = next(os.path.join(dp, "val") for dp, dn, _ in os.walk(root) if "val" in dn)
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    ds = datasets.ImageFolder(val, tf)
    wnids = ImageNetInfo().label_names()            # index in the 1000-way output -> wnid
    to1k = [wnids.index(c) for c in ds.classes]
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:N_PER_SET].tolist()
    xs, ys = zip(*(ds[i] for i in idx))
    return torch.stack(xs), torch.tensor([to1k[y] for y in ys])


teacher = timm.create_model("deit_small_patch16_224", pretrained=not FAKE).to(dev).eval()
student = timm.create_model("deit_tiny_patch16_224", pretrained=not FAKE).to(dev).eval()
_attn = {}
student.blocks[-1].attn.fused_attn = False
student.blocks[-1].attn.attn_drop.register_forward_hook(lambda m, i, o: _attn.__setitem__("a", o))


def fwd_tokens(model, img, keep_idx):
    x = model.patch_embed(img) + model.pos_embed[:, 1:]
    x = torch.gather(x, 1, keep_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    cls = (model.cls_token + model.pos_embed[:, :1]).expand(x.shape[0], -1, -1)
    return model.forward_head(model.norm(model.blocks(torch.cat([cls, x], 1))))


def kl(p, logq):
    return (p * (p.clamp_min(1e-12).log() - logq)).sum(-1)


@torch.no_grad()
def measure(x):
    """Per image, for every keep ratio: logits of the teacher (full, masked) and the student (full, masked), and the
    removed attention mass."""
    out = {"zf": [], "zs": [], **{(k, r): [] for r in KEEPS for k in ("zm", "zsm", "drop")}}
    for i in range(0, len(x), BATCH):
        xb = x[i:i + BATCH].to(dev)
        zs = student(xb).float()
        a = _attn["a"][:, :, 0, 1:].float().mean(1)             # CLS -> patches, head mean (read before any masked pass)
        out["zf"].append(teacher(xb).float().cpu())
        out["zs"].append(zs.cpu())
        for r in KEEPS:
            keep = a.topk(round(196 * r), dim=1).indices
            out[("zm", r)].append(fwd_tokens(teacher, xb, keep).float().cpu())
            out[("zsm", r)].append(fwd_tokens(student, xb, keep).float().cpu())
            out[("drop", r)].append((1 - a.gather(1, keep).sum(1) / a.sum(1)).cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def best_mix(regs, t):
    """Least mean |t - sum_j w_j r_j|^2 over non-negative weights (every active set of at most 3 regressors)."""
    best = float((t * t).sum(1).mean())
    for mask in range(1, 2 ** len(regs)):
        A = [r for j, r in enumerate(regs) if mask >> j & 1]
        G = torch.stack([torch.stack([(a * b).sum(1).mean() for b in A]) for a in A]).double()
        c = torch.stack([(a * t).sum(1).mean() for a in A]).double()
        w = torch.linalg.solve(G + 1e-6 * G.diagonal().mean() * torch.eye(len(A), dtype=torch.float64), c)
        if (w >= 0).all():
            best = min(best, float((t * t).sum(1).mean() - 2 * (w * c).sum() + w @ G @ w))
    return max(best, 0.0)


def placed(score, d, p, floor=0.02, mix=0.5):
    """Noise of the correction, mean d_i (1/pi_i - 1), with pi_i ~ sqrt(score) mixed half and half with uniform."""
    s = score.clamp_min(1e-12).sqrt()
    pi = (mix * p + (1 - mix) * p * s / s.mean()).clamp(floor, 1.0)
    pi = (pi * p / pi.mean()).clamp(floor, 1.0)
    return float((d * (1 / pi - 1)).mean())


def report(name, x, y):
    m, m2 = measure(x), measure(x.flip(-1))
    tf, ps = m["zf"].softmax(1), m["zs"].softmax(1)
    onehot = F.one_hot(y, tf.shape[1]).float()
    half = torch.randperm(len(y), generator=torch.Generator().manual_seed(1))
    fit, held = half[: len(y) // 2], half[len(y) // 2:]
    print(f"\n=== {name}: {len(y)} images, teacher acc {100 * (m['zf'].argmax(1) == y).float().mean():.1f}, "
          f"student acc {100 * (m['zs'].argmax(1) == y).float().mean():.1f}")
    for r in KEEPS:
        zm = m[("zm", r)]
        tm = zm.softmax(1)
        d = ((tf - tm) ** 2).sum(1)
        flip = (zm.argmax(1) != m["zf"].argmax(1)).float()
        k = kl(tf, zm.log_softmax(1))
        top = d.sort(descending=True).values
        sig = {"drop_attn": m[("drop", r)], "disagree": ((tm - ps) ** 2).sum(1),
               "resp": kl(ps, m[("zsm", r)].log_softmax(1)).clamp_min(0)}
        d2 = ((m2["zf"].softmax(1) - m2[("zm", r)].softmax(1)) ** 2).sum(1)
        temps = np.exp(np.linspace(np.log(0.3), np.log(5), 41))
        t_best = min(temps, key=lambda t: float(kl(tf[fit], (zm[fit] / t).log_softmax(1)).mean()))
        tr = (zm / t_best).softmax(1)
        removed = 1 - float(kl(tf[held], (zm[held] / t_best).log_softmax(1)).mean()) / max(float(k[held].mean()), 1e-12)
        v_label, v_free = best_mix([onehot], tf), best_mix([ps, onehot], tf)
        g_early = 1 - best_mix([onehot, tr], tf) / max(v_label, 1e-12)
        g_late = 1 - best_mix([ps, onehot, tr], tf) / max(v_free, 1e-12)
        need = 1 - (1 / (r + P_FULL) - 1) / (1 / P_FULL - 1)
        uni = float((d * (1 / P_FULL - 1)).mean())
        print(f"\n-- keep {round(100 * r)} %  (cost with correction = MaskedKD at {round(100 * (r + P_FULL))} %)")
        print(f"B1 bias     KL {k.mean():.3f}   flips {100 * flip.mean():.1f} %   masked teacher acc "
              f"{100 * (zm.argmax(1) == y).float().mean():.1f}")
        print(f"B2 tail     worst 10 % of images carry {100 * top[: len(top) // 10].sum() / top.sum():.0f} % of the damage")
        for n, s in sig.items():
            auc = roc_auc_score(flip.numpy(), s.numpy()) if 0 < flip.sum() < len(flip) else float("nan")
            print(f"B3 {n:9s} spearman {spearmanr(s.numpy(), d.numpy()).correlation:+.2f}   AUROC(flip) {auc:.2f}   "
                  f"noise vs uniform {placed(s, d, P_FULL) / max(uni, 1e-12):.2f}")
        print(f"B3 oracle    noise vs uniform {placed(d, d, P_FULL) / max(uni, 1e-12):.2f}   "
              f"persistence (image vs mirror) spearman {spearmanr(d.numpy(), d2.numpy()).correlation:+.2f}")
        print(f"B4 shared   one temperature ({t_best:.2f}) removes {100 * removed:.0f} % of the KL")
        print(f"B7 masked answer explains beyond label {100 * g_early:.0f} %, beyond label + student "
              f"{100 * g_late:.0f} %; needs {100 * need:.0f} % to beat 'no masking, full teacher on "
              f"{round(100 * (r + P_FULL))} % of the images'")


for name in SETS:
    report(name, *load(name))
print("""
How to read it
  B1 holds if flips and KL grow clearly as fewer patches are kept (there is something to correct).
  B2 holds if the worst 10 % carry about half of the damage or more.
  B3 holds if a free signal reaches spearman > 0.3 and its 'noise vs uniform' is clearly below 1
     (the oracle line is the best any signal could do).
  B4 holds if one temperature removes a large share (then recalibrate before correcting).
  B7 holds only if 'explains' exceeds 'needs'. If it does not, even early (beyond label), masking does not pay.""")
