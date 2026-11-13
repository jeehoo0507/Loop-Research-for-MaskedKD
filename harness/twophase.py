"""Two-phase corrected distillation (human, 2026-11-13). Method "twophase"; the config is the job's sel_args.

MaskedKD trains on KL(T(x_S) || s(x)): the loss is smallest at the masked teacher's answer, not at the teacher's, so
its step is a biased estimate of the full-KD step. Here the same teacher compute is split differently: a cheap answer
h for every image, and the full teacher T(x) for a random m of the B images of the step. The KD target of image i is

    q_i = h_i + rho * (B / m) * xi_i * (T(x_i) - h_i),        xi_i = 1 for the m drawn images, else 0

and its KD loss is TAU^2 * (logsumexp(z_i) - <q_i, z_i>), z = student logits / TAU, with gradient softmax(z_i) - q_i
in z. Over the draw the mean of q_i is T(x_i) when rho = 1, whatever h is: the step is an unbiased estimate of the
full-KD step (double sampling, Neyman 1938; control variates; prediction-powered inference, Angelopoulos et al.
2023). h only sets how much noise the draw adds: (1/m - 1/B) times the spread of T - h over the images.

cv, the cheap answer h (it never gets a gradient):
  student   the student's own prediction, free. Only the m drawn images teach: full KD on m images, weighted B/m.
  label     the training label (soft, after Mixup/CutMix and smoothing), free.
  masked    the teacher on mask_keep tokens picked by `selector` (MaskedKD's own target; full_n = 0 is MaskedKD).
  recal     the masked answer after one map shared by all images, softmax(a * z + b) (a temperature, a bias per
            class), fitted online on the drawn images to bring it to the full answer.
  fit       a non-negative mix of student, label and (if mask_keep > 0) the recalibrated masked answer, with the
            weights that leave the least of T - h on the drawn images of earlier steps (least squares on an EMA).
  channel   no correction: the student is trained through the map the other way, KL(T(x_S) || softmax(a * z + b)),
            a, b fitted so that the full answer maps to the masked one (forward loss correction, Patrini et al.
            CVPR 2017). The drawn images only fit a and b. Biased if masking is not such a map.
The maps and the weights used at a step were fitted on earlier steps only, so they do not depend on this step's draw.

Budget: the job's keep is the mean number of teacher tokens per image, as for every method: mask_keep (every image)
+ full_n * 196 / B (the drawn images) <= keep. The measured teacher GFLOPs are in metrics.json as for every run.

Diagnostics, per epoch, in history[..]["tp"], over the drawn images (both teacher answers are known there):
mean |T - h|^2 for each cheap answer alone and for the best mix of the free ones ("free") and of all ("all"). The
share the masked teacher explains beyond what is free is 1 - res_all / res_free.
"""
import torch
import torch.nn.functional as F

from . import config as C
from .models import teacher_forward_keepmask

CVS = ("student", "label", "masked", "recal", "fit", "channel")
SELECTORS = ("maskedkd", "rollout", "random")   # harness builtins with a fixed K and no teacher cache
KEYS = {"cv", "mask_keep", "full_n", "selector", "rho"}
FIT_LR = 0.02     # Adam step of the maps
EMA = 0.99        # memory of the mix weights: about 100 steps
RIDGE = 1e-3


def parse(sel_args, budget, batch=None):
    """The checked config of a twophase job; ValueError says what is wrong. budget = the job's keep."""
    batch = int(batch or C.BATCH)
    a = dict(sel_args or {})
    if set(a) - KEYS:
        raise ValueError(f"twophase: unknown keys {sorted(set(a) - KEYS)}; allowed {sorted(KEYS)}")
    cv, k, m = a.get("cv"), a.get("mask_keep", 0), a.get("full_n", 0)
    if cv not in CVS:
        raise ValueError(f"twophase: cv must be one of {CVS}")
    for name, v, hi in (("mask_keep", k, C.NUM_PATCHES), ("full_n", m, batch)):
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= hi:
            raise ValueError(f"twophase: {name} must be an integer in [0, {hi}]")
    if "selector" in a and not k:
        raise ValueError("twophase: selector given without mask_keep")
    sel = a.get("selector", "maskedkd") if k else None
    if k and sel not in SELECTORS:
        raise ValueError(f"twophase: selector must be one of {SELECTORS}")
    if cv in ("masked", "recal", "channel") and not k:
        raise ValueError(f"twophase: cv '{cv}' needs mask_keep >= 1")
    if cv in ("student", "label") and k:
        raise ValueError(f"twophase: cv '{cv}' would pay for a masked pass it does not use")
    if cv in ("student", "recal", "fit", "channel") and m < 1:
        raise ValueError(f"twophase: cv '{cv}' needs full_n >= 1")
    if cv == "channel" and "rho" in a:
        raise ValueError("twophase: cv 'channel' has no correction, so no rho")
    rho = 0.0 if cv == "channel" else a.get("rho", 1.0)
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not 0.0 <= rho <= 1.0:
        raise ValueError("twophase: rho must be in [0, 1]")
    tokens = k + m * C.NUM_PATCHES / batch
    if tokens > float(budget) + 1e-6:
        raise ValueError(f"twophase: mask_keep {k} + full_n {m} x {C.NUM_PATCHES} / {batch} = {tokens:.2f} teacher "
                         f"tokens per image > keep {budget:g}")
    return {"cv": cv, "mask_keep": k, "full_n": m, "selector": sel, "rho": float(rho), "tokens_per_image": tokens}


def target(h, p_full, idx, rho):
    """q [B,C]: h, and for the drawn rows idx h + rho * (B / m) * (p_full - h)."""
    if idx is None or not len(idx) or rho == 0:
        return h
    q = h.clone()
    q[idx] = h[idx] + rho * (h.shape[0] / len(idx)) * (p_full - h[idx])
    return q


def kd_loss(u, q):
    """Per image logsumexp(u) - <q, u>: gradient softmax(u) - q. For q on the simplex it is the cross-entropy to q
    (the KL to q plus q's entropy, which has no gradient)."""
    return torch.logsumexp(u, dim=1) - (q * u).sum(1)


def nnls(G, cols):
    """Non-negative weights on the regressors `cols` that minimise |t - sum_j w_j r_j|^2, from the Gram matrix G of
    (r_0 .. r_{n-1}, t). Returns (w [n], the minimum). At most three regressors, so every active set is tried."""
    G = G.double().cpu()
    n = G.shape[0] - 1
    best_w, best = torch.zeros(n, dtype=torch.float64), float(G[n, n])
    for mask in range(1, 2 ** len(cols)):
        A = [c for i, c in enumerate(cols) if mask >> i & 1]
        M = G[A][:, A]
        w = torch.linalg.solve(M + RIDGE * M.diagonal().mean().clamp_min(1e-12) * torch.eye(len(A), dtype=torch.float64),
                               G[A, n])
        if (w < 0).any():
            continue
        val = float(G[n, n] - 2 * (w * G[A, n]).sum() + w @ M @ w)
        if val < best:
            best, best_w = val, torch.zeros(n, dtype=torch.float64)
            best_w[A] = w
    return best_w.float(), max(best, 0.0)


class AffineMap:
    """u -> a * u + b on logits, fitted by one Adam step per call on mean KL(target || softmax(a * u + b))."""

    def __init__(self, n_classes, device):
        self.p = [torch.ones((), device=device), torch.zeros(n_classes, device=device)]
        self.m = [torch.zeros_like(p) for p in self.p]
        self.v = [torch.zeros_like(p) for p in self.p]
        self.t = 0

    def __call__(self, u):
        return self.p[0] * u + self.p[1]

    def fit(self, u, target_p):
        d = self(u).softmax(1) - target_p
        self.t += 1
        for i, g in enumerate(((d * u).sum(1).mean(), d.mean(0))):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            self.p[i] = self.p[i] - FIT_LR * (self.m[i] / (1 - 0.9 ** self.t)) / ((self.v[i] / (1 - 0.999 ** self.t)).sqrt() + 1e-8)
        self.p[0] = self.p[0].clamp(0.05, 20.0)

    def state_dict(self):
        return {"p": [x.cpu() for x in self.p], "m": [x.cpu() for x in self.m], "v": [x.cpu() for x in self.v], "t": self.t}

    def load_state_dict(self, s, device):
        self.p, self.m, self.v = ([x.to(device) for x in s[k]] for k in ("p", "m", "v"))
        self.t = s["t"]


class TwoPhase:
    """The loss of one twophase run, with what it fits as it goes (the maps, the mix weights) and its diagnostics."""

    def __init__(self, cfg, n_classes, device):
        self.cfg = cfg
        self.rev = AffineMap(n_classes, device)   # masked logits -> the full answer (cv recal, fit)
        self.fwd = AffineMap(n_classes, device)   # full logits -> the masked answer (cv channel)
        self.n = 3 if cfg["mask_keep"] else 2     # regressors of the mix: student, label, recalibrated masked
        self.gram = torch.zeros(self.n + 1, self.n + 1, device=device)
        self.w = torch.zeros(self.n, device=device)
        self.w[1] = 1.0                           # before any pair is seen: the label
        self._acc = {}

    def _regs(self, ps, y, um):
        r = [ps, y] + ([self.rev(um).softmax(1)] if um is not None else [])
        return torch.stack(r, 1)                  # [B, n, C]

    def _cv(self, ps, y, um):
        cv = self.cfg["cv"]
        if cv == "student":
            return ps
        if cv == "label":
            return y
        if cv == "masked":
            return um.softmax(1)
        if cv == "recal":
            return self.rev(um).softmax(1)
        return (self.w.view(1, -1, 1) * self._regs(ps, y, um)).sum(1)

    def step(self, logits, y, x, teacher, keep, segs, idx=None):
        """logits [B,C] of the student (with their graph), y [B,C] soft labels, keep bool [B,196] or None (no
        masked pass). Returns (loss, mean teacher tokens per image). idx: the drawn rows, for tests."""
        cfg, B = self.cfg, x.shape[0]
        m = min(cfg["full_n"], B)
        um = uf = None
        with torch.no_grad(), segs.seg("teacher"):   # under the caller's autocast, like every teacher pass
            if keep is not None:
                um = teacher_forward_keepmask(teacher, x, keep).float() / C.TAU
            if m:
                if idx is None:
                    idx = torch.randperm(B, device=x.device)[:m]
                uf = teacher_forward_keepmask(teacher, x[idx], torch.ones(m, C.NUM_PATCHES, dtype=torch.bool,
                                                                         device=x.device)).float() / C.TAU
        with torch.autocast(x.device.type, enabled=False):
            u = logits.float() / C.TAU
            ps = u.detach().softmax(1)
            if cfg["cv"] == "channel":
                kd = F.kl_div(F.log_softmax(self.fwd(u), dim=1), um.softmax(1), reduction="none").sum(1)
            else:
                h = self._cv(ps, y.float(), um)
                kd = kd_loss(u, target(h, uf.softmax(1) if m else None, idx if m else None, cfg["rho"]))
            ce = torch.sum(-y * F.log_softmax(logits.float(), dim=-1), dim=-1)
            loss = ((1 - C.ALPHA) * ce + C.ALPHA * C.TAU * C.TAU * kd).mean()
            if m:
                with torch.no_grad():
                    self._observe(ps[idx], y.float()[idx], None if um is None else um[idx], uf)
        kept = (keep.sum(1).float().mean().item() if keep is not None else 0.0) + m * C.NUM_PATCHES / B
        return loss, kept

    def _observe(self, ps, y, um, uf):
        """The drawn images of this step: add to the epoch's diagnostics, then refit what later steps use."""
        pf = uf.softmax(1)
        acc = self._acc
        add = lambda k, v: acc.__setitem__(k, acc.get(k, 0.0) + float(v))   # noqa: E731
        add("n", pf.shape[0])
        add("ok_full", (uf.argmax(1) == y.argmax(1)).sum())
        R = torch.cat([self._regs(ps, y, um), pf.unsqueeze(1)], 1)          # [m, n+1, C]
        G = (R.unsqueeze(2) * R.unsqueeze(1)).sum((0, 3))                   # sums of <r_j, r_k> over the pairs
        acc["gram"] = acc.get("gram", 0.0) + G.double().cpu()
        if um is not None:
            pm = um.softmax(1)
            add("res_masked", ((pf - pm) ** 2).sum())
            add("kl_masked", (pf * (pf.clamp_min(1e-8).log() - F.log_softmax(um, dim=1))).sum())
            add("agree_masked", (um.argmax(1) == uf.argmax(1)).sum())
            add("ok_masked", (um.argmax(1) == y.argmax(1)).sum())
        self.gram = EMA * self.gram + (1 - EMA) * G / pf.shape[0]
        if self.cfg["cv"] == "fit":
            self.w = nnls(self.gram, list(range(self.n)))[0].to(self.gram.device)
        if um is not None:
            self.rev.fit(um, pf)
            self.fwd.fit(uf, pm)

    def pop_diag(self):
        """Means over this epoch's drawn images (None if there were none), and reset. res_x = mean |T - h|^2 with h
        = x alone; res_free / res_all = with the best non-negative mix of (student, label) / of all three, and
        w_free / w_all those mixes."""
        acc, self._acc = self._acc, {}
        if not acc:
            return None
        N = acc["n"]
        G = acc["gram"] / N
        n = self.n
        alone = lambda j: float(G[n, n] - 2 * G[j, n] + G[j, j])   # noqa: E731
        w_free, res_free = nnls(G, [0, 1])
        out = {"pairs": int(N), "ok_full": acc["ok_full"] / N, "res_student": alone(0), "res_label": alone(1),
               "res_free": res_free, "w_free": w_free[:2].tolist()}
        if n == 3:
            w_all, res_all = nnls(G, [0, 1, 2])
            out.update({"res_masked": acc["res_masked"] / N, "res_recal": alone(2), "res_all": res_all,
                        "w_all": w_all.tolist(), "kl_masked": acc["kl_masked"] / N,
                        "agree_masked": acc["agree_masked"] / N, "ok_masked": acc["ok_masked"] / N,
                        "map_a": float(self.rev.p[0]), "channel_a": float(self.fwd.p[0])})
        if self.cfg["cv"] == "fit":
            out["w_used"] = self.w.tolist()
        return {k: ([round(float(x), 5) for x in v] if isinstance(v, list) else round(v, 5) if isinstance(v, float) else v)
                for k, v in out.items()}

    def state_dict(self):
        return {"rev": self.rev.state_dict(), "fwd": self.fwd.state_dict(), "gram": self.gram.cpu(), "w": self.w.cpu()}

    def load_state_dict(self, s):
        dev = self.gram.device
        self.rev.load_state_dict(s["rev"], dev)
        self.fwd.load_state_dict(s["fwd"], dev)
        self.gram, self.w = s["gram"].to(dev), s["w"].to(dev)
