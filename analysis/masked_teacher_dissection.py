"""Dissection of the masked-teacher signal (human, 2026-11-13). No training.

MaskedKD trains the student on the full image x against the teacher's answer on the kept tokens x_S:
    loss_KD = KL(T(x_S) || s(x)).
A student that already equals the teacher is penalised by this loss (T(x_S) != T(x)): the loss is not consistent. This
script measures how large that inconsistency is, what it is made of, and which correction could remove it, on this
dataset's clean val images, for trained students of this dataset, token rules R in RULES and K in K_LIST:

  size         KL(T(x) || T(x_S)), accuracy of T(x_S), and the share of images for which T(x_S) is farther from T(x)
               than the one-hot label is (squared distance).
  systematic   how much of the error one map shared by all images removes: z -> a*z + b (one temperature a, one bias
               per class b), fitted on half the images to bring T(x_S) to T(x), scored on the other half (share of the
               KL removed). Also the mixing weight w in T(x_S) ~ (1-w) T(x) + w * (teacher's mean output): how far the
               partial-evidence answer falls back to the teacher's prior.
  bias vs noise in the gradient of the last layer (the student's logits z and its CLS feature h; the KD gradient of
               the classifier is (s - t) (x) h, linear in the target t), per image i:
                 bias^2     = | mean_i (T(x_S)_i - T(x)_i) (x) h_i |^2        what MaskedKD adds to every step
                 noise      = Var_i[(s_i - T(x)_i) (x) h_i] / BATCH            what a minibatch of full KD has anyway
                 V_res      = Var_i[(T(x)_i - lam * T(x_S)_i) (x) h_i]         residual if T(x_S) is a control variate
               An unbiased two-phase estimator (cheap answer for every image, full teacher for a random m of BATCH,
               difference added back with weight BATCH/m; survey statistics' double sampling, prediction-powered
               inference) has variance (1/m - 1/BATCH) * V_res(lam*) and no bias. rho^2 = share of Var[T(x) (x) h]
               the cheap answer explains. Reported per (rule, K), raw and after the systematic map, and as an
               equal-teacher-cost table against MaskedKD at K_REF.
               The label is a control variate too, and it is free. So the masked teacher has to explain what the
               label leaves unexplained: gain = (R2[label + masked] - R2[label]) / (1 - R2[label]). Multifidelity
               Monte Carlo (Peherstorfer, Willcox & Gunzburger 2016) says a cheap model of relative cost w pays for
               itself only if that share exceeds 4w / (1 + w)^2 (necessary; it assumes the best split of the budget).
               The student's own answer is free as well, and it is what plain subsampling leans on: full KD on m of
               the B images, weighted B/m, has V = Var[(T - s) (x) h]. So the two designs trained in corrected/ are
                 images only      V_free = least of Var[(T - mix) (x) h] over non-negative mixes of (student, label)
                 tokens + images  V_all  = the same with the recalibrated masked answer added to the mix
               and gain_beyond_free = 1 - V_all / V_free. "design" compares them at the splits corrected/arms.py
               trains (same teacher GFLOPs as MaskedKD at that K): (1/m - 1/BATCH) * V for each. If images only
               wins, the better use of the same teacher compute is NOT to mask.
  channel      the opposite direction of the systematic map: one channel C(p) = softmax(a * log p + b) fitted so that
               C(T(x)) ~ T(x_S). If masking acts on the teacher's answer like such a channel, training the student
               THROUGH it, KL(T(x_S) || C(s(x))), is minimised by s = T (forward loss correction, from the noisy-label
               literature) and needs the full teacher only to fit a and b.
  direction    one yardstick for every candidate loss, on the student as it is: the angle between the mean
               last-layer gradient of the candidate and that of full KD, and the ratio of their norms. Candidates:
               MaskedKD (s - T(x_S)), recalibrated target, channel-corrected, and same-view matching
               KL(T(x_S) || s(x_S)) (its gradient uses the student's feature on the kept tokens).
  dark         for more than two classes: KL split into the target-class part and the shape of the other classes
               (decoupled KD), the latter against a uniform shape: is there anything left in T(x_S) beyond "how sure"?
  view         whether the student's own answer on the same kept tokens, s(x_S), is closer to T(x_S) than its answer
               on the full image s(x) is: KL(T(x_S) || s(x_S)) vs KL(T(x_S) || s(x)). If so, matching the two on the
               same view is a consistent objective that costs one small student forward.
  groups       where the dataset has groups: accuracy of T(x_S) and mean squared error per group.
Clean val views, eval mode: training uses augmented views, so read the numbers as an optimistic bound on fidelity.
The gradient is the last layer's only (exact for the classifier, a proxy for the rest of the network).
"""
import math

import numpy
import torch

META = {
    "question": "MaskedKD trains the student on the full image against the teacher's answer on a subset of tokens. "
                "How large is that inconsistency, how much of it is a shared (correctable) shift versus per-image "
                "error, is it a bias or a noise in the gradient, and would an unbiased two-phase estimator or a "
                "same-view objective remove it at lower teacher cost?",
    "method": "keep-only masked-teacher fidelity; affine recalibration fitted on a split half; last-layer gradient "
              "bias and variance with the teacher answer as a control variate; two-phase (double) sampling variance",
    "paper": "Son et al. ECCV 2024 (MaskedKD); Neyman 1938 (double sampling); Peherstorfer, Willcox & Gunzburger "
             "SIAM J. Sci. Comput. 2016 (multifidelity Monte Carlo); Angelopoulos et al. Science 2023 "
             "(prediction-powered inference); Guo et al. ICML 2017 (temperature scaling); Beyer et al. CVPR 2022 "
             "(consistent teaching); Patrini et al. CVPR 2017 (forward / backward loss correction); Zhao et al. "
             "CVPR 2022 (decoupled KD); Covert et al. ICLR 2023 (ViT-Shapley surrogate); Abnar & Zuidema ACL 2020",
}

K_LIST = (12, 29, 49, 98)
K_REF = (29, 49, 98)          # MaskedKD budgets the equal-cost table is built against
# budget -> (m of the images-only design, K' and m of the tokens + images design): corrected/arms.py SPLITS (a test
# keeps the two equal). Each spends the teacher GFLOPs of MaskedKD at that budget, within 1.5 %.
DESIGNS = {29: (20, 12, 11), 49: (33, 29, 13)}
# teacher GFLOPs of one pass on K tokens = COST_FIXED + COST_TOKEN * (K + 1): FlopCounterMode on DeiT-B (the patch
# embedding runs on the whole image whatever K is; the rest is linear in the tokens to within 0.1 %)
COST_FIXED, COST_TOKEN = 0.2323, 0.16982
M_FRACS = (1 / 64, 1 / 32, 1 / 16, 1 / 8)
BATCH = 128                   # training batch (config.BATCH): the minibatch the noise is measured for
EVAL_BATCH = 50
DEPTH = 12
EPS = 1e-8
FULL_TOKENS = 197
RULES = ("maskedkd", "rollout", "random")
FIT_STEPS = 400


def _topk(s, k):
    idx = s.topk(k, dim=1).indices
    return torch.zeros_like(s, dtype=torch.bool).scatter_(1, idx, True)


def _scores(attn, rule, rng, n_patches, device):
    if rule == "maskedkd":
        return attn[DEPTH - 1].float().mean(1)[:, 0, 1:]
    if rule == "random":
        return torch.from_numpy(rng.random((attn[0].shape[0], n_patches))).float().to(device)
    r = None
    for a in attn:
        a = a.float().mean(1)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    return r[:, 0, 1:]


def _students(api):
    """Students of this dataset with weights on this server: Full KD (final, and its first snapshot, where the
    full-KD gradient is not yet near zero), CE, and MaskedKD at K=29."""
    got = {}
    runs = api.runs()
    for label, method, keep in (("full", "full", None), ("ce", "ce", None), ("maskedkd29", "maskedkd", 29.0)):
        cands = sorted((r["tier"] != "final", r["seed"], rid) for rid, r in runs.items()
                       if r.get("dataset") == api.dataset and r["method"] == method and r["tier"] in ("final", "screen")
                       and (keep is None or float(r["keep"]) == keep))
        for _, _, rid in cands:
            try:
                got[label] = (rid, api.student(rid))
            except ValueError:
                continue
            snaps = runs[rid].get("snapshots") or []
            if label == "full" and snaps:
                try:
                    got[f"full_e{snaps[0]}"] = (rid, api.student(rid, snaps[0]))
                except ValueError:
                    pass
            break
    return got


def _partial(model, x, keep, k):
    """Logits and the pre-head CLS feature of a ViT on the kept tokens only (harness teacher_forward_masked, which
    returns logits only; every row keeps exactly k tokens here)."""
    idx = keep.float().argsort(dim=1, descending=True, stable=True)[:, :k]
    t = model.patch_embed(x)
    t = model._pos_embed(t)
    cls, patches = t[:, :1], t[:, 1:]
    patches = torch.gather(patches, 1, idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
    t = model.norm(model.blocks(model.norm_pre(model.patch_drop(torch.cat([cls, patches], dim=1)))))
    return model.forward_head(t), t[:, 0]


def _collect(api, student, rng):
    """Per image, as float CPU tensors: label y, feature h, student logits zs, full-teacher logits zf, and per
    (rule, K) the masked-teacher logits zm and the student's logits on the same kept tokens zsp."""
    out = {"y": [], "h": [], "zs": [], "zf": []}
    per = {(r, k): {"zm": [], "zsp": [], "hsp": []} for r in RULES for k in K_LIST}
    for x, y in api.val_batches(batch=EVAL_BATCH):
        with torch.no_grad():
            fwd = api.student_forward(student, x, attn_layers=range(DEPTH), token_layers=[DEPTH - 1])
            out["y"].append(y.cpu())
            out["h"].append(student.norm(fwd.tokens[DEPTH - 1])[:, 0].float().cpu())
            out["zs"].append(fwd.logits.float().cpu())
            out["zf"].append(api.teacher(x).float().cpu())
            for rule in RULES:
                s = _scores(fwd.attn, rule, rng, api.num_patches, x.device)
                for k in K_LIST:
                    keep = _topk(s, k)
                    per[(rule, k)]["zm"].append(api.teacher_forward_keepmask(api.teacher, x, keep).float().cpu())
                    zsp, hsp = _partial(student, x, keep, k)
                    per[(rule, k)]["zsp"].append(zsp.float().cpu())
                    per[(rule, k)]["hsp"].append(hsp.float().cpu())
    return {k: torch.cat(v) for k, v in out.items()}, {key: {n: torch.cat(v) for n, v in d.items()} for key, d in per.items()}


def _kl(p, logq):
    return (p * (p.clamp_min(EPS).log() - logq)).sum(1)


def _fit_affine(zm, tf, rows):
    """Temperature a and class bias b minimising mean KL(tf || softmax(a * zm + b)) on `rows` (convex in a, b):
    zm are the source logits, tf the target probabilities (used in both directions, see the docstring).
    Plain Adam written out: the analysis validator allows torch.autograd.grad but no .backward()."""
    zm, tf = zm[rows], tf[rows]
    a = torch.ones((), requires_grad=True)
    b = torch.zeros(zm.shape[1], requires_grad=True)
    m = [torch.zeros_like(a), torch.zeros_like(b)]
    v = [torch.zeros_like(a), torch.zeros_like(b)]
    for step in range(1, FIT_STEPS + 1):
        loss = _kl(tf, (a * zm + b).log_softmax(1)).mean() + 1e-3 * (b * b).mean()
        grads = torch.autograd.grad(loss, [a, b])
        for i, (p, g) in enumerate(zip([a, b], grads)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            with torch.no_grad():
                p -= 0.05 * (m[i] / (1 - 0.9 ** step)) / ((v[i] / (1 - 0.999 ** step)).sqrt() + 1e-8)
    return a.detach(), b.detach()


def _moment(u, w, h):
    """trace Cov_i[u_i (x) h_i, w_i (x) h_i]."""
    n = u.shape[0]
    second = ((u * w).sum(1) * (h * h).sum(1)).mean()
    return float(second - ((u.T @ h / n) * (w.T @ h / n)).sum())


def _mean_sq(u, h):
    """| mean_i u_i (x) h_i |^2."""
    return float(((u.T @ h / u.shape[0]) ** 2).sum())


def _direction(g, hg, ref, href):
    """Angle (degrees) between mean_i g_i (x) hg_i and mean_i ref_i (x) href_i, and the ratio of their norms."""
    a, b = g.T @ hg / g.shape[0], ref.T @ href / ref.shape[0]
    na, nb = float((a * a).sum().sqrt()), float((b * b).sum().sqrt())
    cos = float((a * b).sum()) / max(na * nb, EPS)
    return {"angle_deg": math.degrees(math.acos(max(-1.0, min(1.0, cos)))), "norm_ratio": na / max(nb, EPS)}


def _dark(tf, tm, y):
    """Decoupled-KD split of KL(tf || tm) given the true class: the binary target / non-target part, the KL between
    the two non-target shapes, and that shape's KL to a uniform one. None for two classes (no shape to speak of)."""
    n_cls = tf.shape[1]
    if n_cls <= 2:
        return None
    hot = torch.zeros_like(tf).scatter_(1, y.view(-1, 1), 1.0).bool()
    pf, pm = tf[hot].clamp(EPS, 1 - EPS), tm[hot].clamp(EPS, 1 - EPS)
    target = pf * (pf / pm).log() + (1 - pf) * ((1 - pf) / (1 - pm)).log()
    shape = lambda t: (t.masked_fill(hot, 0.0) / t.masked_fill(hot, 0.0).sum(1, keepdim=True).clamp_min(EPS))   # noqa: E731
    nf, nm = shape(tf), shape(tm)
    kl = lambda p, q: (p * (p.clamp_min(EPS).log() - q.clamp_min(EPS).log())).masked_fill(hot, 0.0).sum(1)   # noqa: E731
    uni = torch.full_like(tf, 1.0 / (n_cls - 1))
    return {"target_part": float(target.mean()), "shape_kl_masked": float(kl(nf, nm).mean()),
            "shape_kl_uniform": float(kl(nf, uni).mean()),
            "shape_left_vs_uniform": 1 - float(kl(nf, nm).mean()) / max(float(kl(nf, uni).mean()), EPS)}


def _teacher_cost(k):
    """Teacher forward cost on k tokens relative to the full teacher (measured GFLOPs, see COST_FIXED)."""
    return (COST_FIXED + COST_TOKEN * (k + 1)) / (COST_FIXED + COST_TOKEN * FULL_TOKENS)


def _mix(regs, tf, h):
    """Least Var[(tf - sum_j w_j r_j) (x) h] over non-negative w (every active set tried; at most 3 regressors).
    Returns (variance left, weights)."""
    n = len(regs)
    G = [[_moment(regs[i], regs[j], h) for j in range(n)] for i in range(n)]
    c = [_moment(tf, r, h) for r in regs]
    v = _moment(tf, tf, h)
    best, best_w = v, [0.0] * n
    for mask in range(1, 2 ** n):
        A = [i for i in range(n) if mask >> i & 1]
        M = torch.tensor([[G[i][j] for j in A] for i in A], dtype=torch.float64)
        M = M + 1e-6 * M.diagonal().mean().clamp_min(EPS) * torch.eye(len(A), dtype=torch.float64)
        w = torch.linalg.solve(M, torch.tensor([c[i] for i in A], dtype=torch.float64))
        if (w < 0).any():
            continue
        left = v - 2 * sum(float(w[a]) * c[i] for a, i in enumerate(A)) + float(w @ M @ w)
        if left < best:
            best, best_w = left, [0.0] * n
            for a, i in enumerate(A):
                best_w[i] = float(w[a])
    return max(best, 0.0), best_w


def _gradient_stats(tf, tm, ts, onehot, h, rows, w_cost):
    tf, tm, ts, y, h = tf[rows], tm[rows], ts[rows], onehot[rows], h[rows]
    v_f, v_m, v_y = _moment(tf, tf, h), _moment(tm, tm, h), _moment(y, y, h)
    c_fm, c_fy, c_my = _moment(tf, tm, h), _moment(tf, y, h), _moment(tm, y, h)
    lam = c_fm / max(v_m, EPS)
    noise = _moment(ts - tf, ts - tf, h) / BATCH
    bias_sq = _mean_sq(tm - tf, h)
    v_res = v_f - lam * c_fm                                   # control variate: masked answer
    v_res_y = v_f - c_fy * c_fy / max(v_y, EPS)                # control variate: label (free)
    det = max(v_m * v_y - c_my * c_my, EPS)                    # both
    lam_m, lam_y = (c_fm * v_y - c_fy * c_my) / det, (c_fy * v_m - c_fm * c_my) / det
    v_res_my = v_f - lam_m * c_fm - lam_y * c_fy
    r2_y, r2_my = 1 - v_res_y / max(v_f, EPS), 1 - v_res_my / max(v_f, EPS)
    gain = (r2_my - r2_y) / max(1 - r2_y, EPS)
    need = 4 * w_cost / (1 + w_cost) ** 2
    shift = (tm - tf).mean(0)
    v_student = noise * BATCH                                  # control variate: the student's own answer (free)
    v_free, w_free = _mix([ts, y], tf, h)
    v_all, w_all = _mix([ts, y, tm], tf, h)
    var = lambda v: {f"m=B/{round(1 / f)}": (1 / (f * BATCH) - 1 / BATCH) * v / max(noise, EPS) for f in M_FRACS}   # noqa: E731
    return {"bias_sq": bias_sq, "noise_full_kd": noise, "bias_sq_over_noise": bias_sq / max(noise, EPS),
            # how much of the bias is one shift of the class distribution (the classifier's bias term absorbs it)
            "prior_shift_share_of_bias": float((shift * shift).sum() * (h.mean(0) ** 2).sum()) / max(bias_sq, EPS),
            "rho_sq": c_fm * c_fm / max(v_f * v_m, EPS), "lambda_star": lam, "v_full": v_f,
            "v_res": v_res, "v_res_label": v_res_y, "v_res_masked_and_label": v_res_my,
            "r2_label": r2_y, "r2_masked_and_label": r2_my, "gain_beyond_label": gain,
            "gain_needed_to_pay": need, "masking_pays": bool(gain > need),
            "v_student": v_student, "v_free": v_free, "w_free_student_label": w_free,
            "v_all": v_all, "w_all_student_label_masked": w_all,
            "gain_beyond_free": 1 - v_all / max(v_free, EPS),
            "two_phase_var_over_noise": var(v_res), "two_phase_with_label_var_over_noise": var(v_res_my),
            "labels_subsample_var_over_noise": var(v_res_y), "subsample_var_over_noise": var(v_student)}


def _equal_cost(stats):
    """What the teacher compute of MaskedKD at each K_REF buys under each design, as squared last-layer gradient
    error over one minibatch of full KD's own noise. MaskedKD's error is its bias (the same every step); the others
    are unbiased and their error is variance (it averages out over steps).
      labels + subsample   no masking: full teacher on a share pi = budget of the images, labels as control variate
      two-phase            masked teacher (rule, K) on every image, full teacher on pi = budget - cost(K) of them,
                           masked answer and label as control variates; best (rule, K, raw or recalibrated)
      subsample            full teacher on pi = budget of the images, nothing else"""
    def var_at(pi, v, noise):
        return (1 / (pi * BATCH) - 1 / BATCH) * v / max(noise, EPS)
    table = {}
    for kref in K_REF:
        budget = _teacher_cost(kref)
        ref = stats[("maskedkd", kref)]["raw"]
        row = {"teacher_cost": round(budget, 4), "maskedkd_bias_sq_over_noise": ref["bias_sq_over_noise"],
               "labels_subsample_var_over_noise": var_at(budget, ref["v_res_label"], ref["noise_full_kd"]),
               "subsample_var_over_noise": var_at(budget, ref["v_student"], ref["noise_full_kd"])}
        best = None
        for (rule, k), st in stats.items():
            pi = budget - _teacher_cost(k)
            if pi <= 1 / BATCH:   # at least one image must get the full teacher
                continue
            for kind in ("raw", "recal"):
                v = var_at(pi, st[kind]["v_res_masked_and_label"], st[kind]["noise_full_kd"])
                if best is None or v < best["var_over_noise"]:
                    best = {"rule": rule, "k": k, "cheap_signal": kind, "full_teacher_share": round(pi, 4),
                            "var_over_noise": v}
        row["best_two_phase"] = best
        cands = {"labels + subsample": row["labels_subsample_var_over_noise"]}
        if best is not None:
            cands["two-phase"] = best["var_over_noise"]
        row["best_unbiased_design"] = min(cands, key=cands.get)
        # steps after which MaskedKD's accumulated bias outweighs the best unbiased design's accumulated noise
        row["steps_until_bias_dominates"] = min(cands.values()) / max(row["maskedkd_bias_sq_over_noise"], EPS)
        table[f"maskedkd@{kref}"] = row
    return table


def _design(stats):
    """The two designs corrected/ trains, at their own splits: added gradient noise of each over one minibatch of
    full KD's own noise, and which is smaller. MaskedKD's bias^2 on the same scale for reference (it does not
    average out over steps; the others do)."""
    out = {}
    for budget, (m_img, k, m_two) in DESIGNS.items():
        st = stats[("maskedkd", k)]["recal"]
        noise = max(st["noise_full_kd"], EPS)
        img = (1 / m_img - 1 / BATCH) * st["v_free"] / noise
        two = (1 / m_two - 1 / BATCH) * st["v_all"] / noise
        need = 1 - (1 / m_img - 1 / BATCH) / (1 / m_two - 1 / BATCH)
        out[str(budget)] = {"images_only_m": m_img, "tokens_images_k_m": [k, m_two],
                            "images_only_var_over_noise": img, "tokens_images_var_over_noise": two,
                            "gain_beyond_free": st["gain_beyond_free"], "gain_needed": need,
                            "predicted_better": "tokens + images" if two < img else "images only",
                            "maskedkd_bias_sq_over_noise": stats[("maskedkd", budget)]["raw"]["bias_sq_over_noise"],
                            "plain_subsample_var_over_noise": (1 / m_img - 1 / BATCH) * st["v_student"] / noise}
    return out


def _analyse(base, per, groups, rng):
    y, h, zs, zf = base["y"], base["h"], base["zs"], base["zf"]
    n, n_cls = zf.shape
    tf, ts = zf.softmax(1), zs.softmax(1)
    onehot = torch.zeros(n, n_cls).scatter_(1, y.view(-1, 1), 1.0)
    prior = tf.mean(0, keepdim=True)
    perm = torch.from_numpy(rng.permutation(n))
    fit_rows, held = perm[: n // 2], perm[n // 2:]
    out, stats = {}, {}
    for (rule, k), d in per.items():
        zm, zsp, hsp = d["zm"], d["zsp"], d["hsp"]
        tm, tsp = zm.softmax(1), zsp.softmax(1)
        a, b = _fit_affine(zm, tf, fit_rows)
        tr = (a * zm + b).softmax(1)
        fa, fb = _fit_affine(zf, tm, fit_rows)                      # the channel: full answer -> masked answer
        kl_m_f, kl_m_cf = _kl(tm, zf.log_softmax(1)), _kl(tm, (fa * zf + fb).log_softmax(1))
        ref = (ts - tf)[held]                                       # full KD's logit gradient
        through = fa * ((fa * zs + fb).softmax(1) - tm)             # gradient of KL(tm || C(s)) w.r.t. the logits
        kl_raw, kl_rec = _kl(tf, zm.log_softmax(1)), _kl(tf, (a * zm + b).log_softmax(1))
        den = ((prior - tf) ** 2).sum()
        rec = {
            "teacher_cost": round(_teacher_cost(k), 4),
            "size": {"kl_full_to_masked": float(kl_raw.mean()), "masked_teacher_acc": 100 * float((zm.argmax(1) == y).float().mean()),
                     "full_teacher_acc": 100 * float((zf.argmax(1) == y).float().mean()),
                     "share_worse_than_label": float((((tm - tf) ** 2).sum(1) > ((onehot - tf) ** 2).sum(1)).float().mean())},
            "systematic": {"temperature_a": float(a), "bias_abs_mean": float(b.abs().mean()),
                           "kl_heldout_raw": float(kl_raw[held].mean()), "kl_heldout_recalibrated": float(kl_rec[held].mean()),
                           "share_of_kl_removed": 1 - float(kl_rec[held].mean()) / max(float(kl_raw[held].mean()), EPS),
                           "prior_mixing_w": float(((tm - tf) * (prior - tf)).sum() / den.clamp_min(EPS)),
                           "masked_entropy": float(-(tm * tm.clamp_min(EPS).log()).sum(1).mean()),
                           "full_entropy": float(-(tf * tf.clamp_min(EPS).log()).sum(1).mean())},
            "raw": _gradient_stats(tf, tm, ts, onehot, h, held, _teacher_cost(k)),
            "recal": _gradient_stats(tf, tr, ts, onehot, h, held, _teacher_cost(k)),
            "channel": {"temperature_a": float(fa), "bias_abs_mean": float(fb.abs().mean()),
                        "share_of_shift_explained": 1 - float(kl_m_cf[held].mean()) / max(float(kl_m_f[held].mean()), EPS)},
            "direction": {"maskedkd": _direction((ts - tm)[held], h[held], ref, h[held]),
                          "recalibrated_target": _direction((ts - tr)[held], h[held], ref, h[held]),
                          "through_channel": _direction(through[held], h[held], ref, h[held]),
                          "same_view": _direction((tsp - tm)[held], hsp[held], ref, h[held]),
                          "labels_only": _direction((ts - onehot)[held], h[held], ref, h[held])},
            "dark": _dark(tf, tm, y),
            "view": {"kl_masked_teacher_to_student_full_view": float(_kl(tm, zs.log_softmax(1)).mean()),
                     "kl_masked_teacher_to_student_same_view": float(_kl(tm, zsp.log_softmax(1)).mean()),
                     "kl_full_teacher_to_student_full_view": float(_kl(tf, zs.log_softmax(1)).mean()),
                     "student_acc_full_view": 100 * float((zs.argmax(1) == y).float().mean()),
                     "student_acc_same_view": 100 * float((zsp.argmax(1) == y).float().mean())},
        }
        if groups is not None:
            g = torch.from_numpy(groups)
            rec["groups"] = {int(v): {"n": int((g == v).sum()),
                                      "masked_teacher_acc": 100 * float((zm.argmax(1) == y)[g == v].float().mean()),
                                      "full_teacher_acc": 100 * float((zf.argmax(1) == y)[g == v].float().mean()),
                                      "sq_err": float(((tm - tf) ** 2).sum(1)[g == v].mean())} for v in g.unique().tolist()}
        out[f"{rule}@{k}"] = rec
        stats[(rule, k)] = rec
    return out, _equal_cost(stats), _design(stats)


def run(api):
    students = _students(api)
    if not students:
        api.write_json("summary", {"dataset": api.dataset, "error": "no trained student of this dataset on this server"})
        return
    g = api.val_groups()
    groups = None if g is None else g.numpy()
    summary = {"dataset": api.dataset, "k_list": list(K_LIST), "rules": list(RULES), "batch": BATCH,
               "students": {k: v[0] for k, v in students.items()}, "results": {}, "equal_cost": {}, "design": {},
               "headline": {}}
    for label, (rid, student) in students.items():
        rng = numpy.random.default_rng(0)
        with torch.autocast(api.device.type, enabled=False):
            base, per = _collect(api, student, rng)
            res, table, design = _analyse(base, per, groups, rng)
        summary["results"][label] = res
        summary["equal_cost"][label] = table
        summary["design"][label] = design
        summary["n_val"] = int(base["y"].shape[0])
        summary["headline"][label] = {key: {"kl": round(r["size"]["kl_full_to_masked"], 3),
                                            "worse_than_label": round(r["size"]["share_worse_than_label"], 3),
                                            "kl_removed_by_one_map": round(r["systematic"]["share_of_kl_removed"], 3),
                                            "bias_sq_over_noise": round(r["raw"]["bias_sq_over_noise"], 3),
                                            "prior_shift_share_of_bias": round(r["raw"]["prior_shift_share_of_bias"], 3),
                                            "rho_sq": round(r["raw"]["rho_sq"], 3),
                                            "rho_sq_recal": round(r["recal"]["rho_sq"], 3),
                                            "r2_label": round(r["raw"]["r2_label"], 3),
                                            "gain_beyond_label": round(r["raw"]["gain_beyond_label"], 3),
                                            "gain_beyond_label_recal": round(r["recal"]["gain_beyond_label"], 3),
                                            "gain_needed_to_pay": round(r["raw"]["gain_needed_to_pay"], 3),
                                            "gain_beyond_free": round(r["recal"]["gain_beyond_free"], 3),
                                            "masking_pays": r["raw"]["masking_pays"] or r["recal"]["masking_pays"],
                                            "channel_explains": round(r["channel"]["share_of_shift_explained"], 3),
                                            "angle_maskedkd": round(r["direction"]["maskedkd"]["angle_deg"], 1),
                                            "angle_recal": round(r["direction"]["recalibrated_target"]["angle_deg"], 1),
                                            "angle_channel": round(r["direction"]["through_channel"]["angle_deg"], 1),
                                            "angle_same_view": round(r["direction"]["same_view"]["angle_deg"], 1),
                                            "angle_labels": round(r["direction"]["labels_only"]["angle_deg"], 1),
                                            "same_view_kl_ratio": round(r["view"]["kl_masked_teacher_to_student_same_view"]
                                                                        / max(r["view"]["kl_masked_teacher_to_student_full_view"], EPS), 3)}
                                      for key, r in res.items()}
    api.write_json("summary", summary)
