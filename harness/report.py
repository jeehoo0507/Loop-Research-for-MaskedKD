"""Aggregate runs/*/metrics.json into RESULTS_TABLE.md (harness-written; the agent cites run ids from it).

    python -m harness.report            # table + gate verdict + primary metric
    python -m harness.report --gate     # gate verdict only, exit 1 if not passed

Primary metric per method (same tier, same seeds): smallest mean kept tokens whose val acc is within
TOLERANCE %p of Full KD. FLOPs columns are measured, per image, for a whole training step.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict

from . import config as C


def load_runs(dataset=None):
    """Finished runs (smoke runs excluded). dataset=None: all datasets."""
    rs = []
    for m in C.RUNS.glob("*/metrics.json"):
        r = json.loads(m.read_text())
        r.setdefault("dataset", "cifar100")
        if r.get("status") == "done" and r["tier"] != "smoke" and dataset in (None, r["dataset"]):
            rs.append(r)
    return rs


def gate(runs, tier="screen", dataset="cifar100", seed=None):
    """CIFAR gate: screen seed 0. ImageNet-100 gate: final seed 1."""
    seed = (C.SCREEN_SEED if dataset == "cifar100" else C.FINAL_SEEDS[0]) if seed is None else seed

    def acc(method, keep):
        v = [r["val_acc_last"] for r in runs if r["tier"] == tier and r["method"] == method
             and r.get("dataset", "cifar100") == dataset and float(r["keep"]) == keep and r["seed"] == seed]
        return v[-1] if v else None
    ce, full, mk = acc("ce", 0), acc("full", 196), acc("maskedkd", 98)
    if None in (ce, full, mk):
        return None, f"gate incomplete: ce={ce} full={full} maskedkd98={mk}"
    ok1 = full >= ce + C.GATE_KD_OVER_CE
    ok2 = abs(mk - full) <= C.GATE_MASKEDKD98_GAP
    msg = (f"Full KD {full:.2f} vs CE {ce:.2f} (need +{C.GATE_KD_OVER_CE}): {'PASS' if ok1 else 'FAIL'}; "
           f"MaskedKD@98 {mk:.2f} vs Full {full:.2f} (need within {C.GATE_MASKEDKD98_GAP}): {'PASS' if ok2 else 'FAIL'}")
    return ok1 and ok2, msg


def table(runs):
    groups = defaultdict(list)
    for r in runs:
        key = (r["tier"], r["method"], r.get("selector_sha") or "", json.dumps(r.get("sel_args", {}), sort_keys=True),
               float(r["keep"]))
        groups[key].append(r)
    lines = ["| tier | method | sel_args | keep | mean kept | seeds | val acc (mean ± sd) | total GFLOPs/img | teacher | selector | img/s* | run ids |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for (tier, method, sha, args, keep), rs in sorted(groups.items()):
        accs = [r["val_acc_last"] for r in rs]
        mean = statistics.mean(accs)
        sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
        fl = [r["flops"] for r in rs if r.get("flops")]
        f = lambda k: f"{statistics.mean(x[k] for x in fl):.2f}" if fl else "-"
        kept = [r["mean_kept"] for r in rs if r.get("mean_kept") is not None]
        ips = [r["img_per_s_mean"] for r in rs if r.get("img_per_s_mean")]
        rows.append({"tier": tier, "method": method, "sha": sha, "args": args, "keep": keep, "acc": mean,
                     "kept": statistics.mean(kept) if kept else None, "seeds": sorted(r["seed"] for r in rs)})
        lines.append(f"| {tier} | {method}{'@' + sha[:6] if sha and sha != 'builtin' else ''} | {args if args != '{}' else ''} "
                     f"| {keep:g} | {rows[-1]['kept'] if rows[-1]['kept'] is None else round(rows[-1]['kept'], 1)} "
                     f"| {','.join(map(str, rows[-1]['seeds']))} | {mean:.2f} ± {sd:.2f} | {f('total_gflops_per_img')} "
                     f"| {f('teacher_gflops_per_img')} | {f('selector_gflops_per_img')} "
                     f"| {f'{statistics.mean(ips):.0f}' if ips else '-'} | {' '.join(r['id'] for r in rs)} |")
    return lines, rows


def noise(runs, tier="screen"):
    """Seed sigma from maskedkd@98 and full@196 screen runs over several seeds (pooled sd), and the
    promotion threshold max(PROMOTE_MIN, 2*sigma)."""
    groups = defaultdict(list)
    for r in runs:
        if r["tier"] == tier and (r["method"], float(r["keep"])) in (("maskedkd", 98.0), ("full", 196.0)):
            groups[r["method"]].append(r["val_acc_last"])
    num = sum((len(v) - 1) * statistics.variance(v) for v in groups.values() if len(v) > 1)
    dof = sum(len(v) - 1 for v in groups.values() if len(v) > 1)
    if dof == 0:
        return None, C.PROMOTE_MIN, "seed sigma: not measured yet (run harness.submit --baselines)"
    sigma = (num / dof) ** 0.5
    thr = max(C.PROMOTE_MIN, 2 * sigma)
    return sigma, thr, f"seed sigma {sigma:.2f} %p (pooled, dof {dof}) -> promotion threshold {thr:.2f} %p"


def discriminative(runs, tier="screen"):
    """maskedkd - random at each budget (seed 0): where token choice matters on this dataset."""
    out = []
    for k in sorted({float(r["keep"]) for r in runs if r["method"] == "random" and r["tier"] == tier}, reverse=True):
        a = [r["val_acc_last"] for r in runs if r["tier"] == tier and r["method"] == "maskedkd" and float(r["keep"]) == k and r["seed"] == C.SCREEN_SEED]
        b = [r["val_acc_last"] for r in runs if r["tier"] == tier and r["method"] == "random" and float(r["keep"]) == k and r["seed"] == C.SCREEN_SEED]
        if a and b:
            out.append(f"- K={k:g}: maskedkd {a[-1]:.2f} vs random {b[-1]:.2f} (diff {a[-1] - b[-1]:+.2f})")
    return out


def tries(runs, tier="screen"):
    """Distinct screen configurations per selector name (more tries = more chance a win is luck)."""
    cfg = defaultdict(set)
    for r in runs:
        if r["tier"] == tier and r["method"] not in C.BASELINES:
            cfg[r["method"]].add((r.get("selector_sha"), json.dumps(r.get("sel_args", {}), sort_keys=True), float(r["keep"])))
    total = sum(len(v) for v in cfg.values())
    lines = [f"- {m}: {len(v)} configs" for m, v in sorted(cfg.items(), key=lambda t: -len(t[1]))]
    return [f"Total agent screen configs: {total}", *lines]


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def by_seed(runs, method, keep, tier):
    """{seed: run} for one method at one budget (latest run per seed)."""
    out = {}
    for r in sorted(runs, key=lambda r: r["id"]):
        if r["tier"] == tier and r["method"] == method and float(r["keep"]) == float(keep):
            out[r["seed"]] = r
    return out


def paired(A, B):
    """Paired val-acc difference A-B over shared seeds with a 95% t-interval; None with < 2 shared seeds."""
    seeds = sorted(set(A) & set(B))
    if len(seeds) < 2:
        return None
    d = [A[s]["val_acc_last"] - B[s]["val_acc_last"] for s in seeds]
    m, sd = statistics.mean(d), statistics.stdev(d)
    half = T95[len(d) - 1] * sd / len(d) ** 0.5
    fl = lambda X: statistics.mean(X[s]["flops"]["total_gflops_per_img"] for s in seeds) if all(X[s].get("flops") for s in seeds) else None
    fa, fb = fl(A), fl(B)
    return {"seeds": seeds, "diff": m, "lo": m - half, "hi": m + half, "gflops_ratio": fa / fb if fa and fb else None}


def parent_of(r):
    return ((r.get("selector_meta") or {}).get("parent") or "none").strip()


def core(runs, tier="final"):
    """Every agent method at every budget with >= 2 final seeds, against each claim reference at the same
    budget, Full KD, and its META parent. Returns (all lines, significant-only lines)."""
    agent = sorted({(r["method"], float(r["keep"])) for r in runs if r["tier"] == tier and r["method"] not in C.BASELINES})
    lines, sig = [], []
    for m, k in agent:
        A = by_seed(runs, m, k, tier)
        if len(A) < 2:
            continue
        parent = parent_of(A[max(A)])
        refs = [(ref, k) for ref in C.CLAIM_REFS] + [("full", 196.0)] + ([(parent, k)] if parent != "none" else [])
        for ref, rk in refs:
            p = paired(A, by_seed(runs, ref, rk, tier))
            label = f"{m}@{k:g} vs {ref}@{rk:g}" + (" (parent)" if ref == parent and ref not in C.BASELINES else "")
            if p is None:
                lines.append(f"- {label}: missing (needs >= 2 shared seeds)")
                continue
            verdict = "WIN" if p["lo"] > 0 else "LOSS" if p["hi"] < 0 else "no difference"
            g = f", GFLOPs ratio {p['gflops_ratio']:.3f}" if p["gflops_ratio"] else ""
            line = (f"- {label}: {p['diff']:+.2f} %p, 95% CI [{p['lo']:+.2f}, {p['hi']:+.2f}], "
                    f"seeds {','.join(map(str, p['seeds']))}{g} -> {verdict}")
            lines.append(line)
            if verdict != "no difference":
                sig.append(line)
    return lines, sig


def parent_check(runs, thr, tier="screen"):
    """Screen seed 0: does each child selector beat its META parent at the same budget by the promotion threshold?"""
    out = []
    for m, k in sorted({(r["method"], float(r["keep"])) for r in runs if r["tier"] == tier and r["method"] not in C.BASELINES}):
        a = by_seed(runs, m, k, tier).get(C.SCREEN_SEED)
        if a is None or parent_of(a) == "none":
            continue
        b = by_seed(runs, parent_of(a), k, tier).get(C.SCREEN_SEED)
        if b is None:
            out.append(f"- {m}@{k:g}: parent {parent_of(a)}@{k:g} not run at this budget")
            continue
        d = a["val_acc_last"] - b["val_acc_last"]
        out.append(f"- {m}@{k:g} vs parent {parent_of(a)}: {d:+.2f} %p -> "
                   f"{'added part helps' if d >= thr else 'added part UNPROVEN (below threshold ' + format(thr, '.2f') + ')'}")
    return out


def primary(rows):
    out = []
    for tier in sorted({r["tier"] for r in rows}):
        full = [r for r in rows if r["tier"] == tier and r["method"] == "full"]
        if not full:
            continue
        ref = full[0]["acc"]
        best = defaultdict(lambda: None)
        for r in rows:
            if r["tier"] != tier or r["method"] in ("ce", "full") + C.ANALYSIS_ONLY or r["kept"] is None:
                continue
            if r["acc"] >= ref - C.TOLERANCE:
                k = (r["method"], r["sha"], r["args"])
                if best[k] is None or r["kept"] < best[k]:
                    best[k] = r["kept"]
        out.append(f"### {tier}: Full KD val {ref:.2f}; smallest mean kept tokens within {C.TOLERANCE} %p")
        for (m, sha, args), kept in sorted(best.items(), key=lambda t: t[1]):
            out.append(f"- {m}{'@' + sha[:6] if sha and sha != 'builtin' else ''} {args if args != '{}' else ''}: {kept:.1f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--tier", default="screen")
    ap.add_argument("--dataset", default="cifar100", choices=list(C.DATASETS))
    a = ap.parse_args()
    if a.gate:
        tier = a.tier if a.dataset == "cifar100" else "final"
        ok, msg = gate(load_runs(a.dataset), tier, a.dataset)
        print(msg)
        sys.exit(0 if ok else 1)
    body = ["# RESULTS_TABLE (written by harness/report.py; do not edit by hand)", "",
            "*img/s: training throughput averaged over epochs; depends on how many jobs shared the GPU.", ""]
    runs = load_runs("cifar100")
    lines, rows = table(runs)
    ok, msg = gate(runs, "screen", "cifar100")
    sigma, thr, nmsg = noise(runs, "screen")
    core_lines, sig = core(runs, "final")
    body += ["# Stage 1: CIFAR-100", "", f"Gate (screen, seed 0): {msg}", "", f"Noise: {nmsg}", "",
             "## Significant only (final tier, paired 95% CI excludes 0; wins AND losses)", *(sig or ["- none yet"]), "",
             "## Core comparison (final tier: every agent method vs " + ", ".join(C.CLAIM_REFS) + ", full, parent)",
             *(core_lines or ["- no agent method has >= 2 final seeds yet"]), "",
             "## Parent check (screen seed 0: a child must beat its META parent by the promotion threshold)",
             *(parent_check(runs, thr, "screen") or ["- no child selectors yet"]), "",
             "## Primary metric (analysis-only methods excluded: " + ", ".join(C.ANALYSIS_ONLY) + ")", *primary(rows), "",
             "## Where token choice matters (maskedkd - random, screen seed 0)", *(discriminative(runs, "screen") or ["- not measured yet"]), "",
             "## Search effort (screen)", *tries(runs, "screen"), "",
             "## All finished runs", *lines, ""]
    runs2 = load_runs(C.STAGE2)
    lines2, rows2 = table(runs2)
    ok2, msg2 = gate(runs2, "final", C.STAGE2)
    body += ["# Stage 2: ImageNet-100 (fixed methods from stage 1; 'stage2' column in metrics.json says untouched/tuned)", "",
             f"Gate (final, seed {C.FINAL_SEEDS[0]}): {msg2}", "",
             "## Core comparison (final)", *(core(runs2, "final")[0] or ["- none yet"]), "",
             "## Primary metric", *primary(rows2), "", "## All finished runs", *(lines2 if rows2 else ["- none yet"])]
    (C.ROOT / "RESULTS_TABLE.md").write_text("\n".join(body) + "\n")
    print("\n".join(body))


if __name__ == "__main__":
    main()
