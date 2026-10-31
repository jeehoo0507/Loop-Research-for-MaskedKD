"""Aggregate runs/*/metrics.json into RESULTS_TABLE.md (harness-written; the agent cites run ids from it).

    python -m harness.report            # table + gate verdict + primary metric
    python -m harness.report --gate     # gate verdict only, exit 1 if not passed

Primary metric per method (same tier, same seeds): smallest mean kept tokens whose val acc is within
config.tolerance(Full, CE) %p of Full KD (half the KD gain, at most 1 %p) and, on Waterbirds, whose worst-group
acc is within max(1 %p, 2 x WGA seed sigma) of Full KD's. FLOPs columns are measured, per image, for a whole training step.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict

from . import config as C


def load_runs(dataset=None):
    """Finished runs (smoke runs excluded) at the current epoch budget of their tier: when the budget changes
    (agent/bootstrap.sh raises it if the gate fails), runs at the old budget drop out of every table.
    dataset=None: all datasets."""
    rs = []
    for m in C.RUNS.glob("*/metrics.json"):
        r = json.loads(m.read_text())
        r.setdefault("dataset", C.STAGE1)
        want = C.TIER_EPOCHS.get(r["dataset"], {}).get(r["tier"])
        if want is not None and "epochs" in r and r["epochs"] != want:
            continue
        if r.get("lr_override") is not None:                      # lr-sweep runs: read only by harness.boot
            continue
        if r.get("student_lr") is not None and abs(r["student_lr"] - C.student_lr(r["dataset"])) > 1e-12:
            continue                                               # trained at an lr the recipe no longer uses
        if r.get("status") == "done" and r["tier"] != "smoke" and dataset in (None, r["dataset"]):
            rs.append(r)
    return rs


def gate(runs, tier="screen", dataset=C.STAGE1, seed=None, kd_only=False):
    """Stage 1 gate: screen tier, mean over every seed that has all three runs (seed 0 first; agent/bootstrap.sh
    adds seeds 1, 2 when one seed misses). Stage 2 gate: final seed 1. `seed` pins a single seed.
    KD must beat CE by GATE_KD_OVER_CE on accuracy; MaskedKD@98 must not fall more than GATE_MASKEDKD98_GAP below
    Full KD. (Worst-group accuracy is a diagnostic only: the aim is FLOPs-efficient KD, not robustness.)
    kd_only: verdict on the KD-over-CE part alone (agent/bootstrap.sh: a MaskedKD@98 shortfall on the seed mean is
    recorded as a finding about MaskedKD on this dataset and the search goes on; KD not beating CE still stops it)."""

    def vals(method, keep, key):
        """{seed: latest value}"""
        out = {}
        for r in sorted(runs, key=lambda r: r["id"]):
            if (r["tier"] == tier and r["method"] == method and r.get("dataset", C.STAGE1) == dataset
                    and float(r["keep"]) == keep and (seed is None or r["seed"] == seed) and r.get(key) is not None):
                out[r["seed"]] = r[key]
        return out
    trio = (("ce", 0), ("full", 196), ("maskedkd", 98))
    a = {m: vals(m, k, "val_acc_last") for m, k in trio}
    seeds = sorted(set(a["ce"]) & set(a["full"]) & set(a["maskedkd"]))
    if not seeds:
        return None, "gate incomplete: " + ", ".join(f"{m} seeds {sorted(v)}" for m, v in a.items())
    ce, full, mk = (statistics.mean(a[m][s] for s in seeds) for m in ("ce", "full", "maskedkd"))
    ok_acc = full >= ce + C.GATE_KD_OVER_CE
    ok2 = mk >= full - C.GATE_MASKEDKD98_GAP      # MaskedKD@98 reproduces Full KD (being better is fine)
    msg = (f"[{dataset} {tier}, {C.TIER_EPOCHS.get(dataset, {}).get(tier)} epochs, mean of seeds {seeds}] "
           f"Full KD {full:.2f} vs CE {ce:.2f} (need +{C.GATE_KD_OVER_CE})")
    ok1 = ok_acc
    msg += (f": {'PASS' if ok1 else 'FAIL'}; MaskedKD@98 {mk:.2f} vs Full {full:.2f} "
            f"(need >= Full - {C.GATE_MASKEDKD98_GAP}): {'PASS' if ok2 else 'FAIL'}")
    return (ok1 if kd_only else ok1 and ok2), msg


def table(runs):
    groups = defaultdict(list)
    for r in runs:
        key = (r["tier"], r["method"], r.get("selector_sha") or "", json.dumps(r.get("sel_args", {}), sort_keys=True),
               float(r["keep"]))
        groups[key].append(r)
    from .tome import cost_ratio
    lines = ["| tier | method | sel_args | keep | mean kept | seeds | val acc (mean ± sd) | worst-group acc | total GFLOPs/img | teacher | selector | img/s* | run ids |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for (tier, method, sha, args, keep), rs in sorted(groups.items()):
        accs = [r["val_acc_last"] for r in rs]
        mean = statistics.mean(accs)
        sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
        fl = [r["flops"] for r in rs if r.get("flops")]
        f = lambda k: f"{statistics.mean(x[k] for x in fl):.2f}" if fl else "-"
        kept = [r["mean_kept"] for r in rs if r.get("mean_kept") is not None]
        ips = [r["img_per_s_mean"] for r in rs if r.get("img_per_s_mean")]
        wg = [r["val_wga_last"] for r in rs if r.get("val_wga_last") is not None]
        rows.append({"tier": tier, "method": method, "sha": sha, "args": args, "keep": keep, "acc": mean,
                     "wga": statistics.mean(wg) if wg else None,
                     "kept": statistics.mean(kept) if kept else None, "seeds": sorted(r["seed"] for r in rs)})
        wcol = (f"{statistics.mean(wg):.2f} ± {statistics.stdev(wg) if len(wg) > 1 else 0.0:.2f}") if wg else "-"
        over = cost_ratio(keep) if method == "tome" else 1.0
        tag = f" (teacher cost x{over:.2f} of budget: not matched)" if over > 1.05 else ""
        lines.append(f"| {tier} | {method}{'@' + sha[:6] if sha and sha != 'builtin' else ''}{tag} | {args if args != '{}' else ''} "
                     f"| {keep:g} | {rows[-1]['kept'] if rows[-1]['kept'] is None else round(rows[-1]['kept'], 1)} "
                     f"| {','.join(map(str, rows[-1]['seeds']))} | {mean:.2f} ± {sd:.2f} | {wcol} | {f('total_gflops_per_img')} "
                     f"| {f('teacher_gflops_per_img')} | {f('selector_gflops_per_img')} "
                     f"| {f'{statistics.mean(ips):.0f}' if ips else '-'} | {' '.join(r['id'] for r in rs)} |")
    return lines, rows


def noise(runs, tier="screen", key="val_acc_last", floor=C.PROMOTE_MIN):
    """Seed sigma from maskedkd@98 and full@196 screen runs over several seeds (pooled sd), and the
    promotion threshold max(floor, 2*sigma). key="val_wga_last" for worst-group accuracy."""
    groups = defaultdict(list)
    for r in runs:
        if (r["tier"] == tier and (r["method"], float(r["keep"])) in (("maskedkd", 98.0), ("full", 196.0))
                and r.get(key) is not None):
            groups[r["method"]].append(r[key])
    num = sum((len(v) - 1) * statistics.variance(v) for v in groups.values() if len(v) > 1)
    dof = sum(len(v) - 1 for v in groups.values() if len(v) > 1)
    what = "WGA" if key == "val_wga_last" else "acc"
    if dof == 0:
        return None, floor, f"{what} seed sigma: not measured yet (run harness.submit --baselines)"
    sigma = (num / dof) ** 0.5
    thr = max(floor, 2 * sigma)
    return sigma, thr, f"{what} seed sigma {sigma:.2f} %p (pooled, dof {dof}) -> threshold {thr:.2f} %p"


def discriminative(runs, tier="screen"):
    """Each baseline at each budget against Full KD (seed 0): where token choice matters on this dataset."""
    out = []
    full = [r for r in runs if r["tier"] == tier and r["method"] == "full" and r["seed"] == C.SCREEN_SEED]
    for k in sorted({float(r["keep"]) for r in runs if r["method"] == "random" and r["tier"] == tier}, reverse=True):
        cells = []
        for m in ("maskedkd", "rollout", "random", "grid", "center", "tome"):
            v = [r for r in runs if r["tier"] == tier and r["method"] == m and float(r["keep"]) == k and r["seed"] == C.SCREEN_SEED]
            if v:
                w = f" / WGA {v[-1]['val_wga_last']:.1f}" if v[-1].get("val_wga_last") is not None else ""
                cells.append(f"{m} {v[-1]['val_acc_last']:.2f}{w}")
        if cells:
            ref = (f" (Full KD {full[-1]['val_acc_last']:.2f}" + (f" / WGA {full[-1]['val_wga_last']:.1f}"
                   if full[-1].get("val_wga_last") is not None else "") + ")") if full else ""
            out.append(f"- K={k:g}: " + "; ".join(cells) + ref)
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


def paired(A, B, key="val_acc_last"):
    """Paired difference A-B of `key` over shared seeds with a 95% t-interval; None with < 2 shared seeds."""
    seeds = sorted(s for s in set(A) & set(B) if A[s].get(key) is not None and B[s].get(key) is not None)
    if len(seeds) < 2:
        return None
    d = [A[s][key] - B[s][key] for s in seeds]
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
        keys = [("val_acc_last", "acc")]   # worst-group accuracy is a diagnostic, shown in the run table only
        for ref, rk in refs:
            B = by_seed(runs, ref, rk, tier)
            label = f"{m}@{k:g} vs {ref}@{rk:g}" + (" (parent)" if ref == parent and ref not in C.BASELINES else "")
            for key, name in keys:
                p = paired(A, B, key)
                if p is None:
                    lines.append(f"- {label} [{name}]: missing (needs >= 2 shared seeds)")
                    continue
                verdict = "WIN" if p["lo"] > 0 else "LOSS" if p["hi"] < 0 else "no difference"
                g = f", GFLOPs ratio {p['gflops_ratio']:.3f}" if p["gflops_ratio"] else ""
                line = (f"- {label} [{name}]: {p['diff']:+.2f} %p, 95% CI [{p['lo']:+.2f}, {p['hi']:+.2f}], "
                        f"seeds {','.join(map(str, p['seeds']))}{g} -> {verdict}")
                lines.append(line)
                if verdict != "no difference":
                    sig.append(line)
    return lines, sig


def parent_check(runs, thr, tier="screen", thr_wga=None):
    """Screen seed 0: does each child selector beat its META parent at the same budget by the promotion threshold
    (on accuracy, or on worst-group accuracy by its own threshold)?"""
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
        dw = (a["val_wga_last"] - b["val_wga_last"]) if None not in (a.get("val_wga_last"), b.get("val_wga_last")) else None
        helps = d >= thr or (dw is not None and thr_wga is not None and dw >= thr_wga)
        wtxt = f", WGA {dw:+.2f} %p (threshold {thr_wga:.2f})" if dw is not None and thr_wga is not None else ""
        out.append(f"- {m}@{k:g} vs parent {parent_of(a)}: acc {d:+.2f} %p (threshold {thr:.2f}){wtxt} -> "
                   f"{'added part helps' if helps else 'added part UNPROVEN'}")
    return out


def primary(rows, tol_wga=None):
    """Smallest mean kept tokens whose mean accuracy is within tolerance(Full, CE) of Full KD. Entries backed by
    >= CLAIM_MIN_SEEDS seeds can support a claim; the others are screening only and listed apart. (tol_wga is
    unused: worst-group accuracy is a diagnostic, not part of the metric.)"""
    out = []
    for tier in sorted({r["tier"] for r in rows}):
        full = [r for r in rows if r["tier"] == tier and r["method"] == "full"]
        if not full:
            continue
        ref = full[0]["acc"]
        ce = [r["acc"] for r in rows if r["tier"] == tier and r["method"] == "ce"]
        tol = C.tolerance(ref, ce[0] if ce else None)
        best = {True: defaultdict(lambda: None), False: defaultdict(lambda: None)}
        for r in rows:
            if r["tier"] != tier or r["method"] in ("ce", "full") + C.ANALYSIS_ONLY or r["kept"] is None:
                continue
            if r["acc"] >= ref - tol:
                grade = len(r["seeds"]) >= C.CLAIM_MIN_SEEDS
                k = (r["method"], r["sha"], r["args"])
                if best[grade][k] is None or r["kept"] < best[grade][k][0]:
                    best[grade][k] = (r["kept"], len(r["seeds"]))
        gain = f"KD gain {ref - ce[0]:+.2f}, " if ce else "CE not run: "
        out.append(f"### {tier}: Full KD val {ref:.2f} (seeds {','.join(map(str, full[0]['seeds']))}); {gain}"
                   f"tolerance {tol:.2f} %p; smallest mean kept tokens within it")
        for grade, title in ((True, f"claim-grade (>= {C.CLAIM_MIN_SEEDS} seeds)"), (False, "screening only (fewer seeds)")):
            out.append(f"- {title}:" + ("" if best[grade] else " none"))
            for (m, sha, args), (kept, n) in sorted(best[grade].items(), key=lambda t: t[1][0]):
                out.append(f"  - {m}{'@' + sha[:6] if sha and sha != 'builtin' else ''} {args if args != '{}' else ''}: "
                           f"{kept:.1f} ({n} seeds)")
    return out


def stage_section(runs, dataset, tier):
    """Everything RESULTS_TABLE says about one dataset."""
    lines, rows = table(runs)
    ok, msg = gate(runs, tier, dataset)
    sigma, thr, nmsg = noise(runs, "screen")
    sw, thr_w, wmsg = noise(runs, "screen", "val_wga_last", C.WGA_TOL_MIN)
    groups = bool(C.NUM_GROUPS.get(dataset))
    core_lines, sig = core(runs, "final")
    return [f"Gate: {msg}", "", f"Noise: {nmsg}" + (f" (diagnostic: {wmsg})" if groups else ""), "",
            "## Significant only (final tier, paired 95% CI excludes 0; wins AND losses)", *(sig or ["- none yet"]), "",
            "## Core comparison (final tier: every agent method vs " + ", ".join(C.CLAIM_REFS) + ", full, parent)",
            *(core_lines or ["- no agent method has >= 2 final seeds yet"]), "",
            "## Parent check (screen seed 0: a child must beat its META parent by the promotion threshold)",
            *(parent_check(runs, thr, "screen") or ["- no child selectors yet"]), "",
            "## Primary metric (analysis-only methods excluded: " + ", ".join(C.ANALYSIS_ONLY) + ")",
            *primary(rows), "",
            "## Where token choice matters (screen seed 0, each baseline vs Full KD)",
            *(discriminative(runs, "screen") or ["- not measured yet"]), "",
            "## Search effort (screen)", *tries(runs, "screen"), "",
            "## All finished runs", *(lines if rows else ["- none yet"]), ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--tier", default="screen")
    ap.add_argument("--dataset", default=C.STAGE1, choices=list(C.DATASETS))
    ap.add_argument("--kd-only", action="store_true", help="with --gate: exit status from the KD-over-CE part only")
    a = ap.parse_args()
    if a.gate:
        tier = a.tier if a.dataset == C.STAGE1 else "final"
        ok, msg = gate(load_runs(a.dataset), tier, a.dataset, kd_only=a.kd_only)
        print(msg)
        sys.exit(0 if ok else 1)
    body = ["# RESULTS_TABLE (written by harness/report.py; do not edit by hand)", "",
            "*img/s: training throughput averaged over epochs; depends on how many jobs shared the GPU.",
            "Worst-group acc (Waterbirds groups y x place, val split): a diagnostic column only, never used in training or",
            "in any decision here. Decisions use mean accuracy and measured train-step GFLOPs.", "",
            f"# Stage 1: {C.STAGE1} (search)", ""]
    body += stage_section(load_runs(C.STAGE1), C.STAGE1, "screen")
    body += [f"# Stage 2: {C.STAGE2} (fixed methods from stage 1; 'stage2' in metrics.json says untouched/tuned)", ""]
    body += stage_section(load_runs(C.STAGE2), C.STAGE2, "final")
    (C.ROOT / "RESULTS_TABLE.md").write_text("\n".join(body) + "\n")
    print("\n".join(body))


if __name__ == "__main__":
    main()
