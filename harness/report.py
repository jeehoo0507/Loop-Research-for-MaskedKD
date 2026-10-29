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


def load_runs():
    rs = []
    for m in C.RUNS.glob("*/metrics.json"):
        r = json.loads(m.read_text())
        if r.get("status") == "done":
            rs.append(r)
    return rs


def gate(runs, tier="screen"):
    def acc(method, keep):
        v = [r["val_acc_last"] for r in runs if r["tier"] == tier and r["method"] == method
             and float(r["keep"]) == keep and r["seed"] == 0]
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
    lines = ["| tier | method | sel_args | keep | mean kept | seeds | val acc (mean ± sd) | total GFLOPs/img | teacher | selector | run ids |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for (tier, method, sha, args, keep), rs in sorted(groups.items()):
        accs = [r["val_acc_last"] for r in rs]
        mean = statistics.mean(accs)
        sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
        fl = [r["flops"] for r in rs if r.get("flops")]
        f = lambda k: f"{statistics.mean(x[k] for x in fl):.2f}" if fl else "-"
        kept = [r["mean_kept"] for r in rs if r.get("mean_kept") is not None]
        rows.append({"tier": tier, "method": method, "sha": sha, "args": args, "keep": keep, "acc": mean,
                     "kept": statistics.mean(kept) if kept else None, "seeds": sorted(r["seed"] for r in rs)})
        lines.append(f"| {tier} | {method}{'@' + sha[:6] if sha and sha != 'builtin' else ''} | {args if args != '{}' else ''} "
                     f"| {keep:g} | {rows[-1]['kept'] if rows[-1]['kept'] is None else round(rows[-1]['kept'], 1)} "
                     f"| {','.join(map(str, rows[-1]['seeds']))} | {mean:.2f} ± {sd:.2f} | {f('total_gflops_per_img')} "
                     f"| {f('teacher_gflops_per_img')} | {f('selector_gflops_per_img')} | {' '.join(r['id'] for r in rs)} |")
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
        if r["tier"] == tier and r["method"] not in ("ce", "full", "maskedkd", "random") + C.ANALYSIS_ONLY:
            cfg[r["method"]].add((r.get("selector_sha"), json.dumps(r.get("sel_args", {}), sort_keys=True), float(r["keep"])))
    total = sum(len(v) for v in cfg.values())
    lines = [f"- {m}: {len(v)} configs" for m, v in sorted(cfg.items(), key=lambda t: -len(t[1]))]
    return [f"Total agent screen configs: {total}", *lines]


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
    a = ap.parse_args()
    runs = load_runs()
    ok, msg = gate(runs, a.tier)
    if a.gate:
        print(msg)
        sys.exit(0 if ok else 1)
    lines, rows = table(runs)
    sigma, thr, nmsg = noise(runs, a.tier)
    body = ["# RESULTS_TABLE (written by harness/report.py; do not edit by hand)", "",
            f"Gate ({a.tier}, seed 0): {msg}", "", f"Noise: {nmsg}", "",
            "## Primary metric (analysis-only methods excluded: " + ", ".join(C.ANALYSIS_ONLY) + ")", *primary(rows), "",
            "## Where token choice matters (maskedkd - random, screen seed 0)", *(discriminative(runs, a.tier) or ["- not measured yet"]), "",
            "## Search effort (screen)", *tries(runs, a.tier), "",
            "## All finished runs", *lines]
    (C.ROOT / "RESULTS_TABLE.md").write_text("\n".join(body) + "\n")
    print("\n".join(body))


if __name__ == "__main__":
    main()
