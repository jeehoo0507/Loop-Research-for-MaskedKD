"""Paper figures and tables, from runs/*/metrics.json only.

    python -m harness.figures                    # all datasets/tiers that have runs
    python -m harness.figures --methods my_sel   # also draw these agent methods

Writes figures/<dataset>_<tier>_tokens.png (teacher tokens vs val acc), <..>_gflops.png (train-step
GFLOPs vs val acc), and <..>.csv with one row per (method, budget): mean, sd, n, GFLOPs, img/s, run ids.
Points are means over seeds; error bars are one sd when there is more than one seed.
"""
import argparse
import csv
import statistics
from collections import defaultdict

from . import config as C
from .report import load_runs

REF = ("full", "maskedkd", "random", "grid", "center", "tome", "oracle")
STYLE = {"full": ("#111111", "--"), "maskedkd": ("#1f5fbf", "-"), "random": ("#8a8a8a", ":"),
         "tome": ("#b8860b", "-."), "oracle": ("#7a3fbf", ":"),
         "grid": ("#5a9e6f", "--"), "center": ("#c0504d", "--")}


def rows_for(runs):
    g = defaultdict(list)
    for r in runs:
        if r["method"] == "ce":
            continue
        g[(r["method"], float(r["keep"]))].append(r)
    out = []
    for (m, k), rs in sorted(g.items()):
        acc = [r["val_acc_last"] for r in rs]
        fl = [r["flops"]["total_gflops_per_img"] for r in rs if r.get("flops")]
        ips = [r["img_per_s_mean"] for r in rs if r.get("img_per_s_mean")]
        out.append({"method": m, "keep": k, "mean": statistics.mean(acc), "sd": statistics.stdev(acc) if len(acc) > 1 else 0.0,
                    "n": len(acc), "gflops": statistics.mean(fl) if fl else None,
                    "img_per_s": statistics.mean(ips) if ips else None, "runs": " ".join(r["id"] for r in rs)})
    return out


def draw(rows, x, path, title, ce):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in sorted({r["method"] for r in rows}, key=lambda m: (m not in REF, m)):
        pts = sorted((r for r in rows if r["method"] == m and r[x] is not None), key=lambda r: r[x])
        if not pts:
            continue
        color, ls = STYLE.get(m, ("#c0392b", "-"))
        ax.errorbar([p[x] for p in pts], [p["mean"] for p in pts], yerr=[p["sd"] for p in pts], label=m,
                    color=color, linestyle=ls, marker="o", ms=4, capsize=3, lw=2 if m not in REF else 1.4)
    full = [r for r in rows if r["method"] == "full"]
    if full:
        ax.axhspan(full[0]["mean"] - C.TOLERANCE, full[0]["mean"], color="#2e8b57", alpha=0.10, lw=0,
                   label=f"within {C.TOLERANCE:g} %p of Full KD")
    if ce is not None:
        ax.axhline(ce, color="#999999", lw=1, linestyle=(0, (1, 3)), label="CE (no KD)")
    ax.set_xlabel("teacher patch tokens (mean, of 196)" if x == "keep" else "train-step GFLOPs per image")
    ax.set_ylabel("val top-1 (%)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=None, help="agent methods to include (default: all)")
    a = ap.parse_args()
    out = C.ROOT / "figures"
    out.mkdir(exist_ok=True)
    for ds in C.DATASETS:
        runs = load_runs(ds)
        for tier in sorted({r["tier"] for r in runs}):
            rs = [r for r in runs if r["tier"] == tier and (a.methods is None or r["method"] in REF + ("ce",) + tuple(a.methods))]
            rows = rows_for(rs)
            if not rows:
                continue
            ce = [r["val_acc_last"] for r in rs if r["method"] == "ce"]
            ce = statistics.mean(ce) if ce else None
            stem = out / f"{ds}_{tier}"
            with open(f"{stem}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            title = f"{'CIFAR-100' if ds == 'cifar100' else 'ImageNet-100'} · {tier}"
            draw(rows, "keep", f"{stem}_tokens.png", title, ce)
            draw(rows, "gflops", f"{stem}_gflops.png", title, ce)
            print(f"wrote {stem}_tokens.png, {stem}_gflops.png, {stem}.csv")


if __name__ == "__main__":
    main()
