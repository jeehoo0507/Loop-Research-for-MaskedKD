"""Which seeds each Waterbirds run used, so the audit bundle can check that seeds 4-6 were fresh and that no run is left out.

The 05:13 audit could not confirm from the truncated RESULTS_TABLE and the older run_ledger (finished before seeds 4-6
existed) that the new seeds were unused before and that every suffdisc/maskedkd run on them is reported.
This analysis only lists what api.runs() returns (status done), no model is run.
Run ids end in MMDD_HHMMSS_xxxx = submit time.
  fresh_seed_runs   every finished run with seed >= 4, any method, submit order
  suff_mkd_runs     every finished suffdisc and maskedkd run, every tier / K / seed, submit order
  summary           per method: the set of seeds used; and which methods ever used seed >= 4
"""
from collections import defaultdict

META = {
    "question": "were seeds 4-6 used by any run before the H-W30 replication, and is every suffdisc/maskedkd run on them reported?",
    "method": "none (bookkeeping of finished runs via api.runs(); no interpretability method is applied)",
    "paper": "none",
}


def row(rid, r):
    wga = r.get("val_wga_last")
    wga = "-" if wga is None else f"{wga:.2f}"
    return f"{rid} | {r['tier']} | K={r['keep']} | s{r['seed']} | {r['val_acc_last']:.2f} / {wga}"


def run(api):
    runs = {k: v for k, v in api.runs().items() if v.get("dataset") in (None, api.dataset)}
    order = sorted(runs, key=lambda k: k.rsplit("_", 3)[-3] + k.rsplit("_", 3)[-2])
    fresh = [f"{runs[k]['method']}: " + row(k, runs[k]) for k in order if (runs[k]["seed"] or 0) >= 4]
    sm = [f"{runs[k]['method']}: " + row(k, runs[k]) for k in order if runs[k]["method"] in ("suffdisc", "maskedkd")]
    seeds = defaultdict(set)
    for v in runs.values():
        seeds[v["method"]].add(v["seed"])
    api.write_json("fresh_seed_runs", {"n": len(fresh), "rows (submit order)": fresh})
    api.write_json("suff_mkd_runs", {"n": len(sm), "rows (submit order)": sm})
    api.write_json("summary", {"waterbirds_done_runs": len(runs),
                               "seeds_per_method": {m: sorted(s) for m, s in sorted(seeds.items())},
                               "methods_with_seed_ge4": sorted(m for m, s in seeds.items() if max(s) >= 4),
                               "first_seed_ge4_submit": order and next((k for k in order if (runs[k]["seed"] or 0) >= 4), None)})
