"""Complete ledger of finished runs, so the audit bundle can show the whole suffdisc search history.

The audit bundle cuts RESULTS_TABLE at 12k characters, which hides the "All finished runs" rows of suffdisc
(PROGRESS STUCK line). This analysis only lists what api.runs() returns (status done), no model is run.
Run ids end in MMDD_HHMMSS_xxxx = submit time, so the list also shows the submit order.
  suffdisc_runs   every finished suffdisc run: id, tier, K, seed, val acc / WGA (last epoch)
  agent_runs      every finished run of the other agent methods, same columns
  summary         run counts per (method, tier) for all Waterbirds runs, and the counts above
"""
from collections import Counter

META = {
    "question": "which suffdisc runs exist at all (every tier, K, seed) and in which order were they submitted?",
    "method": "none (bookkeeping of finished runs via api.runs(); no interpretability method is applied)",
    "paper": "none",
}

BUILTIN = {"ce", "full", "maskedkd", "random", "grid", "center", "rollout", "tome", "oracle"}


def row(rid, r):
    wga = r.get("val_wga_last")
    wga = "-" if wga is None else f"{wga:.2f}"
    return f"{rid} | {r['tier']} | K={r['keep']} | s{r['seed']} | {r['val_acc_last']:.2f} / {wga}"


def run(api):
    runs = {k: v for k, v in api.runs().items() if v.get("dataset") in (None, api.dataset)}
    order = sorted(runs, key=lambda k: k.rsplit("_", 3)[-3] + k.rsplit("_", 3)[-2])
    suff = [row(k, runs[k]) for k in order if runs[k]["method"] == "suffdisc"]
    agent = [f"{runs[k]['method']}: " + row(k, runs[k]) for k in order if runs[k]["method"] not in BUILTIN | {"suffdisc"}]
    counts = Counter(f"{v['method']}/{v['tier']}" for v in runs.values())
    api.write_json("suffdisc_runs", {"n": len(suff), "rows (submit order)": suff})
    api.write_json("agent_runs", {"n": len(agent), "rows (submit order)": agent})
    api.write_json("summary", {"waterbirds_done_runs": len(runs), "suffdisc_done": len(suff),
                               "other_agent_done": len(agent), "per_method_tier": dict(sorted(counts.items()))})
