"""Test-set evaluation. HUMAN ONLY, once, at the end. The agent loop must never run this
(agent/AGENT_PROMPT.md forbids it and the auditor checks the logs for it).

    python -m harness.test_eval --runs <run_id> [<run_id> ...]

Writes final/test_<timestamp>.json. Never writes into runs/.
"""
import argparse
import json
import time

import torch

from . import config as C
from .data import load_groups, load_split
from .models import make_student
from .train import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dataset", default=C.STAGE1, choices=list(C.DATASETS))
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = load_split("test", a.dataset)
    g = load_groups("test", a.dataset)
    res = {}
    for rid in a.runs:
        m = make_student(a.dataset).to(device)
        m.load_state_dict(torch.load(C.RUNS / rid / "student.pt", map_location="cpu"))
        res[rid] = evaluate(m, x, y, device, g)   # Waterbirds: acc, per-group acc, worst-group acc
        print(rid, res[rid], flush=True)
    out = C.ROOT / "final"
    out.mkdir(exist_ok=True)
    (out / f"test_{time.strftime('%m%d_%H%M%S')}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
