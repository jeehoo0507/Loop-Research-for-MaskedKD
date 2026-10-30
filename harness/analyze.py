"""Run one analysis job (started by the scheduler, never by the agent).

    python -m harness.analyze --job queue/running/<id>.json

Writes analysis_out/<id>/ (the script's json/png, plus log.txt and meta.json).
"""
import argparse
import json
import os
import sys
import time
import traceback

import torch

from . import config as C
from .analysis_api import AnalysisAPI
from .models import make_teacher
from .train import load_teacher
from .validate import load_analysis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    a = ap.parse_args()
    job = json.loads(open(a.job).read())
    out = C.ROOT / "analysis_out" / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    meta = {"id": job["id"], "name": job["method"], "source_sha": job["selector_sha"], "status": "running"}
    try:
        run, script_meta = load_analysis(job["method"], job["selector_src"])
        meta["meta"] = script_meta
        dataset = job.get("dataset", C.STAGE1)
        teacher = load_teacher(device, dataset)[0]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            run(AnalysisAPI(out, device, teacher, dataset))
        meta["status"] = "done"
        rc = 0
    except Exception as e:
        meta["status"] = "error"
        meta["error"] = f"{type(e).__name__}: {e}"
        (out / "error.txt").write_text(traceback.format_exc())
        rc = 1
    meta["seconds"] = round(time.time() - t0, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
