"""What an analysis script (`analysis/<name>.py`) can touch. Offline only: nothing here is billed to
a training step and nothing here can become a result in RESULTS_TABLE.

An analysis file must define:

    META = {
        "question": "which student layer's attention best matches the teacher's important tokens?",
        "method": "Attention Rollout vs occlusion importance",
        "paper": "Abnar & Zuidema 2020; Zeiler & Fergus 2014",
    }

    def run(api):            # api: AnalysisAPI
        for x, y in api.val_batches(limit=1000): ...
        api.write_json("summary", {...})

Rules (checked by harness/validate.py, kind="analysis"): no files, no processes, no test data;
results leave only through api.write_json / api.write_figure into analysis_out/<job id>/.
"""
import json

import torch

from . import config as C
from .data import load_split, normalize_resize
from .models import (make_student, student_forward, teacher_forward_keepmask, teacher_forward_masked,
                     _attn_with_weights)


class AnalysisAPI:
    # model helpers, exposed so scripts need no other harness import
    student_forward = staticmethod(student_forward)
    teacher_forward_keepmask = staticmethod(teacher_forward_keepmask)
    teacher_forward_masked = staticmethod(teacher_forward_masked)
    attn_with_weights = staticmethod(_attn_with_weights)
    num_patches = C.NUM_PATCHES

    def __init__(self, out_dir, device, teacher):
        self._out = out_dir
        self.device = device
        self.teacher = teacher
        self._x, self._y = load_split("val")

    def val_batches(self, batch=100, limit=None):
        """Normalized 224px val images and labels, fixed order."""
        n = len(self._y) if limit is None else min(limit, len(self._y))
        for i in range(0, n, batch):
            j = min(i + batch, n)
            yield normalize_resize(self._x[i:j].to(self.device)), self._y[i:j].to(self.device)

    def runs(self):
        """Finished runs: id -> {method, keep, seed, tier, val_acc_last, snapshots}."""
        out = {}
        for m in sorted(C.RUNS.glob("*/metrics.json")):
            r = json.loads(m.read_text())
            if r.get("status") != "done":
                continue
            snaps = sorted(int(p.stem.split("_e")[1]) for p in m.parent.glob("student_e*.pt"))
            out[r["id"]] = {k: r[k] for k in ("method", "keep", "seed", "tier", "val_acc_last", "epochs")}
            out[r["id"]]["snapshots"] = snaps
        return out

    def student(self, run_id, epoch=None):
        """Trained student of a finished run (final weights, or a snapshot epoch from runs()[id]['snapshots'])."""
        name = "student.pt" if epoch is None else f"student_e{int(epoch)}.pt"
        path = C.RUNS / str(run_id) / name
        if not path.resolve().is_relative_to(C.RUNS.resolve()) or not path.exists():
            raise ValueError(f"no checkpoint {name} for run {run_id}")
        m = make_student()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return m.to(self.device).eval()

    def write_json(self, name, obj):
        (self._out / f"{_safe(name)}.json").write_text(json.dumps(obj, indent=1, default=float))

    def write_figure(self, name, fig):
        fig.savefig(self._out / f"{_safe(name)}.png", dpi=120, bbox_inches="tight")


def _safe(name):
    s = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))[:80]
    return s or "out"
