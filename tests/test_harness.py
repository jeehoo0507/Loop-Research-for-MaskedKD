"""CPU tests for the harness. Run: MASKEDKD_FAKE_DATA=1 python -m pytest -q tests"""
import json
from pathlib import Path
import os

os.environ.setdefault("MASKEDKD_FAKE_DATA", "1")

import pytest
import torch
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy

from harness import config as C
from harness.baselines import MaskedKD, Random, Full
from harness.data import GpuAugment
from harness.models import (make_student, make_teacher, student_forward, teacher_forward_masked,
                            teacher_forward_keepmask, _attn_with_weights)
from harness.selector_api import Context, SelectorBase
from harness.train import train_step, Segments, HarnessViolation, to_keep_mask
from harness.validate import check_source, SelectorRejected, load_selector

torch.manual_seed(0)


@pytest.fixture(scope="module")
def models():
    s = make_student().eval()
    t = make_teacher(pretrained=False).eval()
    return s, t


def test_student_forward_matches_timm(models):
    s, _ = models
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        ref = s(x)
        out = student_forward(s, x, attn_layers=range(12), token_layers=[-1])
    assert torch.allclose(ref, out.logits, atol=1e-4)
    assert all(a.shape == (2, 3, 197, 197) for a in out.attn)
    assert torch.allclose(out.attn[-1].sum(-1), torch.ones(2, 3, 197), atol=1e-5)


def official_maskedkd_teacher(model, x, len_keep):
    """Reimplementation of effl-lab/MaskedKD models_teacher.forward_features (commit 96d052d)."""
    B = x.shape[0]
    x = model.patch_embed(x)
    cls = model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls, x), dim=1)
    x = x + model.pos_embed
    cls_t, x = x[:, :1], x[:, 1:]
    x = torch.gather(x, 1, len_keep[..., None].repeat(1, 1, x.shape[-1]))
    x = torch.cat([cls_t, x], 1)
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    return model.head(x[:, 0])


def test_maskedkd_selection_and_teacher_match_official(models):
    s, t = models
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = student_forward(s, x, attn_layers=[-1])
        # official losses.py: attn.mean(dim=1)[:, 0, 1:] then topk(len_num_keep)
        official_idx = out.attn[-1].mean(dim=1)[:, 0, 1:].topk(98).indices
        ctx = Context(images=x, student=s, out=out, budget=98, num_patches=196, epoch=0, progress=0.0)
        keep = MaskedKD().select(ctx)
        assert keep.sum(1).tolist() == [98, 98]
        assert (keep.gather(1, official_idx)).all()
        ref = official_maskedkd_teacher(t, x, official_idx)
        ours = teacher_forward_keepmask(t, x, keep)   # different token order, same set -> same CLS output
        assert torch.allclose(ref, ours, atol=1e-4)


def test_full_keep_equals_plain_teacher(models):
    _, t = models
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        keep = torch.ones(2, 196, dtype=torch.bool)
        assert torch.allclose(t(x), teacher_forward_keepmask(t, x, keep), atol=1e-4)


def test_variable_k_groups(models):
    _, t = models
    x = torch.randn(3, 3, 224, 224)
    keep = torch.zeros(3, 196, dtype=torch.bool)
    keep[0, :10] = True
    keep[1, :50] = True
    keep[2, 5:15] = True
    with torch.no_grad():
        out = teacher_forward_keepmask(t, x, keep)
        one = teacher_forward_masked(t, x[1:2], torch.arange(50)[None])
    assert torch.allclose(out[1:2], one, atol=1e-4)


def test_augment_shapes():
    aug = GpuAugment(torch.device("cpu"), C.NUM_CLASSES[C.STAGE1])
    x = torch.randint(0, 256, (8, 3, 64, 64), dtype=torch.uint8)
    y = torch.arange(8) % C.NUM_CLASSES[C.STAGE1]
    xa, ya = aug(x, y)
    assert xa.shape == (8, 3, 224, 224) and ya.shape == (8, C.NUM_CLASSES[C.STAGE1])
    assert torch.isfinite(xa).all()
    assert torch.allclose(ya.sum(1), torch.ones(8), atol=1e-5)


GOOD = '''
import torch
from harness.selector_api import SelectorBase, topk_mask
META = {"method": "Attention Rollout", "paper": "Abnar & Zuidema 2020", "why": "uses all layers",
        "vs_maskedkd": "multi-layer instead of last block", "prior_art": "none identical known",
        "parent": "none"}
class Selector(SelectorBase):
    NEEDS = {"attn": list(range(12)), "tokens": []}
    def select(self, ctx):
        r = None
        for a in ctx.out.attn:
            a = a.detach().float().mean(1)
            a = a + torch.eye(a.shape[-1])
            a = a / a.sum(-1, keepdim=True)
            r = a if r is None else a @ r
        return topk_mask(r[:, 0, 1:], int(ctx.budget))
'''


def test_validator_accepts_good_and_rejects_cheats():
    check_source(GOOD, "good")
    cheats = [
        GOOD.replace("import torch\n", "import torch\nimport os\n"),
        GOOD.replace("META = {", "META0 = {"),
        GOOD.replace('"why": "uses all layers"', '"why": ""'),
        GOOD.replace("r = None", "r = torch.load('x')"),
        GOOD.replace("r = None", "r = open('x')"),
        GOOD.replace("r = None", "r = ctx.teacher"),
        GOOD.replace("r = None", "ctx.out.logits.sum().backward(); r = None"),
        GOOD.replace("from harness.selector_api", "from harness.config import ROOT\nfrom harness.selector_api"),
        GOOD.replace("r = None", "r = torch.sys"),
    ]
    for i, src in enumerate(cheats):
        with pytest.raises(SelectorRejected):
            check_source(src, f"cheat{i}")


def _step(method, selector, keep, models):
    s, t = models
    s.train()
    opt = torch.optim.AdamW(s.parameters(), lr=1e-4)
    aug = GpuAugment(torch.device("cpu"), C.NUM_CLASSES[C.STAGE1])
    x, y = aug(torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8), torch.arange(4) % C.NUM_CLASSES[C.STAGE1])
    segs = Segments(True)
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        loss, kept = train_step(s, t, selector, method, keep, x, y, opt, SoftTargetCrossEntropy(), 0, 0.0, segs)
    segs.flops["total"] = fc.get_total_flops()
    return loss, kept, segs.flops


def test_train_step_flops_and_rollout(models):
    sel, meta, sha = load_selector("good", src=GOOD)
    loss, kept, fl = _step("good", sel, 25, models)
    assert kept == 25 and torch.isfinite(torch.tensor(loss))
    assert fl["total"] > fl["teacher"] > 0 and fl["selector"] > 0
    _, _, fl98 = _step("maskedkd", MaskedKD(), 98, models)
    _, _, fl196 = _step("full", Full(), 196, models)
    assert fl98["teacher"] < 0.6 * fl196["teacher"]


class OverBudget(SelectorBase):
    def select(self, ctx):
        return torch.ones(ctx.images.shape[0], 196, dtype=torch.bool)


class Backward(SelectorBase):
    def select(self, ctx):
        ctx.out.logits.float().sum().backward(retain_graph=True)
        return torch.zeros(ctx.images.shape[0], 196, dtype=torch.bool)


class HiddenTeacher(SelectorBase):
    """A cheat the static check would block; here we check its cost still shows up in selector FLOPs."""
    t = None

    def select(self, ctx):
        with torch.no_grad():
            HiddenTeacher.t(ctx.images)
        return torch.zeros(ctx.images.shape[0], 196, dtype=torch.bool)


def test_runtime_guards(models):
    with pytest.raises(HarnessViolation):
        _step("x", OverBudget(), 49, models)
    with pytest.raises(HarnessViolation):
        _step("x", Backward(), 49, models)
    HiddenTeacher.t = models[1]
    _, _, fl = _step("x", HiddenTeacher(), 0, models)
    _, _, fl_full = _step("full", Full(), 196, models)
    assert fl["selector"] >= 0.9 * fl_full["teacher"]   # hidden full teacher forward is billed


def test_to_keep_mask():
    m = to_keep_mask(torch.tensor([[0, 5, 7]]), 1)
    assert m.sum() == 3 and m[0, 5]
    with pytest.raises(HarnessViolation):
        to_keep_mask(torch.tensor([[0, 300]]), 1)


# ---- analysis channel, oracle, seeds ----

GOOD_ANALYSIS = '''
import torch
from harness.analysis_api import AnalysisAPI
META = {"question": "does last-layer attention match teacher attention?", "method": "attention agreement",
        "paper": "Abnar & Zuidema 2020"}
def run(api):
    agree = []
    for x, y in api.val_batches(batch=4, limit=8):
        s = api.student_forward(api.teacher, x, attn_layers=[-1]).attn[-1]
        agree.append(float(s.mean()))
    api.write_json("summary", {"n": len(agree), "mean": sum(agree) / len(agree)})
'''


def test_analysis_validator():
    check_source(GOOD_ANALYSIS, "ga", kind="analysis")
    bad = [
        GOOD_ANALYSIS.replace("api.write_json", "api._out.parent.joinpath('x').write_text"),
        GOOD_ANALYSIS.replace("import torch\n", "import torch\nimport os\n"),
        GOOD_ANALYSIS.replace('"question"', '"q"'),
        GOOD_ANALYSIS.replace("def run(api):", "def go(api):"),
        GOOD_ANALYSIS.replace("agree = []", "agree = []; fig.savefig('/tmp/x.png')"),
    ]
    for i, src in enumerate(bad):
        with pytest.raises(SelectorRejected):
            check_source(src, f"bad{i}", kind="analysis")
    # a selector may still not touch the teacher even though analyses can
    with pytest.raises(SelectorRejected):
        check_source(GOOD.replace("r = None", "r = ctx.teacher"), "sel")


def test_analysis_runs(tmp_path, monkeypatch):
    from harness.analysis_api import AnalysisAPI
    from harness.validate import load_analysis
    run, meta = load_analysis("ga", GOOD_ANALYSIS)
    api = AnalysisAPI(tmp_path, torch.device("cpu"), make_teacher(pretrained=False).eval())
    with torch.no_grad():
        run(api)
    assert json.loads((tmp_path / "summary.json").read_text())["n"] == 2


def test_oracle_step_bills_teacher_to_selector(models):
    _, _, fl = _step("oracle", None, 25, models)
    _, _, fl_full = _step("full", Full(), 196, models)
    assert fl["selector"] >= 0.9 * fl_full["teacher"]


def test_submit_seed_and_oracle_rules(tmp_path, monkeypatch):
    from harness import submit
    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    monkeypatch.setitem(C.TIER_EPOCHS["waterbirds"], "final", 2)
    monkeypatch.setitem(C.TIER_EPOCHS["waterbirds"], "screen", 1)
    with pytest.raises(SystemExit):
        submit.make_job("maskedkd", 49, 0, "final")          # screen seed never in final
    submit.make_job("maskedkd", 49, 1, "final")
    with pytest.raises(SystemExit):
        submit.make_job("oracle", 25, 0, "final")            # analysis-only
    submit.make_job("oracle", 25, 0, "screen")
    ids = submit.baselines()
    assert len(ids) == 2 * len(C.NOISE_SEEDS) + 6 * len(C.BASELINE_KEEPS) + 1   # maskedkd/rollout/random/grid/center/tome + oracle


def test_report_noise_and_discriminative():
    from harness.report import noise, discriminative, tries
    runs = [{"tier": "screen", "method": m, "keep": k, "seed": s, "val_acc_last": v, "selector_sha": "b", "sel_args": {}}
            for m, k, s, v in [("maskedkd", 98, 0, 70.0), ("maskedkd", 98, 1, 70.6), ("full", 196, 0, 71.0),
                               ("full", 196, 1, 70.4), ("random", 25, 0, 60.0), ("maskedkd", 25, 0, 64.0)]]
    sigma, thr, _ = noise(runs)
    assert sigma is not None and thr >= C.PROMOTE_MIN
    assert any("K=25: maskedkd 64.00" in line and "random 60.00" in line for line in discriminative(runs))
    assert tries(runs)[0].startswith("Total agent screen configs: 0")


# ---- ToMe baseline, stage 2, smoke, figures ----

def test_tome_r0_equals_plain_teacher_and_merging_cuts_cost(models):
    from harness.tome import teacher_forward_tome, r_for_budget, token_schedule
    from torch.utils.flop_counter import FlopCounterMode
    _, t = models
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(t(x), teacher_forward_tome(t, x, 0), atol=1e-4)
        with FlopCounterMode(display=False) as f0:
            teacher_forward_tome(t, x, 0)
        with FlopCounterMode(display=False) as f1:
            teacher_forward_tome(t, x, r_for_budget(49))
    assert f1.get_total_flops() < 0.6 * f0.get_total_flops()
    assert r_for_budget(98) <= r_for_budget(49) <= r_for_budget(25)
    assert min(token_schedule(r_for_budget(25))) >= 2


def test_tome_train_step(models):
    loss, kept, fl = _step("tome", None, 49, models)
    assert kept == 49.0 and fl["teacher"] > 0


def _fake_runs(tmp_path, monkeypatch, recs):
    runs = tmp_path / "runs"
    monkeypatch.setattr(C, "RUNS", runs)
    monkeypatch.setattr(C, "ROOT", tmp_path)
    for i, r in enumerate(recs):
        d = runs / f"r{i}"
        d.mkdir(parents=True)
        base = {"id": f"r{i}", "status": "done", "selector_sha": "builtin", "sel_args": {}, "flops": {"total_gflops_per_img": 10.0},
                "img_per_s_mean": 500.0, "dataset": C.STAGE1}
        (d / "metrics.json").write_text(json.dumps({**base, **r}))


def test_stage2_rules(tmp_path, monkeypatch):
    from harness import submit
    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    monkeypatch.setitem(C.TIER_EPOCHS[C.STAGE2], "final", 2)
    (tmp_path / "selectors").mkdir()
    monkeypatch.setattr("harness.validate.SELECTORS_DIR", tmp_path / "selectors")
    (tmp_path / "selectors" / "good.py").write_text(GOOD)
    import hashlib
    sha = hashlib.sha256(GOOD.encode()).hexdigest()
    _fake_runs(tmp_path, monkeypatch, [])
    submit.make_job("ce", 0, 1, "final", dataset=C.STAGE2)                 # baselines always allowed
    with pytest.raises(SystemExit, match="gate"):
        submit.make_job("good", 25, 1, "final", dataset=C.STAGE2)         # stage 2 closed before its gate
    monkeypatch.setattr(C, "STAGE2_EXPLORE", False)
    gate = [{"method": m, "keep": k, "seed": 1, "tier": "final", "val_acc_last": v, "dataset": C.STAGE2}
            for m, k, v in (("ce", 0, 70.0), ("full", 196, 75.0), ("maskedkd", 98, 74.8))]
    cifar = [{"method": "good", "keep": 25, "seed": s, "tier": "final", "val_acc_last": 70.0, "selector_sha": sha}
             for s in (1, 2, 3)]
    import shutil
    shutil.rmtree(tmp_path / "runs", ignore_errors=True)
    _fake_runs(tmp_path, monkeypatch, gate + cifar)
    with pytest.raises(SystemExit, match="audit"):
        submit.make_job("good", 25, 1, "final", dataset=C.STAGE2)         # no audit PASS yet
    with monkeypatch.context() as m:                                     # a human let it on before the audit
        m.setattr(C, "STAGE2_PREAUDIT", {"good": "test"})
        assert submit.make_job("good", 25, 2, "final", dataset=C.STAGE2)["stage2"] == "stage2_untouched"
    (tmp_path / "AUDIT.md").write_text("# AUDIT\n\n## x audit by claude\nclaim good K=25\nVERDICT: PASS\n")
    job = submit.make_job("good", 25, 1, "final", dataset=C.STAGE2)
    assert job["stage2"] == "stage2_untouched" and job["priority"] <= C.STAGE2_PRIORITY


def test_stage2_smoke_run(tmp_path, monkeypatch):
    from harness import submit, train
    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    monkeypatch.setattr(C, "RUNS", tmp_path / "runs")
    monkeypatch.setitem(C.TIER_MAX_STEPS, "smoke", 2)
    monkeypatch.setattr(C, "BATCH", 8)
    job = submit.make_job("maskedkd", 49, 0, "smoke", dataset=C.STAGE2, force=True)
    jp = tmp_path / "job.json"
    jp.write_text(json.dumps(job))
    assert train.run(str(jp)) == 0
    m = json.loads((tmp_path / "runs" / job["id"] / "metrics.json").read_text())
    assert m["dataset"] == C.STAGE2 and m["mean_kept"] == 49.0 and m["img_per_s_mean"] > 0


def test_figures(tmp_path, monkeypatch):
    recs = [{"method": m, "keep": k, "seed": s, "tier": "screen", "val_acc_last": v}
            for m, k, s, v in (("ce", 0, 0, 60.0), ("full", 196, 0, 70.0), ("maskedkd", 98, 0, 69.8),
                               ("maskedkd", 49, 0, 68.5), ("random", 49, 0, 66.0), ("tome", 49, 0, 67.0))]
    _fake_runs(tmp_path, monkeypatch, recs)
    from harness import figures
    monkeypatch.setattr("sys.argv", ["figures"])
    figures.main()
    assert (tmp_path / "figures" / f"{C.STAGE1}_screen_tokens.png").exists()
    assert "maskedkd" in (tmp_path / "figures" / f"{C.STAGE1}_screen.csv").read_text()


def test_boot_rules(tmp_path, monkeypatch):
    from harness import boot
    monkeypatch.setattr(C, "STATE", tmp_path)
    monkeypatch.setattr(C, "EPOCHS_FILE", tmp_path / "epochs.json")
    (tmp_path / "bench.json").write_text(json.dumps({"recommend": {"parallel_jobs": 3}}))
    boot.jobs_from_bench()
    assert json.loads((tmp_path / "epochs.json").read_text())["max_jobs"] == 3
    recs = [{"method": m, "keep": k, "seed": 0, "tier": "screen", "val_acc_last": v}
            for m, k, v in (("full", 196, 90.0), ("maskedkd", 49, 89.5), ("maskedkd", 29, 85.0), ("maskedkd", 12, 87.0))]
    _fake_runs(tmp_path, monkeypatch, recs)
    import io, contextlib
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        boot.pick_keeps()
    assert out.getvalue().strip() == "29,12"      # where MaskedKD falls furthest below Full KD


# ---- lessons from R2E: spatial baselines, parent rule, final-first, significant-only table ----

def _ctx(models, k):
    s, _ = models
    x = torch.randn(3, 3, 224, 224)
    return Context(images=x, student=s, out=None, budget=k, num_patches=196, epoch=0, progress=0.0)


def test_grid_and_center_baselines(models):
    from harness.baselines import Grid, Center
    for cls in (Grid, Center):
        for k in (98, 49, 25, 12):
            m = cls().select(_ctx(models, k))
            assert m.shape == (3, 196) and (m.sum(1) == k).all() and (m[0] == m[1]).all()
    c = Center().select(_ctx(models, 4))[0].nonzero().flatten().tolist()
    assert sorted(c) == [90, 91, 104, 105]          # the 2x2 block at the centre of the 14x14 grid
    g = Grid().select(_ctx(models, 49))[0].nonzero().flatten()
    assert g[0] == 0 and g[-1] == 195


def _write_sel(root, name, parent):
    (root / "selectors").mkdir(exist_ok=True)
    (root / "selectors" / f"{name}.py").write_text(GOOD.replace('"parent": "none"', f'"parent": "{parent}"'))


def test_parent_rule_final_priority_and_refs(tmp_path, monkeypatch):
    from harness import submit, validate
    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(validate, "SELECTORS_DIR", tmp_path / "selectors")
    monkeypatch.setitem(C.TIER_EPOCHS["waterbirds"], "final", 2)
    monkeypatch.setitem(C.TIER_EPOCHS["waterbirds"], "screen", 1)
    with pytest.raises(SelectorRejected):
        check_source(GOOD.replace(',\n        "parent": "none"', ""), "noparent")
    _write_sel(tmp_path, "orphan", "missing_sel")
    with pytest.raises(SystemExit):
        submit.make_job("orphan", 25, 0, "screen")
    _write_sel(tmp_path, "base_sel", "none")
    _write_sel(tmp_path, "child_sel", "base_sel")
    assert submit.make_job("child_sel", 25, 0, "screen")["priority"] == 0
    assert submit.make_job("child_sel", 25, 1, "final")["priority"] == C.FINAL_PRIORITY
    ids = submit.final_refs("child_sel", 25, 1)
    queued = [i for i in ids if not i.startswith("skip")]
    for m in list(C.CLAIM_REFS) + ["full", "base_sel"]:
        assert any(f"_{m}_k" in i for i in queued), (m, ids)
    assert all(i.startswith("skip") for i in submit.final_refs("child_sel", 25, 1))   # second call: all duplicates


def _run(m, k, s, v, tier="final", parent=None, gf=10.0):
    return {"id": f"{tier}_{m}_k{k}_s{s}", "tier": tier, "method": m, "keep": k, "seed": s, "val_acc_last": v,
            "flops": {"total_gflops_per_img": gf}, "selector_sha": "x", "sel_args": {},
            "selector_meta": {"parent": parent} if parent else {"method": f"builtin:{m}"}}


def test_core_significant_and_parent_check():
    from harness.report import core, parent_check
    runs = []
    for s, (a, mk, tm) in zip((1, 2, 3), ((71.0, 69.0, 70.9), (71.2, 69.1, 71.1), (70.9, 68.8, 71.0))):
        runs += [_run("mysel", 25, s, a, parent="maskedkd", gf=9.0), _run("maskedkd", 25, s, mk), _run("tome", 25, s, tm),
                 _run("full", 196, s, 71.5, gf=20.0)]
    lines, sig = core(runs)
    assert any("vs maskedkd@25" in l and "WIN" in l for l in sig)          # +2 every seed
    assert not any("vs tome@25" in l for l in sig)                          # within noise: not significant
    assert any("vs grid@25 [acc]: missing" in l for l in lines)             # missing references are shown, not hidden
    assert any("vs full@196" in l and "LOSS" in l for l in sig)
    scr = [_run("child", 25, 0, 60.2, "screen", parent="mysel"), _run("mysel", 25, 0, 60.0, "screen", parent="maskedkd"),
           _run("maskedkd", 25, 0, 59.0, "screen")]
    pc = parent_check(scr, 0.5)
    assert any(l.startswith("- child@25") and "UNPROVEN" in l for l in pc)
    assert any(l.startswith("- mysel@25") and "helps" in l for l in pc)


def test_report_cli_accepts_dataset(tmp_path, monkeypatch):
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "harness.report", "--gate", "--dataset", "cub"],
                       capture_output=True, text=True, cwd=C.ROOT)
    # the verdict depends on the runs on this machine (incomplete before the gate, a "[cub screen ..." line after it)
    assert "unrecognized arguments" not in r.stderr and ("gate incomplete" in r.stdout or "[cub " in r.stdout)


def test_timm_can_reach_hf_hub():
    """timm silently disables pretrained downloads when its huggingface_hub imports fail (version skew);
    the teacher fine-tune then dies on the server. Catch that on CPU."""
    from timm.models._hub import _has_hf_hub
    assert _has_hf_hub, "pin huggingface_hub to a version timm can import (requirements.txt)"


def test_gate_mean_over_seeds_and_epoch_filter(tmp_path, monkeypatch):
    from harness import report, submit
    from harness.report import gate
    g = lambda m, k, s, v: {"id": f"{m}{s}", "tier": "screen", "method": m, "keep": k, "seed": s, "val_acc_last": v}
    runs = [g("ce", 0, 0, 60.0), g("full", 196, 0, 60.8), g("maskedkd", 98, 0, 60.7)]
    assert gate(runs)[0] is False                              # seed 0 alone: KD only +0.8
    runs += [g("ce", 0, 1, 59.6), g("full", 196, 1, 61.0), g("maskedkd", 98, 1, 60.9),
             g("ce", 0, 2, 59.4), g("full", 196, 2, 61.2), g("maskedkd", 98, 2, 61.0)]
    ok, msg = gate(runs)
    assert ok and "seeds [0, 1, 2]" in msg                     # mean of three seeds: +1.33
    # runs at an old epoch budget drop out once the budget changes
    monkeypatch.setattr(C, "RUNS", tmp_path / "runs")
    monkeypatch.setitem(C.TIER_EPOCHS[C.STAGE1], "screen", 56)
    for i, (ep, v) in enumerate(((28, 60.0), (56, 62.0))):
        d = C.RUNS / f"r{i}"; d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"id": f"r{i}", "status": "done", "tier": "screen", "method": "ce",
                                                     "keep": 0, "seed": 0, "epochs": ep, "val_acc_last": v,
                                                     "dataset": C.STAGE1}))
    assert [r["val_acc_last"] for r in report.load_runs(C.STAGE1)] == [62.0]
    # the gate's extra seeds are reused by --baselines instead of failing as duplicates
    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    ids = submit.gate_jobs(seeds=[4, 5])
    assert submit.gate_jobs(seeds=[4, 5]) == ids
    with pytest.raises(SystemExit):
        submit.gate_jobs(seeds=[1])        # final seeds never at screen
    base = submit.baselines()
    assert all(i in base for i in ids if "_ce_" not in i)


def test_tome_cost_ratio_flags_unmatched_budgets():
    from harness.tome import cost_ratio
    assert cost_ratio(98) <= 1.0 and cost_ratio(49) <= 1.0     # matched
    assert cost_ratio(12) > 1.05                               # r caps out: ToMe uses more teacher compute


def test_evidence_bundle_attaches_cited_runs(tmp_path, monkeypatch):
    from harness import evidence
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    rid = "final_mysel_k25_s1_1030_120000_abcd"
    (tmp_path / "runs" / rid).mkdir(parents=True)
    (tmp_path / "runs" / rid / "metrics.json").write_text(json.dumps(
        {"id": rid, "val_acc_last": 80.1, "val_wga_last": 61.5, "selector_sha": "f" * 64, "teacher_sha": "t" * 64,
         "harness_digest": "d" * 12, "history": [{"epoch": 1, "loss": 2.0}, {"epoch": 2, "val_acc": 80.1}]}))
    (tmp_path / "q" / "done").mkdir(parents=True)
    (tmp_path / "q" / "done" / f"{rid}.json").write_text(json.dumps(
        {"method": "mysel", "selector_sha": "f" * 64, "selector_src": "META = {'parent': 'none'}"}))
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "AUDIT_REQUEST.md").write_text(f"claim: mysel wins, runs {rid}")
    text = evidence.audit()
    assert f"runs/{rid}/metrics.json" in text and '"val_acc": 80.1' in text and '"loss"' not in text
    assert "selector source mysel" in text and "META = {'parent': 'none'}" in text
    summ = text.split("cited runs: summary and identity")[1].split("runs/")[0]   # the summary comes before the metrics
    assert f"{rid} | mysel" in summ and "f" * 64 in summ and "| match |" in summ and "61.5" in summ
    assert "complete job history" in text and f"{rid} | done" in text and "listed 1 jobs of mysel" in text
    assert "selector_api" in evidence.ideas() or "missing: harness/selector_api.py" in evidence.ideas()


def test_tolerance_is_half_the_kd_gain():
    assert C.tolerance(83.50, 82.74) == pytest.approx(0.38)    # pretrained student: small KD gain
    assert C.tolerance(24.38, 21.10) == 1.0                     # large gain: capped at 1 %p
    assert C.tolerance(80.0, None) == 1.0 and C.tolerance(80.0, 80.5) == 0.0
    from harness.report import primary
    rows = [{"tier": "final", "method": m, "acc": a, "kept": k, "sha": "builtin", "args": "{}", "seeds": sd}
            for m, a, k, sd in (("full", 83.5, 196, [1, 2, 3]), ("ce", 82.7, 0, [1, 2, 3]), ("maskedkd", 83.2, 49, [1, 2, 3]),
                                ("maskedkd", 83.0, 25, [1, 2, 3]), ("rollout", 83.4, 12, [0]))]
    out = "\n".join(primary(rows))
    assert "tolerance 0.40" in out and "maskedkd : 49.0 (3 seeds)" in out    # K=25 (-0.5) is outside 0.40
    assert out.index("rollout") > out.index("screening only")                 # one seed: screening only


# ---- Waterbirds: worst-group accuracy, rollout baseline ----

def test_waterbirds_run_records_worst_group(tmp_path, monkeypatch):
    from harness import submit, train
    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    monkeypatch.setattr(C, "RUNS", tmp_path / "runs")
    monkeypatch.setitem(C.TIER_MAX_STEPS, "smoke", 2)
    monkeypatch.setattr(C, "BATCH", 8)
    job = submit.make_job("rollout", 29, 0, "smoke", force=True, dataset="waterbirds")
    jp = tmp_path / "job.json"
    jp.write_text(json.dumps(job))
    assert train.run(str(jp)) == 0
    m = json.loads((tmp_path / "runs" / job["id"] / "metrics.json").read_text())
    assert m["dataset"] == "waterbirds" and m["mean_kept"] == 29.0
    assert m["val_wga_last"] is not None and set(m["val_group_acc_last"]) <= {"0", "1", "2", "3"}
    assert m["val_wga_last"] == min(m["val_group_acc_last"].values())


def test_rollout_baseline(models):
    from harness.baselines import Rollout
    s, _ = models
    x = torch.randn(2, 3, 224, 224)
    out = student_forward(s, x, attn_layers=Rollout.NEEDS["attn"])
    ctx = Context(images=x, student=s, out=out, budget=29, num_patches=196, epoch=0, progress=0.0)
    m = Rollout().select(ctx)
    assert m.shape == (2, 196) and (m.sum(1) == 29).all()


def test_worst_group_is_diagnostic_only():
    from harness.report import gate, primary
    g = lambda m, k, s, a, w: {"id": f"{m}{s}", "tier": "screen", "method": m, "keep": k, "seed": s,
                               "val_acc_last": a, "val_wga_last": w, "dataset": C.STAGE1}
    runs = [g("ce", 0, 0, 90.0, 70.0), g("full", 196, 0, 90.5, 76.0), g("maskedkd", 98, 0, 90.4, 75.0)]
    ok, msg = gate(runs)
    assert not ok                                   # +0.5 accuracy fails, however much WGA improves
    rows = [{"tier": "screen", "method": m, "acc": a, "wga": w, "kept": k, "sha": "builtin", "args": "{}", "seeds": [0, 4, 5]}
            for m, a, w, k in (("full", 91.5, 76.0, 196), ("ce", 90.0, 70.0, 0), ("maskedkd", 91.2, 40.0, 29))]
    assert "maskedkd : 29.0" in "\n".join(primary(rows))   # low WGA does not disqualify


def test_lr_sweep_pick_and_superseded_runs(tmp_path, monkeypatch):
    from harness import boot, report, submit
    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    with pytest.raises(SystemExit):
        submit.make_job("maskedkd", 49, 0, "screen", lr=3e-4, by="lr_sweep")   # only CE may carry an lr
    with pytest.raises(SystemExit):
        submit.make_job("ce", 0, 0, "screen", lr=3e-4)                          # only the sweep may set it
    recs = [{"method": "ce", "keep": 0, "seed": 0, "tier": "screen", "val_acc_last": v, "lr_override": lr,
             "student_lr": lr} for lr, v in ((5e-5, 80.0), (1e-4, 81.0), (3e-4, 81.0))]
    recs.append({"method": "full", "keep": 196, "seed": 0, "tier": "screen", "val_acc_last": 82.0, "student_lr": 1e-4})
    _fake_runs(tmp_path, monkeypatch, recs)
    monkeypatch.setattr(C, "RECIPE_FILE", tmp_path / "recipe.json")
    boot.lr_pick(C.STAGE1)
    d = json.loads((tmp_path / "recipe.json").read_text())[C.STAGE1]
    assert d["student_lr"] == 1e-4 and d["swept"]                  # tie 81.0 -> the smaller lr
    runs = report.load_runs(C.STAGE1)
    assert [r["method"] for r in runs] == ["full"]                 # sweep runs never enter the tables
    (tmp_path / "recipe.json").write_text(json.dumps({C.STAGE1: {"student_lr": 5e-5}}))
    assert report.load_runs(C.STAGE1) == []                        # runs at a superseded lr drop out


# ---- cached teacher map (harness/teacher_cache.py) and lock adoption ----

def test_augment_moves_an_extra_map_without_changing_the_images():
    import random
    import numpy as np
    aug = GpuAugment(torch.device("cpu"), C.NUM_CLASSES[C.STAGE1])
    x0 = torch.randint(0, 256, (8, 3, 64, 64), dtype=torch.uint8)
    y0 = torch.arange(8) % C.NUM_CLASSES[C.STAGE1]
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    x1, y1 = aug(x0, y0)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    x2, y2, tm = aug(x0, y0, extra=torch.ones(8, 1, 14, 14))
    assert torch.equal(x1, x2) and torch.equal(y1, y2)          # same draws, same images and targets
    assert tm.shape == (x2.shape[0], 196) and float(tm.min()) >= 0 and float(tm.max()) <= 1 + 1e-5


def test_teacher_map_only_for_selectors_that_declare_it():
    from harness.train import uses_teacher_cache
    from harness.baselines import TeacherTopK, Rollout
    assert uses_teacher_cache(TeacherTopK(), None) and not uses_teacher_cache(Rollout(), None)
    src = '''
import torch
from harness.selector_api import SelectorBase, topk_mask
META = {"method": "mmm", "paper": "ppp", "why": "www", "vs_maskedkd": "vvv", "prior_art": "aaa", "parent": "none",
        "uses_teacher_cache": True}
class Selector(SelectorBase):
    def select(self, ctx):
        return topk_mask(ctx.teacher_map, int(ctx.budget))
'''
    meta = check_source(src, "tm")
    assert meta["uses_teacher_cache"] is True
    sel, meta, _ = load_selector("tm", src=src)
    assert uses_teacher_cache(sel, meta)


def test_teacher_topk_run_builds_cache_and_charges_it(tmp_path, monkeypatch):
    from harness import submit, train, teacher_cache
    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    monkeypatch.setattr(C, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(C, "STATE", tmp_path / "state")
    monkeypatch.setitem(C.TIER_MAX_STEPS, "smoke", 2)
    monkeypatch.setattr(C, "BATCH", 8)
    monkeypatch.setattr(C, "FLOP_WARMUP", 0)
    monkeypatch.setattr(C, "FLOP_STEPS", 1)
    job = submit.make_job("teacher_topk", 29, 0, "smoke", force=True)
    jp = tmp_path / "job.json"
    jp.write_text(json.dumps(job))
    assert train.run(str(jp)) == 0
    m = json.loads((tmp_path / "runs" / job["id"] / "metrics.json").read_text())
    assert m["mean_kept"] == 29.0 and teacher_cache.cache_path(C.STAGE1).exists()
    f = m["flops"]
    assert f["teacher_cache_gflops_per_img"] > 0
    assert f["selector_gflops_per_img"] >= f["teacher_cache_gflops_per_img"]
    assert teacher_cache.amortized_gflops(35.0, 1000, 100, 7, 128) == pytest.approx(35.0 * 1000 / (100 * 7 * 128))


def test_lock_adopts_only_what_upstream_has(tmp_path, monkeypatch):
    from harness import lock
    monkeypatch.setattr(lock, "LOCK_FILE", tmp_path / "lock")
    calls = {"diff_up": 1}
    def fake_git(*args):
        if args[0] == "rev-parse":
            return 0, "origin/main\n"
        if args[:2] == ("diff", "--quiet"):
            return (calls["diff_up"] if args[2] != "HEAD" else 0), ""
        return 0, ""
    monkeypatch.setattr(lock, "_git", fake_git)
    ok, why = lock.adopt_upstream()
    assert not ok and "differ" in why                       # a local commit or edit is never adopted
    calls["diff_up"] = 0
    class R:
        returncode, stdout, stderr = 0, "ok", ""
    monkeypatch.setattr(lock.subprocess, "run", lambda *a, **k: R())
    ok, why = lock.adopt_upstream()
    assert ok and (tmp_path / "lock").read_text().strip() == lock.harness_digest()


def test_gate_kd_only_ignores_the_maskedkd_clause():
    from harness.report import gate
    g = lambda m, k, s, a: {"id": f"{m}{s}", "tier": "screen", "method": m, "keep": k, "seed": s,
                            "val_acc_last": a, "dataset": C.STAGE1}
    runs = [g("ce", 0, 0, 78.7), g("full", 196, 0, 82.2), g("maskedkd", 98, 0, 81.0)]
    assert gate(runs)[0] is False and gate(runs, kd_only=True)[0] is True
    runs = [g("ce", 0, 0, 82.0), g("full", 196, 0, 82.2), g("maskedkd", 98, 0, 82.2)]
    assert gate(runs, kd_only=True)[0] is False          # no KD gain still stops


# --- per-image teacher budgets (QUEUE 2026-11-01) ---

def test_allocate_and_keep_by_counts():
    from harness.selector_api import allocate, keep_by_counts
    w = torch.tensor([1.0, 1.0, 2.0, 100.0])
    k = allocate(w, 100, kmax=196)          # the capped image hands the rest back, 1:1:2
    assert k.tolist() == [51, 51, 102, 196]
    k = allocate(w, 29, kmax=196)
    assert k.sum() == 29 * 4 and k.argmax() == 3 and k[2] >= k[1]
    k = allocate(torch.zeros(5), 12.5, kmin=1)
    assert k.sum() <= 62 and k.min() >= 1
    m = keep_by_counts(torch.rand(3, 196), torch.tensor([0, 5, 196]))
    assert m.sum(1).tolist() == [0, 5, 196]


def test_per_image_budget_skips_teacher_and_matches_old_loss(models):
    from harness.baselines import DynKD, SalRatio, EntWeight, KDSched, RandAlloc, RandDyn
    # all-or-nothing: images without tokens get no teacher, so the teacher costs a fraction of Full KD
    _, kept_full, f_full = _step("full", Full(), 196, models)
    loss, kept, f = _step("dynkd", DynKD(), 98, models)
    assert kept <= 98 and torch.isfinite(torch.tensor(loss)) and f["teacher"] < 0.6 * f_full["teacher"]
    for sel in (SalRatio(), RandAlloc(), KDSched(), EntWeight(), RandDyn()):
        loss, kept, _ = _step("x", sel, 29, models)
        assert kept <= 29 + 1e-6 and torch.isfinite(torch.tensor(loss))
    # a fixed-K selector gives the same loss as the old batch formula
    s, t = models
    s.eval()                                 # no dropout/drop-path, so both forwards see the same student
    aug = GpuAugment(torch.device("cpu"), C.NUM_CLASSES[C.STAGE1])
    x, y = aug(torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8), torch.arange(4) % C.NUM_CLASSES[C.STAGE1])
    with torch.no_grad():
        out = student_forward(s, x, [-1])
        keep = MaskedKD().select(Context(images=x, student=s, out=out, budget=49.0, num_patches=196, epoch=0, progress=0.0))
        tl = teacher_forward_keepmask(t, x, keep)
        kd = F.kl_div(F.log_softmax(out.logits / C.TAU, 1), F.softmax(tl / C.TAU, 1), reduction="batchmean") * C.TAU ** 2
        old = (1 - C.ALPHA) * SoftTargetCrossEntropy()(out.logits, y) + C.ALPHA * kd
    opt = torch.optim.SGD(s.parameters(), lr=0.0)
    new, _ = train_step(s, t, MaskedKD(), "maskedkd", 49, x, y, opt, SoftTargetCrossEntropy(), 0, 0.0, Segments(False))
    assert abs(new - old.item()) < 1e-3


def test_feedback_reports_masked_teacher_correctness(models):
    from harness.baselines import RandDyn
    seen = {}

    class Spy(RandDyn):
        def observe(self, fb):
            seen["ok"], seen["kept"] = fb.teacher_ok.clone(), fb.kept.clone()
    _step("x", Spy(), 98, models)
    ok, kept = seen["ok"], seen["kept"]
    assert torch.isnan(ok[kept == 0]).all()                      # no teacher, no verdict
    assert set(ok[kept > 0].tolist()) <= {0.0, 1.0}


def test_per_image_budget_rules(models):
    class Greedy(SelectorBase):          # one batch above 2x the budget
        def select(self, ctx):
            return torch.ones(ctx.images.shape[0], 196, dtype=torch.bool)
    with pytest.raises(HarnessViolation):
        _step("g", Greedy(), 29, models)

    class Inflate(SelectorBase):         # KD weights may move signal, never add it
        NEEDS = {"attn": [-1], "tokens": []}
        def select(self, ctx):
            return MaskedKD().select(ctx), torch.full((ctx.images.shape[0],), 2.0)
    with pytest.raises(HarnessViolation):
        _step("i", Inflate(), 29, models)

    seen = {}
    class Watch(SelectorBase):           # observe() sees each image's KD loss, NaN where no teacher ran
        def select(self, ctx):
            keep = torch.zeros(ctx.images.shape[0], 196, dtype=torch.bool)
            keep[0, :58] = True
            return keep
        def observe(self, fb):
            seen["kd"], seen["kept"] = fb.kd.clone(), fb.kept.clone()
    _step("w", Watch(), 29, models)
    assert torch.isfinite(seen["kd"][0]) and torch.isnan(seen["kd"][1:]).all() and seen["kept"].tolist()[0] == 58


def test_per_image_run_measures_flops_over_the_run(tmp_path, monkeypatch):
    from harness import submit, train
    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    monkeypatch.setattr(C, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(C, "STATE", tmp_path / "state")
    monkeypatch.setitem(C.TIER_MAX_STEPS, "smoke", 4)
    monkeypatch.setitem(C.TIER_EPOCHS[C.STAGE1], "smoke", 2)
    monkeypatch.setattr(C, "BATCH", 8)
    monkeypatch.setattr(C, "FLOP_WARMUP", 0)
    monkeypatch.setattr(C, "FLOP_STEPS", 1)
    monkeypatch.setattr(C, "FLOP_FRACS", (0.0, 0.5))
    job = submit.make_job("kd_sched", 29, 0, "smoke", force=True)
    jp = tmp_path / "job.json"
    jp.write_text(json.dumps(job))
    assert train.run(str(jp)) == 0
    m = json.loads((tmp_path / "runs" / job["id"] / "metrics.json").read_text())
    assert m["status"] == "done" and m["mean_kept"] <= 29 + 1e-6 and m["flops"]["windows"] >= 1


def test_scheduler_reaps_jobs_it_lost_the_popen_of():
    import subprocess, time
    from harness.scheduler import job_rc
    p = subprocess.Popen(["sh", "-c", "exit 3"])
    pid = p.pid
    del p                                   # like after os.execv: only the pid is left
    for _ in range(50):
        rc = job_rc(pid)
        if rc is not None:
            break
        time.sleep(0.05)
    assert rc == 3
    q = subprocess.Popen(["sleep", "5"])
    assert job_rc(q.pid) is None
    q.kill(); q.wait()


def test_aircraft_split_and_banner_trim(tmp_path):
    """Aircraft: 90/10 per class of trainval (the CUB rule), variant names with spaces, 20-px banner cut off."""
    from PIL import Image
    from harness import prepare_data as P
    d = tmp_path / "fgvc-aircraft-2013b" / "data"
    (d / "images").mkdir(parents=True)
    variants = ["707-320", "Falcon 2000"]
    (d / "variants.txt").write_text("\n".join(variants) + "\n")
    tv = [(f"{i:07d}", variants[i % 2]) for i in range(40)]
    te = [(f"{100 + i:07d}", variants[i % 2]) for i in range(6)]
    (d / "images_variant_trainval.txt").write_text("".join(f"{i} {v}\n" for i, v in tv))
    (d / "images_variant_test.txt").write_text("".join(f"{i} {v}\n" for i, v in te))
    sp = P.splits_aircraft(tmp_path / "fgvc-aircraft-2013b")
    assert (len(sp["train"]), len(sp["val"]), len(sp["test"])) == (36, 4, 6)
    assert {y for _, y, _ in sp["val"]} == {0, 1} and not {p for p, _, _ in sp["train"]} & {p for p, _, _ in sp["val"]}
    assert sp["test"][1][1] == 1 and sp["test"][1][0].endswith("0000101.jpg")
    im = Image.new("RGB", (300, 220), (0, 0, 255))
    im.paste((255, 0, 0), (0, 200, 300, 220))     # the banner: bottom 20 rows
    im.save(d / "images" / "x.jpg")
    out = P._load((str(d / "images" / "x.jpg"), 224, C.TRIM_BOTTOM["aircraft"]))
    assert out.shape == (3, 224, 224) and out[0].max() < 60   # no red from the banner left
    assert C.EXPECTED_SIZES["aircraft"] == {"train": 5967, "val": 700, "test": 3333}


def test_digest_at_reproduces_the_lock_of_a_commit(tmp_path, monkeypatch):
    """An audit of a run made under an earlier lock recomputes that lock from the commit the run started at."""
    import subprocess as sp
    from harness import lock
    monkeypatch.setattr(C, "ROOT", tmp_path)
    g = lambda *a: sp.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)   # noqa: E731
    g("init", "-q")
    g("config", "user.email", "t@t"), g("config", "user.name", "t")
    (tmp_path / "harness" / "sub").mkdir(parents=True)
    (tmp_path / "harness" / "a.py").write_text("x = 1\n")
    (tmp_path / "harness" / "sub" / "b.py").write_text("y = 2\n")
    (tmp_path / "harness" / "notes.txt").write_text("not locked\n")
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "loop.sh").write_text("echo\n")
    g("add", "-A"), g("commit", "-qm", "one")
    first = lock.harness_digest()
    assert lock.digest_at("HEAD") == first
    (tmp_path / "harness" / "a.py").write_text("x = 3\n")
    g("commit", "-qam", "two")
    assert lock.digest_at("HEAD") == lock.harness_digest() != first
    assert lock.digest_at("HEAD~1") == first and lock.digest_at("nope") is None


def test_caltech256_split_and_holdout_rule(tmp_path, monkeypatch):
    """Caltech-256: 256 classes (clutter dropped), per class 80/10/10. Held-out: a selector runs there only as a final
    run of a source that already has final seeds 1-3 on CUB and Aircraft; baselines always."""
    import hashlib
    from harness import prepare_data as P, submit
    root = tmp_path / "256_ObjectCategories"
    for name, n in (("001.ak47", 20), ("002.bat", 11), ("257.clutter", 5)):
        (root / name).mkdir(parents=True)
        for i in range(n):
            (root / name / f"{i:04d}.jpg").write_bytes(b"x")
        (root / name / "notes.txt").write_text("not an image")
    sp = P.splits_caltech256(root)
    assert sum(map(len, sp.values())) == 31 and {y for _, y, _ in sp["train"]} == {0, 1}
    assert (len(sp["val"]), len(sp["test"])) == (3, 3)          # round(2.0)=2 + round(1.1)=1 each
    assert not {p for p, _, _ in sp["train"]} & {p for p, _, _ in sp["test"]}

    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    (tmp_path / "selectors").mkdir()
    monkeypatch.setattr("harness.validate.SELECTORS_DIR", tmp_path / "selectors")
    (tmp_path / "selectors" / "good.py").write_text(GOOD)
    sha = hashlib.sha256(GOOD.encode()).hexdigest()
    _fake_runs(tmp_path, monkeypatch, [])
    assert submit.make_job("ce", 0, 1, "final", dataset="caltech256")["stage2"] == "holdout_baseline"
    with pytest.raises(SystemExit, match="needs final seeds"):
        submit.make_job("good", 29, 1, "final", dataset="caltech256")
    with pytest.raises(SystemExit):
        submit.make_job("good", 29, 0, "screen", dataset="caltech256")
    import shutil
    shutil.rmtree(tmp_path / "runs", ignore_errors=True)
    _fake_runs(tmp_path, monkeypatch, [{"method": "good", "keep": 29, "seed": s, "tier": "final", "val_acc_last": 70.0,
                                        "selector_sha": sha, "dataset": ds, "epochs": C.TIER_EPOCHS[ds]["final"]}
                                       for ds in (C.STAGE1, C.STAGE2) for s in C.FINAL_SEEDS])
    assert submit.make_job("good", 29, 1, "final", dataset="caltech256")["stage2"] == "holdout"


def test_retired_dataset_takes_no_new_runs(tmp_path, monkeypatch):
    """A retired dataset (config.RETIRED) refuses screen/final submissions; smoke stays for the harness self-test."""
    from harness import submit
    monkeypatch.setattr(C, "RETIRED", ("waterbirds",))
    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    _fake_runs(tmp_path, monkeypatch, [])
    for tier, seed in (("screen", 0), ("final", 1)):
        with pytest.raises(SystemExit, match="retired"):
            submit.make_job("ce", 0, seed, tier, dataset="waterbirds")


def test_cub_lt_split_and_shared_teacher(tmp_path, monkeypatch):
    """CUB long-tail: tail classes keep LT_TAIL_FRACTION of their CUB train images (at least 1), val/test are CUB's,
    group = 1 for a tail class; the teacher is CUB's."""
    from harness import prepare_data as P
    from harness.train import teacher_paths
    root = tmp_path / "CUB_200_2011"
    (root / "images").mkdir(parents=True)
    rows, i = [], 1
    for c in range(200):
        for j in range(10):            # 10 per class: 8 train (CUB val takes 10 % -> 1), 2 test
            rows.append((i, f"{c:03d}/{j}.jpg", c + 1, 1 if j < 8 else 0)); i += 1
    (root / "images.txt").write_text("".join(f"{i} {p}\n" for i, p, _, _ in rows))
    (root / "image_class_labels.txt").write_text("".join(f"{i} {c}\n" for i, _, c, _ in rows))
    (root / "train_test_split.txt").write_text("".join(f"{i} {t}\n" for i, _, _, t in rows))
    cub, lt = P.splits_cub(root), P.splits_cub_lt(root)
    assert lt["val"] == [(p, y, int(y >= 100)) for p, y, _ in cub["val"]] and len(lt["test"]) == len(cub["test"])
    head = [it for it in lt["train"] if it[1] < 100]
    tail = [it for it in lt["train"] if it[1] >= 100]
    assert len(head) == 700 and len(tail) == 100 and all(g == 1 for _, _, g in tail)   # 7 per head, round(0.2*7)=1 per tail
    assert set(tail) <= {(p, y, 1) for p, y, _ in cub["train"]}
    assert teacher_paths("cub_lt") == teacher_paths("cub")


def test_frozen_maskedkd_equals_maskedkd_at_step_zero(models):
    """selectors/maskedkd_frozen.py: at its first call the snapshot is the live student, so its mask equals
    MaskedKD's (eval mode: no drop-path); after the live student changes, the frozen mask does not follow."""
    import importlib.util
    from harness.validate import check_source
    src = (C.ROOT / "selectors" / "maskedkd_frozen.py").read_text()
    check_source(src, "maskedkd_frozen")
    spec = importlib.util.spec_from_file_location("sel_frozen", C.ROOT / "selectors" / "maskedkd_frozen.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    s, _ = models
    s.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = student_forward(s, x, attn_layers=[-1])
        ctx = Context(images=x, student=s, out=out, budget=29, num_patches=196, epoch=0, progress=0.0)
        ref = MaskedKD().select(ctx)
        sel = mod.Selector()
        assert torch.equal(sel.select(ctx), ref)
        with torch.no_grad():
            s.blocks[-1].attn.qkv.weight.add_(torch.randn_like(s.blocks[-1].attn.qkv.weight))
        out2 = student_forward(s, x, attn_layers=[-1])
        ctx2 = Context(images=x, student=s, out=out2, budget=29, num_patches=196, epoch=1, progress=0.5)
        assert torch.equal(sel.select(ctx2), ref)                 # frozen: still the step-0 mask
        assert not torch.equal(MaskedKD().select(ctx2), ref)      # online: follows the changed student
    s.train()


def test_celeba_split_groups_and_caps(tmp_path, monkeypatch):
    """CelebA: official partition by image number, y = blond, group = 2*y + male, train = CELEBA_TRAIN_FRACTION of
    official train, val/test groups capped at CELEBA_EVAL_CAP with minority groups whole; a non-strict held-out
    dataset takes any method as a final run (robustness_check), the strict one keeps holdout_rule."""
    import hashlib
    from harness import prepare_data as P, submit
    root = tmp_path / "celeba"
    (root / "img").mkdir(parents=True)
    rows = []
    for n in range(1, 202600, 101):   # 2006 ids over the official numbering
        rows.append((f"{n:06d}.jpg", 1 if n % 3 == 0 else -1, 1 if n % 2 == 0 else -1))
    with open(root / "attrs.csv", "w", newline="") as fh:
        import csv as _csv
        w = _csv.writer(fh); w.writerow(["image_id", "blond", "male"]); w.writerows(rows)
    monkeypatch.setattr(C, "CELEBA_EVAL_CAP", 30)
    sp = P.splits_celeba(root)
    n_train_official = sum(1 for n in range(1, 202600, 101) if n <= 162770)
    assert len(sp["train"]) == round(C.CELEBA_TRAIN_FRACTION * n_train_official)
    for split in ("val", "test"):
        by_g = {g: sum(1 for _, _, gg in sp[split] if gg == g) for g in range(4)}
        assert max(by_g.values()) == 30 and min(by_g.values()) > 0
        assert all(int(Path(p).name.split(".")[0]) > 162770 for p, _, _ in sp[split])
    assert all(g == 2 * y + (int(Path(p).name.split(".")[0]) % 2 == 0) for p, y, g in sp["train"])
    assert {y for _, y, _ in sp["train"]} == {0, 1}

    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    (tmp_path / "selectors").mkdir()
    monkeypatch.setattr("harness.validate.SELECTORS_DIR", tmp_path / "selectors")
    (tmp_path / "selectors" / "good.py").write_text(GOOD)
    _fake_runs(tmp_path, monkeypatch, [])
    assert submit.make_job("good", 29, 1, "final", dataset="celeba")["stage2"] == "robustness_check"
    with pytest.raises(SystemExit, match="needs final seeds"):
        submit.make_job("good", 29, 1, "final", dataset="caltech256")


def test_layerroll_equals_the_references(models):
    """selectors/layerroll.py is the common form of the reference selectors: layers=[11] is MaskedKD, all 12 blocks
    is the rollout baseline, blocks 6..11 is selectors/lateroll.py, all bit-identical; a block set or a single head
    changes the mask; sel_args reach the selector through load_selector."""
    from harness.baselines import Rollout
    s, _ = models
    s.eval()
    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        out = student_forward(s, x, attn_layers=range(12))
    ctx = Context(images=x, student=s, out=out, budget=29, num_patches=196, epoch=0, progress=0.0)
    mk = lambda **kw: load_selector("layerroll", kwargs=kw)[0]   # noqa: E731
    one, allb, late = mk(layers=[11]), mk(), mk(layers=[6, 7, 8, 9, 10, 11])
    assert one.NEEDS["attn"] == [11] and allb.NEEDS["attn"] == list(range(12))
    assert torch.equal(one.select(ctx), MaskedKD().select(ctx))
    assert torch.equal(mk(layers=[-1]).select(ctx), MaskedKD().select(ctx))
    assert torch.equal(allb.select(ctx), Rollout().select(ctx))
    assert torch.equal(late.select(ctx), load_selector("lateroll")[0].select(ctx))
    assert torch.equal(mk(layers=[9]).select(ctx), load_selector("midattn", kwargs={"layer": 9})[0].select(ctx))
    masks = {name: mk(layers=ls).select(ctx) for name, ls in
             {"b1": [0, 1, 2], "b4": [9, 10, 11], "no_last": list(range(11)), "no_first": list(range(3, 12))}.items()}
    for m in masks.values():
        assert m.shape == (3, 196) and m.dtype == torch.bool and (m.sum(1) == 29).all()
    assert not torch.equal(masks["b1"], masks["b4"]) and not torch.equal(masks["no_last"], allb.select(ctx))
    h0 = mk(layers=[11], head=0).select(ctx)
    ref = torch.zeros(3, 196, dtype=torch.bool).scatter_(1, out.attn[11][:, 0, 0, 1:].topk(29, dim=1).indices, True)
    assert torch.equal(h0, ref) and not torch.equal(h0, one.select(ctx))
    with pytest.raises(ValueError):
        mk(layers=[])


def test_cifar_sets_share_one_split_and_stay_paired(tmp_path, monkeypatch):
    """CIFAR-10 / CIFAR-10-C: every set gets the same rows in every split, a corrupted set takes the severity block
    config.CIFAR_C_SEVERITY, labels that are not the clean test labels are refused, and the stored 32px arrays go
    through the training augmentation and the eval transform like any other dataset."""
    import numpy as np
    from harness import prepare_data as P, data as D
    raw = tmp_path / "data" / "raw"
    (raw / "cifar-10-batches-bin").mkdir(parents=True)
    (raw / "CIFAR-10-C").mkdir()
    n = 200
    rng = np.random.default_rng(0)
    y = np.arange(n) % 10
    x = rng.integers(0, 200, (n, 3, 32, 32), dtype=np.uint8)
    np.concatenate([y[:, None].astype(np.uint8), x.reshape(n, -1)], axis=1).tofile(raw / "cifar-10-batches-bin" / "test_batch.bin")
    blocks = [x.transpose(0, 2, 3, 1) + sev for sev in range(1, 6)]   # severity block s = clean + s
    np.save(raw / "CIFAR-10-C" / "gaussian_noise.npy", np.concatenate(blocks))
    np.save(raw / "CIFAR-10-C" / "labels.npy", np.tile(y, 5))

    clean, noisy = P.cifar_arrays("cifar10", raw), P.cifar_arrays("cifar10c_gaussian_noise", raw)
    assert {k: len(v[1]) for k, v in clean.items()} == {"train": 160, "val": 20, "test": 20}
    for split in ("train", "val", "test"):
        (xc, yc), (xn, yn) = clean[split], noisy[split]
        assert np.array_equal(yc, yn) and np.bincount(yc, minlength=10).tolist() == [len(yc) // 10] * 10
        assert ((xn.astype(int) - xc.astype(int)) == C.CIFAR_C_SEVERITY).all()   # same image, severity-5 block
    rows = P.cifar_split(y)
    assert sorted(rows["train"] + rows["val"] + rows["test"]) == list(range(n))

    np.save(raw / "CIFAR-10-C" / "labels.npy", np.tile(np.roll(y, 1), 5))
    with pytest.raises(SystemExit, match="pairing broken"):
        P.cifar_arrays("cifar10c_gaussian_noise", raw)

    monkeypatch.setattr(C, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setitem(C.EXPECTED_SIZES, "cifar10", {"train": 160, "val": 20, "test": 20})
    P.prepare("cifar10", workers=1)
    monkeypatch.delenv("MASKEDKD_FAKE_DATA")
    xtr, ytr = D.load_split("train", "cifar10")
    assert tuple(xtr.shape) == (160, 3, 32, 32) and torch.equal(ytr, torch.from_numpy(clean["train"][1]))
    assert torch.equal(xtr, torch.from_numpy(clean["train"][0])) and D.load_groups("val", "cifar10") is None
    xa, ya = GpuAugment(torch.device("cpu"), 10)(xtr[:8], ytr[:8])
    assert xa.shape == (8, 3, 224, 224) and ya.shape == (8, 10) and torch.isfinite(xa).all()
    assert D.normalize_resize(D.load_split("val", "cifar10")[0][:4]).shape == (4, 3, 224, 224)
    assert set(C.CIFAR_SETS) <= set(C.DATASETS) and all(C.TIER_EPOCHS[d]["final"] == 75 for d in C.CIFAR_SETS)


def test_surgical_layer_sweep_runs_and_grid_names_its_arms(tmp_path, monkeypatch):
    """analysis/surgical_layer_sweep.py passes the analysis validator and writes, per student and K, every arm with
    its fidelity, group accuracies and the paired difference to L11 (= MaskedKD, so that difference is 0 for L11);
    surgical.submit_grid plans the 11 default arms and surgical.arms maps a finished run back to its arm."""
    from harness.analysis_api import AnalysisAPI
    from harness.validate import analysis_source, load_analysis
    from surgical import arms as A
    from surgical.submit_grid import plan
    run, _ = load_analysis("surgical_layer_sweep", analysis_source("surgical_layer_sweep"))
    run.__globals__.update(K_LIST=(29,), NBOOT=20, BATCH=4)
    student = make_student("waterbirds", pretrained=False).eval()

    class SmallAPI(AnalysisAPI):
        def val_batches(self, batch=100, limit=None):
            return super().val_batches(batch=batch, limit=8)

        def val_groups(self):
            return super().val_groups()[:8]

        def runs(self):
            return {"r_ce": {"method": "ce", "tier": "final", "seed": 1, "dataset": "waterbirds"},
                    "r_other": {"method": "full", "tier": "final", "seed": 1, "dataset": "cub"}}

        def student(self, run_id, epoch=None):
            return student

    api = SmallAPI(tmp_path, torch.device("cpu"), make_teacher("waterbirds", pretrained=False).eval(), "waterbirds")
    run(api)
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["students"] == {"ce": "r_ce"} and s["n_val"] == 8
    block = s["results"]["ce"]["29"]
    assert len(block) == 12 + 12 * 3 + 4 + 7 + 1 and {"L11", "L0H2", "B1 0-2", "no last 0-10", "random"} <= set(block)
    assert block["L11"]["acc_minus_L11"] == 0 and block["L11"]["acc_minus_L11_ci"] == [0, 0]
    assert all(0 <= r["acc"] <= 100 and r["kl"] >= 0 and "worst_group_acc" in r for r in block.values())
    assert len(s["curves"]["ce"]["29"]["blocks"]) == 12 and s["curves"]["ce"]["29"]["measure"] == "worst_group_acc"

    rows = list(plan([1, 4], A.K, extra=False, only=None))
    assert len(rows) == 22 and {r[4] for r in rows if r[3] == 1} == {"final"} and {r[4] for r in rows if r[3] == 4} == {"screen"}
    assert ("CE", "ce", 0, 1, "final", {}) in rows and ("Full KD", "full", 196, 1, "final", {}) in rows
    assert A.arm_of("layerroll", {"layers": [0, 1, 2]}) == "B1 0-2" and A.arm_of("maskedkd", {}) == "last (MaskedKD)"
    assert A.arm_of("layerroll", {"layers": [5]}) is None and A.arm_of("lateroll", {}) == "late 6-11 (lateroll)"
    for name, (method, args) in {**A.SELECT, **A.EXTRA}.items():   # every selecting arm loads and keeps K tokens
        load_selector(method, kwargs=args)


def test_layerroll_trains_on_a_cifar_set_with_its_block_set(tmp_path, monkeypatch):
    """A layerroll job with sel_args runs end to end on a new CIFAR dataset (final seeds allowed, no stage rule) and
    gives the teacher as many tokens as MaskedKD at the same K (the block set changes which tokens, not how many)."""
    from harness import submit, train
    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    monkeypatch.setattr(C, "RUNS", tmp_path / "runs")
    monkeypatch.setitem(C.TIER_MAX_STEPS, "smoke", 2)
    monkeypatch.setattr(C, "BATCH", 8)
    assert submit.make_job("layerroll", 29, 1, "final", {"layers": [0, 1, 2]}, by="human",
                           dataset="cifar10c_gaussian_noise")["stage2"] is None
    ms = {}
    for method, args in (("layerroll", {"layers": [3, 4, 5]}), ("maskedkd", {})):
        job = submit.make_job(method, 29, 0, "smoke", args, force=True, dataset="cifar10c_gaussian_noise")
        jp = tmp_path / f"{method}.json"
        jp.write_text(json.dumps(job))
        assert train.run(str(jp)) == 0
        ms[method] = json.loads((tmp_path / "runs" / job["id"] / "metrics.json").read_text())
    m = ms["layerroll"]
    assert m["dataset"] == "cifar10c_gaussian_noise" and m["mean_kept"] == 29.0 and m["sel_args"] == {"layers": [3, 4, 5]}
    assert m["status"] == "done" and ms["maskedkd"]["mean_kept"] == 29.0   # same token count as MaskedKD at this K


def test_new_server_keeps_the_student_lr_and_reuses_only_its_own_teachers_runs(tmp_path, monkeypatch):
    """The teacher checkpoints are not in git. Fine-tuning a teacher again must not reset a student lr the recipe
    already has, and the surgical scripts reuse a finished KD run only if it was made with this server's teacher
    (a CE run has no teacher and is always reusable)."""
    from harness import finetune_teacher as F
    from harness.train import sha256_file
    from surgical import arms as A
    monkeypatch.setattr(C, "RECIPE_FILE", tmp_path / "recipe.json")
    monkeypatch.setattr(C, "STATE", tmp_path)
    monkeypatch.setattr(C, "STAGE0_OUTPUTS", tmp_path / "none")
    (tmp_path / "recipe.json").write_text(json.dumps({"cub": {"student_lr": 3e-4, "swept": True}}))
    F.write_recipe("cub")
    F.write_recipe("cifar10")
    d = json.loads((tmp_path / "recipe.json").read_text())
    assert d["cub"] == {"student_lr": 3e-4, "swept": True} and d["cifar10"]["student_lr"] == C.STUDENT_LR_DEFAULT

    assert A.teacher_sha("cifar10") is None                       # no checkpoint here
    (tmp_path / "teacher_cifar10.json").write_text(json.dumps({"sha256": "from another server"}))
    assert A.teacher_sha("cifar10") is None                       # a record without its checkpoint
    (tmp_path / "teacher_cifar10.pt").write_bytes(b"weights")
    assert A.teacher_sha("cifar10") is None                       # a checkpoint that is not the recorded one
    sha = sha256_file(tmp_path / "teacher_cifar10.pt")
    (tmp_path / "teacher_cifar10.json").write_text(json.dumps({"sha256": sha}))
    assert A.teacher_sha("cifar10") == sha
    assert A.usable({"method": "ce", "teacher_sha": None}, None) and A.usable({"method": "full", "teacher_sha": sha}, sha)
    assert not A.usable({"method": "full", "teacher_sha": "old"}, sha) and not A.usable({"method": "full", "teacher_sha": None}, None)


def test_autorun_gate_rule(tmp_path, monkeypatch):
    """surgical.autorun reads the gate only when CE and Full KD have all of seeds 1-3, with the plan's rule:
    Full KD - CE mean > 0 and at least 2 of 3 seeds > 0."""
    from surgical import autorun, table
    def runs(full_minus_ce):
        out = {"CE": {}, "Full KD": {}}
        for s, d in zip((1, 2, 3), full_minus_ce):
            out["CE"][s] = {"val_acc_last": 80.0, "flops": None}
            out["Full KD"][s] = {"val_acc_last": 80.0 + d, "flops": None}
        return out
    for diffs, want in (((1.0, 0.5, -0.2), True), ((2.0, -0.5, -0.4), False), ((-0.1, -0.2, 0.1), False)):
        monkeypatch.setattr(table, "arms_by_seed", lambda d, k, r=runs(diffs): r)
        held, text = autorun.gate_verdict("cifar10")
        assert held is want and "per seed" in text
    monkeypatch.setattr(table, "arms_by_seed", lambda d, k: {"CE": {1: {}, 2: {}}, "Full KD": {1: {}, 2: {}}})
    assert autorun.gate_verdict("cifar10") is None


def test_smoke_end_to_end_without_a_teacher_checkpoint(tmp_path, monkeypatch):
    """A teacher record from git without its checkpoint (a new server) must not fail the smoke run: it uses a
    random-init teacher, and the harness's own load_teacher is restored afterwards."""
    from harness import train
    from surgical import smoke
    monkeypatch.setattr(C, "STATE", tmp_path)
    (tmp_path / "teacher.json").write_text(json.dumps({"sha256": "from another server"}))
    monkeypatch.setitem(C.TIER_MAX_STEPS, "smoke", 2)
    monkeypatch.setattr(C, "BATCH", 8)
    monkeypatch.delenv("MASKEDKD_FAKE_DATA")
    monkeypatch.setattr(train, "load_split", lambda split, ds: (torch.randint(0, 256, (16, 3, 64, 64), dtype=torch.uint8),
                                                                torch.arange(16) % 2))
    monkeypatch.setattr(train, "load_groups", lambda split, ds: torch.arange(16) % 4)
    monkeypatch.setattr(train, "make_student", lambda ds: make_student(ds, pretrained=False))
    before = train.load_teacher
    r = smoke.end_to_end("waterbirds")
    assert r["status"] == "done" and r["teacher"].startswith("random-init") and train.load_teacher is before


def test_cut_downloads_are_resumed_or_fetched_again(tmp_path, monkeypatch):
    """A download goes to .part and is renamed only when curl succeeded; an archive left cut short by an older
    download is deleted and fetched again (2026-11-13: a killed CIFAR download was taken as complete)."""
    import io, tarfile as T, subprocess as SP
    from harness import prepare_data as P
    good = tmp_path / "src.tar.gz"
    with T.open(good, "w:gz") as t:
        data = os.urandom(3 << 20)   # incompressible: the archive stays over _fetch's 1 MiB floor
        info = T.TarInfo("d/f.bin"); info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    calls = []
    def fake_run(cmd, **kw):
        out = Path(cmd[cmd.index("-o") + 1]); calls.append(out.name)
        out.write_bytes(good.read_bytes())
        return SP.CompletedProcess(cmd, 0)
    monkeypatch.setattr(P.subprocess, "run", fake_run)
    arc = tmp_path / "raw" / "a.tar.gz"
    arc.parent.mkdir()
    arc.write_bytes(good.read_bytes()[: 1 << 21])   # cut short, but over the size floor: looks complete
    P._extract(["http://x"], arc, ["d/f.bin"], tmp_path / "raw")
    assert (tmp_path / "raw" / "d" / "f.bin").read_bytes() == data
    assert calls == ["a.tar.gz.part"] and arc.exists() and not (tmp_path / "raw" / "a.tar.gz.part").exists()


def test_twophase_config_rules():
    """harness/twophase.py: the config of a twophase job is its sel_args; teacher tokens per image (masked pass on
    every image + full passes on full_n of the batch) may not exceed the job's keep, and a config that pays for a
    pass it does not use, or has nothing to fit its map on, is refused."""
    from harness.twophase import parse
    ok = parse({"cv": "fit", "mask_keep": 12, "full_n": 11}, 29, batch=128)
    assert ok["selector"] == "maskedkd" and ok["rho"] == 1.0 and ok["tokens_per_image"] == 12 + 11 * 196 / 128
    assert parse({"cv": "student", "full_n": 18}, 29, batch=128)["selector"] is None
    assert parse({"cv": "label", "full_n": 32}, 49, batch=128)["tokens_per_image"] == 49.0
    assert parse({"cv": "masked", "mask_keep": 29}, 29, batch=128)["full_n"] == 0          # MaskedKD itself
    assert parse({"cv": "channel", "mask_keep": 25, "full_n": 2}, 29, batch=128)["rho"] == 0.0
    for bad in ({"cv": "fit", "mask_keep": 12, "full_n": 12},        # 30.4 tokens > 29
                {"cv": "student", "full_n": 19},                      # 29.1 tokens > 29
                {"cv": "masked", "full_n": 5},                        # no masked pass
                {"cv": "label", "mask_keep": 12, "full_n": 5},        # pays for a masked pass it does not use
                {"cv": "student"},                                     # no teacher at all
                {"cv": "recal", "mask_keep": 12},                     # nothing to fit the map on
                {"cv": "channel", "mask_keep": 12, "full_n": 2, "rho": 1.0},
                {"cv": "fit", "mask_keep": 12, "full_n": 11, "selector": "lateroll"},
                {"cv": "student", "full_n": 1, "selector": "maskedkd"},
                {"cv": "student", "full_n": 1, "rho": 1.5}, {"cv": "student", "full_n": True},
                {"cv": "nope", "full_n": 1}, {"cv": "student", "full_n": 1, "extra": 1}, None):
        with pytest.raises(ValueError):
            parse(bad, 29, batch=128)


def test_twophase_step_is_unbiased_for_full_kd_whatever_the_cheap_answer():
    """Over every equally likely draw of the m images that get the full teacher, the mean gradient of the twophase
    KD loss in the student's logits is the full-KD gradient softmax(z) - T(x), for any cheap answer h (not on the
    simplex either); with rho < 1 it is not. nnls finds the non-negative mix and drops a regressor that would need
    a negative weight."""
    import itertools
    from harness.twophase import kd_loss, nnls, target
    g = torch.Generator().manual_seed(0)
    B, n_cls, m = 6, 5, 2
    u = torch.randn(B, n_cls, generator=g, requires_grad=True)
    pt = torch.randn(B, n_cls, generator=g).softmax(1)
    want = (u.detach().softmax(1) - pt) / B
    for h in (torch.randn(B, n_cls, generator=g).softmax(1), u.detach().softmax(1), torch.zeros(B, n_cls),
              0.3 * torch.randn(B, n_cls, generator=g)):
        grads = []
        for idx in itertools.combinations(range(B), m):
            idx = torch.tensor(idx)
            grads.append(torch.autograd.grad(kd_loss(u, target(h, pt[idx], idx, 1.0)).mean(), u)[0])
        assert torch.allclose(torch.stack(grads).mean(0), want, atol=1e-6)
    idx = torch.tensor([0, 1])
    half = torch.stack([torch.autograd.grad(kd_loss(u, target(torch.zeros(B, n_cls), pt[torch.tensor(i)], torch.tensor(i), 0.5)).mean(), u)[0]
                        for i in itertools.combinations(range(B), m)]).mean(0)
    assert not torch.allclose(half, want, atol=1e-3)
    assert target(pt, pt[idx], None, 1.0) is pt and target(pt, pt[idx], idx, 0.0) is pt     # no draw, or rho 0: h itself

    r = torch.randn(40, 3, n_cls, generator=g).softmax(2)
    for w_true, cols, w_want in (((0.3, 0.7, 0.0), [0, 1, 2], (0.3, 0.7, 0.0)), ((0.5, 0.5, 0.0), [0], None)):
        t = sum(w * r[:, j] for j, w in enumerate(w_true))
        R = torch.cat([r, t.unsqueeze(1)], 1)
        w, res = nnls((R.unsqueeze(2) * R.unsqueeze(1)).sum((0, 3)), cols)
        if w_want:
            assert torch.allclose(w, torch.tensor(w_want), atol=5e-3) and res < 1e-3
        else:
            assert w[1] == 0 and w[2] == 0 and w[0] > 0 and res > 1e-3
    t = 1.2 * r[:, 0] - 0.4 * r[:, 1]                      # the best free fit has a negative weight on r_1
    R = torch.cat([r, t.unsqueeze(1)], 1)
    w, _ = nnls((R.unsqueeze(2) * R.unsqueeze(1)).sum((0, 3)), [0, 1])
    assert w[1] == 0 and w[0] > 0


def test_twophase_reduces_to_maskedkd_and_to_full_kd(models):
    """cv masked with no full pass is MaskedKD, and a full pass for every image is full KD: same loss up to the
    entropy of the target (twophase uses the cross-entropy to the target, the old path the KL; same gradient).
    A draw of 2 of 4 images with 12 masked tokens costs 12 + 2 * 196 / 4 teacher tokens per image; one step over
    the budget is refused."""
    from harness.twophase import TwoPhase, parse
    s, t = models
    s.eval()                                                # no drop path: the two calls see the same student
    opt = torch.optim.SGD(s.parameters(), lr=0.0)
    n_cls = C.NUM_CLASSES[C.STAGE1]
    aug = GpuAugment(torch.device("cpu"), n_cls)
    x, y = aug(torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8), torch.arange(4) % n_cls)
    crit, dev = SoftTargetCrossEntropy(), torch.device("cpu")

    def ent(logits):
        p = logits.float().softmax(1)
        return float(-(p * p.clamp_min(1e-12).log()).sum(1).mean())

    with torch.no_grad():
        attn = student_forward(s, x, [-1]).attn[-1].float().mean(1)[:, 0, 1:]
        keep = torch.zeros(4, 196, dtype=torch.bool).scatter_(1, attn.topk(12, dim=1).indices, True)
        ent_masked, ent_full = ent(teacher_forward_keepmask(t, x, keep)), ent(t(x))
    old, _ = train_step(s, t, MaskedKD(), "maskedkd", 12, x, y, opt, crit, 0, 0.0, Segments(False))
    tp = TwoPhase(parse({"cv": "masked", "mask_keep": 12}, 12, batch=4), n_cls, dev)
    new, kept = train_step(s, t, MaskedKD(), "twophase", 12, x, y, opt, crit, 0, 0.0, Segments(False), tp=tp)
    assert kept == 12 and abs(new - C.ALPHA * ent_masked - old) < 1e-4 and tp.pop_diag() is None
    old, _ = train_step(s, t, Full(), "full", 196, x, y, opt, crit, 0, 0.0, Segments(False))
    for cv in ("student", "label"):
        tp = TwoPhase(parse({"cv": cv, "full_n": 4}, 196, batch=4), n_cls, dev)
        new, kept = train_step(s, t, None, "twophase", 196, x, y, opt, crit, 0, 0.0, Segments(False), tp=tp)
        assert kept == 196 and abs(new - C.ALPHA * ent_full - old) < 1e-4

    cfg = parse({"cv": "fit", "mask_keep": 12, "full_n": 2}, 110, batch=4)
    tp = TwoPhase(cfg, n_cls, dev)
    segs = Segments(True)
    loss, kept = train_step(s, t, MaskedKD(), "twophase", 110, x, y, opt, crit, 0, 0.0, segs, tp=tp)
    assert kept == 12 + 2 * 196 / 4 == cfg["tokens_per_image"] and torch.isfinite(torch.tensor(loss))
    assert segs.flops["teacher"] > 0 and "selector" in segs.flops    # both teacher passes are billed to the teacher
    d = tp.pop_diag()
    assert d["pairs"] == 2 and {"res_student", "res_label", "res_free", "res_masked", "res_recal", "res_all", "w_all",
                                "w_used", "kl_masked", "map_a", "channel_a"} <= set(d)
    assert d["res_all"] <= d["res_free"] + 1e-6 <= min(d["res_student"], d["res_label"]) + 2e-6 and tp.pop_diag() is None
    other = TwoPhase(cfg, n_cls, dev)
    other.load_state_dict(tp.state_dict())                  # what a resumed run restores
    assert torch.equal(other.w, tp.w) and torch.equal(other.rev.p[1], tp.rev.p[1]) and other.fwd.t == tp.fwd.t == 1
    with pytest.raises(HarnessViolation):
        train_step(s, t, MaskedKD(), "twophase", 100, x, y, opt, crit, 0, 0.0, Segments(False), tp=TwoPhase(cfg, n_cls, dev))
    s.train()


def test_twophase_runs_end_to_end_and_submit_checks_its_budget(tmp_path, monkeypatch):
    """A twophase job trains end to end for every kind of cheap answer, spends exactly the tokens its config says,
    records the config and the per-epoch diagnostics, and a config over its keep is refused when it is queued."""
    from harness import submit, train
    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    monkeypatch.setattr(C, "RUNS", tmp_path / "runs")
    monkeypatch.setitem(C.TIER_MAX_STEPS, "smoke", 2)
    monkeypatch.setattr(C, "BATCH", 8)
    with pytest.raises(SystemExit):
        submit.make_job("twophase", 29, 0, "smoke", {"cv": "student", "full_n": 2}, force=True, dataset="cifar10")   # 49 > 29
    with pytest.raises(SystemExit):
        submit.make_job("twophase", 29, 0, "smoke", {}, force=True, dataset="cifar10")
    assert submit.make_job("twophase", 61, 1, "final", {"cv": "fit", "mask_keep": 12, "full_n": 2}, by="human",
                           dataset="waterbirds")["selector_sha"] == "builtin"
    for args, keep, spent in (({"cv": "fit", "mask_keep": 12, "full_n": 2, "selector": "rollout"}, 61, 61.0),
                              ({"cv": "recal", "mask_keep": 12, "full_n": 2}, 61, 61.0),
                              ({"cv": "student", "full_n": 2}, 49, 49.0),
                              ({"cv": "fit", "full_n": 1}, 29, 24.5),
                              ({"cv": "channel", "mask_keep": 12, "full_n": 1}, 37, 36.5)):
        job = submit.make_job("twophase", keep, 0, "smoke", args, force=True, dataset="cifar10")
        jp = tmp_path / "job.json"
        jp.write_text(json.dumps(job))
        assert train.run(str(jp)) == 0
        m = json.loads((tmp_path / "runs" / job["id"] / "metrics.json").read_text())
        assert m["status"] == "done" and m["mean_kept"] == spent and m["sel_args"] == args
        assert m["selector_meta"]["config"]["cv"] == args["cv"] and m["val_acc_last"] is not None
        d = m["history"][-1]["tp"]
        assert d["pairs"] == 2 * args["full_n"] and d["res_free"] <= min(d["res_student"], d["res_label"]) + 1e-5
        assert ("res_all" in d) == ("mask_keep" in args) and ("w_used" in d) == (args["cv"] == "fit")


def test_dissection_runs_and_checks_the_designs_that_are_trained(tmp_path):
    """analysis/masked_teacher_dissection.py passes the analysis validator, and its "design" block compares exactly
    the two designs corrected/arms.py trains (same splits); adding the masked answer to the free ones cannot leave
    more variance, and a full pass costs 1."""
    from harness.analysis_api import AnalysisAPI
    from harness.validate import analysis_source, load_analysis
    from corrected import arms as A
    run, _ = load_analysis("masked_teacher_dissection", analysis_source("masked_teacher_dissection"))
    G = run.__globals__
    assert G["DESIGNS"] == {b: (s["img"], *s["two"]) for b, s in A.SPLITS.items()} and G["BATCH"] == A.BATCH == C.BATCH
    assert G["_teacher_cost"](196) == 1.0 and abs(G["_teacher_cost"](29) - 5.327 / 33.695) < 2e-3
    G.update(RULES=("maskedkd",), EVAL_BATCH=4, FIT_STEPS=10)
    student = make_student("waterbirds", pretrained=False).eval()

    class SmallAPI(AnalysisAPI):
        def val_batches(self, batch=100, limit=None):
            return super().val_batches(batch=batch, limit=12)

        def val_groups(self):
            return super().val_groups()[:12]

        def runs(self):
            return {"r_ce": {"method": "ce", "tier": "final", "seed": 1, "dataset": "waterbirds", "keep": 0, "snapshots": []}}

        def student(self, run_id, epoch=None):
            return student

    run(SmallAPI(tmp_path, torch.device("cpu"), make_teacher("waterbirds", pretrained=False).eval(), "waterbirds"))
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["students"] == {"ce": "r_ce"} and s["n_val"] == 12 and set(s["design"]["ce"]) == {"29", "49"}
    for budget, d in s["design"]["ce"].items():
        m1, k, m2 = G["DESIGNS"][int(budget)]
        assert d["images_only_m"] == m1 and d["tokens_images_k_m"] == [k, m2]
        assert d["predicted_better"] in ("tokens + images", "images only") and d["gain_beyond_free"] <= 1
        assert abs(d["gain_needed"] - (1 - (1 / m1 - 1 / 128) / (1 / m2 - 1 / 128))) < 1e-9
        assert (d["predicted_better"] == "tokens + images") == (d["gain_beyond_free"] > d["gain_needed"])
    for r in s["results"]["ce"].values():
        for kind in ("raw", "recal"):
            g = r[kind]
            assert g["v_all"] <= g["v_free"] * (1 + 1e-4) + 1e-9 and g["v_free"] <= g["v_student"] * (1 + 1e-4) + 1e-9
            assert all(w >= 0 for w in g["w_all_student_label_masked"])


def test_corrected_arms_spend_the_teacher_compute_of_maskedkd(models):
    """corrected/arms.py: every twophase arm is a valid config at its keep; rand_dyn draws as many images as the
    images-only arms; and, counted by FlopCounterMode on the teacher, no arm of a budget spends more than MaskedKD
    at that K (tokens + images: within 0.5 %)."""
    from torch.utils.flop_counter import FlopCounterMode
    from corrected import arms as A
    from corrected.table import needed_gain
    from harness.twophase import parse
    t = models[1]
    x = torch.randn(1, 3, 224, 224)
    cost = {}
    for k in (12, 25, 29, 45, 49, 196):
        keep = torch.zeros(1, 196, dtype=torch.bool)
        keep[:, :k] = True
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            teacher_forward_keepmask(t, x, keep)
        cost[k] = fc.get_total_flops()
    for budget in A.BUDGETS:
        table, split = A.arms(budget), A.SPLITS[budget]
        assert len(table) == 11 and table[A.MASKEDKD] == ("maskedkd", budget, {})
        for name, (method, keep, args) in table.items():
            if method != "twophase":
                continue
            cfg = parse(args, keep, batch=A.BATCH)
            spent = (cost[cfg["mask_keep"]] if cfg["mask_keep"] else 0) + cfg["full_n"] / A.BATCH * cost[196]
            assert spent <= cost[budget] * (1.005 if cfg["mask_keep"] else 1.0), name
            assert spent >= 0.95 * cost[budget], name                       # and not far below it
            assert A.arm_of(method, keep, args) == (budget, name)
        _, keep, _ = table["images, unweighted (rand_dyn)"]
        assert int(keep * A.BATCH // 196) == split["img"] == table["images"][2]["full_n"]
        assert 0 < needed_gain(budget) < 1
    assert A.arm_of("ce", 0, {}) == (None, "CE") and A.arm_of("maskedkd", 12, {}) is None
    assert A.arm_of("twophase", 29, {"cv": "fit", "mask_keep": 12, "full_n": 10}) is None
    assert set(A.UNBIASED) < set(A.arms(29)) and A.IMAGES in A.UNBIASED and A.TOKENS in A.UNBIASED


def test_corrected_autorun_orders_base_dissection_grid_and_table_reads_them(tmp_path, monkeypatch):
    """corrected.autorun queues the base runs first, the dissection once their students are here, and the grid only
    after the prediction is written; runs that exist are reused. corrected.table puts each run under its arm and
    prints the fixed readings with the prediction next to what training gave."""
    from corrected import arms as A, autorun, table
    from harness import submit
    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    monkeypatch.setattr(C, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(autorun, "PREDICTIONS", tmp_path / "PREDICTIONS.md")
    monkeypatch.setattr(autorun.S, "STATUS", tmp_path / "STATUS.md")
    for mod in (autorun, table):
        monkeypatch.setattr(mod, "teacher_sha", lambda d: "T")
    d, a = "waterbirds", argparse_ns(seeds=[1, 2, 3], seeds_more=[1])
    assert autorun.queue_base(d, a.seeds) == 7 and autorun.queue_base(d, a.seeds) == 0      # 3 CE, 3 Full KD, MaskedKD@29
    st = {"stage": "base"}
    autorun.advance(d, st, a)
    assert st == {"stage": "base"} and not autorun.students_ready(d, 1)

    def finish(job, acc, wga, extra=None):   # the scheduler's part: a queued job becomes a finished run
        j = json.loads(job.read_text())
        r = tmp_path / "runs" / j["id"]
        r.mkdir(parents=True)
        (r / "student.pt").write_bytes(b"")
        (r / "metrics.json").write_text(json.dumps({
            **{k: j[k] for k in ("id", "method", "keep", "seed", "tier", "epochs", "sel_args", "dataset")},
            "status": "done", "val_acc_last": acc, "val_wga_last": wga, "teacher_sha": "T",
            "flops": {"total_gflops_per_img": 9.0, "teacher_gflops_per_img": 5.3}, "history": extra or []}))
        job.rename(tmp_path / "q" / "done" / job.name)

    for job in sorted((tmp_path / "q" / "pending").glob("*.json")):
        finish(job, 90.0, 50.0 + json.loads(job.read_text())["seed"])
    assert autorun.students_ready(d, 1)
    autorun.advance(d, st, a)
    assert st["stage"] == "dissect" and st["analysis"].startswith("analysis_masked_teacher_dissection")
    autorun.advance(d, st, a)
    assert st["stage"] == "dissect"                                                          # still running
    out = tmp_path / "analysis_out" / st["analysis"]
    out.mkdir(parents=True)
    design = {str(b): {"predicted_better": "images only", "images_only_var_over_noise": 0.5,
                       "tokens_images_var_over_noise": 0.9, "gain_beyond_free": 0.1, "gain_needed": table.needed_gain(b),
                       "maskedkd_bias_sq_over_noise": 2.0} for b in A.BUDGETS}
    (out / "summary.json").write_text(json.dumps({"dataset": d, "design": {"ce": design, "full_e25": design, "full": design}}))
    (out / "meta.json").write_text("{}")
    autorun.advance(d, st, a)
    text = (tmp_path / "PREDICTIONS.md").read_text()
    assert st["stage"] == "grid" and "budget 29: predicted better = **images only** (student full_e25" in text
    pending = [json.loads(f.read_text()) for f in (tmp_path / "q" / "pending").glob("*.json")]
    grid = [j for j in pending if j["kind"] == "train"]
    assert len(grid) == 3 * 11 - 1 + 11                       # budget 29 on three seeds (MaskedKD seed 1 exists), 49 on one
    assert {j["seed"] for j in grid if float(j["keep"]) in (49.0, 51.0)} == {1}
    first = [j for j in grid if j["seed"] == 1 and j["method"] == "twophase" and float(j["keep"]) in (29.0, 31.0)]
    assert len(first) == 8 and all(j["priority"] == 12 for j in first)
    assert autorun.queue_grid(d, a.seeds, a.seeds_more) == 0                                 # nothing twice

    tp = {"res_student": 0.5, "res_label": 0.4, "res_masked": 0.6, "res_free": 0.3, "res_all": 0.1, "w_all": [0.2, 0.3, 0.5],
          "w_used": [0.2, 0.3, 0.5]}
    for job in sorted((tmp_path / "q" / "pending").glob("*.json")):
        j = json.loads(job.read_text())
        if j["kind"] != "train":
            continue
        better = j["method"] == "twophase" and j["sel_args"].get("cv") != "channel"
        finish(job, 91.0 + j["seed"] / 10, (70.0 if better else 50.0) + j["seed"], [{"epoch": 1, "tp": tp}])
    lines = table.section(d)
    body = "\n".join(lines)
    assert "### budget 29" in body and "### budget 49" in body and "worst-group acc" in body
    row = next(l for l in lines if l.startswith(f"| {A.TOKENS} |"))
    assert "| 1,2,3 |" in row and "+20.00 [+20.00, +20.00] (n=3)" in row and "| 5.30 | 1.000 |" in row
    r2 = next(l for l in lines if l.startswith("- R2"))
    assert "predicted better: images only" in r2 and "trained better: no difference" in r2
    diag = next(l for l in lines if l.startswith("| 0 % |"))
    assert "| 67 % | 49 % | tokens + images | 0.20 / 0.30 / 0.50 |" in diag
    assert table.section("cifar10") == []


def argparse_ns(**kw):
    import argparse
    return argparse.Namespace(**kw)
