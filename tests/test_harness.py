"""CPU tests for the harness. Run: MASKEDKD_FAKE_DATA=1 python -m pytest -q tests"""
import json
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
    aug = GpuAugment(torch.device("cpu"))
    x = torch.randint(0, 256, (8, 3, 32, 32), dtype=torch.uint8)
    y = torch.arange(8)
    xa, ya = aug(x, y)
    assert xa.shape == (8, 3, 224, 224) and ya.shape == (8, 100)
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
    aug = GpuAugment(torch.device("cpu"))
    x, y = aug(torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8), torch.arange(4))
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
    monkeypatch.setitem(C.TIER_EPOCHS["cifar100"], "final", 2)
    monkeypatch.setitem(C.TIER_EPOCHS["cifar100"], "screen", 1)
    with pytest.raises(SystemExit):
        submit.make_job("maskedkd", 49, 0, "final")          # screen seed never in final
    submit.make_job("maskedkd", 49, 1, "final")
    with pytest.raises(SystemExit):
        submit.make_job("oracle", 25, 0, "final")            # analysis-only
    submit.make_job("oracle", 25, 0, "screen")
    ids = submit.baselines()
    assert len(ids) == 2 * len(C.NOISE_SEEDS) + 5 * len(C.BASELINE_KEEPS) + 1   # maskedkd/random/grid/center/tome + oracle


def test_report_noise_and_discriminative():
    from harness.report import noise, discriminative, tries
    runs = [{"tier": "screen", "method": m, "keep": k, "seed": s, "val_acc_last": v, "selector_sha": "b", "sel_args": {}}
            for m, k, s, v in [("maskedkd", 98, 0, 70.0), ("maskedkd", 98, 1, 70.6), ("full", 196, 0, 71.0),
                               ("full", 196, 1, 70.4), ("random", 25, 0, 60.0), ("maskedkd", 25, 0, 64.0)]]
    sigma, thr, _ = noise(runs)
    assert sigma is not None and thr >= C.PROMOTE_MIN
    assert any("+4.00" in line for line in discriminative(runs))
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
                "img_per_s_mean": 500.0, "dataset": "cifar100"}
        (d / "metrics.json").write_text(json.dumps({**base, **r}))


def test_stage2_rules(tmp_path, monkeypatch):
    from harness import submit
    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    monkeypatch.setitem(C.TIER_EPOCHS["in100"], "final", 2)
    (tmp_path / "selectors").mkdir()
    monkeypatch.setattr("harness.validate.SELECTORS_DIR", tmp_path / "selectors")
    (tmp_path / "selectors" / "good.py").write_text(GOOD)
    import hashlib
    sha = hashlib.sha256(GOOD.encode()).hexdigest()
    _fake_runs(tmp_path, monkeypatch, [])
    submit.make_job("ce", 0, 1, "final", dataset="in100")                 # baselines always allowed
    with pytest.raises(SystemExit, match="gate"):
        submit.make_job("good", 25, 1, "final", dataset="in100")         # stage 2 closed before its gate
    gate = [{"method": m, "keep": k, "seed": 1, "tier": "final", "val_acc_last": v, "dataset": "in100"}
            for m, k, v in (("ce", 0, 70.0), ("full", 196, 75.0), ("maskedkd", 98, 74.8))]
    cifar = [{"method": "good", "keep": 25, "seed": s, "tier": "final", "val_acc_last": 70.0, "selector_sha": sha}
             for s in (1, 2, 3)]
    import shutil
    shutil.rmtree(tmp_path / "runs", ignore_errors=True)
    _fake_runs(tmp_path, monkeypatch, gate + cifar)
    with pytest.raises(SystemExit, match="audit"):
        submit.make_job("good", 25, 1, "final", dataset="in100")         # no audit PASS yet
    (tmp_path / "AUDIT.md").write_text("# AUDIT\n\n## x audit by claude\nclaim good K=25\nVERDICT: PASS\n")
    job = submit.make_job("good", 25, 1, "final", dataset="in100")
    assert job["stage2"] == "stage2_untouched" and job["priority"] <= C.STAGE2_PRIORITY


def test_in100_smoke_run(tmp_path, monkeypatch):
    from harness import submit, train
    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    monkeypatch.setattr(C, "RUNS", tmp_path / "runs")
    monkeypatch.setitem(C.TIER_MAX_STEPS, "smoke", 2)
    monkeypatch.setattr(C, "BATCH", 8)
    job = submit.make_job("maskedkd", 49, 0, "smoke", dataset="in100", force=True)
    jp = tmp_path / "job.json"
    jp.write_text(json.dumps(job))
    assert train.run(str(jp)) == 0
    m = json.loads((tmp_path / "runs" / job["id"] / "metrics.json").read_text())
    assert m["dataset"] == "in100" and m["mean_kept"] == 49.0 and m["img_per_s_mean"] > 0


def test_figures(tmp_path, monkeypatch):
    recs = [{"method": m, "keep": k, "seed": s, "tier": "screen", "val_acc_last": v}
            for m, k, s, v in (("ce", 0, 0, 60.0), ("full", 196, 0, 70.0), ("maskedkd", 98, 0, 69.8),
                               ("maskedkd", 49, 0, 68.5), ("random", 49, 0, 66.0), ("tome", 49, 0, 67.0))]
    _fake_runs(tmp_path, monkeypatch, recs)
    from harness import figures
    monkeypatch.setattr("sys.argv", ["figures"])
    figures.main()
    assert (tmp_path / "figures" / "cifar100_screen_tokens.png").exists()
    assert "maskedkd" in (tmp_path / "figures" / "cifar100_screen.csv").read_text()


def test_boot_rules(tmp_path, monkeypatch):
    from harness import boot
    monkeypatch.setattr(C, "STATE", tmp_path)
    monkeypatch.setattr(C, "EPOCHS_FILE", tmp_path / "epochs.json")
    (tmp_path / "bench.json").write_text(json.dumps({"recommend": {"screen_epochs_for_1h_at_that_parallelism": 37,
                                                                   "parallel_jobs": 3}}))
    boot.epochs_from_bench()
    d = json.loads((tmp_path / "epochs.json").read_text())
    assert d["epochs"]["cifar100"] == {"screen": 37, "final": 74} and d["max_jobs"] == 3
    _fake_runs(tmp_path, monkeypatch, [{"method": "maskedkd", "keep": 49, "seed": 0, "tier": "smoke", "val_acc_last": 1.0,
                                        "img_per_s_mean": 400.0, "dataset": "in100"}])
    boot.epochs_in100("r0")
    d = json.loads((tmp_path / "epochs.json").read_text())
    assert d["epochs"]["in100"]["final"] == int(12 * 3600 / (boot.IN100_TRAIN / 400.0))
    assert d["epochs"]["cifar100"]["screen"] == 37


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
    monkeypatch.setitem(C.TIER_EPOCHS["cifar100"], "final", 2)
    monkeypatch.setitem(C.TIER_EPOCHS["cifar100"], "screen", 1)
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
    assert any("vs grid@25: missing" in l for l in lines)                   # missing references are shown, not hidden
    assert any("vs full@196" in l and "LOSS" in l for l in sig)
    scr = [_run("child", 25, 0, 60.2, "screen", parent="mysel"), _run("mysel", 25, 0, 60.0, "screen", parent="maskedkd"),
           _run("maskedkd", 25, 0, 59.0, "screen")]
    pc = parent_check(scr, 0.5)
    assert any(l.startswith("- child@25") and "UNPROVEN" in l for l in pc)
    assert any(l.startswith("- mysel@25") and "helps" in l for l in pc)


def test_report_cli_accepts_dataset(tmp_path, monkeypatch):
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "harness.report", "--gate", "--dataset", "in100"],
                       capture_output=True, text=True, cwd=C.ROOT)
    assert "unrecognized arguments" not in r.stderr and "gate incomplete" in r.stdout


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
    monkeypatch.setitem(C.TIER_EPOCHS["cifar100"], "screen", 56)
    for i, (ep, v) in enumerate(((28, 60.0), (56, 62.0))):
        d = C.RUNS / f"r{i}"; d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"id": f"r{i}", "status": "done", "tier": "screen", "method": "ce",
                                                     "keep": 0, "seed": 0, "epochs": ep, "val_acc_last": v}))
    assert [r["val_acc_last"] for r in report.load_runs("cifar100")] == [62.0]
    # the gate's extra seeds are reused by --baselines instead of failing as duplicates
    monkeypatch.setattr(C, "QUEUE", tmp_path / "q")
    ids = submit.gate_jobs(seeds=[1, 2])
    assert submit.gate_jobs(seeds=[1, 2]) == ids
    base = submit.baselines()
    assert all(i in base for i in ids if "_ce_" not in i)


def test_scale_epochs(tmp_path, monkeypatch):
    from harness import boot
    monkeypatch.setattr(C, "STATE", tmp_path)
    monkeypatch.setattr(C, "EPOCHS_FILE", tmp_path / "epochs.json")
    (tmp_path / "epochs.json").write_text(json.dumps({"epochs": {"cifar100": {"screen": 28, "final": 56}}, "max_jobs": 4}))
    boot.scale_epochs(2)
    boot.scale_epochs(4)                                       # always relative to the bench rule, not compounding
    d = json.loads((tmp_path / "epochs.json").read_text())
    assert d["epochs"]["cifar100"] == {"screen": 112, "final": 224} and d["gate_scale"] == 4 and d["max_jobs"] == 4


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
        {"id": rid, "val_acc_last": 80.1, "history": [{"epoch": 1, "loss": 2.0}, {"epoch": 2, "val_acc": 80.1}]}))
    (tmp_path / "q" / "done").mkdir(parents=True)
    (tmp_path / "q" / "done" / f"{rid}.json").write_text(json.dumps(
        {"method": "mysel", "selector_sha": "f" * 64, "selector_src": "META = {'parent': 'none'}"}))
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "AUDIT_REQUEST.md").write_text(f"claim: mysel wins, runs {rid}")
    text = evidence.audit()
    assert f"runs/{rid}/metrics.json" in text and '"val_acc": 80.1' in text and '"loss"' not in text
    assert "selector source mysel" in text and "META = {'parent': 'none'}" in text
    assert "selector_api" in evidence.ideas() or "missing: harness/selector_api.py" in evidence.ideas()
