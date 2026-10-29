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
        "vs_maskedkd": "multi-layer instead of last block", "prior_art": "none identical known"}
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
