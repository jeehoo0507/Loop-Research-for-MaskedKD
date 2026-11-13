"""The arms of the corrected-distillation study (human, 2026-11-13). One place, used by autorun and table.

A budget is named after the MaskedKD run it is matched to: every arm of budget 29 spends the teacher GFLOPs of
MaskedKD at K=29 or less (within 0.3 % for the tokens + images arms), measured by the harness in every run.
Images-only arms give the full teacher to `img` of the 128 images of a step and mask nothing. Tokens + images arms
give the masked teacher K' tokens of every image and the full teacher to m of them. The harness counts a full pass
as 196 tokens and a pass on K tokens as K, but a masked pass also runs the patch embedding on the whole image, so in
GFLOPs the images-only arms may draw two or three images more than the token count of MaskedKD allows: their jobs
carry the keep their own tokens need (31 and 51), not 29 and 49.
"""
import math

from harness import config as C

BATCH = 128   # the splits below are for config.BATCH = 128
BUDGETS = (29, 49)
# img: full-teacher images per step when nothing is masked. two: (K', m). chan: (K', m), the pairs only fit the map.
# Teacher GFLOPs per image (FlopCounterMode, DeiT-B): MaskedKD 5.33 at K=29, 8.73 at K=49.
#   29: img 20 -> 5.27, two (12, 11) -> 5.34, chan (25, 2) -> 5.17      49: img 33 -> 8.69, two (29, 13) -> 8.75, chan (45, 2) -> 8.57
SPLITS = {29: {"img": 20, "two": (12, 11), "chan": (25, 2)},
          49: {"img": 33, "two": (29, 13), "chan": (45, 2)}}
REFERENCE = {"CE": ("ce", 0, {}), "Full KD": ("full", C.NUM_PATCHES, {})}
MASKEDKD = "MaskedKD"
IMAGES, TOKENS = "images + fit", "tokens + images, fit"        # the two designs the prediction is about
UNBIASED = ("images", "images + label", IMAGES, "tokens + images, masked", "tokens + images, recal", TOKENS,
            "tokens + images, fit, rollout")


def keep_for(k, m):
    """The smallest whole keep that covers mask_keep k on every image and m full passes per step."""
    return math.ceil(k + m * C.NUM_PATCHES / BATCH - 1e-9)


def arms(budget):
    """name -> (method, keep, sel_args) at one budget. sel_args are exactly what is queued: a finished run is
    matched to its arm by them."""
    s = SPLITS[budget]
    m1, (k2, m2), (kc, mc) = s["img"], s["two"], s["chan"]
    two = lambda cv, **kw: ("twophase", budget, {"cv": cv, "mask_keep": k2, "full_n": m2, **kw})   # noqa: E731
    img = lambda cv: ("twophase", keep_for(0, m1), {"cv": cv, "full_n": m1})                        # noqa: E731
    return {
        MASKEDKD: ("maskedkd", budget, {}),
        "rollout": ("rollout", budget, {}),
        "images, unweighted (rand_dyn)": ("rand_dyn", keep_for(0, m1), {}),   # the same m images, KD weight 1 not B/m
        "images": img("student"),
        "images + label": img("label"),
        IMAGES: img("fit"),
        "tokens + images, masked": two("masked"),
        "tokens + images, recal": two("recal"),
        TOKENS: two("fit"),
        "tokens + images, fit, rollout": two("fit", selector="rollout"),
        "channel": ("twophase", budget, {"cv": "channel", "mask_keep": kc, "full_n": mc}),
    }


def arm_of(method, keep, sel_args):
    """(budget or None for a reference, arm name) of a finished run, or None if it is not an arm of this study."""
    sel_args = dict(sel_args or {})
    for name, (m, k, a) in REFERENCE.items():
        if m == method and a == sel_args:
            return None, name
    for budget in BUDGETS:
        for name, (m, k, a) in arms(budget).items():
            if m == method and float(k) == float(keep) and a == sel_args:
                return budget, name
    return None
