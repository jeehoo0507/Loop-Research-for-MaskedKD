"""The arms of the surgical fine-tuning study (human, 2026-11-12). One place, used by submit_grid and table.

Token selection is always the harness attention rollout; an arm is the set of student blocks it multiplies.
Block groups follow Surgical Fine-Tuning's Block 1-4 (Lee et al. ICLR 2023).
"""
K = 29
B1, B2, B3, B4 = [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]


def _roll(layers):
    return ("layerroll", {"layers": list(layers)})


# name -> (method, sel_args). Builtins are used where an arm is one, so runs that already exist are reused.
SELECT = {
    "last (MaskedKD)": ("maskedkd", {}),          # block 11 alone
    "B1 0-2": _roll(B1),
    "B2 3-5": _roll(B2),
    "B3 6-8": _roll(B3),
    "B4 9-11": _roll(B4),
    "all 0-11 (rollout)": ("rollout", {}),
    "no last 0-10": _roll(range(11)),
    "no first 3-11": _roll(range(3, 12)),
}
EXTRA = {   # not in the default grid
    "no middle 0-2,9-11": _roll(B1 + B4),
    "no middle 0-5,9-11": _roll(B1 + B2 + B4),
    "late 6-11 (lateroll)": ("lateroll", {}),
    "late no last 6-10": _roll(range(6, 11)),
}
REFERENCE = {"CE": ("ce", {}), "Full KD": ("full", {}), "random": ("random", {})}

# What surgical fine-tuning says about each dataset: the block group the shift lives in. "follow" selects with it,
# "avoid" selects with everything but it.
SHIFT = {
    "waterbirds": ("output", "last (MaskedKD)", "no last 0-10"),
    "celeba": ("output", "last (MaskedKD)", "no last 0-10"),
    "cub_lt": ("output", "last (MaskedKD)", "no last 0-10"),
    "cifar10c_gaussian_noise": ("input", "B1 0-2", "no first 3-11"),
    "cifar10c_defocus_blur": ("input", "B1 0-2", "no first 3-11"),
    "cifar10c_frost": ("input", "B1 0-2", "no first 3-11"),
    "cifar10": ("none", None, None),
    "cub": ("none", None, None),
    "aircraft": ("none", None, None),
}


def teacher_sha(dataset):
    """sha256 of this server's teacher checkpoint for `dataset`, or None if there is none here. The checkpoints are
    not in git: a server set up later fine-tunes its own teacher, and KD runs made with another teacher are not the
    same experiment."""
    import json
    from harness.train import sha256_file, teacher_paths
    ckpt, meta = teacher_paths(dataset)
    if not ckpt.exists() or not meta.exists():
        return None
    sha = sha256_file(ckpt)
    return sha if sha == json.loads(meta.read_text()).get("sha256") else None


def usable(run, sha):
    """A finished run this server may reuse and compare: CE (no teacher), or KD with this server's teacher."""
    return run["method"] == "ce" or (sha is not None and run.get("teacher_sha") == sha)


def arm_of(method, sel_args):
    """Arm name of a finished run, or None if it is not an arm of this study."""
    key = (method, {k: v for k, v in (sel_args or {}).items()})
    for name, (m, a) in {**REFERENCE, **SELECT, **EXTRA}.items():
        if m == key[0] and a == key[1]:
            return name
    return None
