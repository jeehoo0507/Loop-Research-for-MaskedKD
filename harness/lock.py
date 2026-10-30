"""Tamper lock for the harness and the agent's own rules.

The expected digest lives OUTSIDE the repo ($MASKEDKD_LOCK, default ~/.maskedkd_lock), written only by
a human via `bash agent/relock.sh`. agent/loop.sh and the scheduler compare against it.

    python -m harness.lock            # print current digest
    python -m harness.lock --check    # exit 1 if it differs from the stored one
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

from . import config as C

LOCKED_GLOBS = ["harness/**/*.py", "tests/**/*.py", "agent/loop.sh", "agent/relock.sh",
                "agent/AGENT_PROMPT.md", "agent/AUDIT_PROMPT.md", "agent/IDEAS_PROMPT.md",
                "requirements.txt", "state/epochs.json", "state/recipe.json"]
LOCK_FILE = Path(os.environ.get("MASKEDKD_LOCK", Path.home() / ".maskedkd_lock"))


def locked_files():
    files = set()
    for g in LOCKED_GLOBS:
        files.update(p for p in C.ROOT.glob(g) if p.is_file() and "__pycache__" not in p.parts)
    return sorted(files)


def harness_digest():
    h = hashlib.sha256()
    for p in locked_files():
        h.update(str(p.relative_to(C.ROOT)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def check():
    if not LOCK_FILE.exists():
        return False, f"no lock file at {LOCK_FILE}; a human must run: bash agent/relock.sh"
    want = LOCK_FILE.read_text().strip()
    got = harness_digest()
    return got == want, f"harness digest {got[:12]} vs locked {want[:12]}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        ok, msg = check()
        print(("OK " if ok else "MISMATCH ") + msg)
        sys.exit(0 if ok else 1)
    print(harness_digest())
