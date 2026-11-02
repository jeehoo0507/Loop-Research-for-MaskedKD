"""Tamper lock for the harness and the agent's own rules.

The expected digest lives OUTSIDE the repo ($MASKEDKD_LOCK, default ~/.maskedkd_lock), written by
`bash agent/relock.sh` or by `--check` when it adopts an update pushed to GitHub (below). agent/loop.sh and the
scheduler compare against it.

    python -m harness.lock            # print current digest
    python -m harness.lock --check    # exit 1 if it differs from the stored one (after trying to adopt)

Adopting (2026-10-31, so a human change pushed from elsewhere does not pause the loop until someone relocks on the
server): on a mismatch, `--check` relocks by itself only if every locked file is exactly what the upstream branch has
(no local commit or uncommitted change touches them, no untracked file under harness/ or tests/) and the tests pass.
The researcher cannot push, so upstream content is human content; anything edited on the server still fails.
"""
import argparse
import hashlib
import os
import subprocess
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


def digest_at(commit):
    """harness_digest() of the locked files as committed at `commit` (same files, order and bytes), or None if git
    fails. Lets an audit check a run made under an earlier lock: its metrics.json records harness_digest and the HEAD
    it started from ("git"); if digest_at(that HEAD) equals the recorded digest, the run used exactly that commit's
    locked files, and `git log <that HEAD>..HEAD -- <locked paths>` shows every change since and who made it."""
    import fnmatch
    from pathlib import PurePosixPath
    try:
        names = subprocess.run(["git", "ls-tree", "-r", "--name-only", commit], cwd=C.ROOT, capture_output=True,
                               text=True, check=True, timeout=120).stdout.split("\n")
    except (subprocess.SubprocessError, OSError):
        return None
    def locked(n):   # same matches as Path.glob(LOCKED_GLOBS): "dir/**/*.py" = any .py at any depth under dir
        for g in LOCKED_GLOBS:
            if "/**/" in g:
                top, base = g.split("/**/")
                if n.startswith(top + "/") and fnmatch.fnmatchcase(n.rsplit("/", 1)[1], base):
                    return True
            elif n == g:
                return True
        return False
    files = sorted(PurePosixPath(n) for n in names if n and "__pycache__" not in n and locked(n))
    h = hashlib.sha256()
    for f in files:
        try:
            data = subprocess.run(["git", "show", f"{commit}:{f}"], cwd=C.ROOT, capture_output=True, check=True,
                                  timeout=120).stdout
        except (subprocess.SubprocessError, OSError):
            return None
        h.update(str(f).encode())
        h.update(b"\0")
        h.update(data)
        h.update(b"\0")
    return h.hexdigest()


def _git(*args):
    r = subprocess.run(["git", *args], cwd=C.ROOT, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                       timeout=120, env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="true"))
    return r.returncode, r.stdout


def adopt_upstream():
    """Relock to the current tree if its locked files equal the upstream branch's and the tests pass.
    Returns (adopted, reason)."""
    paths = ["harness", "tests", "agent/loop.sh", "agent/relock.sh", "agent/AGENT_PROMPT.md", "agent/AUDIT_PROMPT.md",
             "agent/IDEAS_PROMPT.md", "requirements.txt", "state/epochs.json", "state/recipe.json"]
    rc, up = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    up = up.strip() if rc == 0 and up.strip() else "origin/main"
    if _git("diff", "--quiet", up, "--", *paths)[0] != 0:
        return False, f"locked files differ from {up} (local commit or edit)"
    if _git("diff", "--quiet", "HEAD", "--", *paths)[0] != 0:
        return False, "uncommitted change to locked files"
    rc, untracked = _git("status", "--porcelain", "--untracked-files=all", "--", "harness", "tests")
    if rc != 0 or untracked.strip():
        return False, "untracked or modified files under harness/ or tests/"
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=C.ROOT, capture_output=True, text=True,
                       stdin=subprocess.DEVNULL, timeout=900, env=dict(os.environ, MASKEDKD_FAKE_DATA="1"))
    if t.returncode != 0:
        return False, "tests failed: " + (t.stdout + t.stderr).strip()[-300:]
    LOCK_FILE.write_text(harness_digest() + "\n")
    return True, f"adopted {up} ({_git('rev-parse', '--short', 'HEAD')[1].strip()}), tests passed"


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
        if not ok and LOCK_FILE.exists():
            adopted, why = adopt_upstream()
            msg += f"; {why}"
            ok = adopted
        print(("OK " if ok else "MISMATCH ") + msg)
        sys.exit(0 if ok else 1)
    print(harness_digest())
