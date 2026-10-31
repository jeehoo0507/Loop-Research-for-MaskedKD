"""Evidence bundles for Codex, which cannot run shell commands on the server (its sandbox, bubblewrap, does not
start in an unprivileged container). agent/loop.sh puts the bundle into the prompt instead.

    python -m harness.evidence audit   # for an audit of agent/AUDIT_REQUEST.md
    python -m harness.evidence ideas   # for the daily ideas (interface and results only)

Locked with the harness, so the researcher cannot change what the auditor gets to see.
"""
import json
import re
import subprocess
import sys

from . import config as C

LIMIT = 180_000      # characters per bundle
FILE_LIMIT = 12_000  # characters per attached file (the job history gets more: it must be complete)


def section(title, text, limit=FILE_LIMIT):
    text = text if len(text) <= limit else text[:limit] + f"\n... [truncated, {len(text)} chars]"
    return f"\n===== {title} =====\n{text.rstrip()}\n"


def run(*cmd):
    try:
        r = subprocess.run(cmd, cwd=C.ROOT, capture_output=True, text=True)
    except OSError as e:
        return f"(could not run {cmd[0]}: {e})"
    return (r.stdout + r.stderr).strip() or f"(no output, rc={r.returncode})"


def read(rel):
    p = C.ROOT / rel
    return p.read_text(errors="replace") if p.is_file() else f"(missing: {rel})"


def run_metrics(rid):
    m = C.RUNS / rid / "metrics.json"
    if not m.exists():
        return f"(no runs/{rid}/metrics.json)"
    d = json.loads(m.read_text())
    d["history"] = [h for h in d.get("history", []) if "val_acc" in h]   # val points only
    return json.dumps(d, indent=1)


def history():
    """Every job ever queued, in submission order, whatever happened to it: the auditor can see that no run of a
    cited method (failed, timed out, killed, still running) was left out."""
    rows = []
    for state in ("done", "running", "pending"):
        for f in (C.QUEUE / state).glob("*.json"):
            j = json.loads(f.read_text())
            jid = j.get("id", f.stem)
            m = C.RUNS / jid / "metrics.json"
            st = json.loads(m.read_text()).get("status") if m.exists() else "-"
            rows.append((str(j.get("submitted", "")), jid, state, str(j.get("returncode", "-")), str(st), str(j.get("by", ""))))
    rows.sort()
    lines = ["submitted | id | queue state | returncode | metrics status | by"]
    lines += [" | ".join(r) for r in rows]
    bad = [r for r in rows if r[2] == "done" and r[3] not in ("0", "-")]
    lines.append(f"total {len(rows)} jobs; done with non-zero returncode: {len(bad)}")
    return "\n".join(lines)


def audit():
    req = read("agent/AUDIT_REQUEST.md")
    py = sys.executable   # the loop runs this module with the project's .venv python
    out = [section("AUDIT_REQUEST.md", req),
           section("python -m harness.lock (current digest)", run(py, "-m", "harness.lock")),
           section("python -m harness.lock --check", run(py, "-m", "harness.lock", "--check")),
           section("RESULTS_TABLE.md", read("RESULTS_TABLE.md")),
           section("git log -25 --stat (who changed what)", run("git", "log", "-25", "--stat", "--format=%h %ci %s")),
           section("git log -- harness tests (must show no agent commits)",
                   run("git", "log", "-10", "--format=%h %ci %s", "--", "harness", "tests")),
           section("agent/INVALID.md", read("agent/INVALID.md")),
           section("complete job history (all states)", history(), limit=60_000)]
    run_ids = sorted(set(re.findall(r"\b(?:in100_)?(?:screen|final)_[A-Za-z0-9_.-]+_\d{4}_\d{6}_[0-9a-f]{4}\b", req)))
    shas = {}
    for rid in run_ids:
        out.append(section(f"runs/{rid}/metrics.json", run_metrics(rid)))
        job = C.QUEUE / "done" / f"{rid}.json"
        if job.exists():
            j = json.loads(job.read_text())
            if j.get("selector_src") and j.get("selector_sha") not in shas:
                shas[j["selector_sha"]] = j["method"]
                out.append(section(f"selector source {j['method']} (sha {j['selector_sha'][:12]}, from the job snapshot)",
                                   j["selector_src"]))
    for aid in sorted(set(re.findall(r"\banalysis_[A-Za-z0-9_]+_\d{4}_\d{6}_[0-9a-f]{4}\b", req))):
        d = C.ROOT / "analysis_out" / aid
        files = sorted(d.glob("*.json")) if d.is_dir() else []
        out.append(section(f"analysis_out/{aid}", "\n".join(f"--- {f.name}\n{f.read_text()[:4000]}" for f in files)
                           or "(missing)"))
    return "".join(out)


def ideas():
    return "".join(section(f, read(f)) for f in ("harness/selector_api.py", "harness/baselines.py", "RESULTS_TABLE.md"))


if __name__ == "__main__":
    text = {"audit": audit, "ideas": ideas}[sys.argv[1]]()
    print(text if len(text) <= LIMIT else text[:LIMIT] + "\n... [bundle truncated]")
