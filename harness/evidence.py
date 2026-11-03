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

LIMIT = 400_000      # characters per bundle (~110k tokens; 180k cut the cited diff, analyses and tables, 2026-11-03)
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


def history(methods=None):
    """Every job ever queued, in submission order, whatever happened to it: the auditor can see that no run of a
    cited method (failed, timed out, killed, still running) was left out. methods: only jobs of these methods (all
    their datasets, tiers and states), so the list fits the bundle; the totals line still counts every job."""
    rows, n_all, bad_all = [], 0, 0
    for state in ("done", "running", "pending"):
        for f in (C.QUEUE / state).glob("*.json"):
            j = json.loads(f.read_text())
            jid = j.get("id", f.stem)
            m = C.RUNS / jid / "metrics.json"
            st = json.loads(m.read_text()).get("status") if m.exists() else "-"
            n_all += 1
            bad_all += state == "done" and str(j.get("returncode", "-")) not in ("0", "-")
            if methods and j.get("method") not in methods:
                continue
            rows.append((str(j.get("submitted", "")), jid, state, str(j.get("returncode", "-")), str(st), str(j.get("by", ""))))
    rows.sort()
    lines = ["submitted | id | queue state | returncode | metrics status | by"]
    lines += [" | ".join(r) for r in rows]
    bad = [r for r in rows if r[2] == "done" and r[3] not in ("0", "-")]
    lines.append(f"listed {len(rows)} jobs{' of ' + ', '.join(sorted(methods)) if methods else ''}; done with non-zero "
                 f"returncode among them: {len(bad)}. All jobs: {n_all}, done with non-zero returncode: {bad_all}")
    return "\n".join(lines)


LOCKED_PATHS = ["harness", "tests", "agent/loop.sh", "agent/relock.sh", "agent/AGENT_PROMPT.md", "agent/AUDIT_PROMPT.md",
                "agent/IDEAS_PROMPT.md", "requirements.txt", "state/epochs.json", "state/recipe.json"]


def past_locks(run_ids):
    """For cited runs made under an earlier lock: the digest recomputed from the commit the run started at (its
    metrics.json "git"), and every later commit that touched a locked file, with its author (2026-11-02: two audits
    could not verify a past digest from the bundle)."""
    from .lock import digest_at, harness_digest
    now, seen, lines = harness_digest(), {}, []
    for rid in run_ids:
        m = C.RUNS / rid / "metrics.json"
        if not m.exists():
            continue
        d = json.loads(m.read_text())
        rec, commit = d.get("harness_digest"), d.get("git")
        if not rec or rec == now or not commit:
            lines.append(f"{rid}: recorded {str(rec)[:12]} {'= current lock' if rec == now else '(no git commit recorded)'}")
            continue
        if commit not in seen:
            seen[commit] = digest_at(commit)
        got = seen[commit]
        lines.append(f"{rid}: recorded {rec[:12]}, run started at {commit[:12]}, digest recomputed from that commit "
                     f"{str(got)[:12]} -> {'MATCH' if got == rec else 'NO MATCH (the run used uncommitted locked files)'}")
    for commit in seen:
        lines.append(f"\n--- commits touching locked files after {commit[:12]} (author, date, subject) ---")
        lines.append(run("git", "log", "--format=%h | %an | %ci | %s", f"{commit}..HEAD", "--", *LOCKED_PATHS))
        lines.append(f"--- git diff --stat {commit[:12]} HEAD -- locked files ---")
        lines.append(run("git", "diff", "--stat", commit, "HEAD", "--", *LOCKED_PATHS))
    if seen:   # the content, not only titles and stats (2026-11-02 audit: stats cannot show unchanged semantics)
        oldest = run("git", "merge-base", "--octopus", *seen).splitlines()[0] if len(seen) > 1 else next(iter(seen))
        lines.append(f"\n--- git diff {oldest[:12]} HEAD -- locked files except tests (full text) ---")
        lines.append(run("git", "diff", oldest, "HEAD", "--", *[p for p in LOCKED_PATHS if p != "tests"]))
    return "\n".join(lines) or "(no cited runs)"


def method_history(methods, req=""):
    """For each cited agent method: its PROPOSALS.md entries (pre-registration, analysis before implementation) and the
    git history of selectors/<method>.py, so the order analysis -> proposal -> code -> runs can be checked. Entries whose
    id (H-C2b, H-W25, ...) the request cites are included too, even when they name the method differently."""
    props = read("PROPOSALS.md").split("\n- **")
    ids = set(re.findall(r"\bH-[A-Z]+\d+[a-z]?\b", req))
    out = []
    for m in sorted(methods):
        out.append(f"### {m}: git log --follow selectors/{m}.py")
        out.append(run("git", "log", "--follow", "--format=%h | %an | %ci | %s", "--", f"selectors/{m}.py"))
    cited = [b for b in props if (m := re.match(r"(H-[A-Z]+\d+[a-z]?)\b", b)) and m.group(1) in ids]
    out.append(f"### PROPOSALS.md entries cited by id in the request ({len(cited)} of {sorted(ids)}; each cut at 4000 chars)")
    out += ["- **" + b[:4000] for b in cited]
    for m in sorted(methods):
        hits = [b for b in props if re.search(rf"\b{re.escape(m)}\b", b) and b not in cited]
        out.append(f"### {m}: other PROPOSALS.md entries naming it ({len(hits)}; oldest first, each cut at 1500 chars)")
        out += ["- **" + b[:1500] for b in hits]
    return "\n".join(out) or "(no agent method cited)"


def audit():
    """Most specific evidence first and the bulky context last: the bundle is cut at LIMIT characters, and on
    2026-11-02 the cut fell before the cited analysis outputs (NEEDS-MORE)."""
    req = read("agent/AUDIT_REQUEST.md")
    py = sys.executable   # the loop runs this module with the project's .venv python
    out = [section("AUDIT_REQUEST.md", req),
           section("python -m harness.lock (current digest)", run(py, "-m", "harness.lock")),
           section("python -m harness.lock --check", run(py, "-m", "harness.lock", "--check"))]
    run_ids = sorted(set(re.findall(r"\b(?:in100_|aircraft_|waterbirds_)?(?:screen|final)_[A-Za-z0-9_.-]+_\d{4}_\d{6}_[0-9a-f]{4}\b", req)))
    shas = {}
    for rid in run_ids:
        out.append(section(f"runs/{rid}/metrics.json", run_metrics(rid), limit=6_000))
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
        out.append(section(f"analysis_out/{aid}", "\n".join(f"--- {f.name}\n{f.read_text()[:60_000]}" for f in files)
                           or "(missing)", limit=80_000))
    agent_methods = {m for m in shas.values() if m not in C.BASELINES}
    cited_methods = set(shas.values()) | {json.loads((C.RUNS / r / "metrics.json").read_text()).get("method")
                                          for r in run_ids if (C.RUNS / r / "metrics.json").exists()} - {None}
    out += [section("proposal and code history of the cited agent methods", method_history(agent_methods, req),
                    limit=100_000),
            section("past locks of the cited runs (harness.lock.digest_at)", past_locks(run_ids), limit=60_000),
            section("agent/INVALID.md", read("agent/INVALID.md")),
            section("git log -- harness tests (must show no agent commits)",
                    run("git", "log", "-10", "--format=%h %ci %s", "--", "harness", "tests")),
            section("RESULTS_TABLE.md", read("RESULTS_TABLE.md"), limit=60_000),
            section("git log -25 --stat (who changed what)", run("git", "log", "-25", "--stat", "--format=%h %ci %s"), limit=20_000),
            section(f"complete job history of the cited methods (all states): {', '.join(sorted(cited_methods))}",
                    history(cited_methods), limit=40_000)]
    return "".join(out)


def ideas():
    return "".join(section(f, read(f)) for f in ("harness/selector_api.py", "harness/baselines.py", "RESULTS_TABLE.md"))


if __name__ == "__main__":
    text = {"audit": audit, "ideas": ideas}[sys.argv[1]]()
    print(text if len(text) <= LIMIT else text[:LIMIT] + f"\n... [bundle truncated at {LIMIT} of {len(text)} chars;"
          " everything above the job history is complete]")
