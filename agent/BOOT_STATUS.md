# BOOT_STATUS (agent/bootstrap.sh)


- 2026-10-30 11:57 RESET: setting changed (harness/config.py); old results in archive/20261030_1156
- 2026-10-30 11:57 Codex works without a shell here (container blocks its sandbox): GPT audits and adds ideas from files the harness attaches; no emergency researcher
- 2026-10-30 12:12 GPU smoke passed
- 2026-10-30 12:35 teacher: fine-tuned here val acc 93.41 WGA 64.66; recipe: {"waterbirds":{"student_lr":0.0001,"source":"config.STUDENT_LR_DEFAULT(noaim-lab-test-5sweepfound)"}}
- 2026-10-30 12:37 parallel jobs: {"epochs":{},"max_jobs":4,"rule":"epochsfixedat100(aim-lab-test-5recipe);max_jobs=bestbenchparallelism"}
- 2026-10-30 13:05 gate: [waterbirds screen, 100 epochs, mean of seeds [0]] Full KD 90.66 vs CE 89.24 (need +1.0), WGA 65.41 vs 66.17 (or need +2.0): PASS; MaskedKD@98 91.24 vs Full 90.66 (need within 0.5): FAIL
- 2026-10-30 13:59 gate: [waterbirds screen, 100 epochs, mean of seeds [0, 1, 2]] Full KD 90.83 vs CE 89.32 (need +1.0), WGA 66.67 vs 66.42 (or need +2.0): PASS; MaskedKD@98 90.69 vs Full 90.83 (need within 0.5): PASS
- 2026-10-30 13:59 gate PASSED: [waterbirds screen, 100 epochs, mean of seeds [0, 1, 2]] Full KD 90.83 vs CE 89.32 (need +1.0), WGA 66.67 vs 66.42 (or need +2.0): PASS; MaskedKD@98 90.69 vs Full 90.83 (need within 0.5): PASS
- 2026-10-30 13:59 Waterbirds baselines queued (23 jobs)
- 2026-10-30 13:59 CUB (stage 2) teacher, gate and baselines started in background (tmux: stage2)
- 2026-10-30 13:59 research loop started (auditor: codex). Your part from here: agent/QUEUE.md, agent/STOP
- 2026-10-30 15:00 [stage2] CUB teacher: fine-tuned here val acc 84.67
