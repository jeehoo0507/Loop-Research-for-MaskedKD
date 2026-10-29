Read-only. Read harness/selector_api.py, harness/baselines.py and RESULTS_TABLE.md only (do not read
PROGRESS.md, RESULTS.md or PROPOSALS.md, so your ideas stay independent of the other researcher).

Task: propose exactly 2 token-selection methods for the masked teacher in ViT knowledge distillation,
each grounded in a published interpretability method, that could reach Full-KD accuracy (within 1 %p)
with fewer teacher tokens than MaskedKD (student last-block CLS attention top-K), counting the
selector's own FLOPs.

Format, exactly:
IDEA 1: <method name> — <how scores are computed, 2 lines> — basis: <interpretability paper> — closest prior token-pruning work and difference: <1 line> — extra FLOPs: <rough>
IDEA 2: ...
