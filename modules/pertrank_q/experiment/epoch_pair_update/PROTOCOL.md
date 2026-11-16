# Phase 2: jointly select an epoch pair for the deployed B ensemble

This is a separately frozen, post hoc development extension after phase 1.
Phase 1's prespecified primary result remains unchanged. No phase 1 file is
overwritten, and all failed results remain reportable.

The sole update is removal of the same-epoch constraint in two-seed ensemble
selection. For each source and outer target half, the existing five checkpoints
of each seed (0, 50, 100, 200, 300) form exactly 25 pairs. All 25 are evaluated
using the saved inner models. In each of the two inner target exchanges, the
two physical-score matrices are averaged before computing exact P@5. A pair
is eligible only if both exchange improvements over original PertRank are
nonnegative. Select the highest mean improvement; exact ties use smaller total
epochs, then smaller maximum epoch, then lexicographically smaller (e1,e2).
The five diagonal pairs must exactly reproduce phase 1's ensemble selection.

Primary arm B_free_pair uses the original frozen-scale whole-list objective.
Secondary arm B_rank_scale_free_pair uses phase 1's ranking-scale objective.
Both retain the original geometric anchor, query competition inputs and full
list goal. Network structure, training labels, tau, penalties, seeds, source
separation, normalization, target splits and candidate/query pools are fixed.
There is no new inner training and no new hyperparameter value beyond this
explicit 25-pair table. Expanding selection from five diagonal pairs to 25 pairs
increases the degrees of freedom on the same internal validation data and can
increase selection overfitting. No outer score is used for source-specific selection.

For each selected pair, reuse an existing outer fit whenever objective, seed
and epoch match exactly. Only missing fits may be trained, using the unchanged
phase 1 training implementation and permitted calibration half. All six new
prediction pools must be frozen together before phase 2 evaluation. Phase 1
and original B predictions are imported as fixed comparators.

The primary development gate remains: equal-source mean improvement versus
old_B strictly positive and all source improvements versus original PertRank
nonnegative. The stronger stability marker requires at least two sources with
conditional 95% interval lower bounds above 1e-12 versus original PertRank,
and both individual seeds' equal-source improvements positive. These are
exploratory development criteria, not independent confirmation or multiplicity
adjusted inference. All 25 choices, sources, seeds, and both new arms are
reported; the secondary arm cannot replace a failed primary endpoint.

Resource limit: this fixed pair table and its missing outer fits only. No new
objective, source, seed, temperature, feature or additional selection grid is
introduced after phase 2 scoring.
