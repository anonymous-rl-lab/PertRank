# One controlled B optimization experiment

This is an exploratory update on historically accessed references, not a new
independent confirmation. All four arms and both seeds remain in the report.

## Fixed comparisons

| Arm | List objective scale | Epoch selection |
|---|---|---|
| old_B | Frozen training baseline SD | Each seed separately; imported |
| B_joint_selection | Frozen training baseline SD | Same-epoch two-seed score ensemble |
| B_rank_scale | Current eligible prediction SD | Each seed separately |
| B_rank_scale_joint | Current eligible prediction SD | Same-epoch two-seed score ensemble |

Primary arm is B_rank_scale_joint; primary contrast is against old_B. The other
two changed arms separate the effects of objective and selection. Full criteria
and resource bounds are in EXPERIMENT_SCOPE.md. No arm is selected from outer
evaluation results; no additional tuning, temperature search or third factor
is introduced in this experiment.

## Invariants and training

The B architecture, 14 feature columns, source-specific normalization, original
geometric weight anchor, two seeds (2026111511 and 2026111512), target splits,
external donor-bootstrap support labels, tau=0.5, Adam learning rate 0.01,
gradient clipping at 5, score-residual penalty 0.05 and output-head ridge 0.0001
are unchanged. Three sources are fitted separately. Inner fits use K=5 and
outer fits use K=10. Self-query candidates are excluded in all objectives.

The rank-scale objective uses

    normalized_score = (score - eligible_mean(score)) / sqrt(max(
        eligible_population_variance(score),
        (1e-6 * frozen_training_baseline_SD)^2))

for each query. Gradients pass through the mean and variance. The fixed relative
floor protects constant rows; it is not optimized. The physical score residual
penalty remains in the original fixed training scale, and inference emits
physical signed scores rather than normalized scores.

Each inner fit runs 300 epochs, preserving checkpoints at 0, 50, 100, 200 and
300. Epoch zero is reconstructed from the exact initialization. Existing B
inner fits are replayed only to obtain missing early checkpoint predictions;
their final state, normalization and exact training/validation precision curves
must match the predecessor. No predecessor file is overwritten.

For separate-seed selection, each seed selects the largest mean validation
precision improvement subject to nonnegative improvements in both target
exchanges, with earlier epochs breaking exact ties. For ensemble selection,
the same rule is applied to the physical-score mean of both seeds at the same
epoch, not to a mean of their individual precision values. Each inner model has
its own training-only normalization. Epoch zero guarantees a valid fallback.

Outer fits use the selected epoch and the entire allowed calibration half.
Matching original B outer fits are reused when both seed and epoch match;
identical new objective/seed/epoch fits are cached across arms. Every source's
two complete candidate pools are predicted; cross-fold scores are never
concatenated into a single candidate pool.

All six predictions must be locked together before the new scoring program
opens evaluation references. This execution barrier does not erase historical
reference access. Prediction files have scores[arm, seed, query, target] and
retain exact candidate/query identities and self masks. Numerical tests cover
scale invariance, radial gradients, constant rows, self exclusion, permutation,
exact zero initialization and reference-independent normalization.
