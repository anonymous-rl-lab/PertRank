# TRADE numerical sensitivity: separate code review

Review date: 2026-10-11 UTC. Scope: read-only inspection of the sensitivity source, orchestration, collection and replay code, plus checks of saved input identifiers. The full cohort was still running at review; no ranking was computed by this review.

## Decision

The numerical rule and scoring implementation are coherent. The sensitivity is a **post hoc uniformly regularized numerical analysis**, and the original registered H2b remains **not evaluable**. Full-cohort completion and numerical output validation remain prerequisites for reporting its ranking results.

## Verified

1. A direct diff against the frozen `R_TRADE_univariate.R` found exactly two substantive additions: `control=list(eps=1e-4)` in `ash()` and a guard requiring `ash_model$fit_details$optreturn$converged`. The half-uniform family, uniform prior, adaptive component grid and range are preserved. The original adapter, analytic moment formulas and seed 2026 are used.
2. The installed ashr wrapper passes an original default `eps=1e-6` to mixSQP. The retained sensitivity passes `eps=1e-4` uniformly. The optimizer reference shows an additive epsilon inside row-mixture likelihoods in the numerical objective. This changes the optimized objective slightly; it is not an unchanged-estimator rerun or a stronger convergence tolerance. The full-cohort protocol describes this accurately.
3. The Python runner verifies the original source hashes and runtime, finite input effects and positive standard errors, unique axes, output target/replicate/gene counts, valid mixture probabilities and intervals, and independent analytic first/second/variance moments. It checks the resulting scalar against `S/(S+nu)`. Convergence failure raises an exception and prevents a batch completion record.
4. The full runner schedules exactly 561 targets in five groups (2,805 fits). It permits scoring only after every batch has completed. There is no omitted-target, successful-subset or substitute-weight analysis in the retained rule.
5. The collector's score dimensions are arm × five groups × query × direction. Its contrast reduction averages the five groups and two directions, retaining one paired contrast per query. It then uses the frozen 30-block resampling procedure (2,000 draws, seed 2026). These are descriptive intervals for the post hoc sensitivity, rather than a new registered H2b decision.
6. The compact replay reconstructs rankings from original effects, reference and saved scalar weights, then reproduces the stored paired intervals. The original four arms are compared against their registered score arrays.
7. Separate axis-only checks found exact equality of the prediction, query-context and reference target/query axes. All 561 full-axis input files have the same 13,394 gene identifiers, and each target identifier matches the query-context candidate order. No fits or outcome scoring were run for these checks.

## Collection hardening verified after initial review

Before collection was run, the collector was updated to verify every output hash in each completion record, require identical numerical identities across batches, require `eps=1e-4`, and compare the sensitivity-source hash with the actual R file. It now also checks prediction/query/reference axis equality, replicate labels 0–4, and each target's complete gene axis against the corresponding frozen input file. These changes address the principal hardening items below without changing the running fits or the scoring rule.

## Initial small hardening items and remaining observations

- The initial collector read hashes and identities in `COMPLETE.json` without verifying them itself. This is now resolved by the checks described above.
- Query-context axis equality and replicate labels are now asserted. Finite reported log likelihoods and the row-to-target/replicate mapping remain suitable explicit checks in the final output review. The independent axis checks above establish that the present assets align.
- The collector's `i//5` mapping assumes the frozen target-major, replicate-0-to-4 row order. This is the order emitted and validated by the runner. An explicit assertion makes the dependency visible.
- Exact moment reproduction validates mixture arithmetic, not equality to the original default-epsilon estimator. The smoke comparison and its observed differences should remain in the supplementary record.

## Reporting language

Recommended: “A post hoc numerical sensitivity fitted every target and input group with the same increased mixSQP likelihood regularizer (`eps=10^-4`, original default `10^-6`), retaining the mixture family, prior, grid, input genes and scalar-output rule. All fits were required to converge before scoring.”

Avoid describing this as recovery of the original confirmatory H2b, an exact rerun of an unchanged objective, or evidence that all numerical differences are negligible. Earlier unsuccessful numerical candidates were selected using convergence and successful-fit comparisons; their selection record appropriately states that ranking outcomes were not used.

## Completion status

Source and scoring-code review: **PASS**. Saved-axis audit: **PASS**. Full-cohort fit/result verification: **PASS** (completion addendum below).

## Completion addendum

All 2,805 fits completed. A separately written implementation (`review_trade_scores_separate.py`) used full sorting to locate the 25th eligible candidate, assigned exact fractional membership to boundary ties, and excluded each query's own target. It imported no production scoring function. It independently computed 5,940 scores for each of TRADE and PertRank across 594 queries, five groups and both directions. Maximum absolute differences from the saved scores were `9.02e-17` (TRADE) and `2.50e-16` (PertRank).

A separate block-sum/block-count bootstrap implementation reproduced the key contrasts:

| Panel | TRADE mean P@25 | PertRank mean P@25 | PertRank − TRADE | Paired 95% interval |
|---|---:|---:|---:|---|
| Responsive | 0.36901950 | 0.39372749 | 0.02470800 | [0.02248536, 0.02724906] |
| Terciles | 0.12120570 | 0.12948905 | 0.00828334 | [0.00463881, 0.01110911] |

The largest interval/point discrepancy was `6.94e-18`. The fit table contains 2,805 unique target/group pairs, all with finite log likelihoods and 13,394 genes. The machine-readable checks and input hashes are saved in `TRADE_SEPARATE_SCORE_REVIEW.json`.

These results establish the complete-cohort post hoc sensitivity result. The original registered H2b status remains unchanged.
