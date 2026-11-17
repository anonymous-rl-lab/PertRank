# Fixed post hoc selection-baseline comparison

This plan is fixed before fitting the new baseline to the recovered experimental inputs or viewing its evaluation scores. The source studies and reference outcomes have already been analysed in PertRank; this is a post hoc comparator addition, not a new preregistered or unopened confirmation.

## Scientific question

How does the existing PertRank workflow compare with one published selection-oriented empirical-Bayes ranking workflow on the existing nomination tasks? The comparison is between complete workflows. Geometry-specific attribution continues to use the existing matched PertRank-minus-Euclidean contrasts.

## One comparator

Use the original R package rvalues 0.7.1 (Henderson and Newton, Making the cut, DOI 10.1111/rssb.12131), Gaussian sampling model, conjugate Gaussian effect prior, estimated hyperparameters, and no smoothing. Use the package defaults for remaining algorithmic arguments, record them from the installed function, and do not compare prior families or choose configurations using reference scores.

For each query gene and direction, fit on the eligible candidate axis using the original shallow effect and its marginal standard error. Reverse the effect sign for down-regulation. Remove the own-target candidate before fitting and nomination. Rank by ascending native r-value; retain exact ties for the existing expectation-based scorer rather than using an arbitrary ordinal rank.

The input variance is target sample variance / target-cell count plus control sample variance / input-control count, when available in the original statistics. The shared control mean creates cross-candidate covariance. The native method is applied as a marginal working model, without claiming a correctly specified joint candidate posterior or joint optimality. No independent resampling of a separate control mean for each candidate is introduced.

## Predefined numerical handling

- Verify finite effects, finite strictly positive standard errors, candidate identities, query identities, and original normalization scales before fitting.
- If the native fitted prior variance is exactly zero, the conjugate posterior degenerates to a common point. All candidates tie. Report this as a predefined degenerate-prior boundary convention, not a successful native r-value calculation; count these fits separately.
- Other fit failures are retained as failures. Do not switch prior family, adjust parameters based on ranking performance, or substitute another comparator. Resolve verified coding or input errors with an explicit correction record.
- Unit checks use synthetic arrays and input-contract checks only, without reference labels.

## Fixed tasks and scoring

- D4 Rest and D4 stimulated state: original 300 responsive queries per state, five original disjoint shallow input groups, 20 target cells per input, original full candidate pools, K=25.
- External knockout RNA: original D3, D1, D2 inputs; 293, 289, 289 queries and 52, 59, 63 candidates, respectively; original 20-cell inputs, K=10, one shared independent RNA reference.
- Retain original query directions, masks, candidate eligibility, normalization, reference boundary memberships, exact tie handling, query blocks, query weighting and existing estimator predictions.
- D4 intervals: original conditional query-block procedure, 2,000 draws and seed 20261012; equal-state average follows the existing independently spawned state resampling.
- External intervals: original query-block procedure, 2,000 draws and seed 2026100502; preserve original equal-source aggregation and identify the shared reference.
- Validate the recovered evaluation inputs by reproducing existing stored method scores before scoring rvalues.

## Outputs and stopping rule

Save input identities/hashes, software source/version, exact native configuration, fitted hyperparameters, per-query r-values, numerical diagnostics, per-query nomination scores, complete task summaries, paired intervals, and contrasts against PertRank, Euclidean, native FR and ASH where archived predictions permit.

Report every planned task and numerical failure. A difference in P@K times K is the expected difference in reference-supported members per list. Do not use statistical significance or which method wins to decide whether to retain a task. Stop after this one comparator and its correctness checks; no new model development or search for a favourable dataset is part of this experiment.

The smallest portable deliverable will retain query-level sufficient statistics and fixed reference/scoring inputs, not raw single-cell matrices or redundant historical packages.
