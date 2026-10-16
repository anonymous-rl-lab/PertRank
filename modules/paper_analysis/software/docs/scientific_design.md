# Reliability, readout and candidate selection

PertRank answers a gene-specific decision question: which perturbations should occupy a fixed number of follow-up positions for an increased or decreased response? The software keeps reliability calculation separate from the effect estimate used to answer that question. This is the design evaluated in manuscript v25.2.

## One weight from profile-wide information

For target `j`, let `y_j` be its normalized mean expression minus the matched-control mean, and let `V_j` be the estimated covariance of that difference. Target and control sample variances both contribute to `V_j`. The control matrix supplies five orthonormal leading directions, the columns of `U`.

With `M = I − 0.9 U Uᵀ`, the two reliability weights are

```text
Euclidean:  w0_j = clip(1 − trace(V_j) / (y_jᵀ y_j), 0, 1)
PertRank:   wM_j = clip(1 − trace(M V_j) / (y_jᵀ M y_j), 0, 1)
```

Zero usable energy receives weight zero. The implementation uses total energies, total variance contributions and their projections onto the five directions, avoiding a dense gene-by-gene matrix for each target.

“Profile-wide” describes the effect and variance information used to assess reliability. The directional correction uses the five fixed leading directions, rather than all covariance eigenvectors. Controls determine those directions; each candidate's measured effect and sampling variation determine its weight.

## A gene-aligned readout

For query gene `g`, the up score is `w_j × effect[j, g]`; the down score is its negative. A positive scalar preserves the signs and relative gene-effect pattern within a target while changing its competitive strength against other targets. Zero-weight candidates remain in the eligible pool with score zero.

The default cell-input route uses the original target-minus-control gene effects. `rank_effects` can instead use an upstream estimator's labeled gene effects while keeping the fitted weights fixed. The manuscript's FR reuse comparison addresses this estimator choice. It differs from the matched readout experiment, which applies a specified transformation to the same effect vector. Changing an upstream estimator and changing readout coordinates are distinct comparisons.

## Inspect the added geometry separately

Every ranking returns three matched arms:

| Contrast | Interpretation |
|---|---|
| PertRank − Unweighted | Complete reliability-weighting workflow |
| Euclidean − Unweighted | Target-level shrinkage without directional geometry |
| PertRank − Euclidean | Added control geometry, with the effect estimator held fixed |

Apply the same contrasts after `rank_effects` to inspect reuse with an external estimator. Relative differences between the returned scores are not performance estimates: performance requires a separately defined reference and the same eligible candidate and query sets.

The manuscript connects controlled weight–readout comparisons, fixed-setting task results, reuse with FR estimates and one independent knockout RNA reference. Its complete-workflow and geometry-specific results have different scopes. The tutorials demonstrate the archived computations; they do not conduct new external validation or establish an expected gain for a new screen.

## Evaluate the decision actually being made

The ranking operation excludes the queried gene's own perturbation and supplies both a reproducible delivered list and exact expected inclusion for ties at the cutoff. These serve different purposes: the delivered list can be inspected or exported, while expected inclusion supports tie-aware reference scoring.

For a new benchmark, define the candidate pool, queries, response directions, reference and list length before scoring. `evaluate_reference` supplies per-query, per-direction precision against a labeled reference, with exact expected overlap under independent boundary-tie selection. Evaluate every method on the same choices and preserve query-level pairing when estimating contrast uncertainty. The full scientific reproduction bundle supplies the manuscript's reference construction, aggregation and paired resampling analyses. A nomination table alone is not a biological accuracy estimate.
