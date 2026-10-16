# PertRank D4 held-out donor confirmation: experimental results and scientific analysis

Version: complete D4 results archive, 2026-10-08 (Beijing time). This report summarizes the completed frozen analyses. It adds no experiments and does not reselect methods or endpoints.

Upload destination: the `result/` subdirectory of the original `REST_ASSOCIATED_PROJECT`. The associated original registration is `REST_PRIMARY_REGISTRATION`. This results archive is not a new registration.

## 1. Main findings

**D4 met the prespecified primary confirmation and improvement criteria. In the subsequently activated competitive comparison, PertRank outperformed the preselected PR_M method.** On the responsive panel, the P@25 gain was **+0.02352 [95% conditional interval +0.02142, +0.02631]** over Euclidean weighting, **+0.01024 [+0.00772, +0.01270]** over PR_M, and **+0.07742 [+0.06758, +0.08444]** over unweighted ranking. When the five real-cell input groups were evaluated separately, both the point estimate and the interval lower bound were positive for all three comparisons in every group.

This experiment extends the evidence from competitive performance in development donors to confirmation in a held-out donor after the method and competition rules were fixed. It supports reproducible gains within the same study, in resting T cells, for the specified queries and candidate pool. It does not recast within-study donor confirmation as cross-laboratory or cross-guide validation.

## 2. Experimental design and methods

### 2.1 Data, cell partitions and budget

The data source was `D4_Rest.assigned_guide.h5ad` from the public T-cell Perturb-seq resource. **589 targets** met the prespecified requirement of at least 300 eligible cells. All were included under the `min(1200, N)` rule; targets were not selected according to observed performance. The weight-gene axis comprised the **13,209 genes** detected in the input non-targeting controls.

| Resource | Cells per target or control pool | Purpose |
| --- | ---: | --- |
| Planning: positions 0–99 in the fixed within-target cell order | 100 per target | Select query panels and form query blocks; excluded from the five shallow inputs and deep reference |
| Prediction: positions 100–199 | 100 per target, divided into 5 groups of 20 | Re-estimate weights and produce nominations separately in each group; cells are disjoint between groups |
| Truth: positions 200–299 | 100 per target | Construct deep-profile reference lists; read after the predictions were locked |
| Input NTC | 2,000 | Input centering, weight-gene set, control directions and noise statistics |
| Evaluation NTC | 2,000 | Deep-reference centering; disjoint from the input NTC |

Thus, “20 cells” refers to the target-cell budget for each ranking input. Planning and control resources are additional and explicitly accounted for. The five groups contain different real cells, rather than bootstrap samples drawn with replacement from one group. They are not five independent donors.

The primary responsive panel contained 300 queries selected from the planning data under the frozen rule and organized into 30 nonempty query blocks. During query selection, direct self-gene perturbation entries were excluded using the correct Ensembl mapping. During scoring, self-target candidates were again excluded by gene ID. The tercile panel also contained 300 queries and 30 blocks; the two panels together contained 593 distinct queries. Each query was evaluated in both the positive and negative ranking directions. All methods used the same candidate pool, deep reference and exact tie-expectation scorer.

### 2.2 The four tested arms and the meaning of PR_M

| Method | How the weight was calculated | Effect profile used for nomination |
| --- | --- | --- |
| raw | No weighting | Original log-normalized mean difference `y` |
| Euclidean | Overall observed energy and noise subtraction in the original log coordinates | `w_E × y` |
| PertRank | Energy and noise estimated in the original log coordinates using the first 5 directions learned from input NTC to construct `M = I − 0.9 UUᵀ` | `w_M × y` |
| PR_M | Scalar weight estimated in Pearson-residual coordinates under the analogous low-rank metric, using the first 5 NTC directions in those coordinates | `w_PR,M × y`, retaining the original log-effect output |

The scalar followed the frozen clipped form `w = clip(1 − noise / energy, 0, 1)`. Directional noise included uncertainty in both the target mean and the input-control mean, using per-cell projection variances. Per-gene variances were not substituted for missing off-diagonal information.

**PR_M is a scalar competitor that combines Pearson residuals with a low-rank control metric. It is neither Pearson normalization alone nor Euclidean shrinkage alone.** Expected-count proportions for the Pearson residuals were estimated exclusively from aggregate input NTC counts. The calculation used each cell's UMI total over the full gene axis, fixed `theta = 100`, and residual clipping at `±sqrt(2000)`. PR_M directions were also learned exclusively from input NTC; `k = 5` and `beta = 0.9` were not tuned on D4. All four arms retained the same original log output coordinates, separating the effect of weighting from differences in output scale.

Competitor selection was completed before D4 access. For each eligible scalar arm in D3, corrected-query D1 and D2, scores were first averaged across the five input groups and two directions, then across queries, and finally equally across the three donors. Eligible arms were Euclidean, diagonal, TRADE_scalar, PR_E and PR_M. PR_M was the unique winner with mean P@25 = 0.3168875; TRADE_scalar scored 0.3138706. D4 tested this preselected competitor, without rerunning the competitors on D4 and then choosing a favorable opponent. The conclusion is therefore superiority to **the competitor preselected under a fixed development rule**, not superiority to every external method on D4.

### 2.3 Primary estimand and decision criteria

For each query in the primary responsive panel, the PertRank-minus-Euclidean P@25 difference was first averaged across the five input groups and the positive and negative directions, and then averaged across queries. Intervals were two-sided 95% percentile intervals from 2,000 paired query-block bootstrap draws, using fixed random seed 2026. The same resampling draw was applied to every method. Both predictions and reference lists used exact boundary-tie membership probabilities, rather than breaking ties by candidate order.

The frozen criteria were:

- Primary confirmation: the PertRank-minus-Euclidean point estimate must be at least +0.01, with an interval lower bound above zero. Confirmation is disallowed if all PertRank weights are zero in any input group.
- Improvement grade: the same comparison must have a point estimate of at least +0.02, with an interval lower bound above zero. The +0.02 threshold applies to the point estimate.
- Competitive layer: activated only after primary confirmation. For PertRank minus the locked competitor, a lower bound above zero indicates superiority; an upper bound below zero indicates inferiority; otherwise, the comparison is unresolved.
- Practical approximate equivalence: assessed separately, requiring the entire 95% interval to lie within ±0.005. This label is distinct from the directional test.

## 3. Primary responsive-panel results

| Method | Mean P@25 across five groups | Expected number of overlaps per 25 nominations with the deep-reference top-25 |
| --- | ---: | ---: |
| raw | 0.30175 | 7.544 |
| Euclidean | 0.35565 | 8.891 |
| PR_M | 0.36894 | 9.223 |
| PertRank | 0.37917 | 9.479 |

Random P@25 was 0.04245. This was calculated from the actual number of eligible candidates for each query, including the effect of self-target exclusion.

| Comparison | P@25 difference and 95% conditional interval | Additional overlapping members per 25 nominations |
| --- | --- | ---: |
| PertRank − Euclidean | +0.02352 [+0.02142, +0.02631] | 0.588 |
| PertRank − PR_M | +0.01024 [+0.00772, +0.01270] | 0.256 |
| PertRank − raw | +0.07742 [+0.06758, +0.08444] | 1.936 |

**Under the frozen rules, primary confirmation passed, the improvement grade passed, the competitive gate opened, and PertRank was superior to PR_M.** The competitive interval lay entirely above +0.005, so the comparison was not classified as practically approximately equivalent. This reflects an advantage exceeding the equivalence band, not a competitive failure. No input group had all-zero PertRank weights.

### The five real-cell input groups

Each interval below is a paired query-block 95% conditional interval for the indicated input group.

| Real 20-cell input group | PertRank − Euclidean | PertRank − PR_M | PertRank − raw |
| --- | --- | --- | --- |
| 1 | +0.01819 [+0.01320, +0.02374] | +0.00968 [+0.00536, +0.01435] | +0.07500 [+0.06437, +0.08270] |
| 2 | +0.01532 [+0.00984, +0.02086] | +0.01115 [+0.00651, +0.01537] | +0.07439 [+0.06537, +0.08428] |
| 3 | +0.02835 [+0.02490, +0.03301] | +0.01117 [+0.00710, +0.01465] | +0.08028 [+0.07086, +0.08818] |
| 4 | +0.02843 [+0.02357, +0.03319] | +0.01075 [+0.00613, +0.01525] | +0.08300 [+0.07084, +0.09201] |
| 5 | +0.02733 [+0.02340, +0.03144] | +0.00844 [+0.00409, +0.01312] | +0.07444 [+0.06055, +0.08477] |

All five intervals were above zero for each of the three comparisons. The observed advantage was therefore not supported by only one particular 20-cell input group. The formal primary interval nevertheless averaged the five groups before resampling query blocks; it did not treat the 5 × 30 query blocks as mutually independent.

## 4. Tercile panel: prespecified descriptive results

| Method | Mean P@25 |
| --- | ---: |
| raw | 0.07845 |
| Euclidean | 0.10519 |
| PR_M | 0.10890 |
| PertRank | 0.11063 |

| Comparison | P@25 difference and 95% conditional interval |
| --- | --- |
| PertRank − Euclidean | +0.00544 [+0.00296, +0.00764] |
| PertRank − PR_M | +0.00173 [+0.00041, +0.00299] |
| PertRank − raw | +0.03218 [+0.02147, +0.04156] |

The metric increment was substantially smaller on the tercile panel than on the primary responsive panel. The interval for the mean PertRank-minus-PR_M difference was above zero but also entirely within ±0.005. Under the same numerical definitions, this can be described as both statistically positive and practically approximately equivalent. Its status remains descriptive; it is not an additional primary confirmation.

Individual-group results are shown below. In particular, every single-group interval for PertRank minus PR_M on the tercile panel included zero. The statement that every group excluded zero applies to the primary responsive panel and must not be extended to this panel.

| Real 20-cell input group | PertRank − Euclidean | PertRank − PR_M | PertRank − raw |
| --- | --- | --- | --- |
| 1 | +0.00459 [+0.00245, +0.00675] | +0.00204 [-0.00011, +0.00408] | +0.03402 [+0.02122, +0.04538] |
| 2 | +0.00486 [+0.00212, +0.00714] | +0.00222 [-0.00017, +0.00487] | +0.03157 [+0.02128, +0.04012] |
| 3 | +0.00460 [+0.00060, +0.00819] | +0.00067 [-0.00149, +0.00262] | +0.03083 [+0.01957, +0.04079] |
| 4 | +0.00772 [+0.00322, +0.01198] | +0.00226 [-0.00006, +0.00450] | +0.03129 [+0.02003, +0.04112] |
| 5 | +0.00543 [+0.00165, +0.00890] | +0.00147 [-0.00083, +0.00343] | +0.03317 [+0.02173, +0.04310] |

## 5. Secondary cell equivalence and practical effect size

Within the same fixed 100-cell input reservoir, raw inputs of 20, 25, 30, 35, 40, 45 and 50 cells were constructed using the locked anchor and additional-cell rules. All measured depths were retained. Equivalent depth was estimated using the prespecified piecewise-linear inverse mapping, without outcome-dependent smoothing or extrapolation.

| Number of raw input cells | Responsive-panel P@25 |
| ---: | ---: |
| 20 | 0.30175 |
| 25 | 0.32840 |
| 30 | 0.34913 |
| 35 | 0.36719 |
| 40 | 0.38266 |
| 45 | 0.39420 |
| 50 | 0.40505 |

| 20-cell method | Equivalent number of raw cells | 95% conditional inverse-mapping interval | Measured-depth bracket containing the crossing |
| --- | ---: | --- | --- |
| Euclidean | 31.81 | [29.02, 34.12] | 30–35 |
| PertRank | 38.87 | [36.10, 41.33] | 35–40 |

All 2,000 responsive-panel bootstrap draws identified both crossings within the tested depth range. No unreached or boundary draws were discarded. The correspondence between 20-cell PertRank and approximately 38.9 raw cells is an interpretable equivalence in ranking performance. It is conditional on the current planning, control, candidate-pool, query and deep-reference resources; it does not imply an equivalent reduction in total experimental cells or sequencing costs. Piecewise-linear interpolation on a concave depth curve can overestimate the crossing.

The raw tercile-panel curve was comparatively flat and did not reach either the Euclidean or PertRank 20-cell score within the measured 20–50-cell range. A numerical equivalent cell count therefore cannot be given for this panel. The complete results retain all bootstrap statuses; they were not extrapolated into a claim of “more than 50 cells” or an unbounded benefit.

## 6. Weights, stability and scientific interpretation

| Method | Zero-weight fractions across five groups | Mean weights across five groups |
| --- | --- | --- |
| Euclidean | 26.3% / 28.7% / 26.1% / 24.4% / 23.9% | 0.0948 / 0.1008 / 0.0988 / 0.0989 / 0.0988 |
| PertRank | 11.9% / 12.6% / 10.0% / 8.0% / 11.2% | 0.0712 / 0.0722 / 0.0711 / 0.0717 / 0.0698 |
| PR_M | 7.5% / 10.7% / 8.7% / 7.5% / 8.1% | 0.0888 / 0.0900 / 0.0887 / 0.0892 / 0.0868 |

PertRank weights did not collapse to all zero or otherwise degenerate in this analysis. Their mean was lower while fewer targets had zero weight, showing that a higher overall mean weight cannot explain the result. The evaluated quantity was the complete reranking for each query and the resulting exchange of candidates within a fixed-length list.

D4 adds three layers of evidence to the paper's claims:

1. **Reproducible use of the additional metric.** After freezing the method, parameters, candidate eligibility and query rules, the held-out donor still showed a gain of at least +0.02 over Euclidean weighting. This addresses whether the development-stage increment depends entirely on selection in the development data.
2. **A substantive competitive comparison.** PertRank outperformed PR_M, selected from five eligible scalar methods during development. PR_M combines control information, Pearson processing and low-rank geometry. The result shows that fixed PertRank retained an increment over this competitor in the tested task, with raw ranking serving as an additional rather than the sole comparator.
3. **An interpretable effect size.** Per list of 25 nominations, PertRank recovered approximately 0.59 more deep-reference members than Euclidean weighting, 0.26 more than PR_M and 1.94 more than raw ranking. The secondary cell-equivalence analysis expresses performance on the same ranking task in input-depth units.

These are empirical findings about average list quality. They do not uniquely establish a mechanistic explanation. In particular, PR_M itself contains a low-rank metric. PertRank minus PR_M therefore cannot isolate a claim that all gains arise from off-diagonal structure, nor establish that every Pearson preprocessing approach is inferior to PertRank.

## 7. Scope and conclusions that remain unchanged

- This experiment adds one new donor from the same study, using resting D4 T cells and an RNA readout. It adds no laboratory, platform, stimulation condition or functional endpoint.
- Intervals are conditional on the selected input groups, input controls, planning-derived panel, candidate pool and deep reference. Real-cell replacement demonstrates between-group input stability under fixed resources; it does not resample the donor population, control sets or evaluation reference.
- The primary query panel was derived from an independent planning block and therefore shared no cells with the five input groups or the deep reference. Deployment still assumes that planning and control resources are available.
- The score measures recovery of deep-reference lists from the same resource. D4 did not use a cross-guide reference. It does not remove the transfer limitations revealed by earlier guide-separated analyses, or identify shared-guide agreement as purely target-gene effects.
- This analysis adds no protein, functional or wet-laboratory confirmation. An increase in P@25 means more members of the deep-reference list, not necessarily an equal proportional increase in discovered true regulatory relationships.
- Tercile-panel increments were smaller, and individual-group PR_M comparisons were unresolved. Cell equivalence is a secondary estimate and does not determine primary confirmation.

## 8. Locks, archiving and verification

The method, parameters, cell partitions, competitor selection and decision rules were locked before the first access to D4. Once prediction was complete, the actual prediction files, queries, control model, cohort and file hashes were saved in a durable prediction archive before access to the deep reference was opened. The prediction lock bound 101 files. Eight scientific computation files retained their original frozen hashes. Mechanical verification corrections during execution and changes to the access-archiving workflow were documented separately.

The original registration attachment specified subsequent separate registrations for an addendum and a prediction lock. Following the author's later explicit instruction, execution instead used an addendum in the original OSF Project and a durable prediction archive; this is disclosed as a procedural archiving change. **This report does not claim that two additional OSF registrations were approved.** The current `result/` directory archives completed results within the project and is not retrospectively presented as preregistration. Original frozen documents retain the status language used when they were created; the actual execution status should be read alongside the subsequent archiving and access-release records.

After formal scoring, an independently implemented numerical audit recomputed complete ranking tie probabilities, directions, self-target exclusions, paired block bootstrap results and inverse-depth mappings without importing the formal scoring functions. **The audit passed 1,426 checks covering 8,861,488 numerical comparisons, with a maximum absolute difference of 4.97 × 10⁻¹⁴.** This was an actual numerical rerun, not merely a file-hash check. Records of pre-prediction and truth-integrity checks are also retained in the experimental archive.

The numerical values in this report come from the following completed outputs:

- `results/D4_RESULT.json`: formal scores, decisions, group-specific results, weights, depth curves and secondary analyses.
- `results/query_scores_and_depth_bootstrap.npz`: query-level scores and depth-bootstrap arrays.
- `results/INDEPENDENT_NUMERIC_AUDIT.json`: independent numerical audit.
- `competition_selection/COMPETITION_SELECTION_LOCK.json`: competitor selection completed before D4 access.
- `protocol/D4_PROSPECTIVE_PROTOCOL_FINAL.md` and subsequent archiving amendments, locks and access records: the protocol and execution trail.

Paths are relative to `v18_d4_confirmation/` in the complete experimental code-and-data archive. The original numerical outputs and frozen files remain unchanged; this report is a readable synthesis.

## 9. Scope of this four-donor reproduction package

This report is included in the combined D1–D4 GitHub reproduction package. The original D4-only OSF package remains a separate archived deliverable. In the current package, `reference_results/D4.json` contains the unchanged official D4 result, while the original scientific source paths cited above identify its provenance.

The combined package recalculates the D4 scalar weights from retained sufficient moments. For each query and input group, it retains original float64 effects on the union of all frozen methods' signed top-25 lists, including every boundary tie, and the reference tails. Multiplication by the verified frozen weights and reranking this support reproduces the original scores exactly. Interior candidates omitted from this representation cannot support arbitrary new weights, list sizes or query sets.

The raw depth ladder retains exact top-list boundary and tie information sufficient to reproduce its scores and inverse-depth estimates. It does not retain complete higher-depth expression magnitudes or all within-interior rankings. Replay does not reread raw UMI matrices, refit control PCA directions or repeat full planning-gene selection. See [the repository README](../README.md) for the precise reproduction scope. The original complete archive is preserved separately.

Official results-file SHA256: `e89aab5d08a103b3db37b57a230387726bb7933af60328fc6cfa50e36c52e4d0`.
