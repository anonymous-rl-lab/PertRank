# Four-donor results and analysis

This report summarizes the current, archived 20-cell nomination analyses for D1–D4. Numerical sources are `reference_results/D1.json` through `D4.json`. D1 uses the completed full-gene-axis query correction, which replaced four responsive queries; historical D1 panels and cell-equivalent estimates are not substituted for this corrected analysis.

## Scientific question and evidence status

PertRank estimates a scalar weight for each perturbation using a control-informed metric, then applies that weight to the original gene-coordinate effect profile. The central comparison asks whether this metric improves nomination beyond Euclidean weighting. Comparisons with unweighted ranking assess the benefit of the complete weighting procedure.

D3, D1 and D2 were already opened when the complete seven-arm, five-group competition summarized here was performed. These results are development-stage comparisons, including the deterministic D1 correction; this package does not relabel them as new blinded confirmations. D4 is a locked-method, prospective held-out donor from the same study. It tests the frozen PertRank method against Euclidean weighting and a competitor selected before D4 access.

## Shared evaluation

Each donor contributes five mutually disjoint groups of 20 perturbation cells per target. Weights are estimated separately in each group. Each query is evaluated in both directions against the donor's independent deep-reference cells, with the query's own perturbation excluded. Exact boundary-tie expectations are used on both prediction and reference sides.

The score is P@25: the expected fraction of a 25-member nomination list shared with the deep-reference top 25. The two directions and five input groups are averaged within each query before calculating the reported aggregate comparisons. Intervals are the archived, nominal two-sided 95% paired query-block percentile intervals, using 30 query blocks, 2,000 bootstrap draws and seed 2026.

These intervals are conditional on the realized input groups, controls, query panel, candidate pool and reference. Five groups are five real-cell substitutions, not five independent donors. No pooled donor confidence interval is reported. The 20-cell budget refers to prediction input per perturbation; planning, control and evaluation cells remain additional resources.

## Responsive-panel performance

The following are five-group average scores. All methods use the same original effect coordinates for nomination; their scalar weights differ. PR_E and PR_M estimate weights in Pearson-residual coordinates using an Euclidean or low-rank control metric, respectively. They do not replace the final output by Pearson residuals. The diagonal arm uses control-gene variances; TRADE scalar uses the specified perturbation-level adaptation of TRADE estimates.

| Donor | Unweighted | Euclidean | Diagonal | TRADE scalar | PR_E | PR_M | PertRank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D3 | 0.28307 | 0.28177 | 0.27851 | 0.29128 | 0.28989 | 0.30137 | 0.31310 |
| D1 | 0.25568 | 0.28725 | 0.28766 | 0.29362 | 0.29465 | 0.30565 | 0.31337 |
| D2 | 0.33769 | 0.34826 | 0.34910 | 0.35671 | 0.34517 | 0.34365 | 0.36346 |
| D4 | 0.30175 | 0.35565 | Not tested | Not tested | Not tested | 0.36894 | 0.37917 |

D4 tested only the four frozen arms shown. Missing D4 entries are untested comparisons, not failures or tied results. Absolute scores should not be interpreted as a donor difficulty ranking because candidate pools and query panels differ.

| Donor | PertRank − Euclidean | PertRank − PR_M | PertRank − unweighted |
| --- | --- | --- | --- |
| D3 | +0.03133 [+0.02206, +0.03964] | +0.01173 [+0.00888, +0.01432] | +0.03003 [+0.02040, +0.03725] |
| D1 | +0.02612 [+0.02132, +0.03074] | +0.00772 [+0.00608, +0.00937] | +0.05769 [+0.04985, +0.06527] |
| D2 | +0.01519 [+0.01156, +0.01890] | +0.01981 [+0.01675, +0.02361] | +0.02576 [+0.01549, +0.03449] |
| D4 | +0.02352 [+0.02142, +0.02631] | +0.01024 [+0.00772, +0.01270] | +0.07742 [+0.06758, +0.08444] |

Multiplying a precision difference by 25 gives additional expected deep-reference members per list. Against Euclidean weighting, the gains are **0.783, 0.653, 0.380 and 0.588** members for D3, corrected D1, D2 and D4, respectively. Against unweighted ranking, they are **0.751, 1.442, 0.644 and 1.936**. These are improvements in reference-list overlap, not counts of experimentally established causal relationships.

Within each of the four donors, all five responsive-panel input groups have positive PertRank-minus-Euclidean differences and conditional interval lower bounds above zero. This supports repeatability across the tested real-cell input groups with their shared controls, panel and reference fixed.

## Competitive evidence and D4 decision

Across the opened donors, PertRank exceeds the TRADE scalar arm by **+0.02182 [+0.01449, +0.02793]** in D3, **+0.01975 [+0.01443, +0.02486]** in corrected D1, and **+0.00674 [+0.00329, +0.01008]** in D2. These five-group results supersede interpretations based on only one TRADE input group. PertRank also exceeds the diagonal arm in all three responsive panels. Thus the tested variance-scaling and TRADE scalar alternatives do not recover the entire PertRank increment in this development comparison.

The D4 competitor was chosen among Euclidean, diagonal, TRADE scalar, PR_E and PR_M by the highest equal-donor mean responsive P@25 across D3, corrected D1 and D2. Each donor first averaged five groups and two directions within queries. PR_M scored **0.3168875**, ahead of TRADE scalar at **0.3138706**, and was the unique selected competitor. PertRank was not eligible for this selection. The rule, corrected inputs and selection were locked before D4 access.

For D4, confirmation required a PertRank-minus-Euclidean point estimate of at least 0.01, a positive lower 95% bound, and no all-zero PertRank input group. The improvement grade required a point estimate of at least 0.02 with the same positive lower bound. **D4 meets both criteria**, with +0.02352 [+0.02142, +0.02631]. The primary pass opened the hierarchical competition gate; PertRank then exceeded PR_M by **+0.01024 [+0.00772, +0.01270]**, or 0.256 additional reference members per 25. The interval does not lie within the prespecified ±0.005 practical-equivalence band; here that reflects a resolved advantage, not an adverse outcome.

This is evidence against the preselected alternative in an unseen donor, not a D4 benchmark against every development arm. Because PR_M itself uses a low-rank control metric, this contrast does not isolate non-diagonal geometry alone or rule out every possible normalization strategy.

## Scope beyond the responsive panel

On the expression-tercile panel, the five-group PertRank-minus-Euclidean differences are smaller: **+0.00528** in D3, **+0.00417** in corrected D1, **+0.00328** in D2 and **+0.00544** in D4. Their conditional intervals exclude zero, but these panels do not define the D4 primary confirmation. PertRank-minus-PR_M is +0.00118 to +0.00314 across the four donors; each interval lies within ±0.005. Small positive differences and practical equivalence can therefore coexist.

D4's secondary responsive-panel depth analysis places 20-cell PertRank at an unweighted ranking-input equivalent of **38.87 cells [36.10, 41.33]**; the Euclidean equivalent is **31.81 [29.02, 34.12]**. These quantities use measured depth anchors and linear interpolation. They describe conditional ranking-input equivalence, not total experimental-cell or sequencing-cost savings; interpolation on a concave curve can overestimate the crossing. D1's corrected-panel result is not paired with a historical, different-panel depth estimate here.

## What this release establishes

The development comparisons show a reproducible metric increment across the tested shallow T-cell input groups. The frozen D4 analysis adds prospective same-study donor evidence both beyond Euclidean weighting and beyond the preselected Pearson-residual competitor. The benefit remains task dependent and is markedly smaller on tercile panels.

This release does not add independent-laboratory, independent-guide, protein or functional validation, nor does it revise earlier results in those settings. It reproduces the specified ranking experiments from compact sufficient data and preserved code. It does not rerun raw-UMI ingestion, refit control directions, or refit TRADE from expression matrices; TRADE replay begins from retained post-fit moments. Reconstruction scope and exactness checks are documented in the package README.

See [D4_Results_and_Analysis.md](D4_Results_and_Analysis.md) for the detailed D4 design, group-level outcomes, secondary depth results and disclosed archival procedure. Its procedural record distinguishes the original OSF registration and project/prediction archives from additional registrations that were not performed. This four-donor packaging is a post-result reproducibility release, not a new scientific lock.
