# Post hoc robustness analyses for v20

## Control replacement: asset feasibility

The fixed replacement protocol keeps the original 13,394-gene axis, normalization scale, five input groups, query panels, candidates, reference and scoring rules. It specifies two 2,000-cell replacement sets: the independent mechanism-control pool, and a balanced replacement containing 1,000 original input controls and 500 cells from each mechanism-control half. The direction-only analysis changes U and recomputes both target and original-control projection variances. The full fixed-axis analysis also replaces control means and variances. This design is a post hoc sensitivity analysis.

The original execution retained all gene-wise target means and sample variances, but evicted raw target expression checkpoints and their redundant payload packs after verified statistics had been saved. Of 56,100 ranking-input cells, only 100 remain in local raw-row NPZ files. Existing range caches fully cover 644,590,068 of the 2,700,361,416 selected input payload bytes; 2,055,771,348 bytes are absent or only partially covered. All three 2,000-cell control resources are locally available. Arbitrary new-direction target variance cannot be reconstructed from the diagonal gene variances and the five original projected variances. Consequently, complete-cohort control-replacement scoring requires reacquiring the original locked input rows. The present revision records feasibility and does not present a partial-cohort or diagonal-covariance approximation as a replacement-control result.

The inventory is executable (`inventory_control_assets.py`). The prewritten full analysis is `run_control_replacement.py`; it stops if raw rows are missing. `CONTROL_REPLACEMENT_ASSET_REPORT.json` reports the inventory and required missing statistic.

## TRADE numerical sensitivity: method

The original Stim H2b comparison retains its registered not-evaluable status. A separate post hoc sensitivity uses the same TRADE scalar adapter, ashr 2.2-63, R 4.4.3, half-uniform component family, uniform prior, adaptive component grid, complete 13,394-gene vectors, all 561 targets, five input groups and seed 2026. The only retained numerical modification is mixSQP `control=list(eps=1e-4)`, applied uniformly to every fit. The installed ashr adapter used eps=1e-6 in the original execution.

In mixSQP, eps adds a positive floor to row-normalized mixture likelihood values used in numerical objective/derivative calculations. Thus this change regularizes the numerical optimization objective; it is not a change to the mixture prior and is not claimed to reproduce the exact original optimizer. The family, grid, gene set and scalar moment formula remain fixed. The exported ashr log likelihood is evaluated for the fitted mixture without the mixSQP numerical floor.

Before any sensitivity ranking was evaluated, numerical smoke tests were run on the original failure target and two previously successful targets. All five groups of the original failure target converged with eps=1e-4. Across the ten previously successful fits, the largest absolute scalar-weight change was 0.00197844 and the largest absolute unregularized total-log-likelihood difference was 0.118292 across 13,394 genes. Mixture second moments and scalar weights were separately recalculated from the saved component endpoints and probabilities. The uniform rule was frozen in `TRADE_FULL_COHORT_PROTOCOL.json` before complete-cohort fitting and ranking.

Earlier numerical candidates were diagnostic smoke tests: disabling the likelihood SVD approximation reproduced the original failure, and accelerated-EM alternatives were not selected because of numerical warnings/computational cost. Their outputs were not scored. A full sensitivity comparison requires convergence and valid moments for every one of the 2,805 fits; failed fits receive neither imputed weights nor exclusions.

## TRADE numerical sensitivity: results

All 2,805 fits converged, with no fit warnings or errors, in 1,896.92 seconds of wall time using six workers. Per-fit mixture moments and scalar weights were separately recomputed (maximum weight discrepancy 5.56×10⁻¹⁶). All original four-arm scores were reproduced exactly. A compact replay recomputed all five-arm rankings and twelve panel contrasts with zero discrepancy. A separate sort-based top-25 implementation reproduced all 5,940 TRADE scores with maximum discrepancy 9.02×10⁻¹⁷; independent block-sum resampling reproduced both PertRank–TRADE intervals within 6.94×10⁻¹⁸.

| Panel | Raw | Euclidean | TRADE scalar, numerical sensitivity | Pearson metric | PertRank |
|---|---:|---:|---:|---:|---:|
| Responsive | 0.316909 | 0.378579 | 0.369019 | 0.387090 | 0.393727 |
| Expression terciles | 0.089950 | 0.124727 | 0.121206 | 0.127308 | 0.129489 |

PertRank exceeded the numerical TRADE scalar sensitivity on the responsive panel by 0.024708 P@25 (paired query-block 95% interval 0.022485–0.027249), equivalent to 0.617700 additional deep-reference members per 25 nominations. On the tercile panel the difference was 0.008283 (0.004639–0.011109). TRADE scalar sensitivity exceeded raw ranking on both panels: 0.052110 (0.046349–0.058134) and 0.031256 (0.021513–0.038911). Its differences from Euclidean ranking were −0.009560 (−0.011683 to −0.006545) and −0.003522 (−0.005238 to −0.001138), respectively. Differences from the Pearson-metric arm were −0.018071 (−0.020434 to −0.015581) and −0.006102 (−0.008142 to −0.003493).

These results complete a uniformly processed, whole-cohort post hoc numerical sensitivity using the fixed statistical family and a modified optimization floor. They do not change the original registered H2b adjudication. The intervals retain the original fixed-query-block conditional interpretation.

