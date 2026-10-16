# D3 cell-resampling with reliability weights re-estimated

This is a development sensitivity analysis on the opened D3 data. It does not revise the frozen paper estimands or provide a new independent confirmation. It holds input NTCs, control directions, query panel, candidate pool and deep reference fixed. On every draw the original 20 cells per target are sampled with replacement; the same cell multiplicities generate query expression and both newly fitted weights. B=200, seed=202610053, k=5, beta=0.9.

The full source matrix was recovered for the exact original 24,000 target cells plus 2,000 NTCs. The raw query counts, cell IDs, library totals and full feature IDs pass exact checks. Independent audits against archived full-gene sums/squares pass. Two fixed real targets × five draws match explicit copied-cell moments. Uniform multiplicities reproduce the original per-query scores to 2.8e-17; weight differences are at most 1.04e-7 due to archived floating-point centering. A separate reviewer reconstructed the complete first bootstrap draw directly from raw cells: all 1,200 targets, direct mean/variance/projections, and an independent full-sort tie scorer agree (max weight difference 3.6e-15; scores identical). This result is not a Gram implementation error.

## Results

| Panel | Comparison | Original difference | Refit resampling mean | Empirical 2.5–97.5 percentiles | Positive draws |
|---|---|---:|---:|---:|---:|
| Responsive | W1 − w0y | +0.03561 | −0.00344 | [−0.00759, −0.00032] | 2.5% |
| Responsive | W1 − raw | +0.03406 | −0.00383 | [−0.00999, +0.00287] | 12.0% |
| Terciles | W1 − w0y | +0.00554 | −0.00208 | [−0.00369, −0.00053] | 0.0% |
| Terciles | W1 − raw | +0.02286 | +0.00163 | [−0.00065, +0.00389] | 92.0% |

The earlier fixed-weight D3 resampling remained positive: responsive W1 − w0y mean +0.02963 and terciles +0.00388. Both versions must be retained. Keeping weights fixed masks their sensitivity to this particular empirical resampling operation; the fixed-weight result cannot be presented as robustness of the fully re-estimated method.

## What changed in the weights

| Weight | Original mean | Resampling mean | Original zero fraction | Resampling zero fraction | Mean Spearman correlation with original |
|---|---:|---:|---:|---:|---:|
| w0y | 0.1374 | 0.5350 | 8.83% | 0.00% | 0.379 |
| W1 | 0.1127 | 0.5280 | 0.58% | 0.00% | 0.245 |

This is a substantial distributional change, not a small perturbation around the fitted weights. All 200 draw-level quantiles, zero fractions and correlations are saved in `refit_D3_weight_diagnostics.npz` and summarized in its JSON counterpart.

For centered observed cells with squared mean energy a and unbiased total within-cell variance v, ordinary n=20 empirical resampling gives E[a* | X] = a + 19v/400, while E[v* | X] = 19v/20. The empirical mean already contains the original realization's noise; resampling adds further mean variation around that mean. This explains why newly fitted reliability weights can move upward strongly. It is a property of this conditional experiment and does not establish that a fresh set of 20 independent cells would have the reported distribution.

## Allowed interpretation

The existing result is not stable under this ordinary empirical cell-resampling procedure when the weights are refitted. The two conditional procedures tell different stories, and that difference is scientifically relevant. It must be disclosed rather than selecting only the positive fixed-weight analysis.

The empirical percentile ranges are not population confidence intervals, p-values, or proof that W1 fails on new 20-cell samples. No NTC/PCA, query selection, deep truth, donor or guide uncertainty is propagated. No correction, alternative bootstrap, k/beta tuning or source selection was introduced after the negative result. The original frozen method results remain separate.

## Files

- `REFIT_BOOTSTRAP_PROTOCOL.md`, `REFIT_BOOTSTRAP_FREEZE.json`: rules written before the resampling results.
- `prepare_d3_gram_and_mixscale.py`, `d3_prepared/PREPARATION_CHECKS.json`: source identity and exact sufficient-statistic preparation.
- `refit_cell_bootstrap.py`, `refit_D3_results.json`, `refit_D3_draws.npz`: B=200 execution and complete output.
- `weight_refit_diagnostics.py`, `refit_D3_weight_diagnostics.json/.npz`: post-result input-side diagnostics, not additional model selection.
- `../ablations/audit/independent_refit_real_draw_check.json`: independently calculated raw-cell and score checks.
- `audit_refit_cells.py`, `REFIT_AUDIT_FREEZE.json`: an unused audit draft, superseded operationally by the independent audit; it produced no results and is not part of the analysis chain.
