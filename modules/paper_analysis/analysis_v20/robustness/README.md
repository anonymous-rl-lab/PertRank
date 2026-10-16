# v20 robustness analyses

This folder separates the registered Stim result from a post hoc numerical sensitivity and a control-replacement feasibility assessment.

- `TRADE_FULL_COHORT_PROTOCOL.json`: retained uniform numerical rule, frozen before sensitivity ranking; it supersedes the initial numerical candidate in `PROTOCOL.json`.
- `TRADE_SMOKE_VALIDATION.json`: comparison against ten previously successful fits and the five groups of the original failure target.
- `R_TRADE_univariate_eps.R`: original fit function with mixSQP eps=1e-4 and an explicit convergence assertion.
- `run_trade_full.py`: six-worker checkpointed full-cohort refit; all 2,805 fits are required.
- `collect_trade_sensitivity.py`: separately validates saved mixture moments and scores the complete cohort.
- `replay_trade_sensitivity.py`: replays every ranking and all reported intervals from the compact numeric inputs, when the complete-cohort analysis succeeds.
- `CONTROL_REPLACEMENT_ASSET_REPORT.json` and `inventory_control_assets.py`: exact raw-cache and payload inventory.
- `run_control_replacement.py`: fixed-axis direction-only and full control replacement; it checks complete input-row availability before analysis.
- `ROBUSTNESS_METHODS_RESULTS.md`: Methods, findings and analysis status.

Full TRADE refitting requires the archived original `stim_run_20261011` input statistics and R runtime, specified by `PERT_RANK_STIM_ASSET_ROOT`. Batch work can be placed outside the paper with `PERT_RANK_TRADE_WORK`. The paper package contains compact scores, moments, mixture parameters and ranking inputs after successful complete-cohort fitting. It omits duplicate 13,394-gene TSV/NPZ fitting intermediates.

The original registered H2b comparison remains not evaluable. Numerical regularization changes the optimizer's likelihood floor, so the new analysis is labelled a uniformly regularized numerical sensitivity. Every input gene, candidate and replicate is retained. Earlier smoke candidates were selected or rejected using numerical feasibility; no ranking outcomes were used in that choice.

Re-estimating a control geometry needs target covariances in the new directions. Retained per-gene diagonal variances cannot identify those projections. The control-replacement asset report therefore states the missing input requirement rather than reporting a diagonal-covariance substitute.

## Completed results and replay

All 2,805 fits converged without warnings. The responsive-panel PertRank–TRADE sensitivity contrast is 0.024708 P@25 (95% query-block interval 0.022485–0.027249); the tercile contrast is 0.008283 (0.004639–0.011109). Exact values are in `TRADE_SENSITIVITY_RESULT.json`.

From the total package root, choose a new output directory:

```bash
python modules/paper_analysis/reproduce.py --only trade --out /new/path/trade-replay
```

The launcher first restores `TRADE_SENSITIVITY_REPLAY_INPUTS.npz` losslessly in the output directory, then runs both the original replay and independent score review with NumPy and SciPy. The compressed source directory does not contain the restored NPZ for direct execution. The first reproduces all five-arm rankings and twelve contrast intervals. The second uses a separately written sort-based top-25 membership rule and a block-sum bootstrap; it does not call the production score function. The final review is `TRADE_SEPARATE_SCORE_REVIEW.json`.

`trade_eps_runner_execution_v20.py` preserves the exact source bytes used in fitting. `trade_eps_runner.py` is an archive-path adapter; it sets the original runtime paths from `PERT_RANK_STIM_ASSET_ROOT` while keeping the executed numerical implementation unchanged. Identity equality against completed batches is recorded in `RUNNER_PORTABILITY_VERIFICATION.json`.
