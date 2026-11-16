# Reproduction scope

The default chain uses eight modules. Each starts from retained inputs and produces fresh results in a new output directory. It does not download raw atlases or retrain upstream estimators.

| Module | Starting point | Recomputed |
|---|---|---|
| fixed_tasks | Effect/moment inputs, memberships and retained estimator outputs | Weights, rankings, depth curves, paired intervals, fixed D4 Rest/Stim decisions and native-estimator comparisons |
| early_intermediates | Per-query scores, blocks, target statistics and direction/case inputs | Four-source factorial contrasts, 23 retained analyses and diagnostic statistics |
| corrected_intermediates | Guide/query scores, lane counts, prediction/reference memberships and depth scores | Corrected guide, A18-R, A19-R and cell-equivalence contrasts and sensitivity summaries |
| r20_intermediates | Per-query scores and original blocks | Internal and external two/five-input summaries and paired intervals |
| paper_analysis | Losslessly reconstructed effects, weights and reference information | Alignment decomposition, biological strata, query-panel transfer, TRADE sensitivity and CTLA4 lists |
| il2_intermediates | Guide readouts, fixed candidate/list identities and fitted MAGeCK projections | Protein-reference centering, target medians, 20,000 paired draws and fixed-list support |
| sensitivity_replay | Saved resampling draws and fixed fitted coefficients | Empirical draw summaries and Mixscale list comparisons |
| current_paper | Projected query effects, original reference memberships/draws, retained guide statistics and fixed predictions | Independent-RNA four arms, guide diagnostics and current manuscript figures; source tables supplied for inspection |
| pertrank_q (optional) | Retained model inputs and selected checkpoints | Predictions, target-held-out candidate-subpool rankings, selection accounting and conditional intervals |
| software (optional) | Installed package and bundled examples | Interface regression tests, notebook execution and cell-input correctness checks |

Raw count preprocessing, control PCA and upstream model fitting are not repeated where the retained starting point lies downstream. Source-only native adapters remain under `modules/native_workflows/` with their dependencies and attribution; they are not part of the offline replay.

## Explicit unavailable stages

The original A14 per-cell protein input is unavailable; its reported result, protocol and recovery description remain outside numerical replay. Original A19 cell selection is not recovered; A19-R is the separately identified reconstruction. CELLEQ-R likewise retains its reconstruction identity and is distinct from the fixed seven-depth calibration. An unavailable control-replacement ranking is not counted as a successful analysis.

Current-paper tables and figure code are under `modules/current_paper/display/`. Display regeneration alone is not upstream model reproduction; not every source-table cell is freshly computed by the dispatcher. Analytic identities require no fitted model. Each numeric module's receipt states the fields and tolerances it verified.

The three external-RNA inputs share one reference experiment. Q uses reference supervision on calibration targets and evaluates specified held-out subpools. Its intervals are conditional on the selected model. Neither checkpoint replay nor figure rendering adds independent biological validation.
