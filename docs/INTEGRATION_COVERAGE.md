# Evidence coverage

All modules below are included in this reproducibility package. Numerical precision, statistical definitions and the scientific identity of the reported comparisons are retained. Core and exploratory analyses have separate entry points.

| Evidence | Location | Reproduction level |
|---|---|---|
| Independent RNA four-arm comparison | `modules/current_paper/four_arm/` plus checked shared assets | Original effects, scalar weights, projections, rankings, 2,000 block draws and 500 reference draws; source contrasts and accounting are checked against retained results. |
| Retained guide mechanism | `modules/current_paper/guide/` plus shared assets | All 174 entries, 123 eligible entries × 20 partitions × 3 metrics, locked predictions, independent formula checks and fixed-list accounting. |
| PertRank-Q | `modules/pertrank_q/` | Model inputs, selected checkpoints, candidate-subpool statistics, primary/secondary arms and unsuccessful alternatives. Optional inference requires Torch; training is not repeated. |
| Native rvalues fitting | `modules/rvalues_native/` | Exact effects, standard errors, fitting scripts and third-party source/license. The preparation entry restores and verifies the fitting material; R fitting is separately documented. |
| SN4 diagnostics | `modules/sn4_diagnostics/` | Target tables, definitions and source evidence support target-table reaggregation and the original bootstrap. |
| rvalues scoring and program nomination | `modules/new_results/` | Native rvalues membership scoring, D3 program diagnostics and sci-Plex3 frozen evaluation. Core and exploratory computation are separate; the unsuccessful geometry adjustments remain available. |
| Source tables and figures | `modules/current_paper/display/` | Current numerical tables and six main/seven supplementary figure scripts. Rendering is distinct from fitting and numerical rescoring. |

Each module states its retained starting point. Unavailable upstream stages are listed in [reproduction scope](REPRODUCTION_SCOPE.md) and are not counted as completed numerical reproduction. Scientific protocols and unfavorable results remain available with the inputs needed for the declared computations.
