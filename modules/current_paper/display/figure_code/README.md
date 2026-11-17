# Reproduce the current manuscript figures

Run from `modules/current_paper/display/` with Python 3, NumPy and Matplotlib (or use the root `current_paper` workflow to render figures in an output copy):

```bash
python figure_code/prepare_plot_inputs.py
python figure_code/draw_main_figures.py
```

The renderer writes `figures/fig1` through `figures/fig6` as 300-dpi PNG, vector PDF and SVG. It also retains the two supplementary illustrations (`figS6_guide_response` and `figS7_query_refinement`). All required values are packaged locally. Figure generation needs no network connection or original cell matrices.

The scripts extract and render archived estimates and intervals; they do not fit models, rescore candidates, resample cells or run new tests. The displayed two-state rvalues mean and D3 four-arm arrows are arithmetic summaries of saved estimates. These operations do not produce new uncertainty estimates.

## Inputs and traceability

| File | Contents |
| --- | --- |
| `figure_values.json` | Core geometry, native FR/ASH, depth, direction, energy and panel-transfer summaries |
| `design_ablation.json` | Complete six-construction design and D3 four-arm illustration |
| `new_figure_values.json` | External-RNA four-arm, CTLA4, guide and PertRank-Q summaries |
| `program_ranking_values.json` | Source Data extraction of native rvalues and program results |
| `PANEL_SOURCE_MAP.csv` | Main-panel mapping to packaged data files |
| `plotted_values.csv` | Exact plotted scores, contrasts and replacement quantities, with interval levels and semantics |
| `plotted_new_contrasts.csv` | Retained matched contrasts for Figures 1, 3 and Supplementary Figure 7 |

`prepare_plot_inputs.py` refreshes the program and ranking payload from `source_data/rvalues/primary_contrasts.csv` and the two program summary JSON files. It checks agreement with the existing payload to absolute tolerance 1e-12 and writes `PLOT_INPUT_VERIFICATION.json`. Added uncertainty metadata does not alter estimates. `FIGURE_MANIFEST.json` records all plot-input hashes and output names.

## Statistical labels

- Gene nomination plots retain paired conditional query-block 95% intervals.
- Program utility comparisons retain conditional reference-cell intervals. The two frozen primary contrasts use 97.5%; other displayed contrasts use 95%.
- Program P@10 reference-resampling stability ranges are not plotted in the main figures and are never converted to confidence intervals.
- Method bars are point estimates. Their pairwise conclusions come from the separately displayed paired contrasts.
- Replacement responses are weighted by replacement mass and task weights, preserving the whole-list utility identity.
- The three external-RNA input sources share one experiment. The program reference intervals condition on fixed inputs, lists and observed cultures. Neither set of intervals measures between-study replication.

For selective rendering, use, for example:

```bash
python figure_code/draw_main_figures.py --figures 2 3 5
```

Figure 6 combines ranking-input efficiency (a–c), the CTLA4 example (d), the nomination workflow (e) and query-panel reuse (f). The supplementary guide and Q results retain their separate evidence identities.

The figure assets match the current manuscript source. Pagination and legend placement belong to the manuscript LaTeX build, not this rendering workflow.
