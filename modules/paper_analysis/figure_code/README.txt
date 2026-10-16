PertRank v25.2 main figures, retained from v25.1

Run: python figure_code/draw_main_figures.py
Select figures: python figure_code/draw_main_figures.py --only 5 6

Outputs retain fig1_overview_v25 through fig6_panel_transfer_v25 filenames so the manuscript and original result provenance stay aligned. v25.2 retains the v25.1 figure files byte-for-byte. Scientific points, intervals and display labels are unchanged.

Figure 1: placement of control geometry; D3 2x2 development example; D4 native FR rank100 comparison; independent RNA native FR rank300 comparison.
Figure 2: source-specific independent RNA scores and conditional paired contrasts.
Figure 3: fixed-weight readout contrasts; original-readout geometry gains; FR rank300 estimator reuse.
Figure 4: energy, directions, query exclusion and alignment.
Figure 5: fixed settings, measured input equivalents and depth dependence.
Figure 6: transfer of a fixed query panel between states.

The script reads source_data/figures/figure_values.json and the named archival sources. It renders PNG (300 dpi), vector PDF and editable SVG, and exports display maps. Existing figure-data filenames identify their archived sources. No model fit, expression access, statistical resampling or new hypothesis test is performed.

Complete fixed-task statistical replay and raw-count fitting have separate requirements in the included code bundle. Plotting dependencies: Python, NumPy and Matplotlib.
