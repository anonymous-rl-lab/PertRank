# Program source data

`external_summary.json` contains all six frozen arms, all contrasts, complete
cell-line/program/culture-direction stratification, leave-group-out summaries,
replacement accounting and the two original primary decisions. Values are in
input-control program standard deviations for utility, and fractions for P@10.

`external_method_scores.csv`, `overall_contrasts.csv` and
`external_replacement_summary.csv` are plotting-friendly views. The paired
comparison, not overlap of individual-arm intervals, is the comparison unit.
`method_list_scores.csv` includes all 432 method lists; `replacement_lists.csv`
contains all archived paired task-level replacement accounting.

`d3_*` files retain development diagnostics and their complete stratification.
`exploratory_*` retain all completed adjustment branches, including failures.
Original full lists, membership arrays and individual reference projections
are inside the corresponding reproducibility archives.

## Uncertainty semantics

- Utility: `conditional_interval` describes reference-cell measurement
  uncertainty conditional on fixed inputs, predictions and observed cultures.
- P@10: `reference_resampling_range` is a quantile stability range after
  reference Top10 re-ranking. It is not a calibrated confidence interval.
- The two sci-Plex3 primary utility contrasts use 97.5% intervals. Other frozen
  subgroup/secondary contrasts and exploratory intervals use the stated 95%
  level. Exploratory comparisons have no familywise multiplicity correction.

`EXPORT_PROVENANCE.json` maps every normalized JSON to the unchanged original
source and its SHA256. `EXPORT_VERIFICATION.json` checks exact numerical
preservation and replacement identities. Archived files are never overwritten.
