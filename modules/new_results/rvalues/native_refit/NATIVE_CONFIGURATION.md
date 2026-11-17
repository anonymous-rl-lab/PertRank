# Native fitting configuration and complete inputs

The fixed comparison used CRAN `rvalues` 0.7.1 with R 4.5.0. One Gaussian-conjugate configuration was fixed before the new comparison scores were inspected:

- `family=gaussian`, `prior="conjugate"`, `hypers="estimate"`, `alpha.grid=NULL`, `ngrid=NULL`, `smooth="none"`, `control=list()`.
- Each query, direction and shallow input is fitted across eligible candidates. Downregulation negates the signed effect while retaining its SE. The query's own target is excluded before fitting.
- The input SE includes target-cell and shared-control mean variance. The native method consumes marginal SEs, not a joint candidate covariance matrix.
- Ranking uses the negative unrounded native r-value. An exactly zero fitted prior variance gives an explicit common-score boundary tie; it is recorded separately from native success.
- No reference responses enter fitting or configuration selection. The comparison is a fixed post hoc supplementary workflow comparison, not a geometric-component contrast.

`code/run_rvalues_precise.R` and `original_driver.py` are byte-identical copies of the fitted-analysis entry points. The latter is provided here for inspection; its complete data layout and manifest are in the code bundle's `modules/rvalues_native`. Run the complete module's entry point for a native refit. This compact folder intentionally does not provide duplicated effect/SE arrays or the R package source archive.

For a separately prepared input TSV, the unchanged R wrapper can be called with:

```bash
Rscript --vanilla native_refit/code/run_rvalues_precise.R INPUT.tsv OUTPUT.tsv
```

Input columns are `task`, `input_id`, `query_index`, `query_id`, `direction`, `target_index`, `target_id`, `effect`, and `se`. The wrapper requires an installed `rvalues` version 0.7.1. Input candidate identities, own-target exclusion and floating-point precision must match the frozen design. The complete code bundle retains the original 7,742 fit inputs, R dependency source and fitting workflow.

This compact module's default `run.py` recomputes scoring and interval summaries from original native memberships. It does not perform any R fitting.
