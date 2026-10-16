# Early experiments: compact statistical reproduction

Run from the repository root:

```bash
python modules/early_intermediates/reproduce.py --output-dir ../early-intermediates-results
```

Requires Python, NumPy and SciPy. Output directory must be new or empty. No network, model fitting, raw counts, or original experiment reruns are used. Exit 0 means the **supplied statistical replay** matched; exit 1 means integrity or numerical mismatch. `report.json` specifies the actual scope and per-analysis results.

## Starting point and preserved evidence

This compact release starts from saved query-by-direction scores, query block assignments, target energies and weights, control-direction cross-products, and case-gene effect rows. It replaces v1.5.0's full saved-expression states. The export ran the original 23 analyses against their saved inputs, verified their archived outputs, and retained their statistical inputs. `provenance/EXPORT_COMPARISON.json` records the source scripts and archived-result hashes. Later corrected guide and independent RNA analyses are in their separate modules; historical definitions here do not replace those corrections.

* The 14 canonical settings retain **all six arms** (`raw`, `w0y`, `W1`, `AY`, `w0y_AY`, `wM_AY`), per-query scores, blocks and target weights/energy attenuation. The four obsolete D1 “as executed” settings are excluded from this canonical factorial result. Means and all six original paired contrasts are recalculated with the archived 2,000-draw query-block bootstrap, seed 2026. The 14 geometric contrasts are also checked against the current paper's `v25_design_geometry.csv`.
* Twenty original analyses provide saved score means and paired bootstrap inputs (1,650 captured operations in total). Their other descriptive fields remain in `archived_outputs`; the compact entry does not claim to rederive every field merely because an analysis has one captured operation.
* Stage-1 reliability is recalculated from target-level energy vectors, including correlations and energy/noise quintiles. Direction similarity is recalculated from first-direction loadings, control means and restricted direction cross-products. The two D1 case studies are recalculated from their gene-level effects and filtered relation table. These three are numerical descriptive replays, not display-only copies.
* `factorial_14_settings.json`, individual statistic outputs and `report.json` are generated afresh. Numeric tolerance is 1e-12. Original upstream Stage-1/2 export comparisons used their archived 1e-9 tolerance. Bootstrap arithmetic and RNG are unchanged; identical draw-index arrays are memoized for speed.

## What this compact tier does not reproduce

It does not reprocess raw cells, estimate effect spectra or weights from full expression states, or reconstruct candidate rankings from gene-by-target matrices. Those were the starting computations in v1.5.0 and are deliberately removed for the size limit. Weights and per-query scores here are statistical inputs, not outputs newly fitted by this entry. The selected original scripts under `provenance/scripts` document upstream computations but require the full archived v1.5.0 input tree to run.

`provenance/export_from_full_states.py` can export the query intermediates from that separate full archive using `PERTRANK_FULL_SOURCE` and `PERTRANK_EXPORT_WORK`. The compact repository does not include that archive. Supplemental target/control/case intermediate exports are documented in `provenance/export_early_extras.py` and `provenance/compact_control_grams.py`; their source paths refer to the original export workspace.

Legacy A14 is **STORED_RESULT_HASH_ONLY / MISSING_PER_CELL_INPUT**, outside the numerical replay. `saved_only/A14` retains its original result, protocol and code plus a recovery guide for the separate full-archive layout. Its missing 3.08 GB per-cell input has not been downloaded. No hash-only result is reported as a successful numerical replay. A9/A10 target-explanation and direction-replication full-state validation receipts are preserved in the v1.5.0 provenance summary; this compact entry does not claim those two additional reconstructed-parent runs as covered analyses.
