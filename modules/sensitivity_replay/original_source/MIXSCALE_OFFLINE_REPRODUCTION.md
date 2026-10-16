# Rebuilding the official Mixscale comparison

Keep this directory as `strengthening_neighbor/` beside the existing `restored/working_v9_with_v6_inputs/code/repro_v10` tree. The evaluator uses those original locked scientific states; it does not infer a new candidate/query set. Restore the separate exact-cell checkpoint to sibling `d3_cell_restore/` (full counts, cell metadata and gene metadata). No source network read is required after restoring that checkpoint.

Install the isolated Linux R environment from `runtime/explicit_environment.txt` with micromamba, then use `R CMD INSTALL` on the exact upstream Mixscale commit recorded in `runtime/OFFICIAL_INSTALL_PROVENANCE.json`. Required versions and the installed implementation are in the archived sessionInfo files. The upstream commit does not supply a LICENSE file or DESCRIPTION License field. Do not invent a licence or treat a research snapshot as permission to redistribute it. The pinned URL and download hash are supplied for retrieval and verification.

All commands below are from this directory. Use the locked Python dependencies including h5py 3.11 (in the supplied runtime/dependency recipe). In the execution workspace, that Python extension is provided by `../late_recovery/python_deps`.

```bash
export PYTHONPATH=../late_recovery/python_deps
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python3 prepare_d3_gram_and_mixscale.py --counts ../d3_cell_restore/d3_full_counts.npz --metadata ../d3_cell_restore/d3_full_cells_metadata.npz --genes ../d3_cell_restore/gene_metadata.npz --out d3_prepared
runtime/env/bin/Rscript prepare_official_global.R d3_prepared mixscale_global full
runtime/env/bin/Rscript export_compact_driver.R compact_mixscale_helpers.R mixscale_global/global_preprocessed.rds d3_prepared mixscale_compact full
runtime/env/bin/Rscript validate_compact_mixscale.R compact_mixscale_helpers.R mixscale_global/global_preprocessed.rds mixscale_compact d3_prepared compact_equivalence
runtime/env/bin/Rscript execution_audit/validate_de_slot_release_frozen.R compact_mixscale_helpers.R mixscale_compact d3_prepared compact_equivalence de_slot_release_gate
python3 run_with_metrics.py full_fit_metrics.json runtime/env/bin/Rscript run_compact_mixscale_de_released.R compact_mixscale_helpers.R mixscale_compact d3_prepared compact_equivalence mixscale_full 1 1200 2
python3 evaluate_official_mixscale.py --input d3_prepared --fit mixscale_full --out mixscale_evaluated
```

The last fit is long (several hours with two workers). It checkpoints each target independently and verifies an existing target before reuse. Do not edit the running script. Interruption can be resumed using the same command. `RUN_COMPLETE.json` requires all 1,200 native fits; the evaluator rejects partial fits. Readiness of the comparison is not implied merely by these instructions being present.

The 1 GB global RDS and approximately 4 GB compact blocks are reproducible execution caches, not required delivery inputs. The raw-cell checkpoint plus original scientific states, pinned upstream source/environment and scripts rebuild them. The comparison uses native `beta_weight` to rank the original 1,200 targets for each of the original 300 response queries and both query directions; it does not replace the published method's DE purpose with an assertion that it estimates the same quantity as a raw log-normalized mean difference. Native filtered query entries have the predeclared zero fill; errors and nonfinite returned coefficients are never zero-filled.

The four-target gates are execution equivalence checks, not scientific evidence. They preserve original global PCA, nearest NTCs, perturbation signature values, RNA counts/data, feature axes, metadata, control order, native scores and every leave-one-gene-out fit. The final native DE call may release only two unused slots after the exact gate: PRTB and RNA scale.data. No gene/cell subsampling or cross-gene dispersion simplification is used.

For provenance, current active script hashes are in the final runtime freeze and delivery manifest. Earlier frozen paths may name superseded execution versions; `EXECUTION_SOURCE_HASH_ALIASES.json` resolves those exact historical bytes. The execution history documents memory-guard stops and a footer parsing failure caused by editing a live script; the 20 target fits completed before that failure are unchanged and valid checkpoints. Neither event is reported as a scientific method failure.
