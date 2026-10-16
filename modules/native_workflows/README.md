# Native estimator generation paths (opt-in source archive)

This module preserves the code, configurations, source identities and execution records that generated the native FR-Perturb/ASH comparisons and GPerturb resource pilot. It is **not part of the default numerical replay**, and no training was run while assembling it. Frozen prediction/score reproduction is in `../fixed_tasks`; A19-R intermediate-data replay is in `../corrected_intermediates/a19r`.

| Analysis | Generation path | Preserved choices and requirements |
|---|---|---|
| FR-Perturb D3 development / D4 transfer | `benchmark/vendor_work/fr_native_runner.py`, development-grid and locked-transfer scripts; `benchmark/resume_20261014/FR_D3_WINNER_LOCK.json` | Official commit `24e653b6a5c050ab17f3600ebaf222bef32e2d25`; primary rank 100, SPCA alpha 0.1, lasso alpha 0.0001, 50 dictionary iterations, seed 20261012. Rank extension and depth scripts remain separately identified. |
| Native ASH | `benchmark/ash_work/prepare_full_stats.py` → `run_posterior.py` / `run_sharded.py` → `posterior_runner.R` | R 4.4.3, `ashr` 2.2-63, `jsonlite`; all declared genes per target, native default ASH options for selected arm. Solver/row convergence and explicitly enabled retries are recorded, not silently treated as successful fits. |
| Independent RNA native FR/ASH | `a19r/code/prepare_fr.py`, `prepare_ash.py`, `run_native.py`, `guarded_native.py`, `native_diagnostics.py` | Original 52/59/63 candidate pools, input locks, selected run records and all six final native configurations. `code/score_native.py` links generated estimates to the seven-arm fixed task. |
| TRADE scalar adaptation | `trade_scalar/trade_scalar_compact.R` and saved five-group/D4 launchers | A common-coordinate scalar derived from the fitted prior's second moment; **not native end-to-end DESeq2 TRADE**. Upstream `fit_ash()` source is an external requirement, omitted here. |
| TRADE posterior prototype | `benchmark/ash_work/run_posterior.py --mode trade` and R wrapper | Historical generation path, not an additional completed native benchmark. Requires the exact external source identified below; no claim that every archived configuration yielded a paper result. |
| GPerturb | `benchmark/resume_20261014/gperturb_native_pilot.py` and resource protocol/result | Commit `05d5a431413ff9dc443c82b994b571ecc563426d`; real D3 resource pilot only. Archived result is resource-blocked (6.632 GiB peak against a 6.5 GiB guard), with no performance score. |

## Inputs and environments

FR input is raw CSR counts (`data`, `indices`, `indptr`, `shape`), `target_index` (-1 controls), ordered target/gene IDs and optional cell IDs. It contains shallow targets and designated controls only. The adapter applies the official filtering, TP10K/log1p, control centering, factorization/recovery and effect scaling. See `benchmark/vendor_work/requirements.txt`, upstream `environment.yml` and each fit report. Native solver/runtime differences can affect a newly fitted result; stored-score replay is the exact supported verification path.

ASH input is a float64 NPZ containing `effects`, strictly positive `ses`, `row_ids` and `genes` on the full fixed gene axis. Preparation of that axis and standard errors belongs to the recorded input builder. The R installation and third-party runtime binaries are not bundled. The archived README refers to an older runtime rebuilding directory that is not present here; install the stated R packages and supply a valid Rscript launcher before opting into fitting.

GPerturb needs its separate PyTorch environment, full frozen raw count/axis manifest and aligned technical/lane covariates. See `GPerturb_CPU_ENVIRONMENT.json` and `GPerturb_REAL_D3_RESOURCE_GATE_PROTOCOL.json`. This module does not supply the large raw matrix or automatically start the resource pilot.

No large counts, fitted effect matrices, posterior arrays or runtime binaries are copied into this module. Native fitting requires the separate original A19-R count archive, full-gene controls and input-builder products; those raw inputs are not included in this compact package. Historical absolute paths are retained as provenance. A fresh rerun requires rebinding local paths in a new configuration while preserving the recorded input identities; these scripts are not advertised as an automatically relocated, one-command training environment.

## Third-party source and licensing

- **FR-Perturb:** upstream source is retained under its bundled GNU GPL v3 license, `benchmark/vendor_work/FR-Perturb/LICENSE`. Repository: <https://github.com/douglasyao/FR-Perturb>. Source hashes are in `official_manifest.json`.
- **GPerturb:** upstream source is retained with its MIT license and copyright notice, `benchmark/resume_20261014/gperturb_source/LICENSE`. Repository: <https://github.com/hwxing3259/GPerturb>. The pinned source manifest retains exact URLs/hashes.
- **ASH:** the wrappers call the separately installed `ashr` R package. No extracted upstream ASH implementation is redistributed here. Obtain it from the official package distribution and follow that package's license; exact executed package version and options are retained.
- **TRADE:** no upstream TRADE implementation is redistributed. The archived source commit is `3432d108c43325e9df65276563d65144a26b69ce`; the excluded `R_TRADE_univariate.R` digest is recorded in `SOURCE_MANIFEST_SHA256.json`. Supply the matching source from the original authors under applicable permissions before invoking the project adapter. This requirement does not affect the default replay from saved moments.

Project adapters and execution records remain distinct from third-party code; this archive does not assign a new license to upstream files.

`SOURCE_MANIFEST_SHA256.json` maps every archived file to its original archive path, SHA-256 and size. `ASSEMBLY_CHECK.json` records source-hash verification and syntax parsing only, **not fitting or model validation**.
