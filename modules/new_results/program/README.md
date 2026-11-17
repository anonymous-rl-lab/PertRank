# Program nomination: compact reproducibility module

This module covers the fixed D3 program experiment, sci-Plex3 frozen program nomination, and the subsequent exploratory ASH/geometry backward checks. The six predefined programs, all original comparison arms, negative results, and unsuccessful directional adjustments are retained.

## Run

Requires Python 3, NumPy and pandas. Invoke the script by its path from any working directory, and choose a **new** output directory:

```bash
python run.py core --out /absolute/path/new_program_core
python run.py exploratory --out /absolute/path/new_program_exploratory
python run.py all --out /absolute/path/new_program_all
```

`core` runs D3 and the frozen sci-Plex3 evaluation. `exploratory` runs every original and repair arm in D3 and sci-Plex3. `all` runs both. The launcher reconstructs the archived computational materials into `OUT/materials`, runs the scientific routines in fresh processes, verifies the results, and writes `REPORT.json` and `RESULTS.json`. Original detailed reports and logs remain under their experiment directories.

`RESULTS.json` uses `reference_resampling_range` for P@10 reference-resampling quantiles. Utility intervals retain their conditional-reference interpretation. The exporter checks that every original numeric value is unchanged. Original per-experiment summary files remain untouched for exact comparison; their historical P@10 field name is interpreted through this export layer.

## What is actually recomputed

- D3 input effects and standard errors from exact input/control moments; original and program-coordinate-excluded weight formulas from full-profile sufficient statistics; all 54 scoring arms, memberships, 4,000 reference bootstrap draws, 1,000 weight permutations per mode, readout interactions and replacement accounting.
- sci-Plex3 full-profile weights and all six score constructions; exact tie-aware lists, reference response utility and P@10; 4,000 paired reference resamples and all predeclared aggregate, stratified and removal analyses.
- Exploratory signal-protecting, signed adaptive and fixed-enhancement adjustments; all original and adjusted arms, lists, replacement results and 4,000 paired reference resamples in each source.

The application criterion passes; the ASH-readout geometry criterion and original joint extension criterion do not. No outcome is changed or removed by compression.

## Fidelity and compression

No floating-point precision is reduced. Byte shuffling and XZ compression are lossless, duplicate arrays are stored once, and deterministic array/file recipes are checked against their complete original byte fingerprints. Expected output hashes replace bulky bootstrap caches: the draws are newly computed rather than read from those caches.

Two input representation changes are explicit: D3 shallow target/control program measurements and sci-Plex3 shallow target program measurements are replaced by the exact original-order float64 moments used by the scorer; sci-Plex3 fixed control summaries retain the original per-plate moments and program scales. Original candidate and cell identities remain. Every reference-cell program projection needed for resampling is retained in full. These sufficient statistics reproduce the retained computations; **the original shallow cell-level measurements cannot be recovered from their moments**. The full experiment archives preserve those measurements and source recovery materials.

The original program definitions, candidate information, original frozen records, native ASH R scripts/configuration and regenerated native ASH effect/SE input TSV are available in `OUT/materials`. The compact manifests explicitly record the representation change; recreating a historical file or record is not a new scientific freeze.

## Reproducibility boundary

Native ASH posterior means are fixed original outputs. This module does **not** claim an independent ASH refit or raw UMI recovery. Full raw-count preparation resources, the pinned native ASH package sources and original full archives remain in the manuscript reproducibility supplement identified in `SOURCE_ARCHIVES.json`. Native refitting requires that fuller environment; the fit inputs and scripts retained here document the interface and configuration.

`VALIDATION.json` records the completed isolated all-mode check. The original arrays and all scientific summary values matched: D3 37 arrays and 2,399 numbers; sci-Plex3 40 numeric arrays and 11,619 numbers; exploratory replay 223 arrays and 17,208 numbers. This is exact replay from retained statistics/reference projections and fixed native outputs, not a rerun from original counts.
