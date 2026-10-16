# D3 Rest NTC guide supplement

Run the packaged inference replay (Python 3.12 and NumPy 2.3.5 in the recorded environment):

```bash
python replay.py --output /tmp/ntc_replay.json
```

The integration entry point is `replay_to(output_dir: Path) -> dict`; it writes `output_dir/replay.json` and requires an external output directory. `replay(module_dir=None)` returns the report without writing files. Paths are relative to this module, so it can be moved into the D1–D4 repository. Routine replay has no network access or model refits. It regenerates all 2,000 bootstrap draws, all 2,000 permutation draws, all 64 targeting-pair values and all 2,000 targeting bootstrap draws. Every result is compared with the original archived arrays; it does not merely print saved summary values. Selected-cell disjointness and lane compositions are checked too.

## What the compact data retain

`inference_inputs.npz` contains the complete **float64 468 × 468 cross-Gram matrix** of independently centered half-A and half-B guide means, divided by the fixed 13,390-gene count; 64 float64 same-target guide difference energies; and guide/target labels. The matrix is not symmetrized. These are sufficient for exactly the frozen aggregate diagnostic and its resampling, up to floating-point arithmetic order. There is no float32 conversion or decimal rounding. `build_compact.py` records the full source half-means hash and derives these inputs without changing the experiment.

The package also retains exact cell selections, protocol, result and resampling arrays, original pipeline source files, original freeze hashes, source identity, source-audit reports, and the lane metadata. `COMPACT_PROVENANCE.json` lists the omitted large assets and their hashes. Raw counts and gene-level half means are **not** reconstructible from the compact matrix.

## Reproduction levels

| Level | In this package | Additional requirements |
|---|---|---|
| Recompute the published diagnostic, all uncertainty draws and p-values | Yes, using `replay.py` | NumPy |
| Recompute lane composition and selected-cell disjointness | Yes | NumPy |
| Recreate compact inputs from gene-level means | Builder and input hash included | External `half_means.npz` (83.8 MB compressed) |
| Repeat original metadata selection, raw-row acquisition and normalization | Original source code, fixed selection, source identity and hash inventory included | External D3 Rest H5AD; omitted original metadata/raw-control assets listed below |

## Interpretation and identity

This was a **new metadata-frozen recovery selection on 2026-10-08**, not a replay of the unrecovered interrupted NTC selection. It contains 468 NTC guides × two disjoint 20-cell halves (18,720 cells), plus 64 targets × two guides × two 20-cell halves (5,120 cells): 23,840 unique cells total. The normalization scale is 8261.5, and the fixed gene axis has 13,390 genes. No shared control mean is subtracted. Negative point estimates or resampling draws are retained.

The original result was positive: NTC pair energy 0.0031361378 (conditional 95% guide-bootstrap interval 0.0028984009–0.0033684483). Same-target guide energy was 0.0024549014 (target-bootstrap interval 0.0021032450–0.0028490491). The interval conditions on the observed guides, fixed axis and normalization; it is not a donor-population interval. The permutation assumes exchangeable guide spectra and does not adjust lane composition.

The result establishes reproducible **guide-associated expression structure**, not its causal source or the fraction of the main ranking result attributable to artifact. NTC halves are perfectly lane-balanced for 426/468 guides, but splitting does not remove between-guide lane differences. For the targeting controls, between-guide lane total variation has median 0.55 (range 0.35–0.75). Guide efficacy, other guide-associated factors and lane composition cannot be separated by this diagnostic. Nondetection would not have proved absence.

## Full pipeline source and external assets

`original_run/` preserves the original files byte for byte; do not edit its `RUN_FREEZE.json`. `read_selected.py`, `stats_impl.py` and `evaluate.py` retain the exact executed logic. `select_guides.py` retains its historical absolute provenance paths, while `select_guides_portable.py` uses relative frozen assets. The former is an archived source, not the recommended portable entry point.

Full acquisition is intentionally not started by the compact replay. For a full rerun, create a separate working copy and restore all assets listed in `RUN_FREEZE.json`, notably `metadata/obs_metadata.npz` and `frozen_assets/ntc_original/original_input_ntc_raw.npz`, from the full evidence archive or their recorded original source. Confirm their hashes. The raw-control source and original file identities are in `frozen_assets/SOURCE_PROVENANCE.json`; the D3 Rest H5AD URL, ETag and byte size are in `COMPACT_PROVENANCE.json`. The ETag and size guard the selected source row numbers. If the public source changes, stop rather than silently substituting another version.

Metadata can be regenerated through `prepare_metadata.py`, but retain the fixed archived selection rather than overwriting it. `read_selected.py smoke` checks 16 source rows against archived raw controls and an independent h5py read; `read_selected.py full` checks the frozen identities and creates resumable 256-row batches. `evaluate.py` transforms raw counts and generates means and the diagnostic; it refuses to overwrite an existing result. Work in a fresh acquisition copy, keeping archived `RESULT.json` and resampling outputs separately for comparison. The optional original pipeline requires Python, NumPy, SciPy, h5py and requests; exact versions are in `original_run/RUNTIME.json`.

To rebuild this compact representation from a complete archived run without network acquisition:

```bash
python build_compact.py --source-dir /path/to/complete/ntc_guide --output-dir /path/to/new/compact/ntc
```

The builder copies original files and constructs the cross-Gram only. It does not perform a new experiment. Run `replay.py --module-dir /path/to/new/compact/ntc` afterward. The acquisition code's original freeze checks must never be weakened to compensate for missing external files.
