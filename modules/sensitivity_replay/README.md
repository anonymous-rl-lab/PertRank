# Compact v1.5.1 layout

The numerical replay scope is unchanged. `array_payload.tar.xz` contains NPY
arrays in one standard-library xz archive; it contains no nested ZIP or old
release archive. No floating-point precision is reduced. `COMPACT_LAYOUT.json`
records the logical inputs and SHA-256 of every restored array's original bytes.
The launcher materializes inputs under `OUTPUT/restored_inputs`, runs the
unchanged scientific launcher under `OUTPUT/replay`, and publishes both
`OUTPUT/LOSSLESS_RESTORATION.json` and `OUTPUT/REPLAY_REPORT.json`.

All four saved 200-draw groups, original D3 weight diagnostics and full native
Mixscale coefficient/scoring inputs are retained. This module needs no other
module to restore its data. The default remains draw-summary and coefficient
scoring replay; it does not run the 200 original fits or fit native Mixscale.

`ORIGINAL_V150_FILE_MANIFEST_SHA256.json` preserves the previous physical-file
provenance. `FILE_MANIFEST_SHA256.json` describes this compact distribution;
logical data files moved into the payload appear in `COMPACT_LAYOUT.json`.

# Saved cell-sensitivity distributions and completed Mixscale scoring

```bash
python modules/sensitivity_replay/reproduce.py --output-dir /new/path/sensitivity-replay
```

Requires Python and NumPy (verified with Python 3.12.14, NumPy 2.3.5). The output
directory must be new. Successful execution returns zero and writes
`REPLAY_REPORT.json` with `status: PASS`; failed numerical checks return nonzero.
The entry point uses only relative bundled paths and makes no network request.

For SN8, the command recomputes means, sample standard deviations, empirical
2.5%/97.5% quantiles and positive fractions from the **already saved 200 draws**
for each D1/D2/D3 fixed-weight task and the D3 weight-refit task. All scores,
list-overlap distributions and paired method contrasts are checked against the
original result JSONs. This is draw-summary replay: it does not regenerate the
200 cell samples or rerun 200 weight fits. These empirical distributions are
not donor-population confidence intervals. The adverse D3 refit results remain
unchanged and visible in the receipt.

For SN2.5, the completed native Mixscale coefficient matrix is rescored against
the original D3 responsive reference using independently written full-sort
boundary membership and own-target exclusion. All four methods' 2,400 saved
query-direction scores, native-filter coverage, means and three original
2,000-draw query-block contrast intervals are checked numerically. The compact
task was extracted from the original archived sufficient statistics and
prepared target/query identities; its source hashes and exact transformations
are in `COMPACT_EXTRACTION_PROVENANCE.json`. No native R fit is run. This retains
the historical single-input opened-D3 development identity; it does not replace
the later five-input competition or imply external confirmation.

`original_source/` preserves the original execution, protocol and audit scripts
without claiming that their original workstation paths are the portable entry
point. The raw 26,000-by-18,130 D3 count matrix is intentionally not duplicated
here; full cell-level refitting requires the separate shared original inputs.
The lightweight replay verifies actual numerical arrays and summaries rather
than treating a file-hash check as experimental reproduction.
## Archived weight diagnostics

`data/refit_D3_weight_diagnostics.json` and
`data/refit_D3_weight_diagnostics.npz` preserve the original D3 refit
weight-distribution diagnostics byte for byte. These are **archived diagnosis
only**: the fast launcher does not recompute these diagnostics or repeat the
200 cell-level fits. Their original adverse findings remain unchanged.

