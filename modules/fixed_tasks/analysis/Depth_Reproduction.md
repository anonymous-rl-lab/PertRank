# Exact reproduction of the depth calibration

This package reproduces the archived raw-input depth curves at 20, 25, 30, 35, 40, 45 and 50 cells, the weighted 20-cell reference levels, and the corresponding conditional inverse intervals. Both query panels and all five input anchors are retained for each donor. No new experiment or parameter selection is performed.

## What is retained

`data/D*/depth.npz` stores the complete candidate membership strictly above each signed top-25 cutoff and the complete set tied at that cutoff. The two masks are bit-packed. Integer numerators and denominators encode the exact boundary inclusion probability. Consequently, the representation preserves the expected overlap exactly, including very large ties and exclusion of the queried gene's own perturbation. It does not choose one arbitrary top-25 list.

The same representation is retained for the deep reference and the three 20-cell arms: unweighted, Euclidean weighting and PertRank. Below-cutoff ordering and the magnitudes of deeper expression profiles are not retained. This compact representation is sufficient for the archived P@25 analysis and its fixed bootstrap; it is not sufficient for a different K, new weights, new query panels or new expression-based analyses.

Axes are depth × anchor × query × sign × packed candidate byte for raw-depth masks, arm × anchor × query × sign × packed candidate byte for weighted 20-cell masks, and query × sign × packed candidate byte for reference masks. Signs are +1 and −1. Packing uses NumPy's `bitorder='little'`. For each row, membership is `above + tied * numerator / denominator`. There are 25 total expected members per signed query. Candidate identifiers and query identifiers are retained.

## Reproduction

From the repository root:

```bash
python src/replay_depth.py --output reproduced/depth.json
```

The scorer reconstructs inclusion probabilities, recalculates every query score from the candidate axis, averages input anchors and query directions in the frozen order, and recomputes the 2,000 query-block bootstrap draws and inverse summaries. `src/vendor/depth_equivalence.py` is an unchanged copy of the frozen depth mathematics used for D4, which permits the actual number of query blocks; all archived panels in this package have 30 blocks. The original source JSONs and score/bootstrap arrays in `reference_results/` are comparison targets, not inputs to score estimation.

All four donors reproduced with **zero numerical difference** for the query scores, bootstrap arrays and reported depth summaries. The verification record is `DEPTH_REPLAY_VERIFICATION.json`; source-file fingerprints and packing scope are in `provenance/depth_sources.json`.

## D1 panel distinction

**D1's depth calibration is a historical-panel sensitivity analysis.** Its original seven-depth experiment used the fixed R20 panel with 595 union query genes. The current corrected D1 primary analysis uses a different query panel. No corrected-panel D1 depth experiment is claimed or manufactured here. The historical D1 depth result is kept in its own self-contained file and must not be substituted for the corrected primary result.

The responsive-panel PertRank equivalents are:

| Donor | Unweighted-input equivalent of PertRank at 20 cells | Conditional 95% inverse endpoints | Scope |
|---|---:|---:|---|
| D1 | 32.3059 | 29.5436–34.7621 | Historical R20 panel only |
| D2 | 24.8144 | 22.6968–27.2191 | Frozen development panel |
| D3 | 25.4296 | 23.3887–27.3004 | Frozen development panel |
| D4 | 38.8738 | 36.1040–41.3274 | Prespecified secondary confirmation analysis |

These are ranking-input equivalents conditional on the realized 100-cell reservoir, planning resources, controls, queries, candidate pool and deep reference. They are not total experiment-cost or sequencing-cost savings. Adjacent depth levels and the five depth anchors overlap; they are not independent experiments. Linear interpolation may overestimate an inverse crossing on a concave curve. Unidentified boundary draws are retained under the frozen rule; no extrapolation is used. Flat tercile curves must not be presented as finite cell-saving claims when the required crossing is absent.
