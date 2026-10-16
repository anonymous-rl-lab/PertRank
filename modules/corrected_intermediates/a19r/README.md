# A19-R: compact scoring replay from frozen intermediates

This module recalculates the existing A19-R endpoints without duplicating the raw counts or the seven-arm inputs already held in `modules/fixed_tasks`. It is the A19-R fixed-cell development reconstruction, **not** a recovery of the original A19 cell-selection seed.

```bash
python reproduce.py --output-dir /new/output/directory
# An explicit existing fixed-task location can also be used:
python reproduce.py --output-dir /new/output/directory --fixed-root /path/to/fixed_tasks
```

The output directory must be new. The sole runtime dependency is **NumPy** (verified with 2.3.5). Python standard-library modules handle file paths, manifests and reports. The default lookup is the sibling `modules/fixed_tasks` directory; no network or historical machine paths are used for execution.

The entry point actually recalculates:

- raw/Euclidean/PertRank effects and exact top-10 probabilities on all 52/59/63 candidates;
- point scores, own-target exclusions, 2,000 paired query-block resamples and list-turnover accounting;
- both original 500-draw external-reference sensitivities, shared across the three input sources;
- the archived deep diagnostic and the three-bad-control sensitivity from compact saved membership matrices.

Original point/sensitivity JSONs are numeric oracles, not the computed result. Regenerated scientific values are compared at absolute tolerance `1e-12`; all complete 500 × 3 × 4 propagation arrays must match archived array-byte SHA256 digests. `REPRODUCTION_REPORT.json` lists the exact replay depth, checked-field count, discrepancy, hashes and runtime. Generated endpoint JSONs, query scores and propagated arrays are also saved.

## Fixed and recalculated components

The core supplies original gene effects, scalar weights, query blocks, identifiers, masks, reference mean effects and losslessly encoded reference-draw top-10 memberships. The three shallow arms are reranked from effects and checked against stored predictions. `diagnostic_memberships.npz` adds only the frozen deep-arm and author-QC membership matrices. Their provenance is recorded in `PROVENANCE.json`, and their generated point endpoints are checked against the published A19-R reports.

This **does not** normalize raw counts, recompute control PCA or scalar weights from counts, redraw the external reference from raw samples, or refit native methods. The full raw-input route exists in the larger v1.5.0 evidence module; this compact module intentionally starts from its verified intermediates. The archived original evaluator/protocol are retained for provenance; `reproduce.py` is the supported portable entry point.

The deep-arm memberships and author-QC memberships are replayed, not reconstructed from counts. Reference sensitivity ranges condition on the frozen inputs; they are not population confidence intervals. The three source inputs share one independent RNA experiment. The original scorer's historical internal cross-check magnitude and hashes of omitted large float64 reference containers are retained in expected reports but excluded from numeric endpoint matching.
