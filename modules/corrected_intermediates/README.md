# Compact corrected and external evidence

```bash
python reproduce.py --output-dir NEW_DIRECTORY --fixed-root ../fixed_tasks
```

Default `--fixed-root` is the sibling `fixed_tasks` module. Python 3.12 and NumPy 2.3.5 suffice. The default run is offline, takes approximately 25 seconds and writes a new `REPRODUCTION_REPORT.json`. It neither trains models nor opens remote expression data.

This module replaces the large selected-count/state reproduction in v1.5.0 with explicitly defined intermediate-data replay. The old package's raw-count claims do **not** apply to this compact version.

| Component | Retained intermediate data | Calculations actually performed |
|---|---|---|
| Guide legacy and corrected | Per-query directional scores, original block/cluster/donor labels, bootstrap arguments; fixed weight vectors and query-correlation vectors; integer guide-by-lane counts | 410 original statistical calls and 576 score means; original paired/cluster bootstrap, diagnostic summaries, lane summaries and fixed-seed lane-null calibration |
| A18-R | Top-K membership probabilities, strict and secondary per-query support, query blocks | Strict-support means, paired intervals, candidate-turnover support rates and additive identities |
| CELLEQ-R | Per-query scores at the historical measured depth rungs | Mean depth curves, interpolation and input-equivalence values; distinct from current seven-depth calibration |
| A19-R | Existing original predictions and fixed reference/500-draw memberships reused from `fixed_tasks`; tiny added deep-reference/QC membership data | Three-arm reranking and exact tie scoring, conditional query intervals and fixed-reference-draw sensitivity |

Guide score generation, target-weight estimation, control geometry, native model fitting and raw-count normalization are **not** rerun. A19-R reference draws are fixed, not regenerated from counts. The original A19 cell selection is still unrecovered; A19-R remains the documented reconstruction. A18-R and CELLEQ-R also retain reconstruction identities.

Each original result JSON is preserved separately for validation. Its presence is not counted as recalculation. Guide `capture/*/*.json` maps the stored array inputs and original statistical arguments to expected values and original result paths; `capture/*/*.npz` contains the actual query arrays used for fresh computation. Remaining historical constants and narrative decisions stay archived. Descriptive endpoint averages are not promoted to a joint confidence interval.

The two-input external-RNA replacement task in SN1.1 is not supplied by these single-input A19-R intermediates. Protein analyses belong to their separate module. Negative results and both legacy/corrected identities remain in the records.

`SOURCE_MANIFEST_SHA256.json` records all included input/code hashes and the v1.5.0 source module. `VERIFIED_REPRODUCTION_REPORT.json` records an actual execution; individual outputs state their reproduction tier. No raw counts, dense expression states or native fitting runtimes are duplicated here.
