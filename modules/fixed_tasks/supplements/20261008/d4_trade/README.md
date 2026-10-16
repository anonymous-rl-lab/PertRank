# D4 fixed TRADE scalar supplement

This module adds the completed 2026-10-08 **post-hoc descriptive** comparison to
the D1-D4 reproduction repository. The original four D4 arms, selected `PR_M`
competitor, prospective H1/H2 decisions and all original data files stay intact.
The comparison is the fixed TRADE-core scalar adaptation applied to the same
log-effect output, not native DESeq2 TRADE.

The responsive-panel PertRank minus TRADE scalar difference is
**0.028058791577898704**, conditional 95% interval
**[0.02028357242204658, 0.03496841301084284]**. The five input-group differences
are all positive. The interval conditions on the five realized groups, controls,
panel, candidate pool and deep reference; five groups are not five donors.

## Offline replay

Python 3.12 and NumPy are sufficient; the existing repository environment works.
From any location, set the repository path explicitly:

```sh
python /path/to/d4_trade/replay.py \
  --root-base-data /path/to/PertRank_D1_D4_GitHub_Core \
  --output /path/to/new_D4_supplement_verification.json
```

Python interface: `replay(base_repository_root, output_dir=None) -> dict` in
`replay.py`. When an empty/new output directory is supplied, it writes
`query_scores.npz`, `recomputed_panels.json` and `verification.json` there.
Nothing downloads, no R process is launched, and no original result is changed.
The measured standalone replay took approximately five seconds in the build
environment.

It verifies:

- Saved uniform-mixture moments and weights for **all 2,945 fits**, independently
  integrated by two-node Gauss-Legendre quadrature; maximum weight discrepancy
  in the verified build is 1.12e-15. Exact archived float64 weights are then used
  for scoring, preserving original rounding and tie handling.
- **30,000 panel score values**: 5 arms × 5 groups × 300 queries × 2 directions ×
  2 panels. This includes the **24,000 original** four-arm scores and 6,000 added
  TRADE values, rather than 30,000 new or independent observations.
- Every archived supplementary mean, pairwise contrast, query-block bootstrap
  interval and five-group record, with exact equality. Bootstrap seed 2026,
  2,000 draws and all 30 original query blocks are unchanged.
- Original H1/H2 and selected competitor remain unchanged; the original
  adjudicator receives only its original four-arm summaries.

## Why the extra data are small

The core's `data/D4/ranking.npz` contains only the old four arms' retained
top-25 support, not a complete expression matrix. Applying arbitrary new
weights directly to that support would be invalid.

`data/TRADE_support_delta.npz` adds **6,089** missing candidate effects
(58,183 bytes) required by the fixed TRADE signed top-25 lists. Every value is
the original float64 effect; all boundary ties are retained. The builder
compared **17,463,850 membership entries** from all five arms with the complete
input matrix, and they matched exactly. Deep-reference membership also matched.
The audit is in `audit/SUPPORT_DELTA_AUDIT.json` and the reproducible compactor is
`code/build_support_delta.py`. This post-result compaction is lossless for these
five arms at K=25; it is not a prospective data lock or support for new methods,
changed weights, panels, candidate pools or K.

## Contents and reproduction boundary

- `data/`: final weights and latent moments, all saved uniform mixture
  components in NumPy format, and the exact support delta.
- `fits/`: all 74 completed fit batches, original R mixture objects, per-fit TSV
  values, runtime records and completion hashes. NumPy mixtures preserve the
  distribution parameters exported from those RDS objects; the R exporter and
  independent R audit are included in `code/`.
- `code/`: the executed scoring code and exact tie scorer, unchanged byte for
  byte; support compactor and saved-mixture export/audit scripts.
- `results/`: complete five-arm panel scores, CSV means and contrasts, JSON
  summaries and report. `SCORING_COMPLETE.json` remains the historical original
  receipt; it also names the full `scoring_inputs.npz`, which is deliberately
  omitted here. The compact substitute and its validation are explicit above.
- `source/`: exact executed acquisition/fitting runner versions, original
  reader and prediction sources, official TRADE core, study adapter and rules.
  These are preserved **source records**; the archival runner's original paths
  are not rewritten or represented as an executable raw-data workflow here.
- `environment/`, `provenance/`, `audit/`: exact package versions, original
  freezes/releases, smoke and final verification records and source hashes.

The lightweight replay reconstructs fitted-mixture moments, ranking and
conditional inference. It does **not** rerun the 2,945 empirical-Bayes fits or
raw-UMI preprocessing. Those require the separately archived full-gene moments
or the full acquisition package and source H5AD. The original source is
`https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/D4_Rest.assigned_guide.h5ad`
(118,609,904,492 bytes; ETag `27f4103f28edf9cd15e64e247cd9a507-7070`).
This compact extension deliberately omits temporary TSV inputs, source caches,
the raw H5AD, full-gene means/variances and duplicate full effect matrices.

The calculation was newly performed from the original frozen 58,900 D4 input
cells. It does not claim recovery of the earlier interrupted run's partial
fits. No stimulated conditions or new fitting parameters were introduced.

## Third-party source

Read `source/trade/UPSTREAM_NOTICE.md`. The original TRADEtools core, its study
adapter, and their authorship are distinguished. The upstream snapshot has no
usable license declaration; this module does not assign it a new license.

`MANIFEST.json` binds every included file and the five unchanged base files on
which the offline replay depends.
