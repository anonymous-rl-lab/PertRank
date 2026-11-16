# PertRank boundary context and boundary learning, 2026-11-15

This archive reproduces training and evaluation **from saved numerical intermediates**. It includes original20-cell input features, reference replicates, target/query identities, every new fit, frozen normalization, weights, curves, imported historical comparator A, plans and complete result summaries. No earlier experiment package is required. It does not rebuild raw-count preprocessing; provenance and that boundary are recorded in `inputs/DATA_REPORT.md` and `inputs/DATA_PROVENANCE.json`.

## Fixed three-arm experiment

A/base_context imports all prior no-extra-geometry neural gate predictions, with no refit or outcome-based seed selection. B/boundary_context adds input-only list-position, boundary-distance and crowding features, retaining the SoftTopK objective. C/boundary_pair has identical architecture and initialization to B but trains with empirical-reference-support-weighted local pair loss. Every arm retains the original geometric wM anchor and raw signed effect. The extra three original geometry predictor columns are zero; this is not a geometry-free model.

All source models and normalizers are separate. There are48 new inner fits and24 new outer refits, plus12 imported A fits. The sources share one external assay. Targets are held out within the current fit, but reference outcomes were accessed in preceding development: this is exploratory, not a new blind confirmation.

Relative to original PertRank, equal-source external RNA delta P@10 is:

| Arm | Delta |
|---|---:|
| A, historical working model | +0.001231641 |
| B, boundary context | +0.001466258 |
| C, boundary pair loss, declared primary | −0.001205562 |

The declared primary C fails. B has positive point estimates in all sources, with D2 conditional95% interval [+0.000594949,+0.005089405]. B−A is only +0.000234617 and every source comparison interval includes zero. Do not claim a confirmed improvement of B over A or promote B retroactively to primary.

The metric is computed separately in held candidate halves (26–35 candidates), then directions/halves/queries/sources are averaged according to the frozen plan. Do not compare these P@10 values with the older full52/59/63-target task (~0.259). Exact fractional ties and query-self exclusion are preserved.

## Reproduce predictions and scoring

Use Python3.12 and the versions in requirements.txt (CPU PyTorch via its official CPU wheel index).

```bash
python replay_boundary.py
```

This rebuilds all views and six dense prediction arrays from the compact inputs and saved fits, then recomputes scores in a fresh results_replay directory. All1,436 numeric summary fields and all nine result-array SHA-256 values were reproduced exactly in a clean compact test. Original dense predictions, views and result arrays are deliberately omitted from this archive because they are reconstructed. The complete historical/ folder is retained, verified and self-contained; A is independently replayed from its weights.

For a second replay use a new --output-dir. `COMPACT_REPLAY_CHECK.json` records the verified compact test.

## Refit

```bash
python retrain_boundary.py --out ../boundary_retrained
```

The destination must be new. The wrapper preserves historical A and copies all required code/data, runs inner fits, checks the fixed resource-feasibility gate, then runs outer fits and scores only if the gate passes. It does not select new hyperparameters. The frozen run used about108 seconds summed across72 CPU fits. This timing excludes code development and ancillary checks; wall time varies by hardware. Replayed frozen predictions were byte-identical in the tested environment; fresh training depends on software/hardware numerical behavior.

## Diagnostics and limits

- `BOUNDARY_CHECKS.json`:21 numerical/context/gradient checks.
- `audit/SCORING_SYNTHETIC_AUDIT.json`:700 synthetic scoring checks.
- `audit/SAVED_TRAINING_AUDIT.json`:72 saved normalizers recomputed, source/train isolation and reference-free features verified.
- `audit/CALIBRATION_SUPPORT_PROBE.json`: training-side bootstrap support differences, not latent truth or generalization evidence.
- `audit/BOUNDARY_BACKTRACE.json`: post-score split of support contribution within/outside the original local boundary bands; additive accounting, not causal attribution.
- `results/REPORT_zh.md`: all sources, halves, seeds, conditional intervals and entrants/exits.

The optional local-repair diagnostic under audit/ is explicitly post-score. It preserves the existing main results. Its fixed two-slot rule is inherited from the training bands, not selected from a window search. It must be reported with its development identity and all arms, rather than used to overwrite a failed primary result.

The external reference includes calibration targets used in training these gates. These learned corrections therefore require external calibration data; they do not have the same reference-free input requirements as original PertRank.

The fixed post hoc local-list rule did not improve the development conclusion. Its mean delta versus original was A+0.000742, B+0.000745, C−0.001094. All original main results remain unchanged. The full account is `RESULTS_zh.md`; exact local results are in `audit/local_repair/`.

```bash
python replay_local_repair.py
```

This reconstructs the frozen local memberships and their complete source/task numerical summaries. The original diagnostic script is preserved as `audit/local_repair_probe_frozen.py`; its hash matches the diagnostic plan. The working script fixes an empty-own-mask assertion only. `audit/LOCAL_REPAIR_CORRECTION.json` records the repair; the six real tasks and their results were unaffected. Both scripts, correction and independent checks are retained.
