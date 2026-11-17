# Fixed native rvalues comparison

This module recomputes the fixed supplementary selection-method comparison for D4 Rest, D4 stimulation and three independent-RNA input tasks. It retains the original native rvalues membership probabilities, exact tie-aware reference memberships, all comparator query-level scores, query identities and original paired query-block resampling weights.

## Run

Python 3 and NumPy are required. From this folder:

```bash
python run.py --out /absolute/path/to/new-results
```

Or invoke `run.py` by absolute path from any working directory. The result directory must not already exist. The default run needs no downloads or R installation. `REPLAY_CHECK.json` reports verification; `summary.json`, `method_scores.csv`, `paired_contrasts.csv` and task score arrays contain the newly computed results.

The scorer recomputes rvalues list overlap from native fractional membership and reference membership, combines it with the retained baseline query scores, and recomputes every matched contrast and the 2,000 paired query-block interval draws. It then reconstructs the equal-state D4 and equal-source external summaries. The three external tasks share one external reference experiment; their average has no independent-source population confidence interval.

## Lossless storage and checks

`inputs.tar.xz` stores ordinary NPY arrays and original fit-diagnostic TSV files. Every NumPy dtype, shape and array byte sequence is preserved; no precision reduction, rounding or lossy quantization is used. `INPUT_MANIFEST.json` identifies all source arrays and expanded-file hashes. `scorer.py` is byte-identical to the completed comparison's scorer.

The expected summaries and hashes are verification targets only. Predictions, overlaps, contrasts, intervals and aggregate scores are recomputed; the script does not copy the expected summary into its outputs. The check covers all 231 numeric summary values and 30 generated arrays, together with 30 main-report values.

## Native refitting and reproduction scope

This standalone compact module does not refit native rvalues or baseline estimators. Native R fits originally produced the retained membership probabilities; baseline comparator scores are retained at the per-query level. Consequently this replay reproduces the statistical comparison but is not a new end-to-end effect-estimation run.

The complete code bundle retains all native effect/SE inputs and the actual Python/R refitting workflow under `modules/rvalues_native`. The `native_refit` folder here supplies unchanged reference entry points and the exact fixed configuration; its `original_driver.py` must be used with the complete module layout, not as a standalone compact fitting entry point. Third-party terms remain in `native_refit/vendor/GPL-2.txt`.

## Interpretation

This addition is a fixed post hoc workflow comparison. PertRank-minus-rvalues contrasts assess whole workflows. The geometric increment continues to be identified by matched PertRank-minus-Euclidean contrasts; no full-workflow difference is reattributed to geometry.
