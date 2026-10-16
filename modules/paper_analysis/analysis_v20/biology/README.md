# Own-gene evidence analysis — v20

This post hoc module evaluates target-expression and planning-suppression strata in the completed D4 Stim experiment. It includes the original-pool contribution partition, common-pool reranking, and annotations of the five flagged CTLA4 targets. Definitions were recorded in `POSTHOC_PROTOCOL.md` before the new scores were calculated.

## Replay in a separate directory

Use the root package requirements and run from the total package root with a new output directory:

```bash
python modules/paper_analysis/reproduce.py --only biology --out /new/path/biology-replay
```

The launcher restores `compact_inputs.npz` losslessly under the output directory before invoking the original analysis script. The compressed source directory is not a standalone execution directory.

This reconstructs scores, common-subset rankings, contributions, target annotations and conditional intervals from the supplied compact inputs. The module launcher preserves the delivered results by running restored copies in the new output directory. Original raw-cell acquisition and weight fitting remain in the original experiment pipeline.

`compact_inputs.npz` retains exact-order representations of frozen predictions and references, own-gene planning/control means, target identities, original scores and query blocks. Order representations preserve all comparisons and ties required for subset reranking. Input provenance identifies the original extraction. The optional `--extract-run` path is for rebuilding these sufficient inputs from a complete original Stim execution tree.

## Read the results

- `RESULTS.json` and `subset_comparisons.csv`: the full-cohort and restricted-pool numerical results.
- `full_pool_contributions.csv`: contributions within original rankings, summing to the complete method contrasts.
- `target_biology.csv`: continuous own-gene evidence and fixed stratum for every target.
- `CTLA4_flags.csv`: every input group and direction for the five specified candidates.
- `METHODS_RESULTS.md`: definitions and scientific interpretation.
- `VERIFICATION.json`: exact original-score and clean-replay checks.

The expression threshold and suppression ratio operate on mean normalized log1p expression. They provide descriptive own-gene evidence; their meaning is distinct from a percentage transcript knockdown or a functional null. Every method and the deep reference are reranked within the same restricted candidate pool.
