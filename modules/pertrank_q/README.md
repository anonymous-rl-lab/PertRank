# PertRank-Q: anonymous review reproduction module

This module reproduces **PertRank-Q** from retained intermediate inputs and frozen checkpoints. Its internal arm identifier is `B_rank_scale_free_pair`. Both stages' primary and secondary arms, unsuccessful direct comparisons and the original PertRank baseline remain included. The default command recomputes predictions, scores and the recorded inner-selection tables without fitting or changing selection.

## Run

Use Python 3.12 on CPU in an isolated environment:

```bash
python -m pip install numpy==2.3.5
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu
python reproduce.py --output-dir ../q-replay
```

Choose a new output directory outside this package. The command restores losslessly compressed arrays and models, reconstructs six redundant comparator prediction packets, verifies the anonymous release inventory, and executes the scientific scripts. It recomputes both development-stage result tables, exact-tie Top-K membership, paired conditional query-block intervals, 100 fixed-reference sensitivity cases and 300 recorded epoch-pair alternatives. It also checks the paper score table. `passed` means computational reproduction succeeded, not that every experimental success criterion was met.

From the total repository root:

```bash
python reproduce.py --only pertrank_q --output-dir ../q-results
python reproduce.py --with-q --output-dir ../all-results
```

## Reproduction levels

| Entry point | What it does |
|---|---|
| `python reproduce.py --verify-only` | Checks distribution hashes; no numerical replay |
| `python reproduce.py --output-dir ../q-replay` | Reconstructs frozen-checkpoint predictions, complete comparisons, intervals, selection checks and reference sensitivity |
| `python reproduce.py --extract-to ../q-experiment --output-dir ../q-extract-log` | Restores the editable logical experiment; no fitting |
| Retained training scripts, invoked separately in a new workspace | Support training from retained intermediate data; retraining is outside this verified replay |

`--skip-sensitivity` explicitly omits the reference-repeat sensitivity run. It is included by default. Exact replay is checked in the recorded Python/NumPy/Torch CPU environment; other versions are rejected rather than silently weakening equality checks. Full raw-cell preprocessing and subsequent exploratory model branches are outside this compact module.

## Results and task

Equal-source Q precision at 10 is **0.3959913849097158**, with **+0.0017475229401136079** over original PertRank on the identical candidate subpools and **+0.00028126488499435** over the retained prior comparator. Source-specific differences, intervals, both seeds and unfavorable direct arms remain in `experiment/epoch_pair_update/results/SUMMARY.json` and are recomputed by the command.

D3/D1/D2 use 293/289/289 queries and held candidate pools 26/26, 32/27 and 35/28, with K=10 in each pool. Each source is fitted separately. Three sources share one previously used external knockout RNA reference. These approximately 0.396 scores concern held candidate subpools and must not be compared directly with the complete-pool approximately 0.259 precision. Conditional 95% intervals use 2,000 paired query-block bootstrap resamples and condition on selected development models. Reference-repeat sensitivity ranges are empirical ranges, not confidence intervals or success probabilities.

## Files

- `experiment/`: scientific scripts, protocols, complete direct-comparison results, selection records, fit metadata and curves.
- `assets.tar.xz`: losslessly retained arrays, reference replicates, checkpoints and normalizers. No floating-point rounding or raw-cell data are introduced.
- `RESTORATION_MANIFEST.json`: hashes for the anonymous logical release, including six comparator prediction packets reconstructed from checkpoints.
- `restore_historical.py`: packaging adapter for those six packets; no fitting or retuning.
- `paper_data/pertrank_q_scores.csv`: current paper score table; [paper-to-code map](PAPER_CROSSWALK.md).

Private workspace locations in provenance metadata have been replaced with anonymous source-role labels. Numerical assets, model states, scientific code and reported results are unchanged. Redundant verification receipts, duplicate paper summaries and records of unrelated later development branches are omitted. Original component terms retain their scope; see the total bundle's `NOTICE.md`.
