# Native rvalues fitting materials

This module supplies all five tasks' shallow query effects and marginal standard
errors, the Python driver and R wrapper, `rvalues` 0.7.1 source and GPL-2 text,
reference/scoring inputs, fixed protocol and original results.

Numerical inputs retain their original precision. Repeated array values use a
lossless shared store; the stimulated D4 effects and external effects are reused
from the other modules in this same repository, and the resting D4 effects reuse
identical retained support. This saves storage without removing native fitting
inputs. Restoration checks every original array byte string, all 50 restored
files, and the original 49-entry immutable-file manifest.

## Prepare native inputs

Install the dependencies listed at the package root. Then, from the package root,
run:

```bash
python reproduce.py --only rvalues_native --output-dir /absolute/path/to/new_preparation
```

The dispatcher materializes the shared inputs automatically. The resulting native
working directory is `new_preparation/rvalues_native/materials/native_rvalues`.
The command verifies and prepares inputs; it does not fit a model. The lower-level
`prepare_rvalues.py` entry is intended for an already materialized execution tree.

## Refit the native method

Enter the prepared `native_rvalues` directory. The recorded fitting environment
used R 4.5.0 and `rvalues` 0.7.1. Install the supplied native source with a C
compiler and R development headers available:

```bash
R CMD INSTALL vendor/rvalues_0.7.1.tar.gz
python run.py --mode fit --out /absolute/path/to/new_refit --workers 3
```

If needed, add `--rscript /absolute/path/to/Rscript`. The Python driver exports
NPZ effect/SE arrays into TSV and invokes
`Rscript --vanilla code/run_rvalues_precise.R INPUT.tsv OUTPUT.tsv` for each
fixed task and input repetition. The R wrapper receives no evaluation labels.
The driver opens the reference inputs only after fitting, then scores the fixed
memberships using the original tie and aggregation rules. The output path must
be new. The full fitting entry covers 7,742 fits.

## Reproduction levels

| Level | Entry | Reconstructs |
|---|---|---|
| Fixed-output scoring | `new_results` core entry | Scores and intervals from saved native memberships |
| Retained-statistic recomputation | `new_results` core entry | Program weights, lists, utility and resampling; native ASH is fixed |
| Native comparator fitting | Prepared `run.py --mode fit` | rvalues fitting from exact effects/SE, followed by scoring |

The supplied sufficient statistics support this comparator's fitting. Full raw
single-cell matrices and refitting of the other FR/ASH comparators are outside
this module's scope. Native material restoration and fixed-output replay are
verified separately; restoration does not constitute a new R fit.

## Licence identity

The supplied third-party CRAN source is GPL-2, as recorded in its original
DESCRIPTION and `vendor/GPL-2.txt`. This does not assign a licence to PertRank or
the wrapper/assembly code. The project licence field is left blank; existing
third-party rights are unchanged.
