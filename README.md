# PertRank reproducibility package

This package contains the scientific code, compact intermediate inputs, current manuscript source tables and figure scripts, the installable PertRank software, and the optional PertRank-Q checkpoint replay. No author contact details, project repository links, registration identities or Git history are required to run it.

## Start here

| Purpose | Entry |
|---|---|
| Scientific design | [Scientific design](modules/paper_analysis/software/docs/scientific_design.md) |
| Python API | [API reference](modules/paper_analysis/software/docs/api.md) |
| Real Replogle tutorial | [Replogle notebook](modules/paper_analysis/software/tutorials/01_replogle.ipynb) |
| Real T-cell and estimator-reuse tutorial | [T-cell notebook](modules/paper_analysis/software/tutorials/02_tcell.ipynb) |
| Current paper coverage and exact starting points | [Coverage](docs/CURRENT_PAPER_COVERAGE.md) |
| PertRank-Q | [Q instructions and scope](modules/pertrank_q/README.md) |

The three documentation links are relative paths to files included in this package. They work within the repository and from the extracted package. The package is self-contained after dependency installation.

## Scientific replay

Use Python 3.12. Exact input-file restoration was verified with Python 3.12.14 and zlib 1.3.2; see the [storage environment and checks](storage/README.md). Create the environment and all output directories **outside** this package.

```bash
python3.12 -m venv ../pertrank-env
source ../pertrank-env/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --verify-only
python reproduce.py --output-dir ../pertrank-reproduced
```

The default run executes nine scientific workflows, including the independent-RNA four-arm and guide diagnostics, rvalues scoring, and program nomination. PertRank-Q is available through its optional checkpoint workflow. `--verify-only` checks the complete delivered file inventory; it does not run experiments. `--list` shows each workflow's starting point and `--only NAME` selects one. Output directories must be new. The replay runs offline after dependency installation. The dispatcher first restores exact input files into `OUTPUT/_source/`; this execution copy requires additional disk space, while the delivered source directory remains unchanged. All data stores are included in this one repository.

## Optional PertRank-Q

Q requires the separately listed dependencies:

```bash
python -m pip install -r modules/pertrank_q/requirements-q.txt --extra-index-url https://download.pytorch.org/whl/cpu
python reproduce.py --only pertrank_q --output-dir ../pertrank-q-reproduced
# Or all scientific workflows plus Q:
python reproduce.py --with-q --output-dir ../pertrank-all-reproduced
```

Q restores retained model inputs and checkpoints, computes predictions, and repeats the candidate-subpool rankings and conditional intervals. It does not retrain the network. Its target-held-out subpools and K=10 are distinct from the full candidate pools in the unsupervised evaluation. All three external-RNA input sources share one independent reference experiment. Original PertRank remains the reference-free method; Q uses calibration reference labels.

## Installable software

Use a separate environment for the software because its dependency range differs from the pinned scientific replay. Build from a copy, avoiding generated build files inside the checksum-protected review package:

```bash
python -c "from pathlib import Path; import shutil; shutil.copytree(Path('modules/paper_analysis/software'), Path('../pertrank-software-source'))"
python3.12 -m venv ../pertrank-software-env
../pertrank-software-env/bin/python -m pip install -r requirements-software.txt
python reproduce.py --only software --software-python ../pertrank-software-env/bin/python --output-dir ../pertrank-software-check
```

The copied software includes its own README, API documentation, tests and real-data notebooks. The bundled software version is **0.2.0rc3**. Component-specific license notices are retained.

## Scope and integrity

The package uses retained effects, sufficient statistics, reference memberships, per-query scores, model inputs and checkpoints, with each workflow documenting its starting point. It does not claim that all analyses are refitted from raw sequencing counts. A successful display regeneration is distinct from recomputing its underlying experiment. Missing original input and archival-only evidence remain declared in [reproduction scope](docs/REPRODUCTION_SCOPE.md).

Compression is lossless: retained numerical precision is unchanged. The checksums describe the delivered files and restored numerical inputs. The source tables and figure scripts correspond to the current manuscript; reported unfavorable outcomes and their required evidence are retained.

Exit codes: **0** selected declared checks passed; **1** integrity, execution or numerical failure; **2** partial or unavailable coverage. Inspect the module receipts for what was computed.


## Program nomination, ranking comparison and diagnostic workflows

`modules/new_results/` contains native rvalues fixed-output scoring plus D3 program nomination and sci-Plex3 frozen evaluation. Its core entry recomputes weights, lists, utility, statistical summaries and replacement accounting from the declared retained inputs; the native ASH and rvalues fits remain fixed. The exploratory entry retains the unsuccessful geometry adjustments separately.

```bash
python reproduce.py --only new_results --output-dir ../pertrank-new-results
python reproduce.py --only new_results_exploratory --output-dir ../pertrank-program-exploratory
python reproduce.py --only sn4_diagnostics --output-dir ../pertrank-sn4
python reproduce.py --only rvalues_native --output-dir ../pertrank-rvalues-materials
```

The last command restores and verifies the native rvalues fitting materials included in this package. Follow `modules/rvalues_native/README.md` to refit from the exact effects and standard errors using R. All scientific modules, fitting inputs, checkpoints and figure data are delivered together. See [evidence coverage](docs/INTEGRATION_COVERAGE.md) for their locations and reproduction levels.
