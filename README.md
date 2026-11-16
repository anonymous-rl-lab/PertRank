# PertRank: anonymous review reproducibility package

This package contains the scientific code, compact intermediate inputs, current manuscript source tables and figure scripts, the installable PertRank software, and the optional PertRank-Q checkpoint replay. No author contact details, project repository links, registration identities or Git history are required to run it.

## Start here

| Purpose | Entry |
|---|---|
| Scientific design | [Scientific design](https://github.com/anonymous-rl-lab/PertRank/blob/main/modules/paper_analysis/software/docs/scientific_design.md) |
| Python API | [API reference](https://github.com/anonymous-rl-lab/PertRank/blob/main/modules/paper_analysis/software/docs/api.md) |
| Real Replogle tutorial | [Replogle notebook](https://github.com/anonymous-rl-lab/PertRank/blob/main/modules/paper_analysis/software/tutorials/01_replogle.ipynb) |
| Real T-cell and estimator-reuse tutorial | [T-cell notebook](modules/paper_analysis/software/tutorials/02_tcell.ipynb) |
| Current paper coverage and exact starting points | [Coverage](docs/CURRENT_PAPER_COVERAGE.md) |
| PertRank-Q | [Q instructions and scope](modules/pertrank_q/README.md) |
| Audit results | [Audit](docs/AUDIT_zh.md) |

The scientific design, API reference and Replogle tutorial links above open the corresponding files in the anonymous GitHub repository over HTTPS. The same files remain in the software subtree for offline use. PyPI does not publish individual source-archive files as web pages; the generated PyPI description preserves these HTTPS links and converts the remaining local links into explicit source-archive paths.

## Scientific replay

Use Python 3.12. Create the environment and all output directories **outside** this package.

```bash
python3.12 -m venv ../pertrank-env
source ../pertrank-env/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --verify-only
python reproduce.py --output-dir ../pertrank-reproduced
```

The default run executes eight workflows, including current-paper independent-RNA four-arm and guide diagnostics. `--verify-only` checks the complete delivered file inventory; it does not run experiments. `--list` shows each workflow's starting point and `--only NAME` selects one. Output directories must be new. The replay runs offline after dependency installation.

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

The copied software includes its own README, API documentation, tests and real-data notebooks. The source version is **0.2.0rc3** and is supplied for review; this delivery does not publish a new PyPI release. The existing component license is retained; no open-source license is inferred or assigned by this packaging step.

## Scope and integrity

The package uses retained effects, sufficient statistics, reference memberships, per-query scores, model inputs and checkpoints, with each workflow documenting its starting point. It does not claim that all analyses are refitted from raw sequencing counts. A successful display regeneration is distinct from recomputing its underlying experiment. Missing original input and archival-only evidence remain declared in [reproduction scope](docs/REPRODUCTION_SCOPE.md).

Compression is lossless: retained numerical precision is unchanged. The active checksums describe the current delivered files. The current source tables replace obsolete display snapshots; evidence needed to reproduce reported results, including unfavorable outcomes, is retained.

Exit codes: **0** selected declared checks passed; **1** integrity, execution or numerical failure; **2** partial or unavailable coverage. Inspect the module receipts for what was computed.

For GitHub Desktop, see [instructions](docs/GITHUB_DESKTOP_zh.md). The archive has no `.git` directory. Component licenses and independent dataset attribution are described in [NOTICE.md](NOTICE.md). Author-identifying links are omitted from this review copy; the documentation links use the anonymous GitHub repository.
