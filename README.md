# PertRank reproducibility package v1.5.4 — compact intermediates

This is the private anonymous review distribution. Project-owned registration identifiers use role aliases; the confidential mapping is supplied separately. See `docs/REVIEW_IDENTITY.md` and `docs/CODE_AVAILABILITY.txt`.

This release organizes the scientific code and retained intermediate data needed to reproduce the manuscript's reported comparisons. It removes redundant delivery files and keeps the scientific code and numerical arrays identified in the migration inventory, `docs/RELEASE_MIGRATION.csv`. The complete ZIP remains below **100,000,000 bytes**; see the delivered size report for measured byte counts. This version is supplied as one complete package.

## GitHub Desktop

This archive includes an initialized local Git repository with no commits or remote. Extract the `PertRank` folder to a short path, then choose **File → Add Local Repository** and select that folder. Commit the files before publishing; keep the repository private for anonymous review. See [Windows instructions](docs/GITHUB_DESKTOP_zh.md).

## Run

Use Python 3.12. Keep the environment and all generated output outside the package.

```bash
python3.12 -m venv ../pertrank-env
source ../pertrank-env/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --verify-only
python reproduce.py --output-dir ../pertrank-reproduced
```

`--output-dir` must name a new directory. Use `--list` to see individual modules, or `--only NAME` to select one. The command writes numerical comparisons and logs to the output directory and leaves the package inputs unchanged. Python scientific replay is offline; no upstream models are refitted.

The v1.5.3 software update replaced the software with **PertRank 0.2.0rc2**, aligned with manuscript v25.2. It fixes external-estimator reuse and adds an exact-tie decision scorer. Scientific arrays and manuscript tables are byte-identical to v1.5.2; the seven default scientific workflows retain their prior validation. See `validation/SOFTWARE_UPDATE.json` for the current software checks.

## What changed

| Module | Compact starting point |
|---|---|
| `fixed_tasks` | Scientific code, saved statistics and estimator-output replays retained from the v1.4.0 core; packaging changes are recorded in the migration inventory |
| `early_intermediates` | Per-query scores, fixed block labels and retained diagnostic inputs for the early factorial, mechanism and baseline analyses |
| `corrected_intermediates` | Guide/A18/cell-equivalence score and curve intermediates; reuse of the core's independent-RNA predictions/reference memberships |
| `r20_intermediates` | Three sources' query scores and original query blocks for five internal and two external input groups |
| `paper_analysis` | Losslessly factored analysis arrays, original source tables, figure scripts, application and installable software |
| `il2_intermediates` | Guide-level log-ratio effects, fixed candidate/list identities and retained MAGeCK projections |
| `sensitivity_replay` | Saved resampling draws and fitted Mixscale coefficients; no repeated model fitting |
| `native_workflows` | Source-only original adapters, configurations and upstream provenance; outside the default replay |

Intermediate replay regenerates means, contrasts and uncertainty intervals from the supplied lower-level data. It is different from reproducing cell preprocessing, control PCA or estimator training from raw counts. Each module states its exact starting point and records what was checked. Numeric arrays are compressed or factored losslessly; no float downcasting or result rounding is used to meet the size limit.

Removed additions include selected-cell raw counts, full moment/forecast tensors superseded by query-level inputs, duplicate arrays, vendor source/build payloads where unnecessary, and regenerable rendered outputs. Historical A14 remains an explicitly retained result with an unavailable original per-cell input; reconstructed A9/legacy guide identities stay documented. These historical limitations are not converted into raw-data reproduction claims.

## Paper and validation map

`docs/PAPER_COVERAGE.csv` links manuscript sections, tables and figures to the compact modules. `docs/REPRODUCTION_SCOPE.md` explains the levels. Module receipts and root `validation/` distinguish numerical replay from source-table rendering and historical archive retention. `MANIFEST_SHA256.json` checks the full release payload; root `.git/` metadata is outside the payload. Unknown extra payload files or changed inputs are rejected.

Exit codes: **0** means selected commands passed their declared checks; **1** means integrity, execution or numeric failure; **2** identifies partial coverage, a stored-result-only task or an explicit execution limit. The reports, not a generic success label, define what was reproduced.

## Software and tutorials

The AnnData package and two real-data tutorials are in `modules/paper_analysis/software/`. They use a separate environment because the software and frozen numerical replay have different SciPy requirements. Install from a **copy outside this package**: a local pip build writes `build/` and `*.egg-info/` into its source directory, while the reproduction manifest requires the delivered package to remain unchanged.

From the package root, use a new external source directory:

```bash
python -c "from pathlib import Path; import shutil; shutil.copytree(Path('modules/paper_analysis/software'), Path('../pertrank-software-source'))"
python3.12 -m venv ../pertrank-software-env
../pertrank-software-env/bin/python -m pip install -r requirements-software.txt
python reproduce.py --verify-only
python reproduce.py --only software --software-python ../pertrank-software-env/bin/python --output-dir ../pertrank-software-check
```

The copy command refuses to overwrite an existing source directory. The requirements file refers to that external copy and must be used from this package root. The software environment installs the copied package; tests and tutorials are then run by the unified entry point with outputs outside the release. Keep the numeric replay environment active when invoking `reproduce.py`.

Component licences and dataset citations retain their original scope; see `NOTICE.md`. No repository has been published remotely by this packaging release.
