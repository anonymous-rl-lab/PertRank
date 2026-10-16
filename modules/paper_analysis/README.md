# Paper analyses and reusable application — v1.5.4

The numerical replay scope is unchanged. `array_payload.tar.xz` contains NPY
arrays in one standard-library xz archive; it contains no nested ZIP or old
release archive. No floating-point precision is reduced. `COMPACT_LAYOUT.json`
records the logical inputs and SHA-256 of every restored array's original bytes.
The launcher materializes inputs under `OUTPUT/restored_inputs`, runs the
unchanged scientific launcher under `OUTPUT/replay`, and publishes both
`OUTPUT/LOSSLESS_RESTORATION.json` and `OUTPUT/REPLAY_REPORT.json`.

All six default analyses (`s3 biology transfer trade application figures`), all
main/SI figure and table Source Data, and the separately installable AnnData
package and tutorials are retained. `software/` remains directly installable;
its data are not dependent on unpacking the scientific archive.

The required scientific core is the sibling `../fixed_tasks/` module.
For a deliberately separate staging checkout, `PERTRANK_FIXED_TASKS` can point
to that same core directory. Four original Stim inputs are byte-verified and
copied from `supplements/20261011_stim/{data,expected}/`. TRADE's already stored
sparse effect/reference entries also come from this core. All remaining effect
entries are bundled. The 136 overlapping transfer queries are reconstructed
from those exact TRADE arrays. Biology rank codes use the original exact
`np.unique` ranking of the same float64 effects and weights; every restored
array must match its original byte hash before any scientific replay begins.
These changes eliminate duplicated storage; they do not remove data or replace
an effect-level replay with a stored final statistic.

`ORIGINAL_V150_FILE_MANIFEST_SHA256.json` preserves the previous physical-file
provenance. `FILE_MANIFEST_SHA256.json` describes this compact distribution;
logical data files moved into the payload appear in `COMPACT_LAYOUT.json`.

## Analysis scope

This module adds completed analyses and executable source from the v25.2 paper
package to the v1.4.0 scientific replay core. It is self-contained and locates
files relative to its own directory. The original scientific scripts, input
arrays, frozen locks and expected results are preserved. `reproduce.py` runs
copies in a new output directory and checks the regenerated numbers against
the delivered results; it never overwrites the reference package.

## Run the scientific replay

Use the root package's pinned numerical environment (NumPy 2.3.5, SciPy
1.17.0 and Matplotlib) and run these commands from the total package root:

```bash
python -m pip install -r requirements.txt
python modules/paper_analysis/reproduce.py --out /new/path/paper-replay
```

The directory supplied to `--out` must not exist. A subset can be selected:

```bash
python modules/paper_analysis/reproduce.py --only s3 biology transfer trade --out /new/path/statistics-replay
```

| Module | Paper evidence | What the command actually reconstructs |
|---|---|---|
| `s3` | SN15 and registered S3 bridge | Original memberships and coefficients, alignment and weight-transition accounting, paired reference-neutral slopes, 20/30/40-block sensitivity and leave-one-block-out results; independent FWL audit |
| `biology` | SN16 | All target strata, full-pool contributions, common-pool subset rankings, query arrays and CTLA4 flags |
| `transfer` | SN17 | All four-arm rankings, exact boundary ties, 10,000 source-block draws and leave-one-block-out contrasts |
| `trade` | SN18 | Five-arm complete-cohort rankings, twelve intervals, independent sort-based score audit, and all 2,805 latent-mixture second moments and weights |
| `application` | Application and CTLA4 case | Raw-count synthetic software check against the frozen engine, then all five CTLA4 CSVs rebuilt from the original archived effects, weights and reference |
| `figures` | Six main and five supplementary figures | All scientific plots from existing source data; no new model fit or test |

`REPLAY_REPORT.json`, per-command logs and regenerated outputs are written
under `--out`. Array and table comparisons use absolute and relative tolerance
1e-12; the frozen scripts also retain their original stricter internal checks.
Figures are regenerated as PNG/PDF and, where originally supported, SVG.
Image bytes can vary by plotting-library or font version. The figure command
uses the archived numbers and does not validate the biological models anew.

## Install and check the AnnData package

`software/` is the independent installable `pertrank` 0.2.0rc1 source package.
Keep it in a **separate environment**: its tested SciPy 1.16.3 requirement is
distinct from the scientific core's pinned SciPy 1.17.0. No dependency bound
has been silently changed during integration.

```bash
python -m venv /new/path/pertrank-software-env
/new/path/pertrank-software-env/bin/python -m pip install 'modules/paper_analysis/software[test]'
/new/path/pertrank-software-env/bin/python modules/paper_analysis/reproduce.py --only software --out /new/path/software-replay
```

This executes all 61 package tests against the installed package, both real-data
tutorial notebooks (all nine plain-Python code cells), and the older AnnData
adapter's dense/CSR × memory/backed verification. The latter uses a small
compatibility wrapper to retain the legacy synthetic fixture's object-string
metadata when run with pandas 3; it changes no counts or numerical function.
The modern package and historical `application/pertrank/` are isolated in
separate subprocesses. `--software-python` can select an already installed
software interpreter explicitly.

The quickstart is `software/tutorials/00_anndata_quickstart.py`; the two real-data
tutorials are `01_replogle.ipynb` and `02_tcell.ipynb`, with HTML previews. The
fixtures are authentic saved sufficient statistics, not invented cell counts.
The synthetic quickstart and correctness fixtures are labelled as such.

## Execution boundaries and provenance

- This is an offline result replay. No source-cell download, new biological
  experiment, parameter selection, R-mixture refit or donor confirmation runs.
- The original registered Stim TRADE comparison remains not evaluable. The
  completed epsilon=1e-4 numerical sensitivity retains its post hoc identity.
- No control-replacement ranking was completed in the original work. Its asset
  audit and unexecuted source are preserved; missing covariances are never
  approximated to manufacture a result.
- Full raw-source acquisition/refitting scripts remain archival source and
  require the assets named in their own documentation. They are not invoked
  by this compact replay. Their historical workstation paths are provenance,
  not dependencies of the portable entry point.
- `INTEGRATION_INVENTORY.json` maps copied paper files to any exact matches in
  the older core. A small set of frozen functions and inputs is retained in
  each module because its original hash locks require that local layout.
  Nested old release ZIPs, duplicate wheels, build trees and editorial drafts
  are omitted.
- `source_data/manuscript_tables/` contains the final manuscript display tables;
  `source_data/figures/` and `source_data/si/` support the scientific figures.
  Mechanism figures display the already completed energy/direction/query
  evidence. Their underlying analyses retain the original core replay scope.
- `validation/` records this integration's actual successful replay. Historical
  software-release metadata is explicitly retained under `software/provenance/`.
  Existing source attribution and software/data rights are unchanged; this
  source integration does not itself publish a registry or repository release.
