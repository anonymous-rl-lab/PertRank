# PertRank v20: from input counts to nomination lists

This application example calculates PertRank weights from supplied cells and returns original-gene-coordinate nominations. It uses the unchanged v17 cell engine, with a new count-input adapter matching the T-cell study's normalization and numerical precision. It does not load saved rankings, a deep reference or a selected outcome. The separate frozen research replay remains the source for reproducing the manuscript's experiments.

The supported task is a prespecified query list in one prespecified control-matching stratum. Select quality-controlled input cells and matched non-targeting controls before running the program. For the paper's shallow setting, supply 20 distinct cells per perturbation and 2,000 controls. All supplied cells are used. Other input depths are computationally accepted but are not automatically supported by the shallow-setting results. Queries and any planning cells used to select them are additional resources; do not mix planning or evaluation cells into the inputs.

## Run the educational example

The v20 release was verified with Python 3.12.14; exact tested dependencies are in `requirements_verified_v20.txt`. NumPy runs the count-input application; SciPy supports verification and sparse matrices, and AnnData enables `.h5ad` input.

```bash
python -m pip install -r requirements.txt
python make_example.py --out demo_input.npz
python run_counts.py demo_input.npz --out demo_nominations --top-k 3
python verify_application.py
```

The generated example has 2,000 controls, eight perturbations with 20 cells each, 320 genes and three supplied queries. The top-three lists keep the output readable. Its fixed-seed integer counts are synthetic software inputs, not a performance experiment. Output paths must be new so previous runs remain intact.

## Supply an experimental input

Create an ordinary NPZ (no Python objects) with the following arrays. A matrix, cell metadata and query IDs must share exactly the stated axes.

| Array | Shape | Meaning |
|---|---|---|
| `X` | cells × full genes | Dense, nonnegative integer UMI counts on the complete measured gene axis |
| `cell_ids` | cells | Unique identifiers, checked to prevent repeated input rows |
| `gene_ids` | full genes | Unique canonical gene identifiers |
| `target_ids` | cells | Perturbation identifier; NTC labels must not label perturbation cells |
| `target_gene_ids` | cells | Perturbed gene in the `gene_ids` namespace; use `NTC` for controls |
| `batch_ids` | cells | One prespecified matching stratum, such as one donor and condition |
| `is_control` | cells | Boolean or 0/1 input-control indicator |
| `query_ids` | queries | Unique query genes chosen before ranking |

```python
np.savez_compressed(
    "input.npz", X=raw_counts, cell_ids=cell_ids, gene_ids=gene_ids,
    target_ids=cell_target_ids, target_gene_ids=cell_target_gene_ids,
    batch_ids=matching_stratum_ids, is_control=is_input_control,
    query_ids=prespecified_query_ids)
```

Then run:

```bash
python run_counts.py input.npz --out nominations_v20 --top-k 25
```

The adapter calculates library totals **before** retaining genes detected in input controls. It scales each cell to the median total of the input controls and applies `log1p`. It follows the frozen T-cell implementation by storing control log values at float32 precision before float64 moments; target log values remain float64. Five control covariance directions and attenuation 0.9 are fixed. Each target's own sample variance, including variance after projection, contributes to its noise estimate; control variance is not substituted for target variance. A gene not detected in controls cannot be used as a query in this adapter.

The NPZ adapter loads the complete dense matrix. The v20 AnnData adapter below instead batch-reads the full raw gene axis, retains the control matrix for the exact eigendecomposition and processes one target at a time. The control matrix and smaller control/gene Gram matrix must fit in memory. A minimum of six controls and two target cells are numerical requirements only, not recommended experimental budgets. For a specified multi-stratum design or an already normalized gene axis, the original v17 `pertrank` Python interface remains available in this directory. That general interface has its own input conventions; this adapter deliberately has a narrower T-cell path.

## Use AnnData or backed h5ad

Put full-axis raw integer counts in `adata.X`. Use unique canonical gene IDs as `adata.var_names`, unique cell IDs as `adata.obs_names`, and four aligned `obs` columns: `target_ids`, `target_gene_ids`, `batch_ids`, and Boolean `is_control`. The supplied object contains prespecified input cells and input controls from one matching stratum. Write one previously selected query gene ID per line to `queries.txt`.

```bash
python run_h5ad.py input.h5ad --queries queries.txt --out nominations_v20 --top-k 25 --batch-rows 512
python verify_h5ad.py
```

For an in-memory AnnData object:

```python
from run_h5ad import fit_anndata
result, metadata = fit_anndata(adata, prespecified_query_ids, batch_rows=512)
```

Dense and CSR matrices are supported both in memory and through `read_h5ad(..., backed="r")`. The first pass computes library totals over every provided gene and identifies the input-control gene axis. A second pass computes the control model; each perturbation is then processed separately. The complete target-cell matrix is never densified. All supplied controls contribute to the fixed five-direction model, and all supplied target cells contribute to that target's mean and sample variance. Use the full measured gene axis, since a gene-filtered object cannot recover omitted library counts. Raw counts in a named layer can be explicitly assigned to `X` before writing the input file.

`VALIDATION_H5AD_v20.json` records exact parity with the dense adapter for four paths (dense/CSR × in-memory/backed), using shuffled cell order and an extra gene with counts only in target cells. This final check confirms that denominator counts survive control-axis filtering. On the software fixture of 2,160 cells × 321 genes, the four fits took 0.027–0.089 seconds with one BLAS thread; peak process resident memory was 133.1 MiB. These numbers describe the synthetic correctness fixture and include the Python/AnnData process. Actual experiment cost is reported in each `run.json`; the `.h5ad` raw-data interface has not been timed on the original full cohort.

## Inspect the outputs

| File | Use |
|---|---|
| `nominations.csv` | Complete deterministic top-k lists for both directions and Unweighted, Euclidean and PertRank |
| `all_candidate_scores.csv` | Every eligible query–target score, original effect, weight, rank range and boundary-tie inclusion probability |
| `weights.csv` | All target weights and actual target-cell counts |
| `input_statistics.npz` | Original effects, total and projected energies/noise, gene and target axes |
| `control_model_0.npz` | Control mean, variance, directions and projected variances |
| `run.json` | Input hash, parameters, normalization, dimensions and actual resource counts |

For an up list, the signed score is the weighted original effect; for a down list its sign is reversed. The perturbed gene is excluded from its own query by identifier, regardless of target label. Zero weights retain candidates at zero score. Boundary probabilities sum to the list size and describe uniform handling of tied scores; they are **not** biological confidence probabilities. Deterministic hash tie breaking yields a usable list, whereas boundary probabilities retain the exact expected inclusion used by the research evaluations. They must not be confused with uncertainty intervals.

## Verification and limits

`verify_application.py` refits controls from the synthetic counts and checks cell-level effects and weights against the unmodified frozen D4/Stim engine in `verification/frozen_engine.py`. It also checks noise by constructing the full covariance matrix, including off-diagonal terms, and tests complete output generation, own-gene exclusion, tie probability mass and duplicate-cell rejection. The results and numerical tolerances are recorded in `VALIDATION_v20.json`; code identities are in `SOURCE_PROVENANCE.json`.

These checks verify the implemented equations and input-to-output path. They are not new biological validation, a replay of every original raw-data preprocessing step, or a claim that arbitrary screen designs share the paper's benefit. The entry point adds no learned adoption threshold, statistical test, calibrated reliability probability or independent-reference claim. The paper's strongest evidence concerns shallow RNA nomination for responsive queries with matched controls.

The `CTLA4_worked_example.md` and associated tables separately describe one fixed, already-completed Stim nomination task. They show how to inspect shortlist changes, not a new experiment or an external validation.

## Source and rights

The `pertrank/` computational core is copied unchanged from the v17 manuscript's local usage interface. The v20 adapters support dense NPZ and backed AnnData inputs. The archived core exposes its original version 0.1.0; `run.json` identifies this application adapter as v20.0. No package registry release or new software/data license is implied. Rights and source attribution remain those of the manuscript and source experiments.
