# PertRank

**Use control geometry to improve candidate selection while preserving the biological readout.** PertRank combines profile-wide effect and variance information with leading covariance directions from matched controls to calculate one reliability weight per perturbation. It applies that weight to the queried gene effect, producing separate candidate lists for increasing and decreasing the response.

This **0.2.0rc2 private review release** aligns the reusable software with manuscript v25.2. It provides an AnnData interface, reuse of fitted reliability with labeled external estimates, two real-data statistics tutorials, and regression checks against the frozen research implementation. The release is distributed for private review; public PyPI publication, scverse listing and pertpy acceptance are separate steps. The intended distribution and import name is `pertrank`.

## What the software separates

| Step | Information used | Output |
|---|---|---|
| Estimate reliability | Matched controls and each target's own effect and sampling variation across the expression profile | One bounded weight per target |
| Read the response | Original gene effects, or a labeled upstream effect estimate such as FR-Perturb | A response-specific up/down score |
| Allocate shortlist positions | The same eligible candidates and a fixed list length | Candidate lists and exact boundary-tie inclusion probabilities |

The three returned methods make the contribution inspectable: **Unweighted** uses the effect directly, **Euclidean** uses target-level reliability without directional geometry, and **PertRank** adds the fixed control geometry. A complete-workflow comparison against Unweighted and a geometry comparison against Euclidean answer different questions. See [scientific design](docs/scientific_design.md) for the equations, estimator reuse and connection to the manuscript evidence.

## Install the review release

Use Python 3.10 or later. From this directory:

```bash
python -m pip install .
```

To create a wheel from the source tree:

```bash
python -m pip wheel --no-deps . --wheel-dir dist
```

Dependencies are declared in `pyproject.toml`. For development and checks:

```bash
python -m pip install '.[test]'
python -m pytest
python scripts/check_tutorials.py
```

## Rank candidates from AnnData

Start with quality-controlled input cells and their matched non-targeting controls. Keep planning cells and reference cells separate. Gene and cell identifiers must be unique, and perturbed genes must use the same identifier namespace as query genes.

```python
import anndata as ad
import pertrank as pr

adata = ad.read_h5ad("screen_inputs.h5ad", backed="r")
result = pr.rank(
    adata,
    queries=["CTLA4", "IL2"],
    input_scale="counts",
    layer=None,                         # raw counts in X
    full_gene_axis=True,                # assert the full measured axis is present
    strata_keys=["donor", "condition", "batch"],
    target_key="target_ids",
    target_gene_key="target_gene_ids",
    control_key="is_control",
    top_k=25,
)
adata.file.close()

result.nominations.to_csv("nominations.csv", index=False)
result.weights.to_csv("reliability_weights.csv", index=False)
result.all_candidate_scores.to_csv("all_candidate_scores.csv", index=False)
```

The count path accepts one prespecified matching stratum per call. The example's donor, condition and batch columns must therefore each have one value in the supplied object. Split the screen into its prespecified matching strata before calling this path. `strata_keys=[]` explicitly declares that the supplied cells belong to one stratum; use it only when the experimental design supports that declaration.

All supplied input cells contribute to the fit. The package neither selects queries nor chooses a favorable cell subset. The paper's shallow examples use 20 cells per target and 2,000 controls; the numerical minimum is a different concept. See [input contracts](docs/input_contracts.md) for scale, control matching and memory behavior.

## Read the outputs

| Output | Meaning |
|---|---|
| `nominations` | Deterministic top-k lists for up/down responses and the Unweighted, Euclidean and PertRank methods |
| `all_candidate_scores` | Eligible query–target scores, ranks and boundary-tie inclusion probabilities |
| `weights` | Target-level reliability weights and input counts |
| `effects` | Target-by-gene original-coordinate effects |
| `controls`, `statistics`, `metadata` | Control models, sufficient statistics and computation provenance |

The up score is `effect × weight`; the down score reverses its sign. The perturbed gene is excluded from its own query. A zero weight retains the candidate at a zero score. Tie inclusion probabilities describe random selection among equally scored candidates at the cutoff; they are not probabilities of biological correctness.

## Reuse reliability with an existing effect estimate

An external effect matrix must have unique target and gene labels, the same target population, and compatible effect orientation. Align it by labels rather than by row order.

```python
import pandas as pd

external = pd.read_csv("external_effects.csv", index_col=0)
external_result = result.rank_effects(
    external, queries=["CTLA4", "IL2"], top_k=25
)
external_result.nominations.to_csv("external_nominations.csv", index=False)
```

This holds the fitted reliability calculation fixed while replacing the effect estimator. The returned `Unweighted`, `Euclidean` and `PertRank` arms then all use the supplied external effects, so the geometry comparison remains matched. The operation neither fits the upstream estimator nor re-estimates its uncertainty. Record that estimator's configuration and input design alongside the exported ranking.

## Evaluate a fixed shortlist against a separate reference

```python
reference = pd.read_csv("reference_effects.csv", index_col=0)
scores = pr.evaluate_reference(result, reference)
scores.to_csv("reference_precision.csv", index=False)
```

Provide a target-by-gene reference effect table with the same target population and every queried gene. The evaluator returns precision and expected overlap for each method, query and direction, using the result's fixed list length and self-target exclusions. It accounts for boundary ties in both the predicted and reference lists. The reference is used only for scoring. Define its construction independently of the fitted ranking; biological interpretation depends on that design. See [API reference](docs/api.md#evaluate_reference) for the exact scoring rule. Query-level uncertainty and manuscript-specific aggregation remain in the full scientific reproduction bundle.

## Try the interface immediately

```bash
python tutorials/00_anndata_quickstart.py
```

This quickstart creates a small synthetic AnnData solely for a software
check. The two atlas notebooks instead use authentic archived statistics.

## Learn and reproduce

- [API reference](docs/api.md): public functions and result fields.
- [Scientific design](docs/scientific_design.md): reliability, readout and the interpretation of matched comparisons.
- [Input contracts](docs/input_contracts.md): counts, normalized inputs and matching strata.
- [Replogle tutorial](tutorials/01_replogle.ipynb) and [T-cell tutorial](tutorials/02_tcell.ipynb): worked examples using archived real-data sufficient statistics. The tutorials identify the source and processing level of each fixture. They replay candidate ranking from saved statistics; the AnnData example above is the cell-input route.
- `tests/`: checks of the public interface and numerical agreement with the frozen implementation.
- [Community integration proposal](docs/community_integration.md): prepared scverse registration and pertpy contribution scope.
- [Private release and public handoff](docs/release_process.md): installation, artifact verification and the remaining public-release metadata.

The scientific use is candidate prioritization for follow-up experiments. Weights describe estimated profile reliability, not calibrated probabilities of biological validation. The manuscript evaluates decision quality with fixed-length lists, matched exclusions and reference-supported precision; its independent RNA comparison supports the complete workflow, while matched Euclidean contrasts isolate the added geometry.

## Attribution and release status

The computational equations and fixed parameters (`k=5`, `beta=0.9`) are inherited from the manuscript implementation. This release updates software contracts and documentation without fitting new scientific models or selecting new parameters. The accompanying manuscript and source manifests provide method and dataset attribution. Dataset rights remain with their original sources. Public redistribution and ecosystem submission require the rights holder's software-license decision; see [release process](docs/release_process.md).
