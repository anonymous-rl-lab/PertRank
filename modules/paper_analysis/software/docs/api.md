# API reference

Import the public interface with `import pertrank as pr`.

## `rank`

```python
pr.rank(
    adata,
    queries,
    *,
    input_scale,
    strata_keys,
    target_key="target_ids",
    target_gene_key="target_gene_ids",
    control_key="is_control",
    layer=None,
    top_k=25,
    seed=2026,
    batch_rows=512,
    full_gene_axis=False,
    library_totals_key=None,
)
```

Compute reliability weights and query-specific candidate lists from input cells.

| Argument | Contract |
|---|---|
| `adata` | AnnData with unique observation and variable names; only prespecified input cells and matched input controls |
| `queries` | Unique response-gene identifiers on the input gene axis |
| `input_scale` | Required: `"counts"` for raw integer counts or `"log"` for the declared, already normalized log-expression coordinate system |
| `strata_keys` | Required observation column names defining matched controls; `[]` explicitly declares one common stratum |
| `target_key` | Observation column identifying perturbations; control labels and perturbation labels must be disjoint |
| `target_gene_key` | Observation column naming each perturbed gene in the `var_names` namespace; controls use a nonempty placeholder |
| `control_key` | Observation column containing Boolean or 0/1 input-control indicators |
| `layer` | Named matrix layer, or `None` to read `X` |
| `top_k` | Positive integer, no larger than each query's candidate pool after self-target exclusion |
| `seed` | Integer controlling reproducible hash-based tie selection; it does not select cells or fit parameters |
| `batch_rows` | Positive read-chunk size for the count-input path |
| `full_gene_axis` | Required `True` assertion for count input so library totals precede gene filtering |
| `library_totals_key` | Counts-only observation column containing retained full-library totals; if supplied, its values must exactly equal the supplied raw matrix row sums |

The geometry parameters are fixed at five directions and attenuation 0.9. Returns `PertRankResult`. See [input contracts](input_contracts.md) for single-stratum counts, multiple-stratum normalized inputs, control matching and memory behavior.

## `PertRankResult`

| Attribute | Type | Contents |
|---|---|---|
| `nominations` | pandas DataFrame | Delivered query/method/direction top-k lists, scores and tie information |
| `all_candidate_scores` | pandas DataFrame | Eligible candidate scores and rank information |
| `weights` | pandas DataFrame | Target-level weights and input-cell counts |
| `effects` | pandas DataFrame | Effects currently used for ranking, indexed by target and gene identifiers |
| `controls` | mapping | Fitted control model for each matching stratum |
| `statistics` | mapping | Original source-cell effects, energies, variance contributions and labeled axes underlying the reliability fit |
| `metadata` | mapping | Software version, effect origin, scale, fixed parameters, axes and run configuration |

The result is separate from the input AnnData. Inspect or export the returned DataFrames; no persistent output file is written automatically by the ranking operation. `effect_origin` identifies source or external effects, and `pertrank_version` records the generating software. Following external-effect reuse, `effects` and `statistics["effects"]` serve different roles: the former is the current ranking readout, and the latter remains the original reliability input.

## `PertRankResult.rank_effects`

```python
external_result = result.rank_effects(
    external_effects,
    queries=None,
    top_k=None,
    seed=None,
)
```

Apply the fitted target weights to a labeled external effect matrix. `external_effects` is a pandas DataFrame indexed by target identifiers with gene identifiers as columns. Label matching is checked, and row order is reconciled by label. `None` uses the original result's query list, shortlist size or tie seed. Returns a result with new effect-based rankings and the existing reliability weights.

The external matrix must contain every fitted target exactly once. Its columns may be a subset of the retained source-effect gene axis, provided they contain every requested query. The external estimator must use a compatible input population and effect orientation. The operation neither estimates external model parameters nor learns new weights. This method supplies a direct route for using PertRank reliability with another effect estimator. The equivalent top-level call is `pr.rank_effects(result, external_effects, queries=None, top_k=None, seed=None)`.

The external result preserves independent copies of the source `weights`, `controls` and `statistics`. Its `effects`, `nominations` and `all_candidate_scores` use the new estimator. Reuse can be chained: a query-only external matrix does not discard the original source axis for a later external estimate. Calling `from_statistics(external_result.statistics, ...)` replays the original source effects; use `external_result.effects` when exporting the external estimator's ranking inputs. For compact fixtures, the stored source-effect axis can be smaller than the original weight axis, whose size remains recorded separately.

## `from_statistics`

```python
result = pr.from_statistics(
    statistics_mapping,
    queries,
    top_k=25,
    seed=2026,
    effect_gene_subset=False,
    weight_gene_count=None,
)
```

Replay reliability and ranking from an archived input-statistics mapping. Let `T` be the number of targets and `G` the number of genes on the full original weight axis.

| Key | Shape | Meaning |
|---|---|---|
| `gene_ids` | `(G,)` | Unique gene identifiers |
| `target_ids` | `(T,)` | Unique perturbation identifiers |
| `target_gene_ids` | `(T,)` | Perturbed-gene identifiers in the same gene namespace |
| `effects` | `(T, G)` | Original-coordinate target-minus-control effects |
| `energy` | `(T,)` | Sum of squared effects across the full gene axis |
| `noise_trace` | `(T,)` | Total variance contribution to the estimated effect |
| `direction_energy` | `(T, 5)` | Squared effect projections on the five fitted control directions |
| `direction_noise` | `(T, 5)` | Variance contributions on those directions |
| `n_cells` | `(T,)`, optional | Recorded integer target-cell counts, each at least two |
| `batch_ids` | `(T,)`, optional | Recorded matching-stratum identifiers |
| `control_models` | mapping, optional | Archived control model metadata |

By default, these arrays retain the full original gene axis, and `energy` is checked against the sum of squared `effects`. A compact fixture may instead save only query-gene effect columns while preserving the previously computed full-profile energy, noise and projected statistics. Declare that representation explicitly with `effect_gene_subset=True` and supply the original `weight_gene_count`. The implementation records the original weight-axis size separately from the retained effect-axis size and checks that the subset's squared-effect energy does not exceed the archived full-profile energy. It never reconstructs full-profile reliability statistics from the query subset.

Reliability weights are recomputed from those sufficient statistics rather than taken from a saved weight array. Tutorial fixture provenance identifies where the statistics were originally computed.

This entry point supports compact real-data demonstrations and numerical regression. It does not reconstruct missing cells or verify upstream normalization from statistics alone.

## `evaluate_reference`

```python
scores = pr.evaluate_reference(result, reference_effects)
```

Evaluate a `PertRankResult` against a separate, labeled target-by-gene reference effect DataFrame. The reference must contain every fitted target exactly once and all requested query genes; additional reference genes are allowed. Values must be finite, identifiers unique, and response signs compatible with the ranking input. Rows are matched by identifier.

The operation holds the result's query set, top-k, target pool and own-target exclusion fixed. It returns one row per method, query and direction, with `precision_at_k`, `expected_overlap`, `top_k` and `n_candidates`.

For each eligible candidate `j`, let `p_j` and `q_j` denote expected inclusion in the predicted and reference top-k lists under uniform boundary-tie selection. The score is

```text
expected_overlap = sum_j p_j q_j
precision_at_k   = expected_overlap / top_k
```

This expectation uses independent tie selection in the two lists. It does not compare one arbitrary deterministic realization of each list. A fully tied reference remains uninformative even if a deterministic sort would return the same candidate names in both lists.

The evaluator does not fit weights or alter nominations. Reference independence and its experimental interpretation must be established by the study design, rather than inferred from a DataFrame. No confidence intervals, source pooling or biological accuracy probabilities are produced. The method names identify weighting rules: for a result returned by `rank_effects`, `Unweighted` means the supplied external estimator without weights.

## Advanced array interface

The original low-level functions remain available for compatibility with the frozen research implementation. The AnnData interface above is the recommended entry point for a new screen.

| Function | Inputs and return |
|---|---|
| `fit_cells(X, gene_ids, target_ids, batch_ids, is_control, target_gene_ids)` | Finite dense cell-by-gene normalized expression with aligned cell metadata; returns an effects/statistics/control-model dictionary and fixed-method weights |
| `normalize_counts(counts, library_totals, batch_ids, is_control)` | Nonnegative counts, externally supplied positive full-library totals and aligned normalization strata; returns `(log_expression, scales_by_stratum)` using median matched-control totals |
| `reliability_weights(energy, noise_trace, direction_energy, direction_noise, beta=0.9, k=5)` | Nonnegative target energies/noise and target-by-direction statistics; returns the three method weight arrays. Optional algebraic parameters support numerical verification; the high-level fitting interface fixes the manuscript values |
| `nominate(result, query_ids, top_k=25, seed=2026)` | A low-level fit dictionary and query IDs; returns `(records, inclusion_probabilities)`. Probability-array axes are method, query, direction and target |

The array normalization function can accept a selected gene axis when original full-library totals are supplied; the AnnData raw-count path deliberately requires the complete measured axis and computes those totals itself. The low-level functions do not infer cell roles, choose queries or access an evaluation reference.

## Scores and ties

For target `t` and query `g`, the up score is `w[t] * effect[t, g]`; the down score is its negative. The Unweighted method sets all weights to one. Euclidean and PertRank use the same positive-part reliability rule with their respective energies and noise estimates. PertRank attenuates the fixed control directions in the reliability calculation while preserving the supplied gene-coordinate effect for ranking.

For a boundary tie of size `m` with `r` remaining slots, each tied candidate has expected inclusion `r / m`. The deterministic delivered list breaks that tie reproducibly. Neither number is a biological validation probability. A query's own gene perturbation is excluded consistently from every method.

## Offline tutorial data

`from pertrank.datasets import load_example, software_example`

- `load_example("replogle")` returns `(statistics, provenance)` for the complete
  archived 2,000-target task; 300 original-coordinate query columns are kept.
- `load_example("tcell")` returns the complete archived 52-target task with
  293 query columns and the saved FR effect matrix.
- `software_example(seed=2026)` creates a 280-cell, 60-gene synthetic AnnData
  for interface checks only.

Both real fixtures verify their own content hash and work without network
access. Their full geometry axes contain 8,202 and 13,390 genes respectively.
Use the explicit `effect_gene_subset=True` route with those counts.
