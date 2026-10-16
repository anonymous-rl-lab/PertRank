# Input contracts

## Experimental roles

Supply input perturbation cells and input non-targeting controls. Select queries independently of any evaluation reference. Keep planning, reference and validation cells outside the supplied object. PertRank uses all supplied input cells and performs no experimental role assignment.

`obs_names` and `var_names` must be unique. The observation columns identify the perturbation, its perturbed gene, the Boolean control flag and the prespecified control-matching stratum. Perturbed-gene and query identifiers must use the same namespace as `var_names`. Perturbation labels may distinguish guides or other experimental units; choose their meaning before fitting.

## Raw counts

Set `input_scale="counts"` and `full_gene_axis=True`. Counts must be nonnegative integers with positive cell library totals. Use `layer` to select a count layer, or `None` for `X`.

The gene axis must contain every gene contributing to the measured library total, including genes absent from controls. The adapter computes totals first, then retains genes detected in input controls. It scales each cell to the median input-control library total and applies `log1p`. It uses the archived count adapter's numerical convention: control log values are represented at float32 precision before float64 moment calculations, while target log values retain float64 precision.

`full_gene_axis=True` is a user assertion, not a way to recover omitted counts. An HVG-only object is not a valid raw-count input. Query genes must survive the control-detection rule.

If original full-library totals have been retained in an observation column, supply its name as `library_totals_key`. The count adapter checks exact equality with the matrix row sums. This detects a gene-subset mismatch when trustworthy original totals are available; it does not replace the full-axis requirement.

The count path supports a single prespecified matching stratum per call. It reads cells in batches and processes targets separately. Dense and sparse matrices may be supplied in memory or through a supported backed `.h5ad`. The control matrix and the smaller control/gene Gram matrix must fit in memory. `batch_rows` changes the reading chunk size, not the scientific model.

## Already normalized expression

Set `input_scale="log"` for the already normalized gene coordinate system used for estimating effects and covariance. This path uses the supplied values directly: it neither log-transforms nor rescales them. Document the upstream normalization, the full library totals used during that normalization, and the chosen weight-gene axis with the input data.

Avoid double normalization. A matrix of residuals, z scores or latent coordinates is a different representation from the declared log-expression scale and should not be labeled as log expression merely to pass input validation.

The normalized path supports explicitly declared multiple matching strata, with each target belonging to one stratum. Target cells are compared with their own stratum's controls. Pooling a target across strata requires a separately specified aggregation design.

## Matched controls and batch structure

Use `strata_keys` to encode the actual experimental matching variables, such as donor, state and technical batch. A stratum needs controls from the same matching context. Particularly when each target occurs in only one batch, mixing controls across batches can make batch differences look like perturbation effects.

An empty list is an explicit declaration that one common control population is appropriate. PertRank validates the supplied structure; it cannot infer whether a biological or technical matching variable was omitted.

## Reliability and readout

Five resolved control covariance directions and attenuation 0.9 are fixed. Each target's own sample variance contributes to both total and projected noise estimates. Control variance contributes through the matched control mean. The reliability calculation uses full-profile information; the ranked effect remains a gene-specific readout.

Each stratum requires at least six controls and five resolved positive covariance directions. Each target requires at least two cells to calculate sample variance. These are algebraic requirements rather than recommendations about experimental power or depth.

The supplied candidate population determines the ranking. Choose its eligibility rules before examining a reference. The queried gene's own perturbation is excluded by gene identifier. Both up and down lists are returned. Boundary ties use exact expected inclusion, while a deterministic hash rule supplies one reproducible delivered shortlist.

## External effects and archived statistics

`result.rank_effects(...)` applies the fitted target weights to an external target-by-gene effect DataFrame. Labels are required to prevent silent positional alignment. The effect orientation must preserve the meaning of up and down. The external estimator should use a compatible perturbation population and experimental input design. The returned `effects` contains that external readout, while `statistics` preserves an independent copy of the original source reliability inputs; the external estimator's uncertainty is not re-estimated by this operation.

`from_statistics(...)` reconstructs the same reliability calculation from archived energies, variances and effects. It is useful for compact tutorial fixtures and numerical replay. Some compact fixtures retain query-gene effect columns alongside full-profile energy and noise statistics. That representation requires `effect_gene_subset=True` and the original `weight_gene_count`; it does not change the full-profile information used in the archived reliability statistics. A statistics fixture is not a raw-cell dataset and cannot demonstrate recovery of the original normalization or guide-assignment steps.

## Verified dependency combination

This review release bounds SciPy to `<1.17`: AnnData 0.12.3's backed CSR
reader depends on an internal SciPy index helper removed in SciPy 1.17.
The verified dependency versions are recorded in `requirements-tested.txt`,
with the checks actually performed for this release in
[the validation record](validation.md). The CI recipe declares Python 3.10
and 3.12; declared CI targets are separate from completed local checks.
