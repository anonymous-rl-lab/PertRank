# Conditional neural ranking data

Complete original axes are preserved. Each `*_neural_inputs.npz` contains finite float64 `X[query,target,13]`. No normalizer or model was fitted.

| Source | Queries | Targets | Features | Half 0 / 1 | Single-guide targets |
|---|---:|---:|---:|---:|---:|
| D3 | 293 | 52 | 13 | 26 / 26 | 10 |
| D1 | 289 | 59 | 13 | 32 / 27 | 9 |
| D2 | 289 | 63 | 13 | 35 / 28 | 17 |

## Features

- `raw_signed` (effect): Original signed source effect: original20 mean natural-log normalized RNA minus observed archived control mean.
- `pertrank_score` (effect): Original stored PertRank score; exactly raw*wM.
- `euclidean_score` (effect): Original stored Euclidean score; exactly raw*w0.
- `query_se_input` (effect): sqrt(original20 unbiased query sample variance/20), conditional on observed control mean; excludes control uncertainty.
- `wM` (dimensionless): Original fixed PertRank weight.
- `w0` (dimensionless): Original fixed Euclidean weight.
- `n_cells` (count): Original target-cell count, exactly 20 throughout.
- `profile_energy_I` (energy): Squared original20 mean full-profile norm / G; source squared natural-log units.
- `profile_energy_M` (energy): Original20 mean profile energy under fixed M=I-beta*U*U.T / G.
- `control_projection_fraction` (dimensionless): (energy_I-energy_M)/(beta*energy_I); fraction of mean-profile energy in the fixed control subspace.
- `guide_abs_contrast` (effect): Absolute same20 guide mean query difference; missing single-guide contrast gets deterministic zero fill.
- `guide_contrast_available` (indicator): One iff two guides occur in the original20; distinguishes zero fill from measured agreement.
- `guide_min_fraction` (dimensionless): Minor observed guide count divided by original20 count; zero for single-guide targets.

Features use original20 source cells and fixed original control geometry only. Cell and guide identities match both the shallow archive and source role-0 metadata. Same20 query variance, guide contrasts, and full-profile energies were recomputed exactly. No retained/deep expression, planning input, external effect, target ID, or query ID enters X. Geometry and guide statistics reuse these same20 cells and are not independent prior estimates.

## Reference and joint donor handling

`base`, `raw`, `euclidean`, and `reference` match the old paired arrays bitwise. `reference_replicates[Q,T,3].mean(2)` is also bitwise identical to the old reference. Its scale is **unstandardized log2(CPM+1) KO minus batch/donor-matched control mean**, averaged over three donors. No robust scaling or new reference definition was applied.

Batch 1 donor labels are `[1,3,4]`; batch 2 labels are `[1,2,3]`. Tensor positions are not common donor identities across batches. `reference_target_batch`, `reference_donor_labels`, `reference_control_group`, and group batch/donor mappings are included. Donor sensitivity must use shared batch-donor multipliers across candidates in a ranking list, normalized within batch. Candidate-independent donor resampling would destroy the observed joint structure. Same-number person identity across batches was not established; sharing such weights is a separate sensitivity assumption. One KO well per donor does not separate assay noise from biological/well differences.

## Split and checks

`targethalf` exactly preserves the original global stable-ID hash split (seed 2026111504): 75 targets, global halves 38/37, and one half per target across sources. `ownmask` is exactly query-ID/target-ID equality; `query_block` preserves the old blocks. Reference values and donor metadata are exactly consistent for shared source IDs. Saved arrays were reloaded and checked bitwise.

The assay was used in prior work: held targets provide algorithmic separation, not a historically untouched experiment. Predictors and labels are separate arrays, but the runner must enforce target access and fit scalers within training folds only. No labels or Top-K probabilities were constructed during preparation.

`DATA_PROVENANCE.json` records source paths, SHA256 hashes, sizes, exact identity checks, and roundoff bounds. Rebuild with `python prepare_data.py --workspace /path/to/workspace`.
