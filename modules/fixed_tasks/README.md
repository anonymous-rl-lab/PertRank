# Fixed-task replay component

Scientific implementation retained from v1.4.0; review packaging updated in v1.5.2. Run `python reproduce.py --output-dir NEW` from this component directory, or use the total-package root entry point. Packaging backups and a historical HTML display report are retained separately. Current numerical inputs, analysis definitions and result identities remain in this component. Registration aliases are explained in `../../docs/REVIEW_IDENTITY.md`.

This repository contains a compact, offline reproduction of the completed **four-donor resting T-cell experiments and the Stim experiment**: experimental sufficient data, executable analysis code, original result records and English scientific analysis. It preserves the fixed methods and their exact top-25 ranking information, rather than storing only final scores.

**Start with [the four-donor analysis](analysis/FOUR_DONOR_RESULTS.md).** The full held-out D4 report is [here](analysis/D4_Results_and_Analysis.md).

The completed D4 TRADE scalar comparison and D3 NTC-guide diagnostic are available in [supplements/20261008](supplements/20261008/README.md). The later independent-RNA/interaction and external-baseline components are in `supplements/20261014_upgrade` and `supplements/20261014_external_benchmark`. The numerical arrays and analysis definitions retain their original identities; current package metadata and project-owned registration identifiers have been maintained for review.

**Version 1.2 adds the completed Stim experiment** in [supplements/20261011_stim](supplements/20261011_stim/README.md). The standalone Stim ZIP is below **10,000,000 bytes** and contains all inputs needed for its fixed-method numerical replay. It no longer requires the separate 377 MB prediction archive. The size limit applies to the Stim addition, not this complete D1–D4 total package. The original four-donor data and the version 1.1 supplement remain unchanged.

## Quick start

Tested with Python 3.12, NumPy 2.3.5 and SciPy 1.17.0. After extracting the repository:

```bash
python -m pip install -r requirements.txt
python reproduce.py --output-dir ../PertRank_reproduced
```

The replay uses no network connection after dependencies are installed. It requires no original workstation paths, OSF token, R installation or large H5AD download. Use a new output directory for each run. The supplied data and official results are not overwritten.

The command verifies file integrity, reconstructs weights, reranks the retained original effects, recomputes scores and paired intervals, replays competitor selection and D4 decisions, and reproduces the measured depth analysis. It writes a machine-readable verification report and regenerated scientific outputs to the chosen directory. Any mismatch stops the run.

The same command also replays the supplemental D4 five-arm comparison and the NTC guide diagnostic, writing their regenerated outputs under `supplements/` in the chosen output directory. It does not start new fits or download cells.

It then replays Stim under `stim/` in the output directory: scalar weights, both main query panels, the cross-guide analysis, per-target alignment contributions, confidence intervals, wild-cluster tests and the recorded decisions. Successful complete replay ends with `FOUR_DONOR_AND_STIM_REPRODUCTION_PASS`. To replay only Stim, run:

```bash
python supplements/20261011_stim/replay.py --output-dir ../Stim_reproduced
```

## Experiments included

| Donor | Current primary comparison | Candidate targets | Input groups | Role |
| --- | --- | ---: | --- | --- |
| D3 | Seven fixed arms | 1,200 | Five disjoint groups of 20 cells | Opened development data |
| D1 | Seven fixed arms; complete corrected query reselection | 1,127 | Five disjoint groups of 20 cells | Updated analysis of previously opened donor data |
| D2 | Seven fixed arms | 1,200 | Five disjoint groups of 20 cells | Updated analysis of previously opened donor data |
| D4 | Four locked arms, including the preselected competitor | 589 | Five disjoint groups of 20 cells | Prospectively locked, same-study held-out donor confirmation |
| Stim | Four evaluable fixed arms; TRADE not evaluable | 561 main; 26 cross-guide | Five disjoint groups of 20 cells | Completed second-state experiment; actual procedural status retained in its report |

The seven development arms are raw ranking, Euclidean weighting, PertRank, diagonal weighting, the TRADE-core scalar adaptation, PR_E and PR_M. D4 contains raw, Euclidean, PertRank and PR_M. **PR_M combines a Pearson-residual control metric with scalar weighting of the original log-effect output.** It is not Pearson normalization alone.

The added D4 TRADE scalar arm is a **post-hoc supplemental comparison** using the same five locked input groups. It does not replace PR_M, rerun competitor selection with a new arm, or alter the original confirmation decision. The added NTC analysis uses a newly metadata-frozen recovery selection in D3; it is not a claim that the interrupted historical NTC selection was recovered exactly.

Stim preserves its actual outcome: H1 (PertRank versus Euclidean) and H2a (versus PR_M) pass; the improvement threshold of 0.02 is not reached. TRADE remains **not evaluable** after a deterministic fitting failure declared before reference evaluation. Its comparison is not imputed or counted as passed. The Stim source records retain `investigator_directed_deviation` and the unverified independent E4 registration status; this compaction does not change that history.

Both the responsive and expression-tercile panels are included, with both ranking directions, self-target exclusion and exact tie-expectation scoring. The main ranking budget is **20 target cells per group**, with additional planning and control resources. Five groups are repeated inputs within a donor, not five independent donors.

The depth ladder contains measured raw inputs at 20, 25, 30, 35, 40, 45 and 50 cells. **D1 depth results retain the historical query panel and are explicitly labeled as sensitivity results.** They must not be merged with the corrected D1 primary-panel results. No corrected-panel D1 depth curve has been invented or inferred.

## What is recomputed

1. **Weights.** Euclidean, PertRank, diagonal and Pearson scalar weights are recalculated from retained energy/noise statistics. D4 also checks projected noise moments and cell-partition disjointness. For TRADE, the original R-serialized fitted weights are retained and the scalar formula is checked against fitted latent moments; the mixture is not refitted.
2. **Ranking.** Retained original float64 effects are multiplied by the verified frozen weights and ranked again. All candidates tied at any relevant boundary are retained. Full candidate identities and counts preserve self exclusions and random expectations.
3. **Evaluation.** Every panel, arm, input group, sign, pairwise contrast and within-query group-average interval is recalculated and checked against original outputs. Paired query-block bootstraps retain the original seeds and operation order.
4. **Selection and confirmation.** The development rule selecting PR_M is replayed with corrected D1; the D4 primary, improvement and gated competition decisions are recalculated.
5. **Depth equivalence.** Exact signed top-list boundary information regenerates depth scores, paired bootstrap curves and inverse-depth summaries, including unresolved crossings.
6. **Supplemental experiments.** The D4 TRADE scalar weights, five-arm rankings, 30,000 score values and paired summaries are checked. The NTC diagnostic is recalculated from a float64 cross-Gram matrix and target-pair energies, including all 2,000 bootstrap and permutation replicates. See the supplement for the extraction checks, retained fitting records and limits of these sufficient data.
7. **Stim experiment.** Reconstruct fixed scalar weights from retained energy/noise moments, rerank original float64 effects with all boundary ties, and recalculate both main panels, cross-guide scores and the full-axis per-target alignment contributions. Unchanged frozen statistical functions regenerate H1, H2a, the unavailable H2b record and S1–S3, including their intervals and tests. The module checks the regenerated results against the completed experiment.

## Why the data fit in a small package

For each input group and query, the package retains the union of candidates appearing in either signed top-25 list under any frozen arm, including **every boundary tie**, plus the reference tails. Their original float64 effects are retained without quantization. Omitted candidates cannot enter any of the specified frozen lists. Thus multiplying by the same reconstructed weights and reranking this support gives exactly the same top-list membership probabilities.

The compaction was checked against the full original matrices: **165,679,914 candidate membership probabilities matched exactly**, as did all archived per-query/per-sign scores for the four donors. The extraction audit is in `audit/RANKING_COMPACTION_AUDIT.json`. Depth arrays use exact boundary/tie encodings instead of storing redundant expression magnitudes.

This is a **post-result, fixed-method sufficient-data archive**. It supports precise reproduction of these experiments; it is not a general expression-matrix release. A changed method, weight, query set, candidate pool or list size requires the original complete data. The omission of interior values is a storage choice made after evaluation and was not used to select scientific outcomes.

## Repository map

| Path | Contents |
| --- | --- |
| `reproduce.py` | One-command offline reproduction and manifest verification |
| `src/` | Weight, ranking, paired-bootstrap and depth replay |
| `src/vendor/` | Retained scoring functions and source provenance |
| `data/D1/` through `data/D4/` | Sufficient numerical data, axes, assignments and compaction records |
| `reference_results/` | Unmodified original result JSON and numerical outputs |
| `analysis/` | English reports and source-value tables |
| `provenance/` | Original code, locks, correction records and source hashes |
| `audit/` | Construction equivalence and delivery verification records |
| `tools/` | Compaction builders for use with the complete source archive |
| `supplements/20261008/` | D4 TRADE and D3 NTC sufficient data, source code, reference results and replay |
| `supplements/20261011_stim/` | Self-contained Stim sufficient data, frozen statistical source, results, provenance and exact replay |
| `audit/STIM_INTEGRATION.json` | Base-package identity and preservation check for the Stim addition |
| `MANIFEST_SHA256.json` | Integrity inventory for this public package |

## Reproduction scope

The package starts from frozen sufficient statistics and retained effects. It does not reread raw UMI matrices, refit the control PCA, reconstruct all full-gene planning data, or refit the TRADE mixture. Those acquisition/fitting stages belong to the complete source archive. Hashes identify that provenance; a retained historical hash does not mean an omitted original file has been reconstructed.

The interval estimates are conditional on the realized inputs, controls, queries, candidates and reference. D4 adds an unseen donor within the same study, not a new laboratory, guide-separated reference or functional assay. Cell-equivalent depth expresses ranking performance given existing planning/control resources; it is not a measured reduction in total experimental cost.

This package covers the current four-donor competition, real-input and depth experiments. It does not claim to reproduce every historical analysis in the entire PertRank paper, such as the separate external knockout, protein or guide-separation experiments.

The included Stim cross-guide endpoint is distinct from those historical guide-separation experiments. Its 26-target candidate pool is close to the list size of 25; its unresolved contrast must retain that limitation. Stim alignment replay starts from the frozen pre-reference indices and regression design, rather than refitting full-gene planning statistics or the control PCA. The compact data preserve fixed-method numerical results exactly; they do not reconstruct every value in the original full expression matrices.

The NTC result detects reproducible guide-associated expression structure. It does not identify its cause, adjust away lane effects, or estimate a fraction of the primary ranking gain attributable to artifacts. The small supplement omits raw counts, full per-guide expression matrices and full-gene D4 fitting inputs; those require the complete evidence archive. Its fixed-method replay must not be presented as an end-to-end raw-data reconstruction.

## Source data and archiving

The original public donor objects follow the recorded source pattern:

```text
https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/{DONOR}_Rest.assigned_guide.h5ad
```

Here `{DONOR}` is D1, D2, D3 or D4. Original source identities, extraction paths and hashes are preserved in `provenance/` and the per-donor records.

Stim source identities and historical archive hashes are retained separately in its module. Its OSF registration and associated project are STIM_PRIMARY_REGISTRATION and STIM_ASSOCIATED_PROJECT. Access tokens are not needed for compact replay.

Original OSF registration: REST_PRIMARY_REGISTRATION  
Original OSF Project: REST_ASSOCIATED_PROJECT

The D4 report discloses the actual project-based addendum and durable prediction-archive procedure. This GitHub package is a results deposit, not a new registration or a replacement for the historical prediction lock. No private view-only token is included.

## GitHub upload

Extract the ZIP and upload the **contents of this repository folder**, rather than nesting another copy of the folder. All individual files are below 25 MB, and the complete delivery ZIP is below 60 MB. Git LFS is not required for these files. Preserve the directory structure, including `data/` and `reference_results/`; the verifier checks their integrity.

See [rights and third-party provenance](RIGHTS_AND_THIRD_PARTY.md) before applying a repository-wide software or data license. No license is fabricated by this packaging step.
