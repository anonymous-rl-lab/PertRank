# IL2 external protein-support experiment: fixed analysis plan

Version 1.2.0. Scope: the previously audited Schmidt 2022 CRISPRi IL2 FACS dataset and the existing Stim PertRank versus Euclidean lists. This is a retrospective external analysis, not an OSF-preregistered experiment. The source workbook was already downloaded and its metadata and direction-blind count totals audited. This plan is recorded before this workflow computes external high/low contrasts or method results.

## Fixed inputs and mapping

- Retain all ten existing comparisons: five input groups × increasing/decreasing IL2, K=25. Do not change lists, genes, parameters, or choose a favorable direction/group.
- Carry forward the v1.1.0 candidate mapping. Do not combine different external design labels. Analyze all available genome-wide external targeting guides, not only the changed targets, for statistical hit estimation.
- Primary external targeting guides exclude any contiguous 19-nt match in either orientation to the complete source atlas. Require positive high+low counts in each donor. Retain original external NTC measurements, explicitly acknowledging that their sequences are shared.
- All-guide comparison is a predefined sensitivity analysis using the same count-presence rule and estimator, not an alternative primary result.
- No new arbitrary minimum-count threshold; report read depth and single-guide cases.

## Primary external effect and list endpoint

For each guide, library set, and donor, calculate `r = log2((high+1)/(low+1))`. Subtract the median r of the retained external NTCs in that same library and donor. The +1 is added to raw counts. This is an explicitly defined NTC-centered effect estimator, not a claim to reproduce published gene-level LFCs exactly.

For each target and donor, take the median across the target's eligible guides. Let `L_g` be the equal mean of the two donor-specific medians. Positive effects indicate increased IL2 after CRISPRi, matching the positive RNA direction without sign reversal.

The primary statistic is the **observed contribution to mean direction-correct external LFC gain**:

`D_obs = sum_j sum_g observed(g) * (membership_PR_jg - membership_Euc_jg) * direction_j * L_g / 250`.

The denominator is fixed at ten lists ×25, even when a target is missing. Repeated occurrences of a target share one external effect. Report donor-specific, direction-specific, and per-list values. Do not call this P@25 or divide by the number of swaps. Compute full-list mean effects only with observed contributions and report missingness explicitly. ABHD18's unknown continuous contribution is unbounded without an effect bound, so D_obs is not an assumption-free estimate of the complete-list effect gain.

## Prespecified secondary protein-support counts

1. **Two-donor effect support:** a target supports a direction when each donor's oriented NTC-centered median LFC is at least 0.5. This is an effect-size/consistency rule, not an FDR-controlled hit definition. Report counts for swap-in, swap-out, common, and full lists.
2. **MAGeCK support:** use exact MAGeCK 0.5.9.2 if the locally installed version passes its demo smoke test. Normalize counts within each A/B library and high/low sample to that library's mean total across the four high/low donor samples, using totals from the full original guide catalog. Concatenate libraries and run paired donor high versus low, `--norm-method none`, external NTC controls, median gene LFC, and the documented default remaining settings. Count direction-matched `pos|fdr <0.05` and `pos|lfc >0.5` for increasing lists, or `neg|fdr <0.05` and `neg|lfc <-0.5` for decreasing lists. Report all guide exclusions and the testing universe. This adapts the source analysis with an explicitly fixed count scale; it is not an exact reproduction of published FDRs.

If the published tool cannot execute in this environment, preserve the failure and complete the primary and effect-consistency analyses; do not substitute a custom significance method under the MAGeCK label. No result-dependent endpoint changes are allowed.

For either binary endpoint, use net supported swaps divided by250 and supported members per list divided by10. ABHD18 is absent and occurs once as swap-in, so its unknown contribution lies in `[0,1/250]`; do not call it an observed negative. If restricting to at least two eligible guides, the additional SSBP2 swap-in and NAIF1 swap-out yield an unknown contribution in `[-1/250,2/250]`.

## Uncertainty and robustness

- External biological replication is two donors, not ten lists. No donor-population confidence interval or claim of ten independent experiments.
- Report a conditional paired-guide bootstrap for D_obs: 20,000 draws, seed20261011. Resample guide identities with replacement within target×library; carry both donors together. Independently resample NTC identities within library, likewise paired across donors, recompute shared centers, and propagate the same center draw to every target. Use percentile2.5/97.5 intervals. Single-guide strata have no guide-resampling variance. These intervals describe reagent/control sensitivity within these two donors and fixed lists; they are not biological-population coverage guarantees.
- Carry the same guide-bootstrap draws into the fixed two-donor effect-support count endpoint. Do not refit MAGeCK20,000times; its counts are a descriptive fixed-score endpoint here.
- Report removal of each unique changed target including every repeated occurrence, retaining denominator250. Record the range and largest absolute contribution. This diagnoses dominance rather than selecting an improved subset.
- Report the predefined at-least-two-guide observed-component sensitivity and the all-guide sensitivity. Neither replaces the primary result.

## Interpretation and execution

Positive values favor PertRank; negative values favor Euclidean. A positive primary estimate with positive conditional lower bound and positive donor-specific estimates supports a within-case protein-effect improvement, subject to missingness and the listed dependence. Otherwise report mixed, unresolved, or negative evidence as observed. Secondary counts answer a different, threshold-dependent question and must be shown even if they disagree with the primary.

The source atlas used this historical screen during guide selection; controls and the laboratory network are shared and cross-study donor disjointness is unverified. Non-overlapping targeting sequences and a separate protein assay improve evidence separation but do not make this a fully independent prospective replication.

Local CPU-only execution; no model training, epochs, GPU rental, or new data acquisition. Run extractor/check smoke tests, primary scoring and conditional resampling, the fixed MAGeCK secondary analysis, independent numerical verification, then produce a versioned English Markdown report and a compact exact replay package. Preserve intermediate results and logs. No additional experiment is made conditional on obtaining a positive result.
