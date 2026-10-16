# v20 bounded S3 explanation and query-block sensitivity

Status: post hoc analysis of already opened D4 Stim8hr results. This protocol is frozen before computing the new decomposition. It does not alter the registered S3 analysis, H1/H2, fitted weights, candidates, parameters, queries, cell roles, reference or original decisions.

## Inputs and scope

Use the existing exact compact main-task support, query context and pre-reference alignment/design arrays. Reconstruct the original full-candidate top-25 membership probabilities for all four available arms, all five input groups, both signs and both panels solely to verify the archived hashes. Analyze the 300 responsive queries. No raw-cell or remote data read, new fit, changed beta, new list length or reordered candidate pool is permitted.

The saved per-target planning sums may be read to reconstruct planning query correlations and only the prespecified 20/30/40 complete-linkage query-block partitions. Retain only the 300-by-561 responsive planning-effect matrix needed for replay. Reconstructed 30-block labels must match the archived partition (up to label permutation); otherwise do not perform alternative block sensitivity.

## Exact gain/loss decomposition

Use the originally frozen normalized alignment ranks. Eligible targets with rank_A <= 1/3 form the low group, 1/3 < rank_A <= 2/3 the middle group, and rank_A > 2/3 the high group. All 22 originally ineligible targets form a separate unclassified group. Ties are not split. No group-specific reranking occurs. Own-target exclusions and full-pool tie probabilities remain original.

For each query, group and sign, let delta_p = p_PertRank - p_Euclidean and t = p_reference. Sum over each alignment group: positive supported additions = sum(max(delta_p,0)*t); supported losses = sum(max(-delta_p,0)*t); net = additions - losses. Average over five groups and two signs, then equally over all 300 queries. Units are expected reference members per original 25-slot list. Also report positive/negative changed-slot mass and support per changed slot, leaving zero denominators undefined. All four group nets must sum to 25 times the original primary P@25 difference to floating-point precision. Report group counts and frozen signal-to-noise rank distributions.

## Reference-neutral diagnostic

Keep the original 539-target query-specific S3 design masks and covariates, including original self-target exclusion. Replace only reference membership by the mean under a uniform random K-list on each query's eligible full pool: K/N_q. Thus neutral target contribution in P@25 units is delta_p/N_q, averaged over groups and signs. Fit the original query-specific OLS to observed and neutral target contributions, then calculate observed minus neutral rank_A coefficients. Report all three mean coefficients and paired query-block ranges. The neutral model is an analytical reference-agnostic benchmark, not a biological null or a new formal S3 test. The sign of observed-minus-neutral is not automatically good or bad; interpret with actual gains and losses.

## Weight transitions

At each of 2,805 target-by-input combinations, assign exactly one category from original unrounded weights: both zero; Euclidean zero/PertRank positive; Euclidean positive/PertRank zero; both positive and PertRank lower; both positive and equal; both positive and PertRank higher. Zero/equality comparisons are exact on the frozen float64 weights. Use the same supported gain/loss/net and changed-slot accounting by transition, plus alignment-by-transition accounting. These categories partition attribution of changed list memberships; they are not isolated causal interventions. No category-specific reranking.

## Uncertainty and bounded sensitivity

All new intervals are nominal conditional query-block resampling ranges over the represented query modules; they do not estimate donor, control, index-estimation or full experiment uncertainty. Use 2,000 paired block draws, seed 2026, retaining original equal-query weighting and all repeated block multiplicities. The same draws apply to observed/neutral coefficients, alignment groups and transitions. No multiplicity-adjusted new success labels will be assigned.

Report leave-one-original-block-out point estimates for the main PertRank-minus-Euclidean gain, the three alignment coefficient summaries and group net contributions. If the archived planning sums reconstruct the original blocks, also report 20- and 40-block ranges using the same complete-linkage correlation construction; the 30-block primary rules remain unchanged. No other block counts, thresholds, bins, simulations or responsive-query subpanels will be searched.

## Deliverables and verification

Provide a standalone offline script, bundled compact sufficient inputs, input/code/protocol hashes, exact source-membership digest checks, preservation of the original S3 coefficient and primary difference, algebraic decomposition checks, query-level and summary CSV/JSON, and concise Methods/Results text. Keep source archives and manuscript/SI files unchanged. Simulated or regression-implied tercile contributions from external discussion are not treated as observed results.
