# Complete D1 responsive-panel correction before D4

This is an authorized deterministic correction on already opened D1, not new confirmation. D4 is not read by this work.

## Frozen design

The original D1 cohort, metadata quality filters, 1,127 target order, SHA256 cell permutation, original input and evaluation controls, normalization scale, 12,794 weight genes, and five real input groups at positions 100–199 remain fixed. All seven method weights are reused without fitting or modification from the completed v17 competition.

Reconstruct full-gene sums from the original planning cells at positions 0–19, 20–49 and 50–99, and the original deep-reference cells at 200–299. These four blocks contain 20, 30, 50 and 100 cells. Accumulate float64 log1p(count × original scale / total UMI), exactly as the historical D1 target reader. Keep the blocks separate so the historical arithmetic and per-block stored query sums can be checked.

Original response-panel rule: `POOL=((S0+S1+S2)/100 - cm).T`; set each target's own Ensembl gene entry in a copy of POOL to zero; select `np.argsort(-Zp.var(axis=1))[:300]`. The corrected ID mapping is the existing independently audited `d1_own_gene_map.npz`. The old code incorrectly compared target symbols with Ensembl gene IDs. No manual five-gene replacement or reduced candidate query search is allowed. The full keep-gene axis is searched.

The tercile panel is rechecked and retained. Both response-panel and tercile query blocks use the original unzeroed planning profile and the archived `make_blocks`: correlation distance, complete linkage, maximum 30 clusters. The corrected response panel is selected without reading or using method scores. No parameter changes follow its result.

## Reading checks and gate

First read ten targets selected by the historical `SHA256('audit|'+target)` ordering. Verify all old stored query sums in all four blocks, cell permutation identity, exact current input groups, input/evaluation NTC identity and their disjointness. Numerical tolerance for reordered float64 summation is maximum absolute 1e-10 and relative (denominator floor 1e-3) 1e-10; report both. Source ETag, raw per-cell UMI totals, gene IDs and disjoint cell sets are exact checks. Full read begins only after this audit passes. Bounded range transport uses at most eight network workers and deletes temporary payloads after aggregation; compact per-batch gene sums are checkpointed.

## Scoring and strongest competitor

Before corrected scoring, lock the selected panel and its blocks, reconstructed truth and five-group original-coordinate predictions, and reused method weights. Reproduce the uncorrected panel with the newly read planning values. Verify old-query sums and scores against the existing archives before scoring the corrected panel. A tied gene-variance selection discrepancy is reported explicitly; no convenient gene is substituted.

Use the frozen exact boundary-tie-expectation score, original own-target exclusion, top-25, both directions, five groups averaged within each query, and the original 30-block paired bootstrap (2,000 draws, seed 2026). Reuse v17 scoring code and labels. All seven arms are reported: raw, Euclidean, PertRank, diagonal, TRADE_scalar, PR_E and PR_M. This analysis is a correction sensitivity study. It does not amend past frozen decisions. Root will use corrected D1 and unchanged D3/D2 in the separately frozen final strongest-competitor selection rule before any D4 opening.

## Scope

No D4, stimulation-condition or HEK293T data are read. No scores are used for panel selection. The new D1 query panel can differ from the historical panel; both are retained and clearly labeled.
