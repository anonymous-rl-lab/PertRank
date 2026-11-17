# SN4 diagnostic definitions and target-table verification

This module supports the post hoc SN4.2–SN4.4 diagnostics in D3, D1 and D2. It reaggregates retained target-level measurements using the definitions below.

## Run

Requirements: Python 3, NumPy and SciPy.

```sh
python verify_sn4.py
```

The script verifies the compressed original target tables against their uncompressed SHA-256 hashes, rebuilds the diagnostic bins, recomputes the A9 Spearman correlations and support rates, and reproduces the A10 group ratios and 95% percentile intervals with the original 2,000 target resamples and seed 2026. Results are written to `verification_results.json`. All 3,527 target identities align across the two tables. The recovered A10 script hash matches its recorded result hash.

## Definitions

- Within each donor, bins use targets with `w0y > 0` or `W1 > 0`, stable ascending ordering of `d = p_n - p_a`, and `numpy.array_split(..., 5)`.
- Both-zero targets are excluded from these diagnostic bins, not from candidate ranking. All-target A10 summaries include every target.
- SN4.3's nonmonotonic quantity is the own-reference support rate among added nominations: supported expected additions divided by all expected additions within a bin.
- A10 uses the first five input-control directions, disjoint blocks of 20 and 30 target cells, and the shared input-control mean. Its energy and noise quantities are normalized by the number of weight genes.
- Group replication ratios divide group sums. They do not average individual-target ratios. The direction-1 contribution is its share of the reduction in the sum of `A0 - C`.

## Files and scope

The original analysis scripts are retained as source evidence. They require the original full cell-state inputs and are not the portable entry point. `verify_sn4.py` uses the retained target-level intermediate tables; it does not reconstruct raw-cell projections, refit weights or rerun candidate rankings. The A10 archived summary preserves the original metadata, checks, donor summaries and decision reading, with duplicated per-target records supplied separately in the compressed CSV. `SOURCE_MANIFEST.json` records original-byte hashes.

