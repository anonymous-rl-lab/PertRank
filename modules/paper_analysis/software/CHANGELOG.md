# Changelog

## 0.2.0rc2 — manuscript-aligned private review update

### Scientific identity and documentation

- Align the package overview with manuscript v25.2: control geometry enters target reliability, and gene readout follows the biological query.
- Explain complete weighting versus the added geometry through the three matched returned arms.
- Distinguish reuse with an external effect estimator from transforming readout coordinates.
- Document the separate decision evaluator, including exact expected overlap for independently resolved boundary ties and per-query output.
- Update tutorial explanations while retaining their archived real-data inputs and computational examples.
- Replace inherited release/test assertions with a version-specific validation record; distinguish current manifests from initial-release provenance.
- Keep private distribution, public package publication and ecosystem acceptance explicitly separate.

### Correctness and reusable evaluation

- Keep source reliability statistics distinct from external readout effects; preserve the original gene axis for chained estimator reuse.
- Detach copied statistics and metadata so modifying an external result cannot mutate its source result.
- Record implementation version and effect origin with generated results.
- Validate integer counts and configuration, finite scores, identifier alignment and archived gene-axis provenance consistently.
- Export `evaluate_reference` for fixed-pool P@K with identity-based self exclusion and exact expected boundary-tie overlap.
- Verify actual installed-wheel code and data against the release source; configure CI to build and test wheels.

The test suite increases from 61 to 99 checks. Both real-data tutorial input files and the numerical regression oracle remain unchanged. Verified checks and their scope are recorded in `docs/validation.md`.
