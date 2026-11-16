# A19-R bounded recovery and replay protocol

Written before running the recovered scorer on 2026-11-14. This is an audit and feasibility pilot on an already analyzed external reference, not a new blinded confirmation.

## Frozen task

- Original retrieval identities are held separately for editor verification; local content hashes identify every executable input.
- Sources D3_Rest/D1_Rest/D2_Rest; candidate counts 52/59/63, query counts 293/289/289; both query directions; K=10.
- Preserve all original query IDs, target IDs, own-target exclusions, query blocks and scalar weights. No refitting, candidate truncation, query reselection, or parameter tuning.
- Reproduce raw, Euclidean, PertRank scoring from saved effects and the shared independent RNA mean reference. Use exact expected-membership tie handling and the archived scorer (seed 2026100502, 2,000 paired query-block bootstrap draws).
- Before any new readout result, match archived top-10 memberships and mean expected-member counts to absolute tolerance 1e-12.

## New four-arm analysis, conditional on sufficient inputs

The planned arms are w0*y, wM*y, w0*A*y, wM*A*y, with k=5 and beta=0.9 and A=I-(1-sqrt(1-beta))*U*U.T. The primary readout contrast is P10(wM*y)-P10(wM*A*y). Geometry on y and on A*y are reported separately, and their difference (original minus compressed) is secondary. Fixed original candidates, queries and weights are retained. No significance requirement on the interaction is imposed.

The query-coordinate transformation requires either full signed effects and the matching U, or matching signed U.T*y and query rows of U. Scalar energy, squared projections, rankings, or a control basis from another input split are insufficient and will not be substituted. If these inputs are missing, new four-arm scoring is blocked; original three-arm replay is reported only as reproducibility validation.

The three sources share one external experiment. Query-block intervals are conditional descriptive intervals, not donor-level population intervals. Existing reference-draw sensitivity can be rerun if needed but does not supply missing transformed effects. No new reference bootstrap is invented from mean effects. No large raw atlas download or model training is authorized by this pilot.
