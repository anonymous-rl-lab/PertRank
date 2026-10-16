# Final development competitor selection

This selection is completed before D4 access. It uses no D4 observations, metadata, predictions, or truth. This is a development-stage selection, not a confirmatory result.

The eligible arm identifiers are `Euclidean`, `diagonal`, `TRADE_scalar`, `PR_E`, and `PR_M`. `raw` and `PertRank` are reported but are not eligible. All five eligible arms must be present and valid; none may be silently removed.

For each arm, each responsive query first averages its five actual 20-cell groups and its two query signs. The 300 queries are then averaged within each donor. The final value is the equally weighted mean of D3, D1, and D2, irrespective of candidate counts. The largest final value wins. Arms within an absolute difference of 1e-12 from the maximum form a tie set; the lexicographically smallest frozen arm identifier in that set wins.

D3 and D2 use the complete v17 seven-arm query-score archive. D1 must use the full-axis corrected responsive panel, with original five input groups and all seven weights unchanged. The final selector requires `FINAL_CORRECTION_COMPLETE.json`, hash-verified corrected scores/results, the prior panel prediction lock, and a hash-verified independent audit with status PASS. The original D1 panel may only be used in a dry-run calculation check; it cannot determine the final competitor.

The input score arrays must have axes `(7 arms, 5 groups, 300 responsive queries, 2 signs)`. The selector independently reconstructs each mean from these arrays and checks the corresponding result JSON. It records all source hashes, all donor-specific means, all five group means, the donor-equal means, the tie set and the chosen arm. The output field `selected_competitor` is the canonical value consumed by D4 protocol/code and prediction locks.

No confidence interval, significance decision, external endpoint, tercile result or D4 result affects this selection. No provisional winner is written while the corrected D1 result is pending. A historical v17 replay only checks numerical averaging and interfaces.

The code and this rule are frozen before corrected D1 scores exist. Any later implementation correction must preserve its predecessor and explicitly document whether it changes selection mathematics. The final selection is written once and independently reverified by the audit command. Public preregistration and authorization to access D4 remain separate root-controlled gates.
