D3 ORIGINAL 21-CONFIGURATION FR GRID AND DEFAULT ASH

This module reproduces the existing D3 Rest first-input results for 1,200 candidates and 300 queries. All 21 original native FR configurations, raw, Euclidean, PertRank, and the two weighted FR arms are retained. Top-25 and top-10 are reconstructed using the exact original independent boundary-tie membership expectation, with own-target exclusion. The default ASH P@25 is reproduced from its completed posterior means; no unreported ASH confidence interval or P@10 analysis has been added.

Run from the benchmark module:
  python d3_replay.py --output-dir /path/to/new/output

The original float64 predictions were converted to two bit masks per signed query and K: values strictly above the cutoff, and values exactly tied at the cutoff. A tied candidate receives (K - number strictly above) / number tied. This retains the exact top-K marginal probabilities and all fixed-task scores, without quantizing effect values. The reference and scalar baseline memberships are stored once and reused across configurations. The original 2,000 paired query-block bootstrap draws are regenerated with seed 20261012 and the same NumPy algorithm.

This is complete fixed-result numerical reproduction, not a raw-expression or refitting package. Other K values, new query panels or fitting native models require the original archives listed in SOURCE_MANIFEST.json. The full source identities and original expected reports are retained. No new cells, models, tuning, or scientific endpoints were introduced.

The packaging verification recomputed the archived statistics: all 2,521 checked numerical values match exactly (maximum absolute difference 0.0). This validates the completed-result replay, not a new biological experiment.
