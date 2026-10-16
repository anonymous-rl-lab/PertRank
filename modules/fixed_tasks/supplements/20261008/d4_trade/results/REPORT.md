# D4 fixed TRADE-scalar supplementary comparison

Post-hoc descriptive comparison. The original D4 H1/H2 decisions and selected PR_M competitor remain unchanged.
Five input groups come from one donor; they are not five donors. TRADE_scalar is an adaptation that weights the original log-effect output, not the full native TRADE workflow.

| Panel | Arm | P@25 | Reference members per 25 |
|---|---|---:|---:|
| responsive | raw | 0.30175088 | 7.54377 |
| responsive | Euclidean | 0.35564993 | 8.89125 |
| responsive | PertRank | 0.37917333 | 9.47933 |
| responsive | PR_M | 0.36893621 | 9.22341 |
| responsive | TRADE_scalar | 0.35111453 | 8.77786 |
| terciles | raw | 0.07845306 | 1.96133 |
| terciles | Euclidean | 0.10519020 | 2.62975 |
| terciles | PertRank | 0.11062982 | 2.76575 |
| terciles | PR_M | 0.10889608 | 2.72240 |
| terciles | TRADE_scalar | 0.10215482 | 2.55387 |

| Panel | PertRank minus TRADE_scalar [conditional 95% interval] | Statistical direction | Practical equivalence ±0.005 |
|---|---|---|---|
| responsive | +0.02805879 [+0.02028357, +0.03496841] | superiority | not_established |
| terciles | +0.00847500 [+0.00468994, +0.01203309] | superiority | not_established |

Exact independent boundary-tie expectation, own-target exclusion, two signs, original 300-query panels, 30 query blocks, 2,000 paired bootstrap draws, seed 2026.
Groups are averaged within each query/sign before signs are averaged within query and query blocks are resampled. Intervals condition on inputs, controls, panel, candidates and deep reference. Labels can overlap; statistical superiority does not by itself establish a practically large difference.
All old query scores and aggregate reported intervals reproduce exactly. Every new and old query score is cross-checked by an independent full-sort scorer (tolerance 1e-12).
ALL_ARM_MEANS.csv and ALL_CONTRASTS.csv retain each of the five input groups and every method pair. D4_TRADE_RESULT.json records original source paths/hashes, weight provenance and unchanged original decisions.
Compact replay: python score_trade_supplement.py replay --replay-dir RESULTS_DIRECTORY
