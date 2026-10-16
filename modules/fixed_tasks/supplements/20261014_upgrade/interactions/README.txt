Fixed-score sufficient data: geometry and depth interactions, 2026-10-14

Reproduce offline:
  python replay.py --output-dir /path/to/new-output
Requires Python 3.11+ and NumPy from the enclosing environment specification.
The output directory must not already exist. Do not invoke Python with -O.

scores.npz contains all original P@25/P@10 per-repetition, per-arm and per-query
score arrays from the two D4 states and three D3 depths. It also contains the
exact original bootstrap weights and sampled blocks, including the separate
equal-state streams, plus query, candidate and block identity axes.

Storage is lossless. Repeated arrays are shared; integer arrays are narrowed
reversibly; floating-point values use exact dictionary coding; repeated weight
columns are stored once. No floating-point rounding or reduced precision is
used. SUFFICIENT_DATA.json binds each reconstructed array's dtype, shape and
byte digest to its original source. Source file identities are retained too.

Derived bootstrap score summaries are reconstructed as original weights times
the mean per-query score arrays. This avoids storing redundant products.
No fresh bootstrap draws, model fitting or expression access occurs. Both
metrics, all component effects, both interactions, 95%/99% intervals, bootstrap
standard deviations and draw counts are compared to archived RESULTS.json at
absolute tolerance 1e-12. A successful replay writes REPRODUCTION_CHECK.json.

This is complete replay from the fixed benchmark scores. It is not a standalone
raw-cell model-training dataset. The unchanged original analysis script and
freeze are retained in provenance/; replay.py is the portable entry point.
The original post hoc and conditional inference scope remains unchanged.

Packaging checks performed: syntax parsing and lossless decoding of every
retained array to its original dtype, shape and bytes. No statistical endpoint
or interaction analysis was executed during packaging; replay.py was not run.
