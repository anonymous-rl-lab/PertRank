Fixed-task, lossless numerical reproduction of the completed independent-RNA comparison.
All 52/59/63 candidates and 293/289/289 queries retained. Original float64 predictions and weights are unquantized.
Reference resampling preserves exact top-10 inclusion probabilities, including fractional ties, for all 500 draws in each of both schemes. This is sufficient for the archived fixed-ranking experiment, not for new methods/K/candidate sets.
Run: python code/reproduce_scores.py --bundle . --out NEW_EXTERNAL_OUTPUT
Model fitting and scientific endpoint recomputation were not performed during packaging; exact data representation checks were performed.
Full raw counts and full-gene models remain in their separately archived original packages.
