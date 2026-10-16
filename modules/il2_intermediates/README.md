# IL2 protein evidence: compact fixed-readout reproduction

`python reproduce.py --output-dir NEW_EXTERNAL_DIRECTORY` recomputes the completed primary and MAGeCK secondary comparisons, including the null primary result. Inputs retain all 561 candidate mappings, the fixed ten top-25 nomination lists, 4,330 candidate/control guide readouts and all candidate-mapped MAGeCK results in both original modes. Upstream raw counts and MAGeCK fitting software are omitted.

The primary analysis recalculates library/control centering, target medians, nomination contributions and the original paired guide/control bootstrap (20,000 draws, seed 20261011). Eligibility and guide-overlap decisions are fixed intermediates. The secondary analysis uses the unchanged original scoring script and retained six-column candidate projection of the fitted MAGeCK tables. Full original table hashes and projection provenance are recorded. This reproduces readout scores through comparisons; it does not repeat raw-count preparation, sequence overlap checking, or model fitting.

Numerical scorecards are compared with the original expected results at absolute tolerance 1e-12 and relative tolerance 1e-11. `REPRODUCTION_REPORT.json` retains all scientific comparisons and the actual execution time. Python dependencies: NumPy 2.3.5 and pandas 2.2.3. No downloads are performed.
