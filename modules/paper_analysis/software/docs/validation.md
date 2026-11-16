# Current software validation

The delivered review software retains the fixed five-direction geometry with attenuation 0.9. Its core numerical implementation and both authentic tutorial arrays are unchanged. This verification checks implementation and packaging, not new biological performance.

| Check | Observed result |
|---|---|
| Source test suite | 99 passed; no failures or skips |
| Installed wheel test suite | 99 passed; no failures or skips |
| Imported distribution identity | Current version and every installed runtime/resource file matched the source; import was from the wheel |
| Frozen cell implementation | Regression checks passed at the stated tolerance |
| Real-data fixture weights | Both matched archived weights within 1e-12 |
| Replogle tutorial | All 4 code cells passed; 180 exported nomination rows |
| T-cell tutorial | All 5 code cells passed, including saved FR-effect reuse; 180 exported rows |
| AnnData quickstart | Synthetic raw-count input example passed |
| Dependency consistency | `pip check` passed |
| Curation entry points | Syntax and CLI help passed; upstream curation was not rerun |

The suites exercise dense, sparse and backed AnnData, full-axis normalization, control matching, identifier alignment, own-target exclusion, boundary ties, source-statistics provenance and chained external-effect reuse. There were 57 dependency deprecation warnings per suite, from AnnData/Pandas string handling; no test was skipped because of them.

The current environment is recorded in `requirements-tested.txt` and `validation/requirements-resolved.txt`. Python 3.12 was executed; no remote CI or additional Python-version execution is claimed. Notebook cells ran sequentially through the socket-free checker; interactive Jupyter transport was not tested. Retained notebook displays are illustrative saved outputs, while the current complete-cell execution receipt is `validation/tutorials.log`.

The source distribution includes the scientific-design document, API reference and Replogle notebook. The wheel retains runtime code and authentic fixture data. `scripts/check_distribution.py` checks this contract and ensures the PyPI description contains no repository-relative documentation links. Exact build results are recorded in `validation/DISTRIBUTION_CHECK.json`.

The curation scripts accept explicit upstream paths rather than preparation-workspace paths. Exact input hashes and conversion definitions remain in the fixture provenance. These upstream inputs are not needed to execute the tutorials. Original raw-cell preprocessing, scientific estimator training and PertRank-Q are outside these software checks.
