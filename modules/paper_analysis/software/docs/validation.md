# Validation of 0.2.0rc2

This release updates the software to manuscript v25.2. The fixed numerical method remains five control directions with attenuation 0.9. The frozen numerical oracle and both archived tutorial data files are unchanged.

| Check | Observed result |
|---|---|
| Source test suite | 99 passed; no failures or skips |
| Installed wheel test suite | 99 passed; no failures or skips |
| Imported distribution identity | Version and all installed code/data files match the release source |
| Frozen cell implementation | Existing regression checks pass at the stated 1e-12 tolerance |
| Archived fixture weights | Both datasets reproduce their saved/formula weights at 1e-12 tolerance |
| Replogle tutorial | All four code cells executed from the installed wheel |
| T-cell tutorial | All five code cells executed from the installed wheel, including saved FR estimate reuse |
| AnnData quickstart | Full synthetic count-input example executes |
| Decision scoring | Analytic tie examples, identity alignment, self exclusion and external-estimator scores pass |

The tests cover dense, sparse and backed AnnData inputs, normalization on the complete input gene axis, explicit control matching, own-target exclusion, malformed inputs, source-statistics provenance and chained external-effect reuse. The evaluator uses exact expected overlap of independently tie-broken prediction and reference lists. It does not choose a reference, establish its independence or calculate scientific uncertainty intervals.

Testing used Python 3.12.14; package versions are recorded in `requirements-tested.txt`, and the resolved environment is included in the software validation records. The supplied CI also configures Python 3.10, but no remote CI execution is claimed by this private release.

All notebook code cells were executed sequentially with captured table and text output, then rendered to HTML. The environment denied Jupyter's local network-interface operation, so interactive Jupyter transport was not validated; no notebook cell was skipped. The executable `scripts/check_tutorials.py` uses the same socket-free execution route.

Use `scripts/check_installation.py` after installation to verify the imported version and package bytes against this source tree. A source import and a wheel import are distinguished explicitly in its output. Tests and these software checks do not refit scientific baselines or change manuscript results.
