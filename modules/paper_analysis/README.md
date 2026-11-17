# Mechanism, transfer and application replay

This module reconstructs compact arrays from `array_payload.tar.xz`, checks their declared byte hashes, and runs five analyses: `s3`, `biology`, `transfer`, `trade` and `application`. Numerical precision is unchanged. The sibling `fixed_tasks` module supplies shared inputs; no raw data download or new model fitting occurs.

From the package root, use the dispatcher to restore shared inputs automatically:

```bash
python reproduce.py --only paper_analysis --output-dir ../paper-analysis-reproduced
```

The lower-level `modules/paper_analysis/reproduce.py --out ...` command is available
from the restored execution tree at `OUTPUT/_source/`, where the sibling
`fixed_tasks` inputs have been restored and checked.

The output directory must be new. Select a subset with `--only s3 biology transfer trade` before `--out`. The launcher writes `LOSSLESS_RESTORATION.json` and `REPLAY_REPORT.json`; array/table checks use the original numerical tolerances.

| Component | Recomputed |
|---|---|
| s3 | Alignment and weight-transition accounting, reference-neutral slopes, block sensitivity and independent FWL audit |
| biology | Target strata, full-pool contributions and common-pool subset rankings |
| transfer | Four-arm transferred-panel rankings and paired block intervals |
| trade | Five-arm sensitivity rankings and 2,805 fitted-mixture moment/weight checks |
| application | CTLA4 candidate lists, effects, weights and reference support |

Current manuscript figures and numerical table snapshots are supplied by `../current_paper/`; obsolete display copies have been removed. `software/` contains the separately installable API and executable real-data tutorials. The optional software route uses a separate installed environment, as documented in the root README.
