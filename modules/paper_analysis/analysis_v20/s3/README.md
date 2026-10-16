# v20 bounded S3 analysis

This directory contains the post hoc explanatory analysis specified in `POSTHOC_PROTOCOL.md`. `PROTOCOL_LOCK.json` was written before the new statistics were computed. `CODE_LOCK.json` binds the executed analysis code and extracted archived planning matrix. Original registered S3 results and decisions are unchanged.

Run from the total package root with a new output directory:

```bash
python modules/paper_analysis/reproduce.py --only s3 --out /new/path/s3-replay
```

The launcher runs the replay and its independent audit on copies under the output directory.

The replay requires Python, NumPy and SciPy; the current pinned dependencies are in the total package root `requirements.txt`. It refuses to overwrite existing results. All required original compact inputs and fixed numerical functions are bundled. It does not access raw cells, external files or the network. The audit independently checks rank_A coefficients by Frisch–Waugh–Lovell residualization and checks accounting identities against the stored contribution array.

## Principal outputs

- `results/s3_alignment_groups.csv`: four frozen alignment groups, supported gains/losses/net and changed-slot mass; all quantities refer to the original full-pool 25-slot lists.
- `results/s3_coefficients_summary.csv`: original observed rank_A slope, analytical reference-neutral slope, and paired difference.
- `results/s3_weight_transitions.csv`: six mutually exclusive original-weight transitions and their supported contributions.
- `results/s3_alignment_transitions.csv`: the full group-by-transition accounting.
- `results/block_partition_sensitivity.csv` and `leave_one_block_out.csv`: bounded conditional dependence checks.
- `results/query_level_results.csv`, `target_groups.csv` and `replay_statistics.npz`: exact per-query and per-target sufficient results.
- `results/Methods_Results.md` and `INTERPRETATION.md`: methods and scientific reading.

Intervals are nominal conditional resampling ranges across represented query modules. A range does not include new donors, controls, alignment-index estimation or experiment repetition. New bins and diagnostics were fixed in this post hoc protocol, not in the original Stim registration.

`frozen/` contains unchanged source functions from the original compact Stim archive. The existing source identities and attribution remain applicable; no new licence is imposed on the copied scientific sources.
