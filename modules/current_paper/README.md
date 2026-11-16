# Current-paper evidence and figures (current submission)

This module adds the independent-RNA four-arm comparison and retained-cell guide diagnostic used in the current paper, plus its current figure code and numerical source tables. It complements the retained core and the separate `pertrank_q` module. It contains no manuscript PDFs or historical manuscript versions.

From the repository root:

```bash
python modules/current_paper/reproduce.py --output-dir ../current-paper-reproduced
```

Use a new directory outside the repository. NumPy, SciPy and Matplotlib from the root numerical-replay environment are sufficient. Add `--skip-figures` for numeric replay only. The command writes an explicit `REPRODUCTION_REPORT.json` and logs; inputs remain unchanged. The full command is offline after dependencies have been installed.

| Component | Actual executable starting point | Recomputed | Not claimed |
|---|---|---|---|
| External RNA four arms | Original query effects, weights, query-direction projections and fixed external memberships | All four candidate rankings, query scores, 2,000-draw paired block intervals, exact nomination accounting and existing 500-draw reference sensitivity | Cell preprocessing, refitted FR/ASH, a newly unseen RNA experiment |
| Guide diagnostic | Retained-cell Gram/projection statistics, shallow Gram/projection statistics, exact locked prediction files, guide/cell identities | All 174 entries; 123 eligible entries × 20 retained-cell partitions × 3 metrics; independent implementation audit; unchanged-list gains/losses | New model fitting; rerunning the earlier failed selector; independent biological replication across the 20 partitions |
| Current figures | Frozen estimates and intervals in current plotting inputs | Six main figures and all seven supplementary figures; PDF/PNG and SVG where provided | Regeneration of upstream measurements or all model fits |
| Current source tables | Exact current submission exported numerical tables | Available for inspection and current navigation | Every displayed table cell freshly derived from raw data |

`SHARED_ASSETS.json` reuses byte-identical inputs already shipped in sibling modules. Every reused file is checked by SHA256 before being copied into the output-only runtime tree. This avoids duplicating reference draws and predictions already retained in the core. Keep this module inside the complete repository.

The guide replay starts from **fixed predictions**, whose bytes match the original lock. It retains the Gram statistics needed for the complete diagnostic and drops only the unconsumed `query_effect_cells` array from each shallow input file; `guide/COMPACTION.json` records original and compact hashes and retained keys. Exact output-array bytes and CSV bytes are checked against the original diagnostic. The original full guide replay was also rerun during packaging and passed; its receipt is recorded in `PROVENANCE.json`. No numeric downcasting or rounded substitute data are used.

The four-arm point estimates and intervals are compared with the original reports. Reference-sensitivity ranges remain empirical ranges conditional on fixed predictions, not population confidence intervals. The three input sources share one RNA experiment. Guide counts/signs remain descriptive. PertRank-Q's checkpoint replay remains in `../pertrank_q/`; the plot of Q here is only a display of its retained results.

Current numerical source tables and figure inputs use version-neutral filenames. They contain the submission's numerical evidence, not manuscript text. Analytic SM5 identities require no additional fitted model or experimental data; their empirical inputs are the existing nomination accounting and core diagnostics.
