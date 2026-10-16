# Fixed source-panel transfer, v20

The 300 responsive queries previously fixed in D3 Rest were transferred unchanged to D4 Stim8hr. All 300 genes were represented on the frozen Stim control gene axis. The new task used 561 candidates, five existing 20-cell input groups, the existing input controls, and the same independent 100-cell reference. It used zero target-task planning cells to select the query panel. The transferred panel shared 130 genes with the original Stim responsive panel and seven with the Stim tercile panel.

The panel identity, eligibility, scoring, comparator order and source-block resampling rule were written to `PROTOCOL.json` before transferred outcomes were calculated. Source and target datasets had already been analyzed, so this is a post hoc application analysis. The original Stim task was first reconstructed across all 23,760 stored score values; the largest discrepancy was 1.17 × 10⁻¹⁵.

| Method | Mean P@25 | Reference members per 25 |
|---|---:|---:|
| Unweighted | 0.251793 | 6.2948 |
| Euclidean | 0.317454 | 7.9363 |
| Pearson metric variant | 0.324918 | 8.1230 |
| PertRank | 0.332537 | 8.3134 |

PertRank increased P@25 by 0.015083 relative to Euclidean, with a paired source-block percentile interval of [0.013465, 0.016939]. This corresponds to 0.3771 additional deep-reference members per 25 nominations. Relative to unweighted ranking, the gain was 0.080745 [0.072019, 0.086610], or 2.0186 members. Relative to the Pearson metric variant, the gain was 0.007619 [0.005928, 0.009994].

The 30 source-planning blocks were kept intact. Within each query, the five input groups and two directions were averaged before resampling. The 10,000 bootstrap draws (seed 20261012) resampled blocks and divided the summed query contrasts by the corresponding summed block sizes. These are descriptive intervals conditional on the existing inputs, reference, source panel and block partition. Leave-one-block-out PertRank–Euclidean estimates ranged from 0.014835 to 0.015564, with all 30 estimates positive.

## Replay

From the total package root, choose a new output directory:

```bash
python modules/paper_analysis/reproduce.py --only transfer --out /new/path/transfer-replay
```

The launcher restores `transfer_inputs.npz` losslessly in the output directory before running the original script. Direct execution in this compressed source directory does not supply that restored input.

`transfer_inputs.npz` contains exact float64 query effects, reference effects, the four frozen scalar-weight arrays, IDs and source block labels. It preserves the complete 561-candidate pool for every query and both ranking directions. `run_transfer.py` reconstructs all rankings, expected boundary-tie memberships, P@25 and paired comparisons. `source_panel.npz` preserves the original D3 panel and block assignment. `PROVENANCE.json` records coverage and the exact-score audit; `query_scores.csv` and `block_influence.csv` provide inspectable outputs.

The compact inputs support exact nomination and score replay. Original raw-count preprocessing and fitting of the pre-existing weights are documented by the source experiment archives. No target-task planning expression is required by this transferred-query use case.
