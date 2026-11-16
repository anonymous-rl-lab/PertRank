# PertRank-Q paper-to-code map

Q is the reference-supervised extension of PertRank. The retained score table agrees with the current manuscript. Scientific definitions, selected models and numerical results are unchanged by anonymous packaging.

| Paper item | Retained source | Verification output |
|---|---|---|
| SM12: architecture, normalization, anchor and objective | `experiment/baseline_B/`, `experiment/train_optimization.py`, archived normalizers and model weights | Restored inputs and frozen-weight inference |
| SM12.3: score-deviation regularizer | `experiment/train_optimization.py`: `.square().mean()` over the eligibility mask | Scientific implementation retained |
| SN24.1: three-source precision and conditional intervals | `experiment/epoch_pair_update/results/SUMMARY.json`, `paper_data/pertrank_q_scores.csv` | `stage2/REPLAY_PAIR_CHECK.json`, `PERT_RANK_Q_RESULTS.json` |
| SN24.2: two-stage direct comparisons and unsuccessful primary arms | Both result summaries, protocols and frozen plans | Complete numeric comparisons for both stages |
| SN24.1: epoch-pair selection and list changes | `experiment/epoch_pair_update/selections/`, selection audit and attribution code | `SELECTION_REPLAY_CHECK.json`: 12 tables × 25 alternatives |
| Fixed-reference sensitivity | `experiment/audit/reference_sensitivity.py` and retained inputs | All 100 cases and all nine methods |

The complete summary is stored once in `experiment/epoch_pair_update/results/SUMMARY.json`; the paper CSV provides its compact presentation. Replay compares every numeric field in both stage summaries and all 24 numeric cells of the paper CSV. Later exploratory branches described elsewhere in the manuscript are outside this module's checkpoint replay.

Three sources share one external RNA reference. Two held-target subpools per source are scored separately at K=10. The retained Q arm was secondary in the second development-stage plan; computational replay is not a new independent confirmation.
