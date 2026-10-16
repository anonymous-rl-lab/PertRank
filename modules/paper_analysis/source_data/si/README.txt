Supplementary Source Data for PertRank manuscript v21

These tables transcribe previously completed results. They do not add model fits, resampling runs, new reference scores, or candidate selections.

SN19_D3_all_21_FR_configurations.csv
  Source: code v1.4.0 external-benchmark module, provenance/benchmark/FR_D3_WINNER_LOCK.json, full_grid_results.

SN19_all_native_arm_scores.csv and SN19_all_native_contrasts.csv
  Source: the same module, original_reports/g000.json through g021.json, results[K].arms and results[K].contrasts. Original report identities are preserved in source_report.
  g000/g001/g002: D4 Rest/Stim/equal-state, original rank100 FR.
  g003/g004/g005: D4 Rest/Stim/equal-state, post hoc rank300 FR.
  g006/g007/g008: D3 250-candidate ASH depth diagnostic,20/50/100cells.
  g009: D4 Stim395-candidate independent-guide ASH comparison.
  g010/g011/g012: D3 FRcurveA nominalpenalty,100/20/50cells.
  g013/g014/g015: D3 FRcurveB effectivepenalty,100/20/50cells.
  g016/g017/g018: D4 Stim/Rest/equal-state ASH.
  g019/g020/g021: D3 rank150/rank300/rank200 extensions.
  For original FR-family simultaneous interval scope, see SM9.6. Post hoc and ASH contrasts are descriptive. Empty or implicit depth fields indicate the 20-cell task.

SN20_RNA_scores.csv and SN20_RNA_contrasts.csv
  Source: code v1.4.0 retained 20261014_upgrade/native_rna/original_scores/{D3_Rest,D1_Rest,D2_Rest,equal_source}/report.json, means and contrasts. Equal-source summaries have no source-independent pooled confidence interval.

SN20_reference_sensitivity.csv
  Source: the same original RNA reports, reference_sensitivity. Units are expected reference members per ten nominations, not P@10. Two original sets of500 shared reference draws;2.5–97.5percentiles are empirical sensitivity ranges.

SN20_archived_nomination_accounting.csv
  Source: the original A19-R native-method reports, nomination_redistribution. Values are previously saved changes in marginal top-K membership probabilities. Added minus removed reference membership is the archived net difference; no new candidate-level calculation was performed for manuscript v21.

SN21_interactions.csv
  Source: retained 20261014_upgrade/interactions/RESULTS.json, A_geometry_interaction and B_depth_interaction. Columns report original paired descriptive conditional95% intervals. Geometry uses rank300 FR. Depth uses one nested D3 input chain; two D4 states come from one donor.

The original full-precision reports, score arrays, configuration identities, software sources and numerical replay are retained in code release v1.4.0. This manuscript package contains publication-ready table exports; these exports are not substitutes for the code package's replay data.
