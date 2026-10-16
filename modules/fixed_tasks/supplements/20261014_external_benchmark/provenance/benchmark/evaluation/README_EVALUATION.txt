PRIMARY SIX-ARM EVALUATION

EFFECTIVE PRE-RESULT UPDATE: FR SEARCH NOW HAS 21 CONFIGURATIONS.
AMENDMENT_03_FR_SEARCH_21.json is the complete effective FR protocol; the
original six configurations retain their original order, followed by fifteen
additional configurations. Pass --protocol AMENDMENT_03_FR_SEARCH_21.json
to every FR evaluate/select-winner/combine command below. All 21 fit results
and all 21 D3 score reports are required for the winner. The concise original
six examples below illustrate command syntax only, not the current full grid.
Automated execution after all 21 fits complete:
python evaluation/score_development_grid.py
The driver scores every frozen configuration and locks the P25 winner. It
checks the complete fit-grid report and the latest declared scoring-code hash.

Primary code: evaluation/evaluate_six_arm.py
Independent audit: audit/audit_six_arm.py (not imported by primary code)

Only the existing D3_Rest, D4_Rest and D4_Stim8hr expressions are permitted.
D4 scoring enforces root FR_D3_WINNER_LOCK.json before any task/FR data read.
Native FR effect file hashes, official commit, solver validity, configuration,
seed, full raw gene axis, exact input-cell set and per-cell perturbation identity
are checked against frozen metadata. All query/candidate alignment uses exact
stable gene identities. Positive and negative P@K are averaged within query.
No output effect array is rounded.

D3 CONFIGURATION SCORING, ONE COMMAND PER CONFIGURATION
python evaluation/evaluate_six_arm.py evaluate --dataset D3_Rest \
  --config-id default --fr-effects /path/to/default/effects.npz \
  --out evaluation/D3_Rest/default

Repeat for the frozen six IDs in EXECUTION_SPEC.json. The D3 frozen task and
cell selection default to pooled_data/D3_Rest/{frozen_task,selection}.npz.

WINNER LOCK (all six completed D3 reports required)
python evaluation/evaluate_six_arm.py select-winner --reports \
  evaluation/D3_Rest/default/evaluation_report.json \
  evaluation/D3_Rest/rank10/evaluation_report.json \
  evaluation/D3_Rest/rank50/evaluation_report.json \
  evaluation/D3_Rest/lasso_low/evaluation_report.json \
  evaluation/D3_Rest/lasso_high/evaluation_report.json \
  evaluation/D3_Rest/rank50_lasso_high/evaluation_report.json

This creates root FR_D3_WINNER_LOCK.json with status
LOCKED_BEFORE_D4_SCORING, created_utc, config_id, rank, spca_alpha,
lasso_alpha, seed, native-FR P25 objective, all grid outcomes, input hash,
score/effects file hashes, protocol and selector hashes. Highest native-FR
P25 wins; scores within 1e-12 select default, otherwise frozen grid order.
An incomplete/failed/nonconverged grid cannot yield a strongest-in-grid lock.

D4 POOLED EVALUATION (files must be rep0, rep1, rep2, rep3, rep4)
python evaluation/evaluate_six_arm.py evaluate --dataset D4_Rest \
  --fr-effects /path/rep0/effects.npz /path/rep1/effects.npz \
  /path/rep2/effects.npz /path/rep3/effects.npz /path/rep4/effects.npz \
  --out evaluation/D4_Rest

Repeat for --dataset D4_Stim8hr and its five files/output directory.

EQUAL-STATE SUMMARY (Rest first, Stim second)
python evaluation/evaluate_six_arm.py combine --reports \
  evaluation/D4_Rest/evaluation_report.json \
  evaluation/D4_Stim8hr/evaluation_report.json \
  --out evaluation/equal_state

Original Rest/Stim responsive-query panels differ. Each state retains its own
300 queries/30 blocks. For equal-state uncertainty, SeedSequence(20261012)
spawns two independent state RNG streams; each state's original blocks are
resampled, five input repetitions are averaged, then state means are averaged.
These are conditional task-level intervals for the observed D4 donor, not
independent-donor or donor-population intervals. See AMENDMENT_02.

INDEPENDENT-GUIDE D4 STIM (one input repetition)
python evaluation/evaluate_six_arm.py evaluate --dataset D4_Stim8hr \
  --kind independent --fr-effects /path/independent/effects.npz \
  --out evaluation/D4_Stim_independent

Reference, predictions, selections default to the restored independent-guide
execution files. Joint intervals use 2000 reference-cell-by-lane and shared NTC
resamples, jointly with paired source-query-block resampling. Reference seed
9433175891597782681 and batch size 16 reproduce prior reference draws exactly.

OUTPUT FILES PER TASK
compact.npz:
  raw_scores/fr_scores [rep,query,candidate], reference_scores [query,candidate]
  w0/wm [rep,candidate], own_target [query] index or -1, blocks [query]
  queries [query], candidates [candidate], input_reps [rep]
  six_arm_p25 and six_arm_p10 [rep,6,query]
evaluation_arrays.npz:
  sampled_query_blocks [2000,30], query_weights [2000,300]
  six_arm_p25/p10 [rep,6,query]
  query_only_bootstrap_p25/p10 [2000,6]
  independent task also joint_bootstrap_p25/p10 [2000,6]
evaluation_report.json:
  six method points/rep values; five registered contrasts and extra PR-minus-E;
  paired percentile 95%/99% intervals; full provenance, uncertainty scope.
99% intervals apply Bonferroni to the five registered contrasts within one
task and K; the additional PR-minus-E contrast is descriptive and not part
of that five-test family. No family correction across tasks or K is claimed.

EQUAL-STATE OUTPUT
equal_state_report.json and equal_state_arrays.npz preserve each state's
separate draws, bootstrap scores, their equal mean, all contrasts and the
predeclared direct/complementarity resource-allocation gates.

CODE CHECKS BEFORE NEW SCORES
Synthetic mixed exact ties, own-target exclusions, both signs, multiple input
repetitions, unequal block sizes and both K values reproduce an independent
explicit boundary-membership loop. All-zero scores give K/(T-1).
BASELINE_REPLAY_CHECK.json records bitwise reproduction of all previous
independent-guide raw/Euclidean/PertRank per-query values and all 2000 joint
bootstrap method values at K25 and K10. This replay substitutes raw into the
FR slots and does not evaluate newly fitted FR.
