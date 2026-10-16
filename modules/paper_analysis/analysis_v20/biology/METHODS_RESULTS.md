# Biological interpretability of the existing Stim nominations

## Identity and scope

This is a post hoc sensitivity analysis of the opened, completed D4 Stim8hr experiment, specified in `POSTHOC_PROTOCOL.md` before computing the new strata and sensitivity scores. It does not modify the registered task, method, parameters, query panels, reference, or original conclusions. Its question is whether the added-metric benefit is confined to targets with weak measured evidence about their own gene. Low expression is an interpretability risk layer, not a verified biological null.

## Methods

All 561 original targets were retained for annotation and the original-pool decomposition. For a target whose own gene occurred on the frozen 13,394-gene axis, its two 50-cell planning means were contrasted separately with the two independent 1,000-cell mechanism-control means. These mechanism controls were disjoint from the input controls that fitted the weights and from the evaluation controls. The two planning means are also independent of the five ranking inputs and deep-reference cells by the original role allocation.

Let c be the mean of the two control-half means and t the mean of the two target-half means on the normalized log1p expression scale. Four mutually exclusive strata were fixed: own gene unavailable on the frozen axis; low control expression c<0.10; expressed c>=0.10 with both planning contrasts negative and t<=0.80c; and remaining expressed targets. The third stratum is called consistent planning suppression. The expression floor and 0.80 ratio are descriptive thresholds specified before new scoring, not fitted cutoffs, RNA percentage knockdown, significance tests, or validated biological classifiers. Averages of log1p expression cannot supply raw count detection fractions or transcript knockdown percentages. Continuous values for every target are retained.

Full-pool decomposition sums (p_PertRank-p_Euclidean)p_reference/25 over members of each fixed stratum, before averaging the five inputs, both directions, and queries. All membership probabilities come from the complete original pool. Stratum contributions therefore sum exactly to the original gain; the strata are not reranked for this decomposition. Parallel decompositions retain Euclidean-minus-raw and PertRank-minus-raw, and nomination shares are reported by method.

Two additional common candidate pools were prespecified: all measurable expressed targets and the consistent-suppression subset. Both had to contain at least 100 targets before own-target exclusion to be evaluable. In each pool, all three methods and the unchanged unweighted deep reference were reranked with K=25. Original fitted weights, the two 300-query panels, and input allocations were retained. These are separate conditional tasks: their absolute P@25 values are not directly comparable with the full-pool score as evidence of improved efficacy. Original query blocks were sampled in 2,000 paired percentile resamples with seed 2026. Intervals condition on fixed panels, target strata, input samples, controls, and reference, and do not make the strata themselves known without error.

The complete original score tensor was independently reconstructed from exact-order representations of the frozen float64 predictions and reference. Every original query/group/direction/method score agreed with its registered value within 7.8e-16. Order codes preserve all equalities, including zero-score ties, and support exact common-subset reranking without storing a large expression matrix. They are not refitted biological data.

## Results

Of 561 targets, 66 had their own gene unavailable on the frozen axis, 183 had c<0.10, 291 met the consistent-planning-suppression rule, and 21 were expressed but did not meet that rule. The last category means evidence was unconfirmed under the descriptive rule, not that knockdown or perturbation was absent.

| Common candidate pool | Targets | Responsive PertRank−Euclidean P@25 | Tercile PertRank−Euclidean P@25 |
|---|---:|---:|---:|
| Original complete pool | 561 | 0.015148 [0.013088, 0.017870] | 0.004762 [0.003150, 0.006197] |
| Measurable expressed targets | 312 | 0.016910 [0.015234, 0.019111] | 0.003234 [0.001910, 0.004642] |
| Consistent planning suppression | 291 | 0.015949 [0.014187, 0.018429] | 0.002391 [0.001412, 0.003392] |

The original responsive metric gain decomposed into 0.010375 [0.008962, 0.012443] from the 291-target consistent-suppression stratum; 0.004231 [0.002992, 0.005576] from the low-expression stratum; 0.000304 [−0.000101, 0.000694] from unavailable own genes; and 0.000238 [−0.000060, 0.000529] from expressed but unconfirmed targets. This is a decomposition of the fixed ranking endpoint, not a causal allocation of method benefit or an artifact fraction. The corresponding contribution shares are not separate validation rates.

The common 291-target task retained a metric gain of approximately 0.399 deep-reference members per 25 positions. Its complete PertRank-minus-raw gain was also positive (exact values are in `subset_comparisons.csv`). Thus the observed benefit is not confined to targets with low measured own-gene expression or unconfirmed own-gene suppression. Because the evaluation still uses the same guide-associated deep RNA reference, this sensitivity strengthens biological interpretability without establishing gene-specific causation, independent-guide replication, or protein validation. The common pools differ in size and composition, so the larger point estimate in one pool is not evidence that filtering itself improves deployment performance.

## The five CTLA4 annotations

The named entries are perturbation targets; CTLA4 is the response query. SFTPA1 and LYPD3 were shared upward-list members with deep-reference membership in the displayed first input. DEFA6 was shared but outside the deep top 25. DLG4 and CLDN5 were PertRank-only downward-list additions outside that deep top 25. These identities must not be conflated into five PertRank-specific successes or failures.

SFTPA1 and DEFA6 own-gene measurements were unavailable on the frozen positive-input-control axis. This is not evidence that all cells or conditions have zero expression. LYPD3, DLG4 and CLDN5 had mechanism-control mean log1p expression of 0.01963, 0.01153 and 0.008763, respectively, placing all three in the prespecified low-expression layer. LYPD3 had negative own-gene contrasts in both planning halves but remained below the expression floor; DLG4's half contrasts differed in sign; CLDN5's planning mean was zero on the retained axis. None is elevated to a confirmed CTLA4 regulator by these observations. `CTLA4_flags.csv` preserves every group and both directions, including original CTLA4 effects, reference membership, weights, and list changes.

## D3 NTC energy comparison

The exact requested comparison between the D3 Rest NTC D/2 and a distribution of independent target cross-half d_I could not be recovered from the inspected compact D3 assets. The NTC cross-Gram identifies its repeated guide energy, and the D3 weight archive retains five single-input quadratic energies and variance traces, but not full-gene target cross-half inner products. The 64 stored same-target-guide difference energies are different estimands from target-versus-control signal energies. The D4 Stim d_I values belong to another donor/state/gene axis and were not substituted. A noise-subtracted single-input signal-energy estimator could be defined as a separate analysis, but it has not silently replaced the specified cross-half comparison. No artifact percentile or artifact fraction is reported.

## Replay and files

Run `python analyze_biology.py` from this directory, or provide `--out /absolute/path/to/biology`. It uses `compact_inputs.npz`, NumPy, and the Python standard library. `--extract-run /path/to/stim_run_20261011` rebuilds compact inputs from the original local execution assets; this option is not necessary for replay.

- `RESULTS.json`: all primary summaries and numerical parity.
- `target_biology.csv`: all 561 continuous annotations and stratum memberships.
- `full_pool_contributions.csv`: exact stratum decomposition and conditional intervals.
- `nomination_shares.csv`: fraction of nomination membership in each stratum, by panel and method.
- `subset_comparisons.csv`: within-pool method comparisons on both original panels.
- `query_contributions.csv`: all per-query scores and stratum contributions.
- `CTLA4_flags.csv`: all ten input/direction annotations for each of five named targets.
- `compact_inputs.npz`: exact tie-rank inputs, planning evidence and existing mechanism indices; approximately 7.3 MB.
- `INPUT_PROVENANCE.json`: source identities and hashes.

The analysis does not fit an automatic selection gate, remove targets from the registered main analysis, or infer raw-count detection rates or formal knockdown significance from the available mean summaries.
