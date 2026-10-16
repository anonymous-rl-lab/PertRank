# v20 post hoc S3 explanation

## Methods

All rankings used the original 561-candidate pool, own-target exclusions, exact boundary-tie probabilities, five 20-cell inputs and both directions. The 539 valid frozen alignment ranks were split at 1/3 and 2/3; the 22 invalid indices remained a fourth group, and no subgroup was reranked. For each group, positive and negative changes in membership probability were multiplied by the original reference membership and summed to give supported gains and losses per 25-slot list. Their difference is net contribution; the four groups sum to the original metric gain. Six mutually exclusive exact weight-transition categories partition target-by-input combinations using the same accounting. They describe attribution under joint reranking, not isolated interventions.

The analytical reference-neutral diagnostic replaces each candidate's reference membership by K/N_q, retaining every prediction, original S3 design, covariate rank and query-specific exclusion. Observed, neutral and paired-difference rank_A slopes are reported. This reference-agnostic comparator detects structural association; it is not a biological null and does not establish causality.

Nominal conditional ranges use the same 2,000 paired query-block draws (seed 2026), equal-query weighting and 30 original blocks. They quantify sensitivity across represented query modules, not donor or full-experiment uncertainty. A fixed leave-one-block-out analysis and prespecified-in-this-posthoc-protocol 20/40 planning-derived complete-linkage partitions assess block dependence; the archived 30-block partition was exactly reconstructed before these alternatives were used. None replaces any registered decision.

## Results

The metric gain reproduced exactly: 0.0151483176 P@25, or 0.378708 reference members per 25 nominations.

| Alignment group | Targets | Gained | Lost | Net [conditional range] |
|---|---:|---:|---:|---:|
| low | 179 | 0.331159 | 0.095760 | 0.235400 [0.200441, 0.282566] |
| middle | 180 | 0.518334 | 0.206521 | 0.311813 [0.285636, 0.340307] |
| high | 180 | 0.162763 | 0.334261 | -0.171497 [-0.196301, -0.139289] |
| unclassified | 22 | 0.009333 | 0.006341 | 0.002993 [-0.001274, 0.006917] |

| Alignment slope | Mean [conditional range] |
|---|---:|
| observed | -0.0002503550 [-0.0002736534, -0.0002330586] |
| reference_neutral | -0.0000556693 [-0.0000575784, -0.0000537525] |
| observed_minus_neutral | -0.0001946857 [-0.0002179718, -0.0001773089] |

Weight-transition results, exact per-query results and all fixed block sensitivities accompany this report. The observed-minus-neutral sign must be interpreted jointly with actual gains and losses; it is not itself a benefit score. No simulation-derived tercile values are substituted for observed quantities.
