# PertRank D4 Stim8hr: completed fixed-rule evaluation

## Result and execution status

The fixed numerical primary criterion was met. PertRank improved responsive-panel P@25 over Euclidean weighting by 0.015148 (conditional 95% interval [0.013088, 0.017870]). The gated comparison against PR_M also established superiority under the fixed restricted wild-cluster-t rule. The primary increment did not reach the 0.02 improvement-grade threshold. TRADE_scalar was not evaluable.

All 561 main-task and 26 cross-guide reference targets were completed. Independent local verification passed: saved reference means, row identities, assembled references, scores, H1/H2 decisions, S1-S3 results and the access hash chain were checked. All saved score and contribution arrays reproduced exactly; the independent block-sum implementation of the primary interval differed only by floating-point roundoff (<7e-18).

**Procedural qualification:** evaluation proceeded under the investigator's explicit instruction to continue. The investigator asserted that registration had already occurred; the separate Appendix E4 prediction-lock archive and approval were not independently verified before evaluation. The recorded status is `investigator_directed_deviation`, with `independent_e4_registration_verified=false`. The original strict evaluator was preserved, and a separately reviewed execution path retained the same scientific functions. No unverified registration ID, approval time or archive status is represented as verified. Meeting the fixed numerical criterion does not by itself establish complete E4 procedural compliance.

## Fixed task and uncertainty

- Condition: D4, 8-hour stimulation; 561 main-task candidate targets.
- Five frozen 20-cell input groups per target; 100 reserved reference cells per target and 2,000 separate evaluation-control cells.
- Responsive and expression-tercile panels each contain 300 queries; their union contains 594 queries with 6 overlapping queries. Both effect directions are scored.
- The control-geometry parameters remained k=5 and beta=0.9. Predictions, sample roles, panels, competitors, endpoints and decision rules were not changed after reference access.
- Main intervals use the fixed 30-query-block, 2,000-draw percentile bootstrap. They condition on the realized input groups, controls, panel, candidate pool and reference; they are not donor-population or cell-sampling intervals.
- Formal H2 comparisons require the H1 gate, a positive estimate and a null-imposed restricted wild-cluster-t one-sided p <=0.025 (9,999 draws). Percentile direction labels alone do not determine formal H2 superiority.

## All registered arms

P@25 values average the five input groups, 300 queries and two directions. Multiplication by 25 gives the average number of deep-reference members per list; it does not count distinct discoveries across queries.

| Arm | Responsive P@25 | Reference members per 25 | Tercile P@25 | Reference members per 25 |
| --- | ---: | ---: | ---: | ---: |
| raw | 0.316909 | 7.9227 | 0.089950 | 2.2488 |
| Euclidean | 0.378579 | 9.4645 | 0.124727 | 3.1182 |
| TRADE_scalar | Not evaluable | Not evaluable | Not evaluable | Not evaluable |
| PR_M | 0.387090 | 9.6773 | 0.127308 | 3.1827 |
| PertRank | 0.393727 | 9.8432 | 0.129489 | 3.2372 |

PR_M is the prespecified Pearson-residual/control-metric comparator; its final nominations retain the original gene-effect coordinates. It is not an independent published end-to-end algorithm.

## Primary and competitive decisions

| Responsive-panel contrast | P@25 difference | Conditional 95% interval | Additional reference members per 25 | Decision |
| --- | ---: | --- | ---: | --- |
| PertRank - Euclidean | 0.015148 | [0.013088, 0.017870] | 0.3787 | H1 fixed numerical criterion met; improvement grade not met |
| PertRank - PR_M | 0.006637 | [0.005372, 0.008127] | 0.1659 | H2a superiority established; one-sided wild p=0.0001 |
| PertRank - raw | 0.076818 | [0.070485, 0.083324] | 1.9205 | Descriptive full-method benefit |
| PertRank - TRADE_scalar | Not evaluable | Not evaluable | Not evaluable | H2b not evaluable |

H1 requires no all-zero PertRank group, a point estimate >=0.01 and a positive lower 95% bound. No all-zero group occurred. The H1 wild-cluster sensitivity p-value was 0.0001, so the frozen calibration-sensitivity flag was false. The reported p=0.0001 is the finite-resampling value, not p=0.

PertRank - Euclidean was positive in all five input groups (range 0.007789 to 0.023430). This is a direction check; it does not mean every individual group met the pooled 0.01 threshold. The benefits relative to raw, Euclidean and PR_M use different baselines and must not be added.

## Prespecified secondary outcomes

### S1: panel dependence

| Contrast | P@25 difference | Nominal conditional 95% interval |
| --- | ---: | --- |
| Tercile: PertRank - Euclidean | 0.004762 | [0.003150, 0.006197] |
| Tercile: PertRank - PR_M | 0.002181 | [0.000391, 0.003604] |
| Responsive increment minus tercile increment | 0.010386 | [0.007552, 0.014394] |

The larger responsive-panel increment is supported by the prespecified joint union-block analysis, which retains the six overlapping queries. PertRank and PR_M on the tercile panel fall within the fixed +/-0.005 practical-equivalence band according to the nominal secondary interval rule. This panel-specific result does not imply equivalence for the responsive task or universal interchangeability.

### S2: cross-guide task

| Contrast | P@25 difference | Nominal conditional 95% interval |
| --- | ---: | --- |
| PertRank - Euclidean | 0.000227 | [-0.000112, 0.000616] |
| PertRank - raw | 0.000307 | [-0.000339, 0.000859] |

The cross-guide task contained only 26 candidates at K=25. Its scores are near a structural ceiling (raw P@25=0.963813; PertRank=0.964120). Two lists of 25 selected from 26 candidates necessarily share at least 24 members; where self-exclusion leaves 25 candidates, both lists include all candidates. No cross-guide increment was resolved. The narrow range and nominal practical-equivalence label are specific to this nearly saturated task, not strong evidence of general cross-guide equivalence, absence of guide-associated structure, or a negligible artifact fraction.

### S3: frozen alignment prediction

The mean coefficient of the frozen alignment-rank index, adjusting for the frozen signal-to-noise-rank index, was -0.000250355; nominal 95% interval [-0.000273653, -0.000233059]. The predicted negative association was supported. The mechanism design included 539 eligible targets and 22 predefined exclusions; all 561 targets remained in the main ranking task.

This associates stronger signal/noise alignment with a smaller contribution to the PertRank-over-Euclidean nomination gain under the fixed model. It is not a causal decomposition, a validated automatic adoption threshold, or an estimate of guide-artifact removal. Index-estimation, donor and control uncertainty are not included. S1-S3 are nominal secondary analyses outside the H1/H2 family.

## TRADE failure and execution record

The fixed TRADE scalar adapter failed at ENSG00000167130, input-group index 2 (the third group), with `mixsqp_rcpp: solve(): solution not found`. Input moments were valid and the same RNG-state retry reproduced the failure. Forty fits were committed; two additional fits in the failed batch were diagnostic only; the planned 2,805-fit complete-cohort comparison was not completed. The unavailable-arm decision was frozen before reference access. No estimator retuning, target omission, imputation or successful-subset comparison was used. No superiority or equivalence claim against TRADE follows from this run.

The reference reader stopped on invalid remote responses and resumed from immutable checkpoints. Three later stops recorded HTTP 502; the first index-stage failure did not retain its HTTP status. Source identity/range guards were retained, and no invalid response was accepted. Scientific parameters and scoring rules were unchanged.

| Event | UTC timestamp |
| --- | --- |
| Prediction-manifest lock | 2026-10-11T05:42:33.274348+00:00 |
| Investigator-instruction receipt recorded | 2026-10-11T10:52:53.308610+00:00 |
| Investigator-directed execution release | 2026-10-11T10:56:07.383271+00:00 |
| First reference-expression decode | 2026-10-11T10:57:55.705658+00:00 |
| Last reference-expression decode completed | 2026-10-11T11:27:58.118578+00:00 |

The access audit verified 4,200 chained log records. Completed role reads comprised 2,000 evaluation-control cells, 56,100 main-reference cells and 2,600 cross-guide-reference cells, with 60,013 unique cells across these roles. Main and cross-guide reference roles can share cells; reserved reference cells were disjoint from input cells.

## Scientific interpretation

This result extends the observed control-geometry advantage to the D4 8-hour stimulated state under fixed settings, with a smaller gain on expression-tercile queries and a negative alignment association that was specified before reference evaluation. It supports responsive-query nomination as the clearest current use case. The evidence remains within this T-cell study and one stimulated donor. It does not establish cross-platform or donor-population generalization, superiority over TRADE, robust cross-guide validation, or a reduction in total experimental cost. No stimulated-state cell-equivalent calculation was performed.

## Reproduction and authoritative records

This self-contained compact module reproduces the completed fixed-method results
without requiring the 377 MB prediction-lock archive. It retains sufficient
original float64 effects, scalar energy/noise moments, reference tails, frozen
panels and designs, together with unchanged numerical sources. The retrospective
compactor verifies every signed top-25 membership against the full original
arrays, including all boundary ties. See `README.md` for the exact scope and run
command; arbitrary refitting and raw-UMI preprocessing require the full archives.

- Original prediction archive SHA256 (provenance only): `8f6f231006f2a93773f6c092466458014a6b443ff38b951b9732579b448902d1`.
- Full original result: `expected/EVALUATION_REPORT.json`.
- Full result SHA256: `87350eff5a706fcb26c8b618a4783b39130d5b0ad1a4d5ce7c3b22137f60a41b`.
- Original independent audit: `audit/ORIGINAL_FINAL_EVALUATION_AUDIT.json`.
- Original audit SHA256: `0b06fccaba65372e836f08b126bfdc7db7e918b44a41e35d0702a13e6550b20f`.
- Expected scores/contributions: `expected/registered_scores.npz`; consulted only for verification after reconstruction.
- Rounded values in this report do not replace the complete exact JSON/NPZ records.
