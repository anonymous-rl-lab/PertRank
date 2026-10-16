# v20 biological plausibility sensitivity: analysis specification

Status: explicitly post hoc analysis of the opened D4 Stim8hr experiment. This specification is frozen before computing the new strata, subset scores, or contribution summaries; it is not a preregistration and does not change any v18 registered conclusion.

## Questions and scope

Use all 561 original target perturbations, both fixed panels, original five 20-cell inputs, k=5, beta=0.9, K=25, and the original independent deep RNA reference. Assess whether metric benefit is confined to targets with weak own-gene evidence, and annotate the five previously flagged CTLA4 candidates. A low-expression target is a risk stratum, not a known biological negative. NTC guides are a different, targeting-independent diagnostic population. No unopened donor/state is accessed.

## Own-gene evidence, defined without scoring outcomes

Identify the canonical target gene on the frozen 13,394-gene axis. The two target planning-half means (50 cells each) are contrasted with the two independent mechanism-control means (1,000 cells each), not with the input controls used to fit weights or the evaluation controls. Let c=(cA+cB)/2, t=(tA+tB)/2, dA=tA-cA, dB=tB-cB. Values are averages of normalized log1p expression, not raw counts or transcript percentage knockdown.

The four mutually exclusive strata are: (1) own gene unavailable on frozen axis; (2) c<0.10, low control expression on this scale; (3) c>=0.10, dA<0 and dB<0 and t<=0.80*c, consistent planning suppression; (4) all other measurable expressed targets, suppression unconfirmed by this descriptive rule. The 0.10 floor excludes means near the log-expression floor; 0.80 is a fixed descriptive 20% reduction of mean log1p expression, explicitly not a 20% RNA knockdown threshold. Neither is fitted to the method outcome or biologically validated. Continuous values and every target's classification are retained so these cutoffs do not conceal the evidence. No p-value or detection-cell fraction will be fabricated from means.

## Outcomes

1. Original full-pool contribution decomposition: within each query, direction, and input, sum (p_method-p_comparator)*p_reference/25 over targets in each stratum. Average directions and inputs, then queries. Stratum contributions sum exactly to the original all-target contrast; no within-stratum reranking is used for this decomposition. Report PR-E, E-raw, and PR-raw plus nomination membership shares by method. All fixed target strata retained.
2. Two nested common eligibility subsets: all measurable expressed targets (c>=0.10), and the consistent-suppression subset. Rerank every arm and the unweighted deep reference within the same subset, preserving own-target exclusion, K=25, query panels, and original weights. Execute only if >=100 targets remain before own-target exclusions (maximum random P@25 about 0.25). Fewer targets gives not-evaluable for this subset; do not lower K or redefine thresholds to obtain a result. Compare methods only within a subset; changed pool P@25 is not directly compared to full-pool P@25 as an efficacy claim.
3. For every original query, export target-level average contribution, baseline own-gene expression, planning suppression, weights, and alignment where already saved. Produce continuous visualizations as descriptive displays, without fitting or selecting an adoption gate.
4. CTLA4: annotate exactly SFTPA1, DEFA6, LYPD3, CLDN5, DLG4 using all original five groups and both directions. Include original effects, planning own-gene evidence, full-pool membership, and original reference membership. Do not remove them or select a more favorable case.

## Conditional uncertainty and numerical checks

Use the original 30 planning-derived query blocks separately for each panel, 2,000 paired percentile resamples, NumPy default_rng seed 2026, equal-query weighting with unequal blocks retained. Intervals are post hoc conditional sensitivities, not new confirmatory tests or biological-replicate intervals. Recompute original scores and compare to stored registered scores to <=1e-12. Check decomposition and all boundaries, including independent tie probabilities and own-target exclusion. Export first-group and all-group results without picking favorable input groups.

## Optional matched energy context

Inventory only existing D3 Rest assets for target cross-half energy on the same 13,390-gene axis and scale as the existing D3 NTC D. Do not compare D3 Rest D/2 against D4 Stim energies. Independent halves of 20 versus 50 cells change precision rather than requiring a proportional energy rescaling. D/2 is mean centered NTC-guide signal energy, not an artifact fraction or universal negative threshold. If a fully matched target energy cannot be established from existing data, report that limitation and do not invent a percentile.

## Outputs and interpretation

Self-contained analysis code, compact extracted inputs, target and query CSVs, JSON summaries, explicit provenance/hashes, and a concise Methods/Results note. No change to the frozen method, original candidate pool, registered outcomes, or paper text. Possible strengthening: preserved metric gains in a biologically interpretable eligibility subset would show that observed benefit is not confined to weak own-gene-evidence targets. It would not establish functional causation or remove guide-associated confounding.
