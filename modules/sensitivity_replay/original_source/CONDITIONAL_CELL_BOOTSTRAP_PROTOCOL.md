# Conditional query-expression cell bootstrap — 5 October 2026

Status: an additional development/sensitivity analysis on opened data. This does not amend any historical A4–A19 decision. No sealed source is read. The design and code hashes are written before any bootstrap output.

Question: if the observed 20 cells per target are resampled, how sensitive is the list to query-expression sampling **holding the already fitted weights fixed**? This is NOT full-method cell-sampling uncertainty. In particular, it excludes fluctuations in the weight, control PCA, control mean, truth, query selection, guides and donors.

Use D3, D1 (A6a self-gene map), and D2; each source's responsive and tercile 20-cell tasks. Use the original candidate pool, directions, own-gene exclusions, raw truth, frozen Euclidean/PertRank weights (k=5, beta=.9), and tie-expectation P@25. Fixed controls are the archived input-control means.

Counts are transformed by the archived library-size scale: log1p(count * scale / max(total, 1)). Each target's 20 rows are sampled with replacement exactly 20 times, independently across targets. The same draw is shared by all query genes, both panels and all three methods. No target, query, control or truth resampling occurs. Seeds D3=202610053, D1=202610051, D2=202610052. B=200 per donor, no tuning or stopping by outcome.

Pre-computation input check: all three methods' original P@25 reproduce exactly from stored per-cell counts in all six tasks. Archived Y versus count reconstruction has max absolute differences 8.31e-7 for D3 and <4.3e-9 for D1/D2, consistent with the documented float32 stream accumulation and control mean rounding. These are disclosed numerical differences, not adjusted data. Before resampling, require: target order and query gene axis exact; expected-score agreement with the original scorer <=1e-12 for all arms; probability sums equal 25; own-target probability 0. If these fail, stop.

Report the original effect and each bootstrap distribution's mean, SD, 2.5/97.5% empirical quantiles, and fraction >0 for PertRank–Euclidean and PertRank–Unweighted. These quantiles are **conditional perturbation-distribution summaries, not population confidence intervals or p-values**. Report each method's expected top-25 overlap with its original prediction (same independent-tie-expectation convention) to describe list stability. Do not pool the donors or count draws as independent scientific replicates.

A two-draw smoke verifies the code path before the 200-draw run. Its draws are the initial fixed draws and do not set parameters. No claims about superiority to Mixscale arise from this analysis.
