# Boundary context and supervision: exploratory experiment

## Identity and fixed comparisons
The preceding gate failed its primary criterion. Its no-extra-geometry arm
had equal-source delta P@10 +0.001232 with all source intervals including zero.
This historical outcome motivates the working architecture; it is not a new
confirmation. All reference data have been accessed previously. Oracle gates
are nonunique and will not be used as regression targets.
A / base_context: import every frozen prior neural_no_extra_geometry outer fit
and output, including all sources/halves/seeds; no refitting or seed selection.
B / boundary_context: add3 input-only boundary contexts; original SoftTopK loss.
C / boundary_pair (PRIMARY): identical architecture/features/initialization toB;
replace only the ranking objective with local boundary pair supervision.
All retain original wM anchors and raw signed readouts. Original extra geometry
predictor columns remain zero, so these are not geometry-free models.
Primary: C−original PertRank. Explanatory: B−A, C−B, C−A, all paired intervals.
A remains a historically selected development comparator.

## Inputs and contexts
Original20-cell inputs, same75-target global halves, separate D3/D1/D2 fits.
Full pools52/59/63, queries293/289/289. No identity embeddings or source pooling.
Retain previous8 common features and zero original3 additional geometry columns.
Add3 sign-reversal-invariant contexts with query-self targets excluded:
1. Tail depth=max(base−cut_up,cut_down−base)/training baselineSD. Cutoffs are
   midpoints of kth/(k+1)th eligible baseline scores in their direction.
2. Rank depth=((K+0.5)−min(midrank_up,midrank_down))/K, average ranks for exact ties.
3. Fraction of eligible candidates within0.5 training baselineSD of the cutoff
   attaining maximum tail depth; average densities when depths tie.
Contexts use the current available candidate pool at inference. Normalizer
moments use training targets only. Preserve prior seed-specific first11 hidden
input columns and bias. New3 columns use generator(seed+100003), uniform on
±1/sqrt(11); final head remains zero. Removing added contexts must recover prior
path within floating-point tolerance. B/C width8 and identical initialization.
Bounded log-odds correction ±log3, endpoint anchors and signs preserved.

## Supervision and loss
Compute reference TopK membership pi only from the current training target
pool, using exact empirical3-donor bootstrap count enumeration with shared
within-batch draws, independent batches where donor correspondence is unknown.
Pi is empirical reference stability, not latent-truth probability.
For each query/direction, input-only fractional boundary bands are
I=membership(K)−membership(K−2), O=membership(K+2)−membership(K), mass2 each.
For unordered pairs W_ij=(I_i O_j+I_j O_i)|pi_i−pi_j|. Pairwise softplus uses
label sign(pi_i−pi_j), signed scores in training baselineSD units, tau0.5.
Divide by fixed4, average queries/directions; equivalent directed I_i O_j sum
is allowed. Include correct and incorrect pairs; zero reference-support gap
has zero gradient. Exclude self-targets. These fractional bands are boundary
proxies, not exact tie-exchange probabilities. Never break ties by array order.
Both B/C use residual-score penalty0.05 and output-head mean-square ridge1e−4.
Log ranking/penalty/gradient components; equal numerical coefficients do not
imply equal effective regularization. No held-based loss rescaling.

## Staged training
Use prior source-specific balanced split(seed2026111516), both exchanges,
innerK5/outerK10, seeds2026111511/2026111512. Adam lr0.01,clipgrad5,300 inner
epochs; checkpoints0/50/100/200/300. Select maximum mean inner validation gain
versus original baseline requiring both exchanges nonnegative; ties earlier.
Zero epochs allowed. Fit normalization independently in every training pool.
First run numerical/context/gradient checks and all B/C inner fits. Proceed to
outer refits only if at least one new arm has at least4 nonzero selections out
of12 spanning at least2 sources. This is a resource feasibility gate, not
statistical evidence. If neither passes, stop at internal diagnostics. If either
passes, complete both arms for all sources at selected epochs (including0),
import A, ensemble equal seed scores, freeze all sources before held scoring.

## Evaluation and probes
Report exact fractional P@10, self exclusion, separate held pools; average
both directions and halves within query/source, then sources equally. Never
concatenate cross-fold scores. Half-pool P@10 is not comparable with old full-
pool P@10. Three sources share one external assay, not three independent studies.
A clear development signal: positive mean C−original, nonnegative each source,
positive conditional95% lower limits in at least2 sources. Also report C−A.
This is a development convention, not multiplicity-corrected confirmation.
Preserve all sources/halves/seeds, failures, normalizers, weights, checkpoints,
reference-support entrants/exits and probes. Inspect surrogate/real-list gains,
feature ranges, saturation, boundary label mass, finite gradients and replay.
No architecture/band/seed/eligibility/penalty changes from held performance.
Post-score diagnostics cannot select models; oracle scores are not model gains.
