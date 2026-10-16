# PertRank D4 prospective donor confirmation

Status: FINAL PRE-ACCESS EXECUTION PROTOCOL. Primary rules are registered as OSF REST_PRIMARY_REGISTRATION. Corrected D1 and the deterministic competition selection are complete and independently audited; PR_M is the realized competitor. The approved pre-access addendum, later prediction registration and phase-specific access releases remain pending. This final protocol is prepared registration content, not a registration approval or D4 access release.

This document implements the user-approved v16 optimization plan. It restores
that plan's corrected-panel, five-input-group competition rule; the intervening
v16c single-group execution rule is retained as history, not used to select the
D4 competitor. The correction and this choice are made without accessing D4.
The final protocol, selected competitor, code, environment and genuine timestamped OSF
registration receipts must be frozen before any D4 request, including metadata.
An internal hash or mutable project upload is not a registration. The author-submitted primary registration specifies embargoed registrations shared by anonymous view-only links; public release follows publication.

## 1. Question and scope

Does the fixed PertRank control metric retain its gain on an unopened donor of
the same T-cell study, and does it outperform the strongest prespecified scalar
alternative selected on opened donors? This is a new-donor confirmation within
one study, not a new laboratory or platform validation. D3, D1 and D2 are opened
development resources. D4_Rest has not been accessed for this work. Stimulated
conditions and HEK293T remain outside this experiment.

The primary estimand is nomination performance given a planning-derived query
panel and fixed candidate pool, controls and deep reference. The five actual
20-cell inputs assess input sampling under those conditions. They are not five
independent donors. The budget excludes the explicitly available planning and
control resources; it is 20 prediction cells per target.

## 2. Completed prerequisites and competitor selection

The original1127-target D1 cohort's full planning spectrum from cells0–99 and
full deep spectrum from cells200–299 have been reconstructed, using the original
cell assignment, full12794-gene axis and controls. Stored query sums and the
old query panels/blocks reproduced exactly before full-axis own-Ensembl
correction and complete reselection of the300 responsive queries. Four queries
changed; this was not a manual replacement of a preselected list. All seven
existing arms' five-group weights were retained and predictions re-scored.
The independent audit checked84000 old/new scores with zero discrepancy and
252 paired summaries/intervals with maximum discrepancy2.78e-17. Original-panel
results remain a labeled sensitivity, not overwritten.

Eligible non-PertRank scalar arms, all with original log-effect output:
Euclidean, diagonal, TRADE_scalar, PR_E and PR_M. Raw is reported but is not a
fitted scalar competitor. Posterior/native-coordinate arms are not eligible.
Implementations and parameters are the completed v16/v17 definitions; none are
retuned. Any missing or invalid eligible arm blocks selection rather than being
silently dropped. TRADE uses the audited official core and frozen convergence
handling, not a replacement moment formula.

For each arm, average exact tie-expectation P@25 over five actual inputs and two
directions within each query, then queries within donor, then D3/D1/D2 equally.
Use corrected D1 and the existing corrected own-ID D3/D2 panels. Choose the
largest mean. Absolute differences<=1e-12 are ties; select the lexicographically
first frozen arm ID and report the tie set. The selection report and its inputs
are hashed before D4 access. The former group0 ranking is descriptive only. The realized winner is PR_M,
with tie set {PR_M}. Its equally weighted development mean is0.31688749743563355,
compared with0.3138706001865342 for TRADE_scalar,0.3099049550984995 for PR_E,
0.30576011061715797 for Euclidean and0.30508872345078125 for diagonal.
Raw and PertRank were reported but were not eligible. An independent second
reader recomputed all donor/group means from the locked score arrays and
verified the max/tie selection with maximum arithmetic difference5.56e-17.

## 3. Data, identities and deterministic assignment

Use the original public D4_Rest assigned-guide H5AD from the same study's S3
resource: `https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/D4_Rest.assigned_guide.h5ad`.
Record
ETag, size and feature/cell metadata on first authorized access. A later object
identity mismatch stops execution. No complete large object is needed.

Hash namespace: `pertrescue-tcell-probe|D4_Rest|`. Qualified cells have
low_quality=False and nonnegative target/type codes. Targeting cells must be
typed targeting; exclude multi labels and NTC/non-targeting/nan labels. A target
needs at least300 qualified cells. Order eligible symbols by SHA256(namespace+
symbol), take the last min(1200,N). Symbol-to-Ensembl and selected target IDs
must be unambiguous. Fewer than26 eligible candidates stops because self-target
exclusion must leave25; this is a mathematical requirement, not a power filter.

Within each target, sort original row IDs, then permute with NumPy default_rng
seed int(SHA256(namespace+symbol)[:16],16). Select first300 cells. Positions
[0,100) are planning, [100,200) are input, [200,300) are truth. Five consecutive
groups of20 occupy the input block. All sets must be pairwise disjoint.

Sort qualified NTC cells by SHA256(namespace+'ctrl|'+rowID). First2000 are input
controls, next2000 evaluation controls. Fewer4000 stops. Controls and targeting
cells must be disjoint. Input controls define median full-gene UMI scale and
the weight gene set: any positive count among those2000 cells. Evaluation data
never choose genes, directions, parameters, targets or queries.

## 4. Fixed methods, queries and predictions

PertRank uses fixed k=5 and beta=0.9; its directions are learned only from input NTC. Euclidean and the
selected competitor use the same prediction cells. All core outputs retain
the original gene coordinates. Noise calculations include target and control
mean uncertainty with empirical projection variances; diagonal gene moments
are never substituted for missing directional statistics.

The predictor is the D1-regression-checked `prediction/engine.py`; source maps
and exact formulas are part of the registered code attachment. Log NTC rows use
the original float32-row/float64-moment convention and target rows float64.
Pearson definitions retain the frozen NTC-only proportions, theta100 and common
clipping. No D4 parameter optimization is permitted.

Responsive panel: use the100 planning cells per target, subtract input-control
mean, set each target's own correctly mapped gene entry to0, compute across-
target variance with ddof0, and select300 by the frozen NumPy argsort rule.
Use the original, unzeroed planning profiles for complete-linkage correlation
clustering requested at maxclust30. Record actual nonempty labels if ties yield
fewer30; do not split or change clustering after inspection. Bootstrap actual
nonempty groups. Missing/nonfinite queries or invalid clustering stop pending a
documented pre-truth implementation resolution, never outcome-driven filtering.

A300-query tercile panel follows the original control-mean/hash selection and
is descriptive. Query-self candidates are excluded independently at scoring
using exact Ensembl ID equality. Exact boundary tie membership probabilities
are used on both prediction and truth sides. No stable-order tie advantage.

Core arms: raw, Euclidean, PertRank, PR_M (the selected competitor). Save all five groups'
weights and query predictions, actual cell IDs and axes, planning diagnostics,
and the raw20/25/30/35/40/45/50 ladder before opening truth. Additional cells at
each depth come only from the remaining80 input cells under the original v16b
hash ordering; anchors and complements are locked, not selected by performance.

## 5. Primary estimand and inference

Let P(m,q,s,r) denote exact tie-expectation P@25 for method m, query q, direction
s in {negative,positive}, input group r in {1,...,5}. For responsive queries:

`d(q) = sum_{r,s}[P(PertRank,q,s,r)-P(Euclidean,q,s,r)]/10`

`Delta = mean_q d(q)`.

Use2000 paired query-block bootstrap draws, seed2026, two-sided95% percentile
interval. All methods share the same draws. Average groups and directions
within query first, not as150 independent query blocks. Report actual query and
block counts. These conditional intervals do not quantify population-of-donors
uncertainty or repeated control/reference sampling.

Primary confirmation: point Delta>=0.01 AND L95>0.
Improvement grade: point Delta>=0.02 AND L95>0.
The0.02 criterion applies to the point estimate, not to the interval lower end.

Only after primary confirmation passes is the formal competitive layer active:
PertRank minus the locked strongest competitor, with identical paired scoring.
L95>0 means statistical superiority; U95<0 inferiority; otherwise unresolved.
Separately, a whole95% interval within[-0.005,+0.005] is practical approximate
equivalence. Statistical direction and practical equivalence may coexist.
The95% equivalence requirement is stricter than the usual90% TOST interval.
All estimates are reported even if the gate closes, then as descriptive.
The registered duplicate-contrast clause is retained: had Euclidean won the
selection, H2 would coincide with the H1 contrast and be merged, not counted
as independent second evidence. This contingency is inactive because PR_M won.

## 6. Secondary analyses and interpretation

Report every raw/Euclidean/PertRank/competitor score, groupwise difference,
zero-weight fraction, exact candidate count, random reference and tercile score.
Apply the frozen v16b cell-equivalence inverse-depth rule and paired uncertainty
without modifying its interpolation/boundary/nonmonotonic handling. It is a
secondary performance equivalence, not proven cost savings, and cannot rescue
a failed primary result. Report interpolation limitations. Freeze C5 and
planning-task descriptors before truth; add D4 to descriptive deployment maps
without selecting a favorable subset or asserting causal prediction.

Primary failure is reported, including positive but below-threshold estimates.
Nonfinite/invalid method outputs stop scoring until a documented mechanical
resolution is possible. If PertRank's weights are identically zero in any input
group, report the degeneration and do not declare method confirmation even if
some aggregate comparison passes; do not change the weight rule or omit that
group. Report all observed zero-weight fractions, with no new exclusion rule.
Competitive failure/unresolved results do not erase a primary success; nor does
primary success establish competitive superiority. No k/beta changes, alternate
query panels, candidate thresholds, donors or stimulated conditions are used to
rescue outcomes. This experiment does not convert guide-specific replication
into gene-level external validation.

## 7. Two locks and access control

The author has registered the fixed rules as OSF REST_PRIMARY_REGISTRATION. The archived attachment
PertRank_D4_OSF_Registration_anonymized.md is the author-submitted scientific
record. It specifies three OSF registrations, all embargoed for anonymous
review and made public upon publication. This changes access-record packaging,
not scientific choices or thresholds. Preserve the earlier preparation draft.

Before ANY D4 request, both the primary registration and a linked pre-access
addendum must be registered and approved. The addendum records the actual
competitor selected by the already fixed algorithm, every eligible development
arm score, corrected-panel IDs, and exact final protocol/code/environment hashes.
A mutable upload or this local draft is not a substitute. Lock1 binds both
registration receipts and the underlying manifest without circular hashes.
Only root can issue ACCESS_RELEASE for metadata and input. Input means input
NTC, planning and prediction cells. Truth/evaluation-control expression remains
unread and unprocessed; metadata may determine reserved truth row IDs.

After input-only checks, root independently verifies prediction completeness,
cell disjointness, own mapping, axes, finite weights, frozen formulas and source
identity. Root then creates PREDICTION_LOCK hashing all inputs, queries, cell
lists, parameters, weights and predictions. That exact prediction lock must be
archived in the third OSF registration before truth access; execution waits
for verified approval of that registration as well. Its receipt is kept
separate from the registered lock, and ACCESS_RELEASE binds both; the lock must
not contain its own registration receipt. Reader writes PREDICTIONS_READY but
cannot authorize its own release. No truth or score access before root verifies
the registered prediction record and issues the appropriate release.

Range requests can coalesce intervening bytes; only explicitly allowed rows are
decoded and all transient payloads discarded. Source and resume cache hashes
must match. An accidental evaluation exposure is disclosed and confirmation
status reassessed; it is never hidden as ordinary preparation.

## 8. Execution errors, reproducibility and reporting

Mechanical errors before scores may be corrected with old/new hashes, evidence
and timing retained, followed by affected parity checks. Scientific definitions
cannot change under that label. After truth access, any correction/replay must
be disclosed; only a deterministic reproducibility rerun of the same locked
analysis is allowed, not a second selection opportunity. Keep original outputs.

Save compact extraction checkpoints, row-selection provenance, locks, source
hashes, scripts, environment, complete results and verification reports. Separate
preparation, development correction and D4 confirmation in the manifest. Report
results regardless of sign and do not claim completion until numerical and
independent audit checks pass. Primary registration is observed. Explicit approval evidence and the two later registrations remain subject to receipt verification; D4 access has not been released. D4 metadata and expression remain unopened for this project.


## 9. Finalized development records

- `v18_d4_confirmation/d1_correction/FINAL_CORRECTION_COMPLETE.json`
  SHA256 `c7d96d38bb59aa5e66719f403abdec04031a4f86250fe184bbe4e24c1d1dc1c4`.
- `v18_d4_confirmation/d1_correction/INDEPENDENT_CORRECTION_AUDIT.json`
  SHA256 `14bd6b69c4a1d828482da89f4769cfefd259737285828a813415f53218ee0eba`.
- `v18_d4_confirmation/competition_selection/COMPETITION_SELECTION_LOCK.json`
  SHA256 `223b4fb2c69ca4c586ce36e1eea2601b990eee95bac662cf70549ab6f5b665bc`.
- `v18_d4_confirmation/competition_selection/SELECTION_AUDIT_PASS.json`
  SHA256 `b44b20e91ca3bf0900704d2aa3c3314126c480f8cb8c3f73eecfe4cc99c8583d`.
- `v18_d4_confirmation/audit_protocol/FINAL_SELECTION_SECOND_READER_PASS.json`
  SHA256 `379efb0d4ea4796cb17f43a58ff693d229c8d9e5de0c730e556ae8b93a6c92c5`.

The final PREACCESS_MANIFEST separately binds this protocol, the exact execution code closure and environment manifest. Receipt-bearing locks are constructed only after actual OSF evidence is verified; no invented registration or approval timestamp is supplied here.
