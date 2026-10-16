# Completed D4 TRADE and D3 NTC supplemental experiments

This supplement integrates the completed experimental data, executable code,
fixed results and scientific provenance into the D1–D4 reproduction package.
Run the repository-level command:

```bash
python reproduce.py --output-dir ../PertRank_reproduced
```

It verifies the package manifest, replays the original four-donor experiments,
then runs both supplements without network access or model fitting. Outputs go
to `../PertRank_reproduced/supplements/`. NumPy is the only replay dependency.
The original acquisition and R fitting code is retained as provenance; running
those stages requires the full external source data and recorded dependencies.

## Contents and scope

| Module | Included | Recomputed by the compact replay |
| --- | --- | --- |
| [D4 TRADE](d4_trade/README.md) | All 2,945 fitted weights and latent moments; saved fitted mixtures; missing top-list support effects; code, rules, scores, result tables and audits | Scalar weights and mixture moments; five-arm rankings; all 30,000 panel/group/sign/query scores; paired intervals and summaries |
| [D3 NTC](ntc/README.md) | Exact frozen cell/guide selection; float64 centered cross-Gram; 64 target-pair energies; original acquisition/statistical code, results, resampling and audits | Point estimates; all 2,000 NTC bootstrap and permutation replicates; all 2,000 target-pair bootstrap replicates; intervals, p-values and selected-cell lane summaries |

For D4 the old four-arm support is insufficient to evaluate a new weight.
The supplement therefore retains the additional 6,089 original float64 effects
needed by the TRADE lists, including all boundary ties. Construction checked
all 17,463,850 relevant membership probabilities against the full input matrix.
No original D1–D4 numerical data, reference results or confirmation locks are
changed. The new arm remains separate from the historical competitor selection.

For NTC the Gram matrix retains the cross-half relationships required by the
frozen diagnostic. It is nonsymmetric and stored in float64 without rounding.
The compact replay does not reconstruct gene-level half means from it. The
original extraction and independent full-data audits are retained, together
with hashes identifying omitted large assets.

## Results

| Result | Estimate | Conditional 95% bootstrap interval |
| --- | ---: | --- |
| D4 responsive-panel PertRank minus TRADE scalar P@25 | 0.02805879 | [0.02028357, 0.03496841] |
| D4 expression-tercile PertRank minus TRADE scalar P@25 | 0.00847500 | [0.00468994, 0.01203309] |
| NTC reproducible guide-pair energy | 0.00313614 | [0.00289840, 0.00336845] |
| Same-target different-guide pair energy | 0.00245490 | [0.00210325, 0.00284905] |

The responsive D4 comparison corresponds to approximately 0.701 additional
deep-reference members per 25 nominations for PertRank versus the TRADE scalar
adaptation. This is a post-hoc supplemental comparison using the locked inputs,
not an added prospective hypothesis or a test of every use of native TRADE.
PR_M remains the preselected competitor and the original D4 decisions remain
unchanged. The five input groups are not independent donors.

The NTC permutation p-value is 1/2001 (0.0004997501), using the plus-one rule
with 2,000 permutations. Reproducible guide-associated expression structure
was detected. The diagnostic cannot assign this structure to clones, lane,
off-target effects or another cause, and cannot turn its magnitude into an
artifact fraction for the primary ranking result. The same-target comparison
is contextual; no new between-arm significance test is introduced here.

The NTC analysis uses 468 NTC guides and 64 same-target guide pairs, with two
20-cell halves per guide and 23,840 distinct cells in total. Its selection was
newly frozen using metadata during recovery; it does not reproduce the exact
interrupted historical selection. Half balance does not remove differences in
lane composition between guides. Of the NTC guides, 426 have exactly balanced
half-lane counts; target-pair between-guide lane total variation has median
0.55. All scientific estimates are conditional on the recorded selection and
preprocessing, including the disclosed overlap with existing feature controls.

## Size and omitted assets

The entire added supplement, including code and reference results, is below
10,000,000 bytes uncompressed as well as in its separate ZIP. The complete
D1–D4 repository is larger because it retains the existing donor/depth data.
`audit/INTEGRATION_CHANGE_AUDIT.json` records the original archive hash and
unchanged legacy files; the root manifest covers every delivered file.

Raw UMI batches, full-gene D4 fitting moments and the full NTC half-mean matrix
remain in the complete evidence archive. Their hashes are retained here, but
their bytes are not contained in this small supplement. New weights, queries,
candidate sets, list sizes or gene-level diagnostics require those full inputs.
Compression took place after the experiments and did not change their rules,
estimates or fixed decisions.
