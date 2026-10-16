# PertRank D4 Stim8hr: self-contained exact result reproduction

This module reproduces every completed Stim result from a compact set of original
float64 inputs. It is self-contained and requires neither the 377 MB prediction
archive nor an internet connection. It can be used alone or under
`supplements/20261011_stim` in the D1–D4 reproduction package.

## Run

Python 3.12 and the following pinned numerical libraries were used for validation:

```bash
python -m pip install -r requirements.txt
python replay.py --output-dir /path/to/new-stim-output
```

The output directory must not exist. Installation may require internet access;
replay itself prohibits network connections. The validated numerical run takes
approximately five seconds, excluding dependency installation, on the reference
CPU environment. No raw-data download or experimental rerun is performed.

Successful execution writes:

- `VERIFICATION.json`, with `status: STIM_EXACT_REPRODUCTION_PASS`.
- `registered_scores.npz`, recomputed query scores, target contributions and
  regression coefficients, together with their full target/query axes.
- `scientific_report.json`, all recomputed H1/H2 decisions, arm means, pairwise
  intervals, panel interaction, cross-guide results and alignment results.

The replay requires exact equality with the archived numerical outputs. It does
not widen tolerances or round mismatches. NumPy 2.3.5 and SciPy 1.17.0 are required;
the validated original and replay environment is Linux, Python 3.12, with one BLAS
thread. Differences in numerical libraries or hardware may cause the strict
verification to fail and must be investigated rather than silently accepted.

## What is reconstructed

1. Euclidean, PertRank and PR_M scalar weights are recomputed from original
   full-gene energy/noise moments using the unchanged frozen scalar formula.
   The raw arm receives weight one. No stored final weight is used as a shortcut.
2. Original float64 gene effects are multiplied by those weights and ranked using
   the unchanged frozen top-25 membership function, including both directions,
   self-exclusion and the original fractional treatment of boundary ties.
3. Memberships are scattered back onto the complete original candidate axis.
   The original memory layout and reduction order reproduce every score bit for
   bit; no integer or rounded score substitutes are used.
4. The complete S3 target contribution matrix is reconstructed. Zero-contribution
   targets remain in the full 561-target matrix, and the frozen eligibility and
   per-query self-exclusion rules remain in the regression design.
5. Unchanged frozen bootstrap, wild-cluster, gate and regression functions
   reproduce all registered evaluable endpoints. Expected outputs are consulted
   only after reconstruction to verify equality.

The validated output includes 23,760 main scores, 9,000 cross-guide scores,
168,300 target/query contributions and 900 regression coefficients. All 12
scientific report sections are checked for exact equality.

## Why this is small and exact

For each query and input group, `data/main_support.npz` retains the union of every
fixed method's positive and negative top-25 candidates, **all candidates tied at
each boundary**, and both reference tails. Original float64 effect values are
preserved; there is no quantization. A candidate outside this union has zero
nomination probability for all frozen arms and zero reference membership.
Omitting its interior value therefore changes none of the frozen memberships or
endpoints. Every method's memberships were also verified against an independent
full-sort implementation before packaging.

The main task retains 33.2244% of original effect entries. The cross-guide task
retains almost all entries because its 26-candidate pool at K=25 is nearly
saturated. Its small total size permits this without compromising the size limit.

This is **exact sufficient-input reproduction for the completed fixed experiment**.
It is not complete raw-UMI preprocessing or a replacement for the prospective
prediction lock. It cannot support arbitrary new weights, K, candidate pools,
query panels, control refits or other experiments requiring omitted interior
effects. The original full archives remain the sources for those tasks. The
retrospective compaction does not upgrade the registration status.

## Files and provenance

| Path | Purpose |
| --- | --- |
| `data/*_support.npz` | Original float64 support effects, reference values, axes and scalar moments |
| `data/*_query_context.npz` | Frozen panel membership and query blocks |
| `data/alignment_pretruth.*` | Frozen indices, eligibility and regression design |
| `frozen/` | Byte-identical original numerical source modules |
| `compact_support.py` | Support decoding, weight reconstruction and exact ranking/scoring |
| `compact_analysis.py` | Original report orchestration adapted to consume reconstructed scores/contributions |
| `tools/build_compact.py` | Rebuild the compact inputs from the original full local run |
| `audit/*COMPACTION*.json` | Input hashes, exact membership proofs and source identity |
| `expected/` | Original result arrays and report, used only as verification targets |
| `MANIFEST.json` | SHA256 integrity of all module members |
| `RESULTS.md` | English experiment report |

The original prediction-lock archive SHA256 is
`8f6f231006f2a93773f6c092466458014a6b443ff38b951b9732579b448902d1`.
It is recorded for provenance and is **not a runtime dependency**. The original
report and original independent evaluation audit are retained. The public copy of
the continuation receipt omits a private draft-registration URL and records the
original receipt hash and explicit redaction; its scientific/procedural facts
are unchanged. No upstream TRADE source is redistributed.

## Scientific and procedural status

The frozen primary numerical rule and gated PR_M comparison passed. TRADE failed
under the fixed estimator before reference access and remains `not_evaluable`;
there is no TRADE victory or successful-subset comparison. Cross-guide evidence
is limited by the almost saturated 26-candidate task.

The original report records `protocol_status: investigator_directed_deviation`
and `independent_e4_registration_verified: false`. This reflects execution under
the investigator's explicit continuation instruction without independent
verification of the separate E4 archive/approval. The compact package preserves
that qualification and makes no claim of complete E4 procedural compliance.
