# R20 intermediate-data replay

This compact module preserves the archived per-query scores for all three
methods (Unweighted, Euclidean, PertRank), the original query IDs and block
definitions, and the unchanged statistical code. For each of three source
donors it covers five within-screen responsive inputs, five within-screen
tercile inputs, and two disjoint external-RNA inputs. The external analysis
corresponds to Supplementary Table SN1.2. The historical D1 query panel is
retained; later corrected-panel experiments are separate.

```bash
python -m pip install -r requirements.txt
python reproduce.py --output-dir /path/to/new_r20_results
```

The output directory must be new or empty and outside the module. Replay
recomputes all per-input and group-average means, paired contrasts, realised
group ranges/sign counts and conditional query-block intervals, using the
original 2,000 draws and seed 20261006. It also rebuilds the descriptive
cross-source summary. Every statistical result is compared with its historical
value, with absolute tolerance 1e-12 and zero relative tolerance. A successful
run writes `REPRODUCTION_CHECK.json` with `status: PASS`.

`data/` contains scientific intermediates, not final summary statistics:
scores have dimensions method × input group × query × direction. Query blocks
are resampled with the original rule, keeping the two directions paired and
averaging input groups within each query before group-average inference.
`expected/` supplies historical comparison targets, not calculation inputs.

This release starts at query scores. It does not reconstruct effects, weights,
rankings, nomination membership or turnover, and contains no raw cells,
moments or candidate prediction tensors. Historical full-result JSON includes
additional turnover fields solely for provenance; this entry point does not
claim to recompute those fields. A prior verified moments-to-predictions replay
is retained as an explicitly historical record in `provenance/`.

The original score NPZ files are preserved byte-for-byte. Export provenance
identifies the source contexts and external query definitions. The original
freeze is a historical record of a larger archive, not a promise that all its
files are included here. Conditional intervals retain the original fixed
controls, candidate pools, reference and query definitions; the three source
donors share one external RNA experiment.
