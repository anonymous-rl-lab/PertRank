# Source data for PertRank

These files provide numerical sources for the manuscript figures, tables and
completed analyses, including the fixed native rvalues comparison and
predefined-program nomination.

## Navigation by manuscript section

| Main section | Evidence | Source directories |
|---|---|---|
| Results 1 / Fig. 1 | Controlled weight/readout design | `figures/` |
| Results 2 / Fig. 2 | D4 fixed settings, native comparison and FR reuse | `figures/`, `si/`, `rvalues/` |
| Results 3 / Fig. 3 | Independent knockout RNA and readout comparisons | `figures/`, `si/`, `new_results/`, `rvalues/` |
| Results 4 / Fig. 4 | Directional reliability and candidate competition | `figures/`, `new_results/` |
| Results 5 / Fig. 5 | Predefined-program nomination, frozen drug-source evaluation and source comparisons | `programs/` |
| Results 6 / Fig. 6 | Input efficiency, CTLA4 example, query-panel and software reuse | `figures/`, `application/` |

- `figures/`: original method, matched design, D4, mechanism, input-efficiency
  and panel-transfer display sources.
- `si/`: detailed native-method, independent-RNA and paired-interaction tables.
- `application/`: the previously fixed CTLA4 example, with all candidates, all
  five input groups, complete top-25 lists, changed candidates and provenance.
  Its reference is the same-screen deep RNA reference.
- `new_results/`: retained independent-RNA four-arm, guide-diagnostic and
  PertRank-Q results.
- `rvalues/`: complete five-task fixed native comparison, all-method scores,
  paired query-block intervals and D4/external aggregate summaries.
- `programs/`: all six frozen program arms, task/subgroup results, replacement
  accounting, D3 diagnostics and completed exploratory adjustment results.
- `result_dictionary.csv`: units, uncertainty fields and evidence identities
  for the ranking and program-nomination studies.
- `EVIDENCE_MAP.csv`: section/figure links to sources and reproduction entries.
- `ANALYSIS_IDENTITIES.csv`: populations, reference roles and inference scope.

Manuscript table CSVs are generated from the final main text and SI; their
numbering matches the current manuscript. `figure_code/PANEL_SOURCE_MAP.csv` supplies
panel-level provenance. Numerical files retain full precision; rounded values
in the text and figure labels are display choices.

## Quantities that remain distinct

A19-R evaluates complete pools of 52/59/63 candidates at K=10. PertRank-Q
uses held-target subpools of 26/26, 32/27 and 35/28 candidates at K=10; their
absolute precisions are not directly comparable. Three input sources share
one external knockout RNA experiment. The sci-Plex3 extension instead evaluates
six fixed programs in three cell lines of one drug-screen study, using both
culture directions. Cell lines, queries and input repetitions are not counted
as independent studies.

P@K measures reference-list overlap. Program response utility U is the mean
directed held-out response in input-control program-SD units. Whole-workflow
advantages against a native estimator or rvalues do not isolate geometry;
matched PertRank-versus-Euclidean contrasts identify its increment.

Original conditional query-block intervals condition on the supplied input,
reference and selected models. Program utility intervals condition on fixed
predictions, shallow inputs and observed cultures while resampling reference
cells. Program P@10 `reference_resampling_range` describes reference-list
re-ranking stability and is not a calibrated confidence interval. The legacy
program JSON archives retain their original field names; submission exports
correct their semantics without changing numerical values. Full mappings and
checks are in `programs/EXPORT_PROVENANCE.json` and `EXPORT_VERIFICATION.json`.

The earlier four-arm reference-draw ranges are sensitivity summaries rather
than biological-replicate confidence intervals. The D2 original-readout
geometry lower bound of approximately 4.34e-19 is numerical zero. Guide
cross-products and sign counts are descriptive; the 20 partitions reuse
retained cells. The 42 early cross-guide zero-weight entries are distinct from
the four original zero anchors inherited by PertRank-Q.

## Reproduction scope

The core and exploratory computations are documented in
`modules/new_results/README.md` at the repository root. They reconstruct results
from retained statistics, native outputs and reference projections.

The native rvalues fitting materials are included in
`modules/rvalues_native/`, with exact effect/SE inputs and the Python-to-R
entry point. The default rvalues scoring replay uses fixed native outputs;
refitting is a separate optional operation.
