# PertRank v1.5.4: reproduction from retained intermediates

This release retains the numerical intermediates needed to recalculate the
reported comparisons, together with original statistical code, query/group
definitions and provenance. The compact modules start at the layers shown
below. Their receipts describe the calculations actually executed.

`PAPER_COVERAGE.csv` maps 248 manuscript/SI records: 139 sections, 97 displayed
tables, 11 figures and one R20 coverage entry. This is a location and scope
index, not a count of 248 independently executed tests. All displayed tables
and figure source paths are retained. Numerical replay, figure rendering and
historical records are identified separately.

| Module | Current entry point | Actual computation | Retained inputs | Removed upstream inputs/steps |
|---|---|---|---|---|
| `early_intermediates` | `python modules/early_intermediates/reproduce.py --output-dir NEW` | Captured means and paired query-block intervals; 14-setting six-arm factorial contrasts and target-statistic summaries | Per-query/direction scores, blocks, original seeds, selected weight/energy diagnostics, original code and result oracles | Raw expression, full effect/moment states, weight estimation and candidate ranking; uncaptured original fields remain archived |
| `corrected_intermediates` | `python modules/corrected_intermediates/reproduce.py --output-dir NEW` | Guide score/group inference and lane calibration; A18-R strict membership support/turnover; historical CELLEQ-R interpolation; A19-R fixed-effect rankings and reference-membership sensitivities | Query scores/blocks, selected diagnostic vectors, lane-count summaries, memberships, depth scores; A19-R reuses the existing `fixed_tasks` effects/weights/reference memberships | Raw counts, control fitting and new weight estimation; external reference draws are retained memberships rather than regenerated raw-sample draws |
| `fixed_tasks` | `python modules/fixed_tasks/reproduce.py --output-dir NEW` | Existing task-specific weight/ranking calculations, paired intervals, depth curves, native-method comparisons and interactions | Task moments, effects, memberships, scores and reference-draw representations used by the supported entry points | Source-atlas preprocessing and new native-model fitting |
| `paper_analysis` | `python modules/paper_analysis/reproduce.py --out NEW` | S3 decomposition, strata/subset rankings, panel transfer, mixture checks, and all scientific figures | Existing analysis arrays, gene effects, fixed task inputs and paper source data | New experiments; figures use their preserved source values |
| `sensitivity_replay` | `python modules/sensitivity_replay/reproduce.py --output-dir NEW` | Saved-draw empirical summaries, Mixscale list comparisons and paired intervals | Original 200-draw arrays and fitted-output intermediates | Fresh cell resampling/refitting and native Mixscale fitting |
| `r20_intermediates` | `python modules/r20_intermediates/reproduce.py --output-dir NEW` | Means, differences and conditional intervals for five responsive/five tercile inputs and two external RNA inputs per donor; original seed 20261006 and 2,000 draws | Original per-query method scores, query IDs/blocks, unchanged statistical code | Cells, moments, predictions, weight fitting, candidate ranking and nomination redistribution |
| `il2_intermediates` | `python modules/il2_intermediates/reproduce.py --output-dir NEW` | Control centering, target medians, fixed-list contributions and 20,000 paired guide/control bootstrap draws; MAGeCK secondary list scoring | Candidate/control guide readouts, fixed eligibility, candidate mappings/lists and candidate-projected fitted MAGeCK tables | Raw read-count preparation, sequence-overlap recomputation, MAGeCK build/source and refitting |

The corrected module also supports its component entry points
`guide/reproduce.py`, `a18r/reproduce.py`, `celleq/reproduce.py` and
`a19r/reproduce.py`, each with `--output-dir NEW`. A19-R uses the sibling
`fixed_tasks` data by default; `--fixed-root PATH` selects an explicit existing
location. Each component's README and receipt name the particular arrays and
statistics checked. CELLEQ-R retains its historical identity and is separate
from the current seven-depth calibration in `fixed_tasks`.

## Paper displays and numerical evidence

All manuscript table CSVs remain under
`modules/paper_analysis/source_data/manuscript_tables/`. Main figures use
`modules/paper_analysis/figure_code/draw_main_figures.py` and the retained
`source_data/figures/` values. Supplementary figures use the existing
`plot_supplement.py` and `plot_design_attenuation.py` routes. Rendering these
displays does not substitute for the linked module's numerical checks. The
default run does not automatically compare every cell of all 97 display CSVs
against fresh numerical outputs. Where a displayed value is explicitly checked,
the module receipt and code specify that comparison.

`source_data/manuscript_tables/PertRank_Supplement_v25.2_table_060.csv` under
`modules/paper_analysis/` preserves manuscript Table SN9.1, including historical
full-state and raw-CSR verification. That frozen table records earlier work;
current executable starting points are the module layers listed above. Its
directory README and index distinguish those scopes.

Per-query arrays, fixed memberships and guide readouts are calculation inputs;
the expected-result JSONs are comparison targets. Original full-result
reports may include additional fields beyond a compact entry point's scope.
Such fields retain their archival identity. Mathematical arguments, data-source
descriptions and documented unavailable analyses are records rather than
new numerical results. The CSV identifies these rows explicitly.

A14 remains archive-only because its original per-cell protein input is missing.
It is still reported in manuscript SN8.3, is outside the default numerical
replay, and is not validated by a successful early-module run. Original A19
cell selection remains unrecovered; A19-R is the separately identified
reconstruction. Original A9/A10 full-state validation records are historical
provenance, not extra computations claimed by the compact early entry point.

The compression preserves outcome direction and analysis identity. Historical
D1 panels remain distinct from later corrected-panel tasks. A18-R, A19-R and
CELLEQ-R remain labelled reconstructions. Source donors sharing one external
RNA experiment remain three inputs to that experiment. IL2's null result is
included in its default replay. Conditional intervals retain their original
conditioning and are not converted into population or raw-data uncertainty.

## Running and checking

Use the repository dispatcher and `MODULES.json` for the current default set.
Run each module into a fresh output directory and inspect its native receipt.
The R20 receipt explicitly states `per_query_scores_to_means_contrasts_intervals`;
the IL2 and corrected receipts state their readout/membership replay layers.
Historical v1.5.0 moments/raw-input validation records, where retained, are
provenance rather than the claimed starting point of the current compact run.

The scientific runtime retains its module-specific pinned dependencies.
The optional AnnData software tests use their separately documented environment.
No new biological experiment or model fitting is needed for the compact
statistical routes above.
