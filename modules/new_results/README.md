# PertRank: two completed studies

This compact module reproduces the fixed native rvalues comparison and the predefined-program nomination studies from retained numerical inputs. It includes D3 development diagnostics, frozen sci-Plex3 evaluation, and the completed exploratory direction adjustments with their distinct evidence identities.

## Run

Use Python 3.12 and install the declared NumPy and pandas dependencies before an offline run. The output directory must be new and outside this module. Run the following commands from this module directory, or use the absolute path to run.py (for example, python /path/to/new_results/run.py core --out /new/core-output).

```bash
python run.py core --out /new/core-output
python run.py exploratory --out /new/exploratory-output
# Both scopes:
python run.py all --out /new/all-output
```

`python run.py verify` checks the delivered file hashes. Each study verifies its regenerated numerical results against retained expectations and array fingerprints. Reports and logs are written to the chosen output directory.

## What is recomputed

| Study | Starting information | Recomputed results |
|---|---|---|
| Fixed rvalues comparison | Original native membership probabilities, reference memberships, baseline query scores and paired resampling weights | Five-task nomination overlap, paired intervals, equal-state/source summaries and reported contrasts |
| Program nomination | Fixed programs, sufficient input statistics, original native ASH outputs and individual reference-cell projections | Reliability weights, readout-by-weight scores, candidate lists, response utility, P@10 and reference resampling |
| Program diagnostics | Retained directional inputs and fixed adjustment rules | D3 mechanism/permutation diagnostics and the three completed exploratory adjustment branches |

See SCIENTIFIC_SUMMARY_CN.md for the two study findings and their interpretation.

The programs retain all six fixed definitions, both response directions and the original source/replicate identities. The application criterion passed; ASH geometry transfer and the joint criterion did not. The exploratory rules are not presented as a successful replacement algorithm.

## Compression and refitting

Compression uses lossless encoding and shared numerical inputs. When original code uses input cells only through means and variances, those moments are retained in the original numerical precision and computation order. Individual reference-cell values needed for resampling remain available. Arrays are not rounded, resampling counts are not reduced, and unhelpful outcomes are not removed.

These commands recompute scores, lists, uncertainty summaries and completed diagnostics. They do not refit native R models or recover sequencing counts. The integrated code package preserves the full original rvalues fitting inputs and Python/R entry points at `modules/rvalues_native/`; the light module also retains the fixed fitting configuration and script copies for inspection. The study-specific READMEs describe program fitting and source-recovery scope.

## Integrated use

In the complete code package this directory is `modules/new_results/`. The root dispatcher exposes it through `new_results` for core results and `new_results_exploratory` for exploratory diagnostics. The original PertRank and PertRank-Q workflows remain available separately. This delivery does not publish a new GitHub or PyPI release.
