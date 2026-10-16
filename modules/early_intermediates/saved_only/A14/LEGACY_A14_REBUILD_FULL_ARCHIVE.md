# Legacy A14: optional recovery of the missing per-cell SCITO input

**Current status: `STORED_RESULT_HASH_ONLY`.** The frozen analysis script, protocol, panel mapping, original log and result are included. The approximately 3.08 GB per-cell H5AD is absent from this distribution. The packaging work did not download it or rerun A14. A checksum check of the archived JSON is not an input-to-result replay and is not reported as a pass.

This document records the recovery path supported by the saved protocol and source records. It is not a new experimental plan. A14 was a same-experiment protein recovery analysis on previously opened SCITO data; it is distinct from B8's cross-experiment protein comparison and from independent knockout RNA validation.

## Source identity

| Item | Archived information |
|---|---|
| Public deposit | https://figshare.com/articles/dataset/Processed_data/28578974 |
| DOI | https://doi.org/10.6084/m9.figshare.28578974 |
| Official file-list API | https://api.figshare.com/v2/articles/28578974 |
| Exact filename | `raw_contrastiveVI.h5ad` |
| Exact recorded file size | **3,083,988,566 bytes** |
| Recorded, historically verified MD5 | **`f5a38ca2d1dfd7fa2b779788c525072b`** |
| Original H5AD SHA-256 | **Unknown in the recovered records.** The original script embeds the MD5, not a SHA-256. Compute and retain SHA-256 when the file is recovered; do not invent an expected SHA-256. |
| Individual Figshare file ID | **Unknown in the recovered records.** Obtain it from the official file-list API by matching the exact filename, size and MD5. |
| Direct file download URL | **Unknown in the recovered records.** Use the `download_url` returned for that matching file. |
| H5AD dimensions | 3,672,215 cells × 210 features |
| Historical status source | `modules/early_experiments/repro_v10/same_budget_weight/stage3/external/scito/STATUS_after_A12a.md` |

The older SCITO directory README is a retrieval-time snapshot that predates successful file acquisition. The subsequent `STATUS_after_A12a.md` records the filename, size and verified MD5. The deposit's earlier **3,237,310,083-byte total** refers to record metadata, not this individual H5AD. Current endpoint availability was not checked during this packaging task.

The original authors' analysis repository is https://github.com/bjhwang113/SCITO-Perturb-seq, pinned in the archive at commit `e83a4e498236671f838310ab2a817b5e7fec5401`. The separately discussed `gene_zscore_mat.rds` was never recovered and is not the A14 input. The included `scito_gene_protein_z.npz` is an aggregate reference sufficient for B8/A13, but it lacks A14's per-cell partition information and cannot replace the H5AD.

## Exact selection and analysis rules

These rules are taken from the frozen `evaluate_scito_shallow_protein.py` and `AMENDMENT_A14.md`:

1. Read the deposited dense `X` values as supplied. They are already normalized and standardized by the source workflow; the script does not perform a new RNA normalization. Read feature labels from `var/x` and target labels/codes from `obs/guide_target`.
2. Clean antibody prefixes/barcode suffixes exactly as the frozen script specifies. Retain the union of `mapping` and `excluded` keys in `scito_panel_gene_map.json`: **201 protein features**, excluding seven isotype controls plus IgE and IgLightChainL. The mapping's 26 unmapped/excluded-for-gene-matching labels still belong to this 201-protein readout panel; they are not removed from the protein covariance analysis.
3. Pool target categories whose label starts with `NO-TARGET` as controls: the archived run had **27 categories and 32,965 cells**. The script applies no `Type` or `guide_robustness` filter.
4. Retain every non-control target with at least **220 cells**: **6,333 targets** in the archived run. There is no candidate subsampling.
5. For each target label `target`, use `int(sha256(("pertrescue-a14|" + target).encode()).hexdigest()[:16], 16)` as the NumPy `default_rng` seed. Permute that target's original-row-ordered cells and retain the first 220: **20 input**, **100 reference A**, **100 gate B** cells. The three target-cell blocks are disjoint.
6. Use all **201 proteins as queries**. Use the panel map to remove a query protein's own target from its candidate ranking when present; the original run excluded own targets for 57 proteins. The same NTC control mean is used for the three protein blocks.
7. Estimate control covariance and its top 20 directions from the NTC cells. Weights use **k = 5**, **beta = 0.9**, the shallow target moments and target projection variances. The first pass estimates controls; the second pass accumulates target moments, reading chunks of **200,000 cells**. No raw-data streaming occurs in the default repository replay.
8. Evaluate tie-expected **P@25** in both directions, with random tie keys from seeds **1401** and **1402**. Construct **30 query blocks** by `pr10.make_blocks(TA, 30)` and use **2,000 paired block-bootstrap resamples**, seed **2026**, percentile 95% intervals.
9. Before comparing nomination methods, require the 100-cell A/B reference overlap interval to lie above the random candidate rate `25 / (N_targets - 1)`. If this gate fails, the original script stops. Otherwise compare Unweighted, Euclidean weighting and PertRank and retain the original stopping rule. Do not change the panel, depth, k, beta or thresholds to obtain a favorable result.

## Files that anchor the reconstruction

Paths below are relative to `modules/early_experiments/repro_v10/same_budget_weight/`.

| File | SHA-256 |
|---|---|
| `AMENDMENT_A14.md` | `6d1b02256b48da7f45e0832f2d784a0ac750eb5df146f68ea875c314ed1bc5a8` |
| `stage3/evaluate_scito_shallow_protein.py` | `5301f4101f856e6d90b991dba54a0ec65c8d963be3497143ebbabac0ac740a65` |
| `stage3/external/scito/scito_panel_gene_map.json` | `7816bc252afe9531f5752648de4ad763468c3c5aecffa77940e434bf16d53a9f` |
| `stage3/results/scito_shallow_protein.json` | `c86caf672a8a58b5fbf734c1a57d3c9385c57201d7eeb04f9d28380c501a0a21` |

The freeze record is `stage3/A14_FREEZE_SHA256.txt`. The original runtime log is `stage3/logs/a14/evaluate_scito_shallow_protein.log` and records **72 seconds after the local file was available**. That is a historical execution time, not a new measurement or a download-time estimate. The archived run's peak RAM and complete machine specification are **unknown** in the recovered A14 records.

## Optional recovery commands, not executed in this release

Run from the repository root. Obtain metadata first; this command does not download the large file:

```bash
python modules/early_experiments/repro_v10/same_budget_weight/stage3/external/scito/download_figshare.py --out external_data/scito
```

After confirming the exact record entry, the same helper accepts `--file-id` with the ID actually returned by the API. It validates the file's size and supplied/computed MD5 and prints the recovered SHA-256. No file ID is hardcoded here because it was not retained in the recovered records.

Before running the analysis, independently check the recovered file. The frozen analysis script records a hardcoded source MD5 but does not itself hash the supplied H5AD, so this separate check is necessary:

```bash
python - <<'PYCODE'
from pathlib import Path
import hashlib
p = Path("external_data/scito/raw_contrastiveVI.h5ad")
assert p.stat().st_size == 3083988566, "Unexpected file size"
md5, sha = hashlib.md5(), hashlib.sha256()
with p.open("rb") as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
        md5.update(chunk)
        sha.update(chunk)
assert md5.hexdigest() == "f5a38ca2d1dfd7fa2b779788c525072b", "Unexpected MD5"
print("Recovered input SHA-256:", sha.hexdigest())
PYCODE
```

Install the HDF5 dependency needed specifically by this optional original input path, then execute the unchanged frozen script. The variables below are task-specific and resolve from the repository root:

```bash
python -m pip install numpy scipy h5py
export PR10_ROOT="$PWD/modules/early_experiments/repro_v10"
export PR10_DATA="$PR10_ROOT/data"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python "$PR10_ROOT/same_budget_weight/stage3/evaluate_scito_shallow_protein.py" \
  --h5ad "$PWD/external_data/scito/raw_contrastiveVI.h5ad" \
  --out "$PWD/outputs/legacy_a14_rebuilt"
```

Then compare against the archived output, excluding only creation timestamps as in the repository's historical verifier:

```bash
python - <<'PYCODE'
from pathlib import Path
import json, sys
sys.path.insert(0, "modules/early_experiments")
from reproduce_early import compare
expected = Path("modules/early_experiments/repro_v10/same_budget_weight/stage3/results/scito_shallow_protein.json")
actual = Path("outputs/legacy_a14_rebuilt/scito_shallow_protein.json")
diffs = compare(json.loads(actual.read_text()), json.loads(expected.read_text()), tol=1e-12)
if diffs:
    print("A14 RESULT_MISMATCH", diffs[:20])
    raise SystemExit(1)
print("A14 RECOMPUTED_MATCH: the optional input-to-result replay matched the archived result")
PYCODE
```

These commands are a documented recovery route, not evidence that the missing input has been recovered or the calculation rerun in this release. Reproduction remains `STORED_RESULT_HASH_ONLY` until the exact input identity is verified and an actual execution receipt is retained.

## Archived outcome, preserved without reinterpretation

The original result's PertRank minus Euclidean P@25 was **0.0007960**, with 95% interval **[−0.0007447, 0.0021251]**. Its original stopping decision was branch 2: no clear same-budget gain. The archived point-estimate order was Unweighted > PertRank > Euclidean. This recovery document neither changes that result nor proposes further A14 experiments.
