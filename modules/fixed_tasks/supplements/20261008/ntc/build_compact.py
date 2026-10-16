#!/usr/bin/env python3
"""Build the compact NTC inputs from the archived full half-mean checkpoint.

No rounding, downcasting, symmetrization, new selection, or model fit is used.
This script is optional; routine replay uses inference_inputs.npz directly.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source_dir, output_dir):
    source, output = Path(source_dir), Path(output_dir)
    original = output / "original_run"
    original.mkdir(parents=True, exist_ok=True)
    result = json.loads((source / "RESULT.json").read_text())
    assert result["status"] == "COMPLETE"
    assert result["half_means_sha256"] == sha(source / "half_means.npz")
    assert result["selection_sha256"] == sha(source / "selection.npz")
    assert result["freeze_sha256"] == sha(source / "RUN_FREEZE.json")
    with np.load(source / "half_means.npz", allow_pickle=False) as z:
        x = {k: z[k] for k in z.files}
    a, b = x["ntc"][:, 0], x["ntc"][:, 1]
    assert a.dtype == b.dtype == x["target"].dtype == np.float64
    assert a.shape == b.shape == (468, 13390)
    assert x["target"].shape == (64, 2, 2, 13390)
    ac, bc = a - a.mean(axis=0), b - b.mean(axis=0)
    gram = ac @ bc.T / a.shape[1]
    target = x["target"]
    target_vals = np.array([np.dot(target[i, 0, 0] - target[i, 1, 0], target[i, 0, 1] - target[i, 1, 1]) / a.shape[1] for i in range(len(target))])
    np.savez_compressed(output / "inference_inputs.npz", centered_cross_gram=gram,
                        target_pair_energy=target_vals, ntc_guides=x["ntc_guides"],
                        target_names=x["target_names"], target_guides=x["target_guides"],
                        n_genes=np.array(a.shape[1], np.int64), scale=x["scale"])
    names = ["RESULT.json", "PROTOCOL.json", "RUN_FREEZE.json", "RAW_READ_COMPLETE.json",
             "selection.npz", "selection_report.json", "ntc_selected_cells.json", "target_selected_cells.json",
             "resampling.npz", "lane_composition.npz", "lane_composition.json", "RUNTIME.json",
             "read_selected.py", "stats_impl.py", "evaluate.py", "prepare_metadata.py", "select_guides.py",
             "select_guides_portable.py", "freeze_run.py", "SMOKE_PASS.json", "SOURCE_SMOKE_INITIAL.json",
             "PORTABILITY_PATH_CHANGE.json", "RESUME_EVENT_01.json",
             "vendor/metadata_source.py", "vendor/row_transport.py", "vendor/hdf5_chunk_index.py",
             "metadata/hdf_objects.json", "metadata/metadata_report.json",
             "frozen_assets/SOURCE_PROVENANCE.json", "frozen_assets/A15_D3_cohort.npz",
             "frozen_assets/ntc_original/original_input_ntc_metadata.npz",
             "frozen_assets/ntc_original/selection_check.json", "frozen_assets/ntc_original/FULL_READ_PASS.json"]
    digests = {"half_means.npz": sha(source / "half_means.npz")}
    for name in names:
        dst = original / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, dst)
        digests[name] = sha(source / name)
        assert digests[name] == sha(dst)
    omitted = ["half_means.npz", "metadata/obs_metadata.npz", "metadata/indptr.npy",
               "metadata/selected_chunk_manifest.json", "frozen_assets/ntc_original/original_input_ntc_raw.npz"]
    raw_checks = source / "raw_batches"
    raw_summary = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(raw_checks.glob("batch_*.npz")) if ".tmp." not in p.name}
    provenance = {
        "schema": "pertrank.ntc.compact-cross-gram.v1", "status": "DERIVED_FROM_VERIFIED_COMPLETE_RUN",
        "precision": "float64 throughout; no downcast, rounding or symmetrization",
        "cross_gram_definition": "K_ij = <A_i - Abar, B_j - Bbar> / 13390; full nonsymmetric 468x468 matrix",
        "target_definition": "64 values: <A_guide1-A_guide2, B_guide1-B_guide2> / 13390, one per same-target pair",
        "primary_definition": "D = 2 trace(K)/(468-1)",
        "bootstrap_definition": "D_c = 2*(sum_i c_i*K_ii - c.T@K@c/468)/(468-1)",
        "permutation_definition": "D_perm = 2*sum_i K[i,perm(i)]/(468-1)",
        "sufficient_for": ["original primary point estimate", "all 2000 paired-guide bootstrap draws", "all 2000 B-spectrum permutations",
                           "original confidence intervals and p-values", "64 same-target pair energies and all 2000 target bootstrap draws"],
        "insufficient_for": ["recovering raw cell counts", "recovering gene-level half means", "changing gene axes or normalization",
                             "new gene-level diagnoses", "independently rechecking source UMI rows without external source"],
        "compact_sha256": sha(output / "inference_inputs.npz"),
        "original_sha256": digests,
        "omitted_large_artifacts": {name: {"bytes": (source/name).stat().st_size, "sha256": sha(source/name)} for name in omitted},
        "omitted_raw_batches": raw_summary,
        "source_identity": json.loads((source / "selection_report.json").read_text())["source_identity"],
        "historical_identity": "new metadata-frozen recovery selection, not interrupted old NTC selection or mid-run checkpoint",
        "inference_limits": result["interpretation_limits"],
    }
    (output / "COMPACT_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    result = build(args.source_dir, args.output_dir)
    print(json.dumps({"status": result["status"], "compact_sha256": result["compact_sha256"]}, indent=2))
