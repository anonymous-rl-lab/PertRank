#!/usr/bin/env python3
"""Offline NTC inference replay from a float64 cross-Gram matrix.

This does not reconstruct raw UMI counts or gene-level half means. The stored
matrix is K=(A-mean(A))@(B-mean(B)).T/p, not a symmetric covariance matrix.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def replay(module_dir=None):
    root = Path(module_dir or Path(__file__).resolve().parent)
    original = root / "original_run"
    start = time.monotonic()
    provenance = json.loads((root / "COMPACT_PROVENANCE.json").read_text())
    assert sha(root / "inference_inputs.npz") == provenance["compact_sha256"]
    result = json.loads((original / "RESULT.json").read_text())
    freeze = json.loads((original / "RUN_FREEZE.json").read_text())
    verified = []
    for name, digest in freeze["file_sha256"].items():
        path = original / name
        if path.is_file():
            assert sha(path) == digest, ("original freeze mismatch", name)
            verified.append(name)
    for name, expected in provenance["original_sha256"].items():
        if (original / name).is_file():
            assert sha(original / name) == expected, ("source identity mismatch", name)
    assert result["new_replanning_not_interrupted_script_replay"] is True
    data = arrays(root / "inference_inputs.npz")
    selection = arrays(original / "selection.npz")
    expected = arrays(original / "resampling.npz")
    k = data["centered_cross_gram"]
    vals = data["target_pair_energy"]
    n, p, nt = 468, 13390, 64
    assert k.shape == (n, n) and vals.shape == (nt,)
    assert k.dtype == np.float64 and vals.dtype == np.float64
    assert np.isfinite(k).all() and np.isfinite(vals).all()
    assert int(data["n_genes"]) == p
    for name in ("ntc_guides", "target_names", "target_guides"):
        assert np.array_equal(data[name], selection[name]), name
    assert float(data["scale"]) == float(selection["scale"]) == 8261.5
    assert len(selection["keep"]) == p
    allcells = np.r_[selection["ntc_cells"].ravel(), selection["target_cells"].ravel()]
    assert len(allcells) == len(np.unique(allcells)) == 23840
    assert np.array_equal(np.sort(allcells), selection["cell_ids"])
    assert np.array_equal(selection["cell_ids"][selection["ntc_positions"]], selection["ntc_cells"])
    assert np.array_equal(selection["cell_ids"][selection["target_positions"]], selection["target_cells"])
    reps, seed = 2000, 20261008
    assert result["ntc"]["repetitions"] == result["target_same_gene_control"]["repetitions"] == reps
    assert result["ntc"]["seed"] == result["target_same_gene_control"]["seed"] == seed
    # Paired guide resampling is equivalent to count-weighting the cross-Gram.
    # Draw generation is identical to the frozen NumPy default_rng stream;
    # arithmetic below is batched and independent of stats_impl.infer_ntc.
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(reps, n))
    counts = np.zeros((reps, n), np.int64)
    np.add.at(counts, (np.arange(reps)[:, None], indices), 1)
    point = float(2 * np.trace(k) / (n - 1))
    boot = 2 * (counts @ np.diag(k) - np.einsum("ij,ij->i", counts @ k, counts) / n) / (n - 1)
    null = np.array([2 * k[np.arange(n), rng.permutation(n)].sum() / (n - 1) for _ in range(reps)])
    target_rng = np.random.default_rng(seed)
    target_boot = vals[target_rng.integers(0, nt, size=(reps, nt))].mean(axis=1)
    errors = {}

    def check(name, actual, archived):
        a, b = np.asarray(actual), np.asarray(archived)
        assert a.shape == b.shape, name
        errors[name] = float(np.max(np.abs(a - b)))
        assert np.allclose(a, b, rtol=1e-11, atol=1e-12), (name, errors[name])

    check("ntc_point", point, result["ntc"]["pair_energy"])
    check("ntc_centered_guide_energy", point / 2, result["ntc"]["centered_guide_energy"])
    check("all_2000_ntc_bootstrap_draws", boot, expected["ntc_bootstrap"])
    check("all_2000_ntc_permutation_draws", null, expected["ntc_permutation"])
    check("all_64_target_pair_energies", vals, expected["target_pair_energy"])
    check("all_2000_target_bootstrap_draws", target_boot, expected["target_bootstrap"])
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    null_ci = np.quantile(null, [0.025, 0.975]).tolist()
    target_ci = np.quantile(target_boot, [0.025, 0.975]).tolist()
    check("ntc_interval", ci, result["ntc"]["bootstrap_95_percentile"])
    check("ntc_null_interval", null_ci, result["ntc"]["permutation_null_95"])
    check("target_interval", target_ci, result["target_same_gene_control"]["bootstrap_95_percentile"])
    check("target_point", vals.mean(), result["target_same_gene_control"]["pair_energy_mean"])
    p_greater = (1 + int(np.sum(null >= point))) / (reps + 1)
    p_two = (1 + int(np.sum(abs(null) >= abs(point)))) / (reps + 1)
    assert p_greater == result["ntc"]["permutation_p_greater"]
    assert p_two == result["ntc"]["permutation_p_two_sided"]
    # Lane counts are rebuilt from exact selected-cell metadata, not read as a
    # final summary. Half balance does not remove between-guide composition.
    lanes, nl = selection["lane_codes"], len(selection["lane_categories"])
    def lane_counts(positions):
        return np.array([np.bincount(lanes[x], minlength=nl) for x in positions.reshape(-1, 20)]).reshape(*positions.shape[:-1], nl)
    nc = lane_counts(selection["ntc_positions"])
    tc = lane_counts(selection["target_positions"])
    lc = arrays(original / "lane_composition.npz")
    assert np.array_equal(nc, lc["ntc_counts"]) and np.array_equal(tc, lc["target_counts"])
    nh = np.abs(nc[:, 0] - nc[:, 1]).sum(axis=-1) / 40
    th = np.abs(tc[:, :, 0] - tc[:, :, 1]).sum(axis=-1) / 40
    tg = np.abs(tc[:, 0].sum(1) - tc[:, 1].sum(1)).sum(axis=-1) / 80
    check("ntc_half_lane_tv", nh, lc["ntc_half_tv"])
    check("target_half_lane_tv", th, lc["target_half_tv"])
    check("target_between_guide_lane_tv", tg, lc["target_between_guide_tv"])
    ngc = nc.sum(axis=1)
    expected_lane = ngc.sum(1)[:, None] * ngc.sum(0)[None, :] / ngc.sum()
    chi = np.divide((ngc - expected_lane) ** 2, expected_lane, out=np.zeros_like(expected_lane, float), where=expected_lane > 0).sum()
    cv = float(np.sqrt(chi / (ngc.sum() * min(len(ngc) - 1, int(np.sum(expected_lane.sum(0) > 0)) - 1))))
    check("ntc_guide_lane_cramers_v", cv, result["lane_composition"]["selected_ntc_guide_by_lane_cramers_v"])
    return {
        "status": "PASS", "mode": "float64 cross-Gram sufficient-statistic replay",
        "source_raw_counts_reconstructed": False, "gene_level_half_means_reconstructed": False,
        "network_reads": 0, "new_model_fits": 0, "new_experimental_selection": False,
        "ntc_guides": n, "target_pairs": nt, "genes": p, "selected_cells": len(allcells),
        "repetitions": reps, "seed": seed,
        "ntc": {"pair_energy": point, "centered_guide_energy": point / 2, "bootstrap_95_percentile": ci,
                "permutation_p_greater": p_greater, "permutation_p_two_sided": p_two, "permutation_null_95": null_ci},
        "target_same_gene_control": {"pair_energy_mean": float(vals.mean()), "bootstrap_95_percentile": target_ci},
        "lane_composition": {"ntc_half_lane_tv_max": float(nh.max()), "ntc_perfectly_balanced_guides": int((nh == 0).sum()),
                             "target_between_guide_lane_tv_median": float(np.median(tg)), "ntc_guide_lane_cramers_v": cv},
        "negative_estimates_preserved": True, "new_replanning_not_interrupted_script_replay": True,
        "interpretation_limits": result["interpretation_limits"],
        "verified_available_freeze_entries": verified, "errors": errors,
        "compact_sha256": sha(root / "inference_inputs.npz"), "seconds": time.monotonic() - start,
    }


def replay_to(output_dir: Path, module_dir=None) -> dict:
    """Integration entry point; write all generated output outside this module."""
    root = Path(module_dir or Path(__file__).resolve().parent).resolve()
    output = Path(output_dir).resolve()
    if output.is_relative_to(root):
        raise ValueError("Use an external output directory to preserve packaged evidence")
    report = replay(root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "replay.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--module-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    report = replay(args.module_dir)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        if args.output.resolve() in [p.resolve() for p in args.module_dir.rglob("*") if p.is_file() and p.name != "REPLAY_VERIFICATION.json"]:
            raise ValueError("Refuse to overwrite a packaged source or expected result")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
