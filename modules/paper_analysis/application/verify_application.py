#!/usr/bin/env python3
"""Check counts, target-specific covariance and tie outputs against frozen math."""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from make_example import fixture
from run_counts import fit_counts, run


def main():
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("frozen_engine", root/"verification/frozen_engine.py")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    data = fixture()
    x = data["X"].astype(float)
    ctrl = data["is_control"]
    scale = np.median(x[ctrl].sum(1))
    model = frozen.build_control_model(x[ctrl], x[ctrl].sum(1), data["gene_ids"], scale)
    fitted, metadata = fit_counts(data)
    errors = {"Euclidean": 0.0, "PertRank": 0.0, "effects": 0.0,
              "control_covariance_projector": 0.0, "direct_covariance_noise": 0.0}
    current = fitted["control_models"]["example_stratum"]
    errors["control_covariance_projector"] = float(np.max(np.abs(
        current["directions"]@current["directions"].T-model["U"]@model["U"].T)))
    for j, target in enumerate(fitted["target_ids"]):
        counts = x[data["target_ids"] == target]
        log = np.log1p(counts*scale/counts.sum(1)[:, None])
        # Five identical chunks are used only to satisfy frozen shape (5,20,G).
        # Each chunk's moments are computed independently; no inference is made.
        moments = frozen.effect_stats(np.tile(log, (5, 1)), model["cm"], model["U"], model["cv"], model["lam"])
        for arm, suffix in [("Euclidean", "E"), ("PertRank", "M")]:
            w = frozen.scalar(moments["a"+suffix], moments["nu"+suffix])[0]
            errors[arm] = max(errors[arm], float(abs(w-fitted["weights"][arm][j])))
        errors["effects"] = max(errors["effects"], float(np.max(abs(moments["y"][0]-fitted["effects"][j]))))
        logc = np.log1p(x[ctrl]*scale/x[ctrl].sum(1)[:, None]).astype(np.float32).astype(float)
        covariance = np.cov(log, rowvar=False)/20+np.cov(logc, rowvar=False)/2000
        U = current["directions"]
        M = np.eye(log.shape[1])-.9*U@U.T
        direct_noise = float(np.trace(M@covariance))
        compact_noise = fitted["noise_trace"][j]-.9*fitted["direction_noise"][j].sum()
        errors["direct_covariance_noise"] = max(errors["direct_covariance_noise"], abs(direct_noise-compact_noise))
    if max(errors.values()) > 2e-10:
        raise AssertionError(errors)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/"fixture.npz"
        np.savez_compressed(path, **data)
        subprocess.run([sys.executable, str(root/"run_counts.py"), str(path),
                        "--out", str(Path(tmp)/"output"), "--top-k", "3"],
                       check=True, capture_output=True, text=True)
        import csv
        rows = list(csv.DictReader((Path(tmp)/"output/all_candidate_scores.csv").open()))
        for method in ("Unweighted", "Euclidean", "PertRank"):
            for query in data["query_ids"]:
                for direction in ("up", "down"):
                    part = [r for r in rows if r["method"] == method and r["query_id"] == query and r["direction"] == direction]
                    assert abs(sum(float(r["tie_inclusion_probability"]) for r in part)-3) < 1e-12
                    assert all(r["target_gene_id"] != query for r in part)
        from pertrank.core import _boundary_probabilities
        assert np.allclose(_boundary_probabilities(np.array([3., 2., 0., 0., 0.]), 3),
                           np.array([1., 1., 1/3, 1/3, 1/3]), atol=0, rtol=1e-15)
        bad = dict(data, cell_ids=data["cell_ids"].copy())
        bad["cell_ids"][1] = bad["cell_ids"][0]
        try:
            fit_counts(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate-cell input accepted")
    audit = dict(status="PASS", version="v20.0", fixture="Synthetic raw integer counts only",
                 dimensions=list(x.shape), targets=8, target_cells_each=20, input_controls=2000,
                 frozen_engine_cell_level_parity=True, maximum_absolute_errors=errors,
                 full_covariance_check=True, cli_end_to_end=True,
                 own_gene_exclusion_and_probability_mass=True, duplicate_cells_rejected=True,
                 limitations="Software/algebra validation; not real-cell pipeline equivalence or biological performance validation")
    (root/"VALIDATION_v20.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
