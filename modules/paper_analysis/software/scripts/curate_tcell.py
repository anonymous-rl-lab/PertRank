"""Select archived input summaries; do not fit, read raw cells or evaluate references."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / "package_v140/total/PertRank_D1_D4_GitHub_Core/supplements/20261014_upgrade/native_rna"
SOURCES = {
    "moments": ROOT / "package_v130/release/PertRank_Code_v1.3.0/modules/native_rna/raw_inputs/D3_Rest/sufficient_state.npz",
    "predictions": BASE / "legacy/D3_Rest/predictions_locked.npz",
    "fr": BASE / "models/D3_Rest/FR/native/effects.npz",
    "symbols": ROOT / "pertrank_ecosystem/sources/recovered/Pertrescue_Code_v9/code/repro_v10/same_budget_weight/stage3/data/tcell_gene_symbols.json",
}

def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def main():
    s, p, f = (load(SOURCES[k]) for k in ("moments", "predictions", "fr"))
    symbols = json.loads(SOURCES["symbols"].read_text())["symbols"]
    assert np.array_equal(f["targets"], p["target_ids"])
    assert np.array_equal(f["genes"], p["query_ids"])
    b, g, nc = 20, int(s["G"]), int(s["n_control"])
    # Source means over the full gene axis -> sums expected by the public API.
    energy = s["m2"] * g
    noise = s["vb"] * g / b + float(s["trC"]) / nc
    ps, pq = s["Psum"][:, :5], s["Psumsq"][:, :5]
    pe = (ps / b - s["proj_control"][:5]) ** 2
    pv = (pq - ps * ps / b) / (b - 1) / b + s["lam"][:5] / nc
    # Software parity only: no reference or performance endpoint is read.
    sys.path.insert(0, str(ROOT / "paper_v21_1/release/PertRank_Paper_v21_1/application"))
    from pertrank.core import reliability_weights
    weights = reliability_weights(energy, noise, pe, pv)
    errors = {"Euclidean": float(np.max(abs(weights["Euclidean"] - p["w0y"]))),
              "PertRank": float(np.max(abs(weights["PertRank"] - p["W1"])))}
    assert max(errors.values()) < 1e-12
    chosen = np.array(sorted(p["query_ids"])[:3])
    destination = OUT / "tcell_d3_rest.npz"
    np.savez_compressed(
        destination,
        gene_ids=p["query_ids"], gene_symbols=np.array([symbols.get(str(k), str(k)) for k in p["query_ids"]]),
        target_ids=p["target_ids"], target_gene_ids=p["target_ids"], target_symbols=p["target_symbols"],
        effects=p["raw_effect"].T, energy=energy, noise_trace=noise,
        direction_energy=pe, direction_noise=pv, n_cells=np.full(len(energy), b),
        batch_ids=np.full(len(energy), "D3_Rest"), n_control=np.array(nc),
        weight_gene_count=np.array(g), selected_query_ids=chosen,
        archived_weight_Euclidean=p["w0y"], archived_weight_PertRank=p["W1"],
        external_fr_effects=f["effect"],
    )
    report = {
        "dataset": "Primary CD4+ T-cell atlas, donor D3, resting state",
        "fixture_kind": "Authentic archived sufficient statistics and query-effect matrix; not cell-level AnnData",
        "candidate_selection": "All 52 candidates in the completed independent-knockout-RNA comparison; no additional candidate filtering",
        "query_selection": "All 293 archived query columns retained. Tutorial displays first three unique query IDs in lexicographic order, without consulting outcomes.",
        "selected_query_ids": chosen.tolist(),
        "selected_query_symbols": [symbols[str(k)] for k in chosen],
        "n_target_cells": b, "n_control_cells": nc, "weight_gene_count": g,
        "effects_shape": list(p["raw_effect"].T.shape),
        "geometry_scope": "Energies and projected noise are from the full 13,390-gene archived input axis; displayed effect columns are the original 293 queries.",
        "units_conversion": {"energy": "G * m2", "noise_trace": "G * vb / 20 + trC / n_control", "direction_energy": "(Psum/20 - proj_control)^2, first 5 directions", "direction_noise": "(Psumsq - Psum^2/20)/19/20 + eigenvalue/n_control, first 5 directions"},
        "source_study_doi": "10.1016/j.cell.2026.08.002",
        "source_release": "genome-scale-tcell-perturb-seq/marson2025_data",
        "sources": {k: {"path": str(v.relative_to(ROOT)), "sha256": digest(v)} for k, v in SOURCES.items()},
        "fixture": {"file": destination.name, "sha256": digest(destination), "bytes": destination.stat().st_size},
        "archived_weight_max_absolute_error": errors,
        "new_model_fits": 0, "reference_matrices_read": 0, "scientific_performance_evaluations": 0,
        "redistribution": "Derived numerical tutorial fixture from public-source data; retain source-study attribution. Source data are not relicensed by the software license.",
    }
    (OUT / "tcell_d3_rest.provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"fixture": str(destination), "bytes": destination.stat().st_size, "parity": errors}))

if __name__ == "__main__":
    main()
