"""Curate the complete archived Replogle input cohort for software tutorials."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path, help="Archived e1_stage6.npz, not raw expression")
    a = ap.parse_args()
    wpath = next((ROOT / "pertrank_ecosystem/sources/recovered_w1").rglob("e1_w1_stage3.npz"))
    symbolpath = ROOT / "pertrank_ecosystem/sources/recovered/Pertrescue_Code_v9/code/repro_v10/same_budget_weight/stage3/data/tcell_gene_symbols.json"
    symbols = json.loads(symbolpath.read_text())["symbols"]
    # These author archives have legacy object-string arrays; the output fixture
    # is rewritten as ordinary Unicode arrays and needs no pickle support.
    with np.load(wpath, allow_pickle=True) as z:
        needed = ("synthetic", "targets", "checks", "n_genes", "n_control", "query_positions", "proj_sum", "proj_sumsq", "proj_control", "eigenvalues")
        w = {k: z[k] for k in needed}
    assert not bool(w["synthetic"])
    expected = json.loads(str(w["checks"]))["input_sha256"]["e1_stage6.npz"]
    assert digest(a.base) == expected, "Reduced input state differs from the archived W1 input lock"
    with np.load(a.base, allow_pickle=False) as z:
        # Only input statistics and identities are loaded. Evaluation rows,
        # scientific scores, bootstrap samples and truth lists remain unused.
        needed = ("genes", "targets", "cand", "qrows", "om20_q", "cin_q", "cvm", "ncin", "m2_20", "vb20")
        d = {k: z[k] for k in needed}
    targets = d["targets"][d["cand"]].astype(str)
    queries = d["genes"][d["qrows"]].astype(str)
    assert np.array_equal(targets, w["targets"].astype(str))
    assert np.array_equal(d["qrows"], w["query_positions"])
    g, nc, b = int(w["n_genes"]), int(w["n_control"]), 20
    assert g == len(d["genes"]) and nc == int(d["ncin"])
    effects = (d["om20_q"].astype(np.float64) - d["cin_q"].astype(np.float64)[:, None]).T
    energy = d["m2_20"] * g
    noise = (d["vb20"] / b + float(d["cvm"]) / nc) * g
    ps, pq = w["proj_sum"][0, :, :5], w["proj_sumsq"][0, :, :5]
    pe = (ps / b - w["proj_control"][:5]) ** 2
    pv = (pq - b * (ps / b) ** 2) / (b - 1) / b + w["eigenvalues"][:5] / nc
    sys.path.insert(0, str(ROOT / "paper_v21_1/release/PertRank_Paper_v21_1/application"))
    from pertrank.core import reliability_weights
    weights = reliability_weights(energy, noise, pe, pv)
    # Same exact expressions as the frozen evaluate_w1_e1.py, checked here as
    # software algebra without executing its scientific evaluation function.
    e0, v0 = d["m2_20"], d["vb20"] / b + float(d["cvm"]) / nc
    reference = {
        "Euclidean": np.clip(1 - v0 / np.maximum(e0, 1e-300), 0, 1),
        "PertRank": np.clip(1 - (v0 - .9 * pv.sum(1) / g) / np.maximum(e0 - .9 * pe.sum(1) / g, 1e-300), 0, 1),
    }
    errors = {k: float(np.max(abs(weights[k] - reference[k]))) for k in reference}
    assert max(errors.values()) < 1e-12
    chosen = np.array(sorted(queries)[:3])
    destination = OUT / "replogle_k562.npz"
    np.savez_compressed(
        destination, gene_ids=queries, gene_symbols=np.array([symbols.get(str(q), str(q)) for q in queries]),
        target_ids=targets, target_gene_ids=targets,
        target_symbols=np.array([symbols.get(str(t), str(t)) for t in targets]),
        effects=effects, energy=energy, noise_trace=noise,
        direction_energy=pe, direction_noise=pv,
        n_cells=np.full(len(targets), b), batch_ids=np.full(len(targets), "Replogle_K562_archived_cohort"),
        n_control=np.array(nc), weight_gene_count=np.array(g), selected_query_ids=chosen,
        frozen_formula_weight_Euclidean=reference["Euclidean"], frozen_formula_weight_PertRank=reference["PertRank"],
    )
    source_paths = {"reduced_input": a.base.resolve(), "projection_statistics": wpath, "symbol_annotation": symbolpath}
    report = {
        "dataset": "Replogle et al. 2022 K562 genome-wide Perturb-seq",
        "fixture_kind": "Authentic archived sufficient statistics and query-effect matrix; not cell-level AnnData",
        "candidate_selection": "Complete 2,000-candidate principal cohort from the archived input task; reference-screen targets excluded by the historical task rule",
        "query_selection": "All 300 historical reference-informed queries retained. Tutorial displays the first three unique query IDs in lexicographic order, without consulting outcomes.",
        "selected_query_ids": chosen.tolist(), "selected_query_symbols": [symbols.get(str(q), str(q)) for q in chosen],
        "n_target_cells": b, "n_control_cells": nc, "weight_gene_count": g, "effects_shape": list(effects.shape),
        "geometry_scope": "Energies and projected noise retain the full 8,202-gene input axis; query effects retain the original-coordinate 300-column panel.",
        "normalization": "Archived input mean minus archived input-control mean, after the original library-total log1p normalization; no new normalization or fit performed.",
        "units_conversion": {"energy": "8202 * m2_20", "noise_trace": "8202 * (vb20 / 20 + cvm / ncin)", "direction_energy": "(proj_sum[0]/20 - proj_control)^2, first five directions", "direction_noise": "(proj_sumsq[0] - 20*(proj_sum[0]/20)^2)/19/20 + eigenvalue/ncin, first five directions"},
        "source_study_doi": "10.25452/figshare.plus.20029387", "used_mirror": "https://zenodo.org/records/13350497",
        "symbol_annotation_note": "Human Ensembl-to-symbol annotations reused from the archived T-cell feature table; unmatched IDs remain unchanged.",
        "sources": {k: {"path": str(v.relative_to(ROOT)), "sha256": digest(v)} for k, v in source_paths.items()},
        "original_input_lock_sha256_verified": True,
        "fixture": {"file": destination.name, "sha256": digest(destination), "bytes": destination.stat().st_size},
        "frozen_formula_weight_max_absolute_error": errors,
        "new_model_fits": 0, "scientific_performance_evaluations": 0,
        "redistribution": "Derived numerical tutorial fixture from public-source data; retain source-study attribution. Source data are not relicensed by the software license.",
    }
    (OUT / "replogle_k562.provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"fixture": str(destination), "bytes": destination.stat().st_size, "parity": errors}))

if __name__ == "__main__":
    main()
