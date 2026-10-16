#!/usr/bin/env python3
"""Input counts to PertRank nominations, using the unchanged v17 cell engine.

This small adapter implements the fixed T-cell normalization convention:
full-library totals, a positive input-control gene axis, float32-stored log
controls and float64 target log values. All target cells are supplied inputs.
It does not read a deep reference, fit a query panel, or subsample a cohort.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from pertrank.core import ARMS, fit_cells, normalize_counts, nominate

VERSION = "v20.0"


def write_csv(path, rows):
    if not rows:
        raise ValueError("No output rows")
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_counts(data):
    required = {"X", "cell_ids", "gene_ids", "target_ids", "target_gene_ids",
                "batch_ids", "is_control", "query_ids"}
    if required - data.keys():
        raise ValueError("Missing arrays: " + ", ".join(sorted(required-data.keys())))
    counts = np.asarray(data["X"], dtype=float)
    if counts.ndim != 2 or not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("X must be a dense nonnegative integer raw-count matrix")
    n, g = counts.shape
    for name in ("cell_ids", "target_ids", "target_gene_ids", "batch_ids", "is_control"):
        if np.asarray(data[name]).shape != (n,):
            raise ValueError(f"{name} must align to X rows")
    cells = np.asarray(data["cell_ids"]).astype(str)
    if len(np.unique(cells)) != n or np.any(cells == ""):
        raise ValueError("cell_ids must be unique and nonempty")
    genes = np.asarray(data["gene_ids"]).astype(str)
    if genes.shape != (g,) or len(np.unique(genes)) != g:
        raise ValueError("gene_ids must be unique and align to X columns")
    batch = np.asarray(data["batch_ids"]).astype(str)
    if len(np.unique(batch)) != 1:
        raise ValueError("This T-cell adapter requires one prespecified matching stratum; use the generic v17 interface for a specified multi-stratum design")
    if not np.isin(data["is_control"], [False, True, 0, 1]).all():
        raise ValueError("is_control must contain only 0/1 or Boolean values")
    control = np.asarray(data["is_control"], bool)
    if control.sum() < 6:
        raise ValueError("At least six input controls are required to compute five directions")
    # Full-library counts are required in this adapter: no hidden denominator.
    totals = counts.sum(1)
    if "library_totals" in data and not np.array_equal(np.asarray(data["library_totals"], float), totals):
        raise ValueError("X must contain the full raw gene axis; supplied library_totals must equal its row sums")
    keep = np.flatnonzero(counts[control].sum(0) > 0)
    queries = np.asarray(data["query_ids"]).astype(str)
    if queries.ndim != 1 or not len(queries) or len(np.unique(queries)) != len(queries):
        raise ValueError("query_ids must be a nonempty unique one-dimensional list")
    missing = sorted(set(queries)-set(genes[keep]))
    if missing:
        raise ValueError("Queries absent from the input-control weight axis: " + ", ".join(missing[:10]))
    log, scales = normalize_counts(counts[:, keep], totals, batch, control)
    log[control] = log[control].astype(np.float32).astype(np.float64)
    result = fit_cells(log, genes[keep], data["target_ids"], batch,
                       control, data["target_gene_ids"])
    return result, {"normalization_scales": scales, "weight_gene_positions": keep.tolist(),
                    "full_gene_count": g, "weight_gene_count": len(keep),
                    "n_input_controls": int(control.sum()),
                    "n_input_target_cells": int((~control).sum()),
                    "target_depths": sorted(set(result["n_cells"].tolist())),
                    "normalization": "log1p(count * median full-library input-control total / full-library total)",
                    "control_precision": "float32 per log value, then float64 moments",
                    "target_precision": "float64"}


def candidate_rows(result, queries, probabilities, top_k):
    gmap = {g: i for i, g in enumerate(result["gene_ids"])}
    for qi, query in enumerate(queries):
        eligible = np.flatnonzero(result["target_gene_ids"] != query)
        for di, sign in enumerate((1, -1)):
            for ai, arm in enumerate(ARMS):
                score = sign * result["effects"][eligible, gmap[query]] * result["weights"][arm][eligible]
                ordered = np.sort(score)
                for local, target in enumerate(eligible):
                    first = len(score)-int(np.searchsorted(ordered, score[local], side="right"))+1
                    last = len(score)-int(np.searchsorted(ordered, score[local], side="left"))
                    yield {"method": arm, "query_id": str(query), "direction": "up" if sign > 0 else "down",
                           "target_id": str(result["target_ids"][target]),
                           "target_gene_id": str(result["target_gene_ids"][target]),
                           "original_effect": float(result["effects"][target, gmap[query]]),
                           "weight": float(result["weights"][arm][target]), "signed_score": float(score[local]),
                           "rank_first": first, "rank_last": last,
                           "tie_inclusion_probability": float(probabilities[ai, qi, di, target]), "top_k": top_k}


def run(input_path, output, top_k=25, seed=2026):
    t0 = time.perf_counter()
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new output directory to retain previous runs")
    with np.load(input_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    result, metadata = fit_counts(data)
    queries = np.asarray(data["query_ids"]).astype(str)
    metadata['input_format']='dense NPZ raw counts'
    return save_outputs(result,metadata,queries,input_path,output,top_k,seed,t0)


def save_outputs(result,metadata,queries,input_path,output,top_k=25,seed=2026,t0=None):
    """Write the same complete score/list schema for NPZ and AnnData inputs."""
    if t0 is None: t0=time.perf_counter()
    output=Path(output)
    nominations, probabilities = nominate(result, queries, top_k=top_k, seed=seed)
    output.mkdir(parents=True)
    write_csv(output/"nominations.csv", nominations)
    write_csv(output/"all_candidate_scores.csv", list(candidate_rows(result, queries, probabilities, top_k)))
    write_csv(output/"weights.csv", [dict(target_id=str(t), target_gene_id=str(result["target_gene_ids"][i]),
              n_input_cells=int(result["n_cells"][i]), **{a: float(result["weights"][a][i]) for a in ARMS})
              for i, t in enumerate(result["target_ids"])])
    arrays = {k: v for k, v in result.items() if k not in ("weights", "control_models")}
    arrays.update({"weight_"+a: v for a, v in result["weights"].items()})
    np.savez_compressed(output/"input_statistics.npz", **arrays)
    for i, (batch, model) in enumerate(result["control_models"].items()):
        np.savez_compressed(output/f"control_model_{i}.npz", **model, batch_id=batch, gene_ids=result["gene_ids"])
    sha=hashlib.sha256()
    with Path(input_path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): sha.update(chunk)
    report = dict(version=VERSION, input_sha256=sha.hexdigest(),
                  k=5, beta=0.9, top_k=top_k, tie_seed=seed, targets=len(result["target_ids"]),
                  query_ids=queries.tolist(), **metadata, elapsed_seconds=time.perf_counter()-t0,
                  input_selection="All supplied input cells; no sampling and no reference access",
                  scope="Nomination only; no performance, biological confidence or automatic adoption claim")
    (output/"run.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    print(json.dumps(run(args.input, args.out, args.top_k, args.seed), indent=2))
