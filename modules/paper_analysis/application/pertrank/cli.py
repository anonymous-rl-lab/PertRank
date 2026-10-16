"""File interface for fixed-parameter nomination without reference outcomes."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from . import __version__
from .core import ARMS, fit_cells, normalize_counts, nominate


def main():
    parser = argparse.ArgumentParser(description="PertRank: supplied input cells and queries to perturbation lists")
    parser.add_argument("input", help="Dense NPZ with X and identifier arrays; see README")
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--input-space", required=True, choices=("normalized", "counts"))
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026, help="Only for deterministic shortlist ties")
    args = parser.parse_args()
    t0 = time.perf_counter()
    with np.load(args.input, allow_pickle=False) as z:
        required = ("X", "gene_ids", "target_ids", "batch_ids", "is_control", "target_gene_ids", "query_ids")
        missing = set(required)-set(z.files)
        if missing:
            parser.error("Missing input fields: " + ", ".join(sorted(missing)))
        data = {key: z[key] for key in required}
        scales = None
        if args.input_space == "counts":
            if "library_totals" not in z:
                parser.error("Counts input requires library_totals from before gene filtering")
            data["X"], scales = normalize_counts(data["X"], z["library_totals"], data["batch_ids"], data["is_control"])
    queries = data.pop("query_ids")
    result = fit_cells(**data)
    lists, probabilities = nominate(result, queries, args.top_k, args.seed)
    out = Path(args.out)
    if out.exists():
        parser.error("Output directory already exists; choose a new path to preserve previous results")
    out.mkdir(parents=True)
    with (out/"nominations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(lists[0]))
        writer.writeheader()
        writer.writerows(lists)
    with (out/"weights.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target_id", "target_gene_id", "batch_id", "n_cells", *ARMS])
        for j, name in enumerate(result["target_ids"]):
            writer.writerow([name, result["target_gene_ids"][j], result["batch_ids"][j], int(result["n_cells"][j]),
                             *[float(result["weights"][arm][j]) for arm in ARMS]])
    arrays = {key: value for key, value in result.items() if key not in ("control_models", "weights")}
    arrays.update({"weight_"+arm.lower(): value for arm, value in result["weights"].items()})
    np.savez_compressed(out/"input_statistics.npz", **arrays)
    np.savez_compressed(out/"tie_probabilities.npz", probabilities=probabilities,
                        methods=np.array(ARMS), query_ids=queries, directions=np.array(["up", "down"]),
                        target_ids=result["target_ids"])
    controls = {}
    for i, (batch, model) in enumerate(result["control_models"].items()):
        np.savez_compressed(out/f"control_model_{i}.npz", **model, gene_ids=result["gene_ids"], batch_id=batch)
        controls[batch] = dict(file=f"control_model_{i}.npz", n_control=int(model["n_control"]),
                               leading_trace_share=float(model["eigenvalues"].sum()/model["variance"].sum()))
    report = dict(version=__version__, input_space=args.input_space, k=5, beta=0.9, top_k=args.top_k,
                  tie_seed=args.seed, targets=len(result["target_ids"]), queries=len(queries),
                  input_cells=len(data["X"]), genes=len(result["gene_ids"]), controls=controls,
                  normalization_scales=scales, elapsed_seconds=time.perf_counter()-t0,
                  query_selection="Supplied by caller; not performed by this interface",
                  evaluation="No reference outcomes read; no performance score or confidence interval produced",
                  shortlist_ties="SHA-256 ordering; inclusion probabilities saved separately",
                  candidate_rule="All supplied target cells; own-gene candidate excluded per query",
                  warning="Statistics describe selected input cells, not total acquisition or query-selection budget")
    (out/"run.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(out), targets=report["targets"], queries=report["queries"],
                          elapsed_seconds=round(report["elapsed_seconds"], 3))))


if __name__ == "__main__":
    main()
