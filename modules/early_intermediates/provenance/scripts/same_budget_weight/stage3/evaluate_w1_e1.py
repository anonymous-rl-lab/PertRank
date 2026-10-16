"""Stage 3 evaluation of candidate W1 on E1 Replogle, 20 and 50 cells (amendments A4, A4a, A4b, A4d). Frozen with
stage3/read_e1_w1.py before that read produced anything (stage3/STAGE3_FREEZE_SHA256.txt).

W1, k = 5 and beta = 0.9 unchanged, directions from the 2,000 stage-6 input controls on the 8,202 shared genes:

    a_M,j     = m2_j - beta * sum_i (u_i^T Y_j)^2 / G
    nu_M,j(B) = varbar_j / B + cvm / n_C - beta * sum_i [u_i^T Sigma_j u_i / B + lambda_i / n_C] / G
    w_j       = [1 - nu_M,j / a_M,j]_+

with u_i^T Y_j = proj_sum / B - proj_control and u_i^T Sigma_j u_i the variance of that projection over the target's
own B cells. 20 cells use block 0 (the 20 input cells); 50 cells use blocks 0 and 1 together, the same cells w0y uses
in that cell. The Euclidean terms m2, varbar and cvm are the reduced state's, so beta = 0 reproduces the frozen w0y;
the check is absolute, for the reason recorded for E4.

Gatekeepers (A4 section 3, A4d section 4): E1 reference-variance queries at 20 and at 50 cells, W1 against w0y, each
passing unless its 95% interval lies entirely below zero. Failure condition: weights all zero or all equal.
Everything else printed is reported only (A4d section 5).

    python3 stage3/evaluate_w1_e1.py [--data stage3/data/e1_w1_stage3.npz] [--out stage3/results]
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(BUNDLE, "stage2"))
sys.path.insert(0, os.path.join(BUNDLE, "scripts"))
from _paths import checked_sha, sha256  # noqa: E402
import evaluate_stage2 as es  # noqa: E402
import evaluate_w1_e4 as ev  # noqa: E402
from pr10 import block_bootstrap, score_rows  # noqa: E402

K, BETA = ev.K_PRIMARY, ev.BETA_PRIMARY
K_GRID, BETA_GRID = ev.K_GRID, ev.BETA_GRID
FROZEN = {"read_e4_w1.py", "evaluate_w1_e4.py", "read_e1_w1.py", "evaluate_w1_e1.py"}
DEPTHS = ("20", "50")


def w1_weights(Z, D, depth, k, beta, idx=None, tag=""):
    """W1 weights at depth "20" (block 0) or "50" (blocks 0 + 1); idx selects stored directions (default the first
    k); tag "_nonquery" is the A4a sensitivity on the non-query genes, with its own Euclidean terms"""
    idx = np.arange(k) if idx is None else np.asarray(idx)
    B = float(depth)
    ps = Z["proj_sum" + tag]; pq = Z["proj_sumsq" + tag]
    ps, pq = (ps[0], pq[0]) if depth == "20" else (ps[0] + ps[1], pq[0] + pq[1])
    ps = ps[:, idx]; pq = pq[:, idx]
    n_c = float(D["ncin"])
    if tag:
        G = float(Z["n_genes_nonquery"]); m2 = Z[f"m2_{depth}_nonquery"]; vb = Z[f"varbar_{depth}_nonquery"]
        cv = float(Z["cvm_nonquery"]); lam = Z["eigenvalues_nonquery"][idx]; pc = Z["proj_control_nonquery"][idx]
    else:
        G = float(Z["n_genes"]); m2 = D[f"m2_{depth}"]; vb = D[f"vb{depth}"]; cv = float(D["cvm"])
        lam = Z["eigenvalues"][idx]; pc = Z["proj_control"][idx]
    pY = ps / B - pc[None, :]                                  # u_i^T Y_j
    sj = (pq - B * (ps / B) ** 2) / (B - 1.0)                  # u_i^T Sigma_j u_i from the target's own cells
    a = m2 - beta * (pY ** 2).sum(1) / G
    nu = vb / B + cv / n_c - beta * (sj / B + lam[None, :] / n_c).sum(1) / G
    return np.clip(1.0 - nu / np.maximum(a, 1e-300), 0.0, 1.0)


def main():
    t0 = time.time()
    data = es.arg("--data", os.path.join(HERE, "data", "e1_w1_stage3.npz"))
    out_dir = es.arg("--out", os.path.join(HERE, "results"))
    integ = {"plan_sha256": es.plan_sha(), "amendment_A2_sha256": es.amendment_sha()}
    for name in ("A3", "A4", "A4a", "A4b", "A4d"):
        integ[f"amendment_{name}_sha256"] = checked_sha(os.path.join(BUNDLE, f"AMENDMENT_{name}.md"),
                                                       os.path.join(BUNDLE, f"AMENDMENT_{name}_SHA256.txt"),
                                                       f"amendment {name}")
    rec = {}
    for line in open(os.path.join(HERE, "STAGE3_FREEZE_SHA256.txt"), encoding="utf-8"):
        p = line.split()
        if len(p) == 2 and len(p[0]) == 64:
            rec[p[1]] = p[0]
    for rel_name in sorted(FROZEN):
        if rec.get(rel_name) != sha256(os.path.join(HERE, rel_name)):
            sys.exit(f"{rel_name} differs from the Stage 3 freeze record")
    integ["freeze"] = {k: rec[k] for k in sorted(FROZEN)}
    integ["data_sha256"] = sha256(data)

    Z = np.load(data, allow_pickle=True)
    dry = "--dry-run" in sys.argv
    if bool(Z["synthetic"]) and not dry:
        sys.exit("this state was read in --synthetic mode (a code test); it can only be run with --dry-run")
    if str(Z["amendment_A4d_sha256"]) != integ["amendment_A4d_sha256"]:
        sys.exit("the read was done under a different A4d")
    if str(Z["script_sha256"]) != rec["read_e1_w1.py"] and not dry:
        sys.exit("the state was not written by the frozen read_e1_w1.py")
    if dry:
        integ["DRY_RUN"] = "code-path test on a synthetic state; every number in this file is meaningless"
    read_checks = json.loads(str(Z["checks"]))
    SRC = es.sources()["E1"]
    if [str(x) for x in Z["targets"]] != SRC["targets"]:
        sys.exit("targets differ from the reduced state")
    D = es.load("e1_stage6.npz")
    print(f"integrity ok; read values bit-identical to the stage-6 packages: {read_checks['all_values_bit_identical']}",
          flush=True)

    lam_all = Z["eigenvalues"]

    def w1(depth, k, beta, idx=None, tag=""):
        return w1_weights(Z, D, depth, k, beta, idx, tag)

    cells = {c["depth"]: c for c in SRC["cells"]}
    chk = {f"beta_zero_reproduces_w0y_{d}_abs": float(np.max(np.abs(w1(d, K, 0.0) - cells[d]["w0y"]))) for d in DEPTHS}
    print("checks:", json.dumps(chk), flush=True)
    if max(chk.values()) > 1e-6:
        sys.exit(f"beta = 0 does not reproduce the frozen w0y: {chk}")

    def score(c, w):
        return score_rows(w[None, :] * c["Y"], c["T"], c["sq"], key_pred=c["KP"], key_truth=c["KT"])["exp"]

    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ, "checks": chk,
           "read_checks": read_checks,
           "setting": {"k": K, "beta": BETA, "genes": "all 8,202 shared genes",
                       "directions": "the 2,000 stage-6 input controls (A4d section 2)",
                       "data_source": "Zenodo scPerturb mirror of the stage-6 file, MD5 13db594f8f1d2ccb88fec44a13e414dc"},
           "cells": {}, "sensitivity": {}, "decision": {}}
    base_scores = {}
    for d in DEPTHS:
        c = cells[d]
        w = w1(d, K, BETA)
        raw = score(c, np.ones(len(w))); base = score(c, c["w0y"]); sc = score(c, w)
        base_scores[d] = base
        b = block_bootstrap(sc, base, c["blocks"]); br = block_bootstrap(sc, raw, c["blocks"])
        res["cells"][f"reference-variance {d}"] = {
            "role": "gatekeeper", "raw": float(raw.mean()), "w0y": float(base.mean()), "W1": float(sc.mean()),
            "W1_vs_w0y": {"diff": b["diff"], "ci95": b["ci95"], "p_one_sided": b["p_one_sided"]},
            "W1_vs_raw": {"diff": br["diff"], "ci95": br["ci95"]},
            "w0y_vs_raw": {"diff": block_bootstrap(base, raw, c["blocks"])["diff"]},
            "weights": {"mean": float(w.mean()), "zero_fraction": float((w == 0).mean()),
                        "spearman_with_w0y": float(spearmanr(w, c["w0y"]).correlation),
                        "degenerate": bool(np.all(w == 0) or np.ptp(w) == 0)}}
        print(f"reference-variance {d} (gatekeeper): raw {raw.mean():.4f}  w0y {base.mean():.4f}  W1 {sc.mean():.4f}   "
              f"W1 vs w0y {b['diff']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}]", flush=True)

    gates = {f"E1 {d}": res["cells"][f"reference-variance {d}"]["W1_vs_w0y"] for d in DEPTHS}
    degenerate = any(res["cells"][f"reference-variance {d}"]["weights"]["degenerate"] for d in DEPTHS)
    res["decision"] = {
        "gatekeepers": gates,
        "passes": {k: bool(v["ci95"][1] >= 0) and not degenerate for k, v in gates.items()},
        "all_pass": bool(all(v["ci95"][1] >= 0 for v in gates.values()) and not degenerate),
        "degenerate": degenerate,
        "rule": "a gatekeeper fails only if its 95% interval lies entirely below zero; weights all zero or all equal fail"}

    # reported only (A4d section 5)
    for d in DEPTHS:
        c = cells[d]; base = base_scores[d]
        grid = {}
        for k in K_GRID:
            for beta in BETA_GRID:
                b = block_bootstrap(score(c, w1(d, k, beta)), base, c["blocks"])
                grid[f"k{k}_beta{beta}"] = {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]]}
        res["sensitivity"][f"grid_all_genes_{d}"] = grid
        b = block_bootstrap(score(c, w1(d, 15, BETA, idx=np.arange(5, 20))), base, c["blocks"])
        res["sensitivity"][f"directions_6_to_20_only_{d}"] = {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]]}
        b = block_bootstrap(score(c, w1(d, K, BETA, tag="_nonquery")), base, c["blocks"])
        res["sensitivity"][f"non_query_genes_{d}"] = {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]]}
    share = np.cumsum(lam_all) / float(Z["control_trace"])
    res["sensitivity"]["control_trace_share_top_k"] = {str(k): round(float(share[k - 1]), 4) for k in (1, 5, 10, 20)}

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "stage3_w1_e1.json" if not dry else "stage3_w1_e1_DRYRUN.json")
    json.dump(res, open(p, "w", encoding="utf-8"), indent=1)
    print("\ngatekeepers:", json.dumps({k: [round(v["diff"], 4), [round(x, 4) for x in v["ci95"]]] for k, v in gates.items()}))
    print(f"all E1 gatekeepers pass: {res['decision']['all_pass']}")
    print(f"wrote {p} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
