"""Stage 3 evaluation of candidate W1 on E5 KOLF, 20 and 50 cells (amendments A4, A4a, A4b, A4c). Frozen with
stage3/read_e5_w1.py before that read produced anything (stage3/STAGE3_FREEZE_SHA256.txt).

W1 with the target's own batch's directions (A4c section 2), k = 5 and beta = 0.9 unchanged:

    a_M,j     = m2_j - beta * sum_i (u_i^T Y_j)^2 / G
    nu_M,j(B) = varbar_j / B + tr(Sigma_C,b) / (G n_C) - beta * sum_i [u_i^T Sigma_j u_i / B + lambda_i,b / n_C] / G
    w_j       = [1 - nu_M,j / a_M,j]_+

with b the target's batch, u_i^T Y_j = proj_sum / B - proj_control[b], and u_i^T Sigma_j u_i the variance of that
projection over the target's own B cells. The Euclidean terms m2 and varbar come from the read, which checked them
against the reduced state's batch-matched m2_mfull and vb_full, so beta = 0 reproduces the frozen w0y; the check
below is absolute, for the reason recorded for E4.

Gatekeepers (A4c section 5): E5 responsive queries at 20 and at 50 cells, each passing unless its 95% interval lies
entirely below zero. The three tercile cells were descriptive in v10 and stay descriptive here.

    python3 stage3/evaluate_w1_e5.py [--data stage3/data/e5_w1_stage3.npz] [--out stage3/results]
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
FROZEN = {"read_e4_w1.py", "evaluate_w1_e4.py", "read_e4_block1.py", "evaluate_w1_e4_50.py",
          "read_e5_w1.py", "evaluate_w1_e5.py"}
GATE = (("responsive", "20"), ("responsive", "50"))


def main():
    t0 = time.time()
    data = es.arg("--data", os.path.join(HERE, "data", "e5_w1_stage3.npz"))
    out_dir = es.arg("--out", os.path.join(HERE, "results"))
    integ = {"plan_sha256": es.plan_sha(), "amendment_A2_sha256": es.amendment_sha()}
    for name in ("A3", "A4", "A4a", "A4b", "A4c"):
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
    if str(Z["amendment_A4c_sha256"]) != integ["amendment_A4c_sha256"]:
        sys.exit("the read was done under a different A4c")
    SRC = es.sources()["E5"]
    if [str(x) for x in Z["names"]] != SRC["targets"]:
        sys.exit("targets differ from the reduced state")
    print("integrity ok; read checks:", str(Z["checks"])[:300], flush=True)

    tb = Z["target_batch"].astype(int)
    G = float(Z["n_genes"]); n_c = Z["n_control_per_batch"].astype(float)
    U_lam = Z["eigenvalues"]; trC = Z["control_trace"]; pc = Z["proj_control"]

    def w1(depth, k, beta):
        B = float(depth)
        bi = 0 if depth == 20 else None
        if depth == 20:
            ps = Z["proj_sum"][0][:, :k]; pq = Z["proj_sumsq"][0][:, :k]
            m2, vb = Z["m2_20"], Z["varbar_20"]
        else:
            ps = (Z["proj_sum"][0] + Z["proj_sum"][1])[:, :k]
            pq = (Z["proj_sumsq"][0] + Z["proj_sumsq"][1])[:, :k]
            m2, vb = Z["m2_50"], Z["varbar_50"]
        pY = ps / B - pc[tb][:, :k]
        sj = (pq - B * (ps / B) ** 2) / (B - 1.0)
        a = m2 - beta * (pY ** 2).sum(1) / G
        nu = (vb / B + trC[tb] / (G * n_c[tb])
              - beta * (sj / B + U_lam[tb][:, :k] / n_c[tb][:, None]).sum(1) / G)
        return np.clip(1.0 - nu / np.maximum(a, 1e-300), 0.0, 1.0)

    cells = {(c["rule"], c["depth"]): c for c in SRC["cells"]}

    def score(c, w):
        return score_rows(w[None, :] * c["Y"], c["T"], c["sq"], key_pred=c["KP"], key_truth=c["KT"])["exp"]

    chk = {}
    for depth in (20, 50):
        c = cells[("responsive", str(depth))]
        chk[f"beta_zero_reproduces_w0y_{depth}_abs"] = float(np.max(np.abs(w1(depth, K, 0.0) - c["w0y"])))
    print("checks:", json.dumps(chk), flush=True)
    if max(chk.values()) > 1e-6:
        sys.exit(f"beta = 0 does not reproduce the frozen w0y: {chk}")

    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ, "checks": chk,
           "setting": {"k": K, "beta": BETA, "genes": "all 10,940 detected genes",
                       "directions": "the target's own batch's 2,000 input controls (A4c section 2)"},
           "cells": {}, "sensitivity": {}, "decision": {}}
    for (rule, depth), c in cells.items():
        if depth not in ("20", "50"):
            continue
        w = w1(int(depth), K, BETA)
        raw = score(c, np.ones(len(w))); base = score(c, c["w0y"]); sc = score(c, w)
        b = block_bootstrap(sc, base, c["blocks"]); br = block_bootstrap(sc, raw, c["blocks"])
        role = "gatekeeper" if (rule, depth) in GATE else "descriptive"
        res["cells"][f"{rule} {depth}"] = {
            "role": role, "raw": float(raw.mean()), "w0y": float(base.mean()), "W1": float(sc.mean()),
            "W1_vs_w0y": {"diff": b["diff"], "ci95": b["ci95"], "p_one_sided": b["p_one_sided"]},
            "W1_vs_raw": {"diff": br["diff"], "ci95": br["ci95"]},
            "w0y_vs_raw": {"diff": block_bootstrap(base, raw, c["blocks"])["diff"]},
            "weights": {"mean": float(w.mean()), "zero_fraction": float((w == 0).mean()),
                        "spearman_with_w0y": float(spearmanr(w, c["w0y"]).correlation),
                        "degenerate": bool(np.all(w == 0) or np.ptp(w) == 0)}}
        print(f"{rule} {depth} ({role}): raw {raw.mean():.4f}  w0y {base.mean():.4f}  W1 {sc.mean():.4f}   "
              f"W1 vs w0y {b['diff']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}]", flush=True)

    gates = {f"{r} {d}": res["cells"][f"{r} {d}"]["W1_vs_w0y"] for r, d in GATE}
    res["decision"] = {
        "gatekeepers": gates,
        "passes": {k: bool(v["ci95"][1] >= 0) for k, v in gates.items()},
        "all_pass": bool(all(v["ci95"][1] >= 0 for v in gates.values())),
        "degenerate": bool(any(res["cells"][f"{r} {d}"]["weights"]["degenerate"] for r, d in GATE)),
        "rule": "a gatekeeper fails only if its 95% interval lies entirely below zero",
        "gatekeepers_still_not_computed": ["E1 20", "E1 50", "E3 20", "E3 100"],
        "note": "E1 and E3 were unreadable when this ran; see A4c section 1"}

    c = cells[("responsive", "20")]; base = score(c, c["w0y"])
    grid = {}
    for k in K_GRID:
        for beta in BETA_GRID:
            b = block_bootstrap(score(c, w1(20, k, beta)), base, c["blocks"])
            grid[f"k{k}_beta{beta}"] = {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]]}
    res["sensitivity"]["all_genes_responsive_20"] = grid
    share = np.cumsum(U_lam, axis=1) / trC[:, None]
    res["sensitivity"]["control_trace_share_top20_by_batch"] = [round(float(x), 4) for x in share[:, -1]]

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "stage3_w1_e5.json")
    json.dump(res, open(p, "w", encoding="utf-8"), indent=1)
    print("\ngatekeepers:", json.dumps({k: [round(v["diff"], 4), [round(x, 4) for x in v["ci95"]]]
                                        for k, v in gates.items()}))
    print(f"all gatekeepers pass: {res['decision']['all_pass']}")
    print(f"wrote {p} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
