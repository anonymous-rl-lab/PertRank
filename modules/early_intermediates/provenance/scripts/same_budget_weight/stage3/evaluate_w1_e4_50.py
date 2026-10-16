"""Stage 3, the E4 terciles 50-cell gatekeeper (amendments A4, A4a section 2, A4b section 4). Frozen with
stage3/read_e4_block1.py before that read produced anything (stage3/STAGE3_FREEZE_SHA256.txt).

W1 at 50 cells, same metric and same k = 5, beta = 0.9 as the 20-cell round, nothing re-estimated:
    u_i^T Y_j        = (proj_sum_block0 + proj_sum_block1) / 50 - proj_control
    u_i^T Sigma_j u_i = the variance of that projection over the target's own 50 cells
    Euclidean terms m2_50 and varbar_50 are taken from the v10 reduced state, so beta = 0 reproduces the frozen
    50-cell w0y exactly; the check below is absolute, for the reason recorded for the 20-cell round.

Decision (A4 section 3): the gatekeeper E4 terciles 50 passes unless its 95% interval lies entirely below zero.
E4 responsive 50 is reported as descriptive (A4b section 4): it neither adds nor changes a gatekeeper.
The 20-cell cells are re-reported and must match the previous round value for value.
The non-query-gene sensitivity is not computable at 50 cells: it needs both blocks' per-gene sums and the first
read kept only derived scalars. It is reported as not computable rather than approximated.

    python3 stage3/evaluate_w1_e4_50.py [--out stage3/results]
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
FROZEN = {"read_e4_w1.py", "evaluate_w1_e4.py", "read_e4_block1.py", "evaluate_w1_e4_50.py"}


def main():
    t0 = time.time()
    out_dir = es.arg("--out", os.path.join(HERE, "results"))
    d0 = os.path.join(HERE, "data", "e4_w1_stage3.npz")
    d1 = os.path.join(HERE, "data", "e4_w1_stage3_block1.npz")
    integ = {"plan_sha256": es.plan_sha(), "amendment_A2_sha256": es.amendment_sha()}
    for name in ("A3", "A4", "A4a", "A4b"):
        integ[f"amendment_{name}_sha256"] = checked_sha(os.path.join(BUNDLE, f"AMENDMENT_{name}.md"),
                                                       os.path.join(BUNDLE, f"AMENDMENT_{name}_SHA256.txt"),
                                                       f"amendment {name}")
    rec = {}
    for line in open(os.path.join(HERE, "STAGE3_FREEZE_SHA256.txt"), encoding="utf-8"):
        p = line.split()
        if len(p) == 2 and len(p[0]) == 64:
            rec[p[1]] = p[0]
    for rel in sorted(FROZEN):
        if rec.get(rel) != sha256(os.path.join(HERE, rel)):
            sys.exit(f"{rel} differs from the Stage 3 freeze record")
    integ["freeze"] = {k: rec[k] for k in sorted(FROZEN)}
    integ["data_sha256"] = {"block0": sha256(d0), "block1": sha256(d1)}

    Z = np.load(d0, allow_pickle=True); Z1 = np.load(d1, allow_pickle=True)
    if not np.array_equal(Z["names"], Z1["names"]):
        sys.exit("the two reads disagree on the target list")
    SRC = es.sources()["E4"]
    if [str(x) for x in Z["names"]] != SRC["targets"]:
        sys.exit("targets differ from the reduced state")
    D = es.load("e4_tcell.npz")
    cells = {(c["rule"], c["depth"]): c for c in SRC["cells"]}
    G = float(Z["n_genes"]); n_c = float(Z["n_control"]); trC = float(Z["control_trace"])
    lam_all = Z["eigenvalues"]; pc = Z["proj_control"]
    PS = Z["proj_sum"] + Z1["proj_sum"]; PQ = Z["proj_sumsq"] + Z1["proj_sumsq"]

    def w50(k, beta):
        B = 50.0
        pY = PS[:, :k] / B - pc[None, :k]
        sj = (PQ[:, :k] - B * (PS[:, :k] / B) ** 2) / (B - 1.0)
        a = D["m2_50"].astype(np.float64) - beta * (pY ** 2).sum(1) / G
        nu = (D["vb_50"].astype(np.float64) / B + trC / (G * n_c)
              - beta * (sj / B + lam_all[None, :k] / n_c).sum(1) / G)
        return np.clip(1.0 - nu / np.maximum(a, 1e-300), 0.0, 1.0)

    def score(c, w):
        return score_rows(w[None, :] * c["Y"], c["T"], c["sq"], key_pred=c["KP"], key_truth=c["KT"])["exp"]

    chk = {"block1_checks": str(Z1["checks"]),
           "varbar_30_vs_reduced": float(np.max(np.abs(Z1["varbar_30"] - D["vb_30"].astype(np.float64))
                                                / np.abs(D["vb_30"].astype(np.float64))))}
    c50t = cells[("terciles", "50")]
    chk["beta_zero_reproduces_w0y_50_abs"] = float(np.max(np.abs(w50(K, 0.0) - c50t["w0y"])))
    print("checks:", json.dumps({k: v for k, v in chk.items() if k != "block1_checks"}), flush=True)
    print("block 1 read checks:", chk["block1_checks"], flush=True)
    if chk["beta_zero_reproduces_w0y_50_abs"] > 1e-6 or chk["varbar_30_vs_reduced"] > 1e-3:
        sys.exit(f"the 50-cell baseline is not reproduced: {chk}")

    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ, "checks": chk,
           "setting": {"k": K, "beta": BETA, "genes": "all weight genes", "cells": 50}, "cells": {}, "sensitivity": {}}
    w = w50(K, BETA)
    for key, role in ((("terciles", "50"), "gatekeeper"), (("responsive", "50"), "descriptive")):
        c = cells[key]
        raw = score(c, np.ones(len(w))); base = score(c, c["w0y"]); sc = score(c, w)
        b = block_bootstrap(sc, base, c["blocks"]); br = block_bootstrap(sc, raw, c["blocks"])
        res["cells"][" ".join(key)] = {
            "role_this_round": role, "raw": float(raw.mean()), "w0y": float(base.mean()), "W1": float(sc.mean()),
            "W1_vs_w0y": {"diff": b["diff"], "ci95": b["ci95"], "p_one_sided": b["p_one_sided"]},
            "W1_vs_raw": {"diff": br["diff"], "ci95": br["ci95"]},
            "w0y_vs_raw": {"diff": block_bootstrap(base, raw, c["blocks"])["diff"]},
            "weights": {"mean": float(w.mean()), "zero_fraction": float((w == 0).mean()),
                        "spearman_with_w0y": float(spearmanr(w, c["w0y"]).correlation),
                        "degenerate": bool(np.all(w == 0) or np.ptp(w) == 0)}}
        r = res["cells"][" ".join(key)]
        print(f"{key[0]} 50 ({role}): raw {raw.mean():.4f}  w0y {base.mean():.4f}  W1 {sc.mean():.4f}   "
              f"W1 vs w0y {b['diff']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}]", flush=True)

    # the 20-cell cells, re-reported: must equal the previous round
    prev_path = os.path.join(HERE, "results", "stage3_w1_e4.json")
    w20 = ev.w1(Z, K, BETA)[0]
    for key in (("responsive", "20"), ("terciles", "20")):
        c = cells[key]
        b = block_bootstrap(score(c, w20), score(c, c["w0y"]), c["blocks"])
        res["cells"][" ".join(key) + " (re-reported)"] = {"W1_vs_w0y": {"diff": b["diff"], "ci95": b["ci95"]}}
    if os.path.isfile(prev_path):
        prev = json.load(open(prev_path, encoding="utf-8"))
        res["matches_previous_round"] = {
            k: bool(abs(res["cells"][k + " (re-reported)"]["W1_vs_w0y"]["diff"]
                        - prev["cells"][k]["W1_vs_w0y"]["diff"]) < 1e-12) for k in ("responsive 20", "terciles 20")}
        print(f"20-cell cells match the previous round: {res['matches_previous_round']}", flush=True)

    g = res["cells"]["terciles 50"]["W1_vs_w0y"]
    res["decision"] = {"gatekeeper": "E4 terciles 50", "diff": g["diff"], "ci95": g["ci95"],
                       "passes": bool(g["ci95"][1] >= 0),
                       "rule": "a gatekeeper fails only if its 95% interval lies entirely below zero",
                       "gatekeepers_still_not_computed": ["E1 20", "E1 50", "E5 responsive 20", "E5 responsive 50",
                                                          "E3 20", "E3 100"],
                       "note": "E4 responsive 50 is descriptive (A4b section 4) and does not enter any gatekeeper"}
    sens = {}
    c = cells[("terciles", "50")]; base = score(c, c["w0y"])
    for k in K_GRID:
        for beta in BETA_GRID:
            b = block_bootstrap(score(c, w50(k, beta)), base, c["blocks"])
            sens[f"k{k}_beta{beta}"] = {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]]}
    res["sensitivity"] = {"all_weight_genes_terciles_50": sens,
                          "non_query_genes": {"computable": False,
                                              "note": "needs both blocks' per-gene sums; the first read kept only "
                                                      "derived scalars, so this is not approximated"}}
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "stage3_w1_e4_50.json")
    json.dump(res, open(p, "w", encoding="utf-8"), indent=1)
    print(f"\ngatekeeper E4 terciles 50: {g['diff']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]  ->  "
          f"{'passes' if res['decision']['passes'] else 'FAILS'}")
    print(f"wrote {p} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
