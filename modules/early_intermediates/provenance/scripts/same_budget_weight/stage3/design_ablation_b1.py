"""Amendment A6 section 3: design ablation B1. Frozen before it produced anything (stage3/A6_FREEZE_SHA256.txt).

Question: should the control directions change only how much a perturbation is trusted (W1), or also what its
effect is? With A = M^(1/2) = I - c sum_{i<=k} u_i u_i^T, c = 1 - sqrt(1 - beta) (0.683772 at beta = 0.9, k = 5),
the transformed spectrum on the query rows is

    (AY)_q = Y_q - c U_q (U^T Y),

with U^T Y the per-target projection W1 already uses (proj_sum / B - proj_control; E5 per batch). Six arms, the
3 x 2 factorial weight {none, w0y, w_M} x output {Y, AY}:

    raw = Y      w0y = w0y Y      W1 = w_M Y (the product)
    AY           w0y_AY = w0y AY  wM_AY = w_M AY (w0y computed and output in A coordinates)

Cells, both depths per source: E4 D3 responsive and terciles (20, 50), D1 responsive and terciles (20, 50; as
executed, then corrected per A6a), E5 responsive (20, 50), E1 (20, 50), E3 (20, 100). raw, w0y and W1 must
reproduce the frozen results to 1e-12 (A6 section 2). Reading on E4 D3 responsive 20 (A6 section 3), with D1
responsive 20 reported beside it.

    python3 stage3/design_ablation_b1.py [--out stage3/results]
"""
import json
import os
import sys
import time

import numpy as np

import a6_common as A
from a6_common import BETA, C_A, K, block_bootstrap, exp_scores, fmt, summ
import evaluate_stage2 as es  # noqa: E402
import evaluate_w1_e1 as e1m  # noqa: E402
import evaluate_w1_e3 as e3m  # noqa: E402
import evaluate_w1_e4 as ev  # noqa: E402

SCRIPTS = ["design_ablation_b1.py"]
ARMS = ("raw", "w0y", "W1", "AY", "w0y_AY", "wM_AY")
COMPARE = (("AY", "W1"), ("wM_AY", "W1"), ("AY", "raw"), ("wM_AY", "AY"), ("w0y_AY", "w0y"), ("W1", "w0y"))
DATA = os.path.join(A.HERE, "data")


def arms(Y, corr, w0y, wM):
    AY = Y - corr
    return {"raw": Y, "w0y": w0y[None, :] * Y, "W1": wM[None, :] * Y,
            "AY": AY, "w0y_AY": w0y[None, :] * AY, "wM_AY": wM[None, :] * AY}


def run_cell(label, c, corr, w0y, wM, stored, fails):
    E = arms(c["Y"], corr, w0y, wM)
    S = {a: exp_scores(E[a], c) for a in ARMS}
    row = {"means": {a: float(S[a].mean()) for a in ARMS}}
    if stored is not None:
        row["reproduction_max_abs_diff"] = A.check_stored(label, {"raw": row["means"]["raw"], "w0y": row["means"]["w0y"],
                                                                  "W1": row["means"]["W1"]}, stored, fails)
    row["comparisons"] = {f"{a} - {b}": summ(block_bootstrap(S[a], S[b], c["blocks"])) for a, b in COMPARE}
    y2 = (c["Y"] ** 2).sum(0); a2 = ((c["Y"] - corr) ** 2).sum(0)
    row["query_row_energy_removed_by_A"] = {"median_over_targets": float(np.median(1.0 - a2 / np.maximum(y2, 1e-300)))}
    cm = row["comparisons"]
    print(f"{label:34s} raw {row['means']['raw']:.4f} w0y {row['means']['w0y']:.4f} W1 {row['means']['W1']:.4f} "
          f"AY {row['means']['AY']:.4f} w0y_AY {row['means']['w0y_AY']:.4f} wM_AY {row['means']['wM_AY']:.4f} | "
          f"AY-W1 {fmt(cm['AY - W1'])} wM_AY-W1 {fmt(cm['wM_AY - W1'])}", flush=True)
    return row


def reading(row):
    ay, wm = row["comparisons"]["AY - W1"], row["comparisons"]["wM_AY - W1"]
    if (ay["diff"] > 0 and ay["ci95"][0] > 0) or (wm["diff"] > 0 and wm["ci95"][0] > 0):
        return "1: correcting the effect spectrum adds value on these data (product unchanged)"
    if ay["ci95"][1] < 0 and wm["ci95"][1] < 0:
        return ("2: on these data, using the control directions to decide reliability works better than using "
                "them to rewrite the effect")
    return "3: the two uses cannot be told apart on this cell"


def main():
    t0 = time.time()
    out_dir = A.arg("--out", os.path.join(A.HERE, "results"))
    integ = A.integrity(SCRIPTS)
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ,
           "setting": {"k": K, "beta": BETA, "c": float(C_A), "arms": list(ARMS)}, "cells": {}, "reading": {}}
    fails = []
    print(f"A = I - {C_A:.6f} U U^T on the first {K} directions", flush=True)

    # ---------------------------------------------------------------- E4, T cell donor D3
    Z = np.load(os.path.join(DATA, "e4_w1_stage3.npz"), allow_pickle=True)
    Z1 = np.load(os.path.join(DATA, "e4_w1_stage3_block1.npz"), allow_pickle=True)
    D = es.load("e4_tcell.npz")
    if not np.array_equal(Z["keep"], D["keep"]) or not np.array_equal(Z["query_positions"], D["qu"]):
        sys.exit("E4: the stage-3 read and the reduced state are on different gene axes")
    cells = {(c["rule"], c["depth"]): c for c in es.sources()["E4"]["cells"]}
    U = Z["directions"][:, :K]; pc = Z["proj_control"][:K]
    PS50 = Z["proj_sum"] + Z1["proj_sum"]; PQ50 = Z["proj_sumsq"] + Z1["proj_sumsq"]
    pY = {"20": Z["proj_sum"][:, :K] / 20.0 - pc, "50": PS50[:, :K] / 50.0 - pc}
    G = float(Z["n_genes"]); n_c = float(Z["n_control"]); trC = float(Z["control_trace"]); lam = Z["eigenvalues"]

    def w50():   # stage3/evaluate_w1_e4_50.py, w50(K, BETA)
        B = 50.0
        sj = (PQ50[:, :K] - B * (PS50[:, :K] / B) ** 2) / (B - 1.0)
        a = D["m2_50"].astype(np.float64) - BETA * (pY["50"] ** 2).sum(1) / G
        nu = D["vb_50"].astype(np.float64) / B + trC / (G * n_c) - BETA * (sj / B + lam[None, :K] / n_c).sum(1) / G
        return np.clip(1.0 - nu / np.maximum(a, 1e-300), 0.0, 1.0)

    wM = {"20": ev.w1(Z, K, BETA)[0], "50": w50()}
    st = {"20": A.results("stage3_w1_e4.json")["cells"], "50": A.results("stage3_w1_e4_50.json")["cells"]}
    for rule, q in (("responsive", D["q_resp"]), ("terciles", D["q_terc"])):
        for depth in ("20", "50"):
            c = cells[(rule, depth)]
            corr = C_A * (U[q] @ pY[depth].T)
            res["cells"][f"E4 D3 {rule} {depth}"] = run_cell(f"E4 D3 {rule} {depth}", c, corr, c["w0y"], wM[depth],
                                                             st[depth][f"{rule} {depth}"], fails)

    # ---------------------------------------------------------------- D1, as executed and corrected (A6a)
    Zd = A.d1_load(); Ud = Zd["directions"][:, :K]
    stored_d1 = A.results("stage3_d1_confirm.json")["cells"]
    for setting in ("as executed", "corrected"):
        for (rule, depth), c in A.d1_cells(Zd, setting).items():
            corr = C_A * (Ud[c["q"]] @ A.d1_pY(Zd, int(depth)).T)
            lab = f"D1 {rule} {depth} ({setting})"
            res["cells"][lab] = run_cell(lab, c, corr, c["w0y"], c["W1"],
                                         stored_d1[f"{rule} {depth}"] if setting == "as executed" else None, fails)

    # ---------------------------------------------------------------- E5 KOLF, batch-matched directions
    Z = np.load(os.path.join(DATA, "e5_w1_stage3.npz"), allow_pickle=True)
    D = es.load("e5_kolf.npz")
    if not np.array_equal(Z["query_positions"], D["qu"]):
        sys.exit("E5: query positions differ from the reduced state")
    cells = {(c["rule"], c["depth"]): c for c in es.sources()["E5"]["cells"]}
    tb = Z["target_batch"].astype(int); G = float(Z["n_genes"]); n_cb = Z["n_control_per_batch"].astype(float)
    lamb = Z["eigenvalues"]; trCb = Z["control_trace"]; pcb = Z["proj_control"]

    def e5(depth):   # stage3/evaluate_w1_e5.py, w1(depth, K, BETA), and the projections it uses
        B = float(depth)
        if depth == 20:
            ps = Z["proj_sum"][0][:, :K]; pq = Z["proj_sumsq"][0][:, :K]; m2, vb = Z["m2_20"], Z["varbar_20"]
        else:
            ps = (Z["proj_sum"][0] + Z["proj_sum"][1])[:, :K]; pq = (Z["proj_sumsq"][0] + Z["proj_sumsq"][1])[:, :K]
            m2, vb = Z["m2_50"], Z["varbar_50"]
        py = ps / B - pcb[tb][:, :K]
        sj = (pq - B * (ps / B) ** 2) / (B - 1.0)
        a = m2 - BETA * (py ** 2).sum(1) / G
        nu = vb / B + trCb[tb] / (G * n_cb[tb]) - BETA * (sj / B + lamb[tb][:, :K] / n_cb[tb][:, None]).sum(1) / G
        return np.clip(1.0 - nu / np.maximum(a, 1e-300), 0.0, 1.0), py

    st = A.results("stage3_w1_e5.json")["cells"]
    for depth in ("20", "50"):
        c = cells[("responsive", depth)]; w, py = e5(int(depth)); q = D["q_resp"]
        corr = np.zeros_like(c["Y"])
        for b in np.unique(tb):
            j = np.flatnonzero(tb == b)
            corr[:, j] = C_A * (Z["directions"][b][q][:, :K] @ py[j].T)
        res["cells"][f"E5 KOLF responsive {depth}"] = run_cell(f"E5 KOLF responsive {depth}", c, corr, c["w0y"], w,
                                                                st[f"responsive {depth}"], fails)

    # ---------------------------------------------------------------- E1 Replogle and E3 Orion
    for src, fn, mod, depths, key in (("E1", "e1_w1_stage3.npz", e1m, ("20", "50"), "reference-variance"),
                                      ("E3", "e3_w1_stage3.npz", e3m, ("20", "100"), "terciles")):
        Z = np.load(os.path.join(DATA, fn), allow_pickle=True)
        D = es.load("e1_stage6.npz" if src == "E1" else "e3_orion.npz")
        qrows = D["qrows"] if src == "E1" else D["fz_qrows"]
        if not np.array_equal(Z["query_positions"], qrows):
            sys.exit(f"{src}: query positions differ from the reduced state")
        cells = {c["depth"]: c for c in es.sources()[src]["cells"]}
        st = A.results(f"stage3_w1_{src.lower()}.json")["cells"]
        Uq = Z["directions"][:, :K][qrows]; pc = Z["proj_control"][:K]
        for depth in depths:
            if src == "E1":
                ps = Z["proj_sum"][0] if depth == "20" else Z["proj_sum"][0] + Z["proj_sum"][1]
            else:
                ps = Z["proj_sum"][0] if depth == "20" else Z["proj_sum"][1]
            py = ps[:, :K] / float(depth) - pc
            c = cells[depth]
            lab = f"{src} {'Replogle' if src == 'E1' else 'Orion'} {depth}"
            res["cells"][lab] = run_cell(lab, c, C_A * (Uq @ py.T), c["w0y"], mod.w1_weights(Z, D, depth, K, BETA),
                                         st[f"{key} {depth}"], fails)

    if fails:
        sys.exit(f"raw, w0y or W1 do not reproduce the frozen results (A6 section 2); nothing written: {fails}")
    res["reading"] = {"primary E4 D3 responsive 20": reading(res["cells"]["E4 D3 responsive 20"]),
                      "D1 responsive 20 (as executed)": reading(res["cells"]["D1 responsive 20 (as executed)"]),
                      "D1 responsive 20 (corrected)": reading(res["cells"]["D1 responsive 20 (corrected)"])}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "b1_design_ablation.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    for k, v in res["reading"].items():
        print(f"reading, {k}: {v}")
    print(f"wrote {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
