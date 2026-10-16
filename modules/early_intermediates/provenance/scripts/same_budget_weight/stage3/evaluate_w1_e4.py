"""Stage 3 evaluation of candidate W1 on E4, 20 cells (amendments A4 and A4a). Frozen together with
stage3/read_e4_w1.py before the read produced anything (stage3/STAGE3_FREEZE_SHA256.txt).

Candidate W1, at B cells, on a gene set of size G with unit directions u_1..u_k from the input-control covariance:

    M           = I - beta * sum_{i<k} u_i u_i^T
    a_M,j       = Y_j^T M Y_j / G            = m2_j - beta * sum_i (u_i^T Y_j)^2 / G
    nu_M,j(B)   = tr[M (Sigma_j / B + Sigma_C / n_C)] / G
                = [varbar_j / B + tr(Sigma_C) / (G n_C)] - beta * sum_i [u_i^T Sigma_j u_i / B + lambda_i / n_C] / G
    w_j         = [1 - nu_M,j / a_M,j]_+

with Y_j the B-cell mean minus the input-control mean, u_i^T Y_j = proj_sum / B - proj_control, and
u_i^T Sigma_j u_i the variance over the target's own B cells of its projection, from proj_sum and proj_sumsq.
beta = 0 reproduces w0y on the same gene set; a diagonal metric is V1 and is reported from the Stage 2 scalars.

Primary setting, fixed in A4 before the read: k = 5, beta = 0.9, all 13,390 weight genes.
Decision (A4 section 3, A4a section 3): E4 responsive queries at 20 cells against w0y, tie-expectation P@25,
30-block paired bootstrap (B = 2000, seed 2026, keys 101 / 202). Difference >= +0.01 with the 95% interval above
zero continues; >= +0.02 with the interval above zero is worth finalising. Failure condition: weights all zero or
all equal. E4 terciles at 20 cells is the one gatekeeper this read can compute; the 50-cell gatekeepers and the
other sources need their own depth statistics and are not computed here (A4a section 2).
Everything else printed is sensitivity and is not eligible.

    python3 stage3/evaluate_w1_e4.py [--data stage3/data/e4_w1_stage3.npz] [--out stage3/results]
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BUNDLE, "stage2"))
sys.path.insert(0, os.path.join(BUNDLE, "scripts"))
from _paths import AMEND_A3, AMEND_A3_RECORD, AMEND_A4, AMEND_A4_RECORD, checked_sha, sha256  # noqa: E402
import _paths  # noqa: E402
import evaluate_stage2 as es  # noqa: E402
from pr10 import block_bootstrap, score_rows  # noqa: E402

FREEZE = os.path.join(HERE, "STAGE3_FREEZE_SHA256.txt")
FROZEN = {"read_e4_w1.py", "evaluate_w1_e4.py"}
K_PRIMARY, BETA_PRIMARY = 5, 0.9
K_GRID, BETA_GRID = (1, 3, 5, 10, 20), (0.5, 0.9, 0.99)


def freeze_check():
    rec = {}
    for line in open(FREEZE, encoding="utf-8"):
        p = line.split()
        if len(p) == 2 and len(p[0]) == 64:
            rec[p[1]] = p[0]
    for rel in sorted(FROZEN):
        if rec.get(rel) != sha256(os.path.join(HERE, rel)):
            sys.exit(f"{rel} differs from the Stage 3 freeze record; the frozen evaluation cannot run on an edited file")
    return rec


def w1(Z, k, beta, tag=""):
    """W1 weights at 20 cells on the gene set selected by tag ('' = all weight genes, '_nonquery' = sensitivity)"""
    B = 20.0; n_c = float(Z["n_control"])
    G = float(Z["n_genes" if not tag else "n_genes_nonquery"])
    m2 = Z["m2_20" + tag]; vb = Z["varbar_20" + tag]
    lam = Z["eigenvalues" + tag][:k]; trC = float(Z["control_trace" + tag])
    ps = Z["proj_sum" + tag][:, :k]; pq = Z["proj_sumsq" + tag][:, :k]; pc = Z["proj_control" + tag][:k]
    pY = ps / B - pc[None, :]                                  # u_i^T Y_j
    sj = (pq - B * (ps / B) ** 2) / (B - 1.0)                  # u_i^T Sigma_j u_i, from the target's own cells
    a = m2 - beta * (pY ** 2).sum(1) / G
    nu = vb / B + trC / (G * n_c) - beta * (sj / B + lam[None, :] / n_c).sum(1) / G
    return np.clip(1.0 - nu / np.maximum(a, 1e-300), 0.0, 1.0), a, nu


def main():
    t0 = time.time()
    data = es.arg("--data", os.path.join(HERE, "data", "e4_w1_stage3.npz"))
    out_dir = es.arg("--out", os.path.join(HERE, "results"))
    integ = {"plan_sha256": es.plan_sha(), "amendment_A2_sha256": es.amendment_sha(),
             "amendment_A3_sha256": checked_sha(AMEND_A3, AMEND_A3_RECORD, "amendment A3"),
             "amendment_A4_sha256": checked_sha(AMEND_A4, AMEND_A4_RECORD, "amendment A4"),
             "amendment_A4a_sha256": checked_sha(os.path.join(BUNDLE, "AMENDMENT_A4a.md"),
                                                 os.path.join(BUNDLE, "AMENDMENT_A4a_SHA256.txt"), "amendment A4a"),
             "freeze": freeze_check(), "data_sha256": sha256(data)}
    Z = np.load(data, allow_pickle=True)
    for key in ("amendment_A4_sha256", "amendment_A4a_sha256"):
        if str(Z[key]) != integ[key]:
            sys.exit(f"the read was done under a different {key}")
    print("integrity ok; read checks:", str(Z["checks"]), flush=True)

    SRC = es.sources()["E4"]
    if [str(x) for x in Z["names"]] != SRC["targets"]:
        sys.exit("targets differ from the reduced state")
    D = es.load("e4_tcell.npz")
    rel = lambda a, b: float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))
    chk = {"m2_20_vs_reduced": rel(Z["m2_20"], D["m2_20"].astype(np.float64)),
           "varbar_20_vs_reduced": rel(Z["varbar_20"], D["vb_20"].astype(np.float64))}
    w_beta0, a0, nu0 = w1(Z, K_PRIMARY, 0.0)
    # beta = 0 must reproduce w0y. The comparison is ABSOLUTE: w is clipped at zero and 106 of the 1,200 targets
    # sit at or near it, where a relative tolerance diverges on a difference that does not matter for scoring.
    # (Corrected 2026-09-26T01:53Z, before any W1 result existed; see stage3/STAGE3_FREEZE_SHA256.txt.)
    chk["beta_zero_reproduces_w0y_abs"] = float(np.max(np.abs(
        w_beta0 - next(c for c in SRC["cells"] if c["rule"] == "responsive" and c["depth"] == "20")["w0y"])))
    print("checks:", json.dumps(chk), flush=True)
    if chk["m2_20_vs_reduced"] > 1e-3 or chk["varbar_20_vs_reduced"] > 1e-3 or chk["beta_zero_reproduces_w0y_abs"] > 1e-6:
        sys.exit(f"the read does not reproduce the frozen baseline: {chk}")

    cells = {(c["rule"], c["depth"]): c for c in SRC["cells"]}
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ,
           "checks": chk, "primary_setting": {"k": K_PRIMARY, "beta": BETA_PRIMARY, "genes": "all weight genes"},
           "cells": {}, "sensitivity": {}, "decision": {}}

    def score(c, w):
        return score_rows(w[None, :] * c["Y"], c["T"], c["sq"], key_pred=c["KP"], key_truth=c["KT"])["exp"]

    for key in (("responsive", "20"), ("terciles", "20")):
        c = cells[key]
        raw = score(c, np.ones(len(w_beta0))); base = score(c, c["w0y"])
        w, a, nu = w1(Z, K_PRIMARY, BETA_PRIMARY)
        sc = score(c, w)
        vs_w0y = block_bootstrap(sc, base, c["blocks"]); vs_raw = block_bootstrap(sc, raw, c["blocks"])
        degenerate = bool(np.all(w == 0) or np.ptp(w) == 0)
        res["cells"][" ".join(key)] = {
            "role": c["role"], "raw": float(raw.mean()), "w0y": float(base.mean()), "W1": float(sc.mean()),
            "W1_vs_w0y": {"diff": vs_w0y["diff"], "ci95": vs_w0y["ci95"], "p_one_sided": vs_w0y["p_one_sided"]},
            "W1_vs_raw": {"diff": vs_raw["diff"], "ci95": vs_raw["ci95"]},
            "w0y_vs_raw": {"diff": block_bootstrap(base, raw, c["blocks"])["diff"]},
            "weights": {"mean": float(w.mean()), "zero_fraction": float((w == 0).mean()),
                        "spearman_with_w0y": float(spearmanr(w, c["w0y"]).correlation),
                        "degenerate": degenerate}}
        print(f"{key[0]} {key[1]} ({c['role']}): raw {raw.mean():.4f}  w0y {base.mean():.4f}  W1 {sc.mean():.4f}   "
              f"W1 vs w0y {vs_w0y['diff']:+.4f} [{vs_w0y['ci95'][0]:+.4f},{vs_w0y['ci95'][1]:+.4f}]", flush=True)

    p = res["cells"]["responsive 20"]
    d = p["W1_vs_w0y"]; lo = d["ci95"][0]
    reading = ("worth finalising" if d["diff"] >= 0.02 and lo > 0 else
               "continue" if d["diff"] >= 0.01 and lo > 0 else "does not continue")
    if p["weights"]["degenerate"]:
        reading = "does not continue (weights degenerate)"
    res["decision"] = {"primary_cell": "E4 responsive 20", "diff": d["diff"], "ci95": d["ci95"], "reading": reading,
                       "gatekeeper_computed": {"E4 terciles 20": res["cells"]["terciles 20"]["W1_vs_w0y"]},
                       "gatekeepers_not_computed": ["E4 terciles 50", "E1 20", "E1 50", "E5 responsive 20",
                                                    "E5 responsive 50", "E3 20", "E3 100"],
                       "note": "A4a section 2: only the 20-cell statistics were read, so the other gatekeepers "
                               "need their own depth and source before any overall reading"}

    # sensitivity, reported only
    c = cells[("responsive", "20")]; base = score(c, c["w0y"])
    for tag, label in (("", "all_weight_genes"), ("_nonquery", "non_query_genes")):
        g = {}
        for k in K_GRID:
            for beta in BETA_GRID:
                w, a, nu = w1(Z, k, beta, tag)
                b = block_bootstrap(score(c, w), base, c["blocks"])
                g[f"k{k}_beta{beta}"] = {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]],
                                         "mean_weight": round(float(w.mean()), 4),
                                         "zero_fraction": round(float((w == 0).mean()), 4)}
        res["sensitivity"][label] = g
    # the input-only selection rule of A4 section 2
    # A4 section 2's input-only rule: the smallest k whose directions carry 50% of the control trace. The stored
    # directions may not reach that share, in which case k is not determined by the data in hand and the setting
    # is not computable (A4a section 2). Reporting only; never eligible.
    # (Corrected 2026-09-26 per A4b section 2: searchsorted returned the out-of-bounds insertion point, which the
    # first version reported as a selected k of 21.)
    lam = Z["eigenvalues"]; share = np.cumsum(lam) / float(Z["control_trace"])
    nd = int(Z["n_directions"])
    reached = bool(share[-1] >= 0.5)
    k_rule = int(np.searchsorted(share, 0.5) + 1) if reached else None
    res["sensitivity"]["input_only_rule"] = {
        "target_trace_share": 0.5, "beta": 0.9, "n_directions_stored": nd,
        "cumulative_trace_share": share.tolist(), "share_reached_by_stored_directions": round(float(share[-1]), 4),
        "k": k_rule, "computable": reached,
        "note": None if reached else f"the {nd} stored directions carry only {share[-1] * 100:.1f}% of the control "
                                     f"trace, so the rule needs k > {nd}; its value is not determined this round"}
    if reached:
        w, _, _ = w1(Z, k_rule, 0.9)
        b = block_bootstrap(score(c, w), base, c["blocks"])
        res["sensitivity"]["input_only_rule"]["result"] = {"diff": round(b["diff"], 5),
                                                           "ci95": [round(x, 5) for x in b["ci95"]]}

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stage3_w1_e4.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    print(f"\nprimary cell E4 responsive 20, W1 vs w0y: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
          f"  ->  {reading}")
    print(f"sensitivity, all weight genes, best of the grid: "
          f"{max(res['sensitivity']['all_weight_genes'].items(), key=lambda x: x[1]['diff'])}")
    r = res["sensitivity"]["input_only_rule"]
    print(f"input-only selection rule: {'k = ' + str(r['k']) if r['computable'] else r['note']}")
    print(f"wrote {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
