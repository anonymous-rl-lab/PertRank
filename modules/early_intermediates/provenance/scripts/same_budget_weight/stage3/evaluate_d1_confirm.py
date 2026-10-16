"""Amendment A5 section 5: the confirmation decision on the unseen donor D1_Rest. Frozen with
stage3/read_d1_confirm.py before that read produced anything (stage3/STAGE3_FREEZE_SHA256.txt).

Reads only stage3/data/d1_confirm_state.npz. w0y and W1 are both built from that state, so the comparison is
internal to this donor; k = 5 and beta = 0.9 come from evaluate_w1_e4.py and are not re-selected.

    w0y  = [1 - nu / m2]_+                     nu = varbar / B + cvm / n_C
    W1   = [1 - nu_M / a_M]_+                  a_M  = m2 - beta * sum_i (u_i^T Y)^2 / G
                                               nu_M = nu - beta * sum_i [u_i^T Sigma_j u_i / B + lambda_i / n_C] / G
    prediction = w * Y, Y the B-cell mean minus the input-control mean on the query genes
    truth      = block 4 mean minus the evaluation-control mean on the query genes

Decision (A5 section 5): responsive queries at 20 cells, W1 against w0y. Difference >= +0.01 with the 95% interval
above zero confirms; >= +0.02 with the interval above zero confirms at the finalising level; otherwise not
confirmed. Gatekeepers on this donor: terciles 20 and 50, each passing unless its interval lies entirely below
zero. Failure conditions: weights all zero or all equal, or the primary cell below the line.

    python3 stage3/evaluate_d1_confirm.py [--state stage3/data/d1_confirm_state.npz] [--out stage3/results]
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
from _paths import AMEND_A5, AMEND_A5_RECORD, checked_sha, import_pr10, sha256  # noqa: E402
import evaluate_w1_e4 as ev  # noqa: E402

import_pr10()
from pr10 import block_bootstrap, score_rows, self_to_query  # noqa: E402

K, BETA = ev.K_PRIMARY, ev.BETA_PRIMARY
K_GRID, BETA_GRID = ev.K_GRID, ev.BETA_GRID
FROZEN = {"read_d1_confirm.py", "evaluate_d1_confirm.py"}
PRIMARY = ("responsive", "20")
GATE = (("terciles", "20"), ("terciles", "50"))


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    t0 = time.time()
    state = arg("--state", os.path.join(HERE, "data", "d1_confirm_state.npz"))
    out_dir = arg("--out", os.path.join(HERE, "results"))
    a5 = checked_sha(AMEND_A5, AMEND_A5_RECORD, "amendment A5")
    rec = {}
    for line in open(os.path.join(HERE, "STAGE3_FREEZE_SHA256.txt"), encoding="utf-8"):
        p = line.split()
        if len(p) == 2 and len(p[0]) == 64:
            rec[p[1]] = p[0]
    for f in sorted(FROZEN):
        if rec.get(f) != sha256(os.path.join(HERE, f)):
            sys.exit(f"{f} differs from the Stage 3 freeze record")
    Z = np.load(state, allow_pickle=True)
    if str(Z["amendment_A5_sha256"]) != a5:
        sys.exit("the read was done under a different A5")
    integ = {"amendment_A5_sha256": a5, "freeze": {f: rec[f] for f in sorted(FROZEN)},
             "state_sha256": sha256(state), "donor": str(Z["donor"])}
    print(f"donor {str(Z['donor'])}, cohort {len(Z['names'])}, weight genes {int(Z['n_weight_genes'])}, "
          f"read checks {str(Z['checks'])}", flush=True)

    NT = len(Z["names"]); G = float(Z["n_weight_genes"]); n_c = float(Z["n_control"])
    cm_q = Z["cm"][Z["qu"]][:, None]; ce_q = Z["ce"][Z["qu"]][:, None]
    cvm = float(Z["cvm"]); trC = float(Z["control_trace"])
    lam = Z["eigenvalues"]; pc = Z["proj_control"]
    Y = {"20": Z["S0_q"] / 20.0 - cm_q, "50": (Z["S0_q"] + Z["S1_q"]) / 50.0 - cm_q}
    T = Z["S4_q"] / 100.0 - ce_q
    sq_all = self_to_query(Z["self_row"], Z["qu"])
    keys = (np.random.default_rng(101).random(NT), np.random.default_rng(202).random(NT))

    def w0y(depth):
        m2 = Z[f"m2_{depth}"]; vb = Z[f"vb_{depth}"]
        return np.clip(1.0 - (vb / float(depth) + cvm / n_c) / m2, 0.0, 1.0)

    def w1(depth, k, beta):
        B = float(depth); bi = 0 if depth == 20 else None
        ps = Z["proj_sum"][0][:, :k] if depth == 20 else (Z["proj_sum"][0] + Z["proj_sum"][1])[:, :k]
        pq = Z["proj_sumsq"][0][:, :k] if depth == 20 else (Z["proj_sumsq"][0] + Z["proj_sumsq"][1])[:, :k]
        m2 = Z[f"m2_{depth}"]; vb = Z[f"vb_{depth}"]
        pY = ps / B - pc[None, :k]
        sj = (pq - B * (ps / B) ** 2) / (B - 1.0)
        a = m2 - beta * (pY ** 2).sum(1) / G
        nu = vb / B + trC / (G * n_c) - beta * (sj / B + lam[None, :k] / n_c).sum(1) / G
        return np.clip(1.0 - nu / np.maximum(a, 1e-300), 0.0, 1.0)

    # A5 section 7 check 4
    chk = {f"beta_zero_reproduces_w0y_{d}_abs": float(np.max(np.abs(w1(d, K, 0.0) - w0y(d)))) for d in (20, 50)}
    print("check 4:", json.dumps(chk), flush=True)
    if max(chk.values()) > 1e-6:
        sys.exit(f"beta = 0 does not reproduce this donor's w0y: {chk}")

    def cell(rule, depth):
        q = Z["q_resp"] if rule == "responsive" else Z["q_terc"]
        pos = np.searchsorted(Z["qu"], q)
        return {"Y": Y[depth][pos], "T": T[pos], "sq": self_to_query(Z["self_row"], Z["qu"][pos]),
                "blocks": Z["blocks_responsive"] if rule == "responsive" else Z["blocks_terciles"],
                "KP": keys[0], "KT": keys[1]}

    def score(c, w):
        return score_rows(w[None, :] * c["Y"], c["T"], c["sq"], key_pred=c["KP"], key_truth=c["KT"])["exp"]

    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ, "checks": chk,
           "cohort": {"targets": int(NT), "weight_genes": int(G), "eligible": int(Z["n_eligible"]),
                      "cells": int(Z["n_cells"]), "control_trace": trC,
                      "top20_trace_share": round(float(lam.sum() / trC), 4)},
           "setting": {"k": K, "beta": BETA, "genes": "all weight genes",
                       "directions": "this donor's own 2,000 input controls"},
           "cells": {}, "sensitivity": {}, "decision": {}}
    for rule in ("responsive", "terciles"):
        for depth in ("20", "50"):
            c = cell(rule, depth)
            w = w1(int(depth), K, BETA); b0 = w0y(int(depth))
            raw = score(c, np.ones(NT)); base = score(c, b0); sc = score(c, w)
            bb = block_bootstrap(sc, base, c["blocks"]); br = block_bootstrap(sc, raw, c["blocks"])
            role = ("primary" if (rule, depth) == PRIMARY else
                    "gatekeeper" if (rule, depth) in GATE else "descriptive")
            res["cells"][f"{rule} {depth}"] = {
                "role": role, "raw": float(raw.mean()), "w0y": float(base.mean()), "W1": float(sc.mean()),
                "W1_vs_w0y": {"diff": bb["diff"], "ci95": bb["ci95"], "p_one_sided": bb["p_one_sided"]},
                "W1_vs_raw": {"diff": br["diff"], "ci95": br["ci95"]},
                "w0y_vs_raw": {"diff": block_bootstrap(base, raw, c["blocks"])["diff"]},
                "weights": {"mean": float(w.mean()), "zero_fraction": float((w == 0).mean()),
                            "spearman_with_w0y": float(spearmanr(w, b0).correlation),
                            "degenerate": bool(np.all(w == 0) or np.ptp(w) == 0)}}
            print(f"{rule} {depth} ({role}): raw {raw.mean():.4f}  w0y {base.mean():.4f}  W1 {sc.mean():.4f}   "
                  f"W1 vs w0y {bb['diff']:+.4f} [{bb['ci95'][0]:+.4f},{bb['ci95'][1]:+.4f}]", flush=True)

    p = res["cells"][f"{PRIMARY[0]} {PRIMARY[1]}"]
    d = p["W1_vs_w0y"]; lo = d["ci95"][0]
    if p["weights"]["degenerate"]:
        reading = "not confirmed (weights degenerate)"
    elif d["diff"] >= 0.02 and lo > 0:
        reading = "confirmed, at the finalising level"
    elif d["diff"] >= 0.01 and lo > 0:
        reading = "confirmed"
    else:
        reading = "not confirmed"
    gates = {f"{r} {dp}": res["cells"][f"{r} {dp}"]["W1_vs_w0y"] for r, dp in GATE}
    res["decision"] = {"donor": str(Z["donor"]), "primary_cell": "responsive 20", "diff": d["diff"],
                       "ci95": d["ci95"], "reading": reading,
                       "gatekeepers": gates, "gatekeepers_pass": {k: bool(v["ci95"][1] >= 0) for k, v in gates.items()},
                       "all_gatekeepers_pass": bool(all(v["ci95"][1] >= 0 for v in gates.values())),
                       "scope": "an unseen-donor confirmation within the same study, not an external confirmation "
                                "on a new laboratory or platform (A5 section 1)",
                       "does_not_replace": "the four E1 and E3 gatekeepers of A4 (A5 section 9)"}

    c = cell(*PRIMARY); base = score(c, w0y(20))
    grid = {}
    for k in K_GRID:
        for beta in BETA_GRID:
            b = block_bootstrap(score(c, w1(20, k, beta)), base, c["blocks"])
            grid[f"k{k}_beta{beta}"] = {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]]}
    res["sensitivity"]["primary_grid"] = grid
    ctrl = {}
    for label, idx in (("1..5", np.arange(5)), ("6..20", np.arange(5, 20)), ("1 only", np.arange(1))):
        ps = Z["proj_sum"][0][:, idx]; pq = Z["proj_sumsq"][0][:, idx]
        pY = ps / 20.0 - pc[None, idx]
        sj = (pq - 20.0 * (ps / 20.0) ** 2) / 19.0
        a = Z["m2_20"] - BETA * (pY ** 2).sum(1) / G
        nu = Z["vb_20"] / 20.0 + trC / (G * n_c) - BETA * (sj / 20.0 + lam[None, idx] / n_c).sum(1) / G
        w = np.clip(1.0 - nu / np.maximum(a, 1e-300), 0.0, 1.0)
        b = block_bootstrap(score(c, w), base, c["blocks"])
        ctrl[label] = {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]],
                       "control_trace_share": round(float(lam[idx].sum() / trC), 4)}
    res["sensitivity"]["direction_set_control"] = ctrl

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stage3_d1_confirm.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    print(f"\nprimary cell (responsive 20) on {str(Z['donor'])}: {d['diff']:+.4f} "
          f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  ->  {reading}")
    print(f"gatekeepers on this donor all pass: {res['decision']['all_gatekeepers_pass']}")
    print(f"direction-set control: " + "  ".join(f"{k} {v['diff']:+.4f}" for k, v in ctrl.items()))
    print(f"wrote {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
