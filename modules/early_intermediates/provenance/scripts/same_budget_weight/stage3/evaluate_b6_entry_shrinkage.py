"""Amendment A7: the per-entry shrinkage control B6. Frozen before it produced anything (stage3/A7_FREEZE_SHA256.txt).

EB-entry, per query gene g and target j, on the B3 per-cell states (each target's 20 block-0 cells and the 2,000
input controls), float64 throughout:

    y_gj    = mean of the 20 cells - mean of the controls
    s2_gj   = v_gj / 20 + v_Cg / 2000          (sample variances, ddof = 1)
    tau2_g  = max(0, mean_j (y_gj^2 - s2_gj))  (zero-centred normal prior, a pre-set modelling choice)
    post_gj = y_gj tau2_g / (tau2_g + s2_gj);  the whole row is 0 when tau2_g = 0, including 0/0

and targets are ranked by post. No tunable parameter. Cells: D3 and D1, responsive and tercile queries, 20 cells; D1
reproduces its frozen raw, w0y and W1 as executed and is then scored with the A6a own-gene exclusion. Comparisons:
W1 - EB-entry, EB-entry - raw, EB-entry - w0y. Reading on the two responsive 20-cell cells (A7 section 4).

Checks (fatal): the states come from the frozen B3 read; raw, w0y and W1 reproduce the frozen results to 1e-12
absolute; the vectorised EB-entry matches a naive per-entry loop on 10 hash-chosen query genes per donor,
|difference| <= 1e-12 x max(1, |value|).

    python3 stage3/evaluate_b6_entry_shrinkage.py [--out stage3/results]
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

import a6_common as A
from a6_common import BETA, K, block_bootstrap, exp_scores, fmt, summ
import evaluate_b3_baselines as b3  # noqa: E402
import evaluate_stage2 as es  # noqa: E402
import evaluate_w1_e4 as ev  # noqa: E402

A7_FREEZE = os.path.join(A.HERE, "A7_FREEZE_SHA256.txt")
SCRIPTS_A7 = ["evaluate_b6_entry_shrinkage.py"]
N_CELLS = 20
TOL = 1e-12


def eb_entry(X, nc, NT):
    """vectorised EB-entry: returns (NT, Q) posterior means"""
    xt = X[nc:].reshape(NT, N_CELLS, -1); xc = X[:nc]
    y = xt.mean(1) - xc.mean(0)[None, :]
    s2 = xt.var(1, ddof=1) / N_CELLS + xc.var(0, ddof=1)[None, :] / nc
    tau2 = np.maximum(0.0, (y ** 2 - s2).mean(0))
    den = tau2[None, :] + s2
    post = np.where(den > 0, y * tau2[None, :] / np.where(den > 0, den, 1.0), 0.0)
    post[:, tau2 == 0] = 0.0
    return post, tau2


def eb_entry_naive(X, nc, NT, g):
    """the same quantity for one gene column, written as plain loops"""
    xc = [float(v) for v in X[:nc, g]]
    mc = sum(xc) / nc
    vc = sum((v - mc) ** 2 for v in xc) / (nc - 1)
    ys, s2s = [], []
    for j in range(NT):
        cells = [float(v) for v in X[nc + N_CELLS * j: nc + N_CELLS * (j + 1), g]]
        m = sum(cells) / N_CELLS
        v = sum((c - m) ** 2 for c in cells) / (N_CELLS - 1)
        ys.append(m - mc); s2s.append(v / N_CELLS + vc / nc)
    tau2 = max(0.0, sum(y * y - s for y, s in zip(ys, s2s)) / NT)
    out = []
    for y, s in zip(ys, s2s):
        out.append(0.0 if tau2 == 0.0 or tau2 + s == 0.0 else y * tau2 / (tau2 + s))
    return np.array(out)


def integrity():
    out = A.integrity(["evaluate_b3_baselines.py", "read_b3_query_cells.py"])
    out["amendment_A7_sha256"] = A.checked_sha(os.path.join(A.BUNDLE, "AMENDMENT_A7.md"),
                                               os.path.join(A.BUNDLE, "AMENDMENT_A7_SHA256.txt"), "amendment A7")
    rec = A.record(A7_FREEZE)
    for s in SCRIPTS_A7:
        if rec.get(s) != A.sha256(os.path.join(A.HERE, s)):
            sys.exit(f"{s} differs from the A7 freeze record")
    out["freeze_A7"] = {s: rec[s] for s in SCRIPTS_A7}
    return out


def main():
    t0 = time.time()
    out_dir = A.arg("--out", os.path.join(A.HERE, "results"))
    integ = integrity()
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ,
           "setting": {"cells": N_CELLS, "prior": "zero-centred normal, method-of-moments variance per query gene",
                       "k": K, "beta": BETA}, "donors": {}, "cells": {}}
    fails = []
    for donor in ("D3", "D1"):
        S, X, sha = b3.load_state(donor, integ)
        nc = int(S["n_controls"]); NT = len(S["names"])
        chk = {"state_sha256": sha}
        if donor == "D3":
            D = es.load("e4_tcell.npz")
            Z = np.load(os.path.join(A.HERE, "data", "e4_w1_stage3.npz"), allow_pickle=True)
            wM = ev.w1(Z, K, BETA)[0]
            src = {(c["rule"], c["depth"]): c for c in es.sources()["E4"]["cells"]}
            stored = A.results("stage3_w1_e4.json")["cells"]
            qu = D["qu"]; gnames = np.array([str(g) for g in D["genes_keep"]])[qu]
            cells = {}
            for rule, q in (("responsive", D["q_resp"]), ("terciles", D["q_terc"])):
                c = dict(src[(rule, "20")]); c["W1"] = wM; c["q"] = q; cells[rule] = c
                sc = {a: float(exp_scores(E, c).mean()) for a, E in
                      (("raw", c["Y"]), ("w0y", c["w0y"][None, :] * c["Y"]), ("W1", wM[None, :] * c["Y"]))}
                chk[f"reproduction {rule} 20"] = A.check_stored(f"D3 {rule} 20", sc, stored[f"{rule} 20"], fails)
        else:
            Zd = A.d1_load()
            stored = A.results("stage3_d1_confirm.json")["cells"]
            qu = Zd["qu"]; gnames = np.array([str(g) for g in Zd["genes_keep"]])[qu]
            for (rule, depth), c in A.d1_cells(Zd, "as executed").items():
                if depth != "20":
                    continue
                sc = {a: float(exp_scores(E, c).mean()) for a, E in
                      (("raw", c["Y"]), ("w0y", c["w0y"][None, :] * c["Y"]), ("W1", c["W1"][None, :] * c["Y"]))}
                chk[f"reproduction {rule} 20 (as executed)"] = A.check_stored(f"D1 {rule} 20", sc,
                                                                              stored[f"{rule} 20"], fails)
            cells = {rule: c for (rule, depth), c in A.d1_cells(Zd, "corrected").items() if depth == "20"}
        post, tau2 = eb_entry(X, nc, NT)
        pick = sorted(range(len(gnames)), key=lambda i: hashlib.sha256(("b6|" + gnames[i]).encode()).hexdigest())[:10]
        worst = 0.0
        for g in pick:
            nv = eb_entry_naive(X, nc, NT, g)
            worst = max(worst, float(np.max(np.abs(post[:, g] - nv) / np.maximum(1.0, np.abs(nv)))))
        chk["naive_check_genes"] = [str(gnames[g]) for g in pick]
        chk["naive_check_max_scaled_diff"] = worst
        if worst > TOL:
            fails.append(f"{donor}: vectorised EB-entry differs from the naive loop by {worst:.2e}")
        chk["tau2_zero_fraction"] = float((tau2 == 0).mean())
        del X
        for rule, c in cells.items():
            rows = np.searchsorted(qu, c["q"])
            Sc = {"raw": exp_scores(c["Y"], c), "w0y": exp_scores(c["w0y"][None, :] * c["Y"], c),
                  "W1": exp_scores(c["W1"][None, :] * c["Y"], c), "EB-entry": exp_scores(post.T[rows], c)}
            row = {"queries": int(len(rows)), "means": {a: float(v.mean()) for a, v in Sc.items()},
                   "tau2_zero_rows": int((tau2[rows] == 0).sum()),
                   "comparisons": {"W1 - EB-entry": summ(block_bootstrap(Sc["W1"], Sc["EB-entry"], c["blocks"])),
                                   "EB-entry - raw": summ(block_bootstrap(Sc["EB-entry"], Sc["raw"], c["blocks"])),
                                   "EB-entry - w0y": summ(block_bootstrap(Sc["EB-entry"], Sc["w0y"], c["blocks"]))}}
            lab = f"{donor} {rule} 20"
            res["cells"][lab] = row
            m = row["means"]; cm = row["comparisons"]
            print(f"{lab:18s} raw {m['raw']:.4f} w0y {m['w0y']:.4f} W1 {m['W1']:.4f} EB-entry {m['EB-entry']:.4f} | "
                  f"W1-EB {fmt(cm['W1 - EB-entry'])}  EB-raw {fmt(cm['EB-entry - raw'])}  "
                  f"EB-w0y {fmt(cm['EB-entry - w0y'])}", flush=True)
        res["donors"][donor] = chk
        print(f"{donor} checks: {json.dumps(chk)}", flush=True)
    if fails:
        sys.exit(f"B6 checks failed; nothing written: {fails}")
    r = {d: res["cells"][f"{d} responsive 20"]["comparisons"]["W1 - EB-entry"] for d in ("D3", "D1")}
    if any(v["ci95"][1] < 0 for v in r.values()):
        reading = ("1: EB-entry above W1 with its interval above zero on "
                   + ", ".join(d for d, v in r.items() if v["ci95"][1] < 0))
    elif all(v["ci95"][0] > 0 for v in r.values()):
        reading = "2: on these data W1 is above the classical per-entry EB control on both donors"
    else:
        reading = "3: no advantage consistent across the two donors; reported per donor"
    res["reading"] = reading
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "b6_entry_shrinkage.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    print(f"reading: {reading}\nwrote {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
