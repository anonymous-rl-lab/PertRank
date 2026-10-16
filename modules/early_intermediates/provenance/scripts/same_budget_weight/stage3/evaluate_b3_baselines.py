"""Amendment A6 section 5 (with A6a section 3): external task baselines B3. Frozen before it ran
(stage3/A6_FREEZE_SHA256.txt).

Donors D3 and D1, 20 cells, responsive and tercile queries. Every arm uses the same cells (each target's block 0
and the 2,000 input controls), the same queries, the same deep-block truth and the same tie-expected P@25.

  raw, w0y, W1   as in the frozen evaluations. D1 is first scored as executed, which must reproduce
                 stage3_d1_confirm.json, and is then scored with the A6a own-gene exclusion; every comparison uses
                 the corrected exclusion. D3 uses the exclusion E4 has used since Stage 2.
  scMAGeCK-LR    the steps of the author implementation (Bioconductor scMAGeCK: single_gene_matrix_regression,
                 getsolvedmatrix): design cells = input controls + all targets' block 0; genes with non-zero counts
                 in at least 1% of design cells (other query rows get a constant prediction); each gene's values
                 above its 95th percentile over the design cells set to that percentile (numpy 'linear' = R type 7);
                 design matrix = a column of ones (the NegCtrl column of the author code) plus one indicator per
                 target; ridge lambda = 0.01; targets ranked by the selection score. Computed in closed form:
                     s_t = n/(n + lam) (ybar_t - c),
                     c   = (n_C ybar_C + w sum_t ybar_t) / (n_C + lam + T w),  w = n lam / (n + lam),
                 and checked against a direct solve of the full normal equations (sup-norm relative 1e-8).
  scMAGeCK-RRA   MAGeCK alpha-RRA as the author code calls it (-p 0.3): per query gene, all design cells ranked,
                 descending for positive and ascending for negative selection, ties in one fixed random order
                 (default_rng(303) over the stored cell order), u = rank / M; per target
                 rho = min_{k<=j} I_{u_(k)}(k, 21 - k) over its j cells with u <= 0.3, rho = 1 when j = 0.
                 Direction 0 ranks by the positive-selection rho, direction 1 by the negative-selection rho.
  Welch t        20 target cells against the 2,000 input controls; t = 0 where the denominator is 0.
  rank-sum       Mann-Whitney U with mid-ranks, ranked by U / (20 * 2000) - 0.5.

Fatal checks: each state was written by the frozen read under A6; raw, w0y and W1 reproduce the stored results to
1e-12; the LR closed form matches the direct solve; without winsorisation LR scores exactly like the target-mean
difference built from the same stored cells (1e-12), the numerical form of the analytic equivalence.

    python3 stage3/evaluate_b3_baselines.py [--out stage3/results]
"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import betainc

import a6_common as A
from a6_common import BETA, K, block_bootstrap, exp_scores, fmt, summ
import evaluate_stage2 as es  # noqa: E402
import evaluate_w1_e4 as ev  # noqa: E402

SCRIPTS = ["evaluate_b3_baselines.py", "read_b3_query_cells.py"]
N_CELLS, LAM, Q95, MIN_FRAC, ALPHA, RRA_SEED = 20, 0.01, 95.0, 0.01, 0.3, 303
EXTERNAL = ("scMAGeCK-LR", "scMAGeCK-RRA", "Welch t", "rank-sum")


def load_state(donor, integ):
    p = os.path.join(A.HERE, "data", f"b3_{donor}_query_cells.npz")
    S = np.load(p, allow_pickle=True)
    if str(S["amendment_A6_sha256"]) != integ["amendment_A6_sha256"]:
        sys.exit(f"{donor}: the read was done under a different A6")
    if str(S["script_sha256"]) != integ["freeze"]["read_b3_query_cells.py"]:
        sys.exit(f"{donor}: the state was not written by the frozen read_b3_query_cells.py")
    X = np.log1p(S["counts"].astype(np.float64) * (float(S["scale"]) / np.maximum(S["total_counts"], 1.0))[:, None])
    return S, X, A.sha256(p)


def lr_closed(X, nc, NT):
    yc = X[:nc].mean(0); yt = X[nc:].reshape(NT, N_CELLS, -1).mean(1)
    w = N_CELLS * LAM / (N_CELLS + LAM)
    c = (nc * yc + w * yt.sum(0)) / (nc + LAM + NT * w)
    return (N_CELLS / (N_CELLS + LAM)) * (yt - c[None, :])


def lr_direct(X, nc, NT):
    """the full ridge normal equations, (X^T X + lam I) b = X^T y, targets first and the ones column last"""
    M = X.shape[0]
    XtX = np.zeros((NT + 1, NT + 1))
    XtX[np.arange(NT), np.arange(NT)] = N_CELLS
    XtX[:NT, NT] = N_CELLS; XtX[NT, :NT] = N_CELLS; XtX[NT, NT] = M
    XtX += LAM * np.eye(NT + 1)
    Xty = np.vstack([X[nc:].reshape(NT, N_CELLS, -1).sum(1), X.sum(0)[None, :]])
    return np.linalg.solve(XtX, Xty)[:NT]


def rra(X, nc, NT):
    M, Q = X.shape
    key = np.random.default_rng(RRA_SEED).random(M)
    kk = np.arange(1, N_CELLS + 1, dtype=np.float64)
    up = np.ones((NT, Q)); dn = np.ones((NT, Q))
    for g in range(Q):
        v = X[:, g]
        for sgn, out in ((-1.0, up), (1.0, dn)):
            order = np.lexsort((key, sgn * v))
            rank = np.empty(M); rank[order] = np.arange(1, M + 1)
            u = np.sort((rank[nc:] / M).reshape(NT, N_CELLS), axis=1)
            p = betainc(kk[None, :], (N_CELLS + 1 - kk)[None, :], u)
            p[u > ALPHA] = 1.0
            out[:, g] = p.min(1)
    return up, dn


def welch_t(X, nc, NT):
    xt = X[nc:].reshape(NT, N_CELLS, -1); xc = X[:nc]
    den = np.sqrt(xt.var(1, ddof=1) / N_CELLS + xc.var(0, ddof=1)[None, :] / nc)
    num = xt.mean(1) - xc.mean(0)[None, :]
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def rank_sum(X, nc, NT):
    xt = X[nc:].reshape(NT, N_CELLS, -1); sc = np.sort(X[:nc], axis=0)
    U = np.zeros((NT, X.shape[1]))
    for g in range(X.shape[1]):
        lo = np.searchsorted(sc[:, g], xt[:, :, g], "left"); hi = np.searchsorted(sc[:, g], xt[:, :, g], "right")
        U[:, g] = (lo + 0.5 * (hi - lo)).sum(1)
    return U / (N_CELLS * nc) - 0.5


def main():
    t0 = time.time()
    out_dir = A.arg("--out", os.path.join(A.HERE, "results"))
    integ = A.integrity(SCRIPTS)
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ,
           "setting": {"cells": N_CELLS, "k": K, "beta": BETA, "lr_lambda": LAM, "lr_winsor_percentile": Q95,
                       "lr_min_nonzero_fraction": MIN_FRAC, "rra_alpha": ALPHA, "rra_tie_seed": RRA_SEED},
           "donors": {}, "cells": {}}
    fails = []
    for donor in ("D3", "D1"):
        S, X, sha = load_state(donor, integ)
        nc = int(S["n_controls"]); NT = len(S["names"]); C = S["counts"]
        chk = {"state_sha256": sha, "read_checks": json.loads(str(S["checks"]))}
        # reference arms and their exclusion
        if donor == "D3":
            D = es.load("e4_tcell.npz")
            if [str(x) for x in S["names"]] != [str(x) for x in D["names"]]:
                sys.exit("D3: cohort differs")
            Z = np.load(os.path.join(A.HERE, "data", "e4_w1_stage3.npz"), allow_pickle=True)
            wM = ev.w1(Z, K, BETA)[0]
            src = {(c["rule"], c["depth"]): c for c in es.sources()["E4"]["cells"]}
            stored = A.results("stage3_w1_e4.json")["cells"]
            qu = D["qu"]
            cells = {}
            for rule, q in (("responsive", D["q_resp"]), ("terciles", D["q_terc"])):
                c = dict(src[(rule, "20")]); c["W1"] = wM; c["q"] = q
                cells[rule] = c
                sc = {a: float(exp_scores(E, c).mean()) for a, E in
                      (("raw", c["Y"]), ("w0y", c["w0y"][None, :] * c["Y"]), ("W1", wM[None, :] * c["Y"]))}
                chk[f"reproduction {rule} 20"] = A.check_stored(f"D3 {rule} 20", sc, stored[f"{rule} 20"], fails)
        else:
            Zd = A.d1_load()
            if [str(x) for x in S["names"]] != [str(x) for x in Zd["names"]]:
                sys.exit("D1: cohort differs")
            stored = A.results("stage3_d1_confirm.json")["cells"]
            qu = Zd["qu"]
            for (rule, depth), c in A.d1_cells(Zd, "as executed").items():
                if depth != "20":
                    continue
                sc = {a: float(exp_scores(E, c).mean()) for a, E in
                      (("raw", c["Y"]), ("w0y", c["w0y"][None, :] * c["Y"]), ("W1", c["W1"][None, :] * c["Y"]))}
                chk[f"reproduction {rule} 20 (as executed)"] = A.check_stored(f"D1 {rule} 20", sc, stored[f"{rule} 20"],
                                                                              fails)
            cells = {rule: c for (rule, depth), c in A.d1_cells(Zd, "corrected").items() if depth == "20"}
        if not np.array_equal(S["query_genes"], (D["keep"] if donor == "D3" else Zd["keep"])[qu]):
            sys.exit(f"{donor}: the stored query genes differ from the query union")
        # external statistics on the query union, rows in qu order
        lr0 = lr_closed(X, nc, NT)
        d = lr_direct(X, nc, NT)
        chk["lr_closed_vs_direct_supnorm_rel"] = float(np.max(np.abs(lr0 - d)) / np.max(np.abs(d)))
        if chk["lr_closed_vs_direct_supnorm_rel"] > 1e-8:
            fails.append(f"{donor}: LR closed form vs direct solve {chk['lr_closed_vs_direct_supnorm_rel']:.2e}")
        del d
        nzf = (C > 0).mean(0)
        Xw = np.minimum(X, np.percentile(X, Q95, axis=0)[None, :])
        lr = lr_closed(Xw, nc, NT); del Xw
        lr[:, nzf < MIN_FRAC] = 0.0
        const = np.ptp(lr, axis=0) == 0
        chk["lr_rows_filtered"] = int((nzf < MIN_FRAC).sum()); chk["lr_rows_constant"] = int(const.sum())
        rup, rdn = rra(X, nc, NT)
        stat = {"scMAGeCK-LR": lr.T, "Welch t": welch_t(X, nc, NT).T, "rank-sum": rank_sum(X, nc, NT).T,
                "rra_up": -np.log(np.maximum(rup, 1e-300)).T, "rra_dn": np.log(np.maximum(rdn, 1e-300)).T,
                "lr_no_winsor": lr0.T,
                "target_mean_difference": (X[nc:].reshape(NT, N_CELLS, -1).mean(1) - X[:nc].mean(0)[None, :]).T}
        chk["rra_rho_one_fraction"] = {"positive": float((rup == 1).mean()), "negative": float((rdn == 1).mean())}
        del X
        for rule, c in cells.items():
            rows = np.searchsorted(qu, c["q"])
            Sc = {"raw": exp_scores(c["Y"], c), "w0y": exp_scores(c["w0y"][None, :] * c["Y"], c),
                  "W1": exp_scores(c["W1"][None, :] * c["Y"], c)}
            for a in ("scMAGeCK-LR", "Welch t", "rank-sum", "lr_no_winsor", "target_mean_difference"):
                Sc[a] = exp_scores(stat[a][rows], c)
            Sc["scMAGeCK-RRA"] = np.stack([exp_scores(stat["rra_up"][rows], c)[:, 0],
                                           exp_scores(stat["rra_dn"][rows], c)[:, 1]], 1)
            eq = float(np.max(np.abs(Sc["lr_no_winsor"] - Sc["target_mean_difference"])))
            chk[f"lr_no_winsor_equals_mean_difference {rule} (max abs per-query)"] = eq
            if eq > 1e-12:
                fails.append(f"{donor} {rule}: LR without winsorisation differs from the mean difference by {eq:.2e}")
            chk[f"stored_cells_mean_difference_vs_raw {rule}"] = float(
                abs(Sc["target_mean_difference"].mean() - Sc["raw"].mean()))
            row = {"queries": int(len(rows)), "means": {a: float(Sc[a].mean()) for a in
                                                        ("raw", "w0y", "W1") + EXTERNAL},
                   "comparisons": {}}
            for a in EXTERNAL:
                row["comparisons"][f"W1 - {a}"] = summ(block_bootstrap(Sc["W1"], Sc[a], c["blocks"]))
                row["comparisons"][f"{a} - raw"] = summ(block_bootstrap(Sc[a], Sc["raw"], c["blocks"]))
            row["comparisons"]["W1 - raw"] = summ(block_bootstrap(Sc["W1"], Sc["raw"], c["blocks"]))
            row["comparisons"]["W1 - w0y"] = summ(block_bootstrap(Sc["W1"], Sc["w0y"], c["blocks"]))
            lab = f"{donor} {rule} 20"
            res["cells"][lab] = row
            m = row["means"]
            print(f"{lab:18s} raw {m['raw']:.4f} w0y {m['w0y']:.4f} W1 {m['W1']:.4f} | LR {m['scMAGeCK-LR']:.4f} "
                  f"RRA {m['scMAGeCK-RRA']:.4f} t {m['Welch t']:.4f} ranksum {m['rank-sum']:.4f} | "
                  + "  ".join(f"W1-{a} {fmt(row['comparisons'][f'W1 - {a}'])}" for a in EXTERNAL), flush=True)
        res["donors"][donor] = chk
        print(f"{donor} checks: " + json.dumps({k: v for k, v in chk.items() if k != "read_checks"}), flush=True)
    if fails:
        sys.exit(f"B3 checks failed; nothing written: {fails}")
    res["reading"] = {}
    for lab in ("D3 responsive 20", "D1 responsive 20"):
        beat = [a for a in EXTERNAL if res["cells"][lab]["comparisons"][f"W1 - {a}"]["ci95"][1] < 0]
        res["reading"][lab] = (f"external arm(s) above W1 with the interval above zero: {beat}" if beat else
                               "no external arm above W1 with its interval above zero")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "b3_external_baselines.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res["reading"], indent=1))
    print(f"wrote {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
