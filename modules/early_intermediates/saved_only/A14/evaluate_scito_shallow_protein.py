"""Amendment A14: shallow protein recovery inside SCITO. One 20-cell input block per target, an independent deep
protein truth from the same experiment, and the frozen W1 against w0y on the same budget.

Two passes over the authors' deposited per-cell file, on the 201 panel proteins:
    pass 1  the NTC cells only: control mean cm, per-feature variance cv (ddof 1), cvm = cv.mean(), trC = cv.sum(),
            the full control covariance, its top NDIR eigenvectors U and eigenvalues lam, proj_control = U^T cm.
    pass 2  per-target block sums and sums of squares, and the per-target sums and sums of squares of the cells'
            projections U^T x, which pass 1 must precede because U is not known before it. Blocks: 0 = 20 cells (the
            shallow input), A = 100 cells (the deep truth), B = 100 cells (the gate of A14 section 4), taken from the
            target's own hashed cell permutation and therefore disjoint.
Weights are the project's own formulas, ported from a6_common.d1_w0y and d1_w1 with k = 5 and beta = 0.9, and the
beta = 0 identity against w0y is checked.

Order of operations is the amendment's: the gate runs first and involves no nomination at all; if the deep truth
does not replicate above the random candidate rate the script stops before any weight touches a nomination.

    python3 stage3/evaluate_scito_shallow_protein.py --h5ad <file> [--out stage3/results]
"""
import hashlib
import json
import os
import re
import sys
import time

import h5py
import numpy as np

import a6_common as A
from a6_common import block_bootstrap, fmt, summ
from pr10 import make_blocks, score_rows

K, BETA, KTOP, NDIR = 5, 0.9, 25, 20
N0, NA, NB = 20, 100, 100
MIN_CELLS = N0 + NA + NB
SALT = "pertrescue-a14|"
CHUNK = 200_000
PREFIXES = ("anti-human-mouse-rat-", "anti-mouse-human-", "anti-human-mouse-", "anti-human-",
            "anti-mouse-", "Mouse-", "Rat-", "ArmenianHamsterIgG-")
SUFFIX = re.compile(r"[-_]barcode\d+$")
PANEL = os.path.join(A.HERE, "external", "scito", "scito_panel_gene_map.json")


def clean(n):
    for p in PREFIXES:
        if n.startswith(p):
            n = n[len(p):]
            break
    return SUFFIX.sub("", n).strip()


def decode(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def setup(h5):
    P = json.load(open(PANEL, encoding="utf-8"))
    panel = set(P["mapping"]) | set(P["excluded"])
    f = h5py.File(h5, "r")
    cl = np.array([clean(x) for x in decode(f["var/x"][:])])
    keep = np.flatnonzero(np.isin(cl, list(panel)))
    if len(keep) != 201:
        sys.exit(f"expected the 201 panel proteins, matched {len(keep)}")
    gt = decode(f["obs/guide_target/categories"][:])
    codes = f["obs/guide_target/codes"][:].astype(np.int64)
    is_ntc = np.array([t.startswith("NO-TARGET") for t in gt])
    return f, f["X"], cl[keep], keep, gt, codes, is_ntc


def pass1(X, keep, codes, is_ntc, G, t0):
    """the control side: mean, per-feature variance, covariance and its top directions"""
    ntc = is_ntc[codes]
    s = np.zeros(G); CP = np.zeros((G, G)); n = 0
    for a in range(0, len(codes), CHUNK):
        b = min(a + CHUNK, len(codes)); m = ntc[a:b]
        if not m.any():
            continue
        blk = np.asarray(X[a:b, :], dtype=np.float64)[:, keep][m]
        s += blk.sum(0); CP += blk.T @ blk; n += len(blk)
    cm = s / n
    C = (CP - n * np.outer(cm, cm)) / (n - 1.0)
    cv = np.diag(C).copy(); trC = float(cv.sum())
    w, V = np.linalg.eigh(C)
    U = np.ascontiguousarray(V[:, ::-1][:, :NDIR]); lam = w[::-1][:NDIR].copy()
    print(f"pass 1: {n} NTC cells, control trace {trC:.2f}, top {K} directions carry {lam[:K].sum() / trC:.4f} "
          f"of it [{time.time() - t0:.0f}s]", flush=True)
    return {"cm": cm, "cv": cv, "cvm": float(cv.mean()), "trC": trC, "U": U, "lam": lam, "n_control": n}


def pass2(X, keep, codes, gt, is_ntc, G, t0):
    """per-target block moments and projection moments; the blocks come from each target's own hashed permutation"""
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(gt) + 1))
    eligible, blk_of = [], {}
    for t in np.flatnonzero(~is_ntc):
        cells = order[bounds[t]:bounds[t + 1]]
        if len(cells) < MIN_CELLS:
            continue
        rng = np.random.default_rng(int(hashlib.sha256((SALT + gt[t]).encode()).hexdigest()[:16], 16))
        pm = cells[rng.permutation(len(cells))][:MIN_CELLS]
        eligible.append(t); blk_of[t] = (pm[:N0], pm[N0:N0 + NA], pm[N0 + NA:])
    eligible = np.array(eligible); NT = len(eligible)
    slot = np.full(len(codes), -1, np.int8); row = np.full(len(codes), -1, np.int64)
    for i, t in enumerate(eligible):
        for b, cc in enumerate(blk_of[t]):
            slot[cc] = b; row[cc] = i
    return eligible, NT, slot, row


def accumulate(X, keep, slot, row, NT, G, U, t0):
    S = np.zeros((3, NT, G)); Q = np.zeros((3, NT, G))
    PS = np.zeros((NT, NDIR)); PQ = np.zeros((NT, NDIR))          # block 0 only: the weights use the 20 cells
    for a in range(0, len(slot), CHUNK):
        b = min(a + CHUNK, len(slot))
        sl = slot[a:b]
        if (sl < 0).all():
            continue
        blk = np.asarray(X[a:b, :], dtype=np.float64)[:, keep]
        rw = row[a:b]
        for bi in range(3):
            sel = np.flatnonzero(sl == bi)
            if not len(sel):
                continue
            np.add.at(S[bi], rw[sel], blk[sel])
            np.add.at(Q[bi], rw[sel], blk[sel] ** 2)
            if bi == 0:
                p = blk[sel] @ U
                np.add.at(PS, rw[sel], p)
                np.add.at(PQ, rw[sel], p ** 2)
        print(f"   pass 2: {b}/{len(slot)} cells [{time.time() - t0:.0f}s]", flush=True)
    return S, Q, PS, PQ


def weights(S, Q, PS, PQ, ctl, G, beta=BETA, k=K):
    """a6_common.d1_w0y and d1_w1, ported verbatim onto the protein axis"""
    B = float(N0); n_c = float(ctl["n_control"])
    mu = S[0] / B
    m2 = ((mu - ctl["cm"][None, :]) ** 2).mean(1)
    vb = ((Q[0] - B * mu ** 2) / (B - 1.0)).mean(1)
    w0y = np.clip(1.0 - (vb / B + ctl["cvm"] / n_c) / np.maximum(m2, 1e-300), 0.0, 1.0)
    pc = (ctl["U"].T @ ctl["cm"])[:k]
    pY = PS[:, :k] / B - pc[None, :]
    sj = (PQ[:, :k] - B * (PS[:, :k] / B) ** 2) / (B - 1.0)
    a = m2 - beta * (pY ** 2).sum(1) / float(G)
    nu = vb / B + ctl["trC"] / (float(G) * n_c) - beta * (sj / B + ctl["lam"][None, :k] / n_c).sum(1) / float(G)
    W1 = np.clip(1.0 - nu / np.maximum(a, 1e-300), 0.0, 1.0)
    return w0y, W1


def main():
    t0 = time.time()
    h5 = A.arg("--h5ad"); out_dir = A.arg("--out", os.path.join(A.HERE, "results"))
    if not h5 or not os.path.isfile(h5):
        sys.exit("--h5ad <raw_contrastiveVI.h5ad> is required")
    integ = {"amendment_A14_sha256": A.checked_sha(os.path.join(A.BUNDLE, "AMENDMENT_A14.md"),
                                                   os.path.join(A.BUNDLE, "AMENDMENT_A14_SHA256.txt"), "amendment A14"),
             "script_sha256": A.sha256(os.path.abspath(__file__)), "panel_map_sha256": A.sha256(PANEL),
             "source_h5ad_md5": "f5a38ca2d1dfd7fa2b779788c525072b"}
    f, X, prot, keep, gt, codes, is_ntc = setup(h5)
    G = len(prot)
    ctl = pass1(X, keep, codes, is_ntc, G, t0)
    eligible, NT, slot, row = pass2(X, keep, codes, gt, is_ntc, G, t0)
    print(f"eligible targets (>= {MIN_CELLS} cells): {NT}", flush=True)
    S, Q, PS, PQ = accumulate(X, keep, slot, row, NT, G, ctl["U"], t0)
    targets = gt[eligible]

    floor = KTOP / float(NT - 1)
    P = json.load(open(PANEL, encoding="utf-8"))["mapping"]
    tpos = {t: i for i, t in enumerate(targets)}
    sq = np.full(NT, -1, np.int64)                       # sq[candidate] = the query it is excluded from
    n_own = 0
    for qi, p in enumerate(prot):
        c = tpos.get(P.get(p, "\x00"), -1)
        if c >= 0:
            sq[c] = qi; n_own += 1
    TA = (S[1] / float(NA) - ctl["cm"][None, :]).T       # (queries, candidates)
    TB = (S[2] / float(NB) - ctl["cm"][None, :]).T
    Y = (S[0] / float(N0) - ctl["cm"][None, :]).T
    bl = make_blocks(TA, 30)
    kp = np.random.default_rng(1401).random(NT); kt = np.random.default_rng(1402).random(NT)
    print(f"candidates {NT}, random rate {floor:.5f}; queries {G}; blocks {len(np.unique(bl))}; "
          f"own-gene column excluded for {n_own} of {G} proteins", flush=True)

    # --- gate (A14 section 4): the deep truth must replicate; no nomination involved ---
    gs = score_rows(TB, TA, sq, key_pred=kp, key_truth=kt)["exp"]
    gb = block_bootstrap(gs, np.full_like(gs, floor), bl)
    cor = np.array([np.corrcoef(TA[i], TB[i])[0, 1] for i in range(G)])
    gate = {"overlap_vs_random": summ(gb), "random_rate": floor, "median_overlap": float(np.median(gs.mean(axis=1))),
            "median_across_target_correlation": float(np.median(cor)), "passes": bool(gb["ci95"][0] > 0)}
    print(f"\ngate: deep truth A vs B top-{KTOP} overlap median {gate['median_overlap']:.4f}, above random "
          f"{fmt(gb)}, median correlation {gate['median_across_target_correlation']:.3f} -> "
          f"{'PASSES' if gate['passes'] else 'FAILS'}", flush=True)
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ,
           "note": "amendment A14; a same-experiment split inside SCITO, development on opened data, not an "
                   "independent confirmation and not part of the A4 to A8 evidence levels",
           "design": {"candidates": NT, "queries": G, "random_rate": floor,
                      "cells": {"input": N0, "truth": NA, "gate": NB}, "k": K, "beta": BETA, "ktop": KTOP,
                      "blocks": int(len(np.unique(bl))), "own_gene_excluded_for": n_own,
                      "control_ntc_cells": ctl["n_control"], "control_trace": ctl["trC"],
                      "top5_share_of_control_trace": float(ctl["lam"][:K].sum() / ctl["trC"])},
           "gate": gate}
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, "scito_shallow_protein.json")
    if not gate["passes"]:
        out["reading"] = {"branch": 1, "text": "gate failed: the deep protein truth does not replicate above the "
                          "random candidate rate, so no nomination method can be scored on it; A14 stops here"}
        json.dump(out, open(dest, "w", encoding="utf-8"), indent=1)
        print("\n" + out["reading"]["text"])
        return

    # --- the one comparison (A14 sections 3, 5 and 6) ---
    w0y, W1 = weights(S, Q, PS, PQ, ctl, G)
    z0, _ = weights(S, Q, PS, PQ, ctl, G, beta=0.0)
    E = {"raw": Y, "w0y": w0y[None, :] * Y, "W1": W1[None, :] * Y}
    sc = {a: score_rows(v, TA, sq, key_pred=kp, key_truth=kt)["exp"] for a, v in E.items()}
    arms = {a: {"mean": float(v.mean()),
                "above_random": summ(block_bootstrap(v, np.full_like(v, floor), bl))} for a, v in sc.items()}
    prim = summ(block_bootstrap(sc["W1"], sc["w0y"], bl))
    out["weights"] = {"beta0_max_abs_vs_w0y": float(np.abs(z0 - w0y).max()),
                      "W1_zero_fraction": float((W1 <= 0).mean()), "w0y_zero_fraction": float((w0y <= 0).mean()),
                      "W1_mean": float(W1.mean()), "w0y_mean": float(w0y.mean()),
                      "W1_minus_w0y_mean": float((W1 - w0y).mean()),
                      "W1_quantiles": [float(q) for q in np.percentile(W1, [5, 25, 50, 75, 95])]}
    out["arms"] = arms; out["W1_vs_w0y"] = prim
    out["W1_vs_raw"] = summ(block_bootstrap(sc["W1"], sc["raw"], bl))
    out["w0y_vs_raw"] = summ(block_bootstrap(sc["w0y"], sc["raw"], bl))
    clear, weak = 0.90 * floor, 0.45 * floor
    d, lo = prim["diff"], prim["ci95"][0]
    if lo <= 0 or d < weak:
        branch, reading = 2, ("no clear same-budget gain; A14 section 6 branch 2: stop, and do not change the query "
                              "set, the depth, k, beta or the thresholds, and build no further explanation around "
                              "this dataset")
    elif d < clear:
        branch, reading = 3, "a weak gain: at or above 0.45 times the random rate with the lower bound above zero"
    else:
        branch, reading = 4, "a clear gain: at or above 0.90 times the random rate with the lower bound above zero"
    out["thresholds"] = {"weak": weak, "clear": clear}
    out["reading"] = {"branch": branch, "text": reading,
                      "positioning": "a same-experiment split, so it adds generality across measurement modality; it "
                                     "is not independent validation and implies nothing about the RNA results"}
    json.dump(out, open(dest, "w", encoding="utf-8"), indent=1)
    print(f"\nbeta = 0 reproduces w0y to {out['weights']['beta0_max_abs_vs_w0y']:.2e}; "
          f"W1 clipped to zero on {out['weights']['W1_zero_fraction']:.3f} of targets "
          f"(w0y {out['weights']['w0y_zero_fraction']:.3f})")
    for a in ("raw", "w0y", "W1"):
        print(f"  {a:4s} P@25 {arms[a]['mean']:.5f}  above random {fmt(arms[a]['above_random'])}")
    print(f"W1 - w0y {fmt(prim)}   thresholds: weak {weak:.5f}, clear {clear:.5f}")
    print(f"branch {branch}: {reading} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
