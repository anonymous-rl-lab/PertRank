"""Amendment A11, question 1: are the three donors' leading input-control directions the same directions, and what
genes carry the first one? Post hoc; stored states only.

    python3 stage3/direction1_q1_similarity.py [--out stage3/results]

Writes results/direction1_similarity.json and results/direction1_loadings.csv (the first direction's loading of every
weight gene of each donor, sign-aligned to D3).
"""
import csv
import json
import os
import time

import numpy as np
from scipy.stats import spearmanr

import a6_common as A
import evaluate_stage2 as es  # noqa: E402

TOPN = 40
FAMILIES = {"ribosomal (RPL/RPS)": ("RPL", "RPS"), "mitochondrial (MT-)": ("MT-",),
            "histone (H1/H2/H3/H4)": ("H1", "H2", "H3", "H4"), "MHC (HLA-)": ("HLA-",),
            "interferon-stimulated (IFI/ISG/MX/OAS)": ("IFI", "ISG", "MX", "OAS")}


def donors():
    D = es.load("e4_tcell.npz")
    Z3 = np.load(os.path.join(A.HERE, "data", "e4_w1_stage3.npz"), allow_pickle=True)
    if not np.array_equal(D["keep"], Z3["keep"]):
        raise SystemExit("D3: the W1 state and the reduced state keep different genes")
    Z1 = A.d1_load()
    Z2 = np.load(os.path.join(A.HERE, "data", "d2_prospective_state.npz"), allow_pickle=True)
    out = {"D3": (np.array([str(g) for g in D["genes_keep"]]), Z3["directions"], D["cm_keep"].astype(np.float64),
                  Z3["eigenvalues"], float(Z3["control_trace"])),
           "D1": (np.array([str(g) for g in Z1["genes_keep"]]), Z1["directions"], Z1["cm"], Z1["eigenvalues"], float(Z1["control_trace"])),
           "D2": (np.array([str(g) for g in Z2["genes_keep"]]), Z2["directions"], Z2["cm"], Z2["eigenvalues"], float(Z2["control_trace"]))}
    for d, (g, U, cm, lam, tr) in out.items():
        if len(g) != U.shape[0] or len(g) != len(cm):
            raise SystemExit(f"{d}: gene axis, directions and control mean disagree")
    return out


def restrict(U, rows):
    """directions restricted to shared genes: unit columns, plus an orthonormal basis of their span (QR).
    Restriction can destroy orthogonality, so subspace shares use the QR basis, not a sum of squared cosines."""
    X = U[rows]
    N = X / np.linalg.norm(X, axis=0, keepdims=True)
    Q, _ = np.linalg.qr(X)
    off = float(np.abs(N.T @ N - np.eye(N.shape[1])).max())
    return N, Q, off


def pair(ga, Ua, gb, Ub, nrep=5):
    shared = sorted(set(ga) & set(gb))
    ia = {g: i for i, g in enumerate(ga)}; ib = {g: i for i, g in enumerate(gb)}
    Na, Qa, offa = restrict(Ua, [ia[g] for g in shared])
    Nb, Qb, offb = restrict(Ub, [ib[g] for g in shared])
    cos = Na.T @ Nb
    share_ab = [float(((Qb.T @ Na[:, i]) ** 2).sum()) for i in range(nrep)]
    share_ba = [float(((Qa.T @ Nb[:, j]) ** 2).sum()) for j in range(nrep)]
    naive_ab = [float((cos[i, :] ** 2).sum()) for i in range(nrep)]
    naive_ba = [float((cos[:, j] ** 2).sum()) for j in range(nrep)]
    gap = max(max(abs(x - y) for x, y in zip(share_ab, naive_ab)), max(abs(x - y) for x, y in zip(share_ba, naive_ba)))
    return len(shared), cos, share_ab, share_ba, (offa, offb, gap)


def main():
    t0 = time.time()
    out_dir = A.arg("--out", os.path.join(A.HERE, "results"))
    integ = {"amendment_A11_sha256": A.checked_sha(os.path.join(A.BUNDLE, "AMENDMENT_A11.md"),
                                                   os.path.join(A.BUNDLE, "AMENDMENT_A11_SHA256.txt"), "amendment A11"),
             "script_sha256": A.sha256(os.path.abspath(__file__))}
    sym = json.load(open(os.path.join(A.HERE, "data", "tcell_gene_symbols.json"), encoding="utf-8"))["symbols"]
    dn = donors()
    names = ["D3", "D1", "D2"]
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n, cos, sab, sba, chk = pair(dn[a][0], dn[a][1], dn[b][0], dn[b][1])
            pairs[f"{a} vs {b}"] = {
                "shared_genes": n,
                "cosine_directions_1_to_5": [[round(float(x), 4) for x in row[:5]] for row in cos[:5]],
                f"share of each {a} direction inside {b} top-20 subspace": [round(x, 3) for x in sab],
                f"share of each {b} direction inside {a} top-20 subspace": [round(x, 3) for x in sba],
                "orthonormality_after_restriction": {
                    f"max |off-diagonal| {a}": round(chk[0], 6), f"max |off-diagonal| {b}": round(chk[1], 6),
                    "max |QR share - sum of squared cosines|": round(chk[2], 6)}}
    sign = {"D3": 1.0, "D1": float(np.sign(pairs["D3 vs D1"]["cosine_directions_1_to_5"][0][0])),
            "D2": float(np.sign(pairs["D3 vs D2"]["cosine_directions_1_to_5"][0][0]))}
    per = {}; rows = []; tops = {}
    for d in names:
        g, U, cm, lam, tr = dn[d]
        u = U[:, 0] * sign[d]; sq = u ** 2
        order = np.argsort(-np.abs(u))
        tops[d] = [g[i] for i in order[:TOPN]]
        fam = {}
        for fname, pref in FAMILIES.items():
            mask = np.array([any(sym.get(x, "").startswith(p) for p in pref) for x in g])
            fam[fname] = {"genes": int(mask.sum()), "share_of_squared_loading": round(float(sq[mask].sum()), 4),
                          "mean_signed_loading": (round(float(u[mask].mean()), 5) if mask.any() else None)}
        per[d] = {"sign_relative_to_D3": sign[d],
                  "eigenvalue_share_of_direction_1": round(float(lam[0] / tr), 4),
                  "spearman_loading_vs_log_control_mean": round(float(spearmanr(u, np.log1p(cm)).correlation), 3),
                  "spearman_abs_loading_vs_log_control_mean": round(float(spearmanr(np.abs(u), np.log1p(cm)).correlation), 3),
                  "share_of_squared_loading_on_positive_side": round(float(sq[u > 0].sum()), 3),
                  "genes_carrying_half_of_squared_loading": int(np.searchsorted(np.cumsum(np.sort(sq)[::-1]), 0.5) + 1),
                  "top_loading_genes": [{"symbol": sym.get(g[i], g[i]), "gene": g[i], "loading": round(float(u[i]), 5),
                                         "control_mean": round(float(cm[i]), 4)} for i in order[:TOPN]],
                  "families": fam}
        for i in order:
            rows.append({"donor": d, "gene": g[i], "symbol": sym.get(g[i], g[i]), "loading_direction_1": float(u[i]),
                         "control_mean": float(cm[i])})
    overlap = {f"{a} & {b}": len(set(tops[a]) & set(tops[b])) for i, a in enumerate(names) for b in names[i + 1:]}
    overlap["all three"] = len(set(tops["D3"]) & set(tops["D1"]) & set(tops["D2"]))
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ,
           "note": "amendment A11 question 1; post hoc; directions are the stored input-control eigenvectors",
           "pairs": pairs, "per_donor": per, f"top{TOPN}_overlap": overlap}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "direction1_similarity.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(out_dir, "direction1_loadings.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for k, v in pairs.items():
        print(k, "shared", v["shared_genes"], "cos(1,1) %.3f" % v["cosine_directions_1_to_5"][0][0])
    for d in names:
        p = per[d]
        print(d, "sign", p["sign_relative_to_D3"], "lam1 share", p["eigenvalue_share_of_direction_1"],
              "rho(load, log cm)", p["spearman_loading_vs_log_control_mean"], "top:",
              [x["symbol"] for x in p["top_loading_genes"][:12]])
    print("overlap", overlap, f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
