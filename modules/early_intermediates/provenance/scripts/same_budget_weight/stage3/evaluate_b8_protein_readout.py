"""Amendments A12 and A12a: B8, the locked 20-cell nominations of the three opened T cell donors scored against
an independent protein readout (the recomputed SCITO CRISPRa gene x protein matrix).

Primary    responsive lists; prediction = locked Y20 x weight, ranked within the candidates; truth = the protein's
           Z with the sign reversed (our "up" direction takes -Z, "down" takes +Z); tie-expected P@25; paired block
           bootstrap on the donor's 30 responsive blocks, B = 2000, seed 2026; W1 - w0y pooled over the decidable
           donors with equal weight, the pooled interval being the average of the donors' endpoints.
Diagnostic direction-free arm: one list per query, predictions ranked by |w y|, truth by |Z| (A12a section 5). It
           interprets a primary interval that crosses zero; it is never a second primary quantity.

    python3 stage3/evaluate_b8_protein_readout.py [--out stage3/results]
"""
import csv
import json
import os
import sys
import time

import numpy as np

import a6_common as A
from a6_common import block_bootstrap, exp_scores, fmt, summ
import cross_donor_b2 as cdb  # noqa: E402
import evaluate_d2_prospective as e2  # noqa: E402
import lock_b7_sources as L  # noqa: E402
from pr10 import score_rows, self_to_query  # noqa: E402

KTOP, MIN_C, MIN_Q = 25, 150, 5
SCITO = os.path.join(A.HERE, "external", "scito", "scito_gene_protein_z.npz")
PANEL = os.path.join(A.HERE, "external", "scito", "scito_panel_gene_map.json")


def integrity():
    out = {}
    for n in ("A12", "A12a"):
        out[f"amendment_{n}_sha256"] = A.checked_sha(os.path.join(A.BUNDLE, f"AMENDMENT_{n}.md"),
                                                     os.path.join(A.BUNDLE, f"AMENDMENT_{n}_SHA256.txt"), f"amendment {n}")
    rec = A.record(os.path.join(A.HERE, "A12_FREEZE_SHA256.txt"))
    for s in ("compute_scito_zscores.py", "evaluate_b8_protein_readout.py"):
        h = A.sha256(os.path.join(A.HERE, s))
        if rec.get(s) != h:
            sys.exit(f"{s} differs from the A12 freeze record")
        out.setdefault("freeze_A12", {})[s] = h
    out["scito_matrix_sha256"] = A.sha256(SCITO)
    out["panel_map_sha256"] = A.sha256(PANEL)
    return out


def donors(Lk):
    """the three opened donors' locked 20-cell predictions, weights, responsive lists and tie keys"""
    sym = json.load(open(os.path.join(A.HERE, "data", "tcell_gene_symbols.json"), encoding="utf-8"))["symbols"]
    out = {}
    for d in ("D3", "D1"):
        S = {k[len(d) + 1:]: Lk[k] for k in Lk.files if k.startswith(d + "_")}
        out[d] = {"names": [str(x) for x in S["names"]], "ids": np.array([str(x) for x in S["ids"]]),
                  "genes": [str(g) for g in S["genes"]], "Y20": S["Y20"], "w0y": S["w0y"], "W1": S["W1"],
                  "resp": [str(g) for g in S["resp"]], "blocks": np.asarray(S["resp_blocks"]),
                  "key_pred": S["key_pred"], "key_truth": S["key_truth"], "sym": sym}
    Z2 = np.load(os.path.join(A.HERE, "data", "d2_prospective_state.npz"), allow_pickle=True)
    g2 = np.array([str(g) for g in Z2["genes_keep"]]); qu = Z2["qu"]
    NT = len(Z2["names"])
    out["D2"] = {"names": [str(x) for x in Z2["names"]], "ids": np.array([str(x) for x in Z2["ids"]]),
                 "genes": [str(g) for g in g2[qu]], "Y20": Z2["S0_q"] / 20.0 - Z2["cm"][qu][:, None],
                 "w0y": e2.w0y(Z2, 20), "W1": e2.w1(Z2, 20), "resp": [str(g) for g in g2[Z2["q_resp"]]],
                 "blocks": np.asarray(Z2["blocks_responsive"]),
                 "key_pred": np.random.default_rng(101).random(NT),
                 "key_truth": np.random.default_rng(202).random(NT), "sym": sym}
    return out


def turnover_support(E1, E0, T, sq, directionless):
    """expected slots changed and the truth-side support of added and removed candidates"""
    from pr10.scoring import _boundary
    NQ = T.shape[0]; add = rem = mass = 0.0; nc = []
    for qi in range(NQ):
        keep = np.flatnonzero(sq != qi)
        for sgn in ((1.0,) if directionless else (1.0, -1.0)):
            p1 = _boundary(E1[qi, keep] * sgn, KTOP)[0]; p0 = _boundary(E0[qi, keep] * sgn, KTOP)[0]
            pt = _boundary(T[qi, keep] * sgn, KTOP)[0]
            up = np.clip(p1 - p0, 0, None); dn = np.clip(p0 - p1, 0, None)
            add += up @ pt; rem += dn @ pt; mass += up.sum(); nc.append(KTOP / len(keep))
    return {"slots_changed_total": float(mass), "support_added": (float(add / mass) if mass > 0 else None),
            "support_removed": (float(rem / mass) if mass > 0 else None),
            "support_difference": (float((add - rem) / mass) if mass > 0 else None),
            "random_candidate_rate": float(np.mean(nc))}


def main():
    t0 = time.time()
    out_dir = A.arg("--out", os.path.join(A.HERE, "results"))
    integ = integrity()
    P = json.load(open(PANEL, encoding="utf-8"))["mapping"]
    Sc = np.load(SCITO, allow_pickle=True)
    s_t = [str(x) for x in Sc["targets"]]; s_p = [str(x) for x in Sc["proteins"]]
    Z = Sc["Z"].astype(np.float64)
    trow = {t: i for i, t in enumerate(s_t)}; pcol = {p: i for i, p in enumerate(s_p)}
    gene_to_label = {g: l for l, g in P.items()}
    Lk = np.load(L.LOCK_NPZ, allow_pickle=True)
    D = donors(Lk)
    res = {"sources": {}, "pooled": {}}
    rows = []
    for d, S in D.items():
        names, ids, sym = S["names"], S["ids"], S["sym"]
        grow = {g: i for i, g in enumerate(S["genes"])}
        C = np.array([j for j, n in enumerate(names) if n in trow], np.int64)
        ct = np.array([trow[names[j]] for j in C], np.int64)
        info = {"candidates": int(len(C)), "random_floor": float(KTOP / max(len(C) - 1, 1)),
                "cohort": len(names), "arms": {}}
        res["sources"][d] = info
        # matched queries: responsive genes whose symbol is a mapped panel protein present in the matrix
        q, prot = [], []
        for g in S["resp"]:
            lab = gene_to_label.get(sym.get(g, g))
            if lab is not None and lab in pcol and g in grow:
                q.append(g); prot.append(pcol[lab])
        info["matched_queries"] = len(q)
        info["decidable"] = bool(len(C) >= MIN_C and len(q) >= MIN_Q)
        info["query_proteins"] = [s_p[p] for p in prot]
        if not info["decidable"]:
            info["note"] = "too few candidates or matched queries"
            continue
        keep_q = [S["resp"].index(g) for g in q]
        bl = S["blocks"][keep_q]
        Y = S["Y20"][[grow[g] for g in q]][:, C]
        Zq = Z[np.ix_(ct, prot)].T                                  # (queries, candidates)
        sq = cdb.id_exclusion(ids[C], q)
        E = {"raw": Y, "w0y": S["w0y"][C][None, :] * Y, "W1": S["W1"][C][None, :] * Y}
        c_signed = {"T": -Zq, "sq": sq, "KP": S["key_pred"][C], "KT": S["key_truth"][C]}
        c_mag = {"T": np.abs(Zq), "sq": sq, "KP": S["key_pred"][C], "KT": S["key_truth"][C]}
        for arm, c, dirless in (("signed (primary)", c_signed, False), ("magnitude (diagnostic)", c_mag, True)):
            if dirless:
                sc = {a: score_rows(np.abs(v), c["T"], c["sq"], key_pred=c["KP"], key_truth=c["KT"],
                                    pred_dir_invariant=True)["exp"][:, :1] for a, v in E.items()}
            else:
                sc = {a: exp_scores(v, c) for a, v in E.items()}
            info["arms"][arm] = {
                "queries": len(q), "blocks": int(len(np.unique(bl))),
                "means": {a: float(v.mean()) for a, v in sc.items()},
                "W1_vs_w0y": summ(block_bootstrap(sc["W1"], sc["w0y"], bl)),
                "W1_vs_raw": summ(block_bootstrap(sc["W1"], sc["raw"], bl)),
                "w0y_vs_raw": summ(block_bootstrap(sc["w0y"], sc["raw"], bl)),
                "turnover": turnover_support(np.abs(E["W1"]) if dirless else E["W1"],
                                             np.abs(E["w0y"]) if dirless else E["w0y"], c["T"], c["sq"], dirless)}
            print(f"{d:3s} {arm:22s} |C| {len(C):4d} q {len(q):3d}  raw {info['arms'][arm]['means']['raw']:.4f} "
                  f"w0y {info['arms'][arm]['means']['w0y']:.4f} W1 {info['arms'][arm]['means']['W1']:.4f}  "
                  f"W1-w0y {fmt(info['arms'][arm]['W1_vs_w0y'])}", flush=True)
        for g, p in zip(q, prot):
            rows.append({"donor": d, "query_gene": g, "symbol": sym.get(g, g), "panel_protein": s_p[p],
                         "candidates": len(C)})
    dec = [d for d in D if res["sources"][d].get("decidable")]
    for arm in ("signed (primary)", "magnitude (diagnostic)"):
        bs = [res["sources"][d]["arms"][arm]["W1_vs_w0y"] for d in dec]
        res["pooled"][arm] = {"donors": dec, "diff": float(np.mean([b["diff"] for b in bs])),
                              "ci95": [float(np.mean([b["ci95"][0] for b in bs])),
                                       float(np.mean([b["ci95"][1] for b in bs]))]}
    pr = res["pooled"]["signed (primary)"]
    reading = ("the list changes gain more support in the independent protein readout" if pr["ci95"][0] > 0
               else "no support seen in the independent protein readout")
    mg = res["pooled"]["magnitude (diagnostic)"]
    if pr["ci95"][0] <= 0:
        reading += ("; direction-free arm supported" if mg["ci95"][0] > 0 else "; direction-free arm not supported")
    res["reading"] = reading
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ,
           "note": "B8 under amendments A12 and A12a; truth is a recomputed SCITO CRISPRa matrix, not the authors' own",
           "calibration": json.loads(str(Sc["calibration"])), **res}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(out_dir, "b8_protein_readout.json"), "w", encoding="utf-8"), indent=1)
    with open(os.path.join(out_dir, "b8_protein_readout_queries.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\npooled primary:", fmt(pr), "| diagnostic:", fmt(mg))
    print("reading:", reading, f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
