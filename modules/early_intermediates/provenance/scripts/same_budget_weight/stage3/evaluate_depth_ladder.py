"""Amendments A13 and A13a: the depth ladder. Does measuring more cells buy agreement with the independent protein
readout, and does the deep list reach demonstrable absolute correspondence at all.

Primary   Delta = P@25(100 cells) - P@25(20 cells), tie-expected, paired by query. Answers only the return on more
          cells (A13a section 2). Pooled as the equal-weight average of the three donors' means; the primary pooling
          convention is A12's endpoint averaging, with the joint and cluster conventions reported beside it under
          their stated assumptions (A13a section 3).
Second    P@25(100 cells) - the random-candidate rate. Answers whether the deep list reaches correspondence at all.
          It has no pairing, so its interval is wider by construction and only supports "established / not
          established" (A13a section 2).
Arms      unweighted profiles at 20 cells (block 0), 50 cells (block 2, D1 and D2), 100 cells (block 4). The
          per-target cell blocks are disjoint, so no rung shares cells with another. Centring does not affect the
          primary arm: it shifts a whole query row, and the tie-expected P@25 depends only on within-row order
          (A13a section 5), so D3's mixed centring does not downgrade its primary arm.
Reading   the four branches of A13a section 6, applied mechanically against the working bound of 0.0090.

No weights anywhere: W1, w0y, k and beta are untouched, and B8's own numbers are not recomputed.

    python3 stage3/evaluate_depth_ladder.py [--out stage3/results]
"""
import json
import os
import time

import numpy as np

import a6_common as A
from a6_common import block_bootstrap, exp_scores, summ
import cross_donor_b2 as cdb  # noqa: E402
import evaluate_b8_protein_readout as B8  # noqa: E402
import lock_b7_sources as L  # noqa: E402
import sizing_depth_ladder as SZ  # noqa: E402
from pr10 import score_rows  # noqa: E402

KTOP = B8.KTOP
BOUND = 0.0090          # A13a section 4: the largest null half-width of the primary convention over 32 permutations


def integrity():
    out = {}
    for n in ("A12", "A12a", "A12b", "A13", "A13a"):
        out[f"amendment_{n}_sha256"] = A.checked_sha(os.path.join(A.BUNDLE, f"AMENDMENT_{n}.md"),
                                                     os.path.join(A.BUNDLE, f"AMENDMENT_{n}_SHA256.txt"), f"amendment {n}")
    out["script_sha256"] = A.sha256(os.path.abspath(__file__))
    out["sizing_script_sha256"] = A.sha256(os.path.join(A.HERE, "sizing_depth_ladder.py"))
    out["scito_matrix_sha256"] = A.sha256(B8.SCITO)
    out["panel_map_sha256"] = A.sha256(B8.PANEL)
    return out


def pooled_endpoints(per):
    """the A12 convention: average the donors' point estimates and their interval endpoints"""
    return {"diff": float(np.mean([p["diff"] for p in per])),
            "ci95": [float(np.mean([p["ci95"][0] for p in per])), float(np.mean([p["ci95"][1] for p in per]))]}


def main():
    t0 = time.time()
    out_dir = A.arg("--out", os.path.join(A.HERE, "results"))
    integ = integrity()
    Sc = np.load(B8.SCITO, allow_pickle=True)
    Z = Sc["Z"].astype(np.float64)
    trow = {t: i for i, t in enumerate(str(x) for x in Sc["targets"])}
    pcol = {p: i for i, p in enumerate(str(x) for x in Sc["proteins"])}
    s_p = [str(x) for x in Sc["proteins"]]
    g2l = {g: l for l, g in json.load(open(B8.PANEL, encoding="utf-8"))["mapping"].items()}
    Lk = np.load(L.LOCK_NPZ, allow_pickle=True)
    D = B8.donors(Lk)
    pack, rec, donor = {}, [], []
    for d, S in D.items():
        grow = {g: i for i, g in enumerate(S["genes"])}
        C = np.array([j for j, n in enumerate(S["names"]) if n in trow], np.int64)
        ct = np.array([trow[S["names"][j]] for j in C], np.int64)
        q, prot = [], []
        for g in S["resp"]:
            lab = g2l.get(S["sym"].get(g, g))
            if lab is not None and lab in pcol and g in grow:
                q.append(g); prot.append(pcol[lab])
        if not (len(C) >= B8.MIN_C and len(q) >= B8.MIN_Q):
            continue
        bl = S["blocks"][[S["resp"].index(g) for g in q]]
        YY, centring = SZ.rungs(d, S, Lk)
        pack[d] = {"Y": {k: v[[grow[g] for g in q]][:, C] for k, v in YY.items()},
                   "Zq": Z[np.ix_(ct, prot)].T, "sq": cdb.id_exclusion(S["ids"][C], q),
                   "KP": S["key_pred"][C], "KT": S["key_truth"][C], "bl": bl, "q": q,
                   "floor": KTOP / max(len(C) - 1, 1), "centring": centring, "candidates": int(len(C))}
        for g, b, p in zip(q, bl, prot):
            rec.append((d, int(b), int(p))); donor.append(d)
    donor = np.array(donor)
    comp = SZ.components(rec)
    blkid = np.array([f"{d}_{b}" for d, b, _ in rec])

    res = {"donors": {}, "pooled": {}}
    for arm, dirless in (("signed (primary)", False), ("magnitude (diagnostic)", True)):
        dep = {}
        per_delta, per_floor = [], []
        for d, p in pack.items():
            c = {"T": (np.abs(p["Zq"]) if dirless else -p["Zq"]), "sq": p["sq"], "KP": p["KP"], "KT": p["KT"]}
            sc = {}
            for k, Yk in p["Y"].items():
                if dirless:
                    sc[k] = score_rows(np.abs(Yk), c["T"], c["sq"], key_pred=c["KP"], key_truth=c["KT"],
                                       pred_dir_invariant=True)["exp"][:, :1]
                else:
                    sc[k] = exp_scores(Yk, c)
            info = res["donors"].setdefault(d, {"candidates": p["candidates"], "queries": len(p["q"]),
                                                "blocks_nonempty": int(len(np.unique(p["bl"]))),
                                                "random_floor": p["floor"], "centring": p["centring"], "arms": {}})
            a = info["arms"][arm] = {"rungs": {k: float(v.mean()) for k, v in sc.items()},
                                    "delta_100_vs_20": summ(block_bootstrap(sc["100"], sc["20"], p["bl"])),
                                    "deep_minus_floor": summ(block_bootstrap(sc["100"], np.full_like(sc["100"], p["floor"]), p["bl"])),
                                    "shallow_minus_floor": summ(block_bootstrap(sc["20"], np.full_like(sc["20"], p["floor"]), p["bl"]))}
            if "50" in sc:      # descriptive only, amendment A13a section 6
                a["shape_50_vs_20"] = summ(block_bootstrap(sc["50"], sc["20"], p["bl"]))
                a["shape_100_vs_50"] = summ(block_bootstrap(sc["100"], sc["50"], p["bl"]))
            per_delta.append(a["delta_100_vs_20"]); per_floor.append(a["deep_minus_floor"])
            dep[d] = (sc["100"] - sc["20"]).mean(axis=1)
            print(f"{d:3s} {arm:22s} " + " ".join(f"{k:>3s}c {v:.4f}" for k, v in sorted(a['rungs'].items(), key=lambda x: float(x[0])))
                  + f"  floor {p['floor']:.4f}  delta {a['delta_100_vs_20']['diff']:+.4f} "
                    f"[{a['delta_100_vs_20']['ci95'][0]:+.4f}, {a['delta_100_vs_20']['ci95'][1]:+.4f}]", flush=True)
        dd = np.concatenate([dep[d] for d in pack])
        pooled = {"endpoints (primary)": pooled_endpoints(per_delta)}
        for key, grp, note in (("joint", blkid, "treats records in different donors as independent; optimistic"),
                               ("cluster", comp, "same donor block or same protein, fully dependent inside; pessimistic")):
            r = SZ.donor_mean_bootstrap(dd, grp, donor)
            pooled[key] = {"diff": r["diff"], "ci95": r["ci95"], "groups": r["groups"],
                           "empty_donor_draws": r["empty_donor_draws"], "assumption": note}
        res["pooled"][arm] = {"delta_100_vs_20": pooled,
                              "deep_minus_floor_endpoints": pooled_endpoints(per_floor)}
    # amendment A13a section 6, applied mechanically
    pr = res["pooled"]["signed (primary)"]["delta_100_vs_20"]["endpoints (primary)"]
    fl = res["pooled"]["signed (primary)"]["deep_minus_floor_endpoints"]
    lo, hi = pr["ci95"]
    if lo > 0:
        branch, reading = 1, ("more cells does buy agreement; " + ("the deep list also reaches demonstrable absolute "
                              "correspondence" if fl["ci95"][0] > 0 else "the gain exists but neither depth reaches "
                              "demonstrable absolute correspondence"))
    elif hi < 0:
        branch, reading = 3, ("more cells makes agreement worse; the 20-cell agreement is not coming from a better "
                              "estimate of the CRISPRi effect. Reported on its own, not folded into another branch")
    elif hi < BOUND:
        branch, reading = 2, (f"a mean depth gain of {BOUND:.4f} or more is excluded (about {BOUND * KTOP:.2f} hits "
                              "per list of 25); this says nothing by itself about whether correspondence exists")
    else:
        branch, reading = 4, "undetermined at this resolution; a legitimate result, and no branch is forced"
    res["reading"] = {"branch": branch, "working_bound": BOUND, "text": reading,
                      "B8_unchanged": "B8's reading stays 'no support seen in the independent protein readout'; no "
                                      "verdict on W1 follows from this amendment (A13a sections 1 and 6)"}
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ,
           "note": "amendments A13 and A13a; unweighted depth arms only, no W1 or w0y anywhere",
           "structure": {"records": len(rec), "distinct_proteins": len({p for _, _, p in rec}),
                         "blocks_nonempty_total": int(len(np.unique(blkid))), "clusters": int(comp.max() + 1)},
           **res}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(out_dir, "depth_ladder.json"), "w", encoding="utf-8"), indent=1)
    print(f"\npooled primary delta (endpoint averaging): {pr['diff']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    for k in ("joint", "cluster"):
        v = res["pooled"]["signed (primary)"]["delta_100_vs_20"][k]
        print(f"  {k:8s} {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  ({v['groups']} groups)")
    print(f"deep minus floor (endpoint averaging): {fl['diff']:+.4f} [{fl['ci95'][0]:+.4f}, {fl['ci95'][1]:+.4f}]")
    print(f"branch {branch}: {reading} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
