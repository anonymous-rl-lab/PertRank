"""Amendment A12b section 3: how far each B8 arm sits above the random-candidate floor, with an interval.

The B8 result document reported three bare point estimates ("best arm minus the random floor") and drew a
conclusion about the truth from them. Amendment A12b retracts that conclusion and replaces the bare numbers with
this quantity: for every decidable donor, every arm (raw, w0y, W1) and both judging directions, the arm's
tie-expected P@25 minus the donor's random-candidate rate, bootstrapped exactly as B8 bootstraps its paired
differences (B = 2000, seed 2026, the donor's responsive blocks). Only the subtrahend changes, from the other arm's
per-query scores to a constant.

It is a statement about the truth as an instrument, never about W1: see amendment A12b sections 2 and 3. Nothing
here re-runs or alters B8, whose frozen scripts and stored numbers stay as they are.

    python3 stage3/b8_floor_headroom.py [--out stage3/results]
"""
import json
import os
import time

import numpy as np

import a6_common as A
from a6_common import block_bootstrap, exp_scores, fmt, summ
import cross_donor_b2 as cdb  # noqa: E402
import evaluate_b8_protein_readout as B8  # noqa: E402
import lock_b7_sources as L  # noqa: E402
from pr10 import score_rows  # noqa: E402

KTOP = B8.KTOP


def main():
    t0 = time.time()
    out_dir = A.arg("--out", os.path.join(A.HERE, "results"))
    integ = {"amendment_A12b_sha256": A.checked_sha(os.path.join(A.BUNDLE, "AMENDMENT_A12b.md"),
                                                    os.path.join(A.BUNDLE, "AMENDMENT_A12b_SHA256.txt"), "amendment A12b"),
             "script_sha256": A.sha256(os.path.abspath(__file__)), **B8.integrity()}
    P = json.load(open(B8.PANEL, encoding="utf-8"))["mapping"]
    Sc = np.load(B8.SCITO, allow_pickle=True)
    s_t = [str(x) for x in Sc["targets"]]; s_p = [str(x) for x in Sc["proteins"]]
    Z = Sc["Z"].astype(np.float64)
    trow = {t: i for i, t in enumerate(s_t)}; pcol = {p: i for i, p in enumerate(s_p)}
    gene_to_label = {g: l for l, g in P.items()}
    D = B8.donors(np.load(L.LOCK_NPZ, allow_pickle=True))
    res = {}
    for d, S in D.items():
        names, ids, sym = S["names"], S["ids"], S["sym"]
        grow = {g: i for i, g in enumerate(S["genes"])}
        C = np.array([j for j, n in enumerate(names) if n in trow], np.int64)
        ct = np.array([trow[names[j]] for j in C], np.int64)
        floor = KTOP / max(len(C) - 1, 1)
        q, prot = [], []
        for g in S["resp"]:
            lab = gene_to_label.get(sym.get(g, g))
            if lab is not None and lab in pcol and g in grow:
                q.append(g); prot.append(pcol[lab])
        if not (len(C) >= B8.MIN_C and len(q) >= B8.MIN_Q):
            res[d] = {"decidable": False}
            continue
        bl = S["blocks"][[S["resp"].index(g) for g in q]]
        Y = S["Y20"][[grow[g] for g in q]][:, C]
        Zq = Z[np.ix_(ct, prot)].T
        sq = cdb.id_exclusion(ids[C], q)
        E = {"raw": Y, "w0y": S["w0y"][C][None, :] * Y, "W1": S["W1"][C][None, :] * Y}
        info = {"decidable": True, "candidates": int(len(C)), "queries": len(q), "random_floor": float(floor),
                "arms": {}}
        for arm, T, dirless in (("signed (primary)", -Zq, False), ("magnitude (diagnostic)", np.abs(Zq), True)):
            c = {"T": T, "sq": sq, "KP": S["key_pred"][C], "KT": S["key_truth"][C]}
            if dirless:
                sc = {a: score_rows(np.abs(v), c["T"], c["sq"], key_pred=c["KP"], key_truth=c["KT"],
                                    pred_dir_invariant=True)["exp"][:, :1] for a, v in E.items()}
            else:
                sc = {a: exp_scores(v, c) for a, v in E.items()}
            info["arms"][arm] = {a: summ(block_bootstrap(v, np.full_like(v, floor), bl)) for a, v in sc.items()}
            for a, v in info["arms"][arm].items():
                print(f"{d:3s} {arm:22s} {a:4s} above floor {fmt(v)}"
                      f"  {'excludes the floor' if v['ci95'][0] > 0 else 'does not exclude the floor'}", flush=True)
        res[d] = info
    # the reading written down in amendment A12b section 3, applied mechanically
    any_clear = [(d, arm, a) for d, i in res.items() if i.get("decidable")
                 for arm, ar in i["arms"].items() for a, v in ar.items() if v["ci95"][0] > 0]
    reading = ("the truth has some discriminating power: " + ", ".join(f"{d}/{arm}/{a}" for d, arm, a in any_clear)
               + "; B8's pooled null is then a null about W1's list changes on this transfer task, which weakens "
                 "explanation (b) of amendment A12b section 2"
               if any_clear else
               "no arm's interval excludes the floor on any donor: the truth's discriminating power is NOT "
               "established, which is not the same as showing it has none; the three explanations of amendment "
               "A12b section 2 stay unseparated")
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ,
           "note": "amendment A12b section 3; a statement about the truth as an instrument, never about W1. "
                   "B8's own numbers are unchanged.",
           "donors": res, "arms_clearing_the_floor": [{"donor": d, "arm": arm, "weighting": a} for d, arm, a in any_clear],
           "reading": reading}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(out_dir, "b8_floor_headroom.json"), "w", encoding="utf-8"), indent=1)
    print("\nreading:", reading, f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
