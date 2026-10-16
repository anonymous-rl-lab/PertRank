"""Amendment A6 section 4 (with A6a section 3): cross-donor support B2. Frozen before it produced anything
(stage3/A6_FREEZE_SHA256.txt).

The 20-cell predictions of one T-cell donor (raw, w0y, W1, each weight computed on that donor exactly as before)
are scored against the other donor's deep-block truth, on the targets both donors carry and on the source donor's
own query lists restricted to genes inside the other donor's query union. Own-gene entries are excluded through
Ensembl identifiers (D3: inputs/own_gene_map.npz; D1: the A6a map). The same targets and queries scored against
the source donor's own truth are the within-donor reference.

Candidate turnover, per query and direction, from the prediction-side boundary probabilities p of the tie-expected
scoring: added mass max(p_W1 - p_w0y, 0), removed mass max(p_w0y - p_W1, 0); the support rate of each is its
average truth-side boundary probability in the other donor. Both lists carry 25 slots, so

    cross-donor (W1 - w0y) = turnover / 25 * (support of added - support of removed),

which is checked to 1e-12.

    python3 stage3/cross_donor_b2.py [--out stage3/results]
"""
import json
import os
import sys
import time

import numpy as np

import a6_common as A
from a6_common import BETA, K, block_bootstrap, exp_scores, fmt, self_to_query, summ
import evaluate_stage2 as es  # noqa: E402
import evaluate_w1_e4 as ev  # noqa: E402
from pr10.scoring import _boundary  # noqa: E402

SCRIPTS = ["cross_donor_b2.py"]
KTOP = 25


def donor_d3():
    D = es.load("e4_tcell.npz")
    Z = np.load(os.path.join(A.HERE, "data", "e4_w1_stage3.npz"), allow_pickle=True)
    O = np.load(os.path.join(A.BUNDLE, "inputs", "own_gene_map.npz"), allow_pickle=True)
    if [str(x) for x in O["names"]] != [str(x) for x in D["names"]]:
        sys.exit("D3 own-gene map is for a different cohort")
    cell = next(c for c in es.sources()["E4"]["cells"] if (c["rule"], c["depth"]) == ("responsive", "20"))
    return {"label": "D3", "names": np.array([str(x) for x in D["names"]]), "ids": np.array([str(x) for x in O["ids"]]),
            "own_row": O["own_row"].astype(np.int64), "genes": np.array([str(g) for g in D["genes_keep"]]),
            "qu": D["qu"], "S0": D["S0_q"].astype(np.float64), "S4": D["S4_q"].astype(np.float64),
            "cm": D["cm_q"].astype(np.float64), "ce": D["ce_q"].astype(np.float64),
            "lists": {"responsive": (D["q_resp"], D["blocks_responsive"]), "terciles": (D["q_terc"], D["blocks_terciles"])},
            "w0y": cell["w0y"], "W1": ev.w1(Z, K, BETA)[0]}


def donor_d1():
    Z = A.d1_load()
    M = np.load(A.D1_MAP, allow_pickle=True)
    own = A.d1_own_row(Z)
    return {"label": "D1", "names": np.array([str(x) for x in Z["names"]]), "ids": np.array([str(x) for x in M["ids"]]),
            "own_row": own, "genes": np.array([str(g) for g in Z["genes_keep"]]), "qu": Z["qu"],
            "S0": Z["S0_q"], "S4": Z["S4_q"], "cm": Z["cm"][Z["qu"]], "ce": Z["ce"][Z["qu"]],
            "lists": {"responsive": (Z["q_resp"], Z["blocks_responsive"]), "terciles": (Z["q_terc"], Z["blocks_terciles"])},
            "w0y": A.d1_w0y(Z, 20), "W1": A.d1_w1(Z, 20)}


def id_exclusion(ids_targets, ids_queries):
    pos = {g: i for i, g in enumerate(ids_queries)}
    return np.array([pos.get(g, -1) for g in ids_targets], np.int64)


def turnover(Ea, Eb, T, sq):
    NQ = T.shape[0]
    add = np.zeros((NQ, 2)); rem = np.zeros((NQ, 2)); mass = np.zeros((NQ, 2)); nc = np.zeros((NQ, 2))
    for qi in range(NQ):
        keep = np.flatnonzero(sq != qi)
        for di, sgn in enumerate((1.0, -1.0)):
            pa = _boundary(Ea[qi, keep] * sgn, KTOP)[0]; pb = _boundary(Eb[qi, keep] * sgn, KTOP)[0]
            pt = _boundary(T[qi, keep] * sgn, KTOP)[0]
            up = np.clip(pa - pb, 0, None); dn = np.clip(pb - pa, 0, None)
            add[qi, di] = up @ pt; rem[qi, di] = dn @ pt; mass[qi, di] = up.sum(); nc[qi, di] = len(keep)
    return add, rem, mass, nc


def ratio_bootstrap(add, rem, mass, blocks, B=2000, seed=2026):
    """the same block draws as pr10.block_bootstrap, applied to the pooled support rates"""
    rng = np.random.default_rng(seed)
    bl = np.unique(blocks); mem = {b: np.flatnonzero(blocks == b) for b in bl}
    a = add.sum(1); r = rem.sum(1); m = mass.sum(1)
    out = []
    for _ in range(B):
        ix = np.concatenate([mem[b] for b in rng.choice(bl, len(bl), replace=True)])
        out.append((a[ix].sum() - r[ix].sum()) / max(m[ix].sum(), 1e-300))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    t0 = time.time()
    out_dir = A.arg("--out", os.path.join(A.HERE, "results"))
    integ = A.integrity(SCRIPTS)
    d3, d1 = donor_d3(), donor_d1()
    fails = []
    # fatal checks (A6 section 4 as amended by A6a section 3): the identifier-based exclusion equals the exclusion
    # each donor's evaluation uses, on that donor's own lists and all of its targets
    for d in (d3, d1):
        for rule, (q, _) in d["lists"].items():
            a = id_exclusion(d["ids"], d["genes"][q]); b = self_to_query(d["own_row"], q)
            if not np.array_equal(a, b):
                fails.append(f"{d['label']} {rule}: identifier exclusion differs in {(a != b).sum()} targets")
    common = sorted(set(d3["names"]) & set(d1["names"]))
    i3 = np.array([int(np.flatnonzero(d3["names"] == n)[0]) for n in common])
    i1 = np.array([int(np.flatnonzero(d1["names"] == n)[0]) for n in common])
    if not np.array_equal(d3["ids"][i3], d1["ids"][i1]):
        fails.append("shared targets carry different identifiers in the two maps")
    if fails:
        sys.exit(f"B2 checks failed; nothing written: {fails}")
    N = len(common)
    kp, kt = np.random.default_rng(101).random(N), np.random.default_rng(202).random(N)
    print(f"shared targets: {N}", flush=True)
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ,
           "shared_targets": N, "setting": {"k": K, "beta": BETA, "cells": 20, "K": KTOP}, "directions": {}}

    for src, tgt, ts, tt in ((d3, d1, i3, i1), (d1, d3, i1, i3)):
        name = f"{src['label']} -> {tgt['label']}"
        res["directions"][name] = {}
        tgt_row = {g: r for r, g in enumerate(tgt["genes"][tgt["qu"]])}
        for rule, (q, blocks) in src["lists"].items():
            qid = src["genes"][q]
            m = np.array([g in tgt_row for g in qid])
            qs, bl, ids = q[m], blocks[m], qid[m]
            rs = np.searchsorted(src["qu"], qs); rt = np.array([tgt_row[g] for g in ids])
            Y = src["S0"][rs][:, ts] / 20.0 - src["cm"][rs][:, None]
            T = {"cross": tgt["S4"][rt][:, tt] / 100.0 - tgt["ce"][rt][:, None],
                 "within": src["S4"][rs][:, ts] / 100.0 - src["ce"][rs][:, None]}
            sq = id_exclusion(src["ids"][ts], ids)
            E = {"raw": Y, "w0y": src["w0y"][ts][None, :] * Y, "W1": src["W1"][ts][None, :] * Y}
            cellres = {"queries": int(len(qs)), "queries_in_source_list": int(len(q)), "blocks": int(len(np.unique(bl))),
                       "own_entries_excluded": int((sq >= 0).sum())}
            for truth in ("cross", "within"):
                c = {"T": T[truth], "sq": sq, "KP": kp, "KT": kt}
                S = {a: exp_scores(E[a], c) for a in E}
                r = {"means": {a: float(S[a].mean()) for a in S},
                     "W1_vs_w0y": summ(block_bootstrap(S["W1"], S["w0y"], bl)),
                     "W1_vs_raw": summ(block_bootstrap(S["W1"], S["raw"], bl)),
                     "w0y_vs_raw": summ(block_bootstrap(S["w0y"], S["raw"], bl))}
                add, rem, mass, nc = turnover(E["W1"], E["w0y"], T[truth], sq)
                ident = float(abs(((add - rem) / KTOP).mean() - (S["W1"] - S["w0y"]).mean()))
                if ident > 1e-12:
                    sys.exit(f"{name} {rule} {truth}: the turnover identity fails by {ident:.2e}")
                M = mass.sum()
                r["turnover"] = {"slots_changed_per_list": float(mass.mean()),
                                 "support_added": float(add.sum() / M) if M > 0 else None,
                                 "support_removed": float(rem.sum() / M) if M > 0 else None,
                                 "support_difference": float((add.sum() - rem.sum()) / M) if M > 0 else None,
                                 "support_difference_ci95": ratio_bootstrap(add, rem, mass, bl) if M > 0 else None,
                                 "support_random_candidate": float((KTOP / nc).mean()),
                                 "identity_error": ident}
                cellres[truth] = r
                tv = r["turnover"]
                print(f"{name:9s} {rule:10s} {truth:6s} n={len(qs):3d}  raw {r['means']['raw']:.4f} w0y "
                      f"{r['means']['w0y']:.4f} W1 {r['means']['W1']:.4f}  W1-w0y {fmt(r['W1_vs_w0y'])}  "
                      f"W1-raw {fmt(r['W1_vs_raw'])}  added {tv['support_added']:.3f} removed "
                      f"{tv['support_removed']:.3f} (random {tv['support_random_candidate']:.3f}, "
                      f"{tv['slots_changed_per_list']:.2f} slots)", flush=True)
            res["directions"][name][f"{rule} 20"] = cellres
    reading = {}
    for name, d in res["directions"].items():
        b = d["responsive 20"]["cross"]["W1_vs_w0y"]
        reading[name] = ("W1's improvement is supported in the other donor" if b["ci95"][0] > 0 else
                         "no evidence of cross-donor support")
    res["reading"] = reading
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "b2_cross_donor.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(reading, indent=1))
    print(f"wrote {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
