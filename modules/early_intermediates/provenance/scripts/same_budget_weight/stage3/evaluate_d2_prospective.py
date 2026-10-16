"""Amendment A8 sections 4 to 7: the D2 prospective evaluation. Frozen before it ran (stage3/A8_FREEZE_SHA256.txt);
its code paths were run beforehand on the dry-run state built from the opened donor D1.

Test 1  D2 confirmation, as A5: W1 against w0y (k = 5, beta = 0.9) on D2's own cohort, own-gene exclusion through
        D2's perturbed_gene_id map; primary D2 responsive 20, gatekeepers D2 terciles 20 and 50.
Test 2  the locked 20-cell predictions of D3 and D1 scored against D2's cross-donor truth (first 100 cells of each
        target, >= 100 eligible cells), within C_s = source cohort that D2 measured; decidable when |C_s| >= 150;
        pooled over the two sources with equal weights, pooled interval = the average of the two intervals'
        endpoints (A8 section 5).
Table   W1-specific and w0y-specific relations of the source responsive lists, with their support in D2 and in the
        other opened donor (A8 section 6); JSON and CSV.
Second  D2 as a source against D3 and D1 deep truth; the external baselines and EB-entry on D2's block 0 (A8 section 7).

    python3 stage3/evaluate_d2_prospective.py [--out stage3/results]
    python3 stage3/evaluate_d2_prospective.py --dry-run --state <dry state> --b3 <dry S3> --out <tmp> [--dry-min 10]
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

import a6_common as A
from a6_common import BETA, K, block_bootstrap, exp_scores, fmt, self_to_query, summ
import cross_donor_b2 as cdb  # noqa: E402
import evaluate_b3_baselines as b3  # noqa: E402
import evaluate_b6_entry_shrinkage as b6  # noqa: E402
import evaluate_w1_e4 as ev  # noqa: E402
import lock_b7_sources as L  # noqa: E402
from pr10.scoring import _boundary  # noqa: E402

SCRIPTS = ["evaluate_d2_prospective.py", "read_d2_prospective.py", "lock_b7_sources.py"]
KTOP, MIN_C, Z_SUPPORT = 25, 150, 1.96
EXTERNAL = ("EB-entry", "scMAGeCK-LR", "scMAGeCK-RRA", "Welch t", "rank-sum")


def load(dry):
    sp = A.arg("--state", os.path.join(A.HERE, "data", "d2_prospective_state.npz"))
    p3 = A.arg("--b3", os.path.join(A.HERE, "data", "b3_D2_query_cells.npz"))
    Z = np.load(sp, allow_pickle=True); S3 = np.load(p3, allow_pickle=True)
    if bool(Z["dry"]) != dry or bool(S3["dry"]) != dry:
        sys.exit("a dry-run state can only be evaluated with --dry-run, and a real one only without it")
    return Z, S3, sp, p3


def w0y(Z, d):
    return A.d1_w0y(Z, d)


def w1(Z, d, k=K, beta=BETA, idx=None):
    if idx is None:
        return A.d1_w1(Z, d, k, beta)
    B = float(d); G = float(Z["n_weight_genes"]); n_c = float(Z["n_control"]); trC = float(Z["control_trace"])
    lam = Z["eigenvalues"][idx]; pc = Z["proj_control"][idx]
    ps = Z["proj_sum"][0][:, idx] if d == 20 else (Z["proj_sum"][0] + Z["proj_sum"][1])[:, idx]
    pq = Z["proj_sumsq"][0][:, idx] if d == 20 else (Z["proj_sumsq"][0] + Z["proj_sumsq"][1])[:, idx]
    pY = ps / B - pc[None, :]
    sj = (pq - B * (ps / B) ** 2) / (B - 1.0)
    a = Z[f"m2_{d}"] - beta * (pY ** 2).sum(1) / G
    nu = Z[f"vb_{d}"] / B + trC / (G * n_c) - beta * (sj / B + lam[None, :] / n_c).sum(1) / G
    return np.clip(1.0 - nu / np.maximum(a, 1e-300), 0.0, 1.0)


def d2_cells(Z):
    NT = len(Z["names"]); qu = Z["qu"]
    cm_q = Z["cm"][qu][:, None]; ce_q = Z["ce"][qu][:, None]
    Y = {"20": Z["S0_q"] / 20.0 - cm_q, "50": (Z["S0_q"] + Z["S1_q"]) / 50.0 - cm_q}
    T = Z["S4_q"] / 100.0 - ce_q
    kp, kt = np.random.default_rng(101).random(NT), np.random.default_rng(202).random(NT)
    out = {}
    for rule in ("responsive", "terciles"):
        q = Z["q_resp"] if rule == "responsive" else Z["q_terc"]
        pos = np.searchsorted(qu, q)
        for d in ("20", "50"):
            out[(rule, d)] = {"Y": Y[d][pos], "T": T[pos], "sq": self_to_query(Z["self_row"], qu[pos]), "q": q,
                              "blocks": Z["blocks_responsive"] if rule == "responsive" else Z["blocks_terciles"],
                              "KP": kp, "KT": kt, "w0y": w0y(Z, int(d)), "W1": w1(Z, int(d))}
    return out


def test1(Z, fails):
    chk = {f"beta_zero_reproduces_w0y_{d}_abs": float(np.max(np.abs(w1(Z, d, K, 0.0) - w0y(Z, d)))) for d in (20, 50)}
    if max(chk.values()) > 1e-6:
        fails.append(f"beta = 0 does not reproduce D2's w0y: {chk}")
    cells = d2_cells(Z); res = {"checks": chk, "cells": {}}
    for (rule, d), c in cells.items():
        raw = exp_scores(c["Y"], c); base = exp_scores(c["w0y"][None, :] * c["Y"], c)
        sc = exp_scores(c["W1"][None, :] * c["Y"], c)
        role = ("primary" if (rule, d) == ("responsive", "20") else
                "gatekeeper" if rule == "terciles" else "descriptive")
        res["cells"][f"{rule} {d}"] = {
            "role": role, "raw": float(raw.mean()), "w0y": float(base.mean()), "W1": float(sc.mean()),
            "W1_vs_w0y": summ(block_bootstrap(sc, base, c["blocks"])), "W1_vs_raw": summ(block_bootstrap(sc, raw, c["blocks"])),
            "w0y_vs_raw": summ(block_bootstrap(base, raw, c["blocks"])),
            "own_entries_excluded": int((c["sq"] >= 0).sum()),
            "weights": {"mean": float(c["W1"].mean()), "zero_fraction": float((c["W1"] == 0).mean()),
                        "spearman_with_w0y": float(spearmanr(c["W1"], c["w0y"]).correlation),
                        "degenerate": bool(np.all(c["W1"] == 0) or np.ptp(c["W1"]) == 0)}}
        r = res["cells"][f"{rule} {d}"]
        print(f"test 1  {rule:10s} {d}  raw {r['raw']:.4f} w0y {r['w0y']:.4f} W1 {r['W1']:.4f}  W1-w0y "
              f"{fmt(r['W1_vs_w0y'])}", flush=True)
    p = res["cells"]["responsive 20"]; dd = p["W1_vs_w0y"]
    if p["weights"]["degenerate"]:
        reading = "not confirmed (weights degenerate)"
    elif dd["diff"] >= 0.02 and dd["ci95"][0] > 0:
        reading = "confirmed, at the finalising level"
    elif dd["diff"] >= 0.01 and dd["ci95"][0] > 0:
        reading = "confirmed"
    else:
        reading = "not confirmed"
    gates = {k: bool(res["cells"][k]["W1_vs_w0y"]["ci95"][1] >= 0) for k in ("terciles 20", "terciles 50")}
    res["reading"] = {"A5_section_5": reading, "gatekeepers_pass": gates}
    c = cells[("responsive", "20")]; base = exp_scores(c["w0y"][None, :] * c["Y"], c); grid = {}
    for k in ev.K_GRID:
        for beta in ev.BETA_GRID:
            b = block_bootstrap(exp_scores(w1(Z, 20, k, beta)[None, :] * c["Y"], c), base, c["blocks"])
            grid[f"k{k}_beta{beta}"] = {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]]}
    res["primary_grid"] = grid
    lam = Z["eigenvalues"]; trC = float(Z["control_trace"]); ctrl = {}
    for label, idx in (("1..5", np.arange(5)), ("6..20", np.arange(5, 20)), ("1 only", np.arange(1))):
        b = block_bootstrap(exp_scores(w1(Z, 20, idx=idx)[None, :] * c["Y"], c), base, c["blocks"])
        ctrl[label] = {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]],
                       "control_trace_share": round(float(lam[idx].sum() / trC), 4)}
    res["direction_set_control"] = ctrl
    return res, cells


def truth2(Z):
    n = float(Z["s2_cells_per_target"]); nev = float(Z["n_control"])
    mu = Z["T2_sum"] / n
    T = mu - Z["ev_truth_mean"][None, :]
    v = (Z["T2_sumsq"] - n * mu ** 2) / (n - 1.0)
    se = np.sqrt(np.maximum(v, 0.0) / n + Z["ev_truth_var"][None, :] / nev)
    z = np.where(se > 0, T / np.where(se > 0, se, 1.0), 0.0)
    return T, z                                   # (S2 targets, truth genes)


def top_sets(v, keep, key):
    """deterministic top 25 among `keep` (value descending, key ascending), and the boundary probabilities"""
    order = np.lexsort((key[keep], -v[keep]))[:KTOP]
    p, ntie, r = _boundary(v[keep], KTOP)
    return set(keep[order].tolist()), dict(zip(keep.tolist(), p.tolist())), bool(ntie > r)


def test2_and_table(Z, Lk, min_c):
    T2, Zs = truth2(Z)
    tg = [str(g) for g in Z["truth_genes"]]; tg_row = {g: i for i, g in enumerate(tg)}
    s2_row = {str(n): i for i, n in enumerate(Z["s2_names"])}
    out = {"sources": {}, "pooled": {}}; relations = []
    src = {d: {k[len(d) + 1:]: Lk[k] for k in Lk.files if k.startswith(d + "_")} for d in ("D3", "D1")}
    for s, other in (("D3", "D1"), ("D1", "D3")):
        S = src[s]; O = src[other]
        names = [str(x) for x in S["names"]]; ids = np.array([str(x) for x in S["ids"]])
        C = np.array([j for j, n in enumerate(names) if n in s2_row], np.int64)
        c2 = np.array([s2_row[names[j]] for j in C], np.int64)
        info = {"candidates": int(len(C)), "random_floor": float(KTOP / max(len(C) - 1, 1)),
                "decidable": bool(len(C) >= max(min_c, 2 * KTOP)), "lists": {}}
        out["sources"][s] = info
        if len(C) < 2 * KTOP:
            info["note"] = "too few candidates to score"
            continue
        genes_s = [str(g) for g in S["genes"]]; grow = {g: i for i, g in enumerate(genes_s)}
        for rule, lst, blk in (("responsive", S["resp"], S["resp_blocks"]), ("terciles", S["terc"], S["terc_blocks"])):
            keep_q = [i for i, g in enumerate(lst) if str(g) in tg_row]
            q = [str(lst[i]) for i in keep_q]; bl = np.asarray(blk)[keep_q]
            missing = [str(g) for g in lst if str(g) not in tg_row]
            rs = np.array([grow[g] for g in q]); rt = np.array([tg_row[g] for g in q])
            Y = S["Y20"][rs][:, C]; T = T2[c2][:, rt].T
            sq = cdb.id_exclusion(ids[C], q)
            E = {"raw": Y, "w0y": S["w0y"][C][None, :] * Y, "W1": S["W1"][C][None, :] * Y}
            c = {"T": T, "sq": sq, "KP": S["key_pred"][C], "KT": S["key_truth"][C]}
            Sc = {a: exp_scores(E[a], c) for a in E}
            add, rem, mass, nc = cdb.turnover(E["W1"], E["w0y"], T, sq)
            ident = float(abs(((add - rem) / KTOP).mean() - (Sc["W1"] - Sc["w0y"]).mean()))
            if ident > 1e-12:
                sys.exit(f"{s} {rule}: turnover identity fails by {ident:.2e}")
            Mm = mass.sum()
            info["lists"][rule] = {
                "queries": len(q), "missing_genes": missing, "blocks": int(len(np.unique(bl))),
                "means": {a: float(v.mean()) for a, v in Sc.items()},
                "W1_vs_w0y": summ(block_bootstrap(Sc["W1"], Sc["w0y"], bl)),
                "W1_vs_raw": summ(block_bootstrap(Sc["W1"], Sc["raw"], bl)),
                "w0y_vs_raw": summ(block_bootstrap(Sc["w0y"], Sc["raw"], bl)),
                "turnover": {"slots_changed_per_list": float(mass.mean()),
                             "support_added": float(add.sum() / Mm) if Mm > 0 else None,
                             "support_removed": float(rem.sum() / Mm) if Mm > 0 else None,
                             "support_difference": float((add.sum() - rem.sum()) / Mm) if Mm > 0 else None,
                             "support_difference_ci95": cdb.ratio_bootstrap(add, rem, mass, bl) if Mm > 0 else None,
                             "random_candidate_rate": float((KTOP / nc).mean()), "identity_error": ident}}
            r = info["lists"][rule]
            print(f"test 2  {s} -> D2 {rule:10s} |C| {len(C)}  raw {r['means']['raw']:.4f} w0y {r['means']['w0y']:.4f} "
                  f"W1 {r['means']['W1']:.4f}  W1-w0y {fmt(r['W1_vs_w0y'])}", flush=True)
            if rule != "responsive":
                continue
            # ---- the relation table (A8 section 6), responsive lists only
            onames = [str(x) for x in O["names"]]; opos = {n: i for i, n in enumerate(onames)}
            shared = [j for j, n in enumerate(names) if n in opos]            # s cohort order
            sh_o = np.array([opos[names[j]] for j in shared], np.int64)
            o_grow = {str(g): i for i, g in enumerate(O["genes"])}
            for qi, g in enumerate(q):
                keep = np.flatnonzero(sq != qi)
                for di, sgn in enumerate((1.0, -1.0)):
                    top_w1, p_w1, tie_w1 = top_sets(E["W1"][qi] * sgn, keep, c["KP"])
                    top_w0, p_w0, tie_w0 = top_sets(E["w0y"][qi] * sgn, keep, c["KP"])
                    top_t, _, _ = top_sets(T[qi] * sgn, keep, c["KT"])
                    o_top = None
                    if g in o_grow:
                        vo = O["T"][o_grow[g]][sh_o] * sgn
                        so = cdb.id_exclusion(O["ids"][sh_o].astype(str), [g])
                        ko = np.flatnonzero(so != 0)
                        o_top = {shared[i] for i in top_sets(vo, ko, O["key_truth"][sh_o])[0]}
                    for kind, members in (("W1-specific", top_w1 - top_w0), ("w0y-specific", top_w0 - top_w1)):
                        for m in sorted(members):
                            j = int(C[m]); t = names[j]
                            meas_o = o_top is not None and t in opos
                            row = {"source": s, "target": t, "target_id": str(ids[j]), "gene": g,
                                   "direction": "up" if di == 0 else "down", "kind": kind,
                                   "p_W1": p_w1.get(m, 0.0), "p_w0y": p_w0.get(m, 0.0),
                                   "boundary_tie": bool(tie_w1 or tie_w0), "effect_20": float(Y[qi, m]),
                                   "D2_rank_support": bool(m in top_t),
                                   "D2_z": float(Zs[c2[m], rt[qi]]),
                                   "D2_effect_support": bool(Zs[c2[m], rt[qi]] * sgn >= Z_SUPPORT),
                                   "other_donor": other,
                                   "other_donor_support": (bool(j in o_top) if meas_o else None)}
                            row["measurable_donors"] = 1 + int(meas_o)
                            row["supporting_donors"] = int(row["D2_rank_support"]) + int(bool(row["other_donor_support"]))
                            relations.append(row)
    dec = [s for s in ("D3", "D1") if out["sources"][s]["decidable"]]
    for rule in ("responsive", "terciles"):
        if not dec:
            out["pooled"][rule] = {"decidable": False}
            continue
        bs = [out["sources"][s]["lists"][rule]["W1_vs_w0y"] for s in dec]
        tv = [out["sources"][s]["lists"][rule]["turnover"] for s in dec]
        pooled = {"sources": dec, "diff": float(np.mean([b["diff"] for b in bs])),
                  "ci95": [float(np.mean([b["ci95"][0] for b in bs])), float(np.mean([b["ci95"][1] for b in bs]))]}
        if all(t["support_difference"] is not None for t in tv):
            pooled["support_difference"] = float(np.mean([t["support_difference"] for t in tv]))
            pooled["support_difference_ci95"] = [float(np.mean([t["support_difference_ci95"][0] for t in tv])),
                                                 float(np.mean([t["support_difference_ci95"][1] for t in tv]))]
        out["pooled"][rule] = pooled
    return out, relations


def summarise(relations):
    out = {}
    for kind in ("W1-specific", "w0y-specific"):
        R = [r for r in relations if r["kind"] == kind]
        n = len(R)
        mo = [r for r in R if r["other_donor_support"] is not None]
        strata = {}
        for m in (1, 2):
            Rm = [r for r in R if r["measurable_donors"] == m]
            strata[f"measurable {m}"] = {f"supported by {k}": sum(r["supporting_donors"] == k for r in Rm)
                                         for k in range(m + 1)}
        out[kind] = {"relations": n,
                     "D2_rank_support_rate": (sum(r["D2_rank_support"] for r in R) / n) if n else None,
                     "D2_effect_support_rate": (sum(r["D2_effect_support"] for r in R) / n) if n else None,
                     "other_donor_measurable": len(mo),
                     "other_donor_support_rate": (sum(bool(r["other_donor_support"]) for r in mo) / len(mo)) if mo else None,
                     "by_measurable_donors": strata}
    return out


def secondary_sources(Z, Lk):
    """D2's own 20-cell lists against D3 and D1 deep truth (A8 section 7.1), as B2"""
    names2 = [str(x) for x in Z["names"]]; ids2 = np.array([str(x) for x in Z["ids"]])
    g2 = np.array([str(g) for g in Z["genes_keep"]])[Z["qu"]]; row2 = {g: i for i, g in enumerate(g2)}
    Y2 = Z["S0_q"] / 20.0 - Z["cm"][Z["qu"]][:, None]
    wts = {"raw": np.ones(len(names2)), "w0y": w0y(Z, 20), "W1": w1(Z, 20)}
    genes_keep = np.array([str(g) for g in Z["genes_keep"]])
    out = {}
    for t in ("D3", "D1"):
        O = {k[3:]: Lk[k] for k in Lk.files if k.startswith(t + "_")}
        on = [str(x) for x in O["names"]]; op = {n: i for i, n in enumerate(on)}
        common = sorted(set(names2) & set(on))
        i2 = np.array([names2.index(n) for n in common], np.int64); io = np.array([op[n] for n in common], np.int64)
        orow = {str(g): i for i, g in enumerate(O["genes"])}
        N = len(common); kp, kt = np.random.default_rng(101).random(N), np.random.default_rng(202).random(N)
        out[t] = {"shared_targets": N}
        for rule, qidx, blk in (("responsive", Z["q_resp"], Z["blocks_responsive"]),
                                ("terciles", Z["q_terc"], Z["blocks_terciles"])):
            qg = [str(genes_keep[i]) for i in qidx]
            m = [k for k, g in enumerate(qg) if g in orow]
            if N < 2 * KTOP or not m:
                out[t][rule] = {"queries": len(m), "note": "too few shared targets or queries"}
                continue
            q = [qg[k] for k in m]; bl = np.asarray(blk)[m]
            Y = Y2[[row2[g] for g in q]][:, i2]; T = O["T"][[orow[g] for g in q]][:, io]
            c = {"T": T, "sq": cdb.id_exclusion(ids2[i2], q), "KP": kp, "KT": kt}
            Sc = {a: exp_scores(w[i2][None, :] * Y, c) for a, w in wts.items()}
            out[t][rule] = {"queries": len(q), "means": {a: float(v.mean()) for a, v in Sc.items()},
                            "W1_vs_w0y": summ(block_bootstrap(Sc["W1"], Sc["w0y"], bl)),
                            "W1_vs_raw": summ(block_bootstrap(Sc["W1"], Sc["raw"], bl))}
            print(f"second  D2 -> {t} {rule:10s} shared {N} queries {len(q)}  W1-w0y "
                  f"{fmt(out[t][rule]['W1_vs_w0y'])}", flush=True)
    return out


def secondary_external(Z, S3, cells):
    if [str(x) for x in S3["names"]] != [str(x) for x in Z["names"]] or not np.array_equal(
            S3["query_genes"], Z["keep"][Z["qu"]]):
        sys.exit("the S3 state does not belong to this D2 state")
    C = S3["counts"]
    X = np.log1p(C.astype(np.float64) * (float(S3["scale"]) / np.maximum(S3["total_counts"], 1.0))[:, None])
    nc = int(S3["n_controls"]); NT = len(S3["names"])
    nzf = (C > 0).mean(0)
    Xw = np.minimum(X, np.percentile(X, b3.Q95, axis=0)[None, :])
    lr = b3.lr_closed(Xw, nc, NT); del Xw
    lr[:, nzf < b3.MIN_FRAC] = 0.0
    rup, rdn = b3.rra(X, nc, NT)
    stat = {"EB-entry": b6.eb_entry(X, nc, NT)[0].T, "scMAGeCK-LR": lr.T, "Welch t": b3.welch_t(X, nc, NT).T,
            "rank-sum": b3.rank_sum(X, nc, NT).T, "rra_up": -np.log(np.maximum(rup, 1e-300)).T,
            "rra_dn": np.log(np.maximum(rdn, 1e-300)).T}
    out = {}
    for rule in ("responsive", "terciles"):
        c = cells[(rule, "20")]; rows = np.searchsorted(Z["qu"], c["q"])
        Sc = {"raw": exp_scores(c["Y"], c), "W1": exp_scores(c["W1"][None, :] * c["Y"], c)}
        for a in ("EB-entry", "scMAGeCK-LR", "Welch t", "rank-sum"):
            Sc[a] = exp_scores(stat[a][rows], c)
        Sc["scMAGeCK-RRA"] = np.stack([exp_scores(stat["rra_up"][rows], c)[:, 0],
                                       exp_scores(stat["rra_dn"][rows], c)[:, 1]], 1)
        row = {"means": {a: float(v.mean()) for a, v in Sc.items()}, "comparisons": {}}
        for a in EXTERNAL:
            row["comparisons"][f"W1 - {a}"] = summ(block_bootstrap(Sc["W1"], Sc[a], c["blocks"]))
            row["comparisons"][f"{a} - raw"] = summ(block_bootstrap(Sc[a], Sc["raw"], c["blocks"]))
        out[f"{rule} 20"] = row
        print(f"second  D2 external {rule:10s} " + "  ".join(f"W1-{a} {fmt(row['comparisons'][f'W1 - {a}'])}"
                                                           for a in EXTERNAL), flush=True)
    return out


def main():
    t0 = time.time()
    dry = "--dry-run" in sys.argv
    out_dir = A.arg("--out", os.path.join(A.HERE, "results"))
    min_c = int(A.arg("--dry-min", 10)) if dry else MIN_C
    if dry:
        integ = {"DRY_RUN": "code-path test; every number in these files is meaningless"}
    else:
        integ = L.a8_integrity(SCRIPTS)
    lock = A.record(L.A8_LOCK)
    for rel, sha in lock.items():
        if A.sha256(os.path.join(A.HERE, rel)) != sha:
            sys.exit(f"{rel} differs from the A8 lock record")
    Z, S3, sp, p3 = load(dry)
    if not dry:
        if str(Z["amendment_A8_sha256"]) != integ["amendment_A8_sha256"]:
            sys.exit("the D2 read was done under a different A8")
        if str(Z["script_sha256"]) != integ["freeze_A8"]["read_d2_prospective.py"]:
            sys.exit("the D2 state was not written by the frozen read_d2_prospective.py")
    integ["state_sha256"] = A.sha256(sp); integ["s3_sha256"] = A.sha256(p3); integ["lock"] = lock
    Lk = np.load(L.LOCK_NPZ, allow_pickle=True)
    fails = []
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ,
           "donor": str(Z["donor"]), "read_checks": json.loads(str(Z["checks"])),
           "cohort": {"S1_targets": int(len(Z["names"])), "eligible_300": int(Z["n_eligible"]),
                      "S2_targets": int(len(Z["s2_names"])), "S2_in_S1": int(Z["s2_in_s1"].sum()),
                      "weight_genes": int(Z["n_weight_genes"]),
                      "own_gene_on_weight_axis": int((Z["self_row"] >= 0).sum())},
           "setting": {"k": K, "beta": BETA, "K": KTOP, "min_candidates": min_c}}
    t1, cells = test1(Z, fails)
    if fails:
        sys.exit(f"test 1 checks failed; nothing written: {fails}")
    res["test1"] = t1
    t2, relations = test2_and_table(Z, Lk, min_c)
    res["test2"] = t2
    pr = t2["pooled"]["responsive"]
    if not pr.get("decidable", True) and "diff" not in pr:
        reading2 = "undecidable: no source has enough candidates"
    else:
        note = "" if len(pr["sources"]) == 2 else f" (only {pr['sources'][0]} decidable)"
        reading2 = ("the pooled list changes of the two sources gain more support in D2" if pr["ci95"][0] > 0
                    else "no support seen in D2") + note
    res["test2"]["reading"] = reading2
    res["relation_table_summary"] = summarise(relations)
    res["secondary"] = {"D2_as_source": secondary_sources(Z, Lk), "external_baselines": secondary_external(Z, S3, cells)}
    os.makedirs(out_dir, exist_ok=True)
    pre = "dry_" if dry else ""
    path = os.path.join(out_dir, f"{pre}b7_d2_prospective.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    rp = os.path.join(out_dir, f"{pre}b7_relation_table.json")
    json.dump({"note": "A8 section 6; descriptive; z >= 1.96 is uncorrected", "relations": relations},
              open(rp, "w", encoding="utf-8"), indent=1)
    if relations:
        with open(os.path.join(out_dir, f"{pre}b7_relation_table.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(relations[0].keys())); w.writeheader(); w.writerows(relations)
    print(f"test 1: {t1['reading']['A5_section_5']}, gatekeepers {t1['reading']['gatekeepers_pass']}")
    if "diff" in pr:
        print(f"test 2 pooled responsive: {pr['diff']:+.4f} [{pr['ci95'][0]:+.4f}, {pr['ci95'][1]:+.4f}] -> {reading2}")
    else:
        print(f"test 2: {reading2}")
    print(f"relations: {len(relations)}; wrote {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
