"""Amendment A9: the target-level explanation table of W1's adjustments. A post hoc analysis of opened data (D3, D1,
D2); it reads only stored states and results and changes no earlier result.

For every target of each source donor's cohort it records the input side of W1's one weight change, and, in every
scoring pool of the original tests, the nominations that change brings in or takes out and how many of them the
pool's truth supports. The attribution is the tie-expected score itself: per list, E[P@25] = sum_t p_pred(t) p_T(t)
/ 25, so a target's net contribution sum_lists (p_W1 - p_w0y) p_T / (25 L) adds up over the cohort to the pool's
mean W1 - w0y exactly.

  input side    a = |y|^2 / G, nu = tr(V) / G, p_a = sum_{i<=5} (u_i'y)^2 / |y|^2,
                p_n = sum_{i<=5} [u_i' Sigma_t u_i / 20 + lambda_i / n_C] / tr(V), d = p_n - p_a;
                w0y = [1 - nu / a]_+, W1 = [1 - nu (1 - beta p_n) / (a (1 - beta p_a))]_+
  pools         own cohort and own deep truth (A4, A5 as corrected by A6a, A8 test 1); D2 truth (A8 test 2, sources
                D3 and D1); the other opened donor's deep truth on shared targets (B2 for D3 and D1, A8 section 7.1
                for D2)

    python3 stage3/explain_w1_targets.py [--out stage3/results]

Checks (A9 section 5) stop the run before anything is written.
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import rankdata, spearmanr

import a6_common as A
from a6_common import BETA, K
import cross_donor_b2 as cdb  # noqa: E402
import evaluate_d2_prospective as e2  # noqa: E402
import lock_b7_sources as L  # noqa: E402
from pr10.scoring import _boundary  # noqa: E402

KTOP = 25
CASES = ("USP22", "SUPT7L", "TADA2B", "RFX5", "TP53")
POOL_FIELDS = ("add", "add_sup", "rem", "rem_sup", "net")


def a9_integrity():
    out = {"amendment_A9_sha256": A.checked_sha(os.path.join(A.BUNDLE, "AMENDMENT_A9.md"),
                                                os.path.join(A.BUNDLE, "AMENDMENT_A9_SHA256.txt"), "amendment A9"),
           "script_sha256": A.sha256(os.path.abspath(__file__))}
    for rel in ("data/b7_lock_sources.npz", "data/d2_prospective_state.npz", "data/e4_w1_stage3.npz",
                "data/d1_confirm_state.npz", "data/d1_own_gene_map.npz", "results/b7_relation_table.json",
                "results/b7_d2_prospective.json", "results/b2_cross_donor.json", "results/stage3_w1_e4.json",
                "results/d1_self_gene_correction.json"):
        out[rel] = A.sha256(os.path.join(A.HERE, rel))
    return out


# ------------------------------------------------------------------ input side
def comps_e4(Z):
    """D3 (the E4 W1 state): the pieces of stage3/evaluate_w1_e4.w1 at k = 5"""
    B = 20.0; n_c = float(Z["n_control"]); G = float(Z["n_genes"])
    m2 = Z["m2_20"].astype(np.float64); vb = Z["varbar_20"].astype(np.float64)
    lam = Z["eigenvalues"][:K]; trC = float(Z["control_trace"])
    ps = Z["proj_sum"][:, :K]; pq = Z["proj_sumsq"][:, :K]; pc = Z["proj_control"][:K]
    return pieces(m2, vb, ps, pq, pc, lam, trC, G, n_c, B)


def comps_d(Z):
    """D1 and D2: the pieces of a6_common.d1_w1 at 20 cells, k = 5"""
    B = 20.0; n_c = float(Z["n_control"]); G = float(Z["n_weight_genes"])
    m2 = Z["m2_20"].astype(np.float64); vb = Z["vb_20"].astype(np.float64)
    lam = Z["eigenvalues"][:K]; trC = float(Z["control_trace"])
    ps = Z["proj_sum"][0][:, :K]; pq = Z["proj_sumsq"][0][:, :K]; pc = Z["proj_control"][:K]
    return pieces(m2, vb, ps, pq, pc, lam, trC, G, n_c, B)


def pieces(m2, vb, ps, pq, pc, lam, trC, G, n_c, B):
    pY = ps / B - pc[None, :]                                  # u_i'y
    sj = (pq - B * (ps / B) ** 2) / (B - 1.0)                  # u_i' Sigma_t u_i from the target's own cells
    e_dir = (pY ** 2).sum(1) / G
    nu = vb / B + trC / (G * n_c)
    n_dir = (sj / B + lam[None, :] / n_c).sum(1) / G
    w0y = np.clip(1.0 - nu / m2, 0.0, 1.0)
    w1 = np.clip(1.0 - (nu - BETA * n_dir) / np.maximum(m2 - BETA * e_dir, 1e-300), 0.0, 1.0)
    return {"a": m2, "nu": nu, "p_a": e_dir / m2, "p_n": n_dir / nu, "w0y_formula": w0y, "W1_formula": w1}


# ------------------------------------------------------------------ list side
def pool_moves(Y, T, sq, w_new, w_old):
    """expected additions and removals per candidate, weighted by the truth's inclusion probability"""
    NQ, N = T.shape
    out = {k: np.zeros(N) for k in ("add", "add_sup", "rem", "rem_sup")}
    En = w_new[None, :] * Y; Eo = w_old[None, :] * Y
    lists = 0; rnd = 0.0; diff = 0.0
    for qi in range(NQ):
        keep = np.flatnonzero(sq != qi)
        for sgn in (1.0, -1.0):
            pn = _boundary(En[qi, keep] * sgn, KTOP)[0]; po = _boundary(Eo[qi, keep] * sgn, KTOP)[0]
            pt = _boundary(T[qi, keep] * sgn, KTOP)[0]
            up = np.clip(pn - po, 0, None); dn = np.clip(po - pn, 0, None)
            out["add"][keep] += up; out["rem"][keep] += dn
            out["add_sup"][keep] += up * pt; out["rem_sup"][keep] += dn * pt
            diff += (pn @ pt - po @ pt) / KTOP; lists += 1; rnd += KTOP / len(keep)
    out["net"] = (out["add_sup"] - out["rem_sup"]) / (KTOP * lists)
    return out, lists, rnd / lists, diff / lists


def det_counts(Y, sq, w_new, w_old, key):
    """deterministic top 25 under the locked prediction-side key, as the A8 relation table"""
    NQ, N = Y.shape
    n_new = np.zeros(N, np.int64); n_old = np.zeros(N, np.int64)
    for qi in range(NQ):
        keep = np.flatnonzero(sq != qi)
        for sgn in (1.0, -1.0):
            a = set(keep[np.lexsort((key[keep], -(w_new[keep] * Y[qi, keep] * sgn)))[:KTOP]].tolist())
            b = set(keep[np.lexsort((key[keep], -(w_old[keep] * Y[qi, keep] * sgn)))[:KTOP]].tolist())
            for m in a - b:
                n_new[m] += 1
            for m in b - a:
                n_old[m] += 1
    return n_new, n_old


# ------------------------------------------------------------------ the pools, as the original tests built them
def lock_source(Lk, s):
    return {k[len(s) + 1:]: Lk[k] for k in Lk.files if k.startswith(s + "_")}


def pools_for(s, S, Other, oname, Z2, T2, s2_row, tg_row):
    names = [str(x) for x in S["names"]]; ids = np.array([str(x) for x in S["ids"]])
    grow = {str(g): i for i, g in enumerate(S["genes"])}
    resp = [str(g) for g in S["resp"]]
    out = {}
    # own cohort, own deep truth (A4 primary; A5 as corrected by A6a)
    rs = np.array([grow[g] for g in resp])
    out["own"] = {"idx": np.arange(len(names)), "Y": S["Y20"][rs], "T": S["T"][rs],
                  "sq": cdb.id_exclusion(ids, resp), "key": S["key_pred"]}
    # D2 truth (A8 test 2)
    C = np.array([j for j, n in enumerate(names) if n in s2_row], np.int64)
    c2 = np.array([s2_row[names[j]] for j in C], np.int64)
    q = [g for g in resp if g in tg_row]
    rq = np.array([grow[g] for g in q]); rt = np.array([tg_row[g] for g in q])
    out["D2"] = {"idx": C, "Y": S["Y20"][rq][:, C], "T": T2[c2][:, rt].T, "sq": cdb.id_exclusion(ids[C], q),
                 "key": S["key_pred"][C]}
    # the other opened donor (B2, rebuilt from the lock as lock_b7_sources.b2_from_lock)
    common = sorted(set(names) & set(str(x) for x in Other["names"]))
    onames = [str(x) for x in Other["names"]]
    ts = np.array([names.index(n) for n in common]); tt = np.array([onames.index(n) for n in common])
    orow = {str(g): i for i, g in enumerate(Other["genes"])}
    q = [g for g in resp if g in orow]
    rs2 = np.array([grow[g] for g in q]); rt2 = np.array([orow[g] for g in q])
    N = len(common)
    out[oname] = {"idx": ts, "Y": S["Y20"][rs2][:, ts], "T": Other["T"][rt2][:, tt],
                  "sq": cdb.id_exclusion(ids[ts], q), "key": np.random.default_rng(101).random(N)}
    return out


def pools_d2(Z2, Lk):
    names2 = [str(x) for x in Z2["names"]]; ids2 = np.array([str(x) for x in Z2["ids"]])
    cells = e2.d2_cells(Z2)[("responsive", "20")]
    out = {"own": {"idx": np.arange(len(names2)), "Y": cells["Y"], "T": cells["T"], "sq": cells["sq"],
                   "key": cells["KP"]}}
    genes_keep = np.array([str(g) for g in Z2["genes_keep"]])
    g2 = genes_keep[Z2["qu"]]; row2 = {g: i for i, g in enumerate(g2)}
    Y2 = Z2["S0_q"] / 20.0 - Z2["cm"][Z2["qu"]][:, None]
    for t in ("D3", "D1"):
        O = lock_source(Lk, t)
        on = [str(x) for x in O["names"]]; op = {n: i for i, n in enumerate(on)}
        common = sorted(set(names2) & set(on))
        i2 = np.array([names2.index(n) for n in common], np.int64); io = np.array([op[n] for n in common], np.int64)
        orow = {str(g): i for i, g in enumerate(O["genes"])}
        qg = [str(genes_keep[i]) for i in Z2["q_resp"]]
        q = [g for g in qg if g in orow]
        N = len(common)
        out[t] = {"idx": i2, "Y": Y2[[row2[g] for g in q]][:, i2], "T": O["T"][[orow[g] for g in q]][:, io],
                  "sq": cdb.id_exclusion(ids2[i2], q), "key": np.random.default_rng(101).random(N)}
    return out


# ------------------------------------------------------------------ summaries (A9 section 6)
def rate(num, den):
    return float(num / den) if den > 0 else None


def concentration(m):
    net = m["net"]
    out = {"net_positive": int((net > 1e-15).sum()), "net_negative": int((net < -1e-15).sum()),
           "net_zero": int((np.abs(net) <= 1e-15).sum())}
    for k in ("add_sup", "rem_sup"):
        tot = m[k].sum(); v = np.sort(m[k])[::-1]
        out[f"{k}_total"] = float(tot)
        out[f"{k}_share_top10"] = rate(v[:10].sum(), tot)
        out[f"{k}_share_top50"] = rate(v[:50].sum(), tot)
    return out


def bins(order_value, active, pools_full, rnd, n_bins=5, extra=None):
    """equal-count bins of the active targets by order_value (ascending); per bin and pool: moves and support"""
    act = np.flatnonzero(active)
    srt = act[np.argsort(order_value[act], kind="stable")]
    rows = []
    for b, members in enumerate(np.array_split(srt, n_bins), start=1):
        row = {"bin": b, "targets": int(len(members)),
               "value_min": float(order_value[members].min()), "value_max": float(order_value[members].max())}
        if extra:
            for k, f in extra.items():
                row[k] = f(members)
        row["pools"] = {}
        for pname, full in pools_full.items():
            inp = members[~np.isnan(full["add"][members])]
            A_ = float(np.nansum(full["add"][inp])); As = float(np.nansum(full["add_sup"][inp]))
            R_ = float(np.nansum(full["rem"][inp])); Rs = float(np.nansum(full["rem_sup"][inp]))
            row["pools"][pname] = {"targets_in_pool": int(len(inp)), "additions": A_, "addition_support_rate": rate(As, A_),
                                   "removals": R_, "removal_support_rate": rate(Rs, R_),
                                   "net": float(np.nansum(full["net"][inp])), "random_rate": rnd[pname]}
        rows.append(row)
    return rows


def top_vs_rest(rows, pname, which):
    """A9 section 7: the addition rate of the top bin against the rest, or the removal rate of the bottom bin"""
    if which == "add":
        sel, rest = rows[-1], rows[:-1]; num, den = "addition_support_rate", "additions"
    else:
        sel, rest = rows[0], rows[1:]; num, den = "removal_support_rate", "removals"
    p = sel["pools"][pname]
    if p[num] is None:
        return None
    rn = sum((r["pools"][pname][num] or 0.0) * r["pools"][pname][den] for r in rest)
    rd = sum(r["pools"][pname][den] for r in rest)
    if rd <= 0:
        return None
    return {"bin_rate": p[num], "rest_rate": rn / rd, "difference": p[num] - rn / rd}


def main():
    t0 = time.time()
    out_dir = A.arg("--out", os.path.join(A.HERE, "results"))
    integ = a9_integrity()
    Lk = np.load(L.LOCK_NPZ, allow_pickle=True)
    Z2 = np.load(os.path.join(A.HERE, "data", "d2_prospective_state.npz"), allow_pickle=True)
    Z3 = np.load(os.path.join(A.HERE, "data", "e4_w1_stage3.npz"), allow_pickle=True)
    Z1 = A.d1_load()
    rel = json.load(open(os.path.join(A.HERE, "results", "b7_relation_table.json"), encoding="utf-8"))["relations"]
    b7 = A.results("b7_d2_prospective.json"); b2 = A.results("b2_cross_donor.json")
    stored = {("D3", "own"): A.results("stage3_w1_e4.json")["cells"]["responsive 20"]["W1_vs_w0y"]["diff"],
              ("D1", "own"): A.results("d1_self_gene_correction.json")["cells"]["corrected"]["responsive 20"]["W1_vs_w0y"]["diff"],
              ("D2", "own"): b7["test1"]["cells"]["responsive 20"]["W1_vs_w0y"]["diff"],
              ("D3", "D2"): b7["test2"]["sources"]["D3"]["lists"]["responsive"]["W1_vs_w0y"]["diff"],
              ("D1", "D2"): b7["test2"]["sources"]["D1"]["lists"]["responsive"]["W1_vs_w0y"]["diff"],
              ("D3", "D1"): b2["directions"]["D3 -> D1"]["responsive 20"]["cross"]["W1_vs_w0y"]["diff"],
              ("D1", "D3"): b2["directions"]["D1 -> D3"]["responsive 20"]["cross"]["W1_vs_w0y"]["diff"],
              ("D2", "D3"): b7["secondary"]["D2_as_source"]["D3"]["responsive"]["W1_vs_w0y"]["diff"],
              ("D2", "D1"): b7["secondary"]["D2_as_source"]["D1"]["responsive"]["W1_vs_w0y"]["diff"]}
    T2, _ = e2.truth2(Z2)
    s2_row = {str(n): i for i, n in enumerate(Z2["s2_names"])}
    tg_row = {str(g): i for i, g in enumerate(Z2["truth_genes"])}
    S3, S1 = lock_source(Lk, "D3"), lock_source(Lk, "D1")
    src = {"D3": {"S": S3, "comps": comps_e4(Z3), "state_names": Z3["names"],
                  "pools": pools_for("D3", S3, S1, "D1", Z2, T2, s2_row, tg_row)},
           "D1": {"S": S1, "comps": comps_d(Z1), "state_names": Z1["names"],
                  "pools": pools_for("D1", S1, S3, "D3", Z2, T2, s2_row, tg_row)},
           "D2": {"S": {"names": Z2["names"], "ids": Z2["ids"], "w0y": e2.w0y(Z2, 20), "W1": e2.w1(Z2, 20)},
                  "comps": comps_d(Z2), "state_names": Z2["names"], "pools": pools_d2(Z2, Lk)}}
    checks = {"weights": {}, "identity_violations": {}, "pool_sums": {}, "relation_table": {}}
    fails = []
    for s, v in src.items():
        S, c = v["S"], v["comps"]
        if [str(x) for x in v["state_names"]] != [str(x) for x in S["names"]]:
            fails.append(f"{s}: state and lock cohorts differ")
        dW1 = float(np.max(np.abs(c["W1_formula"] - S["W1"])))
        dw0 = float(np.max(np.abs(c["w0y_formula"] - S["w0y"])))
        checks["weights"][s] = {"W1_max_abs_diff": dW1, "w0y_max_abs_diff": dw0}
        if dW1 > 1e-12 or dw0 > (2e-7 if s == "D3" else 1e-12):
            fails.append(f"{s}: weights not reproduced ({dW1:.1e}, {dw0:.1e})")
        ok = (c["w0y_formula"] > 0) & (c["W1_formula"] > 0)
        lhs = np.sign(c["W1_formula"] - c["w0y_formula"]); rhs = np.sign(c["p_n"] - c["p_a"])
        viol = int((ok & (lhs != rhs)).sum())
        checks["identity_violations"][s] = {"unclipped_targets": int(ok.sum()), "violations": viol}
        if viol:
            fails.append(f"{s}: identity violated in {viol} targets")
    # pools
    moves = {}
    for s, v in src.items():
        S = v["S"]; w0 = np.asarray(S["w0y"], np.float64); w1 = np.asarray(S["W1"], np.float64)
        for pname, P in v["pools"].items():
            m, lists, rnd, diff = pool_moves(P["Y"], P["T"], P["sq"], w1[P["idx"]], w0[P["idx"]])
            gap = abs(m["net"].sum() - stored[(s, pname)])
            checks["pool_sums"][f"{s} -> {pname}"] = {"stored": stored[(s, pname)], "sum_of_net": float(m["net"].sum()),
                                                      "abs_diff": float(gap), "lists": lists}
            if gap > 1e-12:
                fails.append(f"{s} -> {pname}: contributions sum to {m['net'].sum():.6f}, stored {stored[(s, pname)]:.6f}")
            moves[(s, pname)] = (m, lists, rnd)
            print(f"{s} -> {pname:3s}: {len(P['idx'])} candidates, {lists} lists, W1 - w0y {m['net'].sum():+.4f} "
                  f"(stored {stored[(s, pname)]:+.4f})", flush=True)
    # relation table, deterministic counts in the D2 pool
    agg = {}
    for r in rel:
        a = agg.setdefault((r["source"], r["target"]), {"W1-specific": 0, "w0y-specific": 0})
        a[r["kind"]] += 1
    for s in ("D3", "D1"):
        S = src[s]["S"]; P = src[s]["pools"]["D2"]; names = [str(x) for x in S["names"]]
        w0 = np.asarray(S["w0y"], np.float64); w1 = np.asarray(S["W1"], np.float64)
        n_new, n_old = det_counts(P["Y"], P["sq"], w1[P["idx"]], w0[P["idx"]], P["key"])
        bad = 0
        for k, j in enumerate(P["idx"]):
            want = agg.get((s, names[j]), {"W1-specific": 0, "w0y-specific": 0})
            if (n_new[k], n_old[k]) != (want["W1-specific"], want["w0y-specific"]):
                bad += 1
        checks["relation_table"][s] = {"targets_compared": int(len(P["idx"])), "mismatches": bad}
        if bad:
            fails.append(f"{s}: deterministic D2-pool counts differ from the relation table in {bad} targets")
    print(json.dumps(checks, indent=1), flush=True)
    if fails:
        sys.exit("A9 checks failed; nothing written: " + "; ".join(fails))

    # per-target table
    reltab = {}
    for r in rel:
        e = reltab.setdefault((r["source"], r["target"]), {f"{k}_{f}": 0 for k in ("W1spec", "w0yspec")
                                                          for f in ("n", "D2_rank", "other_measurable", "other_support", "both")})
        k = "W1spec" if r["kind"] == "W1-specific" else "w0yspec"
        e[f"{k}_n"] += 1; e[f"{k}_D2_rank"] += int(r["D2_rank_support"])
        e[f"{k}_other_measurable"] += int(r["other_donor_support"] is not None)
        e[f"{k}_other_support"] += int(bool(r["other_donor_support"]))
        e[f"{k}_both"] += int(r["measurable_donors"] == 2 and r["supporting_donors"] == 2)
    targets = []; summary = {}
    for s, v in src.items():
        S, c = v["S"], v["comps"]; names = [str(x) for x in S["names"]]; ids = [str(x) for x in S["ids"]]
        N = len(names)
        w0 = np.asarray(S["w0y"], np.float64); w1 = np.asarray(S["W1"], np.float64)
        r0 = rankdata(-w0, method="average"); r1 = rankdata(-w1, method="average")
        d = c["p_n"] - c["p_a"]
        full = {}
        for pname, P in v["pools"].items():
            m, lists, rnd = moves[(s, pname)]
            arr = {f: np.full(N, np.nan) for f in POOL_FIELDS}
            for f in POOL_FIELDS:
                arr[f][P["idx"]] = m[f]
            full[pname] = arr
        for j in range(N):
            row = {"source": s, "target": names[j], "target_id": ids[j],
                   "a": float(c["a"][j]), "nu": float(c["nu"][j]), "p_a": float(c["p_a"][j]), "p_n": float(c["p_n"][j]),
                   "d": float(d[j]), "w0y": float(w0[j]), "W1": float(w1[j]),
                   "ratio": (float(w1[j] / w0[j]) if w0[j] > 0 else None),
                   "rank_w0y": float(r0[j]), "rank_W1": float(r1[j]), "rank_change": float(r0[j] - r1[j])}
            for pname in ("own", "D2", "D3", "D1"):
                for f in POOL_FIELDS:
                    x = full[pname][f][j] if pname in full else np.nan
                    row[f"{pname}_{f}"] = None if np.isnan(x) else float(x)
            rt_ = reltab.get((s, names[j])) if s in ("D3", "D1") else None
            for k in ("W1spec", "w0yspec"):
                for f in ("n", "D2_rank", "other_measurable", "other_support", "both"):
                    row[f"rel_{k}_{f}"] = (rt_ or {}).get(f"{k}_{f}", 0) if s in ("D3", "D1") else None
            targets.append(row)
        active = (w0 > 0) | (w1 > 0)
        rnd = {pname: moves[(s, pname)][2] for pname in v["pools"]}
        ratio = np.where(w0 > 0, w1 / np.where(w0 > 0, w0, 1.0), np.nan)
        extra = {"d_median": lambda mem: float(np.median(d[mem])),
                 "ratio_median": lambda mem: float(np.nanmedian(ratio[mem])),
                 "rank_change_median": lambda mem: float(np.median((r0 - r1)[mem])),
                 "w0y_median": lambda mem: float(np.median(w0[mem]))}
        summary[s] = {"targets": N, "both_weights_zero": int((~active).sum()),
                      "pools": {pname: {"candidates": int(len(P["idx"])), "lists": moves[(s, pname)][1],
                                        "random_rate": moves[(s, pname)][2], "W1_minus_w0y": stored[(s, pname)],
                                        "concentration": concentration(moves[(s, pname)][0])}
                                for pname, P in v["pools"].items()},
                      "bins_by_d": bins(d, active, full, rnd, extra=extra),
                      "bins_by_w0y": bins(w0, active, full, rnd, extra=extra),
                      "consistency": {}}
        for pname in v["pools"]:
            if pname == "own":
                continue
            a_ = full["own"]["net"]; b_ = full[pname]["net"]
            both = ~np.isnan(b_)
            nz = both & (np.abs(a_) > 1e-15) & (np.abs(b_) > 1e-15)
            rho = spearmanr(a_[both], b_[both]).correlation if both.sum() > 2 else None
            summary[s]["consistency"][f"own vs {pname}"] = {
                "targets_in_both": int(both.sum()), "nonzero_in_both": int(nz.sum()),
                "same_sign": int((np.sign(a_[nz]) == np.sign(b_[nz])).sum()),
                "same_sign_fraction": rate((np.sign(a_[nz]) == np.sign(b_[nz])).sum(), nz.sum()),
                "spearman": (None if rho is None else float(rho))}
    # reading (A9 section 7): four cross-donor pools of the sources D3 and D1
    cross = [("D3", "D2"), ("D3", "D1"), ("D1", "D2"), ("D1", "D3")]
    reading = {}
    for which in ("add", "rem"):
        comp = {f"{s} -> {p}": top_vs_rest(summary[s]["bins_by_d"], p, which) for s, p in cross}
        signs = [None if v_ is None else np.sign(v_["difference"]) for v_ in comp.values()]
        agree = all(x is not None for x in signs) and len(set(signs)) == 1 and signs[0] != 0
        reading["top_bin_additions" if which == "add" else "bottom_bin_removals"] = {
            "pools": comp, "same_direction_in_all_four": bool(agree),
            "direction": (None if not agree else ("higher" if signs[0] > 0 else "lower"))}
    separable = any(v_["same_direction_in_all_four"] for v_ in reading.values())
    reading["reading"] = ("gains and losses are separable by the input-side structure" if separable else
                          "the input-side structure does not separate gains from losses")
    cases = [row for row in targets if row["target"] in CASES]
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ,
           "note": "amendment A9; post hoc explanation on opened data; responsive lists, 20 cells, k = 5, beta = 0.9",
           "checks": checks, "summary": summary, "reading": reading, "cases": cases, "targets": targets}
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, "w1_target_explanation.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    cp = os.path.join(out_dir, "w1_target_explanation_targets.csv")
    with open(cp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(targets[0].keys()))
        w.writeheader(); w.writerows(targets)
    print(f"reading: {reading['reading']}")
    for k in ("top_bin_additions", "bottom_bin_removals"):
        print(k, json.dumps(reading[k]))
    print(f"wrote {jp} and {cp} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
