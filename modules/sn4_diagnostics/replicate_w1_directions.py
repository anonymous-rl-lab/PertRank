"""Amendment A10: does the change along the control directions that W1 presses down reappear in an independent block
of the same target's cells? A post hoc analysis of opened data (D3, D1, D2); stored states only.

For target t, direction u_i (i <= 5, the directions W1 uses) and block b (block 0: cells 0..19, the cells W1 uses;
block 1: cells 20..49, disjoint), both centred on the same input-control mean:

  ybar_b,i = proj_sum_b,i / B_b - (input-control mean projection on u_i)
  n_b,i    = s_b,i / B_b + lambda_i / n_C          s_b,i: the target's own variance along u_i in block b

  E0 = sum_i ybar_0,i^2,  N0 = sum_i n_0,i,  A0 = E0 - N0       the apparent signal along the directions (block 0)
  C  = sum_i (ybar_0,i ybar_1,i - lambda_i / n_C)               the part of block 0's change that block 1 repeats
  A1 = sum_i (ybar_1,i^2 - n_1,i)                               block 1's own apparent signal, sign-free

all divided by G (per-gene units, as amendment A9). The shared control mean adds lambda_i / n_C to the expectation of
the product; it is removed exactly. Groups are A9's five bins of d = p_n - p_a; the primary quantity is
r = sum C / sum A0 in each bin, with a 95% percentile interval from 2,000 resamples of targets (seed 2026).

    python3 stage3/replicate_w1_directions.py [--out stage3/results]

The checks of A10 section 5 stop the run before anything is written.
"""
import csv
import json
import os
import sys
import time

import numpy as np

import a6_common as A
from a6_common import K

B0, B1 = 20.0, 30.0
CASES = ("TP53", "RFX5", "USP22", "SUPT7L", "TADA2B")
NBOOT, SEED = 2000, 2026


def a10_integrity():
    out = {"amendment_A10_sha256": A.checked_sha(os.path.join(A.BUNDLE, "AMENDMENT_A10.md"),
                                                 os.path.join(A.BUNDLE, "AMENDMENT_A10_SHA256.txt"), "amendment A10"),
           "script_sha256": A.sha256(os.path.abspath(__file__))}
    for rel in ("data/e4_w1_stage3.npz", "data/e4_w1_stage3_block1.npz", "data/d1_confirm_state.npz",
                "data/d2_prospective_state.npz", "results/w1_target_explanation.json"):
        out[rel] = A.sha256(os.path.join(A.HERE, rel))
    return out


def block(ps, pq, B, pc):
    ybar = ps / B - pc[None, :]
    s = (pq - B * (ps / B) ** 2) / (B - 1.0)
    return ybar, s


def donor_arrays(label):
    """per-target, per-direction block-0 and block-1 quantities, from the stored states"""
    if label == "D3":
        Z = np.load(os.path.join(A.HERE, "data", "e4_w1_stage3.npz"), allow_pickle=True)
        Zb = np.load(os.path.join(A.HERE, "data", "e4_w1_stage3_block1.npz"), allow_pickle=True)
        if [str(x) for x in Z["names"]] != [str(x) for x in Zb["names"]]:
            sys.exit("D3: the two block states list different targets")
        G = float(Z["n_genes"]); p0 = (Z["proj_sum"][:, :K], Z["proj_sumsq"][:, :K])
        p1 = (Zb["proj_sum"][:, :K], Zb["proj_sumsq"][:, :K])
        extra = {"m2_20": Z["m2_20"].astype(np.float64), "varbar_20": Z["varbar_20"].astype(np.float64),
                 "m2_30": Zb["m2_30"].astype(np.float64), "varbar_30": Zb["varbar_30"].astype(np.float64),
                 "trC": float(Z["control_trace"])}
        sizes_ok = True
    else:
        Z = A.d1_load() if label == "D1" else np.load(os.path.join(A.HERE, "data", "d2_prospective_state.npz"),
                                                        allow_pickle=True)
        G = float(Z["n_weight_genes"])
        p0 = (Z["proj_sum"][0][:, :K], Z["proj_sumsq"][0][:, :K]); p1 = (Z["proj_sum"][1][:, :K], Z["proj_sumsq"][1][:, :K])
        extra = None
        sizes_ok = bool(json.loads(str(Z["checks"])).get("block_sizes_ok", False))
    n_c = float(Z["n_control"]); lam = Z["eigenvalues"][:K].astype(np.float64); pc = Z["proj_control"][:K]
    y0, s0 = block(p0[0], p0[1], B0, pc); y1, s1 = block(p1[0], p1[1], B1, pc)
    n0 = s0 / B0 + lam[None, :] / n_c; n1 = s1 / B1 + lam[None, :] / n_c
    shared = lam / n_c
    per_dir = {"A0": (y0 ** 2 - n0) / G, "C": (y0 * y1 - shared[None, :]) / G, "A1": (y1 ** 2 - n1) / G}
    out = {"names": [str(x) for x in Z["names"]], "G": G, "n_c": n_c, "lam": lam, "y0": y0, "y1": y1,
           "sd0": np.sqrt(n0), "sd1": np.sqrt(n1), "per_dir": per_dir,
           "E0": (y0 ** 2).sum(1) / G, "N0": n0.sum(1) / G, "E1": (y1 ** 2).sum(1) / G, "N1": n1.sum(1) / G,
           "shared_share": float(shared.sum() * len(y0) / n0.sum()), "sizes_ok": sizes_ok, "extra": extra}
    out["A0"] = out["E0"] - out["N0"]; out["A1"] = out["E1"] - out["N1"]; out["C"] = per_dir["C"].sum(1)
    return out


def a9_rows(a9, label):
    return [r for r in a9["targets"] if r["source"] == label]


def a9_bins(rows):
    d = np.array([r["d"] for r in rows]); w0 = np.array([r["w0y"] for r in rows]); w1 = np.array([r["W1"] for r in rows])
    act = np.flatnonzero((w0 > 0) | (w1 > 0))
    srt = act[np.argsort(d[act], kind="stable")]
    return np.array_split(srt, 5), d


def ratio_ci(num, den, idx, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(idx); out = np.empty(NBOOT)
    for b in range(NBOOT):
        s = idx[rng.integers(0, n, n)]
        dn = den[s].sum()
        out[b] = num[s].sum() / dn if dn != 0 else np.nan
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return [float(lo), float(hi)]


def group_summary(D, idx):
    sA0 = float(D["A0"][idx].sum()); sC = float(D["C"][idx].sum()); sA1 = float(D["A1"][idx].sum())
    sE0 = float(D["E0"][idx].sum()); sN0 = float(D["N0"][idx].sum())
    g = {"targets": int(len(idx)), "sum_E0": sE0, "sum_N0": sN0, "sum_A0": sA0, "sum_C": sC, "sum_A1": sA1,
         "r_C_over_A0": (sC / sA0 if sA0 > 0 else None), "r_C_over_A0_ci95": (ratio_ci(D["C"], D["A0"], idx) if sA0 > 0 else None),
         "A1_over_A0": (sA1 / sA0 if sA0 > 0 else None), "A1_over_A0_ci95": (ratio_ci(D["A1"], D["A0"], idx) if sA0 > 0 else None),
         "C_over_E0": sC / sE0, "N0_over_E0": sN0 / sE0,
         "per_direction": [{"direction": i + 1, "sum_A0": float(D["per_dir"]["A0"][idx, i].sum()),
                            "sum_C": float(D["per_dir"]["C"][idx, i].sum()),
                            "sum_A1": float(D["per_dir"]["A1"][idx, i].sum())} for i in range(K)]}
    return g


def removal_rates(rows, members, pools):
    out = {}
    for p in pools:
        rem = sum(rows[j][f"{p}_rem"] or 0.0 for j in members if rows[j][f"{p}_rem"] is not None)
        sup = sum(rows[j][f"{p}_rem_sup"] or 0.0 for j in members if rows[j][f"{p}_rem"] is not None)
        out[p] = {"removals": rem, "supported": sup, "rate": (sup / rem if rem > 0 else None)}
    return out


def main():
    t0 = time.time()
    out_dir = A.arg("--out", os.path.join(A.HERE, "results"))
    integ = a10_integrity()
    a9 = json.load(open(os.path.join(A.HERE, "results", "w1_target_explanation.json"), encoding="utf-8"))
    pools_of = {"D3": ("own", "D2", "D1"), "D1": ("own", "D2", "D3"), "D2": ("own", "D3", "D1")}
    checks, fails, res, targets = {}, [], {}, []
    for label in ("D3", "D1", "D2"):
        D = donor_arrays(label); rows = a9_rows(a9, label)
        # check 1: E0, N0 against A9
        if [r["target"] for r in rows] != D["names"]:
            fails.append(f"{label}: target order differs from A9")
        e_a9 = np.array([r["p_a"] * r["a"] for r in rows]); n_a9 = np.array([r["p_n"] * r["nu"] for r in rows])
        relE = float(np.max(np.abs(D["E0"] - e_a9) / np.maximum(np.abs(e_a9), 1e-300)))
        relN = float(np.max(np.abs(D["N0"] - n_a9) / np.maximum(np.abs(n_a9), 1e-300)))
        # check 2: bins against A9
        bins, d = a9_bins(rows)
        want = a9["summary"][label]["bins_by_d"]
        bins_ok = all(len(b) == w["targets"] and float(d[b].min()) == w["value_min"] and float(d[b].max()) == w["value_max"]
                      for b, w in zip(bins, want))
        checks[label] = {"E0_max_rel_diff_vs_A9": relE, "N0_max_rel_diff_vs_A9": relN, "bins_match_A9": bool(bins_ok),
                         "block_sizes_ok": D["sizes_ok"], "shared_control_term_share_of_N0": D["shared_share"]}
        if relE > 1e-12 or relN > 1e-12:
            fails.append(f"{label}: E0/N0 differ from A9 ({relE:.1e}, {relN:.1e})")
        if not bins_ok:
            fails.append(f"{label}: bins differ from A9")
        if not D["sizes_ok"]:
            fails.append(f"{label}: block sizes not confirmed by the read checks")
        # groups
        allidx = np.arange(len(rows))
        grp = {"all targets": group_summary(D, allidx)}
        for k, b in enumerate(bins, start=1):
            g = group_summary(D, b)
            g["d_min"], g["d_max"] = float(d[b].min()), float(d[b].max())
            grp[f"bin {k}"] = g
        res[label] = {"groups": grp}
        # secondary 1: D3 off-axis comparator
        if D["extra"] is not None:
            X = D["extra"]; G = D["G"]; nc = D["n_c"]
            nu20 = X["varbar_20"] / B0 + X["trC"] / (G * nc); nu30 = X["varbar_30"] / B1 + X["trC"] / (G * nc)
            O0 = (X["m2_20"] - nu20) - D["A0"]; O1 = (X["m2_30"] - nu30) - D["A1"]
            nu_a9 = np.array([r["nu"] for r in rows])
            checks[label]["nu20_max_rel_diff_vs_A9"] = float(np.max(np.abs(nu20 - nu_a9) / nu_a9))
            off = {}
            for name, idx in [("all targets", allidx)] + [(f"bin {k}", b) for k, b in enumerate(bins, start=1)]:
                s0, s1 = float(O0[idx].sum()), float(O1[idx].sum())
                off[name] = {"sum_O0": s0, "sum_O1": s1, "O1_over_O0": (s1 / s0 if s0 > 0 else None),
                             "O1_over_O0_ci95": (ratio_ci(O1, O0, idx) if s0 > 0 else None),
                             "A1_over_A0": grp[name]["A1_over_A0"]}
            res[label]["off_axis_D3"] = off
        # secondary 2: link to the list costs, bottom bin
        b1 = bins[0]
        pos = np.array([j for j in b1 if D["A0"][j] > 0])
        ratio = D["C"][pos] / D["A0"][pos]
        med = float(np.median(ratio))
        lo_half = pos[ratio <= med]; hi_half = pos[ratio > med]
        res[label]["bottom_bin_link"] = {
            "targets_with_positive_A0": int(len(pos)), "median_C_over_A0": med,
            "high_replication": {"targets": int(len(hi_half)), "removal": removal_rates(rows, hi_half, pools_of[label])},
            "low_replication": {"targets": int(len(lo_half)), "removal": removal_rates(rows, lo_half, pools_of[label])}}
        # per-target rows
        bin_of = np.zeros(len(rows), np.int64)
        for k, b in enumerate(bins, start=1):
            bin_of[b] = k
        for j, r in enumerate(rows):
            row = {"source": label, "target": r["target"], "target_id": r["target_id"], "d_bin": int(bin_of[j]),
                   "d": r["d"], "w0y": r["w0y"], "W1": r["W1"],
                   "E0": float(D["E0"][j]), "N0": float(D["N0"][j]), "A0": float(D["A0"][j]), "C": float(D["C"][j]),
                   "E1": float(D["E1"][j]), "N1": float(D["N1"][j]), "A1": float(D["A1"][j]),
                   "C_over_A0": (float(D["C"][j] / D["A0"][j]) if D["A0"][j] > 0 else None)}
            for i in range(K):
                row[f"ybar0_u{i + 1}"] = float(D["y0"][j, i]); row[f"ybar1_u{i + 1}"] = float(D["y1"][j, i])
                row[f"sd0_u{i + 1}"] = float(D["sd0"][j, i]); row[f"sd1_u{i + 1}"] = float(D["sd1"][j, i])
            targets.append(row)
    print(json.dumps(checks, indent=1), flush=True)
    if fails:
        sys.exit("A10 checks failed; nothing written: " + "; ".join(fails))
    # reading (A10 section 6)
    bottom = {l: res[l]["groups"]["bin 1"] for l in ("D3", "D1", "D2")}
    above = all(g["r_C_over_A0_ci95"] is not None and g["r_C_over_A0_ci95"][0] > 0.5 for g in bottom.values())
    below = all(g["r_C_over_A0_ci95"] is not None and g["r_C_over_A0_ci95"][1] < 0.5 for g in bottom.values())
    reading = ("mostly present in independent cells" if above else "mostly absent in independent cells" if below
               else "partly present")
    calib = {l: {"r_all": res[l]["groups"]["all targets"]["r_C_over_A0"],
                 "r_all_ci95": res[l]["groups"]["all targets"]["r_C_over_A0_ci95"],
                 "ci_excludes_1": bool(not (res[l]["groups"]["all targets"]["r_C_over_A0_ci95"][0] <= 1.0 <=
                                            res[l]["groups"]["all targets"]["r_C_over_A0_ci95"][1])),
                 "r_bottom_over_r_all": bottom[l]["r_C_over_A0"] / res[l]["groups"]["all targets"]["r_C_over_A0"]}
             for l in ("D3", "D1", "D2")}
    link = []
    for l in ("D3", "D1"):
        L_ = res[l]["bottom_bin_link"]
        for p in pools_of[l]:
            h = L_["high_replication"]["removal"][p]["rate"]; lo = L_["low_replication"]["removal"][p]["rate"]
            link.append(None if h is None or lo is None else h > lo)
    link_reading = ("targets whose pressed change repeats lose trusted candidates more often when cut"
                    if all(x is True for x in link) else "no link seen")
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ,
           "note": "amendment A10; post hoc; block 0 = cells 0..19, block 1 = cells 20..49; top 5 input-control directions",
           "checks": checks, "donors": res,
           "reading": {"bottom_bin_r": {l: {"r": g["r_C_over_A0"], "ci95": g["r_C_over_A0_ci95"]} for l, g in bottom.items()},
                       "reading": reading, "calibration": calib, "link_comparisons_high_gt_low": link,
                       "link_reading": link_reading},
           "cases": [t for t in targets if t["target"] in CASES], "targets": targets}
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, "w1_direction_replication.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    cp = os.path.join(out_dir, "w1_direction_replication_targets.csv")
    with open(cp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(targets[0].keys()))
        w.writeheader(); w.writerows(targets)
    print("reading:", reading, json.dumps(out["reading"]["bottom_bin_r"]))
    print("calibration:", json.dumps(calib))
    print("link:", link_reading, link)
    print(f"wrote {jp} and {cp} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
