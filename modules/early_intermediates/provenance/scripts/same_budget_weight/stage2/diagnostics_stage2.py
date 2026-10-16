"""Stage 2 post-hoc diagnostics. Written after stage2_results.json existed, to find out why no candidate helped;
nothing here enters or revisits the A2 4.5 decision.

D1  oracle by metric: for E4 and E5, the weight built from the deep block A energy under each metric m (and the
    20-cell noise under the same m) is scored in the same cells. It separates two things V1 changes at once:
    which energy is measured (the metric) and how well 20 cells estimate it. The 20-cell change against w0y is
    split exactly into a metric part (oracle m minus oracle V0) and an estimation part (the remainder), each
    with the same paired block bootstrap.
D2  whose weight V1 moves: for each query and direction, the truth top 25 are the hits; the V1 / V0 weight ratio
    of hits is compared with that of all targets (E4, E5; responsive and tercile queries).
D3  variance calibration by energy: kappa = mean (s20 - s30)^2 / mean (tau2_20 + tau2_30) within quintiles of
    the average energy (s20 + s30) / 2, V0 (E1, E4, E5).
D4  what L1 and L1kappa did to the weights: agreement with w0y, and the weight given to targets that w0y sets to
    zero (s20 <= 0).
D5  the query genes under V1 (E4): their input-control variance, the V1 gene weight they get, and their share
    of the across-target effect variance.

    python3 stage2/diagnostics_stage2.py [--scalars stage2/scalars/stage2_scalars.json] [--out stage2/results]
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate_stage2 as es  # noqa: E402
from pr10 import block_bootstrap, score_rows  # noqa: E402

SCAL = es.arg("--scalars", os.path.join(HERE, "scalars", "stage2_scalars.json"))
OUT = es.arg("--out", os.path.join(HERE, "results"))
S = json.load(open(SCAL, encoding="utf-8"))
SRC = es.sources()
METRICS = ("V0", "V1_x1", "V1_x0.25", "V1_x4", "V2")


def arr(rec, block, k, q):
    return np.array(rec["blocks"][block][k][q])


def w20(rec, k):
    return np.clip(1.0 - arr(rec, "20", k, "nu") / arr(rec, "20", k, "a"), 0.0, 1.0)


def oracle(rec, k):
    sA = np.maximum(arr(rec, "A", k, "a") - arr(rec, "A", k, "nu"), 0.0)
    v = arr(rec, "20", k, "nu")
    return sA / (sA + v)


def energy(rec, block, k="V0"):
    return arr(rec, block, k, "a") - arr(rec, block, k, "nu")


def cell(src, rule, depth="20"):
    return next(c for c in SRC[src]["cells"] if c["rule"] == rule and c["depth"] == depth)


def score(c, w):
    return score_rows(w[None, :] * c["Y"], c["T"], c["sq"], key_pred=c["KP"], key_truth=c["KT"])["exp"]


def vs(c, w, base):
    b = block_bootstrap(score(c, w), base, c["blocks"])
    return {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]]}


def hits(c, K=25):
    """per query and direction, the truth top K (self gene excluded as in scoring); returns hit counts per target"""
    T = c["T"]; cnt = np.zeros(T.shape[1])
    for qi in range(T.shape[0]):
        keep = np.flatnonzero(c["sq"] != qi)
        for sgn in (1.0, -1.0):
            cnt[keep[np.argsort(-sgn * T[qi, keep], kind="stable")[:K]]] += 1
    return cnt


out = {"note": "post hoc, written after the pre-registered evaluation; not part of the A2 4.5 decision",
       "scalars_sha256": es.sha256(SCAL)}

# D1 and D2
for src in ("E4", "E5"):
    rec = S["sources"][src]; d1 = {}; d2 = {}
    ms = [m for m in METRICS if m in rec["blocks"]["20"]]
    for rule in ("responsive", "terciles"):
        c = cell(src, rule); base = score(c, c["w0y"])
        d1[rule] = {"w0y_score": round(float(base.mean()), 5),
                    "20_cells": {m: vs(c, w20(rec, m), base) for m in ms},
                    "oracle_deepA": {m: vs(c, oracle(rec, m), base) for m in ms},
                    "decomposition_vs_V0": {}}
        o0 = score(c, oracle(rec, "V0")); v0 = score(c, w20(rec, "V0"))
        for m in ms[1:]:   # 20-cell change = metric part (oracle m - oracle V0) + estimation part (the rest)
            om = score(c, oracle(rec, m)); vm = score(c, w20(rec, m))
            mp = block_bootstrap(om, o0, c["blocks"]); ep = block_bootstrap(vm - om + o0, v0, c["blocks"])
            d1[rule]["decomposition_vs_V0"][m] = {
                "metric_part": {"diff": round(mp["diff"], 5), "ci95": [round(x, 5) for x in mp["ci95"]]},
                "estimation_part": {"diff": round(ep["diff"], 5), "ci95": [round(x, 5) for x in ep["ci95"]]}}
        h = hits(c); w0 = w20(rec, "V0"); w1 = w20(rec, "V1_x1"); both = (w0 > 0) & (w1 > 0)
        lr = np.log(w1[both] / w0[both]); hh = h[both]
        d2[rule] = {"targets_with_both_weights_positive": int(both.sum()),
                    "median_log_ratio_V1_over_V0_all": round(float(np.median(lr)), 4),
                    "median_log_ratio_hit_weighted": round(float(np.average(lr, weights=hh)) if hh.sum() else float("nan"), 4),
                    "spearman_log_ratio_vs_hit_count": round(float(spearmanr(lr, hh).correlation), 4),
                    "median_log_ratio_top_decile_by_hits": round(float(np.median(lr[hh >= np.quantile(hh, 0.9)])), 4)}
    out[f"D1_oracle_by_metric_{src}"] = d1
    out[f"D2_whose_weight_V1_moves_{src}"] = d2

# D3
for src in ("E1", "E4", "E5"):
    rec = S["sources"][src]
    s20, s30 = energy(rec, "20"), energy(rec, "30"); t = arr(rec, "20", "V0", "tau2") + arr(rec, "30", "V0", "tau2")
    avg = (s20 + s30) / 2.0; qs = np.quantile(avg, [0.2, 0.4, 0.6, 0.8]); bins = np.digitize(avg, qs)
    out[f"D3_kappa_by_energy_quintile_{src}"] = {
        "overall": round(float(np.mean((s20 - s30) ** 2) / np.mean(t)), 3),
        "by_quintile_low_to_high": [round(float(np.mean((s20[bins == b] - s30[bins == b]) ** 2) / np.mean(t[bins == b])), 3)
                                    for b in range(5)]}

# D4
for src in ("E1", "E4", "E5"):
    rec = S["sources"][src]; kappa = es.kappa_for(rec)
    W, _ = es.weights_for(src, rec, "20", kappa); base = w20(rec, "V0"); zero = base == 0
    out[f"D4_L1_weights_{src}"] = {
        "kappa": round(kappa, 3), "share_of_targets_w0y_zero": round(float(zero.mean()), 4),
        **{k: {"spearman_with_w0y": round(float(spearmanr(W[k], base).correlation), 4),
               "max_abs_diff_from_w0y": round(float(np.max(np.abs(W[k] - base))), 4),
               "mean_weight_where_w0y_zero": round(float(W[k][zero].mean()), 4) if zero.any() else None,
               "mean_weight_elsewhere": round(float(W[k][~zero].mean()), 4)} for k in ("L1", "L1kappa")},
        "w0y_mean_weight_elsewhere": round(float(base[~zero].mean()), 4)}

# D5 which genes the queries ask about, and what V1 does to them (E4, from the reduced state's control sums)
D = es.load("e4_tcell.npz"); keep = D["keep"]; cm = D["audit_ctrl_s"][0] / 2000.0
cv = ((D["audit_ctrl_q"][0] - 2000.0 * cm ** 2) / 1999.0)[keep]
mv1 = 1.0 / (cv + np.median(cv)); rv = D["resp_var"]
out["D5_query_genes_under_V1_E4"] = {
    "all_genes_median_control_variance": round(float(np.median(cv)), 4),
    **{rule: {"median_control_variance": round(float(np.median(cv[q])), 4),
              "mean_V1_gene_weight_relative_to_average": round(float(mv1[q].mean() / mv1.mean()), 3),
              "share_of_genes": round(len(q) / len(cv), 4),
              "share_of_V1_metric_mass": round(float(mv1[q].sum() / mv1.sum()), 4),
              "share_of_across_target_effect_variance": round(float(rv[q].sum() / rv.sum()), 4)}
       for rule, q in (("responsive", D["q_resp"]), ("terciles", D["q_terc"]))}}

os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, "stage2_diagnostics.json")
json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
print(f"wrote {p}")
