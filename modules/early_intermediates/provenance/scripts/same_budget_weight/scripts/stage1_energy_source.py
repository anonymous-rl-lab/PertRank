"""Stage 1 of PLAN_same_budget_weight.md: which error dominates the 20-cell weight's energy estimate.

Fixed 20-cell prediction Y20 and noise term nu20; only the energy source m changes:
    s_m = [mean_g(y_m^2) - nu_m]_+ ,  w_{m->20} = s_m / (s_m + nu20)      (own 20 cells: exactly w0y)
Sources: own 0-19, disjoint 20-49, union 0-49, union 0-99, deep A 100-199, oracle deep B (diagnostic),
and the split cross-moment X between 0-19 and 20-49, recovered from stored m2 values:
    m2_50 = 0.16 m2_20 + 0.36 m2_30 + 0.48 X   (same control centring), s_X = [X - cvm/n_c]_+
Orion: disjoint 20-99 (80 cells), union 0-99, X(0-19, 20-99) from m2_100 = 0.04 m2_20 + 0.64 m2_80 + 0.32 X.
Scoring: tie-expectation precision@25, 30-block paired bootstrap, B=2000, seed 2026 (package code).
Readings for the primary setting (T cell responsive) are fixed in the plan before this script ran.

Usage (from the bundle folder):  python3 scripts/stage1_energy_source.py [--out DIR]
Paths are resolved by scripts/_paths.py (PR10_ROOT, PR10_DATA).

Amendment A2 (external audit of Stage 1) changed three things here, none of them in the main
energy-source comparison: deep B, centred on the evaluation control, now subtracts the evaluation-
control variance (was the input-control variance); the covariate check uses average ranks for ties
and is also reported on mapped targets only; the reading label attached in the plan is kept for the
record and the wording adopted in A2 is added beside it.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr, pearsonr

from _paths import OWN_MAP, amendment_sha, data_file, import_pr10, out_dir, plan_sha

import_pr10()
from pr10 import block_bootstrap, nu, score_rows, self_to_query, w0y, w_signal  # noqa: E402

OWN = np.load(OWN_MAP, allow_pickle=True)["own_row"]
PLAN_SHA = plan_sha()
A2_SHA = amendment_sha()


def keys(n):
    return np.random.default_rng(101).random(n), np.random.default_rng(202).random(n)


def summ(bb):
    return {"diff": bb["diff"], "ci95": bb["ci95"], "p_one_sided": bb["p_one_sided"]}


def corr(a, b):
    return {"pearson": float(pearsonr(a, b)[0]), "spearman": float(spearmanr(a, b).correlation)}


def evaluate(Y, T, self_q, blocks, KP, KT, W, refs):
    """W: dict name -> per-target weight (must contain 'own20' = w0y); refs: names used as ordering references"""
    sr = lambda E: score_rows(E, T, self_q, key_pred=KP, key_truth=KT)
    raw = sr(Y)
    S = {k: sr(w[None, :] * Y) for k, w in W.items()}
    out = {"raw": float(raw["exp"].mean()), "sources": {}}
    for k, w in W.items():
        rec = {"score": float(S[k]["exp"].mean()),
               "vs_w0y": summ(block_bootstrap(S[k]["exp"], S["own20"]["exp"], blocks)),
               "vs_raw": summ(block_bootstrap(S[k]["exp"], raw["exp"], blocks)),
               "w_mean": float(w.mean()), "w_zero_frac": float(np.mean(w == 0))}
        for r in refs:
            if r in W and r != k:
                rec[f"corr_with_{r}"] = corr(w, W[r])
        out["sources"][k] = rec
    pairs = [("deepA", "union100"), ("union50", "disjoint30"), ("crossX", "union50"), ("disjoint80", "union100"),
             ("crossX80", "union100")]
    out["contrasts"] = {f"{a} - {b}": summ(block_bootstrap(S[a]["exp"], S[b]["exp"], blocks))
                        for a, b in pairs if a in S and b in S}
    return out


def variance_heterogeneity(s20, s30, vb20, sbar):
    """two unbiased estimates of the same s_j from disjoint cell blocks. The blocks share the input control,
    so their errors are positively correlated (shared control-mean error; about 1% of the 20-cell noise per
    gene on the T cell probe): E[d2] = tau_20^2 + tau_30^2 - 2 cov. Report how d2 varies with vb (noise
    level) and with the energy itself."""
    d2 = (s20 - s30) ** 2
    res = {"mean_d2": float(d2.mean()),
           "spearman_d2_vs_vb20": float(spearmanr(d2, vb20).correlation),
           "spearman_d2_vs_sbar": float(spearmanr(d2, sbar).correlation)}
    for nm, x in (("vb20", vb20), ("sbar", sbar)):
        q = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
        b = np.digitize(x, q)
        res[f"mean_d2_by_{nm}_quintile"] = [float(d2[b == i].mean()) for i in range(5)]
    return res


def covariate_information(x, s20, sA, w20, mapped, B=2000, seed=2026):
    """does x carry information about the deep-A energy beyond the 20-cell estimate? Partial Spearman with
    average ranks for ties, and the R^2 increment of rank(sA) on rank(s20) when rank(x) is added.
    x = 0 marks targets whose own gene is not in the weight gene set (off the gene axis, or zero mean in
    the input controls), so the check is reported on all targets and on the mapped targets only; for the
    latter a bootstrap over targets gives an interval for the partial correlation."""
    r = lambda v: (rankdata(v, method="average") - 1) / (len(v) - 1)

    def one(xv, s, a):
        rs, rx, ra = r(s), r(xv), r(a)
        Xb = np.c_[np.ones_like(rs), rs]
        res_a = ra - Xb @ np.linalg.lstsq(Xb, ra, rcond=None)[0]
        res_x = rx - Xb @ np.linalg.lstsq(Xb, rx, rcond=None)[0]
        X2 = np.c_[Xb, rx]
        fit2 = ra - X2 @ np.linalg.lstsq(X2, ra, rcond=None)[0]
        tss = np.sum((ra - ra.mean()) ** 2)
        return {"partial_spearman_x_sA_given_s20": float(np.corrcoef(res_a, res_x)[0, 1]),
                "r2_rank_sA_on_s20": float(1 - np.sum(res_a ** 2) / tss),
                "r2_rank_sA_on_s20_and_x": float(1 - np.sum(fit2 ** 2) / tss), "n": int(len(xv))}

    out = {"all_targets": one(x, s20, sA), "mapped_only": one(x[mapped], s20[mapped], sA[mapped])}
    rng = np.random.default_rng(seed); idx = np.flatnonzero(mapped)
    bs = [one(x[b], s20[b], sA[b])["partial_spearman_x_sA_given_s20"]
          for b in (rng.choice(idx, len(idx), replace=True) for _ in range(B))]
    out["mapped_only"]["partial_ci95_bootstrap_over_targets"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    z = w20 == 0
    out["zero_weight"] = {"n": int(z.sum()), "n_mapped": int(np.sum(z & mapped)), "n_unmapped": int(np.sum(z & ~mapped)),
                          "mean_sA_mapped": float(sA[z & mapped].mean()) if np.any(z & mapped) else None,
                          "mean_sA_unmapped": float(sA[z & ~mapped].mean()) if np.any(z & ~mapped) else None}
    return out


RESULTS = {"plan_sha256": PLAN_SHA, "amendment_A2_sha256": A2_SHA, "scoring": "tie expectation (exp), 30-block paired bootstrap, B=2000, seed 2026"}

# ---------------- E4 CD4+ T cell probe (corrected own-gene map) ----------------
D = np.load(data_file("e4_tcell.npz"), allow_pickle=True)
NP = len(D["names"]); KP, KT = keys(NP); cvm = float(D["cvm"])
nn = {"20": 20, "30": 30, "50": 50, "100": 100, "A": 100, "B": 100}
NU = {k: nu(D[f"vb_{k}"], n, cvm) for k, n in nn.items()}
# A2: deep B is centred on the evaluation control, so its noise term takes the evaluation-control variance
_cs, _cq = D["audit_ctrl_s"][1], D["audit_ctrl_q"][1]
cve = float(((_cq - 2000 * (_cs / 2000.0) ** 2) / 1999.0)[D["keep"]].mean())
NU["B"] = nu(D["vb_B"], 100, cve)
SIG = {k: np.maximum(D[f"m2_{k}"] - NU[k], 0.0) for k in nn}
X = (D["m2_50"] - 0.16 * D["m2_20"] - 0.36 * D["m2_30"]) / 0.48 - cvm / 2000.0
nu20 = NU["20"]
W4 = {"own20": w0y(nu20, D["m2_20"]), "disjoint30": w_signal(SIG["30"], nu20), "union50": w_signal(SIG["50"], nu20),
      "union100": w_signal(SIG["100"], nu20), "deepA": w_signal(SIG["A"], nu20), "oracleB": w_signal(SIG["B"], nu20),
      "crossX": w_signal(np.maximum(X, 0.0), nu20)}
assert np.allclose(w_signal(SIG["20"], nu20), W4["own20"])
e4 = {}
for rule, q in (("terciles", D["q_terc"]), ("responsive", D["q_resp"])):
    pos = np.searchsorted(D["qu"], q)
    Y20 = D["S0_q"][pos] / 20.0 - D["cm_q"][pos][:, None]
    T = D["S4_q"][pos] / 100.0 - D["ce_q"][pos][:, None]
    e4[rule] = evaluate(Y20, T, self_to_query(OWN, q), D[f"blocks_{rule}"], KP, KT, W4, refs=("deepA", "oracleB"))
s20u = D["m2_20"] - NU["20"]; s30u = D["m2_30"] - NU["30"]
e4["variance_heterogeneity"] = variance_heterogeneity(s20u, s30u, D["vb_20"], (20 * s20u + 30 * s30u) / 50)
xcov = np.where(OWN >= 0, D["cm_keep"][np.clip(OWN, 0, None)], 0.0)
e4["covariate_target_gene_control_mean"] = covariate_information(xcov, s20u, SIG["A"], W4["own20"], OWN >= 0)
e4["control_variance"] = {"input": cvm, "evaluation": cve}
RESULTS["E4_tcell"] = e4

# ---------------- E5 KOLF2.1J probe: batch-matched (primary) and pooled ----------------
D = np.load(data_file("e5_kolf.npz"), allow_pickle=True)
NP = len(D["names"]); KP, KT = keys(NP); tb = D["target_batch"]; qu = D["qu"]
cvm_m = D["cvm_mfull"]; cvm_p = float(D["cvm_pfull"])
e5 = {"matched": {}, "pooled": {}}
nuM = {a: D[f"vb_full_{a}"] / int(a) + cvm_m[tb] / 2000.0 for a in ("20", "30", "50", "100")}
sM = {a: np.maximum(D[f"m2_mfull_{a}"] - nuM[a], 0.0) for a in nuM}
XM = (D["m2_mfull_50"] - 0.16 * D["m2_mfull_20"] - 0.36 * D["m2_mfull_30"]) / 0.48 - cvm_m[tb] / 2000.0
WM = {"own20": w0y(nuM["20"], D["m2_mfull_20"]), "disjoint30": w_signal(sM["30"], nuM["20"]),
      "union50": w_signal(sM["50"], nuM["20"]), "union100": w_signal(sM["100"], nuM["20"]),
      "crossX": w_signal(np.maximum(XM, 0.0), nuM["20"])}
nuP = {a: nu(D[f"vb_full_{a}"], (100 if a in "AB" else int(a)), cvm_p) for a in ("20", "30", "50", "100", "A", "B")}
_ps, _pq = D["audit_ctrl_pooled_s"][1], D["audit_ctrl_pooled_q"][1]     # A2: evaluation control for deep B
cvp_eval = float(((_pq - 2000 * (_ps / 2000.0) ** 2) / 1999.0).mean())
nuP["B"] = nu(D["vb_full_B"], 100, cvp_eval)
sP = {a: np.maximum(D[f"m2_pfull_{a}"] - nuP[a], 0.0) for a in nuP}
XP = (D["m2_pfull_50"] - 0.16 * D["m2_pfull_20"] - 0.36 * D["m2_pfull_30"]) / 0.48 - cvm_p / 2000.0
WP = {"own20": w0y(nuP["20"], D["m2_pfull_20"]), "disjoint30": w_signal(sP["30"], nuP["20"]),
      "union50": w_signal(sP["50"], nuP["20"]), "union100": w_signal(sP["100"], nuP["20"]),
      "deepA": w_signal(sP["A"], nuP["20"]), "oracleB": w_signal(sP["B"], nuP["20"]),
      "crossX": w_signal(np.maximum(XP, 0.0), nuP["20"])}
for rule, q in (("terciles", D["q_terc"]), ("responsive", D["q_resp"])):
    pos = np.searchsorted(qu, q); self_q = self_to_query(D["self_row"], q)
    S0 = D["S0_q"][pos]; S4 = D["S4_q"][pos]
    cmB = D["cmB_q"][:, pos][tb].T; ceB = D["ceB_q"][:, pos][tb].T
    cm = D["cm_q"][pos][:, None]; ce = D["ce_q"][pos][:, None]
    blocks = D[f"blocks_{rule}"]
    e5["matched"][rule] = evaluate(S0 / 20.0 - cmB, S4 / 100.0 - ceB, self_q, blocks, KP, KT, WM, refs=())
    e5["pooled"][rule] = evaluate(S0 / 20.0 - cm, S4 / 100.0 - ce, self_q, blocks, KP, KT, WP, refs=("deepA", "oracleB"))
s20m = D["m2_mfull_20"] - nuM["20"]; s30m = D["m2_mfull_30"] - nuM["30"]
e5["matched"]["variance_heterogeneity"] = variance_heterogeneity(s20m, s30m, D["vb_full_20"], (20 * s20m + 30 * s30m) / 50)
e5["note"] = ("matched state has no deep-A or deep-B energy in the reduced state; covariate check skipped "
              "because the only deep energies are pooled-control and batch-confounded")
e5["pooled"]["control_variance"] = {"input": cvm_p, "evaluation": cvp_eval}
RESULTS["E5_kolf"] = e5

# ---------------- E3 Orion HCT116, frozen gene set ----------------
D = np.load(data_file("e3_orion.npz"), allow_pickle=True)
g = lambda k: D[f"fz_{k}"]
NT = len(D["cohort"]); KP, KT = keys(NT); cvm = float(g("cvm"))
qrows = g("qrows"); self_q = self_to_query(g("self_row"), qrows)
Y20 = g("sh_s_q") / 20.0 - g("cin_q")[:, None]
T = g("ev_s_q") / 100.0 - g("ecm_q")[:, None]
nu20 = nu(g("vb20"), 20, cvm)
s80 = np.maximum(g("m2_80") - nu(g("vb20"), 80, cvm), 0.0)          # 80-cell variance implied by the 20-cell block
s100 = np.maximum(g("m2_100") - nu(g("vb20"), 100, cvm), 0.0)
X80 = (g("m2_100") - 0.04 * g("m2_20") - 0.64 * g("m2_80")) / 0.32 - cvm / 2000.0
W3 = {"own20": w0y(nu20, g("m2_20")), "disjoint80": w_signal(s80, nu20), "union100": w_signal(s100, nu20),
      "crossX80": w_signal(np.maximum(X80, 0.0), nu20)}
RESULTS["E3_orion"] = {"terciles": evaluate(Y20, T, self_q, g("blocks"), KP, KT, W3, refs=()),
                       "note": "the 80- and 100-cell noise terms use the 20-cell per-cell variance (not stored for the 80-cell block); the cross-moment needs no noise term"}

# ---------------- E1 Replogle K562 genome-wide ----------------
D = np.load(data_file("e1_stage6.npz"), allow_pickle=True)
cand = D["cand"]; qrows = D["qrows"]
self_q = self_to_query(D["self_row"][cand], qrows)
KPa, KTa = keys(len(D["targets"])); KP, KT = KPa[cand], KTa[cand]
cvm = float(D["cvm"]); ncin = int(D["ncin"])
Y20 = D["om20_q"].astype(np.float64) - D["cin_q"].astype(np.float64)[:, None]
NU1 = {a: nu(D[f"vb{a}"], int(a), cvm, ncin) for a in ("20", "30", "50")}
S1 = {a: np.maximum(D[f"m2_{a}"] - NU1[a], 0.0) for a in NU1}
X1 = (D["m2_50"] - 0.16 * D["m2_20"] - 0.36 * D["m2_30"]) / 0.48 - cvm / ncin
W1 = {"own20": w0y(NU1["20"], D["m2_20"]), "disjoint30": w_signal(S1["30"], NU1["20"]),
      "union50": w_signal(S1["50"], NU1["20"]), "crossX": w_signal(np.maximum(X1, 0.0), NU1["20"])}
e1 = {"reference_variance": evaluate(Y20, D["Tq"], self_q, D["blocks"], KP, KT, W1, refs=())}
s20r = D["m2_20"] - NU1["20"]; s30r = D["m2_30"] - NU1["30"]
e1["variance_heterogeneity"] = variance_heterogeneity(s20r, s30r, D["vb20"], (20 * s20r + 30 * s30r) / 50)
RESULTS["E1_replogle"] = e1

# ---------------- pre-declared reading, primary setting ----------------
P = RESULTS["E4_tcell"]["responsive"]
sc = {k: v["score"] for k, v in P["sources"].items()}
AU = P["contrasts"]["deepA - union100"]
a_u_includes_zero = AU["ci95"][0] <= 0 <= AU["ci95"][1]
theory = (sc["union50"] >= sc["disjoint30"]) and a_u_includes_zero and (sc["union50"] >= sc["crossX"])
coupling = ((AU["diff"] > 0 and not a_u_includes_zero) or
            (sc["disjoint30"] > sc["union50"] and sc["crossX"] > sc["union50"]))
RESULTS["reading_primary"] = {
    "setting": "E4 T cell, responsive queries, 20-cell prediction",
    "union50_ge_disjoint30": bool(sc["union50"] >= sc["disjoint30"]),
    "deepA_minus_union100_ci_includes_zero": bool(a_u_includes_zero),
    "union50_ge_crossX": bool(sc["union50"] >= sc["crossX"]),
    "reading": "theory holds (variance dominant)" if theory else ("coupling present" if coupling else "mixed"),
    "reading_label_status": "label attached to this outcome in the hashed plan; withdrawn by amendment A2 as overstated",
    "reading_adopted_A2": ("supports prioritising better energy estimation; no additional advantage of independent cells "
                           "over the partly overlapping input pool detected") if theory else None}

out = os.path.join(out_dir(sys.argv), "stage1_results.json")
json.dump(RESULTS, open(out, "w"), indent=1)
print("wrote", out)
print(json.dumps(RESULTS["reading_primary"], indent=1))
