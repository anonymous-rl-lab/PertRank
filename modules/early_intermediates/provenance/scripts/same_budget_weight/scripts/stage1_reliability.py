"""Stage 1, descriptive: how reliable is the 20-cell energy estimate as an ordering of targets?

Two disjoint input blocks give two unbiased estimates of the same per-target energy s_j (the blocks share the
input control, so their errors are positively correlated through the control mean; about 1% of the 20-cell
noise per gene on the T cell probe):
    s_m = m2_m - nu_m      (unclipped; m2 = mean_g y^2 over the weight gene set, nu = noise of the m-cell mean)
Reported: their correlation across targets (T cell 20 vs 30 cells, and the two 100-cell deep blocks A vs B
as a reference; KOLF batch-matched 20 vs 30; Replogle 20 vs 30), and how the squared difference
d2 = (s_20 - s_30)^2 varies with the energy itself and with the per-cell noise level vb (T cell),
each quintile's mean d2 divided by the overall mean.

This file reproduces the computation that produced results/stage1_reliability.json during Stage 1 (it was
first run inline). Rounding (3 decimals for correlations, 2 for ratios) is part of that output and is kept.
Amendment A2: deep B is centred on the evaluation control, so s_B now subtracts the evaluation-control
variance. The change is a constant shift across targets and leaves every correlation unchanged.

Usage (from the bundle folder):  python3 scripts/stage1_reliability.py [--out DIR]
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, pearsonr

from _paths import amendment_sha, data_file, import_pr10, out_dir, plan_sha

plan_sha()          # refuse to run if the plan differs from its recorded hash
amendment_sha()     # likewise for amendment A2
import_pr10()
from pr10 import nu  # noqa: E402


def rel(a, b):
    return {"pearson": round(float(pearsonr(a, b)[0]), 3), "spearman": round(float(spearmanr(a, b).correlation), 3)}


out = {}

# E4 CD4+ T cell probe
D = np.load(data_file("e4_tcell.npz"))
cvm = float(D["cvm"])
s = lambda k, n: D[f"m2_{k}"] - nu(D[f"vb_{k}"], n, cvm)
s20, s30, sA = s("20", 20), s("30", 30), s("A", 100)
_cs, _cq = D["audit_ctrl_s"][1], D["audit_ctrl_q"][1]          # evaluation control (A2)
cve = float(((_cq - 2000 * (_cs / 2000.0) ** 2) / 1999.0)[D["keep"]].mean())
sB = D["m2_B"] - nu(D["vb_B"], 100, cve)
out["T cell"] = {"s20 vs s30 (independent 20 and 30 cells)": rel(s20, s30),
                 "sA vs sB (independent 100 and 100 cells)": rel(sA, sB),
                 "s20 vs sA": rel(s20, sA)}
d2 = (s20 - s30) ** 2
sbar = (20 * s20 + 30 * s30) / 50
q = np.quantile(sbar, [.2, .4, .6, .8]); b = np.digitize(sbar, q)
out["T cell"]["d2 by energy quintile / overall mean"] = [round(float(d2[b == i].mean() / d2.mean()), 2) for i in range(5)]
qv = np.quantile(D["vb_20"], [.2, .4, .6, .8]); bv = np.digitize(D["vb_20"], qv)
out["T cell"]["d2 by vb quintile / overall mean"] = [round(float(d2[bv == i].mean() / d2.mean()), 2) for i in range(5)]

# E5 KOLF2.1J probe, batch-matched controls
D = np.load(data_file("e5_kolf.npz"))
tb = D["target_batch"]; cm = D["cvm_mfull"]
sm = lambda a: D[f"m2_mfull_{a}"] - (D[f"vb_full_{a}"] / int(a) + cm[tb] / 2000)
out["KOLF matched"] = {"s20 vs s30": rel(sm("20"), sm("30"))}

# E1 Replogle K562 genome-wide
D = np.load(data_file("e1_stage6.npz"))
cvm = float(D["cvm"]); nc = int(D["ncin"])
sr = lambda a: D[f"m2_{a}"] - nu(D[f"vb{a}"], int(a), cvm, nc)
out["Replogle"] = {"s20 vs s30": rel(sr("20"), sr("30"))}

path = os.path.join(out_dir(sys.argv), "stage1_reliability.json")
json.dump(out, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", path)
print(json.dumps(out, indent=1, ensure_ascii=False))
