"""Adversarial checks on the Stage 3 W1 result. Post hoc, separate from the frozen evaluation, and not part of the
A4 decision. The primary cell moved +0.0356, which is 67% of the deep-block oracle's +0.053 at the same cell, so
the question is whether the gain is specific to what W1 claims to use.

  C1  is it the LEADING control directions, or would any of the stored directions do? The metric is rebuilt from
      directions 6..20 and 16..20, whose control eigenvalues are much smaller. W1's claim is that the discount
      works because those directions carry a large share of the noise, so a non-leading set should do much less.
  C2  does W1's weight agree better with the weight built from 100 independent deep cells? That weight is the
      Stage 1 oracle; w0y reaches Pearson 0.619 with it.
  C3  is the gain carried by a few targets? The primary cell is rescored with the 25, 50 and 100 targets of
      largest |w_W1 - w0y| removed from the candidate set.
  C4  how much of the deep-block oracle gain does W1 capture, and does the oracle still beat it?

    python3 stage3/diagnostics_w1_e4.py [--data stage3/data/e4_w1_stage3.npz] [--out stage3/results]
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(BUNDLE, "stage2"))
import evaluate_stage2 as es  # noqa: E402
import evaluate_w1_e4 as ev  # noqa: E402
from pr10 import block_bootstrap, score_rows, w_signal  # noqa: E402

Z = np.load(es.arg("--data", os.path.join(HERE, "data", "e4_w1_stage3.npz")), allow_pickle=True)
OUT = es.arg("--out", os.path.join(HERE, "results"))
SRC = es.sources()["E4"]
c = next(x for x in SRC["cells"] if x["rule"] == "responsive" and x["depth"] == "20")
S2 = json.load(open(os.path.join(BUNDLE, "stage2", "scalars", "stage2_scalars.json"), encoding="utf-8"))["sources"]["E4"]

B, n_c, G = 20.0, float(Z["n_control"]), float(Z["n_genes"])


def weights(idx, beta):
    """W1 restricted to the given direction indices"""
    lam = Z["eigenvalues"][idx]; ps = Z["proj_sum"][:, idx]; pq = Z["proj_sumsq"][:, idx]; pc = Z["proj_control"][idx]
    pY = ps / B - pc[None, :]; sj = (pq - B * (ps / B) ** 2) / (B - 1.0)
    a = Z["m2_20"] - beta * (pY ** 2).sum(1) / G
    nu = Z["varbar_20"] / B + float(Z["control_trace"]) / (G * n_c) - beta * (sj / B + lam[None, :] / n_c).sum(1) / G
    return np.clip(1.0 - nu / np.maximum(a, 1e-300), 0.0, 1.0)


def score(w, keep=None):
    Y, T, sq, KP, KT, bl = c["Y"], c["T"], c["sq"], c["KP"], c["KT"], c["blocks"]
    if keep is not None:
        Y = Y[:, keep]; T = T[:, keep]; sq = sq[keep]; KP = KP[keep]; KT = KT[keep]; w = w[keep]
    return score_rows(w[None, :] * Y, T, sq, key_pred=KP, key_truth=KT)["exp"]


base = score(c["w0y"]); raw = score(np.ones(len(c["w0y"])))
w5 = weights(np.arange(5), 0.9)
out = {"note": "post hoc adversarial checks, not part of the A4 decision",
       "primary_cell": "E4 responsive 20", "w0y": float(base.mean()), "raw": float(raw.mean()),
       "W1_primary_setting": float(score(w5).mean())}

print("C1  which directions carry the gain (beta 0.9, primary cell, vs w0y)")
lam = Z["eigenvalues"]
for label, idx in (("1..5 (the primary setting)", np.arange(5)), ("6..20", np.arange(5, 20)),
                   ("16..20", np.arange(15, 20)), ("1 only", np.arange(1)), ("11..20", np.arange(10, 20))):
    w = weights(idx, 0.9); b = block_bootstrap(score(w), base, c["blocks"])
    share = float(lam[idx].sum() / float(Z["control_trace"]))
    out.setdefault("C1_direction_sets", {})[label] = {"diff": round(b["diff"], 5),
                                                      "ci95": [round(x, 5) for x in b["ci95"]],
                                                      "control_trace_share": round(share, 4),
                                                      "n_directions": int(len(idx))}
    print(f"  {label:26s} {b['diff']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}]   "
          f"these directions carry {share * 100:.1f}% of the control trace")

print("\nC2  agreement with the weight from 100 independent deep cells")
sA = np.maximum(np.array(S2["blocks"]["A"]["V0"]["a"]) - np.array(S2["blocks"]["A"]["V0"]["nu"]), 0.0)
nu20 = np.array(S2["blocks"]["20"]["V0"]["nu"]); wA = w_signal(sA, nu20)
out["C2_pearson_with_deepA_weight"] = {"w0y": round(float(np.corrcoef(c["w0y"], wA)[0, 1]), 4),
                                       "W1": round(float(np.corrcoef(w5, wA)[0, 1]), 4)}
out["C2_spearman_with_deepA_weight"] = {"w0y": round(float(spearmanr(c["w0y"], wA).correlation), 4),
                                        "W1": round(float(spearmanr(w5, wA).correlation), 4)}
print(f"  pearson  w0y {out['C2_pearson_with_deepA_weight']['w0y']:.3f}   W1 {out['C2_pearson_with_deepA_weight']['W1']:.3f}")
print(f"  spearman w0y {out['C2_spearman_with_deepA_weight']['w0y']:.3f}   W1 {out['C2_spearman_with_deepA_weight']['W1']:.3f}")

print("\nC3  is the gain carried by a few targets? drop the ones W1 moves most")
d = np.abs(w5 - c["w0y"]); order = np.argsort(-d)
for n in (0, 25, 50, 100, 200):
    keep = np.setdiff1d(np.arange(len(w5)), order[:n])
    b = block_bootstrap(score(w5, keep), score(c["w0y"], keep), c["blocks"])
    out.setdefault("C3_drop_most_moved_targets", {})[str(n)] = {"diff": round(b["diff"], 5),
                                                                "ci95": [round(x, 5) for x in b["ci95"]]}
    print(f"  drop {n:>3d} targets: {b['diff']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}]")

print("\nC4  against the deep-block oracle at the same cell")
o = block_bootstrap(score(wA), base, c["blocks"]); v = block_bootstrap(score(w5), score(wA), c["blocks"])
out["C4_oracle"] = {"oracle_vs_w0y": round(o["diff"], 5), "W1_vs_w0y": round(float(score(w5).mean() - base.mean()), 5),
                    "share_of_oracle_gain": round(float((score(w5).mean() - base.mean()) / o["diff"]), 3),
                    "W1_vs_oracle": {"diff": round(v["diff"], 5), "ci95": [round(x, 5) for x in v["ci95"]]}}
print(f"  oracle vs w0y {o['diff']:+.4f};  W1 captures {out['C4_oracle']['share_of_oracle_gain'] * 100:.0f}% of it;"
      f"  W1 vs oracle {v['diff']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]")

os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, "stage3_w1_diagnostics.json")
json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
print(f"\nwrote {p}")
