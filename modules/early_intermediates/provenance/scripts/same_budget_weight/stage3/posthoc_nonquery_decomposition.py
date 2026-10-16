"""POST-HOC diagnostic, not pre-declared in A4, A4a or A4d, and not part of any decision.

The A4a sensitivity "non-query genes" compares W1 computed entirely on the non-query genes with the frozen w0y on
all weight genes, so it mixes two changes: dropping the query genes from the weight's gene set, and changing the
metric. This splits the two, on the same state files the frozen evaluations read:

    w0y on non-query genes  vs  w0y on all genes          (the gene-set change alone)
    W1 on non-query genes   vs  w0y on non-query genes    (the metric change on the same genes)

The two add up exactly to the reported "W1 non-query vs w0y". Scoring and bootstrap as in v10.

    python3 stage3/posthoc_nonquery_decomposition.py [--e1 stage3/data/e1_w1_stage3.npz] [--e4 stage3/data/e4_w1_stage3.npz]
                                                     [--out stage3/results]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(BUNDLE, "stage2"))
sys.path.insert(0, os.path.join(BUNDLE, "scripts"))
import evaluate_stage2 as es  # noqa: E402
import evaluate_w1_e4 as ev  # noqa: E402
from evaluate_w1_e1 import w1_weights  # noqa: E402
from pr10 import block_bootstrap, score_rows  # noqa: E402


def score(c, w):
    return score_rows(w[None, :] * c["Y"], c["T"], c["sq"], key_pred=c["KP"], key_truth=c["KT"])["exp"]


def split(c, w0nq, w1nq):
    r = lambda b: {"diff": round(b["diff"], 5), "ci95": [round(x, 5) for x in b["ci95"]]}
    base = score(c, c["w0y"]); s0 = score(c, w0nq); s1 = score(c, w1nq)
    return {"gene_set_change__w0y_nonquery_vs_w0y": r(block_bootstrap(s0, base, c["blocks"])),
            "metric_change__W1_nonquery_vs_w0y_nonquery": r(block_bootstrap(s1, s0, c["blocks"])),
            "reported_sensitivity__W1_nonquery_vs_w0y": r(block_bootstrap(s1, base, c["blocks"]))}


def main():
    e1 = es.arg("--e1", os.path.join(HERE, "data", "e1_w1_stage3.npz"))
    e4 = es.arg("--e4", os.path.join(HERE, "data", "e4_w1_stage3.npz"))
    SRC = es.sources(); out = {"status": "post-hoc diagnostic; not pre-declared; not part of any decision"}
    Z = np.load(e1, allow_pickle=True); D = es.load("e1_stage6.npz")
    for c in SRC["E1"]["cells"]:
        d = c["depth"]
        out[f"E1 reference-variance {d}"] = split(c, w1_weights(Z, D, d, 5, 0.0, tag="_nonquery"),
                                                  w1_weights(Z, D, d, 5, 0.9, tag="_nonquery"))
    Z4 = np.load(e4, allow_pickle=True)
    c = next(x for x in SRC["E4"]["cells"] if x["rule"] == "responsive" and x["depth"] == "20")
    out["E4 D3 responsive 20"] = split(c, ev.w1(Z4, 5, 0.0, "_nonquery")[0], ev.w1(Z4, 5, 0.9, "_nonquery")[0])
    out_dir = es.arg("--out", os.path.join(HERE, "results")); os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "posthoc_nonquery_decomposition.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1)); print(f"wrote {p}")


if __name__ == "__main__":
    main()
