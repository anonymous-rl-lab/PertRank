"""Amendment A11, question 2: on D1, which responsive query genes carry TP53's and RFX5's change along direction 1,
and how far those responses are kept in independent cells (block 1), in D1's deep truth, and in D3's and D2's truth.
Post hoc; stored states, the A8 lock and the A8 relation table only.

    python3 stage3/direction1_q2_cases.py [--out stage3/results]

Writes results/direction1_cases_D1.json and results/direction1_cases_D1_genes.csv.
"""
import csv
import json
import os
import time

import numpy as np
from scipy.stats import spearmanr

import a6_common as A
import lock_b7_sources as L  # noqa: E402

CASES = ("TP53", "RFX5")
HALF = 0.5


def agree(x, y):
    m = (x != 0) & (y != 0)
    return {"genes": int(m.sum()), "sign_agreement": (float((np.sign(x[m]) == np.sign(y[m])).mean()) if m.sum() else None),
            "spearman": (float(spearmanr(x[m], y[m]).correlation) if m.sum() > 2 else None)}


def main():
    t0 = time.time()
    out_dir = A.arg("--out", os.path.join(A.HERE, "results"))
    integ = {"amendment_A11_sha256": A.checked_sha(os.path.join(A.BUNDLE, "AMENDMENT_A11.md"),
                                                   os.path.join(A.BUNDLE, "AMENDMENT_A11_SHA256.txt"), "amendment A11"),
             "script_sha256": A.sha256(os.path.abspath(__file__))}
    sym = json.load(open(os.path.join(A.HERE, "data", "tcell_gene_symbols.json"), encoding="utf-8"))["symbols"]
    Z1 = A.d1_load()
    Lk = np.load(L.LOCK_NPZ, allow_pickle=True)
    Z2 = np.load(os.path.join(A.HERE, "data", "d2_prospective_state.npz"), allow_pickle=True)
    rel = json.load(open(os.path.join(A.HERE, "results", "b7_relation_table.json"), encoding="utf-8"))["relations"]
    names1 = [str(x) for x in Z1["names"]]; genes1 = np.array([str(g) for g in Z1["genes_keep"]])
    qu = Z1["qu"]; qgenes = genes1[qu]
    pos_resp = np.searchsorted(qu, Z1["q_resp"])           # responsive queries as rows of the *_q arrays
    resp_genes = qgenes[pos_resp]
    u1 = Z1["directions"][:, 0]; pc1 = float(Z1["proj_control"][0])
    # D3 truth on D3's query union, D2's S2 truth on the 692 truth genes
    d3_genes = [str(g) for g in Lk["D3_genes"]]; d3_row = {g: i for i, g in enumerate(d3_genes)}
    d3_names = [str(x) for x in Lk["D3_names"]]
    s2 = [str(x) for x in Z2["s2_names"]]; tg = [str(g) for g in Z2["truth_genes"]]; tg_row = {g: i for i, g in enumerate(tg)}
    n100 = float(Z2["s2_cells_per_target"])
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ,
           "note": "amendment A11 question 2; D1 source; responsive list; direction 1 = the first stored input-control "
                   "direction of D1 (its sign is opposite to D3's, see direction1_similarity.json)",
           "targets": {}}
    rows = []
    for t in CASES:
        j = names1.index(t)
        y0 = Z1["S0_q"][:, j] / 20.0 - Z1["cm"][qu]
        y1 = Z1["S1_q"][:, j] / 30.0 - Z1["cm"][qu]
        T1 = Z1["S4_q"][:, j] / 100.0 - Z1["ce"][qu]
        c1 = float(Z1["proj_sum"][0][j, 0] / 20.0 - pc1)
        comp = c1 * u1[qu]
        # cross-donor truths on the responsive genes, where present
        j3 = d3_names.index(t); j2 = s2.index(t)
        T3 = np.full(len(qu), np.nan); T2 = np.full(len(qu), np.nan)
        for k, g in enumerate(qgenes):
            if g in d3_row:
                T3[k] = Lk["D3_T"][d3_row[g], j3]
            if g in tg_row:
                T2[k] = Z2["T2_sum"][j2, tg_row[g]] / n100 - Z2["ev_truth_mean"][tg_row[g]]
        r = pos_resp
        along = (np.abs(comp[r]) >= HALF * np.abs(y0[r])) & (np.sign(comp[r]) == np.sign(y0[r])) & (y0[r] != 0)
        groups = {"along direction 1": r[along], "not along direction 1": r[~along]}
        info = {"c1 (block-0 projection on direction 1)": c1,
                "share of block-0 responsive energy carried by direction 1": float((comp[r] ** 2).sum() / (y0[r] ** 2).sum()),
                "groups": {}}
        for gname, idx in groups.items():
            g = {"genes": int(len(idx)),
                 "share of block-0 responsive energy": float((y0[idx] ** 2).sum() / (y0[r] ** 2).sum()),
                 "block 1 (independent cells)": agree(y0[idx], y1[idx]),
                 "D1 deep truth": agree(y0[idx], T1[idx]),
                 "D3 truth": agree(y0[idx][~np.isnan(T3[idx])], T3[idx][~np.isnan(T3[idx])]),
                 "D2 truth": agree(y0[idx][~np.isnan(T2[idx])], T2[idx][~np.isnan(T2[idx])])}
            info["groups"][gname] = g
        # the w0y-specific relations of this target (D1 source) split by the grouping
        along_set = set(qgenes[r[along]].tolist())
        split = {}
        for r_ in rel:
            if r_["source"] != "D1" or r_["target"] != t or r_["kind"] != "w0y-specific":
                continue
            key = "along direction 1" if r_["gene"] in along_set else "not along direction 1"
            s = split.setdefault(key, {"relations": 0, "D2 rank support": 0, "D2 effect support": 0,
                                       "other donor measurable": 0, "other donor support": 0})
            s["relations"] += 1; s["D2 rank support"] += int(r_["D2_rank_support"]); s["D2 effect support"] += int(r_["D2_effect_support"])
            s["other donor measurable"] += int(r_["other_donor_support"] is not None)
            s["other donor support"] += int(bool(r_["other_donor_support"]))
        for s in split.values():
            s["D2 rank support rate"] = s["D2 rank support"] / s["relations"]
            s["D2 effect support rate"] = s["D2 effect support"] / s["relations"]
            s["other donor support rate"] = (s["other donor support"] / s["other donor measurable"]) if s["other donor measurable"] else None
        info["w0y-specific relations split"] = split
        # top responses along direction 1, for reading
        order = r[along][np.argsort(-np.abs(y0[r[along]]))]
        info["largest responses along direction 1"] = [
            {"symbol": sym.get(qgenes[k], qgenes[k]), "block0": round(float(y0[k]), 3), "component": round(float(comp[k]), 3),
             "block1": round(float(y1[k]), 3), "D1 truth": round(float(T1[k]), 3),
             "D3 truth": (None if np.isnan(T3[k]) else round(float(T3[k]), 3)),
             "D2 truth": (None if np.isnan(T2[k]) else round(float(T2[k]), 3))} for k in order[:25]]
        out["targets"][t] = info
        for k in r:
            rows.append({"target": t, "gene": qgenes[k], "symbol": sym.get(qgenes[k], qgenes[k]),
                         "along_direction_1": bool(k in set(r[along].tolist())), "block0": float(y0[k]),
                         "direction1_component": float(comp[k]), "block1": float(y1[k]), "D1_truth": float(T1[k]),
                         "D3_truth": (None if np.isnan(T3[k]) else float(T3[k])), "D2_truth": (None if np.isnan(T2[k]) else float(T2[k])),
                         "loading_direction_1": float(u1[qu][k])})
        print(t, json.dumps({k: v for k, v in info.items() if k != "largest responses along direction 1"}, indent=1), flush=True)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "direction1_cases_D1.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(out_dir, "direction1_cases_D1_genes.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
