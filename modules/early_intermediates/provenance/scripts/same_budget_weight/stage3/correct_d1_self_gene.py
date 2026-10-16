"""Amendment A6a: the own-gene correction of the D1 confirmation. Frozen before it produced anything
(stage3/A6_FREEZE_SHA256.txt).

read_d1_confirm.py matched target gene symbols against a gene axis that carries Ensembl identifiers, so self_row is
-1 for all 1,127 targets: own-target entries were never excluded at scoring, and the responsive rule never zeroed
them before ranking genes by variance (A6a section 1). This script

  1. builds the target -> Ensembl map from the D1 obs (perturbed_gene_name paired with perturbed_gene_id over the
     eligible targeting cells; exactly one identifier per target, and the same identifier as D3's own_gene_map.npz
     for every shared target), and stores it as stage3/data/d1_own_gene_map.npz;
  2. checks the frozen responsive list: only own genes change when own entries are zeroed, and only downwards, so
     if every own gene on the list keeps a variance at or above the list's 300th variance, the corrected list is
     the frozen list (sufficient condition, from the stored input-block sums);
  3. rescores the four D1 cells with the own gene excluded, beside the as-executed scoring, which must reproduce
     stage3/results/stage3_d1_confirm.json to 1e-12 (A6 section 2).

    python3 stage3/correct_d1_self_gene.py [--out stage3/results]

The first run reads the D1 obs from the public bucket (about a minute) to build the map; later runs use the map.
"""
import json
import os
import sys
import time

import numpy as np

import a6_common as A
from a6_common import block_bootstrap, exp_scores, fmt, summ

SCRIPTS = ["correct_d1_self_gene.py"]
URL = "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/D1_Rest.assigned_guide.h5ad"


def build_map(Z):
    sys.path.insert(0, os.path.join(A.BUNDLE, "stage2", "v2_mask"))
    import h5py
    from rangefile import RangeFile
    f = h5py.File(RangeFile(URL, block_size=4 << 20, max_blocks=16), "r")
    o = f["obs"]

    def cat(name):
        x = o[name]
        c = np.array([v.decode() if isinstance(v, bytes) else v for v in x["categories"][:]])
        return c, x["codes"][:].astype(np.int64)

    gn, gnc = cat("perturbed_gene_name"); gi, gic = cat("perturbed_gene_id"); tc, tcode = cat("guide_type")
    lq = o["low_quality"][:]
    f.close()
    tgt = int(np.flatnonzero(tc == "targeting")[0])
    ok = (~lq) & (gnc >= 0) & (gic >= 0) & (tcode == tgt)
    pairs = np.unique(np.stack([gnc[ok], gic[ok]], 1), axis=0)
    code = {str(n): i for i, n in enumerate(gn)}
    names = [str(x) for x in Z["names"]]
    ids = []
    for n in names:
        u = np.unique(pairs[pairs[:, 0] == code[n], 1]) if n in code else np.array([], np.int64)
        if len(u) != 1:
            sys.exit(f"target {n} has {len(u)} gene identifiers among its eligible cells; the map is not one-to-one")
        ids.append(str(gi[u[0]]))
    gk = {str(g): i for i, g in enumerate(Z["genes_keep"])}
    own = np.array([gk.get(i, -1) for i in ids], np.int64)
    O = np.load(os.path.join(A.BUNDLE, "inputs", "own_gene_map.npz"), allow_pickle=True)
    d3 = {str(n): str(i) for n, i in zip(O["names"], O["ids"])}
    shared = [n for n in names if n in d3]
    bad = [n for n, i in zip(names, ids) if n in d3 and d3[n] != i]
    if bad:
        sys.exit(f"{len(bad)} shared targets map to a different identifier than in D3's map, e.g. {bad[:5]}")
    np.savez_compressed(A.D1_MAP, names=np.array(names), ids=np.array(ids), own_row=own,
                        genes_keep=Z["genes_keep"], source=URL)
    return {"built_from": URL, "targets": len(names), "own_gene_on_weight_axis": int((own >= 0).sum()),
            "shared_with_D3_checked": len(shared)}


def main():
    t0 = time.time()
    out_dir = A.arg("--out", os.path.join(A.HERE, "results"))
    integ = A.integrity(SCRIPTS)
    Z = A.d1_load()
    integ["state_sha256"] = A.sha256(A.D1_STATE)
    if not os.path.isfile(A.D1_MAP):
        print("built the D1 own-gene map:", json.dumps(build_map(Z)), flush=True)
    integ["map_sha256"] = A.sha256(A.D1_MAP)
    own = A.d1_own_row(Z)
    info = {"source": "perturbed_gene_id of the D1 obs, paired with perturbed_gene_name over eligible targeting cells",
            "targets": int(len(own)), "own_gene_on_weight_axis": int((own >= 0).sum()),
            "frozen_self_row_all_minus_one": bool((Z["self_row"] == -1).all())}
    for rule, q in (("responsive", Z["q_resp"]), ("terciles", Z["q_terc"])):
        info[f"{rule}_queries_that_are_own_genes"] = int(np.isin(q, own[own >= 0]).sum())
    print("map:", json.dumps(info), flush=True)

    # 2. the frozen responsive list, from the stored input-block sums (blocks 0, 1, 2 = the 100-cell input side)
    qu = Z["qu"]
    pool = (Z["S0_q"] + Z["S1_q"] + Z["S2_q"]) / 100.0 - Z["cm"][qu][:, None]
    pr = np.searchsorted(qu, Z["q_resp"])
    v = pool[pr].var(1); v300 = float(v.min())
    others = np.setdiff1d(np.arange(len(qu)), pr)
    listed_is_top = bool(pool[others].var(1).max() <= v300) if len(others) else True
    risk = []
    for r, g in zip(pr, Z["q_resp"]):
        js = np.flatnonzero(own == g)
        if len(js):
            row = pool[r].copy(); row[js] = 0.0
            vz = float(row.var())
            if vz < v300:
                risk.append({"gene": str(Z["genes_keep"][g]), "targets": [str(Z["names"][j]) for j in js],
                             "variance_frozen": float(v[list(pr).index(r)]), "variance_zeroed": vz})
    qcheck = {"own_genes_on_list": info["responsive_queries_that_are_own_genes"], "variance_300th": v300,
              "frozen_list_is_top300_of_stored_union": listed_is_top, "genes_below_the_line_after_zeroing": risk,
              "corrected_list_equals_frozen": len(risk) == 0}
    print("responsive list:", json.dumps({k: v_ for k, v_ in qcheck.items() if k != "genes_below_the_line_after_zeroing"}),
          f"at risk: {[r['gene'] for r in risk]}", flush=True)

    # 3. rescoring
    stored = A.results("stage3_d1_confirm.json")["cells"]
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "integrity": integ, "map": info,
           "responsive_list_check": qcheck, "cells": {}, "reproduction": {}, "reading": {}}
    fails = []
    for setting in ("as executed", "corrected"):
        cells = A.d1_cells(Z, setting)
        for (rule, depth), c in cells.items():
            key = f"{rule} {depth}"
            raw = exp_scores(c["Y"], c); base = exp_scores(c["w0y"][None, :] * c["Y"], c)
            sc = exp_scores(c["W1"][None, :] * c["Y"], c)
            row = {"raw": float(raw.mean()), "w0y": float(base.mean()), "W1": float(sc.mean()),
                   "own_entries_excluded": int((c["sq"] >= 0).sum())}
            if setting == "as executed":
                res["reproduction"][key] = A.check_stored(f"D1 {key}", row, stored[key], fails)
            b = block_bootstrap(sc, base, c["blocks"]); br = block_bootstrap(sc, raw, c["blocks"])
            row.update({"W1_vs_w0y": summ(b), "W1_vs_raw": summ(br),
                        "w0y_vs_raw": summ(block_bootstrap(base, raw, c["blocks"]))})
            res["cells"].setdefault(setting, {})[key] = row
            print(f"{setting:12s} {key:14s} raw {row['raw']:.4f}  w0y {row['w0y']:.4f}  W1 {row['W1']:.4f}  "
                  f"W1 vs w0y {fmt(b)}  (own entries excluded: {row['own_entries_excluded']})", flush=True)
    if fails:
        sys.exit(f"the as-executed scoring does not reproduce the frozen D1 results: {fails}")
    for setting in ("as executed", "corrected"):
        p = res["cells"][setting]["responsive 20"]["W1_vs_w0y"]
        lvl = ("confirmed, at the finalising level" if p["diff"] >= 0.02 and p["ci95"][0] > 0 else
               "confirmed" if p["diff"] >= 0.01 and p["ci95"][0] > 0 else "not confirmed")
        gates = {k: bool(res["cells"][setting][k]["W1_vs_w0y"]["ci95"][1] >= 0) for k in ("terciles 20", "terciles 50")}
        res["reading"][setting] = {"A5_section_5": lvl, "gatekeepers_pass": gates}
    res["reading"]["same_reading"] = res["reading"]["as executed"] == res["reading"]["corrected"]
    res["reading"]["weights_unchanged"] = "the weights do not depend on the exclusion; only the scoring changes"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "d1_self_gene_correction.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    print(f"A5 reading, as executed: {res['reading']['as executed']['A5_section_5']}; corrected: "
          f"{res['reading']['corrected']['A5_section_5']}; wrote {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
