"""Stage 2 evaluation (amendment A2 section 4.5, amendment A3), run on the scalars produced by
stage2/workspace/stage2_scalars.py. Frozen, with the workspace script and the V2 mask, before any Stage 2 output
existed (stage2/STAGE2_FREEZE_SHA256.txt).

Checks, each fatal: the plan, A2 and A3 are unchanged; this file, the scalars' producing script and the V2 mask
match the freeze record; the scalars carry the same plan and A2 hashes; every source's targets are the reduced
state's targets in the same order; V0 reproduces the reduced state's m2 and noise term (relative 1e-9).

Weights, all at the same number of cells as the prediction they multiply:
  w0y           the v10 baseline, from the reduced state
  V1_x1         A2 4.2 primary metric;  V1_x0.25, V1_x4 sensitivity (reported, not eligible)
  V2            E4 only (A2 4.2); identical to w0y elsewhere
  L1            A2 4.3: grid-mixture prior fitted across the source's targets on (s_hat, tau2) of V0, posterior mean
                s*, weight s* / (s* + nu)
  L1kappa       A3 diagnostic: tau2 multiplied by kappa = mean (s_20 - s_30)^2 / mean (tau2_20 + tau2_30); not
                eligible (kappa uses the 30-cell block, outside the 20-cell budget); not computed for Orion
Scoring as in v10: tie-expectation precision@25, 30-block paired bootstrap (B = 2000, seed 2026), keys 101 / 202.

Decisions (A2 4.5), per eligible candidate (V1_x1, V2, L1), all differences against w0y:
  primary  E4 T cell responsive queries, 20 cells: >= +0.01 and 95% interval above zero -> continue;
           >= +0.02 and interval above zero -> worth finalising
  gatekeepers  E1 Replogle 20 and 50; E4 terciles 20 and 50; E5 KOLF matched responsive 20 and 50; E3 Orion 20
           and 100 (Orion also reported separately): the 95% interval must not lie entirely below zero
  a candidate continues only if the primary cell says so and every gatekeeper passes; if none continues the
  outcome is "no effective same-budget improvement found in this round".

Order of computation (stage2/README.md, section 4): the scalars file may hold a subset of the sources, provided it
holds E4 (the primary cell). Gatekeepers of absent sources are reported as not computed. The rule above is
unchanged: if no eligible candidate passes the primary cell the outcome is final without them; otherwise it is
reported as pending until they are computed.

    python3 stage2/evaluate_stage2.py --scalars stage2_scalars.json [--out DIR]
    python3 stage2/evaluate_stage2.py --dry-run [--out DIR]     (code-path test on placeholder scalars built from
                                                                 the reduced states; results are meaningless)
A scalars file carrying "placeholder": true (built by stage2/tests/test_evaluate_paths.py) runs every check of a
real evaluation and is labelled meaningless in the output in the same way as a dry run.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BUNDLE, "scripts"))
from _paths import (AMEND_A3, AMEND_A3_RECORD, DATA, OWN_MAP, amendment_sha, checked_sha, import_pr10,  # noqa: E402
                    plan_sha, sha256)

import_pr10()
from pr10 import block_bootstrap, nu, score_rows, self_to_query, w0y, w_signal  # noqa: E402

FREEZE = os.path.join(HERE, "STAGE2_FREEZE_SHA256.txt")
FROZEN = {"workspace/stage2_scalars.py", "evaluate_stage2.py", "v2_mask/v2_mask_e4.npz"}
RTOL = 1e-9
ELIGIBLE = ("V1_x1", "V2", "L1")
REPORTED = ("V1_x1", "V1_x0.25", "V1_x4", "V2", "L1", "L1kappa")
PRIMARY = ("E4", "responsive", "20")
GATEKEEPERS = (("E1", "reference-variance", "20"), ("E1", "reference-variance", "50"),
               ("E4", "terciles", "20"), ("E4", "terciles", "50"),
               ("E5", "responsive", "20"), ("E5", "responsive", "50"),
               ("E3", "terciles", "20"), ("E3", "terciles", "100"))


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def keys(n):
    return np.random.default_rng(101).random(n), np.random.default_rng(202).random(n)


def summ(bb):
    return {"diff": bb["diff"], "ci95": bb["ci95"], "p_one_sided": bb["p_one_sided"]}


# ------------------------------------------------------------------ integrity
def freeze_record():
    rec = {}
    for line in open(FREEZE, encoding="utf-8"):
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            rec[parts[1]] = parts[0]
    return rec


def integrity(S, dry):
    out = {"plan_sha256": plan_sha(), "amendment_A2_sha256": amendment_sha(),
           "amendment_A3_sha256": checked_sha(AMEND_A3, AMEND_A3_RECORD, "amendment A3")}
    if dry:
        out["freeze"] = "skipped (dry run)"
        return out
    rec = freeze_record()
    for rel in FROZEN:
        if rec.get(rel) != sha256(os.path.join(HERE, rel)):
            sys.exit(f"{rel} differs from the freeze record; the frozen evaluation cannot run on an edited file")
    for key, want in (("script_sha256", rec["workspace/stage2_scalars.py"]), ("v2_mask_sha256", rec["v2_mask/v2_mask_e4.npz"]),
                      ("plan_sha256", out["plan_sha256"]), ("amendment_A2_sha256", out["amendment_A2_sha256"])):
        if S.get(key) != want:
            sys.exit(f"scalars file: {key} {S.get(key)} != {want}")
    out["freeze"] = {k: rec[k] for k in sorted(FROZEN)}
    return out


# ------------------------------------------------------------------ reduced states, rows and baselines
def load(name):
    return np.load(os.path.join(DATA, name), allow_pickle=True)


def sources():
    """per source: targets, the reduced (m2, nu) per block for the V0 check, and the scoring cells"""
    OWN = np.load(OWN_MAP, allow_pickle=True)["own_row"]
    out = {}

    D = load("e1_stage6.npz"); cand = D["cand"]; cvm, nc = float(D["cvm"]), int(D["ncin"])
    KPa, KTa = keys(len(D["targets"])); KP, KT = KPa[cand], KTa[cand]
    sq = self_to_query(D["self_row"][cand], D["qrows"]); cin = D["cin_q"].astype(np.float64)[:, None]
    base = {b: w0y(nu(D[f"vb{b}"], int(b), cvm, nc), D[f"m2_{b}"]) for b in ("20", "50")}
    Y = {"20": D["om20_q"].astype(np.float64) - cin, "50": D["om50_q"].astype(np.float64) - cin}
    out["E1"] = {"label": "E1 Replogle K562", "targets": [str(t) for t in D["targets"][cand]],
                 "v0": {b: (D[f"m2_{b}"], D[f"vb{b}"] / int(b) + cvm / nc) for b in ("20", "30", "50")},
                 "cells": [dict(rule="reference-variance", depth=b, role="gatekeeper", Y=Y[b], T=D["Tq"], sq=sq, blocks=D["blocks"],
                                KP=KP, KT=KT, w0y=base[b]) for b in ("20", "50")]}

    D = load("e3_orion.npz"); g = lambda k: D[f"fz_{k}"]; KP, KT = keys(len(D["cohort"])); cvm = float(g("cvm"))
    sq = self_to_query(g("self_row"), g("qrows")); cin = g("cin_q")[:, None]
    T = g("ev_s_q") / 100.0 - g("ecm_q")[:, None]
    base = {"20": w0y(nu(g("vb20"), 20, cvm), g("m2_20")), "100": w0y(nu(g("vb20"), 100, cvm), g("m2_100"))}
    Y = {"20": g("sh_s_q") / 20.0 - cin, "100": g("in_s_q") / 100.0 - cin}
    out["E3"] = {"label": "E3 Orion HCT116 (frozen scope)", "targets": [str(t) for t in D["cohort"]],
                 "v0": {"20": (g("m2_20"), g("vb20") / 20 + cvm / 2000), "100": (g("m2_100"), g("vb20") / 100 + cvm / 2000)},
                 "cells": [dict(rule="terciles", depth=b, role="gatekeeper", Y=Y[b], T=T, sq=sq, blocks=g("blocks"),
                                KP=KP, KT=KT, w0y=base[b]) for b in ("20", "100")]}

    D = load("e4_tcell.npz"); KP, KT = keys(len(D["names"])); cvm = float(D["cvm"])
    base = {b: w0y(nu(D[f"vb_{b}"], int(b), cvm), D[f"m2_{b}"]) for b in ("20", "50", "100")}
    cells = []
    for rule, q in (("terciles", D["q_terc"]), ("responsive", D["q_resp"])):
        pos = np.searchsorted(D["qu"], q); S = [D[f"S{b}_q"][pos] for b in range(5)]; cm = D["cm_q"][pos][:, None]
        T = S[4] / 100.0 - D["ce_q"][pos][:, None]; sq = self_to_query(OWN, q)
        Y = {"20": S[0] / 20.0 - cm, "50": (S[0] + S[1]) / 50.0 - cm, "100": (S[0] + S[1] + S[2]) / 100.0 - cm}
        for b in ("20", "50", "100"):
            role = ("primary" if (rule, b) == ("responsive", "20") else
                    "gatekeeper" if rule == "terciles" and b in ("20", "50") else "descriptive")
            cells.append(dict(rule=rule, depth=b, role=role, Y=Y[b], T=T, sq=sq, blocks=D[f"blocks_{rule}"], KP=KP, KT=KT, w0y=base[b]))
    out["E4"] = {"label": "E4 CD4+ T cell probe", "targets": [str(t) for t in D["names"]],
                 "v0": {b: (D[f"m2_{b}"], D[f"vb_{b}"] / n + cvm / 2000) for b, n in (("20", 20), ("30", 30), ("50", 50), ("100", 100), ("A", 100))},
                 "cells": cells}

    D = load("e5_kolf.npz"); KP, KT = keys(len(D["names"])); tb = D["target_batch"]; cmm = D["cvm_mfull"]
    base = {b: w0y(D[f"vb_full_{b}"] / int(b) + cmm[tb] / 2000.0, D[f"m2_mfull_{b}"]) for b in ("20", "50", "100")}
    cells = []
    for rule, q in (("terciles", D["q_terc"]), ("responsive", D["q_resp"])):
        pos = np.searchsorted(D["qu"], q); S = [D[f"S{b}_q"][pos] for b in range(5)]
        cmB = D["cmB_q"][:, pos][tb].T; ceB = D["ceB_q"][:, pos][tb].T
        T = S[4] / 100.0 - ceB; sq = self_to_query(D["self_row"], q)
        Y = {"20": S[0] / 20.0 - cmB, "50": (S[0] + S[1]) / 50.0 - cmB, "100": (S[0] + S[1] + S[2]) / 100.0 - cmB}
        for b in ("20", "50", "100"):
            role = "gatekeeper" if rule == "responsive" and b in ("20", "50") else "descriptive"
            cells.append(dict(rule=rule, depth=b, role=role, Y=Y[b], T=T, sq=sq, blocks=D[f"blocks_{rule}"], KP=KP, KT=KT, w0y=base[b]))
    out["E5"] = {"label": "E5 KOLF2.1J probe, batch-matched controls", "targets": [str(t) for t in D["names"]],
                 "v0": {b: (D[f"m2_mfull_{b}"], D[f"vb_full_{b}"] / int(b) + cmm[tb] / 2000.0) for b in ("20", "30", "50", "100")},
                 "cells": cells}
    return out


# ------------------------------------------------------------------ L1
def npmle_posterior_mean(s_hat, tau2, K=300, max_iter=2000, gap_tol=1e-9):
    """grid mixture (NPMLE) over s >= 0 with likelihood N(s_hat | s, tau2); returns the posterior mean.
    The mixing weights maximise sum_j log (L pi)_j on the simplex, a convex problem, solved by the constrained
    Newton method of Wang (2007): each step minimises || (L x) / (L pi) - 2 ||^2 over x >= 0, sum x = 1 (the
    second-order expansion of the log-likelihood) by nonnegative least squares on the current support plus the
    grid point of largest directional derivative, followed by a backtracking line search that never lowers the
    likelihood. Stopping rule: the NPMLE optimality certificate max_k (1/n) sum_j L_jk / (L pi)_j - 1 <= gap_tol."""
    tau = np.sqrt(tau2)
    lo = 1e-3 * float(np.median(tau)); hi = max(float(np.max(s_hat)), 0.0) + 4.0 * float(np.max(tau))
    u = np.concatenate([[0.0], np.geomspace(lo, hi, K - 1)])
    logL = -0.5 * ((s_hat[:, None] - u[None, :]) / tau[:, None]) ** 2 - np.log(tau)[:, None]
    L = np.exp(logL - logL.max(1, keepdims=True))
    n = len(s_hat)
    pi = np.full(K, 1.0 / K); f = L @ pi; ll = float(np.log(f).sum())
    it = 0; g = float(np.max(L.T @ (1.0 / f)) / n - 1.0)
    while g > gap_tol and it < max_iter:
        it += 1
        d = L.T @ (1.0 / f) / n
        S = np.union1d(np.flatnonzero(pi > 0), [int(np.argmax(d))])
        A = L[:, S] / f[:, None]
        wrow = np.sqrt(n) * 10.0
        x, _ = nnls(np.vstack([A, wrow * np.ones(len(S))]), np.concatenate([2.0 * np.ones(n), [wrow]]), maxiter=50 * len(S))
        if x.sum() <= 0:
            break
        new = np.zeros(K); new[S] = x / x.sum()
        step = 1.0
        while True:
            cand = (1.0 - step) * pi + step * new; fc = L @ cand
            with np.errstate(divide="ignore"):
                lc = float(np.log(fc).sum())      # -inf if a target loses all likelihood: rejected below
            if lc >= ll or step < 1e-12:
                break
            step *= 0.5
        if lc < ll:
            break
        pi, f, ll = cand, fc, lc          # no pruning: NNLS already returns exact zeros off the support
        g = float(np.max(L.T @ (1.0 / f)) / n - 1.0)
    post = (L * pi) @ u / (L @ pi)
    return post, {"iterations": it, "optimality_gap": g, "converged": bool(g <= gap_tol),
                  "support_points_with_mass": int((pi > 0).sum()), "grid": [lo, hi]}


def weights_for(src, rec, block, kappa):
    B = rec["blocks"][block]; W = {}
    for k in ("V1_x1", "V1_x0.25", "V1_x4", "V2"):
        if k in B:
            W[k] = np.clip(1.0 - np.array(B[k]["nu"]) / np.array(B[k]["a"]), 0.0, 1.0)
    a, v, t2 = (np.array(B["V0"][q]) for q in ("a", "nu", "tau2"))
    s_hat = a - v; info = {}
    post, info["L1"] = npmle_posterior_mean(s_hat, t2); W["L1"] = post / (post + v)
    if kappa is not None:
        post, info["L1kappa"] = npmle_posterior_mean(s_hat, kappa * t2); W["L1kappa"] = post / (post + v)
    return W, info


def kappa_for(rec):
    b = rec["blocks"]
    if "30" not in b:
        return None
    s20 = np.array(b["20"]["V0"]["a"]) - np.array(b["20"]["V0"]["nu"]); s30 = np.array(b["30"]["V0"]["a"]) - np.array(b["30"]["V0"]["nu"])
    return float(np.mean((s20 - s30) ** 2) / np.mean(np.array(b["20"]["V0"]["tau2"]) + np.array(b["30"]["V0"]["tau2"])))


# ------------------------------------------------------------------ main
def main():
    t0 = time.time(); dry = "--dry-run" in sys.argv
    out_dir = arg("--out", os.path.join(HERE, "results_dry_run" if dry else "results"))   # created only when writing
    SRC = sources()
    if not dry and arg("--scalars") is None:
        sys.exit("give --scalars <stage2_scalars.json> (or --dry-run)")
    S = placeholder_scalars(SRC) if dry else json.load(open(arg("--scalars"), encoding="utf-8"))
    unknown = sorted(set(S["sources"]) - set(SRC))
    if unknown:
        sys.exit(f"unknown sources in the scalars file: {unknown}")
    present = [s for s in SRC if s in S["sources"]]
    if PRIMARY[0] not in present:
        sys.exit(f"the scalars file must hold {PRIMARY[0]}, the source of the primary cell")
    meaningless = dry or bool(S.get("placeholder"))
    res = {"dry_run": dry, "placeholder_scalars": meaningless, "integrity": integrity(S, dry),
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sources_present": present,
           "sources": {}, "decisions": {}}
    if meaningless:
        print("=" * 72 + "\nPLACEHOLDER SCALARS: exercises the code; every number below is meaningless\n" + "=" * 72)
    for src in present:
        info = SRC[src]; rec = S["sources"][src]
        if rec["targets"] != info["targets"]:
            sys.exit(f"{src}: targets differ from the reduced state")
        chk = {}
        for b, (m2, nuv) in info["v0"].items():
            a = np.array(rec["blocks"][b]["V0"]["a"]); v = np.array(rec["blocks"][b]["V0"]["nu"])
            chk[b] = [float(np.max(np.abs(a - m2) / np.abs(m2))), float(np.max(np.abs(v - nuv) / np.abs(nuv)))]
            if max(chk[b]) >= RTOL:
                sys.exit(f"{src} block {b}: V0 does not reproduce the reduced state ({chk[b]})")
        kappa = None if src == "E3" else kappa_for(rec)
        W = {}; L1info = {}
        for b in sorted({c["depth"] for c in info["cells"]}):
            W[b], L1info[b] = weights_for(src, rec, b, kappa)
        cells = []
        for c in info["cells"]:
            sr = lambda E: score_rows(E, c["T"], c["sq"], key_pred=c["KP"], key_truth=c["KT"])["exp"]
            raw = sr(c["Y"]); base = sr(c["w0y"][None, :] * c["Y"])
            row = {"rule": c["rule"], "depth": c["depth"], "role": c["role"], "raw": float(raw.mean()), "w0y": float(base.mean()),
                   "w0y_vs_raw": summ(block_bootstrap(base, raw, c["blocks"])), "candidates": {}}
            for k in REPORTED:
                if k == "V2" and src != "E4":
                    row["candidates"][k] = {"identical_to_w0y": True, "vs_w0y": {"diff": 0.0, "ci95": [0.0, 0.0], "p_one_sided": 1.0}}
                    continue
                if k not in W[c["depth"]]:
                    continue
                sc = sr(W[c["depth"]][k][None, :] * c["Y"])
                row["candidates"][k] = {"score": float(sc.mean()), "vs_w0y": summ(block_bootstrap(sc, base, c["blocks"])),
                                        "vs_raw": summ(block_bootstrap(sc, raw, c["blocks"]))}
            cells.append(row)
        diag = diagnostics(src, rec, W, kappa)
        res["sources"][src] = {"label": info["label"], "v0_check_max_rel_diff": chk, "kappa_A3": kappa, "L1_fit": L1info,
                               "cells": cells, "diagnostics": diag}
        print(f"{src} done ({time.time() - t0:.0f}s)", flush=True)
    res["decisions"] = decide(res["sources"])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stage2_results.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    report(res)
    print(f"wrote {path}")


def diagnostics(src, rec, W, kappa):
    d = {}
    b = rec["blocks"]
    if "A" in b:   # agreement of each 20-cell weight with the weight from 100 independent deep cells (diagnostic)
        v20 = np.array(b["20"]["V0"]["nu"]); sA = np.maximum(np.array(b["A"]["V0"]["a"]) - np.array(b["A"]["V0"]["nu"]), 0.0)
        wA = w_signal(sA, v20)
        base = np.clip(1.0 - v20 / np.array(b["20"]["V0"]["a"]), 0.0, 1.0)
        d["pearson_with_deepA_weight_20cells"] = {"w0y": float(np.corrcoef(base, wA)[0, 1]),
                                                  **{k: float(np.corrcoef(w, wA)[0, 1]) for k, w in W["20"].items()}}
        d["spearman_energy_20_vs_deepA_same_metric"] = {}
        for k in [x for x in ("V0", "V1_x1", "V2") if x in b["20"]]:
            s20 = np.array(b["20"][k]["a"]) - np.array(b["20"][k]["nu"]); sa = np.array(b["A"][k]["a"]) - np.array(b["A"][k]["nu"])
            d["spearman_energy_20_vs_deepA_same_metric"][k] = float(spearmanr(s20, sa).correlation)
    if kappa is not None:
        d["tau2_underestimation_kappa"] = kappa
    return d


def decide(sources_res):
    cells = {(s, c["rule"], c["depth"]): c for s, r in sources_res.items() for c in r["cells"]}
    found = {k for k, c in cells.items() if c["role"] == "gatekeeper"}
    assert found <= set(GATEKEEPERS) and cells[PRIMARY]["role"] == "primary", "cell roles differ from A2 4.5"
    missing = [g for g in GATEKEEPERS if g not in cells]
    prim = cells[PRIMARY]
    out = {}
    for k in ELIGIBLE:
        p = prim["candidates"][k]["vs_w0y"]
        lo = p["ci95"][0]
        primary = ("worth finalising" if p["diff"] >= 0.02 and lo > 0 else "continue" if p["diff"] >= 0.01 and lo > 0 else "does not continue")
        g = {f"{s} {r} {d}": {"diff": cells[(s, r, d)]["candidates"][k]["vs_w0y"]["diff"],
                              "ci95": cells[(s, r, d)]["candidates"][k]["vs_w0y"]["ci95"],
                              "passes": cells[(s, r, d)]["candidates"][k]["vs_w0y"]["ci95"][1] >= 0}
             for (s, r, d) in GATEKEEPERS if (s, r, d) in cells}
        computed_pass = all(v["passes"] for v in g.values())
        if primary == "does not continue":
            status = "does not continue (primary cell)"
        elif missing:
            status = "pending: gatekeepers not computed"
        else:
            status = "continues" if computed_pass else "does not continue (gatekeeper)"
        out[k] = {"primary": {"diff": p["diff"], "ci95": p["ci95"], "reading": primary}, "gatekeepers": g,
                  "gatekeepers_not_computed": [" ".join(x) for x in missing],
                  "all_gatekeepers_pass": computed_pass if not missing else None,
                  "computed_gatekeepers_pass": computed_pass, "status": status, "continues": status == "continues"}
    cont = [k for k in ELIGIBLE if out[k]["continues"]]
    pend = [k for k in ELIGIBLE if out[k]["status"].startswith("pending")]
    if cont:
        out["outcome"] = "candidates that continue: " + ", ".join(cont)
    elif pend:
        absent = sorted({s for s, _, _ in missing})
        out["outcome"] = (f"pending: the primary cell passes for {', '.join(pend)}; the gatekeepers of {', '.join(absent)} "
                          "must be computed before a decision")
    else:
        out["outcome"] = "no effective same-budget improvement found in this round"
    out["final"] = not pend
    out["note"] = ("three eligible candidates are reported in full; each is judged only on its own pre-declared line; "
                   "V1 sensitivity and L1kappa are reported and not eligible")
    return out


def report(res):
    for src, r in res["sources"].items():
        print(f"\n{r['label']}   kappa(A3) = {r['kappa_A3']}")
        for c in r["cells"]:
            parts = [f"{k} {v['vs_w0y']['diff']:+.4f} [{v['vs_w0y']['ci95'][0]:+.4f},{v['vs_w0y']['ci95'][1]:+.4f}]"
                     for k, v in c["candidates"].items() if not v.get("identical_to_w0y")]
            print(f"  {c['role'][:4]} {c['rule'][:10]:10s} {c['depth']:>3s}  w0y-raw {c['w0y_vs_raw']['diff']:+.4f} | " + " | ".join(parts))
    print("\nDecisions (A2 4.5), against w0y:")
    for k in ELIGIBLE:
        d = res["decisions"][k]
        failed = [g for g, v in d["gatekeepers"].items() if not v["passes"]]
        nc = f"; not computed: {len(d['gatekeepers_not_computed'])}" if d["gatekeepers_not_computed"] else ""
        print(f"  {k:6s} primary {d['primary']['diff']:+.4f} [{d['primary']['ci95'][0]:+.4f},{d['primary']['ci95'][1]:+.4f}] "
              f"-> {d['primary']['reading']}; gatekeepers computed {len(d['gatekeepers'])}, failing {failed or 'none'}{nc}; "
              f"{d['status']}")
    print(f"  outcome: {res['decisions']['outcome']}")


def placeholder_scalars(SRC):
    """dry run only: V0 from the reduced states; every other variant copies V0; tau2 is an arbitrary proxy"""
    S = {"sources": {}}
    for src, info in SRC.items():
        blocks = {}
        for b, (m2, nuv) in info["v0"].items():
            s = np.maximum(m2 - nuv, 0.0); t2 = (4.0 * s * nuv + 2.0 * nuv ** 2) / 5000.0
            v0 = {"a": m2.tolist(), "nu": nuv.tolist(), "tau2": t2.tolist()}
            blocks[b] = {"V0": v0, "V1_x1": v0, "V1_x0.25": v0, "V1_x4": v0, **({"V2": v0} if src == "E4" else {})}
        S["sources"][src] = {"targets": info["targets"], "blocks": blocks}
    return S


if __name__ == "__main__":
    main()
