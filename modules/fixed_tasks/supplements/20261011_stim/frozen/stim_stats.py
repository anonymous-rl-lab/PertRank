"""Registered Stim scoring and decision layer; pure local numerical functions.

No data access lives here. Truth-bearing functions may be called only by the
evaluation stage after the second registration release. Seed/constants and
tie expectations follow the registered attachment, not historical defaults.
"""
from __future__ import annotations

import numpy as np

import stats_frozen_exact_score as exact
import stats_frozen_summary as historical
from stats_frozen_wild_cluster_t import wild_cluster_t

SEED = 2026
DRAWS = 2000
K = 25
ARMS = ("raw", "Euclidean", "TRADE_scalar", "PR_M", "PertRank")
SCOPE = ("conditional on realized input cell groups, fixed controls, query panel, "
         "candidate pool and reference; not population or cell-sampling uncertainty")


def finite_array(name, value, ndim=None):
    a = np.asarray(value, dtype=np.float64)
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains nonfinite values; no entries are dropped")
    return a


def block_draws(differences, blocks):
    """Same ordered concatenated-query resampling as frozen exact_score.py."""
    d = finite_array("differences", differences, 1)
    b = np.asarray(blocks)
    if b.ndim != 1 or b.shape != d.shape or d.size == 0:
        raise ValueError("nonempty query differences and block labels must align")
    if b.dtype.kind in "fc" and not np.isfinite(b).all():
        raise ValueError("nonfinite query block labels")
    labels = np.unique(b)
    members = {v: np.flatnonzero(b == v) for v in labels}
    rng = np.random.default_rng(SEED)
    return np.array([
        d[np.concatenate([members[v] for v in rng.choice(labels, len(labels), replace=True)])].mean()
        for _ in range(DRAWS)
    ])


def paired_block_interval(differences, blocks):
    d = finite_array("differences", differences, 1)
    draws = block_draws(d, blocks)
    return {
        "difference": float(d.mean()),
        "conditional_query_block_95_interval": np.quantile(draws, [.025, .975]).tolist(),
        "conditional_query_block_97_5_interval": np.quantile(draws, [.0125, .9875]).tolist(),
        "bootstrap_nonpositive_count": int(np.count_nonzero(draws <= 0)),
        "bootstrap_fraction_nonpositive": float(np.mean(draws <= 0)),
        "draws": DRAWS, "seed": SEED, "query_blocks": int(len(np.unique(blocks))),
        "scope": SCOPE,
    }


def score_predictions(effects, truth, weights, query_ids, target_ids):
    """Return (scores[A,5,Q,2], arm_names) from original log effects[5,T,Q]."""
    y = finite_array("effects", effects, 3)
    ref = finite_array("truth", truth, 2)
    target_ids = np.asarray(target_ids).astype(str)
    query_ids = np.asarray(query_ids).astype(str)
    if y.shape != (5, len(target_ids), len(query_ids)) or ref.shape != y.shape[1:]:
        raise ValueError("effect/truth axes must be five groups, target, query")
    own = exact.own_mask(query_ids, target_ids)
    names = tuple(name for name in ARMS if name in weights)
    if set(weights) != set(names) or not names:
        raise ValueError("unexpected/empty method arm; missing arms must be declared separately")
    scores = np.empty((len(names), 5, len(query_ids), 2), dtype=np.float64)
    for ai, name in enumerate(names):
        w = finite_array(f"{name} weights", weights[name], 2)
        if w.shape != y.shape[:2] or np.any((w < 0) | (w > 1)):
            raise ValueError(f"invalid {name} scalar weights")
        if name == "raw" and not np.array_equal(w, np.ones_like(w)):
            raise ValueError("raw weights must all equal one")
        for group in range(5):
            scores[ai, group] = exact.score((w[group, :, None] * y[group]).T, ref.T, own, K)
    return scores, names


def summarize_scores(scores, arm_names, blocks):
    """All arm means and available pairwise contrasts, preserving D4 summaries.

    ``statistical_direction`` in the historical structure is descriptive only;
    formal H2 conclusions are exclusively returned by ``adjudicate_main``.
    """
    scores = finite_array("scores", scores, 4)
    names = list(arm_names)
    if scores.shape != (len(names), 5, len(blocks), 2):
        raise ValueError("scores must have arm, five groups, query, two signs axes")
    if len(names) != len(set(names)) or not set(names).issubset(ARMS):
        raise ValueError("invalid method names")
    if np.any((scores < -1e-14) | (scores > 1+1e-14)):
        raise ValueError("P@25 outside [0,1]")
    out = historical.summarize(scores, names, blocks, exact)
    out["arm_names"] = names
    out["interpretation"] = "All percentile-interval direction and equivalence labels are descriptive."
    ave = scores.mean(1)
    for name, i, j in historical.all_contrasts(names):
        interval = paired_block_interval((ave[i] - ave[j]).mean(1), blocks)
        row = out["repetition_average"][name]
        if row["conditional_query_block_95_interval"] != interval["conditional_query_block_95_interval"]:
            raise AssertionError("new/frozen block bootstrap mismatch")
        row["conditional_query_block_97_5_interval"] = interval["conditional_query_block_97_5_interval"]
    return out


def adjudicate_main(scores, arm_names, blocks, pertrank_weights, unavailable_arms=None):
    """Preserve D4 H1; use registered restricted wild t for gated H2a/H2b.

    Missing competitor arms are not victories and never silently imputed. The
    same WCR seed and responsive labels recreate the identical multiplier matrix
    for every contrast. Its other tails are descriptive, not formal claims.
    """
    names = list(arm_names)
    scores = finite_array("scores", scores, 4)
    if not {"PertRank", "Euclidean"}.issubset(names):
        raise ValueError("H1 cannot be adjudicated without PertRank and Euclidean")
    summary = summarize_scores(scores, names, blocks)
    w = finite_array("PertRank weights", pertrank_weights, 2)
    if w.shape[0] != 5 or w.shape[1] < 26 or np.any((w < 0) | (w > 1)):
        raise ValueError("expected valid PertRank weights [5,T>=26]")
    zero_groups = np.flatnonzero(np.all(w == 0, axis=1)).tolist()
    main = summary["repetition_average"]["PertRank_minus_Euclidean"]
    lo = main["conditional_query_block_95_interval"][0]
    numerical = bool(main["difference"] >= .01 and lo > 0)
    confirm = bool(numerical and not zero_groups)
    mean_groups = scores.mean(1)
    pr_idx = names.index("PertRank")
    wild = {}
    for other in names:
        if other != "PertRank":
            wild["PertRank_minus_" + other] = wild_cluster_t(
                (mean_groups[pr_idx] - mean_groups[names.index(other)]).mean(1), blocks)
    h1_wild = wild["PertRank_minus_Euclidean"]
    sensitive = bool(confirm and (h1_wild["status"] != "ok" or h1_wild["p_upper"] > .025))
    unavailable_arms = {} if unavailable_arms is None else dict(unavailable_arms)
    competitive = {}
    for other in ("PR_M", "TRADE_scalar"):
        if other not in names:
            competitive[other] = {
                "status": "not_evaluable", "formal_superiority": False,
                "reason": unavailable_arms.get(other, "arm_missing_without_supplied_reason"),
            }
            continue
        test = wild["PertRank_minus_" + other]
        if not confirm:
            status = "not_activated"
        elif test["status"] != "ok":
            status = "not_evaluable"
        elif test["estimate"] > 0 and test["p_upper"] <= .025:
            status = "superiority_established"
        else:
            status = "superiority_not_established"
        competitive[other] = {
            "status": status, "formal_superiority": status == "superiority_established",
            "positive_estimate": bool(test["estimate"] > 0),
            "p_upper": test["p_upper"], "one_sided_threshold": .025,
            "percentile_intervals_role": "descriptive_only",
        }
    return {
        "summary": summary, "wild_tests": wild,
        "decision": {
            "confirmation": confirm,
            "improvement_grade": bool(not zero_groups and main["difference"] >= .02 and lo > 0),
            "numerical_primary_threshold_pass": numerical,
            "method_valid_for_confirmation": not zero_groups,
            "allzero_pertrank_groups": zero_groups,
            "calibration_sensitive": sensitive,
            "competitive_gate_open": confirm, "competitors": competitive,
            "confirmation_rule": "no allzero PertRank group AND point >=0.01 AND nominal lower95 >0",
            "competition_rule": "H1 gate AND positive point AND null-imposed wild cluster t p_upper<=0.025",
            "closed_gate_rule": "No secondary or descriptive result rescues H1; non-superiority is not inferiority/equivalence.",
        },
    }
