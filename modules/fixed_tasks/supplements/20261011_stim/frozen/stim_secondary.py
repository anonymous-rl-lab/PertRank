"""Registered S1/S3 computations, with input-only preparation separated.

Role-disjointness is the caller's audited metadata responsibility. These pure
functions never access network/files. S3 indices/designs are input-only;
contributions and regression require a separately released truth stage.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

import stats_frozen_exact_score as exact
from stim_stats import finite_array, paired_block_interval, SEED, DRAWS, K, SCOPE


def _unique_ids(name, ids):
    a = np.asarray(ids).astype(str)
    if a.ndim != 1 or len(a) == 0 or len(np.unique(a)) != len(a):
        raise ValueError(f"{name} must contain nonempty unique identifiers")
    return a


def union_panel_bootstrap(responsive_differences, tercile_differences,
                          responsive_query_ids, tercile_query_ids,
                          union_query_ids, union_blocks):
    """S1 common-union block draw; retain duplicated block and overlapping genes.

    Union clustering is prepared from original planning profiles before truth.
    Each resampled panel mean weights its selected queries equally. Empty-panel
    draws are discarded and counted, with no data-dependent seed or draw cap.
    """
    dr = finite_array("responsive differences", responsive_differences, 1)
    dt = finite_array("tercile differences", tercile_differences, 1)
    qr = _unique_ids("responsive query IDs", responsive_query_ids)
    qt = _unique_ids("tercile query IDs", tercile_query_ids)
    qu = _unique_ids("union query IDs", union_query_ids)
    b = np.asarray(union_blocks)
    if dr.shape != qr.shape or dt.shape != qt.shape or b.shape != qu.shape:
        raise ValueError("panel scores, query IDs and union block axes do not align")
    if b.dtype.kind in "fc" and not np.isfinite(b).all():
        raise ValueError("nonfinite union block labels")
    if set(qu) != (set(qr) | set(qt)):
        raise ValueError("union query axis must be exactly the union of the two panels")
    mapping = {q: i for i, q in enumerate(qu)}
    ir = np.array([mapping[q] for q in qr], dtype=int)
    it = np.array([mapping[q] for q in qt], dtype=int)
    labels = np.unique(b)
    members = {v: np.flatnonzero(b == v) for v in labels}
    rng = np.random.default_rng(SEED)
    samples = np.empty(DRAWS)
    rejected = 0
    accepted = 0
    while accepted < DRAWS:
        selected = np.concatenate([members[v] for v in rng.choice(labels, len(labels), replace=True)])
        multiplicity = np.bincount(selected, minlength=len(qu))
        nr, nt = multiplicity[ir].sum(), multiplicity[it].sum()
        if nr == 0 or nt == 0:
            rejected += 1
            continue
        samples[accepted] = np.dot(multiplicity[ir], dr)/nr - np.dot(multiplicity[it], dt)/nt
        accepted += 1
    interval = np.quantile(samples, [.025, .975]).tolist()
    return {
        "responsive_increment": float(dr.mean()), "tercile_increment": float(dt.mean()),
        "difference_of_increments": float(dr.mean()-dt.mean()),
        "nominal_95_interval": interval,
        "directional_prediction_supported_nominally": bool(interval[0] > 0),
        "union_queries": len(qu), "overlapping_queries": len(set(qr) & set(qt)),
        "union_blocks": len(labels), "draws": DRAWS, "seed": SEED,
        "empty_panel_draws_rejected": rejected, "scope": SCOPE,
        "formal_family": "prespecified secondary; outside H1/H2, cannot rescue H1",
    }


def alignment_indices(planning_half_mean, planning_var_trace,
                      planning_proj_var_trace, mechanism_control_mean,
                      input_control_var_trace, input_control_proj_var_trace, U):
    """S3 input-only indices from float64 means and sample-covariance traces.

    planning_half_mean[T,2,G] is first/second 50-cell mean. Planning traces are
    over all 100 cells, ddof=1; input control traces use 2000 rows, ddof=1.
    Mechanism controls are independent sets of 1000, transformed in float64.
    Arrays with undefined indices retain NaN for audit and are excluded only
    from this regression. Every target remains in all main ranking lists.
    """
    halves = finite_array("planning half means", planning_half_mean, 3)
    controls = finite_array("mechanism control means", mechanism_control_mean, 2)
    u = finite_array("U", U, 2)
    nt, two, ng = halves.shape
    if two != 2 or controls.shape != (2, ng) or u.shape != (ng, 5):
        raise ValueError("expected half means [T,2,G], controls [2,G], U[G,5]")
    if not np.allclose(u.T @ u, np.eye(5), rtol=0, atol=1e-8):
        raise ValueError("U must have orthonormal columns")
    traces = finite_array("planning covariance traces", planning_var_trace, 1)
    ptraces = finite_array("planning projected covariance traces", planning_proj_var_trace, 1)
    ctrace = float(input_control_var_trace)
    cptrace = float(input_control_proj_var_trace)
    if traces.shape != (nt,) or ptraces.shape != (nt,) or not np.isfinite([ctrace,cptrace]).all():
        raise ValueError("covariance trace axes/nonfinite values")
    if np.any(traces < -1e-10) or np.any(ptraces < -1e-10) or min(ctrace,cptrace) < -1e-10:
        raise ValueError("negative covariance trace")
    h = halves - controls[None, :, :]
    d_i = np.einsum("tg,tg->t", h[:,0], h[:,1])/ng
    proj = h @ u
    d_p = np.einsum("tk,tk->t", proj[:,0], proj[:,1])/ng
    vtrace = traces/20 + ctrace/2000
    pvtrace = ptraces/20 + cptrace/2000
    nu = vtrace/ng
    with np.errstate(divide="ignore", invalid="ignore"):
        noise_alignment = pvtrace/vtrace
        a = d_p/d_i-noise_alignment
        r = d_i/nu
    eligible = (d_i > 0) & (nu > 0) & np.isfinite(a) & np.isfinite(r) & np.isfinite(d_p)
    rank_a = np.full(nt, np.nan)
    rank_r = np.full(nt, np.nan)
    n = int(eligible.sum())
    if n:
        rank_a[eligible] = rankdata(a[eligible], method="average")/n
        rank_r[eligible] = rankdata(r[eligible], method="average")/n
    return {
        "d_I": d_i, "d_P": d_p, "nu20": nu, "r": noise_alignment,
        "A": a, "R": r, "eligible": eligible, "rank_A": rank_a, "rank_R": rank_r,
        "eligible_targets": n, "excluded_targets": nt-n,
        "exclusion_nonpositive_dI": int(np.count_nonzero(d_i <= 0)),
        "exclusion_nonpositive_nu20": int(np.count_nonzero(nu <= 0)),
        "exclusion_nonfinite_index": int(np.count_nonzero(~np.isfinite(a) | ~np.isfinite(r))),
    }


def prepare_alignment_design(indices, target_ids, query_ids):
    """Input-only, whole-endpoint feasibility check before any truth access."""
    tids = _unique_ids("target IDs", target_ids)
    qids = _unique_ids("query IDs", query_ids)
    eligible = np.asarray(indices["eligible"], dtype=bool)
    ra = np.asarray(indices["rank_A"], dtype=np.float64)
    rr = np.asarray(indices["rank_R"], dtype=np.float64)
    if eligible.shape != tids.shape or ra.shape != tids.shape or rr.shape != tids.shape:
        raise ValueError("mechanism indices/target axis mismatch")
    if not np.isfinite(ra[eligible]).all() or not np.isfinite(rr[eligible]).all():
        raise ValueError("nonfinite eligible ranks")
    design = np.column_stack([np.ones(len(tids)),ra,rr])
    masks = eligible[None,:] & (qids[:,None] != tids[None,:])
    counts = masks.sum(axis=1)
    ranks = np.array([np.linalg.lstsq(design[m], np.zeros(int(m.sum())), rcond=None)[2]
                      if m.any() else 0 for m in masks], dtype=int)
    invalid = (counts < 26) | (ranks != 3)
    return {
        "status": "not_evaluable" if invalid.any() else "ok",
        "reason": "at_least_one_query_has_fewer_than_26_targets_or_rank_deficient_design" if invalid.any() else None,
        "target_ids": tids, "query_ids": qids, "design": design,
        "query_target_masks": masks, "query_target_counts": counts,
        "query_design_ranks": ranks, "invalid_query_ids": qids[invalid],
        "eligible_targets": int(eligible.sum()),
        "rank_scope": "all mechanism-eligible targets before per-query self exclusion",
    }


def nomination_contributions(effects, truth, pertrank_weights, euclidean_weights,
                            query_ids, target_ids):
    """Truth-stage g[T,Q], using full candidate membership before S3 filtering."""
    y = finite_array("original effects", effects, 3)
    ref = finite_array("truth", truth, 2)
    wp = finite_array("PertRank weights", pertrank_weights, 2)
    we = finite_array("Euclidean weights", euclidean_weights, 2)
    tids = _unique_ids("target IDs", target_ids)
    qids = _unique_ids("query IDs", query_ids)
    if y.shape != (5,len(tids),len(qids)) or ref.shape != y.shape[1:] or wp.shape != y.shape[:2] or we.shape != wp.shape:
        raise ValueError("contribution effect/weight axes mismatch")
    if np.any((wp < 0)|(wp > 1)) or np.any((we < 0)|(we > 1)):
        raise ValueError("invalid scalar weights")
    own = exact.own_mask(qids,tids)
    p_truth = exact.probabilities(ref.T,own,K)
    contributions = np.zeros_like(ref)
    for group in range(5):
        p_pr = exact.probabilities((wp[group,:,None]*y[group]).T,own,K)
        p_e = exact.probabilities((we[group,:,None]*y[group]).T,own,K)
        contributions += ((p_pr-p_e)*p_truth).reshape(2,len(qids),len(tids)).sum(axis=0).T/(5*2*K)
    return contributions


def alignment_regression(contributions, frozen_design, blocks, *, target_ids, query_ids):
    """Truth-stage OLS, all queries or none, using the frozen pretruth designs."""
    tids = _unique_ids("target IDs",target_ids)
    qids = _unique_ids("query IDs",query_ids)
    if not np.array_equal(tids,frozen_design["target_ids"]) or not np.array_equal(qids,frozen_design["query_ids"]):
        raise ValueError("design/contribution identifiers mismatch")
    if frozen_design["status"] != "ok":
        return {"status":"not_evaluable", "reason":frozen_design["reason"],
                "invalid_query_ids":np.asarray(frozen_design["invalid_query_ids"]).tolist()}
    g = finite_array("nomination contributions",contributions,2)
    if g.shape != (len(tids),len(qids)) or np.asarray(blocks).shape != qids.shape:
        raise ValueError("contribution or block axes mismatch")
    design = np.asarray(frozen_design["design"],dtype=np.float64)
    masks = np.asarray(frozen_design["query_target_masks"],dtype=bool)
    coefs = np.empty((len(qids),3))
    for q,m in enumerate(masks):
        coef,_,rank,_ = np.linalg.lstsq(design[m],g[m,q],rcond=None)
        if m.sum() < 26 or rank != 3 or not np.isfinite(coef).all():
            raise ValueError("frozen pretruth design changed or nonfinite OLS result")
        coefs[q] = coef
    interval = paired_block_interval(coefs[:,1],blocks)
    return {
        "status":"ok", "mean_rank_A_coefficient":float(coefs[:,1].mean()),
        "nominal_95_interval":interval["conditional_query_block_95_interval"],
        "negative_prediction_supported_nominally":bool(interval["conditional_query_block_95_interval"][1] < 0),
        "coefficients":coefs, "coefficient_order":["intercept","rank_A","rank_R"],
        "query_ids":qids, "seed":SEED, "draws":DRAWS,
        "scope":"conditional association; index-estimation, donor and control uncertainty not included",
        "formal_family":"prespecified secondary; outside H1/H2, cannot rescue H1",
    }
