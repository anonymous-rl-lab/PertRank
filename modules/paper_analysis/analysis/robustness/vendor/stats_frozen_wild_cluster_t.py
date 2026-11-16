"""Prespecified restricted wild query-cluster bootstrap for a paired mean.

This is the intercept-only WCR-C bootstrap with CR1 studentization. It is an
approximate cluster-inference procedure, not a randomization of method labels
and not a guarantee of finite-sample coverage. Queries within a cluster share
one Rademacher multiplier. Unequal cluster sizes retain equal-query weighting.

Scientific defaults: null mean 0; 9,999 draws; NumPy default_rng(20261011).
No biological data, data access, or study-specific file paths are included.
"""

from __future__ import annotations

import numpy as np


def _studentized(mean: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Extended-real t for bootstrap draws; a zero/zero draw has t=0."""
    out = np.zeros_like(mean, dtype=np.float64)
    np.divide(mean, se, out=out, where=se > 0)
    zero_se = se == 0
    out[zero_se & (mean > 0)] = np.inf
    out[zero_se & (mean < 0)] = -np.inf
    return out


def wild_cluster_t(
    differences,
    blocks,
    *,
    draws: int = 9999,
    seed: int = 20261011,
    null_mean: float = 0.0,
    return_distribution: bool = False,
) -> dict:
    """Test an equal-query paired mean against a prespecified null mean.

    ``differences`` must already average input groups and signs within query.
    ``blocks`` contains one fixed, planning-derived cluster label per query.
    Inputs are not filtered or imputed. Formal superiority uses the upper-tail
    p value and a positive estimate; its multiplicity gate is applied outside
    this numerical function. Percentile intervals are a separate procedure.

    With B blocks, n queries, block sizes n_b and sums A_b of d-null_mean:
        estimate-null_mean = sum(A_b)/n
        se^2 = B/(B-1) * sum((A_b-n_b*mean)^2) / n^2.
    The null-restricted residual is d-null_mean, NOT d-estimate. Bootstrap
    block sums are v_b*A_b, with independent equiprobable v_b in {-1,+1}.
    Each draw recomputes its equal-query mean and the same CR1 standard error.

    Fewer than two blocks or zero observed CR1 standard error is not evaluable.
    There is no minimum-block significance shortcut. Block-size diagnostics
    describe imbalance; they do not certify effective statistical independence.
    """
    d = np.asarray(differences, dtype=np.float64)
    labels = np.asarray(blocks)
    if d.ndim != 1 or labels.ndim != 1 or len(d) != len(labels) or len(d) == 0:
        raise ValueError("Differences and blocks must be nonempty aligned 1-D arrays.")
    if not np.all(np.isfinite(d)) or not np.isfinite(null_mean):
        raise ValueError("Nonfinite differences/null are invalid; no rows are dropped.")
    if isinstance(draws, bool) or int(draws) != draws or draws < 1:
        raise ValueError("draws must be a positive integer.")
    if labels.dtype.kind in "fc" and not np.all(np.isfinite(labels)):
        raise ValueError("Nonfinite block labels are invalid.")
    _, inverse = np.unique(labels, return_inverse=True)
    sizes = np.bincount(inverse).astype(np.float64)
    block_count = len(sizes)
    query_count = len(d)
    weights = sizes / query_count
    estimate = float(np.mean(d))
    result = {
        "method": "restricted_wild_cluster_t_CR1_intercept_only",
        "estimate": estimate,
        "null_mean": float(null_mean),
        "queries": query_count,
        "blocks": block_count,
        "block_sizes": sizes.astype(int).tolist(),
        "maximum_block_query_fraction": float(np.max(weights)),
        "size_effective_blocks": float(1.0 / np.sum(weights * weights)),
        "draws": int(draws),
        "seed": int(seed),
        "se": None,
        "t": None,
        "p_upper": None,
        "p_lower": None,
        "p_two_sided": None,
        "status": "not_evaluable",
    }
    if block_count < 2:
        result["reason"] = "fewer_than_two_nonempty_blocks"
        return result
    residuals_null = d - null_mean
    block_sums = np.bincount(inverse, weights=residuals_null, minlength=block_count)
    centered_mean = float(np.sum(block_sums) / query_count)
    observed_scores = block_sums - sizes * centered_mean
    variance_factor = block_count / (block_count - 1) / (query_count * query_count)
    se = float(np.sqrt(variance_factor * np.sum(observed_scores * observed_scores)))
    result["se"] = se
    if se == 0.0 or not np.isfinite(se):
        result["reason"] = "zero_or_nonfinite_observed_cluster_standard_error"
        return result
    observed_t = centered_mean / se
    rng = np.random.default_rng(seed)
    multipliers = rng.integers(0, 2, size=(int(draws), block_count), dtype=np.int8)
    multipliers = multipliers * 2 - 1
    star_sums = multipliers * block_sums[None, :]
    star_means = np.sum(star_sums, axis=1) / query_count
    star_scores = star_sums - star_means[:, None] * sizes[None, :]
    star_se = np.sqrt(variance_factor * np.sum(star_scores * star_scores, axis=1))
    star_t = _studentized(star_means, star_se)
    denominator = int(draws) + 1
    result.update(
        status="ok",
        t=float(observed_t),
        p_upper=float((1 + np.count_nonzero(star_t >= observed_t)) / denominator),
        p_lower=float((1 + np.count_nonzero(star_t <= observed_t)) / denominator),
        p_two_sided=float((1 + np.count_nonzero(np.abs(star_t) >= abs(observed_t))) / denominator),
        zero_se_bootstrap_draws=int(np.count_nonzero(star_se == 0)),
    )
    if return_distribution:
        result["bootstrap_t"] = star_t
    return result

