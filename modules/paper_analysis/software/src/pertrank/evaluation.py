"""Evaluate frozen nominations against an independently supplied effect reference."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .api import PertRankResult, _options, _queries
from .core import ARMS, _boundary_probabilities, _strings


def evaluate_reference(result: PertRankResult, reference_effects: pd.DataFrame) -> pd.DataFrame:
    """Compute exact expected P@K on the result's fixed candidate pool.

    ``reference_effects`` has target identifiers as its index and response-gene
    identifiers as columns. Every fitted target and requested query must be
    present; extra reference genes are permitted. Both axes are matched by
    identifier. The reference must use the same effect orientation as the
    supplied predictions. Its construction and independence remain the caller's
    responsibility.

    The result's shortlist size, queries, candidate pool and own-target
    exclusions are retained. At each boundary tie, marginal inclusion is the
    number of remaining slots divided by tie size. Expected overlap is the dot
    product of prediction and reference inclusion probabilities, integrating
    independent uniform tie breaking. Deterministic display lists are not used.

    Returns one row per weighting arm, query and direction with
    ``precision_at_k``, ``expected_overlap``, ``top_k`` and ``n_candidates``.
    The arm names describe weighting of the supplied effect estimator: after
    ``rank_effects``, ``Unweighted`` means that external estimator without a
    reliability weight. Scores measure reference-list agreement, not a
    probability of biological validity. No model fitting, confidence intervals
    or pooling across experiments occurs, and neither input is modified.
    """
    if not isinstance(result, PertRankResult) or not isinstance(reference_effects, pd.DataFrame):
        raise TypeError("Supply a PertRankResult and a target-by-gene reference DataFrame")
    queries = _queries(result.metadata["query_ids"])
    top_k = result.metadata["top_k"]
    _options(top_k, result.metadata["tie_seed"])
    targets = _strings(result.statistics["target_ids"], len(result.statistics["target_ids"]), "target IDs")
    own = _strings(result.statistics["target_gene_ids"], len(targets), "target gene IDs")
    if len(set(targets)) != len(targets):
        raise ValueError("Fitted target IDs must be unique")
    ref_targets = _strings(reference_effects.index, len(reference_effects), "reference target IDs")
    ref_genes = _strings(reference_effects.columns, reference_effects.shape[1], "reference gene IDs")
    if len(set(ref_targets)) != len(ref_targets) or len(set(ref_genes)) != len(ref_genes):
        raise ValueError("Reference target and gene IDs must be unique")
    if set(ref_targets) != set(targets):
        raise ValueError("Reference effects must contain exactly the fitted target IDs")
    if set(queries) - set(ref_genes):
        raise ValueError("Reference effects must include every requested query")
    reference = reference_effects.copy()
    reference.index, reference.columns = ref_targets, ref_genes
    if not np.isfinite(reference.to_numpy(dtype=float)).all():
        raise ValueError("Reference effects must contain finite values")
    reference = reference.loc[targets, queries]

    scores = result.all_candidate_scores
    required = {"method", "query_id", "direction", "target_id", "target_gene_id",
                "signed_score", "tie_inclusion_probability", "top_k"}
    if not isinstance(scores, pd.DataFrame) or not required.issubset(scores.columns):
        raise ValueError("Result requires complete all_candidate_scores, including tie probabilities")
    scores = scores.loc[:, sorted(required)].copy()
    for key in ("method", "query_id", "direction", "target_id", "target_gene_id"):
        scores[key] = _strings(scores[key], len(scores), key)
    group_keys = ["method", "query_id", "direction"]
    expected_groups = {(arm, query, direction) for arm in ARMS for query in queries
                       for direction in ("up", "down")}
    groups = scores.groupby(group_keys, sort=False)
    if set(groups.indices) != expected_groups:
        raise ValueError("Candidate scores must contain exactly every method, query and direction")
    if not np.all(scores["top_k"].to_numpy() == top_k):
        raise ValueError("Candidate scores must use the result's fixed top_k")
    rows = []
    for arm in ARMS:
        for query in queries:
            eligible = targets[own != query]
            eligible_own = own[own != query]
            if len(eligible) < top_k:
                raise ValueError("Fewer than top_k candidates remain after own-target exclusion")
            truth = reference.loc[eligible, query].to_numpy(dtype=float)
            for direction, sign in (("up", 1), ("down", -1)):
                group = groups.get_group((arm, query, direction))
                if (group["target_id"].duplicated().any()
                        or set(group["target_id"]) != set(eligible)):
                    raise ValueError("Candidate scores must contain the complete eligible pool exactly once")
                group = group.set_index("target_id").loc[eligible]
                if not np.array_equal(group["target_gene_id"].to_numpy(), eligible_own):
                    raise ValueError("Candidate own-target identifiers disagree with fitted target identities")
                values = group["signed_score"].to_numpy(dtype=float)
                saved_p = group["tie_inclusion_probability"].to_numpy(dtype=float)
                if not np.isfinite(values).all() or not np.isfinite(saved_p).all():
                    raise ValueError("Candidate scores and tie probabilities must be finite")
                predicted_p = _boundary_probabilities(values, top_k)
                if not np.allclose(saved_p, predicted_p, rtol=0, atol=1e-12):
                    raise ValueError("Stored tie probabilities disagree with candidate scores")
                reference_p = _boundary_probabilities(sign * truth, top_k)
                overlap = float(predicted_p @ reference_p)
                rows.append({"method": arm, "query_id": str(query), "direction": direction,
                             "precision_at_k": overlap / top_k, "expected_overlap": overlap,
                             "top_k": int(top_k), "n_candidates": len(eligible)})
    answer = pd.DataFrame.from_records(rows)
    answer.attrs["scope"] = (
        "Fixed-query, fixed-pool reference-list agreement; independent uniform boundary ties. "
        "No biological probability, uncertainty interval or across-experiment inference."
    )
    return answer
