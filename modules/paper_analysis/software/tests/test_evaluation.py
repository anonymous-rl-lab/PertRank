"""Closed-form decision-scoring checks; synthetic examples are not experiments."""
from copy import deepcopy

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from pertrank.api import from_statistics
from pertrank.evaluation import evaluate_reference


def frozen_result(values, *, top_k=2, seed=2026):
    effects = np.asarray(values, dtype=float)[:, None]
    n = len(effects)
    statistics = {
        "target_ids": np.array([f"T{i}" for i in range(n)]),
        "target_gene_ids": np.array([f"G{i}" for i in range(n)]),
        "gene_ids": np.array(["G0"]), "effects": effects,
        "energy": np.square(effects[:, 0]), "noise_trace": np.zeros(n),
        "direction_energy": np.zeros((n, 5)), "direction_noise": np.zeros((n, 5)),
    }
    return from_statistics(statistics, ["G0"], top_k=top_k, seed=seed)


def reference(result, values):
    return pd.DataFrame({"G0": values}, index=result.effects.index)


def test_independent_tie_expectation_not_deterministic_list_agreement():
    # Three eligible candidates compete for two slots in both lists.
    # Each marginal is 2/3: expected overlap=3*(2/3)^2, P@2=2/3.
    result = frozen_result([999, 1, 1, 1])
    ref = reference(result, [-999, 7, 7, 7])
    answer = evaluate_reference(result, ref)
    np.testing.assert_allclose(answer.precision_at_k, 2/3, atol=1e-15)
    np.testing.assert_allclose(answer.expected_overlap, 4/3, atol=1e-15)
    assert (answer.n_candidates == 3).all()
    assert (answer.top_k == 2).all()
    # Deterministic nomination seed and delivered list order cannot affect scoring.
    other = frozen_result([999, 1, 1, 1], seed=34)
    other.nominations = other.nominations.iloc[::-1]
    assert_frame_equal(answer, evaluate_reference(other, ref))


def test_own_query_exclusion_applies_before_both_rankings_and_signs():
    result = frozen_result([1000, 4, 3, 2, 1], top_k=1)
    ref = reference(result, [-1000, 1, 2, 3, 4])
    # Both reference extrema differ from prediction extrema after excluding T0.
    answer = evaluate_reference(result, ref)
    assert (answer.precision_at_k == 0).all()
    assert (answer.n_candidates == 4).all()
    ref.loc["T0", "G0"] = 1e6
    assert_frame_equal(answer, evaluate_reference(result, ref))
    perfect = evaluate_reference(result, result.effects)
    assert (perfect.precision_at_k == 1).all()


def test_reference_axis_order_extra_genes_and_inputs_remain_unchanged():
    result = frozen_result([0, 4, 3, 2, 1])
    ref = reference(result, [0, 4, 3, 2, 1])
    baseline = evaluate_reference(result, ref)
    ref["unqueried"] = np.arange(len(ref))
    ref = ref.iloc[::-1, ::-1]
    before = ref.copy(deep=True)
    original_scores = result.all_candidate_scores.copy(deep=True)
    assert_frame_equal(baseline, evaluate_reference(result, ref))
    assert_frame_equal(ref, before)
    assert_frame_equal(result.all_candidate_scores, original_scores)


def test_external_estimator_uses_external_scores_with_fixed_source_weights():
    result = frozen_result([0, 4, 3, 2, 1], top_k=1)
    external = -result.effects
    reused = result.rank_effects(external)
    assert (evaluate_reference(result, external).precision_at_k == 0).all()
    assert (evaluate_reference(reused, external).precision_at_k == 1).all()
    assert_frame_equal(result.weights, reused.weights)


@pytest.mark.parametrize("kind", ["missing_target", "extra_target", "duplicate_target",
                                 "missing_query", "duplicate_gene", "nonfinite"])
def test_incompatible_reference_is_rejected_without_silent_intersection(kind):
    result = frozen_result([0, 4, 3, 2, 1])
    ref = result.effects.copy()
    if kind == "missing_target":
        ref = ref.iloc[1:]
    elif kind == "extra_target":
        ref.loc["extra"] = 1
    elif kind == "duplicate_target":
        ref.index = ["T0", "T1", "T1", "T3", "T4"]
    elif kind == "missing_query":
        ref.columns = ["different"]
    elif kind == "duplicate_gene":
        ref = pd.concat([ref, ref], axis=1)
    elif kind == "nonfinite":
        ref.iloc[0, 0] = np.inf
    with pytest.raises(ValueError):
        evaluate_reference(result, ref)


@pytest.mark.parametrize("kind", ["missing_candidate", "duplicate_candidate", "missing_arm",
                                 "nonfinite_score", "wrong_probability", "wrong_k", "wrong_own_id"])
def test_incomplete_or_inconsistent_prediction_output_is_rejected(kind):
    result = deepcopy(frozen_result([0, 4, 3, 2, 1]))
    table = result.all_candidate_scores
    if kind == "missing_candidate":
        result.all_candidate_scores = table.iloc[1:]
    elif kind == "duplicate_candidate":
        result.all_candidate_scores = pd.concat([table, table.iloc[:1]], ignore_index=True)
    elif kind == "missing_arm":
        result.all_candidate_scores = table[table.method != "Euclidean"]
    elif kind == "nonfinite_score":
        table.loc[0, "signed_score"] = np.nan
    elif kind == "wrong_probability":
        table.loc[0, "tie_inclusion_probability"] = .123
    elif kind == "wrong_k":
        table.loc[0, "top_k"] = 1
    elif kind == "wrong_own_id":
        table.loc[0, "target_gene_id"] = "G0"
    with pytest.raises(ValueError):
        evaluate_reference(result, result.effects)
