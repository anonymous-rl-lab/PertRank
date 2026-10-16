"""Mathematical and shortlist checks independent of AnnData plumbing."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pertrank.core import nominate, reliability_weights


def test_reference_snapshot_hash():
    directory = Path(__file__).parent
    provenance = json.loads((directory/'REFERENCE_PROVENANCE.json').read_text())
    assert hashlib.sha256((directory/'reference_core_v21_1.py').read_bytes()).hexdigest() == provenance['sha256']


def test_weights_have_target_noise_and_control_projection_contributions():
    w = reliability_weights([4.], [1.], [[1.]], [[.8]], k=1, beta=.9)
    np.testing.assert_allclose(w['Euclidean'], [.75], atol=1e-15)
    np.testing.assert_allclose(w['PertRank'], [1-(1-.9*.8)/(4-.9)], atol=1e-15)
    zero = reliability_weights([0.], [1.], [[0.]], [[0.]], k=1)
    assert zero['PertRank'][0] == 0
    assert zero['Euclidean'][0] == 0
    assert zero['Unweighted'][0] == 1


def test_projected_energy_cannot_exceed_total():
    with pytest.raises(ValueError, match='exceed'):
        reliability_weights([1.], [1.], [[2.]], [[0.]], k=1)


def tied_result():
    return dict(gene_ids=np.array(['G0']), target_ids=np.array([f'T{i}' for i in range(6)]),
                target_gene_ids=np.array([f'G{i}' for i in range(6)]),
                effects=np.array([[999.],[3.],[2.],[0.],[0.],[0.]]),
                weights={a:np.ones(6) for a in ('Unweighted','Euclidean','PertRank')})


def test_boundary_ties_and_own_query_exclusion():
    records, p = nominate(tied_result(), ['G0'], top_k=3, seed=25)
    assert not any(r['target_id'] == 'T0' for r in records)
    np.testing.assert_allclose(p[:,0,0], np.tile([0,1,1,1/3,1/3,1/3], (3,1)))
    np.testing.assert_allclose(p.sum(-1), 3)
    assert p[:,0,:,0].sum() == 0
    assert len(records) == 3*2*3


def test_delivery_tie_breaking_is_id_stable_under_row_permutation():
    result = tied_result()
    expected, _ = nominate(result, ['G0'], top_k=3, seed=25)
    ix = np.array([3,5,1,0,4,2])
    shuffled = dict(result, target_ids=result['target_ids'][ix],
                    target_gene_ids=result['target_gene_ids'][ix], effects=result['effects'][ix],
                    weights={k:v[ix] for k,v in result['weights'].items()})
    actual, _ = nominate(shuffled, ['G0'], top_k=3, seed=25)
    assert actual == expected


@pytest.mark.parametrize('query,top_k', [(['missing'],2), (['G0','G0'],2), (['G0'],6), (['G0'],0)])
def test_invalid_nominations_rejected(query,top_k):
    with pytest.raises(ValueError):
        nominate(tied_result(), query, top_k=top_k)


@pytest.mark.parametrize('options', [{'top_k': True}, {'seed': True}, {'seed': 1.5}])
def test_low_level_nomination_requires_integer_configuration(options):
    with pytest.raises(ValueError, match='integer'):
        nominate(tied_result(), ['G0'], **{'top_k': 3, **options})


@pytest.mark.parametrize('issue', ['nan_effect', 'bad_effect_shape', 'nan_weight',
                                  'negative_weight', 'large_weight', 'bad_weight_shape',
                                  'duplicate_targets', 'missing_own_id', 'missing_arm'])
def test_low_level_nomination_rejects_invalid_score_inputs(issue):
    result = tied_result()
    if issue == 'nan_effect': result['effects'][2, 0] = np.nan
    elif issue == 'bad_effect_shape': result['effects'] = np.ones((5, 1))
    elif issue == 'nan_weight': result['weights']['PertRank'][2] = np.nan
    elif issue == 'negative_weight': result['weights']['PertRank'][2] = -0.1
    elif issue == 'large_weight': result['weights']['PertRank'][2] = 1.1
    elif issue == 'bad_weight_shape': result['weights']['PertRank'] = np.ones(5)
    elif issue == 'duplicate_targets': result['target_ids'][2] = result['target_ids'][1]
    elif issue == 'missing_own_id': result['target_gene_ids'] = ['G0', 'NaT', 'G2', 'G3', 'G4', 'G5']
    elif issue == 'missing_arm': del result['weights']['PertRank']
    with pytest.raises(ValueError):
        nominate(result, ['G0'], top_k=3)


@pytest.mark.parametrize('options', [{'k': True}, {'beta': True}, {'beta': [0.9]}])
def test_weight_parameters_reject_boolean_or_nonscalar_options(options):
    with pytest.raises(ValueError, match='integer k'):
        reliability_weights([4.], [1.], [[1.]], [[.8]], **{'k': 1, **options})


def test_low_level_normalization_rejects_fractional_counts():
    from pertrank.core import normalize_counts
    with pytest.raises(ValueError, match='integer raw counts'):
        normalize_counts([[1.2, 3.], [2., 2.]], [4.2, 4.], ['A', 'A'], [True, False])


def test_cell_statistics_match_direct_covariance_trace(counts_adata):
    from pertrank.core import fit_cells, normalize_counts
    a = counts_adata
    controls = a.obs['control'].to_numpy()
    keep = np.asarray(a.X[controls].sum(0)) > 0
    normalized, _ = normalize_counts(a.X[:,keep], a.X.sum(1), a.obs['stratum'], controls)
    normalized[controls] = normalized[controls].astype(np.float32).astype(np.float64)
    fit = fit_cells(normalized, np.asarray(a.var_names)[keep], a.obs['perturbation'],
                    a.obs['stratum'], controls, a.obs['target_gene'])
    control = normalized[controls]
    control_covariance = np.cov(control, rowvar=False, ddof=1)/len(control)
    U = fit['control_models']['A']['directions']
    geometry = np.eye(sum(keep)) - .9*U@U.T
    for i,target in enumerate(fit['target_ids']):
        target_rows = normalized[a.obs['perturbation'].to_numpy()==target]
        covariance = np.cov(target_rows, rowvar=False, ddof=1)/len(target_rows) + control_covariance
        direct_noise = np.trace(geometry@covariance)
        compact_noise = fit['noise_trace'][i]-.9*fit['direction_noise'][i].sum()
        np.testing.assert_allclose(compact_noise, direct_noise, atol=1e-12, rtol=1e-12)
        effect = target_rows.mean(0)-control.mean(0)
        direct_energy = effect@geometry@effect
        compact_energy = fit['energy'][i]-.9*fit['direction_energy'][i].sum()
        np.testing.assert_allclose(compact_energy, direct_energy, atol=1e-12, rtol=1e-12)
