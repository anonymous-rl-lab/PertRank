"""Adapter regression and misuse prevention on small synthetic cell inputs."""
import inspect

import anndata as ad
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from scipy import sparse

from pertrank import rank, rank_effects, from_statistics


KEYS = dict(input_scale='counts', strata_keys=['stratum'], target_key='perturbation',
            target_gene_key='target_gene', control_key='control', top_k=2,
            full_gene_axis=True, batch_rows=11)


def run(a, **changes):
    return rank(a, ['G0','G10'], **{**KEYS, **changes})


def assert_same(left,right):
    for name in ('effects','weights','nominations','all_candidate_scores'):
        assert_frame_equal(getattr(left,name), getattr(right,name), check_exact=False, check_dtype=False, rtol=1e-12, atol=1e-12)


def test_counts_adapter_matches_frozen_v211_and_preserves_full_denominator(counts_adata,frozen_core):
    a=counts_adata
    result=run(a)
    ctrl=a.obs['control'].to_numpy()
    keep=np.flatnonzero(a.X[ctrl].sum(0)>0)
    log,scales=frozen_core.normalize_counts(a.X[:,keep], a.X.sum(1), a.obs['stratum'], ctrl)
    log[ctrl]=log[ctrl].astype(np.float32).astype(np.float64)
    frozen=frozen_core.fit_cells(log, np.asarray(a.var_names)[keep], a.obs['perturbation'],
                                 a.obs['stratum'],ctrl,a.obs['target_gene'])
    for name in ('effects','energy','noise_trace','direction_energy','direction_noise'):
        np.testing.assert_allclose(result.statistics[name],frozen[name],rtol=1e-12,atol=1e-12)
    for arm in ('Unweighted','Euclidean','PertRank'):
        np.testing.assert_allclose(result.weights[arm],frozen['weights'][arm],rtol=1e-12,atol=1e-12)
    assert result.metadata['normalization_scales']==scales
    assert result.metadata['weight_gene_count']==30
    assert result.metadata['full_gene_count']==31
    assert 'G30' not in result.effects.columns
    # Show that the fixture actually detects the forbidden early filtering.
    wrong,_=frozen_core.normalize_counts(a.X[:,keep],a.X[:,keep].sum(1),a.obs['stratum'],ctrl)
    assert np.max(np.abs(wrong[~ctrl]-log[~ctrl]))>.1
    records,_=frozen_core.nominate(frozen,['G0','G10'],top_k=2)
    assert_frame_equal(result.nominations,pd.DataFrame(records),check_exact=False,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('storage',['csr','csc','backed_dense','backed_csr'])
def test_storage_invariance(counts_adata,storage,tmp_path):
    expected=run(counts_adata)
    a=counts_adata.copy()
    if storage in ('csr','backed_csr'): a.X=sparse.csr_matrix(a.X)
    if storage=='csc': a.X=sparse.csc_matrix(a.X)
    if storage.startswith('backed'):
        path=tmp_path/'input.h5ad'
        a.write_h5ad(path)
        a=ad.read_h5ad(path,backed='r')
    try: assert_same(run(a),expected)
    finally:
        if a.isbacked: a.file.close()


def test_named_layer_is_used_without_modifying_anndata(counts_adata):
    a=counts_adata
    expected=run(a)
    a.layers['raw']=sparse.csr_matrix(a.X)
    a.X=np.full(a.shape,np.nan)
    old_obs=a.obs.copy(deep=True)
    old_var=a.var.copy(deep=True)
    actual=run(a,layer='raw')
    assert_same(actual,expected)
    assert actual.metadata['layer']=='raw'
    assert np.isnan(a.X).all()
    assert_frame_equal(a.obs,old_obs)
    assert_frame_equal(a.var,old_var)
    assert len(a.uns)==0


def test_library_total_check_detects_prefiltered_input(counts_adata):
    a=counts_adata
    a.obs['full_total']=a.X.sum(1)
    assert_same(run(a,library_totals_key='full_total'),run(a))
    subset=a[:,:-1].copy()
    with pytest.raises(ValueError,match='library totals'):
        run(subset,library_totals_key='full_total')


def test_log_input_remains_unchanged_and_matches_frozen_core(counts_adata,frozen_core):
    a=counts_adata
    a.X=np.log1p(a.X)
    expected=frozen_core.fit_cells(a.X,a.var_names,a.obs['perturbation'],a.obs['stratum'],
                                   a.obs['control'],a.obs['target_gene'])
    result=run(a,input_scale='log',full_gene_axis=False)
    np.testing.assert_allclose(result.effects,expected['effects'],rtol=1e-12,atol=1e-12)
    for method in ('Euclidean','PertRank'):
        np.testing.assert_allclose(result.weights[method],expected['weights'][method],rtol=1e-12,atol=1e-12)


def test_multistratum_log_uses_matched_controls(counts_adata):
    a=counts_adata
    a.X=np.log1p(a.X)
    b=a.copy()
    b.X=b.X+2
    b.obs_names=[x+'_B' for x in b.obs_names]
    b.obs['stratum']='B'
    b.obs['perturbation']=b.obs['perturbation']+'_B'
    together=ad.concat([a,b])
    result=run(together,input_scale='log')
    ar=run(a,input_scale='log')
    br=run(b,input_scale='log')
    for source in (ar,br):
        np.testing.assert_allclose(result.effects.loc[source.effects.index],source.effects,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(result.weights.loc[source.weights.index,['Euclidean','PertRank']],
                                   source.weights[['Euclidean','PertRank']],rtol=1e-12,atol=1e-12)
    assert set(result.controls)=={'A','B'}


@pytest.mark.parametrize('issue',['duplicate_cells','duplicate_genes','missing_target',
 'missing_gene','missing_stratum','missing_control_column','bad_control','few_controls',
 'target_spans_strata','missing_matching_controls','negative','noninteger','nan','zero_library',
 'target_single_cell','inconsistent_own_id'])
def test_invalid_inputs_fail_explicitly(counts_adata,issue):
    a=counts_adata
    opts={}
    if issue=='duplicate_cells': a.obs_names=[a.obs_names[0]]*a.n_obs
    elif issue=='duplicate_genes': a.var_names=[a.var_names[0]]*a.n_vars
    elif issue=='missing_target': a.obs.loc[a.obs.index[-1],'perturbation']=None
    elif issue=='missing_gene': a.var_names=list(a.var_names[:-1])+['']
    elif issue=='missing_stratum': a.obs.loc[a.obs.index[-1],'stratum']=None
    elif issue=='missing_control_column': del a.obs['control']
    elif issue=='bad_control': a.obs['control']=np.repeat(2,a.n_obs)
    elif issue=='few_controls': a=a[np.r_[0:5,24:a.n_obs]].copy()
    elif issue=='target_spans_strata':
        a.X=np.log1p(a.X); opts['input_scale']='log'
        a.obs.loc[a.obs.index[:12],'stratum']='B'
        a.obs.loc[a.obs.index[-1],'stratum']='B'
    elif issue=='missing_matching_controls':
        a.X=np.log1p(a.X); opts['input_scale']='log'
        a.obs.loc[a.obs.index[-8:],'stratum']='B'
    elif issue=='negative': a.X[-1,0]=-1
    elif issue=='noninteger': a.X[-1,0]=.1
    elif issue=='nan': a.X[-1,0]=np.nan
    elif issue=='zero_library': a.X[-1]=0
    elif issue=='target_single_cell': a=a[np.r_[0:25,32:a.n_obs]].copy()
    elif issue=='inconsistent_own_id': a.obs.loc[a.obs.index[-1],'target_gene']='different'
    with pytest.raises(ValueError): run(a,**opts)


@pytest.mark.parametrize('change',[{'input_scale':'scaled'}, {'full_gene_axis':False},
 {'strata_keys':None}, {'strata_keys':'stratum'}, {'strata_keys':['absent']},
 {'layer':'missing'}, {'batch_rows':0}, {'seed':1.2}])
def test_explicit_contract_required(counts_adata,change):
    with pytest.raises(ValueError): run(counts_adata,**change)


def test_reference_or_evaluation_inputs_are_not_supported(counts_adata):
    forbidden={'reference','truth','evaluation','deep_reference','query_selection'}
    assert not forbidden.intersection(inspect.signature(rank).parameters)
    with pytest.raises(TypeError):
        rank(counts_adata,['G0'],**KEYS,reference=np.zeros((1,1)))


def test_from_statistics_replays_frozen_predictions(counts_adata):
    fit=run(counts_adata)
    replay=from_statistics(fit.statistics,['G0','G10'],top_k=2)
    assert_same(replay,fit)
    corrupt={**fit.statistics,'weights':{a:np.zeros(6) for a in ('Unweighted','Euclidean','PertRank')}}
    assert_same(from_statistics(corrupt,['G0','G10'],top_k=2),fit)
    incomplete={**fit.statistics,'effects':fit.statistics['effects'][:,:5],
                'gene_ids':fit.statistics['gene_ids'][:5]}
    with pytest.raises(ValueError,match='energy'):
        from_statistics(incomplete,['G0'],top_k=2)


def test_external_effects_are_joined_by_target_and_gene_ids(counts_adata):
    fitted=run(counts_adata)
    external=fitted.effects[['G0','G10']]*2.5
    expected=rank_effects(fitted,external,top_k=2)
    reversed_both=external.iloc[::-1,::-1]
    actual=fitted.rank_effects(reversed_both,top_k=2)
    assert_frame_equal(actual.nominations,expected.nominations)
    assert_frame_equal(actual.all_candidate_scores,expected.all_candidate_scores)
    assert_frame_equal(actual.weights,fitted.weights)
    assert actual.metadata['weight_gene_count']==30
    assert actual.metadata['external_effect_gene_count']==2
    np.testing.assert_allclose(actual.effects.loc[external.index,external.columns],external)
    np.testing.assert_allclose(fitted.effects[['G0','G10']]*2.5,external)


def test_external_readout_preserves_source_statistics_and_allows_chained_reuse(counts_adata):
    fitted = run(counts_adata)
    external = fitted.rank_effects(fitted.effects[['G0', 'G10']] * 2.5)
    np.testing.assert_array_equal(external.statistics['gene_ids'], fitted.statistics['gene_ids'])
    np.testing.assert_array_equal(external.statistics['effects'], fitted.statistics['effects'])
    assert external.metadata['effect_origin'] == 'external'
    assert fitted.metadata['effect_origin'] == 'source'
    assert external.metadata['source_effect_gene_count'] == 30
    assert external.metadata['effect_gene_count'] == 2
    # A first query-only readout must not erase the fitted source-gene axis.
    second_readout = fitted.effects[['G5']] * 3
    direct = fitted.rank_effects(second_readout, queries='G5')
    chained = external.rank_effects(second_readout, queries='G5')
    assert_same(chained, direct)
    replay = from_statistics(external.statistics, ['G0', 'G10'], top_k=2)
    assert_same(replay, fitted)


def test_reused_result_does_not_alias_source_statistics_or_metadata(counts_adata):
    fitted = run(counts_adata)
    expected_effects = fitted.statistics['effects'].copy()
    expected_weights = fitted.statistics['weights']['PertRank'].copy()
    expected_directions = fitted.controls['A']['directions'].copy()
    expected_scale = fitted.metadata['normalization_scales']['A']
    external = fitted.rank_effects(fitted.effects[['G0', 'G10']] * 2)
    external.statistics['effects'][0, 0] += 100
    external.statistics['weights']['PertRank'][0] = -1
    external.controls['A']['directions'][0, 0] += 100
    external.metadata['normalization_scales']['A'] = -1
    np.testing.assert_array_equal(fitted.statistics['effects'], expected_effects)
    np.testing.assert_array_equal(fitted.statistics['weights']['PertRank'], expected_weights)
    np.testing.assert_array_equal(fitted.controls['A']['directions'], expected_directions)
    assert fitted.metadata['normalization_scales']['A'] == expected_scale


def test_metadata_records_running_software_version(counts_adata):
    import pertrank
    assert run(counts_adata).metadata['pertrank_version'] == pertrank.__version__


@pytest.mark.parametrize('issue',['missing_target','extra_target','duplicate_target','duplicate_gene',
                                  'unknown_gene','missing_query','nan','positional_array'])
def test_external_effects_require_complete_labelled_alignment(counts_adata,issue):
    fitted=run(counts_adata)
    external=fitted.effects[['G0','G10']].copy()
    if issue=='missing_target': external=external.iloc[:-1]
    elif issue=='extra_target': external.loc['new']=0
    elif issue=='duplicate_target': external.index=['T0']*len(external)
    elif issue=='duplicate_gene': external.columns=['G0','G0']
    elif issue=='unknown_gene': external.columns=['G0','unknown']
    elif issue=='missing_query': external=external[['G0']]
    elif issue=='nan': external.iloc[0,0]=np.nan
    elif issue=='positional_array': external=external.to_numpy()
    with pytest.raises((ValueError,TypeError)): rank_effects(fitted,external)


def test_archived_query_projection_requires_explicit_full_axis_provenance(counts_adata):
    fitted=run(counts_adata)
    stats=fitted.statistics
    ix=[0,10]
    subset={**stats,'effects':stats['effects'][:,ix],'gene_ids':stats['gene_ids'][ix]}
    replay=from_statistics(subset,['G0','G10'],top_k=2,effect_gene_subset=True,weight_gene_count=30)
    assert_frame_equal(replay.nominations,fitted.nominations)
    assert replay.metadata['weight_gene_count']==30
    assert replay.metadata['effect_gene_count']==2
    with pytest.raises(ValueError):
        from_statistics(subset,['G0','G10'],top_k=2,effect_gene_subset=True)
    impossible={**subset,'effects':subset['effects']*1000}
    with pytest.raises(ValueError):
        from_statistics(impossible,['G0','G10'],top_k=2,effect_gene_subset=True,weight_gene_count=30)


@pytest.mark.parametrize('count', [30.0, True])
def test_archived_weight_axis_count_requires_an_integer(counts_adata, count):
    fitted = run(counts_adata)
    with pytest.raises(ValueError, match='positive integer'):
        from_statistics(fitted.statistics, ['G0'], top_k=2, weight_gene_count=count)


def test_counts_multistratum_is_not_silently_pooled(counts_adata):
    counts_adata.obs.loc[counts_adata.obs.index[:12],'stratum']='B'
    with pytest.raises(ValueError,match='one prespecified matching stratum'):
        run(counts_adata)


def test_query_absent_in_controls_and_boolean_top_k_fail(counts_adata):
    with pytest.raises(ValueError,match='query|Queries'):
        rank(counts_adata,['G30'],**KEYS)
    with pytest.raises(ValueError,match='top_k'):
        run(counts_adata,top_k=True)
