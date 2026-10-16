import numpy as np
import pytest
import pertrank as pr
from pertrank.datasets import load_example, software_example


@pytest.mark.parametrize('name,n_candidates,n_weight_genes', [('replogle',2000,8202),('tcell',52,13390)])
def test_archived_fixtures(name,n_candidates,n_weight_genes):
    data, provenance = load_example(name)
    assert len(data['target_ids']) == n_candidates
    assert int(data['weight_gene_count']) == n_weight_genes
    result = pr.from_statistics(data, data['selected_query_ids'], top_k=10,
        effect_gene_subset=True, weight_gene_count=n_weight_genes)
    for method in ['Euclidean','PertRank']:
        np.testing.assert_allclose(result.weights[method],data[('archived_weight_' if name=='tcell' else 'frozen_formula_weight_')+method],rtol=0,atol=1e-12)
    assert len(result.nominations)==180
    assert result.metadata['weight_gene_count']==n_weight_genes
    assert provenance['new_model_fits']==0


def test_installation_example():
    adata=software_example()
    result=pr.rank(adata,['G20'],input_scale='counts',full_gene_axis=True,strata_keys=['batch'],top_k=3)
    assert result.effects.shape==(8,60)
    assert len(result.nominations)==18


def test_missing_dataset():
    with pytest.raises(ValueError):
        load_example('unknown')
