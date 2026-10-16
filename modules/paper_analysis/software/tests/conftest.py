"""Synthetic inputs exercise software contracts, not biological performance."""
import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def counts_adata():
    rng = np.random.default_rng(3472)
    n_control, n_target, depth, n_gene = 24, 6, 8, 31
    n = n_control + n_target * depth
    loading = rng.normal(0, .18, (5, n_gene))
    rate = np.exp(1 + rng.normal(size=(n, 5)) @ loading)
    for j in range(n_target):
        rate[n_control+j*depth:n_control+(j+1)*depth, j:j+5] *= 1.8
    counts = rng.poisson(rate).astype(np.float64)
    # This gene is absent in input controls but contributes to full-library
    # denominators in perturbations: filtering before normalization is wrong.
    counts[:n_control, -1] = 0
    counts[n_control:, -1] = np.arange(n-n_control) % 9 + 25
    targets = ['NTC'] * n_control + [f'T{j}' for j in range(n_target) for _ in range(depth)]
    own = ['NTC'] * n_control + [f'G{j}' for j in range(n_target) for _ in range(depth)]
    obs = pd.DataFrame({'perturbation':targets, 'target_gene':own,
                        'control':np.arange(n)<n_control, 'stratum':['A']*n},
                       index=pd.Index([f'cell{i}' for i in range(n)], dtype=object))
    for key in ('perturbation', 'target_gene', 'stratum'):
        obs[key] = obs[key].astype(object)
    return ad.AnnData(counts, obs=obs, var=pd.DataFrame(index=pd.Index([f'G{j}' for j in range(n_gene)], dtype=object)))


@pytest.fixture
def frozen_core():
    path = Path(__file__).with_name('reference_core_v21_1.py')
    spec = importlib.util.spec_from_file_location('_reference_core_v21_1', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
