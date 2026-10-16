#!/usr/bin/env python3
"""Reproduce seven fixed development arms from compact scalar statistics.

This does not fit control directions, recompute full-gene moments, or refit the
TRADE mixture. TRADE's original R-serialized weight is retained for exact ranking;
its formula is independently checked against retained fitted latent moments.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ARMS = ('raw', 'Euclidean', 'PertRank', 'diagonal', 'TRADE_scalar', 'PR_E', 'PR_M')


def array_sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def scalar(a, nu):
    """Frozen diagonal/Pearson scalar convention; float64 throughout."""
    assert np.isfinite(a).all() and np.isfinite(nu).all()
    assert np.min(a) >= -1e-10 and np.min(nu) >= -1e-10
    a, nu = np.maximum(a, 0), np.maximum(nu, 0)
    w = np.zeros_like(a)
    pos = a > 0
    w[pos] = np.clip(1-nu[pos]/a[pos], 0, 1)
    return w


def metric_weights(z, beta):
    """Original R20 operation order, with target-first moment arrays."""
    m2, vb = z['log_m2'], z['log_vb']
    pm, pv = z['log_projection_mean'], z['log_projection_variance']
    lam, trc = z['log_control_eigenvalues'], float(z['log_control_trace'])
    g, nc = int(z['weight_gene_count']), int(z['n_control'])
    a = m2-beta*(pm*pm).sum(-1)/g
    nu = vb/20+trc/(g*nc)-beta*(pv/20+lam/nc).sum(-1)/g
    assert min(float(a.min()), float(nu.min())) >= -1e-10
    return np.clip(1-nu/np.maximum(a, 1e-300), 0, 1).T


def reproduce_weights(path):
    """Return [arm, group, target] weights and a fully specified audit."""
    with np.load(path, allow_pickle=False) as z:
        assert tuple(z['arm_names'].astype(str)) == ARMS
        recorded = z['recorded_weights']
        assert recorded.shape[:2] == (7, 5)
        trade_formula = z['trade_S']/(z['trade_S']+z['trade_noise'])
        trade_saved = z['trade_serialized_weight']
        trade_error = float(np.max(np.abs(trade_formula-trade_saved)))
        moment_error = float(np.max(np.abs(z['trade_S']-z['trade_TI']-z['trade_latent_mean']**2)))
        assert trade_error <= 2e-14
        assert moment_error <= 1e-10
        weights = np.stack([
            np.ones_like(recorded[0]),
            metric_weights(z, 0),
            metric_weights(z, .9),
            scalar(z['diagonal_energy'], z['diagonal_noise']).T,
            trade_saved.T,
            scalar(z['pearson_energy_E'], z['pearson_noise_E']),
            scalar(z['pearson_energy_M'], z['pearson_noise_M']),
        ])
        checks = {}
        for i, arm in enumerate(ARMS):
            assert np.array_equal(weights[i], recorded[i]), ('weight mismatch', arm)
            checks[arm] = {
                'exact_against_frozen_weights': True,
                'maximum_absolute_error': float(np.max(np.abs(weights[i]-recorded[i]))),
                'array_sha256': array_sha256(weights[i]),
                'mode': 'original R-serialized fitted weight; scalar formula independently checked'
                        if arm == 'TRADE_scalar' else 'recalculated from sufficient statistics',
            }
        audit = {
            'donor': str(z['donor']), 'status': 'PASS',
            'shape': list(weights.shape), 'arms': checks,
            'TRADE_formula_maximum_absolute_error': trade_error,
            'TRADE_latent_moment_identity_maximum_absolute_error': moment_error,
            'TRADE_mixture_refit': False,
            'control_direction_refit': False,
            'raw_cell_read': False,
            'scope': 'fixed scalar-statistic replay; not raw-data preprocessing or statistical refitting',
        }
        return weights, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output', type=Path)
    a = ap.parse_args()
    result = {d: reproduce_weights(a.repo/'data'/d/'weight_stats.npz')[1]
              for d in ('D1', 'D2', 'D3')}
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if a.output:
        a.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
