#!/usr/bin/env python3
"""Reproduce fixed-score interactions offline; no fits or new sampling."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ATOL = 1e-12


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def array_sha(a):
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(json.dumps(list(a.shape), separators=(',', ':')).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def load_data():
    manifest = json.loads((HERE / 'SUFFICIENT_DATA.json').read_text())
    if sha(HERE / 'scores.npz') != manifest['archive_sha256']:
        raise ValueError('Sufficient-data archive SHA256 differs')
    data = {}
    with np.load(HERE / 'scores.npz', allow_pickle=False) as z:
        decoded = {}
        for key, r in manifest['arrays'].items():
            if r['codec'] == 'column_dictionary':
                columns = z[key + '_values'][z[key + '_codes']]
                a = columns[:, z[key + '_column_index']]
            elif r['codec'] == 'dictionary':
                a = z[key + '_values'][z[key + '_codes']]
            elif r['codec'] == 'narrow_integer':
                a = z[key + '_data'].astype(np.dtype(r['dtype']))
            else:
                a = z[key + '_data']
            if a.dtype.str != r['dtype'] or list(a.shape) != r['shape'] or array_sha(a) != r['array_sha256']:
                raise ValueError('Lossless reconstruction differs: ' + key)
            decoded[key] = a
        for name, key in manifest['bindings'].items():
            tag, field = name.split('/', 1)
            data.setdefault(tag, {})[field] = decoded[key]
    return data, manifest


def summary(point, draws):
    return {'point': float(point),
            'descriptive_conditional_95_interval': np.quantile(draws, [.025, .975]).tolist(),
            'descriptive_conditional_99_interval': np.quantile(draws, [.005, .995]).tolist(),
            'bootstrap_sd': float(np.std(draws, ddof=1)), 'draws': len(draws)}


def compare(a, b, key='result'):
    if isinstance(b, dict):
        if set(a) != set(b):
            raise AssertionError('Different keys: ' + key)
        return sum(compare(a[k], b[k], key + '.' + k) for k in b)
    if isinstance(b, list):
        if len(a) != len(b):
            raise AssertionError('Different sequence: ' + key)
        return sum(compare(x, y, key + f'[{i}]') for i, (x, y) in enumerate(zip(a, b)))
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        if not np.isfinite(a) or abs(a - b) > ATOL:
            raise AssertionError(f'Numerical mismatch at {key}: {a} != {b}')
    elif a != b:
        raise AssertionError('Mismatch: ' + key)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if not __debug__:
        parser.error('Do not use python -O')
    if args.output_dir.exists():
        parser.error('--output-dir must be a new directory')
    data, manifest = load_data()
    expected_file = HERE / 'RESULTS.json'
    if sha(expected_file) != manifest['original_results_sha256']:
        raise ValueError('Archived RESULTS.json differs')
    expected = json.loads(expected_file.read_text())
    checks = 0
    for tag in ['rest', 'stim', 'n20', 'n50', 'n100']:
        d = data[tag]
        assert len(np.unique(d['queries'])) == 300 and len(np.unique(d['blocks'])) == 30
        weights, sampled = d['query_weights'], d['sampled_query_blocks']
        assert sampled.shape == (2000, 30) and weights.shape == (2000, 300)
        ids = np.unique(d['blocks'])
        counts = np.stack([(sampled == i).sum(1) for i in range(len(ids))], axis=1)
        counts = counts[:, np.searchsorted(ids, d['blocks'])]
        np.testing.assert_allclose(weights, counts / counts.sum(1)[:, None], rtol=0, atol=ATOL)
        checks += 1
    for tag in ['n50', 'n100']:
        for field in ['queries', 'candidates', 'blocks', 'query_weights', 'sampled_query_blocks']:
            np.testing.assert_array_equal(data['n20'][field], data[tag][field])
            checks += 1
    result = {'A_geometry_interaction': {}, 'B_depth_interaction': {}}
    for k in [25, 10]:
        state_points, state_combined_boots = [], []
        for index, tag in enumerate(['rest', 'stim']):
            d = data[tag]
            scores = d[f'six_arm_p{k}'].mean(0)
            point = scores.mean(1)
            boot = d['query_weights'] @ scores.T
            combined_boot = data['equal'][f'state_{index}_query_weights'] @ scores.T
            # Reconstructed saved block draws validate both original streams.
            sampled = data['equal'][f'state_{index}_sampled_blocks']
            labels = np.unique(d['blocks'])
            counts = np.stack([(sampled == i).sum(1) for i in range(len(labels))], axis=1)
            counts = counts[:, np.searchsorted(labels, d['blocks'])]
            np.testing.assert_allclose(counts / counts.sum(1)[:, None],
                                       data['equal'][f'state_{index}_query_weights'], rtol=0, atol=ATOL)
            state_points.append(point)
            state_combined_boots.append(combined_boot)
            pairs = {'raw_geometry': (point[2]-point[1], boot[:, 2]-boot[:, 1]),
                     'FR_geometry': (point[5]-point[4], boot[:, 5]-boot[:, 4]),
                     'I': ((point[5]-point[4])-(point[2]-point[1]),
                           (boot[:, 5]-boot[:, 4])-(boot[:, 2]-boot[:, 1]))}
            result['A_geometry_interaction'].setdefault(tag, {})[str(k)] = {
                name: summary(*values) for name, values in pairs.items()}
            checks += 1
        point = np.mean(state_points, axis=0)
        boot = (state_combined_boots[0] + state_combined_boots[1]) / 2
        pairs = {'raw_geometry': (point[2]-point[1], boot[:, 2]-boot[:, 1]),
                 'FR_geometry': (point[5]-point[4], boot[:, 5]-boot[:, 4]),
                 'I': ((point[5]-point[4])-(point[2]-point[1]),
                       (boot[:, 5]-boot[:, 4])-(boot[:, 2]-boot[:, 1]))}
        result['A_geometry_interaction'].setdefault('equal_state', {})[str(k)] = {
            name: summary(*values) for name, values in pairs.items()}
        depth, qs, bs = {}, {}, {}
        for tag in ['n20', 'n50', 'n100']:
            d = data[tag]
            scores = d[f'four_arm_p{k}'].mean(0)
            boot = d['query_weights'] @ scores.T
            qs[tag] = scores[2] - scores[3]
            bs[tag] = boot[:, 2] - boot[:, 3]
            depth[tag] = summary(qs[tag].mean(), bs[tag])
        q, b = qs['n20']-qs['n100'], bs['n20']-bs['n100']
        np.testing.assert_allclose(data['n20']['query_weights'] @ q, b, rtol=0, atol=ATOL)
        depth['J_n20_minus_n100'] = summary(q.mean(), b)
        result['B_depth_interaction'][str(k)] = depth
        checks += 2
    comparisons = sum(compare(result[k], expected[k], k) for k in result)
    result['scope'] = expected['scope']
    result['status'] = 'PASS_FIXED_SCORE_REPRODUCTION'
    receipt = {'status': 'PASS_FIXED_SCORE_REPRODUCTION', 'scientific_values_compared': comparisons,
               'source_consistency_checks': checks, 'absolute_tolerance': ATOL,
               'new_model_fits': 0, 'expression_matrices_read': False, 'new_bootstrap_draws': 0,
               'data_sha256': manifest['archive_sha256'], 'numpy_version': np.__version__}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output_dir / 'REPRODUCTION_CHECK.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
