#!/usr/bin/env python3
"""Replay D3's archived per-query P@25 scores for one versus five inputs.

This checks the existing numeric reconciliation. It does not rerun the historical
32 input/axis/weight checks, read expression data, fit methods or draw bootstrap
samples. Original historical checks remain in the preserved source report.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent / 'd3_reconciliation'
ATOL = 1e-12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('--output-dir must be a new directory')
    manifest = json.loads((HERE / 'SOURCE_HASHES.json').read_text())
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Source file SHA-256 mismatch: ' + name)
    source = json.loads((HERE / 'D3_TASK_RECONCILIATION.json').read_text())
    with np.load(HERE / 'D3_TASK_RECONCILIATION_SCORES.npz', allow_pickle=False) as z:
        arms = z['arm_names'].tolist()
        scores = z['five_rep_query_direction_scores']
        native_scores = z['native_rep0_query_scores']
        if scores.shape != (3, 5, 300, 2) or native_scores.shape != (3, 300):
            raise ValueError('Unexpected archived score axes')
        if len(z['responsive_query_ids']) != 300 or len(z['responsive_query_blocks']) != 300:
            raise ValueError('Unexpected query axes')
        if arms != ['raw', 'Euclidean', 'PertRank']:
            raise ValueError('Unexpected arm order')
    # Preserve the original script's exact order of averaging operations.
    means = scores.mean((2, 3))
    average = scores.mean(1).mean((1, 2))
    result = {}
    checks = 0
    for i, arm in enumerate(arms):
        rep0 = float(native_scores[i].mean())
        result[arm] = {
            'per_rep_P25': means[i].tolist(),
            'five_rep_average_P25': float(average[i]),
            'native_benchmark_rep0_P25': rep0,
            'five_rep_minus_rep0': float(average[i] - rep0),
        }
        for key, expected in source['scores'][arm].items():
            got = np.asarray(result[arm][key])
            expected = np.asarray(expected)
            if got.shape != expected.shape or not np.allclose(got, expected, atol=ATOL, rtol=0):
                raise AssertionError('Reconciliation value mismatch: ' + arm + '.' + key)
            checks += got.size
    if not np.allclose(native_scores, scores[:, 0].mean(-1), atol=ATOL, rtol=0):
        raise AssertionError('Single-input scorer agreement failed')
    args.output_dir.mkdir(parents=True)
    report = {
        'status': 'PASS_ARCHIVED_D3_RECONCILIATION_SCORE_REPLAY',
        'scores': result,
        'numeric_values_compared_with_original_report': int(checks),
        'absolute_tolerance': ATOL,
        'historical_input_checks_rerun': False,
        'scope': manifest['scope'],
        'expression_matrices_read': False,
        'new_model_fits': 0,
        'new_bootstrap_draws': 0,
    }
    (args.output_dir / 'D3_RECONCILIATION_REPLAY.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.output_dir / 'D3_input_repetition_scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'single_development_input_P25', 'five_input_mean_P25', 'difference'])
        for arm in arms:
            r = result[arm]
            writer.writerow([arm, r['native_benchmark_rep0_P25'], r['five_rep_average_P25'], r['five_rep_minus_rep0']])
    print(json.dumps({'status': report['status'], 'numeric_values_compared': int(checks)}))


if __name__ == '__main__':
    main()
