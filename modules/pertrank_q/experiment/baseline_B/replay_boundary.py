#!/usr/bin/env python3
"""Rebuild frozen predictions, rescore, and compare every saved numeric field."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

from rebuild_boundary import ROOT, rebuild, checked_relative
from common import dump, sha


def numeric_fields(value, prefix=''):
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)):
        return {prefix: value}
    if isinstance(value, dict):
        return {key: number for name, child in value.items()
                for key, number in numeric_fields(child, f'{prefix}/{name}').items()}
    if isinstance(value, list):
        return {key: number for index, child in enumerate(value)
                for key, number in numeric_fields(child, f'{prefix}/{index}').items()}
    return {}


def main(output_dir='results_replay', reference_dir='results'):
    output, reference = checked_relative(output_dir), checked_relative(reference_dir)
    if output.exists():
        raise RuntimeError(f'Replay output must be a fresh directory: {output}')
    reference_summary = reference / 'SUMMARY.json'
    if not reference_summary.is_file():
        raise RuntimeError(f'Complete frozen scoring is required first: {reference_summary}')
    original = json.loads(reference_summary.read_text())
    rebuilt = rebuild()
    environment = os.environ.copy()
    runtime_paths = [p for p in sys.path if Path(p).name == '.runtime' and Path(p).is_dir()]
    if runtime_paths:
        environment['PYTHONPATH'] = os.pathsep.join(runtime_paths + [environment.get('PYTHONPATH', '')])
    completed = subprocess.run([sys.executable, str(ROOT / 'score_boundary.py'), '--output-dir', output_dir],
                               cwd=ROOT, env=environment, check=True, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / 'score.log').write_text(completed.stdout)
    replayed = json.loads((output / 'SUMMARY.json').read_text())
    expected_numbers, actual_numbers = numeric_fields(original), numeric_fields(replayed)
    assert expected_numbers.keys() == actual_numbers.keys(), 'SUMMARY numeric field paths differ'
    differences = {key: abs(actual_numbers[key] - value) for key, value in expected_numbers.items()}
    differing = {key: value for key, value in differences.items() if value != 0}
    assert not differing, ('SUMMARY numeric values differ', differing)
    ignored = {'scoring_lock_sha256', 'files'}
    assert {k: v for k, v in original.items() if k not in ignored} == {
        k: v for k, v in replayed.items() if k not in ignored}, 'SUMMARY content differs'
    array_hashes = {}
    original_arrays_checked, original_arrays_missing = [], []
    for name, expected in original['files'].items():
        original_path = checked_relative(name)
        suffix = original_path.relative_to(reference)
        replay_path = output / suffix
        if original_path.is_file():
            assert sha(original_path) == expected, ('Changed frozen scored file', name)
            original_arrays_checked.append(name)
        else:
            # Compact packages retain the authoritative hashes in SUMMARY;
            # the original derived arrays need not be duplicated in the ZIP.
            original_arrays_missing.append(name)
        assert sha(replay_path) == expected, ('Replayed scored arrays differ', name)
        array_hashes[str(suffix)] = expected
    assert len(array_hashes) == len(replayed['files']), 'Scored array file count differs'
    report = dict(passed=True, training_performed=False, tasks=len(rebuilt['tasks']),
                  reconstructed_predictions_byte_identical=True,
                  numeric_summary_fields_compared=len(expected_numbers),
                  maximum_numeric_summary_difference=max(differences.values(), default=0),
                  scoring_arrays_byte_identical=True, scoring_array_sha256=array_hashes,
                  scoring_arrays_match_recorded_sha256=True,
                  original_scoring_array_files_checked=len(original_arrays_checked),
                  original_scoring_array_files_missing=len(original_arrays_missing),
                  original_scoring_array_paths_missing=original_arrays_missing,
                  original_summary_sha256=sha(reference_summary),
                  replay_summary_sha256=sha(output / 'SUMMARY.json'),
                  note='Scoring timestamps, their lock hashes, and output-directory paths are intentionally fresh.')
    dump(output / 'REPLAY_CHECK.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'scoring_array_sha256'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='results_replay')
    parser.add_argument('--reference-dir', default='results')
    arguments = parser.parse_args()
    main(arguments.output_dir, arguments.reference_dir)
