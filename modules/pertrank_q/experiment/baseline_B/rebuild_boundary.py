#!/usr/bin/env python3
"""Rebuild compact-package intermediates; require every locked SHA to match.

No training or model selection occurs. All three prediction locks and saved
fit hashes are verified before the full source inputs are opened. Reference
arrays are then copied only to reconstruct the original calibration/evaluation
views; frozen inference consumes the reference-free held-feature views.
"""
from pathlib import Path
import json
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
for runtime in (ROOT / '.runtime', ROOT.parent / 'pertrank_neural_rank_20261115' / '.runtime'):
    if runtime.is_dir():
        sys.path.insert(0, str(runtime))
        break

import numpy as np
import torch
from common import SOURCES, load, sha, subset, dump, inner_ids
from gate_model import ReliabilityGate
from score_boundary import ARMS, SEEDS
from base_training import GateNormalizer, predict
from boundary_components import BoundaryNormalizer


def checked_relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe package-relative path: {value}')
    return ROOT / path


def prediction_barrier():
    """Read no input NPZ or evaluation arrays until every source is frozen."""
    lock_paths = {s: ROOT / 'predictions' / f'{s}_PREDICTION_LOCK.json' for s in SOURCES}
    missing = [str(p) for p in lock_paths.values() if not p.is_file()]
    if missing:
        raise RuntimeError('All source prediction locks are required before rebuild: ' + ', '.join(missing))
    plan_path = ROOT / 'PLAN_LOCK.json'
    plan = json.loads(plan_path.read_text())
    plan_hash = sha(plan_path)
    assert plan['arms'] == list(ARMS) and plan['seeds'] == list(SEEDS)
    assert set(plan['tasks']) == {f'{s}_held{h}' for s in SOURCES for h in (0, 1)}
    for name, expected in plan['code_hashes'].items():
        assert sha(checked_relative(name)) == expected, ('Changed locked code', name)
    for name, expected in plan.get('input_hashes', {}).items():
        assert sha(checked_relative(name)) == expected, ('Changed input', name)
    imported_path = ROOT / 'historical' / 'IMPORTED_A.json'
    assert sha(imported_path) == plan['imported_a_sha256'], 'Changed historical A manifest'
    imported = json.loads(imported_path.read_text())
    for name, expected in imported['files'].items():
        assert sha(checked_relative(name)) == expected, ('Changed historical A file', name)
    assert sha(ROOT / 'BOUNDARY_CHECKS.json') == plan['checks_sha256']
    feasible_path = ROOT / 'FEASIBILITY.json'
    feasible = json.loads(feasible_path.read_text())
    assert feasible['proceed_to_outer'] and feasible['plan_sha256'] == plan_hash
    for source, expected in feasible['selection_sha256'].items():
        assert sha(ROOT / 'selections' / f'{source}.json') == expected
    locks = {}
    for source, path in lock_paths.items():
        lock = json.loads(path.read_text())
        assert lock['plan_sha256'] == plan_hash, source
        assert lock['feasibility_sha256'] == sha(feasible_path), source
        assert set(lock['tasks']) == {f'{source}_held{h}' for h in (0, 1)}, source
        for task, entry in lock['tasks'].items():
            assert entry['path'] == f'predictions/{task}.npz'
            assert set(entry['fits']) == {f'{a}_seed{s}' for a in ARMS for s in SEEDS}
            path = checked_relative(entry['path'])
            if path.exists():
                assert sha(path) == entry['sha256'], ('Changed existing prediction', task)
            for fit in entry['fits'].values():
                directory = checked_relative(fit['outer_fit'])
                assert sha(directory / 'weights.pt') == fit['weights_sha256'], str(directory)
                assert sha(directory / 'normalization.npz') == fit['normalization_sha256'], str(directory)
        locks[source] = lock
    return plan, locks


def write_checked(path, payload, expected, *, json_payload=False):
    """Regenerate even existing files, verify bytes, preserve existing files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert sha(path) == expected, ('Changed existing file', str(path))
    handle, temporary_name = tempfile.mkstemp(prefix='.rebuild_', suffix=path.suffix, dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        if json_payload:
            dump(temporary, payload)
        else:
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **payload)
        actual = sha(temporary)
        assert actual == expected, ('Reconstructed bytes differ', str(path), expected, actual)
        if not path.exists():
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def frozen_predict(features, fit_directory, arm):
    """The locked trainer's own normalization and inference, without fitting."""
    assert not any(k.startswith('reference') for k in features)
    normalization = (GateNormalizer if arm == 'base_context' else BoundaryNormalizer).from_file(fit_directory / 'normalization.npz')
    saved = torch.load(fit_directory / 'weights.pt', map_location='cpu', weights_only=True)
    model = ReliabilityGate(**saved['config']).double()
    model.load_state_dict(saved['state_dict'])
    model.eval()
    context = normalization.tensors(features, 'neural_no_extra_geometry') if arm == 'base_context' else normalization.tensors(features, arm, 10)
    return predict(model, context)


def rebuild():
    plan, locks = prediction_barrier()
    checked = {}
    input_hashes = {}
    for source in SOURCES:
        input_path = ROOT / 'inputs' / f'{source}_neural_inputs.npz'
        input_hashes[str(input_path.relative_to(ROOT))] = sha(input_path)
        full = load(input_path)
        for half in (0, 1):
            task = f'{source}_held{half}'
            metadata, entry = plan['tasks'][task], locks[source]['tasks'][task]
            calibration = subset(full, np.flatnonzero(full['targethalf'] != half))
            held = subset(full, np.flatnonzero(full['targethalf'] == half))
            features = {k: v for k, v in held.items() if not k.startswith('reference')}
            evaluation = {k: held[k] for k in ('reference', 'reference_replicates', 'ownmask', 'query_ids', 'target_ids', 'query_block')}
            view = ROOT / 'views' / source / f'held{half}'
            write_checked(view / 'calibration.npz', calibration, metadata['calibration_sha256'])
            write_checked(view / 'held_features.npz', features, metadata['held_features_sha256'])
            write_checked(view / 'INNER_SPLIT.json', inner_ids({source: calibration}, 2026111516),
                          metadata['inner_split_sha256'], json_payload=True)
            write_checked(ROOT / 'evaluation' / source / f'held{half}.npz', evaluation, metadata['evaluation_sha256'])
            scores = np.empty((len(ARMS), len(SEEDS)) + features['base'].shape)
            for ai, arm in enumerate(ARMS):
                for si, seed in enumerate(SEEDS):
                    fit = entry['fits'][f'{arm}_seed{seed}']
                    scores[ai, si] = frozen_predict(features, checked_relative(fit['outer_fit']), arm)
            historical_entry = json.loads((ROOT / 'historical' / 'IMPORTED_A.json').read_text())['tasks'][task]
            historical = load(checked_relative(historical_entry['path']))
            assert np.array_equal(scores[0], historical['scores']), ('Imported A replay differs', task)
            path = checked_relative(entry['path'])
            existing_difference = None
            if path.exists():
                existing_difference = float(np.max(np.abs(scores - load(path)['scores'])))
                assert existing_difference == 0., ('Frozen score replay differs', task, existing_difference)
            payload = dict(scores=scores, arm_names=np.array(ARMS), seeds=np.array(SEEDS),
                           **{k: features[k] for k in ('base', 'raw', 'euclidean', 'ownmask', 'query_ids', 'query_block', 'target_ids')})
            write_checked(path, payload, entry['sha256'])
            checked[task] = dict(prediction_sha256=sha(path), prediction_bytes_match_lock=True,
                                 existing_score_max_absolute_difference=existing_difference,
                                 calibration_held_evaluation_and_split_hashes_match=True)
    report = dict(passed=True, training_performed=False, all_source_locks_verified_before_input_load=True,
                  plan_sha256=sha(ROOT / 'PLAN_LOCK.json'), input_sha256=input_hashes,
                  torch_version=str(torch.__version__), historical_A_replayed_exactly=True, tasks=checked)
    dump(ROOT / 'REBUILD_CHECK.json', report)
    return report


if __name__ == '__main__':
    report = rebuild()
    print(json.dumps(dict(passed=report['passed'], tasks=len(report['tasks']),
                          message='All reconstructed views, splits and predictions match locked SHA-256 bytes.')))
