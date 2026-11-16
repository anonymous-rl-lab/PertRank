#!/usr/bin/env python3
"""Restore six omitted historical packets exactly from retained frozen weights.

This is a packaging adapter, not a replacement for locked experiment code.
No model is fitted, selected or rescored. All original inputs, scientific
modules, historical fit files and regenerated packets are SHA-256 checked.
Existing packets are checked and preserved; mismatches fail without overwrite.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile

SOURCES = ('D3', 'D1', 'D2')
SEEDS = (2026111511, 2026111512)
PAYLOAD_KEYS = ('base', 'raw', 'euclidean', 'ownmask', 'query_ids', 'query_block', 'target_ids')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_path(root, relative):
    value = Path(relative)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError('Unsafe package-relative path: ' + str(relative))
    path = root / value
    if not path.resolve().is_relative_to(root):
        raise ValueError('Path escapes the baseline root: ' + str(relative))
    return path


def require_hash(path, expected):
    if not path.is_file() or digest(path) != expected:
        raise RuntimeError('Original SHA-256 mismatch: ' + str(path))


def restore(base_root):
    root = Path(base_root).resolve()
    plan_path = root / 'PLAN_LOCK.json'
    plan = json.loads(plan_path.read_text())
    imported_path = root / 'historical' / 'IMPORTED_A.json'
    require_hash(imported_path, plan['imported_a_sha256'])
    imported = json.loads(imported_path.read_text())
    expected_tasks = {f'{s}_held{h}' for s in SOURCES for h in (0, 1)}
    if set(imported['tasks']) != expected_tasks:
        raise RuntimeError('Unexpected historical task set')
    missing = []
    for task, record in imported['tasks'].items():
        path = checked_path(root, record['path'])
        if path.exists():
            require_hash(path, record['sha256'])
        else:
            missing.append(task)
    for relative, expected in plan['code_hashes'].items():
        require_hash(checked_path(root, relative), expected)
    for relative, expected in plan['input_hashes'].items():
        require_hash(checked_path(root, relative), expected)
    # Verify all historical fitted assets. The only permitted absent imported
    # files are precisely the six reconstructible prediction packets.
    prediction_names = {v['path'] for v in imported['tasks'].values()}
    for relative, expected in imported['files'].items():
        path = checked_path(root, relative)
        if relative not in prediction_names or path.exists():
            require_hash(path, expected)
    for record in imported['tasks'].values():
        if set(record['fits']) != {f'base_context_seed{s}' for s in SEEDS}:
            raise RuntimeError('Unexpected historical fit set')
        for fit in record['fits'].values():
            directory = checked_path(root, fit['outer_fit'])
            require_hash(directory / 'weights.pt', fit['weights_sha256'])
            require_hash(directory / 'normalization.npz', fit['normalization_sha256'])

    # Import the exact retained modules only after checking their hashes.
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    common = importlib.import_module('common')
    base_training = importlib.import_module('base_training')
    gate_model = importlib.import_module('gate_model')
    for module in (common, base_training, gate_model):
        if Path(module.__file__).resolve().parent != root:
            raise RuntimeError('A scientific module was imported from another directory')

    packets = {}
    with tempfile.TemporaryDirectory(prefix='pertrank_historical_restore_') as td:
        temporary_root = Path(td)
        # Generate and hash all missing files before installing any of them.
        for source in SOURCES:
            full = common.load(root / 'inputs' / f'{source}_neural_inputs.npz')
            for half in (0, 1):
                task = f'{source}_held{half}'
                record = imported['tasks'][task]
                destination = checked_path(root, record['path'])
                if task not in missing:
                    packets[task] = {'path': record['path'], 'sha256': record['sha256'],
                                     'status': 'existing_verified', 'bytes': destination.stat().st_size}
                    continue
                held = common.subset(full, np.flatnonzero(full['targethalf'] == half))
                features = {k: v for k, v in held.items() if not k.startswith('reference')}
                scores = np.empty((len(SEEDS),) + features['base'].shape)
                for index, seed in enumerate(SEEDS):
                    fit = record['fits'][f'base_context_seed{seed}']
                    directory = checked_path(root, fit['outer_fit'])
                    normalization = base_training.GateNormalizer.from_file(directory / 'normalization.npz')
                    saved = torch.load(directory / 'weights.pt', map_location='cpu', weights_only=True)
                    model = gate_model.ReliabilityGate(**saved['config']).double()
                    model.load_state_dict(saved['state_dict'])
                    model.eval()
                    context = normalization.tensors(features, 'neural_no_extra_geometry')
                    scores[index] = base_training.predict(model, context)
                # Original packet member order and NumPy dtypes are preserved.
                payload = dict(scores=scores, seeds=np.array(SEEDS),
                               **{k: features[k] for k in PAYLOAD_KEYS})
                temporary = temporary_root / f'{task}.npz'
                np.savez_compressed(temporary, **payload)
                require_hash(temporary, record['sha256'])
                packets[task] = {'path': record['path'], 'sha256': record['sha256'],
                                 'status': 'regenerated_exactly', 'bytes': temporary.stat().st_size}
        for task in missing:
            record = imported['tasks'][task]
            destination = checked_path(root, record['path'])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                require_hash(destination, record['sha256'])
                packets[task]['status'] = 'existing_verified'
                continue
            content = (temporary_root / f'{task}.npz').read_bytes()
            try:
                with destination.open('xb') as stream:
                    stream.write(content)
            except FileExistsError:
                require_hash(destination, record['sha256'])
                packets[task]['status'] = 'existing_verified'
            require_hash(destination, record['sha256'])
    return {'passed': True, 'training_performed': False, 'model_selection_performed': False,
            'all_original_sha256_match': True, 'packets': packets,
            'restored_packet_count': sum(v['status'] == 'regenerated_exactly' for v in packets.values()),
            'numpy_version': np.__version__, 'torch_version': str(torch.__version__),
            'source_plan_sha256': digest(plan_path), 'adapter_sha256': digest(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', type=Path, required=True, help='Restored baseline_B directory')
    parser.add_argument('--report', type=Path, help='Optional new JSON report path; existing files are never replaced')
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError('Report exists; choose a new path')
    try:
        report = restore(args.base_root)
    except Exception as error:
        report = {'passed': False, 'training_performed': False,
                  'model_selection_performed': False, 'error': repr(error)}
        print(json.dumps(report, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open('x') as stream:
                json.dump(report, stream, indent=2)
                stream.write('\n')
        raise
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
