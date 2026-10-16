#!/usr/bin/env python3
"""Execution-only fresh replay of the already frozen D3 B200 calculation.

No scientific source, seed, number of draws, input, or scoring rule is changed.
Preparation uses the standard library only. Full replay needs numpy/scipy.
"""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import zipfile

PREFIX = 'restored/working_v9_with_v6_inputs/code/repro_v10/'
EXPECTED = ('refit_D3_results.json', 'refit_D3_draws.npz',
            'refit_D3_weight_diagnostics.json', 'refit_D3_weight_diagnostics.npz')
ATOL = 1e-12
RTOL = 1e-10


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def safe_member(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or str(p) != name:
        raise ValueError('Unsafe archive member: ' + name)
    return p


def copy_checked(z, name, record, dest):
    safe_member(name)
    if z.namelist().count(name) != 1:
        raise ValueError('Missing or duplicate ZIP member: ' + name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with z.open(name) as source, dest.open('xb') as out:
        shutil.copyfileobj(source, out, length=1024 * 1024)
    got = {'bytes': dest.stat().st_size, 'sha256': sha(dest)}
    if got != {'bytes': record['bytes'], 'sha256': record['sha256']}:
        raise ValueError('Package hash/size mismatch: ' + name)
    return got


def prepare(args, work):
    # All outputs go into a new directory; original packages remain read-only.
    if work.exists() and (not work.is_dir() or any(work.iterdir())):
        raise ValueError('--workdir must be nonexistent or completely empty')
    if work.is_symlink():
        raise ValueError('--workdir must not be a symlink')
    work.mkdir(parents=True, exist_ok=True)
    audit = work / 'replay_audit'
    audit.mkdir()
    copied = {}
    with zipfile.ZipFile(args.shared_package) as shared:
        manifest = json.loads(shared.read('SHARED_INPUTS_MANIFEST.json'))
        if len(manifest) != 453 or any(not n.startswith(PREFIX) for n in manifest):
            raise ValueError('Expected the frozen 453-file shared reproduction tree')
        for name, rec in sorted(manifest.items()):
            copied[name] = copy_checked(shared, name, rec, work / safe_member(name))
        write_json(audit / 'shared_inputs_manifest.json', manifest)
    with zipfile.ZipFile(args.refit_package) as refit:
        manifest = json.loads(refit.read('MANIFEST.json'))
        prefix = 'strengthening_neighbor/'
        freeze = json.loads(refit.read(prefix + 'REFIT_BOOTSTRAP_FREEZE.json'))
        needed = set(freeze['files']) | {'REFIT_BOOTSTRAP_FREEZE.json',
                    'weight_refit_diagnostics.py', 'd3_prepared/d3_fixed_control_gram.npz'}
        for relative in sorted(needed):
            name = prefix + relative
            copied[name] = copy_checked(refit, name, manifest[name], work / safe_member(name))
        for name in EXPECTED:
            member = prefix + name
            dest = work / 'expected' / name
            copied[str(dest.relative_to(work))] = copy_checked(refit, member, manifest[member], dest)
        for name, digest in freeze['files'].items():
            if sha(work / prefix / name) != digest:
                raise ValueError('Scientific freeze mismatch: ' + name)
        result = json.loads((work / 'expected' / 'refit_D3_results.json').read_text())
        if result['B'] != 200 or result['smoke'] is not False:
            raise ValueError('Expected a full B200 archive, not a smoke run')
        if result['freeze'] != freeze:
            raise ValueError('Archived result belongs to another scientific freeze')
        if result['gram_sha256'] != sha(work / prefix / 'd3_prepared/d3_fixed_control_gram.npz'):
            raise ValueError('Gram checkpoint does not match the archived run')
        for relative in needed:
            if relative.endswith('.py'):
                ast.parse((work / prefix / relative).read_text(), filename=relative)
    required = ['same_budget_weight/stage3/a6_common.py',
                'same_budget_weight/stage3/evaluate_w1_e4.py',
                'same_budget_weight/stage3/evaluate_d2_prospective.py',
                'same_budget_weight/stage2/evaluate_stage2.py',
                'same_budget_weight/stage3/data/e4_w1_stage3.npz',
                'same_budget_weight/stage3/data/b3_D3_query_cells.npz']
    for rel in required:
        if not (work / PREFIX / rel).is_file():
            raise ValueError('Missing required original path: ' + rel)
    report = {
        'status': 'PREPARED_ONLY_NOT_REPLAYED',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'identity': 'execution-only wrapper; original scientific code and B200 unchanged',
        'packages': {n: {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                     for n, p in [('refit', args.refit_package), ('shared', args.shared_package)]},
        'shared_file_count': 453, 'all_copied_files': copied,
        'archived_B': result['B'], 'numerical_comparison_atol': ATOL,
        'numerical_comparison_rtol': RTOL, 'commands': [],
        'interpretation': 'conditional empirical resampling with fixed input NTC mean, covariance directions and deep reference; weights refitted; not a joint/population confidence interval',
    }
    write_json(audit / 'replay_summary.json', report)
    return audit, report


def compare_outputs(work):
    import numpy as np
    files = {}
    for name in EXPECTED:
        want = work / 'expected' / name
        got = work / 'strengthening_neighbor' / name
        item = {'archived_sha256': sha(want), 'replayed_sha256': sha(got),
                'numeric_values': 0, 'nonexact_numeric_values': 0,
                'max_abs_difference': 0.0, 'max_relative_difference': 0.0,
                'mismatches': [], 'numeric_field_differences': []}

        def mismatch(path, reason):
            item['mismatches'].append({'path': path, 'reason': reason})

        def nums(path, old, new):
            a, b = np.asarray(old), np.asarray(new)
            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                mismatch(path, 'non-finite numeric value'); return
            d = np.abs(b.astype(float) - a.astype(float))
            r = d / np.maximum(np.abs(a.astype(float)), np.finfo(float).tiny)
            if not np.all(np.isfinite(r)):
                r = np.minimum(r, np.finfo(float).max)
            count = int(np.count_nonzero(d))
            md = float(d.max(initial=0)); mr = float(r.max(initial=0))
            item['numeric_values'] += int(a.size)
            item['nonexact_numeric_values'] += count
            item['max_abs_difference'] = max(item['max_abs_difference'], md)
            item['max_relative_difference'] = max(item['max_relative_difference'], mr)
            if count:
                row = {'path': path, 'count': count, 'max_abs_difference': md,
                       'max_relative_difference': mr}
                if a.size == 1:
                    row.update(archived=float(a), replayed=float(b), signed_difference=float(b)-float(a))
                item['numeric_field_differences'].append(row)
            if not np.all(d <= ATOL + RTOL * np.abs(a)):
                mismatch(path, 'numeric difference exceeds declared tolerance')

        def walk(path, a, b):
            if isinstance(a, dict) and isinstance(b, dict):
                if a.keys() != b.keys():
                    mismatch(path, 'dictionary keys differ')
                for k in sorted(a.keys() & b.keys()):
                    walk(path + '/' + k, a[k], b[k])
            elif isinstance(a, list) and isinstance(b, list):
                if len(a) != len(b): mismatch(path, 'list lengths differ')
                for i, (x, y) in enumerate(zip(a, b)): walk(path + '/' + str(i), x, y)
            elif isinstance(a, bool) or isinstance(b, bool):
                if type(a) is not type(b) or a != b: mismatch(path, 'boolean differs')
            elif isinstance(a, int) or isinstance(b, int):
                if type(a) is not type(b) or a != b: mismatch(path, 'integer count or type differs')
                if isinstance(a, (int, float)) and isinstance(b, (int, float)): nums(path, a, b)
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                nums(path, a, b)
            elif type(a) is not type(b) or a != b:
                mismatch(path, 'type or nonnumeric value differs')

        if name.endswith('.json'):
            walk('$', json.loads(want.read_text()), json.loads(got.read_text()))
        else:
            with np.load(want, allow_pickle=False) as a, np.load(got, allow_pickle=False) as b:
                if set(a.files) != set(b.files): mismatch('$', 'NPZ keys differ')
                for k in sorted(set(a.files) & set(b.files)):
                    x, y = a[k], b[k]
                    if x.shape != y.shape or x.dtype != y.dtype:
                        mismatch(k, 'shape or dtype differs'); continue
                    if np.issubdtype(x.dtype, np.number):
                        nums(k, x, y)
                        if np.issubdtype(x.dtype, np.integer) and not np.array_equal(x, y):
                            mismatch(k, 'integer array differs')
                    elif not np.array_equal(x, y): mismatch(k, 'nonnumeric array differs')
        item['within_tolerance'] = not item['mismatches']
        item['numerically_exact'] = not item['mismatches'] and item['nonexact_numeric_values'] == 0
        files[name] = item
    return files


def run(args, work, audit, report):
    env = os.environ.copy()
    for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[v] = '1'
    commands = [
        ('environment', [args.python, '-c',
          'import json,sys,platform,numpy,scipy; print(json.dumps(dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=numpy.__version__,scipy=scipy.__version__),indent=2))']),
        ('refit_B200', [args.python, str(work / 'strengthening_neighbor/refit_cell_bootstrap.py'),
                       '--gram', str(work / 'strengthening_neighbor/d3_prepared/d3_fixed_control_gram.npz')]),
        ('weight_diagnostics_B200', [args.python, str(work / 'strengthening_neighbor/weight_refit_diagnostics.py')]),
    ]
    report['thread_environment'] = {k: env[k] for k in env if k.endswith('_NUM_THREADS') or k == 'VECLIB_MAXIMUM_THREADS'}
    report['status'] = 'RUNNING'
    write_json(audit / 'replay_summary.json', report)
    for label, cmd in commands:
        start = time.monotonic()
        print('Running:', label, flush=True)
        with (audit / (label + '.log')).open('w') as log:
            completed = subprocess.run(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
        report['commands'].append({'name': label, 'argv': cmd, 'cwd': str(work),
                                   'returncode': completed.returncode,
                                   'seconds': time.monotonic() - start})
        write_json(audit / 'replay_summary.json', report)
        if completed.returncode:
            raise RuntimeError(label + ' failed; see replay_audit/' + label + '.log')
    comparison = compare_outputs(work)
    write_json(audit / 'field_comparison.json', comparison)
    report['all_fields_within_tolerance'] = all(v['within_tolerance'] for v in comparison.values())
    report['all_fields_numerically_exact'] = all(v['numerically_exact'] for v in comparison.values())
    if not report['all_fields_within_tolerance']:
        raise RuntimeError('Output mismatch; see replay_audit/field_comparison.json')
    report['status'] = 'REPLAY_PASS_EXACT' if report['all_fields_numerically_exact'] else 'REPLAY_PASS_WITHIN_TOLERANCE'
    write_json(audit / 'replay_summary.json', report)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--refit-package', type=Path, required=True)
    ap.add_argument('--shared-package', type=Path, required=True)
    ap.add_argument('--workdir', type=Path, required=True)
    ap.add_argument('--prepare-only', action='store_true', help='verify/copy inputs only; no B200 and no diagnostics')
    ap.add_argument('--python', default=sys.executable, help='Python with numpy/scipy; use this interpreter to launch wrapper too')
    args = ap.parse_args()
    args.refit_package = args.refit_package.resolve(strict=True)
    args.shared_package = args.shared_package.resolve(strict=True)
    # Check symlink before resolve so it cannot redirect an intended new directory.
    if args.workdir.is_symlink(): raise ValueError('--workdir must not be a symlink')
    work = args.workdir.resolve()
    audit = None; report = None
    try:
        audit, report = prepare(args, work)
        if not args.prepare_only:
            run(args, work, audit, report)
        print(report['status'] + ': ' + str(audit / 'replay_summary.json'), flush=True)
    except Exception as exc:
        if report is not None:
            report['status'] = 'FAILED'
            report['error'] = str(exc)
            write_json(audit / 'replay_summary.json', report)
        raise


if __name__ == '__main__':
    main()
