#!/usr/bin/env python3
"""Restore exact retained inputs; replay checkpoints and original statistics.

No fitting, model selection or retuning is performed. The scientific scripts
under experiment/ are unchanged. Private provenance paths are anonymized;
release inventories track the documented metadata changes.
"""
from pathlib import Path, PurePosixPath
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def numeric(value, prefix=''):
    if isinstance(value, bool):
        return {}
    if isinstance(value, (float, int)):
        return {prefix: value}
    if isinstance(value, dict):
        return {k2: v2 for k, v in value.items() for k2, v2 in numeric(v, prefix+'/'+k).items()}
    if isinstance(value, list):
        return {k2: v2 for k, v in enumerate(value) for k2, v2 in numeric(v, prefix+'/'+str(k)).items()}
    return {}

def verify_distribution():
    manifest = read(ROOT/'DISTRIBUTION_SHA256.json')
    for name, digest in manifest.items():
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('Unsafe distribution path')
        if sha(ROOT/name) != digest:
            raise ValueError('Distribution checksum mismatch: '+name)
    return len(manifest)

def runtime_info():
    expected = {'numpy': '2.3.5', 'torch': '2.15.0+cpu'}
    found = {}
    for package, version in expected.items():
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError('Missing '+package+'. Install requirements-q.txt as described in README.md.') from exc
        if found[package] != version:
            raise RuntimeError(f'Exact replay uses {package}=={version}; found {found[package]}. See README.md.')
    found['python'] = sys.version
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use Python 3.12 for the recorded exact-replay environment.')
    return found

def command(argv, log):
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run([sys.executable, *map(str, argv)], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    Path(log).write_text(result.stdout)
    if result.returncode:
        raise RuntimeError('Replay command failed; see '+str(log)+'\n'+result.stdout[-4000:])

def restore(destination, logdir):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Preserve existing workspace; choose a new --extract-to path.')
    shutil.copytree(ROOT/'experiment', destination)
    inventory = read(ROOT/'RESTORATION_MANIFEST.json')
    allowed = set(inventory['stored_binary_files'])
    seen = set()
    with tarfile.open(ROOT/'assets.tar.xz', 'r:xz') as archive:
        for member in archive:
            p = PurePosixPath(member.name)
            if (not member.isfile() or p.is_absolute() or '..' in p.parts
                    or member.name not in allowed or member.name in seen):
                raise ValueError('Unsafe or unexpected asset member: '+member.name)
            target = destination.joinpath(*p.parts)
            if target.exists():
                raise ValueError('Asset would overwrite existing source: '+member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('wb') as sink:
                shutil.copyfileobj(source, sink)
            seen.add(member.name)
    if seen != allowed:
        raise ValueError('Binary asset membership differs from the restoration manifest')
    command([ROOT/'restore_historical.py', '--base-root', destination/'baseline_B'],
            logdir/'restore_historical.log')
    for name, digest in inventory['all_release_files'].items():
        if sha(destination/name) != digest:
            raise ValueError('Restored logical file differs from release: '+name)
    return {'all_release_files_checked': len(inventory['all_release_files']),
            'historical_prediction_packets_reconstructed': len(inventory['regenerated_files']),
            'all_release_sha256_match': True,
            'no_precision_reduction': True}

def replay(work, out, skip_sensitivity):
    reports = {}
    stages = [('stage1', work/'replay_optimization.py', 'REPLAY_OPTIMIZATION_CHECK.json'),
              ('stage2', work/'epoch_pair_update/replay_pair_update.py', 'REPLAY_PAIR_CHECK.json')]
    for stage, script, filename in stages:
        print('Replaying '+stage+' from frozen checkpoints...', flush=True)
        command([script, '--output-root', out/stage], out/(stage+'.log'))
        reports[stage] = read(out/stage/filename)
        if not reports[stage].get('passed'):
            raise RuntimeError(stage+' did not pass')
    if not skip_sensitivity:
        print('Replaying the 100 fixed-reference sensitivity cases...', flush=True)
        command([work/'audit/reference_sensitivity.py', '--rebuild',
                 '--experiment-root', work, '--base-root', work/'baseline_B',
                 '--output-root', out/'reference_sensitivity'], out/'reference_sensitivity.log')
        expected = read(work/'audit/reference_sensitivity/REFERENCE_SENSITIVITY_RESULTS.json')
        actual = read(out/'reference_sensitivity/REFERENCE_SENSITIVITY_RESULTS.json')
        e, a = numeric(expected), numeric(actual)
        if e.keys() != a.keys():
            raise ValueError('Reference-sensitivity numeric fields differ')
        maximum = max((abs(a[k]-v) for k,v in e.items()), default=0)
        if maximum != 0:
            raise ValueError('Reference-sensitivity numbers differ: '+str(maximum))
        array_match = sha(out/'reference_sensitivity/REFERENCE_CASES.npz') == expected['arrays_sha256']
        if not array_match:
            raise ValueError('Reference-sensitivity array checksum differs')
        reports['reference_sensitivity'] = {'passed': True, 'numeric_fields_compared': len(e),
                                           'maximum_numeric_difference': maximum,
                                           'scored_array_sha256_match': True, 'cases': 100}
    else:
        reports['reference_sensitivity'] = {'executed': False, 'reason': 'Explicit --skip-sensitivity'}
        command([work/'baseline_B/rebuild_boundary.py'], out/'baseline_for_selection_check.log')
    print('Checking all 300 archived inner epoch-pair alternatives...', flush=True)
    command([work/'epoch_pair_update/check_pair_selection.py', '--base-root', work/'baseline_B',
             '--require-selections', '--output', out/'SELECTION_REPLAY_CHECK.json'],
            out/'selection_check.log')
    reports['selection_verification'] = read(out/'SELECTION_REPLAY_CHECK.json')
    if not reports['selection_verification']['passed']:
        raise ValueError('Archived selection verification failed')
    summary = read(work/'epoch_pair_update/results/SUMMARY.json')
    qname = 'B_rank_scale_free_pair'
    paper_fields = 0
    with (ROOT/'paper_data/pertrank_q_scores.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            value = summary['sources'][row['source']]
            contrast = value['contrasts']['original_PertRank'][qname]
            pairs = {'original_PertRank': value['means']['original_PertRank'],
                     'PertRank_Q': value['means'][qname], 'delta_P_at_10': contrast['delta'],
                     'conditional95_low': contrast['conditional95'][0],
                     'conditional95_high': contrast['conditional95'][1],
                     'n_queries': value['queries'], 'query_blocks': value['query_blocks'], 'K': summary['K']}
            for key, expected in pairs.items():
                if float(row[key]) != expected:
                    raise ValueError('Paper source data differ: '+row['source']+'/'+key)
                paper_fields += 1
    reports['paper_source_data'] = {'passed': True, 'fields_compared': paper_fields,
                                   'maximum_numeric_difference': 0}
    results = {'method': 'PertRank-Q', 'archived_arm': qname,
               'identity': 'Phase-2 prespecified secondary development arm',
               'equal_source_aggregate': summary['equal_source_aggregate'][qname],
               'source_results': {s: {'precision':v['means'][qname],
                  'contrasts': {c: cv[qname] for c, cv in v['contrasts'].items()}}
                  for s,v in summary['sources'].items()},
               'primary_gates_unchanged': summary['primary_gates']}
    write(out/'PERT_RANK_Q_RESULTS.json', results)
    return reports, results

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--verify-only', action='store_true', help='Distribution hashes only; not numerical reproduction')
    parser.add_argument('--extract-to', type=Path, help='Restore exact logical experiment files; do not fit models')
    parser.add_argument('--skip-sensitivity', action='store_true', help='Explicitly omit only the reference-repeat sensitivity replay')
    args = parser.parse_args()
    checked = verify_distribution()
    if args.verify_only:
        print(json.dumps({'distribution_files_checked': checked, 'passed': True, 'numerical_replay_performed': False}))
        return
    runtime = runtime_info()
    if args.output_dir is None:
        parser.error('--output-dir is required for numerical replay or extraction')
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Choose a new or empty --output-dir; existing results are preserved')
    out.mkdir(parents=True, exist_ok=True)
    report = {'passed': False, 'training_performed': False, 'model_selection_changed': False,
              'archived_selection_recomputed_for_verification': True,
              'runtime': runtime, 'distribution_files_checked': checked}
    start = time.monotonic()
    try:
        if args.extract_to:
            report['restoration'] = restore(args.extract_to.resolve(), out)
            report.update(passed=True, numerical_replay_performed=False,
                          restored_experiment=str(args.extract_to.resolve()))
        else:
            with tempfile.TemporaryDirectory(prefix='pertrank_q_exact_') as temp:
                work = Path(temp)/'base_B2'
                report['restoration'] = restore(work, out)
                report['replay'], report['q_result'] = replay(work, out, args.skip_sensitivity)
            report.update(passed=True, numerical_replay_performed=True,
                          temporary_workspace_removed=True)
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-start
        write(out/'REPRODUCTION_REPORT.json', report)
    print(json.dumps({'passed': report['passed'], 'report': str(out/'REPRODUCTION_REPORT.json'),
                      'wall_seconds': report['wall_seconds']}, indent=2))

if __name__ == '__main__':
    main()
