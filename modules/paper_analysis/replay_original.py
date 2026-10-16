#!/usr/bin/env python3
"""Offline v25.2 paper-analysis replay; original frozen scripts stay unchanged."""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT = ['s3', 'biology', 'transfer', 'trade', 'application', 'figures']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equivalent(actual, expected, path='root'):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (path, 'keys')
        for key in expected:
            equivalent(actual[key], expected[key], path + '.' + str(key))
    elif isinstance(expected, list):
        assert len(actual) == len(expected), (path, 'length')
        for i, (a, b) in enumerate(zip(actual, expected)):
            equivalent(a, b, path + '[' + str(i) + ']')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


def compare_file(actual, expected):
    if actual.suffix == '.npz':
        with np.load(actual, allow_pickle=False) as a, np.load(expected, allow_pickle=False) as e:
            assert set(a.files) == set(e.files)
            for key in e.files:
                if e[key].dtype.kind in 'fc':
                    np.testing.assert_allclose(a[key], e[key], atol=1e-12, rtol=1e-12, equal_nan=True)
                else:
                    np.testing.assert_array_equal(a[key], e[key])
            return {'file': expected.name, 'arrays': len(e.files), 'status': 'PASS'}
    if actual.suffix == '.json':
        equivalent(json.loads(actual.read_text()), json.loads(expected.read_text()))
    elif actual.suffix == '.csv':
        with actual.open(newline='') as f:
            a = list(csv.reader(f))
        with expected.open(newline='') as f:
            e = list(csv.reader(f))
        assert len(a) == len(e)
        assert a[0] == e[0]
        for i, (ar, er) in enumerate(zip(a[1:], e[1:]), 1):
            assert len(ar) == len(er)
            for j, (av, ev) in enumerate(zip(ar, er)):
                try:
                    af, ef = float(av), float(ev)
                except ValueError:
                    assert av == ev, (expected.name, i, j)
                else:
                    assert np.isclose(af, ef, atol=1e-12, rtol=1e-12, equal_nan=True), (expected.name, i, j, af, ef)
    else:
        assert actual.read_bytes() == expected.read_bytes()
    return {'file': expected.name, 'status': 'PASS'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New output directory; bundled results are never overwritten.')
    parser.add_argument('--only', nargs='+', choices=DEFAULT + ['software'], default=DEFAULT)
    parser.add_argument('--software-python', default=sys.executable, help='Python in the separately installed software environment.')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Choose a new --out directory: ' + str(out))
    out.mkdir(parents=True)
    work = out / 'work'
    work.mkdir()
    logs = out / 'logs'
    logs.mkdir()
    report = {'status': 'RUNNING', 'scope': 'Existing-result replay and software contract tests; no new experiment, raw-source access or model selection.', 'modules': {}, 'python': sys.version, 'numpy': np.__version__}
    initial = {str(p.relative_to(ROOT)): digest(p) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')

    def copy(rel):
        dst = work / rel
        shutil.copytree(ROOT / rel, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        return dst

    def run(name, script, *arguments, python=None, extra_env=None):
        cmd = [python or sys.executable, str(script), *map(str, arguments)]
        started = time.monotonic()
        runenv = env.copy()
        if extra_env:
            runenv.update(extra_env)
        result = subprocess.run(cmd, cwd=work, env=runenv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (logs / (name + '.log')).write_text(result.stdout)
        assert result.returncode == 0, (name, result.returncode, str(logs / (name + '.log')), result.stdout[-2000:])
        return {'status': 'PASS', 'seconds': round(time.monotonic() - started, 3), 'log': 'logs/' + name + '.log'}

    try:
        for module in args.only:
            started = time.monotonic()
            checks = []
            runs = []
            if module == 's3':
                dst = copy('analysis_v20/s3')
                shutil.rmtree(dst / 'results')
                runs.append(run('s3_replay', dst / 'replay_s3.py', '--output-dir', dst / 'results'))
                runs.append(run('s3_independent_audit', dst / 'audit_s3.py'))
                for p in (ROOT / 'analysis_v20/s3/results').iterdir():
                    if p.suffix in ['.csv', '.npz'] or p.name == 'RESULT.json':
                        checks.append(compare_file(dst / 'results' / p.name, p))
            elif module == 'biology':
                dst = copy('analysis_v20/biology')
                runs.append(run('biology', dst / 'analyze_biology.py', '--out', dst))
                for p in (ROOT / 'analysis_v20/biology').iterdir():
                    if p.suffix == '.csv' or p.name in ['query_scores.npz', 'RESULTS.json']:
                        checks.append(compare_file(dst / p.name, p))
            elif module == 'transfer':
                dst = copy('analysis_v20/transfer')
                runs.append(run('transfer', dst / 'run_transfer.py'))
                for name in ['transfer_scores.npz', 'RESULTS.json', 'query_scores.csv', 'block_influence.csv']:
                    checks.append(compare_file(dst / name, ROOT / 'analysis_v20/transfer' / name))
            elif module == 'trade':
                dst = copy('analysis_v20/robustness')
                runs.append(run('trade_replay', dst / 'replay_trade_sensitivity.py'))
                runs.append(run('trade_independent_scores', dst / 'review_trade_scores_separate.py'))
                with gzip.open(dst / 'TRADE_SENSITIVITY_MIXTURES.json.gz', 'rt') as f:
                    mixtures = json.load(f)
                with (dst / 'TRADE_SENSITIVITY_FIT_MOMENTS.csv').open() as f:
                    moments = list(csv.DictReader(f))
                assert len(mixtures) == len(moments) == 2805
                errors = {'latent_second_moment': 0.0, 'weight': 0.0}
                for row in moments:
                    g = mixtures[row['target'] + '::' + row['replicate']]['fitted_g']
                    pi, a, b = [np.asarray(g[k], float) for k in ['pi', 'a', 'b']]
                    assert np.all(pi >= 0) and abs(pi.sum() - 1) < 1e-6
                    second = float(np.sum(pi * (a*a + a*b + b*b) / 3))
                    weight = second / (second + float(row['nu']))
                    errors['latent_second_moment'] = max(errors['latent_second_moment'], abs(second - float(row['S'])))
                    errors['weight'] = max(errors['weight'], abs(weight - float(row['weight'])))
                assert max(errors.values()) < 1e-10, errors
                checks.append({'status': 'PASS', 'fits': 2805, 'mixture_moment_max_abs_errors': errors, 'scope': 'Recomputes moments from fitted mixture parameters; does not refit R mixtures or reconstruct raw-cell standard errors.'})
                checks.append({'status': 'PRESERVED_NOT_RUN', 'analysis': 'control replacement', 'reason': 'Original asset audit reported missing raw input rows; no control-replacement ranking result exists.'})
            elif module == 'application':
                dst = copy('application')
                runs.append(run('application_counts', dst / 'verify_application.py'))
                # Recreate the exact original prediction interfaces from full query effects,
                # references and weights retained by the completed TRADE sensitivity.
                runroot = work / 'ctla4_archived_interfaces'
                (runroot / 'input').mkdir(parents=True)
                (runroot / 'evaluation').mkdir()
                with np.load(ROOT / 'analysis_v20/robustness/TRADE_SENSITIVITY_REPLAY_INPUTS.npz', allow_pickle=False) as z:
                    pred = {'original_effects': z['effects'], 'target_ids': z['target_ids'], 'query_gene_ids': z['query_gene_ids']}
                    for arm in ['raw', 'Euclidean', 'PertRank']:
                        pred['weight_' + arm] = z['weight_' + arm]
                        pred['prediction_' + arm] = z['effects'] * z['weight_' + arm][:, :, None]
                    np.savez_compressed(runroot / 'input/main_nontrade_predictions.npz', **pred)
                    np.savez_compressed(runroot / 'evaluation/main_reference.npz', effects=z['reference'], target_ids=z['target_ids'], query_gene_ids=z['query_gene_ids'])
                shutil.copy2(ROOT / 'analysis_v20/s3/inputs/original_registered_scores.npz', runroot / 'evaluation/registered_scores.npz')
                ctla4 = work / 'ctla4_rebuilt'
                runs.append(run('ctla4', dst / 'build_CTLA4_example.py', '--run-root', runroot, '--query-context', ROOT / 'analysis_v20/s3/inputs/main_query_context.npz', '--out', ctla4))
                for p in (ROOT / 'application').glob('CTLA4*.csv'):
                    checks.append(compare_file(ctla4 / p.name, p))
            elif module == 'figures':
                dst = copy('figure_code')
                copy('source_data')
                for name in ['draw_main_figures.py', 'plot_supplement.py', 'plot_design_attenuation.py']:
                    runs.append(run(name.removesuffix('.py'), dst / name))
                figures = list((work / 'manuscript/figures').glob('*'))
                assert len(list((work / 'manuscript/figures').glob('*.png'))) == 11
                assert len(list((work / 'manuscript/figures').glob('*.pdf'))) == 11
                assert all(p.stat().st_size > 1000 for p in figures)
                checks.append({'status': 'PASS', 'main_figures': 6, 'supplementary_figures': 5, 'output_files': len(figures), 'scope': 'Renders archived data; generated image bytes can differ by Matplotlib/font version.'})
            elif module == 'software':
                dst = copy('software')
                # Use an installed software environment, not the old application module.
                runs.append(run('software_identity', dst / 'scripts/check_installation.py', python=args.software_python))
                runs.append(run('software_tests', '-m', 'pytest', '-q', dst / 'tests', python=args.software_python))
                runs.append(run('software_tutorials', dst / 'scripts/check_tutorials.py', python=args.software_python))
                legacy = work / 'legacy_h5ad_application'
                shutil.copytree(ROOT / 'application', legacy, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
                runs.append(run('application_h5ad', ROOT / 'verify_legacy_h5ad.py', '--application-dir', legacy, python=args.software_python))
            report['modules'][module] = {'status': 'PASS', 'seconds': round(time.monotonic() - started, 3), 'runs': runs, 'comparisons': checks}
            print(module + ': PASS', flush=True)
        assert initial == {rel: digest(ROOT / rel) for rel in initial}, 'Bundled source changed during replay'
        report['bundled_files_unchanged'] = True
        report['status'] = 'PASS'
    except Exception as exc:
        report['status'] = 'FAIL'
        report['error'] = repr(exc)
        raise
    finally:
        (out / 'REPLAY_REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'modules': args.only, 'report': str(out / 'REPLAY_REPORT.json')}, indent=2))


if __name__ == '__main__':
    main()
