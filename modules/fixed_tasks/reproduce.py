#!/usr/bin/env python3
"""One-command, offline, exact replay of D1-D4 and the fixed Stim experiment."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse, csv, hashlib, json, platform, socket, sys, time, subprocess
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from reproduce_development_weights import reproduce_weights
from reproduce_d4_weights import reproduce as reproduce_d4
import replay_scores, replay_depth, replay_supplements, replay_stim


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest():
    path = ROOT / 'MANIFEST_SHA256.json'
    manifest = json.loads(path.read_text())
    expected = manifest['files']
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()
              and not any(x in ('.git','__pycache__') for x in p.relative_to(ROOT).parts)
              and p.name != 'MANIFEST_SHA256.json' and p.suffix != '.pyc'}
    if actual != set(expected):
        raise AssertionError({'missing': sorted(set(expected)-actual), 'extra': sorted(actual-set(expected))})
    for rel, digest in expected.items():
        target = (ROOT / rel).resolve()
        if not target.is_relative_to(ROOT) or sha(target) != digest:
            raise AssertionError('Manifest mismatch: ' + rel)
    return {'status':'PASS', 'files':len(expected), 'manifest_sha256':sha(path)}


def deny_network(*args, **kwargs):
    raise RuntimeError('Network access is disabled during this offline reproduction')


def block_network():
    original = socket.socket
    class OfflineSocket(original):
        connect = deny_network
        connect_ex = deny_network
    socket.socket = OfflineSocket
    socket.create_connection = deny_network
    socket.getaddrinfo = deny_network


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def replay_competition(arrays):
    path = ROOT / 'provenance/code/v18_d4_confirmation/competition_selection/COMPETITION_SELECTION_LOCK.json'
    lock = json.loads(path.read_text())
    donors = ['D3','D1','D2']; donor_means = {}
    for donor in donors:
        a = arrays[donor]['responsive_scores']
        means = a.mean(axis=1).mean(axis=2).mean(axis=1)
        names = list(arrays[donor]['arm_names'].astype(str))
        donor_means[donor] = dict(zip(names, map(float,means)))
        archived = lock['donor_results'][donor + '_Rest']['means']
        if donor_means[donor] != archived:
            raise AssertionError(('competitor selection donor mean mismatch', donor))
    pooled = {name:float(np.mean([donor_means[d][name] for d in donors])) for name in names}
    if pooled != lock['equal_donor_means']:
        raise AssertionError('competitor equal-donor means differ')
    eligible = lock['eligible_arms']; top = max(pooled[a] for a in eligible)
    tied = sorted(a for a in eligible if top-pooled[a] <= lock['tie_absolute_tolerance'])
    if tied != lock['tie_set'] or tied[0] != lock['selected_competitor']:
        raise AssertionError('selected competitor differs')
    return {'status':'PASS','selected_competitor':tied[0],'equal_donor_means':pooled,
            'D1_panel':'corrected full-axis','new_selection':False,'historical_lock_sha256':sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT.parent/'PertRank_reproduced')
    parser.add_argument('--external-benchmark-only', action='store_true', help='Replay only the v1.4.0 external benchmark supplement')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError('Choose an output directory outside the immutable repository')
    if out.exists():
        raise FileExistsError('Use a new output directory; existing results will not be overwritten: '+str(out))
    start = time.monotonic()
    manifest = verify_manifest()
    if args.external_benchmark_only:
        subprocess.run([sys.executable, str(ROOT/'supplements/20261014_external_benchmark/reproduce.py'), '--output-dir', str(out)], check=True)
        return
    block_network()
    out.mkdir(parents=True)
    print('MANIFEST_PASS', manifest['files'], 'files; network blocked',flush=True)
    weights={}; weight_audits={}
    for donor in ('D1','D2','D3'):
        weights[donor],weight_audits[donor] = reproduce_weights(ROOT/'data'/donor/'weight_stats.npz')
    weights['D4'],weight_audits['D4'] = reproduce_d4(ROOT)
    for donor in weights:
        with np.load(ROOT/'data'/donor/'ranking.npz',allow_pickle=False) as z:
            if not np.array_equal(weights[donor],z['weights']):
                raise AssertionError(('reconstructed weights differ from sparse support lock',donor))
        print('WEIGHT_PASS',donor,flush=True)
    scores,arrays = replay_scores.run(ROOT,weights_by_donor=weights)
    selection = replay_competition(arrays)
    print('COMPETITOR_SELECTION_PASS',selection['selected_competitor'],flush=True)
    depth={}
    for donor in ('D1','D2','D3','D4'):
        depth[donor]=replay_depth.replay_donor(ROOT,donor)
        print('DEPTH_PASS',donor,depth[donor]['verification']['max_absolute_difference'],flush=True)
        np.savez_compressed(out/(donor+'_recomputed_query_scores.npz'),**arrays[donor])
        save_json(out/(donor+'_recomputed_results.json'),{
            'panels':scores['donors'][donor]['recomputed_panels'],
            'decision':scores['donors'][donor].get('recomputed_decision'),
            'depth':depth[donor]['recomputed'],
            'depth_panel_scope':depth[donor]['verification']['panel_scope'],
        })
    save_json(out/'weights_verification.json',weight_audits)
    save_json(out/'competition_selection_replay.json',selection)
    save_json(out/'depth_replay.json',depth)
    save_json(out/'scores_replay.json',scores)
    with (out/'responsive_arm_means.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['donor','arm','P_at_25','expected_members_per_25'])
        for donor in ('D1','D2','D3','D4'):
            means=scores['donors'][donor]['recomputed_panels']['responsive']['repetition_average']['means']
            for arm,value in means.items():writer.writerow([donor,arm,value,25*value])
    summary_fields=sum(z['numeric_fields'] for d in scores['donors'].values() for z in d['summary_checks'].values())
    supplements = replay_supplements.run(ROOT, out/'supplements')
    stim = replay_stim.run(ROOT, out/'stim')
    subprocess.run([sys.executable, str(ROOT/'supplements/20261014_upgrade/reproduce.py'), '--output-dir', str(out/'upgrade')], check=True)
    upgrade = json.loads((out/'upgrade/SUPPLEMENT_REPRODUCTION.json').read_text())
    subprocess.run([sys.executable, str(ROOT/'supplements/20261014_external_benchmark/reproduce.py'), '--output-dir', str(out/'external_benchmark')], check=True)
    external_benchmark = json.loads((out/'external_benchmark/BENCHMARK_REPRODUCTION.json').read_text())
    report={
        'status':'FOUR_DONOR_STIM_UPGRADE_AND_EXTERNAL_BENCHMARK_REPRODUCTION_PASS',
        'package_version':'1.4.0',
        'original_four_donor_status':'FOUR_DONOR_EXACT_REPRODUCTION_PASS',
        'identity':'post-result exact fixed-method sufficient-data reproduction; not a new experiment or preregistration',
        'environment':{'python':platform.python_version(),'numpy':np.__version__},
        'network_access':False,'network_blocked':True,'manifest':manifest,
        'weight_reproduction':weight_audits,
        'per_query_score_values':sum(d['score_values'] for d in scores['donors'].values()),
        'per_query_scores_exact':True,'summary_numeric_fields':summary_fields,
        'maximum_summary_difference':max(z['maximum_absolute_error'] for d in scores['donors'].values() for z in d['summary_checks'].values()),
        'depth_verification':{d:v['verification'] for d,v in depth.items()},
        'competitor_selection':selection,
        'D4_decision':scores['donors']['D4']['recomputed_decision'],
        'all_weights_equal_frozen_weights':True,
        'raw_UMI_reloaded':False,'control_PCA_refitted':False,'TRADE_mixture_refitted':False,
        'D1_depth_panel':'historical sensitivity; distinct from corrected primary panel',
        'supplement_reproduction':supplements,
        'stim_reproduction':stim,
        'upgrade_reproduction':upgrade,
        'external_benchmark_reproduction':external_benchmark,
        'elapsed_seconds':time.monotonic()-start,
    }
    save_json(out/'VERIFICATION.json',report)
    print(report['status'], 'seconds',round(report['elapsed_seconds'],2),flush=True)


if __name__=='__main__':
    main()
