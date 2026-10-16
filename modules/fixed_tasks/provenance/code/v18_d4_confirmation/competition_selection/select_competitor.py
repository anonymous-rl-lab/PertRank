#!/usr/bin/env python3
"""Select one development competitor; never read D4 or access a network."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DONORS = ('D3_Rest', 'D1_Rest', 'D2_Rest')
ARMS = ('raw', 'Euclidean', 'PertRank', 'diagonal', 'TRADE_scalar', 'PR_E', 'PR_M')
ELIGIBLE = ('Euclidean', 'diagonal', 'TRADE_scalar', 'PR_E', 'PR_M')
TIE_ATOL = 1e-12
V17 = ROOT/'v17_experiments/evaluation/results/combined/ALL_SCORES_COMPLETE.json'
V17_HASH = '7549b7df2f07681d00d9167c68b4b9419a4b11ac76b7f92b1e9a2354ed904a60'
CORRECTION = ROOT/'v18_d4_confirmation/d1_correction'
FINAL = HERE/'COMPETITION_SELECTION_LOCK.json'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def dump_new(p, obj):
    with Path(p).open('x') as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')

def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))

def resolve_path(value, expected=None):
    p = Path(value)
    if not p.is_absolute():
        p = ROOT/p
    p = p.resolve()
    assert p.is_relative_to(ROOT), ('source outside workspace', p)
    if expected is not None:
        assert p == expected.resolve(), ('unexpected source', p, expected)
    return p

def checked(p, expected_hash, sources):
    p = Path(p).resolve()
    got = sha(p)
    assert got == expected_hash, ('hash mismatch', rel(p), got, expected_hash)
    sources[rel(p)] = got
    return p

def frozen(sources):
    p = HERE/'SELECTION_CODE_FREEZE.json'
    f = read(p)
    for path, expected in f['files'].items():
        checked(resolve_path(path), expected, sources)
    sources[rel(p)] = sha(p)
    return sha(p)

def donor_scores(donor, rp, sp, sources):
    assert donor in DONORS
    result = read(rp)
    assert result['donor'] == donor
    with np.load(sp, allow_pickle=False) as z:
        assert tuple(z['arm_names'].astype(str)) == ARMS
        a = z['responsive_scores'].copy()
    assert a.shape == (7, 5, 300, 2), (donor, a.shape)
    assert np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all()
    # Explicit hierarchy: average groups and signs within query, then queries.
    within_query = a.mean(axis=1).mean(axis=2)
    means = within_query.mean(axis=1)
    group_means = a.mean(axis=(2, 3))
    archived = result['panels']['responsive']['repetition_average']['means']
    errors = {arm: abs(float(means[i])-float(archived[arm])) for i, arm in enumerate(ARMS)}
    assert max(errors.values()) <= 1e-14, (donor, errors)
    # Reversed summation order is an independent check of the linear averaging.
    alternative = a[:, ::-1, ::-1, ::-1].sum(axis=(1, 2, 3))/(5*300*2)
    assert np.max(np.abs(alternative-means)) <= 1e-14
    sources[rel(rp)] = sha(rp)
    sources[rel(sp)] = sha(sp)
    return dict(means={arm: float(means[i]) for i, arm in enumerate(ARMS)},
                group_means={arm: group_means[i].tolist() for i, arm in enumerate(ARMS)},
                shape=list(a.shape), archived_mean_max_abs_error=max(errors.values()),
                alternate_sum_max_abs_error=float(np.max(np.abs(alternative-means))))

def load_v17(sources):
    checked(V17, V17_HASH, sources)
    z = read(V17)
    assert set(z['donors']) == set(DONORS)
    return z

def load_d1_corrected(sources):
    cp = CORRECTION/'FINAL_CORRECTION_COMPLETE.json'
    c = read(cp)
    assert c['status'] == 'PASS', 'D1 correction incomplete'
    sources[rel(cp)] = sha(cp)
    expected = {
        'result': CORRECTION/'D1_corrected_results.json',
        'query_scores': CORRECTION/'D1_corrected_query_scores.npz',
        'panel_prediction_lock': CORRECTION/'PANEL_PREDICTION_LOCK.json',
    }
    paths = {}
    for prefix, ep in expected.items():
        paths[prefix] = checked(resolve_path(c[prefix+'_path'], ep), c[prefix+'_sha256'], sources)
    ap = checked(resolve_path(c['independent_audit_path']), c['independent_audit_sha256'], sources)
    assert read(ap)['status'] == 'PASS', 'independent D1 audit did not pass'
    r = read(paths['result'])
    assert r['score_file_sha256'] == sha(paths['query_scores'])
    assert r['panel_lock_sha256'] == sha(paths['panel_prediction_lock'])
    panel_lock = read(paths['panel_prediction_lock'])
    pp = (CORRECTION/panel_lock['prediction_file']).resolve()
    assert pp.is_relative_to(CORRECTION)
    checked(pp, panel_lock['prediction_sha256'], sources)
    assert panel_lock['inputs_unchanged'] and panel_lock['all_seven_weight_arms_unchanged']
    assert panel_lock['original_panel_reproduced_exactly']
    assert panel_lock['no_D4_access']
    return paths['result'], paths['query_scores']

def aggregate(corrected):
    sources = {}
    freeze = frozen(sources)
    v17 = load_v17(sources)
    results = {}
    for donor in DONORS:
        if donor == 'D1_Rest' and corrected:
            rp, sp = load_d1_corrected(sources)
        else:
            d = v17['donors'][donor]
            rp = checked(resolve_path(d['result_path']), d['result_sha256'], sources)
            sp = checked(resolve_path(d['query_scores_path']), d['query_scores_sha256'], sources)
        results[donor] = donor_scores(donor, rp, sp, sources)
    equal = {arm: float(np.mean([results[d]['means'][arm] for d in DONORS])) for arm in ARMS}
    return dict(eligible_arms=list(ELIGIBLE), donor_order=list(DONORS), donor_weight=1/3,
                panel='responsive', queries_per_donor=300, input_groups=5, query_signs=2,
                donor_results=results, equal_donor_means=equal, sources_sha256=sources,
                selection_code_freeze_sha256=freeze, D1_panel='corrected full-axis' if corrected else 'historical replay only',
                no_D4_access=True)

def winner(means):
    assert all(np.isfinite(means[a]) for a in ELIGIBLE)
    top = max(means[a] for a in ELIGIBLE)
    tie = sorted(a for a in ELIGIBLE if top-means[a] <= TIE_ATOL)
    return tie[0], tie

def selftests():
    # The tie threshold is absolute, with lexicographic selection from the full tie set.
    z = {a: 0.2 for a in ELIGIBLE}
    assert winner(z) == ('Euclidean', sorted(ELIGIBLE))
    z['TRADE_scalar'] += 0.5e-12
    assert winner(z)[0] == 'Euclidean'
    z['TRADE_scalar'] += 2e-12
    assert winner(z) == ('TRADE_scalar', ['TRADE_scalar'])
    try:
        winner({a: (float('nan') if a == 'PR_E' else 0.1) for a in ELIGIBLE})
    except AssertionError:
        pass
    else:
        raise AssertionError('non-finite candidate not rejected')

def dryrun():
    selftests()
    r = aggregate(False)
    # Deliberately no winner or tie set for an obsolete D1 panel.
    r.update(status='PASS', identity='historical v17 averaging/interface replay only; not a selection', created_utc=stamp())
    dump_new(HERE/'V17_REPLAY_PASS.json', r)
    print('PASS: all three historical donor means reconstructed; no competitor selected.')

def final():
    selftests()
    assert not FINAL.exists(), 'final selection exists; use audit, never overwrite'
    assert read(HERE/'V17_REPLAY_PASS.json')['status'] == 'PASS'
    r = aggregate(True)
    selected, tie = winner(r['equal_donor_means'])
    r.update(status='LOCKED', identity='final corrected-panel five-group development competitor selection',
             created_utc=stamp(), selected_competitor=selected, tie_set=tie, tie_absolute_tolerance=TIE_ATOL,
             replay_record_sha256=sha(HERE/'V17_REPLAY_PASS.json'),
             selection_rule_path=rel(HERE/'SELECTION_RULE.md'), selection_rule_sha256=sha(HERE/'SELECTION_RULE.md'))
    dump_new(FINAL, r)
    print(json.dumps({'selected_competitor': selected, 'equal_donor_means': r['equal_donor_means'], 'lock_sha256': sha(FINAL)}, indent=2))

def audit():
    lock = read(FINAL)
    current = aggregate(True)
    for key, value in current.items():
        assert value == lock[key], ('selection record mismatch', key)
    selected, tie = winner(current['equal_donor_means'])
    assert selected == lock['selected_competitor'] and tie == lock['tie_set']
    assert lock['tie_absolute_tolerance'] == TIE_ATOL
    assert lock['replay_record_sha256'] == sha(HERE/'V17_REPLAY_PASS.json')
    report = dict(status='PASS', created_utc=stamp(), selected_competitor=selected,
                  selection_lock_path=rel(FINAL), selection_lock_sha256=sha(FINAL),
                  source_file_count=len(current['sources_sha256']), no_D4_access=True,
                  audit='complete source-hash replay, raw-query-score averaging, and deterministic tie-rule verification')
    dump_new(HERE/'SELECTION_AUDIT_PASS.json', report)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['dryrun', 'final', 'audit'], required=True)
    a = p.parse_args()
    {'dryrun': dryrun, 'final': final, 'audit': audit}[a.mode]()
