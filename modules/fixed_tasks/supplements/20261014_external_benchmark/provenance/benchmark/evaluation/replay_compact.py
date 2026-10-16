#!/usr/bin/env python3
"""Replay saved six-arm task arrays without full FR effect matrices or fitting.

This is a post-save reproduction entry point, not a route for selecting models
or bypassing the D3 winner gate during first scoring. Original immutable score
report/compact hashes must agree; the scientific scoring functions are shared
with the locked primary evaluator. Use the separate audit scorer for an
independent implementation of point/query-bootstrap checks.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import time
import numpy as np
import evaluate_six_arm as e


def replay(source, out, reference_path=None):
    source, out = Path(source), Path(out)
    report_path = source/'evaluation_report.json'
    report = json.loads(report_path.read_text())
    e.require_dataset(report['dataset'])
    if report['status'] != 'COMPLETE':
        raise ValueError('Only complete archived evaluations can be replayed')
    compact_path, arrays_path = source/'compact.npz', source/'evaluation_arrays.npz'
    if e.sha(compact_path) != report['compact_sha256'] or e.sha(arrays_path) != report['evaluation_arrays_sha256']:
        raise ValueError('Archived compact/array hashes disagree with original report')
    z, saved = e.load_npz(compact_path), e.load_npz(arrays_path)
    e.validate_compact(z, report['dataset'], report['kind'])
    reference = None
    if report['kind'] == 'independent':
        if reference_path is None:
            raise ValueError('Full independent replay requires original reference-cell NPZ')
        reference_path = Path(reference_path)
        if e.sha(reference_path) != report['provenance']['reference_sha256']:
            raise ValueError('Reference-cell archive differs from original evaluation')
        reference = e.load_npz(reference_path)
        if not np.array_equal(reference['target_ids'].astype(str), z['candidates']) or not np.array_equal(reference['query_ids'].astype(str), z['queries']):
            raise ValueError('Reference-cell identity axes mismatch')
    out.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    results, replayed, rseed = e.score_arrays(z, draws=2000, reference=reference)
    checks = {}
    if set(replayed) != set(saved):
        raise ValueError('Archived bootstrap array schema differs from replay')
    for name in sorted(saved):
        a, b = replayed[name], saved[name]
        if a.shape != b.shape:
            raise AssertionError('Replay shape mismatch: '+name)
        error = float(np.max(np.abs(a-b)))
        if error > 1e-12:
            raise AssertionError('Replay numeric mismatch: '+name+' '+str(error))
        checks[name] = {'max_abs_error':error, 'bitwise_equal':bool(np.array_equal(a,b))}
    np.savez_compressed(out/'replayed_arrays.npz', **replayed)
    result = dict(status='PASS', original_report_sha256=e.sha(report_path),
        compact_sha256=e.sha(compact_path), old_arrays_sha256=e.sha(arrays_path),
        reference_sha256=e.sha(reference_path) if reference is not None else None,
        dataset=report['dataset'], kind=report['kind'], results=results,
        reference_seed=rseed, checks=checks, elapsed_seconds=time.monotonic()-begin,
        scoring_code_sha256=e.sha(e.__file__), replay_code_sha256=e.sha(__file__),
        scope=report['scope'], reuses_saved_predictions=True, refits_native_FR=False)
    e.save_json(out/'REPLAY_REPORT.json',result)
    print(json.dumps({'status':'PASS','path':str(out/'REPLAY_REPORT.json'),'maximum_array_error':max(c['max_abs_error'] for c in checks.values())}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    a=p.parse_args()
    replay(a.source,a.out,a.reference)
