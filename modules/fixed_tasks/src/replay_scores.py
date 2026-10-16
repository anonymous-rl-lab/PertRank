#!/usr/bin/env python3
"""Reconstruct exact top-25 rankings, paired scores and frozen result summaries.

The compact input retains raw float64 effects on the union of every frozen arm's
positive/negative top-25 boundary (including all ties). Omitted entries cannot
enter any archived top-25 list. This representation is lossless for the fixed
arms and K=25, not for arbitrary new weights, queries or effect analyses.

Membership vectors are restored on the FULL original candidate axis before
multiplication and summation. This preserves the archived floating-point
summation order as well as the exact independent boundary-tie expectation.
"""
from pathlib import Path
import argparse
import json
import sys
import time
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'vendor'))
import exact_score as exact
import frozen_summary as frozen

PANELS = (('responsive','responsive_positions','blocks_responsive'),
          ('terciles','tercile_positions','blocks_terciles'))
K = 25
_CACHE = {}


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def signed_membership(ids, values, n_candidates, k=K):
    """Dense [positive/negative, original candidate] top-K probabilities."""
    ids = np.asarray(ids, dtype=int)
    values = np.asarray(values, dtype=np.float64)
    if len(ids) != len(values) or len(ids) < k:
        raise ValueError('incomplete retained row')
    if np.any(ids < 0) or np.any(ids >= n_candidates) or np.any(np.diff(ids) <= 0):
        raise ValueError('candidate indices must be unique, sorted and in range')
    if not np.isfinite(values).all():
        raise ValueError('nonfinite retained effect')
    signed = np.full((2, n_candidates), -np.inf, dtype=np.float64)
    signed[0, ids] = values
    signed[1, ids] = -values
    kth = np.partition(signed, n_candidates-k, axis=1)[:, -k, None]
    if not np.isfinite(kth).all():
        raise ValueError('fewer than K retained eligible candidates')
    above, tied = signed > kth, signed == kth
    slots = k-above.sum(1)
    probability = above.astype(np.float64) + tied*(slots/tied.sum(1))[:, None]
    if not np.allclose(probability.sum(1), k, rtol=0, atol=1e-12):
        raise AssertionError('invalid probability sum')
    return probability


def truth_membership(data):
    """Return [direction, query, candidate] deep-reference memberships."""
    nq = len(data['query_ids'])
    nt = len(data['target_ids'])
    offsets = data['truth_offsets']
    if len(offsets) != nq+1 or offsets[0] != 0:
        raise ValueError('invalid truth row pointers')
    result = np.empty((2,nq,nt), dtype=np.float64)
    for q in range(nq):
        sl = slice(int(offsets[q]), int(offsets[q+1]))
        ids = data['truth_ids'][sl]
        if 'own_index' in data and int(data['own_index'][q]) in ids:
            raise AssertionError('self-target retained in reference')
        result[:,q] = signed_membership(ids, data['truth_values'][sl], nt)
    return result


def cached_bootstrap(a, b, blocks, seed=2026):
    """Frozen paired query-block bootstrap with cached resampling indices.

    The cache removes redundant RNG/index construction. For every contrast,
    the same default_rng(seed), rng.choice, concatenation and d[index].mean
    operations are used as in the original exact.query_bootstrap function.
    """
    d = np.asarray(a)-np.asarray(b)
    if d.ndim != 2 or d.shape[1] != 2:
        raise ValueError('query bootstrap expects [query,two directions]')
    d = d.mean(1)
    blocks = np.asarray(blocks)
    if len(blocks) != len(d):
        raise ValueError('query/block length mismatch')
    labels = np.unique(blocks)
    key = (blocks.dtype.str, tuple(blocks.tolist()), int(seed))
    first = key not in _CACHE
    if first:
        members = {v:np.flatnonzero(blocks==v) for v in labels}
        rng = np.random.default_rng(seed)
        _CACHE[key] = tuple(np.concatenate([members[v] for v in rng.choice(labels,len(labels),replace=True)])
                            for _ in range(exact.BOOTSTRAPS))
    draws = np.array([d[index].mean() for index in _CACHE[key]])
    nonpositive = int(np.count_nonzero(draws<=0))
    result = {'difference':float(d.mean()),
              'conditional_query_block_95_interval':np.quantile(draws,[.025,.975]).tolist(),
              'bootstrap_nonpositive_count':nonpositive,
              'bootstrap_fraction_nonpositive':nonpositive/exact.BOOTSTRAPS,
              'draws':exact.BOOTSTRAPS,'seed':seed,'query_blocks':len(labels),
              'scope':'conditional on the realized input cell groups, fixed controls, query panel, candidate pool and reference; not a population or cell-sampling confidence interval'}
    if first:
        if result != exact.query_bootstrap(a,b,blocks,seed=seed):
            raise AssertionError('cached resampling differs from frozen scorer')
    return result


def evaluate_donor(data, weights=None):
    """Return (panel summaries, per-query arrays, D4 decision or None).

    Only weights identical to the archived frozen arm values are supported by
    the sparse representation. New method development requires full raw data.
    """
    archived = np.asarray(data['weights'])
    weights = archived if weights is None else np.asarray(weights)
    if not np.array_equal(weights, archived):
        raise ValueError('new weights are outside the compact support guarantee')
    arms = list(np.asarray(data['arm_names']).astype(str))
    nq, nt = len(data['query_ids']), len(data['target_ids'])
    if weights.shape != (len(arms),5,nt):
        raise ValueError('invalid weight axes')
    if not np.isfinite(weights).all() or np.any((weights<0)|(weights>1)):
        raise ValueError('invalid scalar weights')
    if not np.array_equal(weights[arms.index('raw')],np.ones((5,nt))):
        raise AssertionError('raw weight differs from one')
    offsets = data['Y_offsets']
    if len(offsets) != 5*nq+1 or offsets[0] != 0:
        raise ValueError('invalid input row pointers')
    reference = truth_membership(data)
    all_scores = np.empty((len(arms),5,nq,2), dtype=np.float64)
    for group in range(5):
        for q in range(nq):
            row = group*nq+q
            sl = slice(int(offsets[row]),int(offsets[row+1]))
            ids = data['Y_ids'][sl]
            values = data['Y_values'][sl]
            if 'own_index' in data and int(data['own_index'][q]) in ids:
                raise AssertionError('self-target retained in prediction')
            for a in range(len(arms)):
                p = signed_membership(ids, values*weights[a,group,ids], nt)
                all_scores[a,group,q] = (p*reference[:,q]).sum(1)/K
    summaries, arrays = {}, {'arm_names':np.asarray(arms)}
    for panel, pk, bk in PANELS:
        pos = np.asarray(data[pk],dtype=int)
        blocks = np.asarray(data[bk])
        if len(pos) != 300 or len(np.unique(pos)) != 300 or len(blocks) != 300:
            raise ValueError('invalid frozen query panel')
        scores = np.ascontiguousarray(all_scores[:,:,pos])
        summaries[panel] = frozen.summarize(scores,arms,blocks,SimpleNamespace(query_bootstrap=cached_bootstrap))
        arrays[panel+'_scores'] = scores
        arrays[panel+'_query_ids'] = data['query_ids'][pos]
        arrays[panel+'_query_blocks'] = blocks
    decision = None
    donor = str(np.asarray(data['donor']).item()) if 'donor' in data else ''
    if donor.startswith('D4'):
        competitor = 'PR_M'
        zero = np.flatnonzero(np.all(weights[arms.index('PertRank')]==0,axis=1)).tolist()
        decision = frozen.adjudicate(summaries['responsive'],competitor,zero)
    return summaries, arrays, decision


def compare_required(actual, expected, path='', counters=None):
    """Require exact equality for every selected archived result field."""
    if counters is None:
        counters = {'scalar_fields':0,'numeric_fields':0,'maximum_absolute_error':0.0}
    if isinstance(expected,dict):
        for k,v in expected.items():
            if k not in actual:
                raise AssertionError(('missing result field',path,k))
            compare_required(actual[k],v,path+'/'+k,counters)
    elif isinstance(expected,list):
        if len(actual) != len(expected):
            raise AssertionError(('result length mismatch',path))
        for i,(x,y) in enumerate(zip(actual,expected)):
            compare_required(x,y,path+'/'+str(i),counters)
    else:
        counters['scalar_fields'] += 1
        if isinstance(expected,(int,float)) and not isinstance(expected,bool):
            counters['numeric_fields'] += 1
            counters['maximum_absolute_error'] = max(counters['maximum_absolute_error'],abs(float(actual)-float(expected)))
        if actual != expected:
            raise AssertionError(('archived result mismatch',path,actual,expected))
    return counters


def run(root, weights_by_donor=None):
    """Replay four donors; returns a JSON report and recomputed score arrays."""
    root = Path(root)
    started = time.monotonic()
    report = {'status':'RUNNING','scope':'fixed candidate pools, panels, controls, effects and archived method fits; exact ranking and conditional query-block inference replay','donors':{}}
    produced = {}
    for donor in ('D1','D2','D3','D4'):
        data = load_npz(root/'data'/donor/'ranking.npz')
        official = json.loads((root/'reference_results'/(donor+'.json')).read_text())
        reference = load_npz(root/'reference_results'/(donor+'_query_scores.npz'))
        summaries, arrays, decision = evaluate_donor(data, None if weights_by_donor is None else weights_by_donor[donor])
        checks = {'score_values':0,'per_query_scores_exact':True,'summary_checks':{}}
        for panel,_,_ in PANELS:
            got,want = arrays[panel+'_scores'],reference[panel+'_scores']
            if not np.array_equal(got,want):
                delta = float(np.max(np.abs(got-want)))
                raise AssertionError((donor,panel,'per-query scores differ',delta))
            checks['score_values'] += int(got.size)
            expected = {key:official['panels'][panel][key] for key in ('repetitions','repetition_average')}
            checks['summary_checks'][panel] = compare_required(summaries[panel],expected,donor+'/'+panel)
            for key in (panel+'_query_ids',panel+'_query_blocks'):
                if key in reference and not np.array_equal(arrays[key],reference[key]):
                    raise AssertionError((donor,key,'context mismatch'))
        if decision is not None:
            compare_required(decision,official['decision'],donor+'/decision')
            checks['prospective_decision_exact'] = True
        checks['arms'] = list(data['arm_names'].astype(str))
        checks['groups'] = 5
        checks['queries_in_union'] = len(data['query_ids'])
        checks['candidates'] = len(data['target_ids'])
        checks['recomputed_panels'] = summaries
        if decision is not None:
            checks['recomputed_decision'] = decision
        report['donors'][donor] = checks
        produced[donor] = arrays
        print('EXACT_SCORE_REPLAY_PASS',donor,flush=True)
    report.update(status='FOUR_DONOR_EXACT_SCORE_REPLAY_PASS',elapsed_seconds=time.monotonic()-started,
                  bootstrap_draws=2000,bootstrap_seed=2026,interval='two-sided 95% paired query-block percentile; average five groups within each query and direction first')
    return report,produced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=HERE.parent)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('refusing to overwrite an existing verification report')
    report,_ = run(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
