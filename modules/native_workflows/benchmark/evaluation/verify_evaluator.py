#!/usr/bin/env python3
"""Scientific checks without opening any new FR effects or protected donor."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import evaluate_six_arm as e


def simple_membership(x, k, excluded):
    result = np.zeros(len(x))
    valid = np.delete(np.arange(len(x)), excluded)
    y = x[valid]
    cutoff = sorted(y)[-k]
    above, tied = valid[y > cutoff], valid[y == cutoff]
    result[above] = 1
    result[tied] = (k-len(above))/len(tied)
    return result


def synthetic():
    rng = np.random.default_rng(312)
    q, t, r = 9, 35, 2
    z = dict(raw_scores=rng.integers(-2, 3, size=(r,q,t)).astype(float),
             fr_scores=rng.normal(size=(r,q,t)), reference_scores=rng.integers(-2,3,size=(q,t)).astype(float),
             w0=rng.random((r,t)), wm=rng.random((r,t)),
             queries=np.array(['g'+str(x) for x in range(q)]),
             candidates=np.array(['g'+str(x) for x in range(t)]),
             blocks=np.array([0,0,1,2,2,2,3,4,4]))
    _, arrays, _ = e.score_arrays(z, draws=300)
    arms = e.effects_for_arms(z)
    for k in (25,10):
        check = np.zeros((r,6,q))
        for ri in range(r):
            for ai in range(6):
                for qi in range(q):
                    check[ri,ai,qi] = sum(np.dot(simple_membership(s*arms[ri,ai,qi],k,qi),
                        simple_membership(s*z['reference_scores'][qi],k,qi)) for s in (1,-1))/(2*k)
        np.testing.assert_allclose(arrays[f'six_arm_p{k}'], check, atol=1e-12, rtol=0)
        np.testing.assert_allclose(arrays[f'query_only_bootstrap_p{k}'], arrays['query_weights']@check.mean(0).T, atol=1e-12, rtol=0)
    for name in ['raw_scores','fr_scores','reference_scores']: z[name][:] = 0
    _, arrays, _ = e.score_arrays(z, draws=10)
    for k in (25,10): np.testing.assert_allclose(arrays[f'six_arm_p{k}'], k/(t-1), atol=1e-15, rtol=0)
    return {'status':'PASS', 'checks':['mixed exact ties', 'both signs', 'own-target exclusion',
        'two input reps', 'unequal query blocks', 'K25/K10', 'all-zero analytical chance']}


def historical():
    begin = time.monotonic()
    base = e.ROOT/'independent/execution/D4_Stim8hr'
    z, reference = e.assemble_independent(base/'prediction.npz', base/'reference.npz')
    z['fr_scores'] = z['raw_scores'].copy()  # No new FR score is opened.
    e.validate_compact(z, 'D4_Stim8hr', 'independent')
    result, arrays, _ = e.score_arrays(z, reference=reference)
    previous = e.load_npz(base/'evaluation/evaluation_arrays.npz')
    checks = {'status':'PASS', 'new_FR_evaluated':False}
    for k, pk, jk in [(25,'primary_scores_by_query','joint_primary_scores'), (10,'K10_scores_by_query','joint_K10_scores')]:
        checks[f'K{k}_points_max_error'] = float(np.max(np.abs(arrays[f'six_arm_p{k}'][0,:3]-previous[pk])))
        checks[f'K{k}_joint_max_error'] = float(np.max(np.abs(arrays[f'joint_bootstrap_p{k}'][:,:3]-previous[jk])))
        assert checks[f'K{k}_points_max_error'] < 1e-12
        assert checks[f'K{k}_joint_max_error'] < 1e-12
    checks['query_draws_identical'] = bool(np.array_equal(arrays['sampled_query_blocks'],previous['sampled_blocks']))
    checks['query_weights_identical'] = bool(np.array_equal(arrays['query_weights'],previous['query_weights']))
    assert checks['query_draws_identical'] and checks['query_weights_identical']
    checks['elapsed_seconds'] = time.monotonic()-begin
    return checks


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-baseline',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result = historical() if a.old_baseline else synthetic()
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
