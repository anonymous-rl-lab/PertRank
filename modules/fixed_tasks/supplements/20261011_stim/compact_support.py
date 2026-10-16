"""Exact fixed-method top-25 replay from original float64 support values.

No rounding, alternative ranking, cached scores, or imputed interior effects.
All memberships are scattered to the original full candidate axis before sums.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import frozen_engine as engine
import stats_frozen_exact_score as exact

K = 25

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def array_digest(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.asarray(arrays[key])
        h.update(key.encode()); h.update(a.dtype.str.encode())
        h.update(json.dumps(a.shape).encode()); h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def reconstruct_weights(data):
    names = tuple(data['arm_names'].astype(str))
    moments = {name: data['moment_' + name] for name in
               ('log_aE', 'log_nuE', 'log_aM', 'log_nuM')}
    weights = {'raw': np.ones_like(moments['log_aE']),
               'Euclidean': engine.scalar(moments['log_aE'], moments['log_nuE']),
               'PertRank': engine.scalar(moments['log_aM'], moments['log_nuM'])}
    if 'PR_M' in names:
        weights['PR_M'] = engine.scalar(data['moment_pr_aM'], data['moment_pr_nuM'])
    assert set(weights) == set(names)
    return {name: weights[name] for name in names}

def probability_row(values):
    values = np.asarray(values, dtype=np.float64)
    return exact.probabilities(values[None, :], np.zeros((1, len(values)), bool), K)

def validate_support(data):
    tids, qids = data['target_ids'].astype(str), data['query_gene_ids'].astype(str)
    nt, nq = len(tids), len(qids)
    assert nt >= 26 and len(set(tids)) == nt and len(set(qids)) == nq
    assert int(data['k']) == K and int(data['groups']) == 5
    for prefix, rows in [('Y', 5*nq), ('truth', nq)]:
        ids, values, offsets = [data[prefix + '_' + suffix] for suffix in ('ids', 'values', 'offsets')]
        assert values.dtype == np.float64 and np.isfinite(values).all()
        assert offsets.shape == (rows+1,) and offsets[0] == 0 and offsets[-1] == len(ids) == len(values)
        assert np.all(np.diff(offsets.astype(np.int64)) >= K)
        for r in range(rows):
            selected = ids[offsets[r]:offsets[r+1]].astype(int)
            assert np.all((selected >= 0) & (selected < nt)) and np.all(np.diff(selected) > 0)
            assert np.all(tids[selected] != qids[r % nq])
    return nt, nq

def reconstruct(data, *, responsive_positions=None):
    nt, nq = validate_support(data)
    names = tuple(data['arm_names'].astype(str))
    weights = reconstruct_weights(data)
    # Original scorer receives transposed [target,query] effects. Its membership
    # arrays are Fortran-contiguous; preserve that layout and reduction order.
    truth_p = np.zeros((2*nq, nt), np.float64, order='F')
    for q in range(nq):
        lo, hi = data['truth_offsets'][q:q+2]
        ids = data['truth_ids'][lo:hi].astype(int)
        p = probability_row(data['truth_values'][lo:hi])
        truth_p[q, ids], truth_p[nq+q, ids] = p[0], p[1]
    scores = np.empty((len(names), 5, nq, 2), np.float64)
    # The original operation order is group, positive+negative sum, division,
    # then cumulative addition. Keep the complete target axis, including zeros.
    contributions = None
    if responsive_positions is not None:
        rp = np.asarray(responsive_positions, int)
        contributions = np.zeros((nt, len(rp)), np.float64)
    membership_hash = hashlib.sha256()
    for group in range(5):
        group_memberships = {}
        for name in names:
            p = np.zeros_like(truth_p)
            for q in range(nq):
                row = group*nq+q
                lo, hi = data['Y_offsets'][row:row+2]
                ids = data['Y_ids'][lo:hi].astype(int)
                values = data['Y_values'][lo:hi] * weights[name][group, ids]
                pp = probability_row(values)
                p[q, ids], p[nq+q, ids] = pp[0], pp[1]
            group_memberships[name] = p
            scores[names.index(name), group] = (p*truth_p).sum(1).reshape(2, nq).T/K
            membership_hash.update(p.tobytes())
        if contributions is not None:
            ix = np.concatenate([rp, nq+rp])
            pp, pe = group_memberships['PertRank'][ix], group_memberships['Euclidean'][ix]
            pt = truth_p[ix]
            contributions += ((pp-pe)*pt).reshape(2,len(rp),nt).sum(axis=0).T/(5*2*K)
    out = {'scores': scores, 'arm_names': names, 'weights': weights,
           'target_ids': data['target_ids'], 'query_gene_ids': data['query_gene_ids'],
           'prediction_membership_sha256': membership_hash.hexdigest(),
           'truth_membership_sha256': hashlib.sha256(truth_p.tobytes()).hexdigest()}
    if contributions is not None:
        out['alignment_contributions'] = contributions
    return out
