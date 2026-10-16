"""Exact R20 scoring functions; historical default seed retained, callers pass2026."""
import numpy as np
BOOTSTRAPS=2000
BOOTSTRAP_SEED=20261006

def finite(name, x):
    if not np.isfinite(x).all():
        raise ValueError(f'{name} contains nonfinite values')

def own_mask(query_ids, target_ids):
    q = np.asarray(query_ids).astype(str); t = np.asarray(target_ids).astype(str)
    if len(np.unique(q)) != len(q) or len(np.unique(t)) != len(t):
        raise ValueError('query and target identifiers must be unique on their axes')
    return q[:, None] == t[None, :]

def probabilities(effects, own, k):
    """Exact independent tie membership, signed row order + queries then - queries."""
    y = np.asarray(effects, dtype=np.float64)
    if y.ndim != 2 or own.shape != y.shape:
        raise ValueError('effects/own mask shape mismatch')
    finite('effects', y)
    if np.any(y.shape[1]-own.sum(1) < k):
        raise ValueError('fewer than K eligible candidates')
    s = np.concatenate([y, -y], axis=0)
    mask = np.concatenate([own, own], axis=0)
    s[mask] = -np.inf
    kth = np.partition(s, s.shape[1]-k, axis=1)[:, -k, None]
    above, tied = s > kth, s == kth
    slots = k-above.sum(1)
    p = above.astype(np.float64) + tied*(slots/tied.sum(1))[:, None]
    if not np.allclose(p.sum(1), k, rtol=0, atol=1e-12) or p[mask].any():
        raise AssertionError('invalid top-K membership')
    return p

def score(effects, truth, own, k):
    q = effects.shape[0]
    p = probabilities(effects, own, k)
    t = probabilities(truth, own, k)
    return (p*t).sum(1).reshape(2, q).T/k

def query_bootstrap(a, b, blocks, seed=BOOTSTRAP_SEED):
    """Conditional paired query-block interval; cell groups first averaged by caller."""
    d = np.asarray(a)-np.asarray(b)
    if d.ndim != 2 or d.shape[1] != 2:
        raise ValueError('query bootstrap expects [query, two directions]')
    d = d.mean(1)
    blocks = np.asarray(blocks)
    if len(blocks) != len(d):
        raise ValueError('query blocks length mismatch')
    labels = np.unique(blocks)
    members = {v: np.flatnonzero(blocks == v) for v in labels}
    rng = np.random.default_rng(seed)
    draws = np.array([d[np.concatenate([members[v] for v in rng.choice(labels, len(labels), replace=True)])].mean()
                      for _ in range(BOOTSTRAPS)])
    nonpositive = int(np.count_nonzero(draws <= 0))
    return {'difference': float(d.mean()), 'conditional_query_block_95_interval': np.quantile(draws, [.025, .975]).tolist(),
            'bootstrap_nonpositive_count': nonpositive, 'bootstrap_fraction_nonpositive': nonpositive/BOOTSTRAPS,
            'draws': BOOTSTRAPS, 'seed': seed, 'query_blocks': len(labels),
            'scope': 'conditional on the realized input cell groups, fixed controls, query panel, candidate pool and reference; not a population or cell-sampling confidence interval'}
