"""Draw-index memoization only; all RNG and reduction operations are unchanged.
Paired block bootstrap over query blocks, exactly as the Stage 6, 7 and 8 evaluators ran it:
the per-query difference averaged over the two directions, 30 blocks resampled with replacement,
2,000 resamples, numpy default_rng(2026), percentile intervals."""
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=64)
def _draws(dtype,shape,raw,B,seed):
    blocks=np.frombuffer(raw,dtype=np.dtype(dtype)).reshape(shape)
    rng=np.random.default_rng(seed);bl=np.unique(blocks)
    mem={b:np.flatnonzero(blocks==b) for b in bl}
    return tuple(np.concatenate([mem[b] for b in rng.choice(bl,len(bl),replace=True)]) for _ in range(B))


def block_bootstrap(pa, pb, blocks, B=2000, seed=2026, margin=None):
    d = (pa - pb).mean(axis=1)
    blocks=np.ascontiguousarray(blocks)
    draws=_draws(blocks.dtype.str,blocks.shape,blocks.tobytes(),B,seed)
    bs = np.array([d[ix].mean() for ix in draws])
    out = {"diff": float(d.mean()),
           "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           "lower_one_sided_95": float(np.percentile(bs, 5)),
           "p_one_sided": float(np.mean(bs <= 0))}
    if margin is not None:
        out["p_le_margin"] = float(np.mean(bs <= margin))
    return out
