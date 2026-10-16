#!/usr/bin/env python3
"""Create a count-level software fixture; not biological validation data."""
import argparse
from pathlib import Path
import numpy as np


def fixture(n_control=2000):
    rng = np.random.default_rng(18012026)
    g, nt, depth = 320, 8, 20
    n = n_control + nt*depth
    loading = rng.normal(size=(5, g))*.25
    latent = rng.normal(size=(n, 5))
    log_rate = 1.0 + latent@loading
    for i in range(nt):
        rows = slice(n_control+i*depth, n_control+(i+1)*depth)
        log_rate[rows, i:i+8] += .8
        log_rate[rows, 20:40] += (i-3)*.09
    counts = rng.poisson(np.exp(np.clip(log_rate, -3, 4))).astype(np.uint16)
    target = np.array(["NTC"]*n_control + [f"target_{i:02d}" for i in range(nt) for _ in range(depth)])
    own = np.array(["NTC"]*n_control + [f"GENE_{i:03d}" for i in range(nt) for _ in range(depth)])
    return dict(X=counts, cell_ids=np.array([f"cell_{i:05d}" for i in range(n)]),
                gene_ids=np.array([f"GENE_{i:03d}" for i in range(g)]), target_ids=target,
                target_gene_ids=own, batch_ids=np.repeat("example_stratum", n),
                is_control=np.arange(n)<n_control,
                query_ids=np.array(["GENE_000", "GENE_021", "GENE_050"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **fixture())
    print(f"Created synthetic counts: {args.out}; use --top-k 3 for this eight-target example.")
