import numpy as np

def donor_mean_bootstrap(diff, groups, donor, B=2000, seed=2026):
    """bootstrap the equal-weight average of the three donors' means, resampling whole groups with replacement.

    The statistic is the A12 estimand, not a 110-record average: each resample recomputes every donor's own mean
    over whatever of its records were drawn, then averages the donors equally (a donor drawn empty is dropped from
    that resample, which is recorded)."""
    rng = np.random.default_rng(seed)
    gs = np.unique(groups); mem = {g: np.flatnonzero(groups == g) for g in gs}
    ds = list(dict.fromkeys(donor)); empty = 0
    stat = []
    for _ in range(B):
        ix = np.concatenate([mem[g] for g in rng.choice(gs, len(gs), replace=True)])
        per = [diff[ix][donor[ix] == d].mean() for d in ds if np.any(donor[ix] == d)]
        empty += len(ds) - len(per)
        stat.append(np.mean(per))
    s = np.asarray(stat)
    per0 = [diff[donor == d].mean() for d in ds]
    return {"diff": float(np.mean(per0)), "ci95": [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))],
            "groups": int(len(gs)), "empty_donor_draws": int(empty)}

