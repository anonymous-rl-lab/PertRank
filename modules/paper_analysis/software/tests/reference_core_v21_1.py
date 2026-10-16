"""Fixed k=5, beta=0.9 PertRank; no query selection or evaluation data input.

The equations follow stage3/read_e4_w1.py and evaluate_w1_e4.py in the
research archive. This module accepts only the cells already selected as
inputs. It does not infer experimental strata or recreate source cohorts.
"""
import hashlib
import numpy as np

ARMS = ("Unweighted", "Euclidean", "PertRank")
K_DIRECTIONS, BETA = 5, 0.9


def _strings(values, n, name):
    a = np.asarray(values)
    if a.ndim != 1 or len(a) != n:
        raise ValueError(f"{name} must have length {n}")
    a = a.astype(str)
    if any(not s or s != s.strip() or s.lower() in ("none", "nan", "<na>") for s in a):
        raise ValueError(f"{name} contains a missing or whitespace-padded identifier")
    return a


def _matrix(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("X must be a nonempty finite dense cell-by-gene matrix")
    return x


def _controls(values, n):
    a = np.asarray(values)
    if a.shape != (n,) or not np.isin(a, [False, True, 0, 1]).all():
        raise ValueError("is_control must be a Boolean or 0/1 vector")
    return a.astype(bool)


def normalize_counts(counts, library_totals, batch_ids, is_control):
    """log1p(count * median matched-input-control total / full-library total).

    library_totals MUST be totals before weight-gene filtering. Control
    strata must be the intended normalization strata; all input controls
    are used and no 2,000-control subset is chosen here. Zero-total cells
    are rejected instead of silently modifying their denominator.
    """
    x = _matrix(counts)
    totals = np.asarray(library_totals, dtype=np.float64)
    batch = _strings(batch_ids, len(x), "batch_ids")
    ctrl = _controls(is_control, len(x))
    if (x < 0).any() or totals.shape != (len(x),) or not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("Counts must be nonnegative and full-library totals finite and positive")
    if np.any(x.sum(1) > totals + 1e-7 * np.maximum(1, totals)):
        raise ValueError("Selected-gene counts exceed the supplied full-library total")
    output = np.empty_like(x)
    scales = {}
    for b in np.unique(batch):
        take = batch == b
        if not np.any(take & ctrl):
            raise ValueError(f"No input controls in normalization stratum {b}")
        scale = float(np.median(totals[take & ctrl]))
        output[take] = np.log1p(x[take] * (scale / totals[take])[:, None])
        scales[str(b)] = scale
    return output, scales


def reliability_weights(energy, noise_trace, direction_energy, direction_noise,
                        beta=BETA, k=K_DIRECTIONS):
    """Evaluate the paper formula from sums, not means, over the gene axis.

    Direction arrays are target by direction. Projected sample variances
    include each target's own cell variance/B and matched controls/n_C.
    beta/k are exposed for algebraic verification; fit_cells always uses
    the fixed paper values and performs no parameter selection.
    """
    e, v = np.asarray(energy, float), np.asarray(noise_trace, float)
    pe, pv = np.asarray(direction_energy, float), np.asarray(direction_noise, float)
    if e.ndim != 1 or v.shape != e.shape or pe.ndim != 2 or pv.shape != pe.shape or pe.shape[0] != len(e):
        raise ValueError("Expected target energies and target-by-direction statistics")
    if not isinstance(k, (int, np.integer)) or not 0 <= k <= pe.shape[1] or not 0 <= beta < 1:
        raise ValueError("Require an available integer k and 0 <= beta < 1")
    if not all(np.isfinite(a).all() and (a >= 0).all() for a in (e, v, pe, pv)):
        raise ValueError("All sufficient statistics must be finite and nonnegative")
    se, sv = pe[:, :k].sum(1), pv[:, :k].sum(1)
    if np.any(se > e + 1e-10*np.maximum(1, e)) or np.any(sv > v + 1e-10*np.maximum(1, v)):
        raise ValueError("Projected contributions exceed total; check axes and orthogonality")
    def weight(a, n):
        ratio = np.zeros_like(a)
        np.divide(n, a, out=ratio, where=a > 0)
        return np.where(a > 0, np.clip(1-ratio, 0, 1), 0)
    return {"Unweighted": np.ones_like(e), "Euclidean": weight(e, v),
            "PertRank": weight(np.maximum(e-beta*se, 0), np.maximum(v-beta*sv, 0))}


def _directions(centered):
    """Leading exact covariance eigenvectors via the smaller Gram matrix."""
    n, g = centered.shape
    if min(n-1, g) < K_DIRECTIONS:
        raise ValueError("Fixed k=5 requires at least six controls and five weight genes per stratum")
    gram = centered @ centered.T if n < g else centered.T @ centered
    ev, vec = np.linalg.eigh(gram / (n-1))
    order = np.argsort(ev)[::-1][:K_DIRECTIONS]
    lam = ev[order]
    if lam[-1] <= 1e-12 * max(float(lam[0]), np.finfo(float).tiny):
        raise ValueError("Control covariance has fewer than five resolved positive directions")
    u = centered.T @ vec[:, order] if n < g else vec[:, order]
    u /= np.linalg.norm(u, axis=0)
    # Signs do not affect the scalar; canonical signs simplify saved files.
    signs = np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(K_DIRECTIONS)])
    u *= signs
    if not np.allclose(u.T @ u, np.eye(K_DIRECTIONS), atol=1e-8, rtol=0):
        raise ValueError("Computed control directions are not orthonormal")
    lam = np.var(centered @ u, axis=0, ddof=1)
    return u, lam


def fit_cells(X, gene_ids, target_ids, batch_ids, is_control, target_gene_ids):
    """Compute weights and original-coordinate effects from input cells.

    target_gene_ids is a per-cell identifier in the same namespace as
    gene_ids. Control rows may use any nonempty placeholder. A target
    must belong to one control-matching stratum; multi-stratum pooling
    needs an explicit experimental design and is rejected here.
    """
    x = _matrix(X)
    n, g = x.shape
    genes = _strings(gene_ids, g, "gene_ids")
    targets = _strings(target_ids, n, "target_ids")
    batch = _strings(batch_ids, n, "batch_ids")
    own = _strings(target_gene_ids, n, "target_gene_ids")
    ctrl = _controls(is_control, n)
    if len(set(genes)) != len(genes):
        raise ValueError("gene_ids must be unique and use one declared namespace")
    names = np.unique(targets[~ctrl])
    if len(names) == 0:
        raise ValueError("At least one perturbation target is required")
    if set(names) & set(targets[ctrl]):
        raise ValueError("A target_id cannot label both controls and perturbations")
    stats, models = [], {}
    for b in np.unique(batch[~ctrl]):
        xc = x[ctrl & (batch == b)]
        if len(xc) < 6:
            raise ValueError(f"Stratum {b} needs at least six matched input controls")
        cm = xc.mean(0)
        u, lam = _directions(xc-cm)
        models[str(b)] = dict(directions=u, eigenvalues=lam, mean=cm,
                              variance=xc.var(0, ddof=1), n_control=len(xc))
    for name in names:
        rows = (~ctrl) & (targets == name)
        bs, gs = np.unique(batch[rows]), np.unique(own[rows])
        if len(bs) != 1:
            raise ValueError(f"Target {name} spans strata; supply a prespecified pooling design")
        if len(gs) != 1:
            raise ValueError(f"Target {name} has inconsistent target_gene_ids")
        if rows.sum() < 2:
            raise ValueError(f"Target {name} needs at least two input cells for sample variance")
        m = models[str(bs[0])]
        xt = x[rows]
        y = xt.mean(0)-m["mean"]
        var = xt.var(0, ddof=1)
        py = y @ m["directions"]
        pv = (xt @ m["directions"]).var(0, ddof=1)
        b = len(xt)
        stats.append((y, float(y@y), float(var.sum()/b+m["variance"].sum()/m["n_control"]),
                      py*py, pv/b+m["eigenvalues"]/m["n_control"], b, str(bs[0]), str(gs[0])))
    result = dict(gene_ids=genes, target_ids=names, effects=np.array([s[0] for s in stats]),
                  energy=np.array([s[1] for s in stats]), noise_trace=np.array([s[2] for s in stats]),
                  direction_energy=np.array([s[3] for s in stats]), direction_noise=np.array([s[4] for s in stats]),
                  n_cells=np.array([s[5] for s in stats]), batch_ids=np.array([s[6] for s in stats]),
                  target_gene_ids=np.array([s[7] for s in stats]), control_models=models)
    result["weights"] = reliability_weights(result["energy"], result["noise_trace"],
                                            result["direction_energy"], result["direction_noise"])
    return result


def _boundary_probabilities(v, k):
    threshold = np.partition(v, len(v)-k)[len(v)-k]
    above, tied = v > threshold, v == threshold
    p = above.astype(float)
    p[tied] = (k-int(above.sum())) / int(tied.sum())
    return p


def nominate(result, query_ids, top_k=25, seed=2026):
    """Return lists and tie-expectation inclusion probabilities, no truth.

    Stable SHA-256 keys provide reproducible delivered shortlists. The
    probabilities separately implement independent boundary-tie selection;
    they are not calibrated probabilities of biological correctness.
    """
    if not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    queries = _strings(query_ids, len(query_ids), "query_ids")
    if len(queries) == 0:
        raise ValueError("At least one supplied query is required")
    if len(set(queries)) != len(queries):
        raise ValueError("query_ids must be unique")
    gmap = {g: i for i, g in enumerate(result["gene_ids"])}
    missing = [q for q in queries if q not in gmap]
    if missing:
        raise ValueError(f"Queries not present on the input gene axis: {missing[:5]}")
    nt = len(result["target_ids"])
    probabilities = np.zeros((len(ARMS), len(queries), 2, nt))
    records = []
    for qi, q in enumerate(queries):
        keep = np.flatnonzero(result["target_gene_ids"] != q)
        if len(keep) < top_k:
            raise ValueError(f"Query {q}: fewer than top_k candidates after own-target exclusion")
        for di, sign in enumerate((1, -1)):
            direction = "up" if sign == 1 else "down"
            keys = np.array([hashlib.sha256(f"{seed}|{q}|{direction}|{result['target_ids'][j]}".encode()).hexdigest()
                             for j in keep])
            for ai, arm in enumerate(ARMS):
                score = sign * result["effects"][keep, gmap[q]] * result["weights"][arm][keep]
                p = _boundary_probabilities(score, top_k)
                probabilities[ai, qi, di, keep] = p
                selected = np.lexsort((keys, -score))[:top_k]
                for rank, z in enumerate(selected, 1):
                    j = keep[z]
                    records.append(dict(method=arm, query_id=str(q), direction=direction, rank=rank,
                                        target_id=str(result["target_ids"][j]),
                                        target_gene_id=str(result["target_gene_ids"][j]),
                                        effect=float(result["effects"][j, gmap[q]]),
                                        weight=float(result["weights"][arm][j]), signed_score=float(score[z]),
                                        tie_inclusion_probability=float(p[z])))
    return records, probabilities
