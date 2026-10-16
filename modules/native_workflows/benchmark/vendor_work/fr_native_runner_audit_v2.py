#!/usr/bin/env python
"""Audited adapter around the unmodified, commit-locked FR-Perturb Python core.

No reference expression is accepted. Input NPZ fields: data, indices, indptr,
shape (CSR raw integer counts), target_index (-1 NTC, 0..T-1 targets), targets,
genes; optional cell_ids. This adapter uses official Scanpy preprocessing and
official factorize_recover / scale_effs. It captures full native precision before
the CLI's three-significant-figure text export, with no algorithm replacement.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import resource
import sys
import time
import traceback
import warnings

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "deps"), str(HERE / "FR-Perturb")]
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_key, "2")

import numpy as np
import scipy.sparse as sparse
from threadpoolctl import threadpool_limits

COMMIT = "24e653b6a5c050ab17f3600ebaf222bef32e2d25"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path, value):
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def run_fr(counts, target_index, targets, genes, output_dir, *, rank=20,
           spca_alpha=0.1, lasso_alpha=0.0001, spca_iter=50,
           seed=20261012, threads=2, cell_ids=None, input_provenance=None):
    """Fit official FR with full raw gene counts and all supplied shallow cells.

    Returns paths to effects.npz and fit_report.json. All target cells and NTCs
    participate in the factorization. No deep or reference values are accepted.
    """
    import anndata
    import scanpy
    import sklearn
    import spams

    core = importlib.import_module("FR_Perturb.FR_Perturb")
    utils = importlib.import_module("FR_Perturb.utils")
    manifest = json.loads((HERE / "official_manifest.json").read_text())
    if manifest["commit"] != COMMIT:
        raise AssertionError("FR vendor manifest commit mismatch")
    for rel, expected in manifest["files"].items():
        if sha256(HERE / "FR-Perturb" / rel) != expected:
            raise AssertionError("FR vendor source was modified: " + rel)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    report = dict(status="RUNNING", official_repository="https://github.com/douglasyao/FR-Perturb",
                  official_commit=COMMIT, parameters=dict(rank=rank, spca_alpha=spca_alpha,
                  lasso_alpha=lasso_alpha, spca_iter=spca_iter, seed=seed, threads=threads),
                  preprocessing="official Scanpy filter_genes(min_cells=1), TP10K, log1p, NTC centering",
                  output_scale="official scale_effs LOWESS; preserve before signif(3)",
                  numerical_precision="native dtype, without decimal export rounding",
                  rng_note="NumPy and Python seeds set; SPAMS exposes no random seed parameter; reproducibility is tested separately",
                  dictionary_convergence="native fixed iteration budget; no native convergence flag",
                  input_provenance=input_provenance, events=[], warnings=[])
    for rel in ("run_FR_Perturb.py", "FR_Perturb/FR_Perturb.py", "FR_Perturb/utils.py", "environment.yml"):
        report.setdefault("official_source_sha256", {})[rel] = sha256(HERE / "FR-Perturb" / rel)
    report["runner_sha256"] = sha256(__file__)
    report["versions"] = {name: importlib.metadata.version(name) for name in
        ("numpy", "scipy", "pandas", "scikit-learn", "scanpy", "anndata", "spams-bin", "statsmodels", "h5py", "threadpoolctl")}
    report["python"] = sys.version
    _json(out / "fit_report.json", report)
    logger = logging.getLogger("fr_native." + str(out))
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(out / "fit.log")
    sh = logging.StreamHandler()
    for handler in (fh, sh):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)

    old_train, old_lasso_spams, old_lasso_sklearn = spams.trainDL, spams.lasso, core.Lasso
    fitted_lasso = []

    def event(name, fun, *args, **kwargs):
        begin = time.time()
        logger.info("START %s", name)
        value = fun(*args, **kwargs)
        report["events"].append(dict(stage=name, seconds=time.time()-begin))
        logger.info("FINISH %s %.3fs", name, time.time()-begin)
        _json(out / "fit_report.json", report)
        return value

    def timed_train(*args, **kwargs):
        kwargs["numThreads"] = threads  # resource limit only; native algorithm unchanged
        return event("spams_trainDL", old_train, *args, **kwargs)

    def timed_project(*args, **kwargs):
        kwargs["numThreads"] = threads
        return event("spams_lasso_projection", old_lasso_spams, *args, **kwargs)

    class CapturedLasso(old_lasso_sklearn):
        def fit(self, *args, **kwargs):
            result = event("sklearn_lasso_recovery", super().fit, *args, **kwargs)
            fitted_lasso.append(self)
            return result

    try:
        targets = np.asarray(targets).astype(str)
        genes = np.asarray(genes).astype(str)
        ti = np.asarray(target_index, dtype=np.int64)
        if cell_ids is not None:
            cell_ids = np.asarray(cell_ids).astype(str)
            if len(cell_ids) != len(ti) or len(np.unique(cell_ids)) != len(ti):
                raise ValueError("cell_ids must match rows and be unique")
        x = sparse.csr_matrix(counts, copy=True)
        if x.shape != (len(ti), len(genes)):
            raise ValueError("count axes do not match labels")
        if len(set(targets)) != len(targets) or len(set(genes)) != len(genes):
            raise ValueError("non-unique target/gene IDs")
        if (x.data < 0).any() or not np.isfinite(x.data).all() or not np.equal(x.data, np.floor(x.data)).all():
            raise ValueError("FR input must be finite non-negative raw integer counts")
        if (ti < -1).any() or (ti >= len(targets)).any():
            raise ValueError("invalid target index")
        if np.sum(ti == -1) < 100:
            raise ValueError("Require >=100 NTCs to avoid official all-cell-centering fallback")
        if not np.all(np.bincount(ti[ti >= 0], minlength=len(targets)) > 0):
            raise ValueError("every target must have supplied cells")
        totals = np.asarray(x.sum(axis=1)).ravel()
        if np.any(totals == 0):
            raise ValueError("Empty cells would change fair input population; reject rather than silently filter")
        if len(genes) <= rank or len(ti) <= rank:
            raise ValueError("rank must be smaller than both cell and gene counts")
        report["input"] = dict(n_cells=x.shape[0], n_genes=x.shape[1], n_targets=len(targets),
              n_ntc=int(np.sum(ti == -1)), nnz=x.nnz,
              cells_per_target=np.bincount(ti[ti >= 0], minlength=len(targets)).tolist(),
              full_umi_min=int(totals.min()), full_umi_max=int(totals.max()),
              counts_dtype=str(x.dtype))
        np.random.seed(seed)
        random.seed(seed)
        dat = anndata.AnnData(x)
        dat.var_names = genes
        scanpy.pp.filter_genes(dat, min_cells=1)
        retained_genes = dat.var_names.to_numpy().astype(str)
        if dat.n_vars <= rank:
            raise ValueError("rank must be smaller than input-expressed gene count")
        full_index = {g: i for i, g in enumerate(genes)}
        retained_index = np.array([full_index[g] for g in retained_genes])
        report["input"]["n_genes_retained"] = dat.n_vars
        report["input"]["removed_zero_genes"] = genes[~np.isin(np.arange(len(genes)), retained_index)].tolist()
        p_mat = sparse.csr_matrix((np.ones(len(ti), dtype=np.float64),
                    (np.arange(len(ti)), np.where(ti == -1, len(targets), ti))),
                    shape=(len(ti), len(targets)+1))
        logger.info("Official preprocessing %d cells x %d expressed genes", dat.n_obs, dat.n_vars)
        with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scanpy.pp.normalize_total(dat, target_sum=10000)
            logmeanexp = np.squeeze(np.array(np.log(np.mean(dat.X, axis=0))))
            scanpy.pp.log1p(dat)
            ctrl_exp = dat.X[ti == -1].mean(axis=0)
            # Same elementwise centering as the CLI, in-place storage only.
            # scipy sparse-minus-dense otherwise allocates two dense matrices.
            exp_mat = dat.X.toarray(order="C")
            exp_mat -= np.asarray(ctrl_exp)
            exp_mat = np.asfortranarray(exp_mat.T)
            report["input"]["normalized_dtype"] = str(exp_mat.dtype)
            report["input"]["dense_factorize_bytes"] = exp_mat.nbytes
            del dat, x
            spams.trainDL, spams.lasso, core.Lasso = timed_train, timed_project, CapturedLasso
            B, U, U_tilde, W, p_out, alpha = core.factorize_recover(
                exp_mat, p_mat, rank, spca_alpha, log=logger, alpha=lasso_alpha, iter=spca_iter)
            del exp_mat
            if not np.isfinite(B).all():
                raise FloatingPointError("Nonfinite native FR output")
            np.savez_compressed(out / "native_unscaled.npz", B=B, U=U, U_tilde=U_tilde, W=W,
                    genes=retained_genes, targets=targets, logmeanexp=logmeanexp,
                    perturbation_names=np.concatenate([targets, ['__NTC__']]),
                    ntc_mean=np.asarray(ctrl_exp).ravel(), retained_gene_index=retained_index)
            scaled, scale_factors = event("official_LOWESS_scaling", utils.scale_effs, B, logmeanexp)
            if not np.isfinite(scaled).all() or not np.all(np.isfinite(scale_factors) & (scale_factors > 0)):
                raise FloatingPointError("Official scaling failed; unscaled output preserved for diagnosis")
            report["warnings"] = [dict(category=w.category.__name__, message=str(w.message)) for w in caught]
        model = fitted_lasso[-1]
        # Independent analytic audit is valid because our benchmark design is one-hot.
        group_count = np.asarray(p_out.sum(axis=0)).ravel()
        means = np.asarray(p_out.T @ U_tilde) / group_count[:, None]
        analytic = np.sign(means) * np.maximum(np.abs(means) - alpha * p_out.shape[0] / group_count[:, None], 0)
        error = float(np.max(np.abs(U-analytic)))
        error_tolerance = 2e-5 * max(1.0, float(np.max(np.abs(analytic))))
        report["recover_audit"] = dict(analytic_one_hot_soft_threshold_max_error=error,
            analytic_audit_tolerance=error_tolerance,
            analytic_audit_pass=bool(error <= error_tolerance),
            n_iter=np.asarray(model.n_iter_).tolist(), dual_gap=np.asarray(model.dual_gap_).tolist(),
            max_iter=model.max_iter, tolerance=model.tol,
            convergence_warning=any(w["category"] == "ConvergenceWarning" for w in report["warnings"]))
        from native_recovery_audit import audit_recovery
        report['recover_audit']['legacy_coefficient_gate_is_diagnostic_only'] = True
        report['recover_audit']['acceptance_audit'] = audit_recovery(
            U, U_tilde, ti, alpha, model.dual_gap_, model.tol,
            report['recover_audit']['convergence_warning'])
        report['recovery_audit_source_sha256'] = sha256(HERE / 'native_recovery_audit.py')
        if not report['recover_audit']['acceptance_audit']['pass_']:
            raise AssertionError('Native LASSO failed native stopping and exact objective audit')
        # Positive per-gene scaling must preserve target order, apart from explicitly
        # counted floating-point merges. Check every retained gene, both signs.
        old_order = np.argsort(B[:-1], axis=0, kind="stable")
        new_order = np.argsort(scaled[:-1], axis=0, kind="stable")
        report["scaling_order_audit"] = dict(
            rank_positions_changed=int(np.count_nonzero(old_order != new_order)),
            all_target_rankings_identical=bool(np.array_equal(old_order, new_order)))
        if not np.array_equal(old_order, new_order):
            raise AssertionError("Official positive scaling unexpectedly changed native target rankings")
        report["factor_diagnostics"] = dict(W_dtype=str(W.dtype), B_dtype=str(B.dtype),
            U_dtype=str(U.dtype), U_tilde_dtype=str(U_tilde.dtype),
            dictionary_shape=list(W.shape), effect_nonzero_fraction=float(np.mean(B != 0)),
            activity_nonzero_fraction=float(np.mean(U_tilde != 0)),
            dictionary_norms=np.linalg.norm(W, axis=1).tolist())
        effect_unscaled = np.zeros((len(targets), len(genes)), dtype=B.dtype)
        effect_scaled = np.zeros((len(targets), len(genes)), dtype=scaled.dtype)
        effect_unscaled[:, retained_index] = B[:-1]
        effect_scaled[:, retained_index] = scaled[:-1]
        np.savez_compressed(out / "effects.npz", effect=effect_scaled, effect_unscaled=effect_unscaled,
                  targets=targets, genes=genes, retained_gene_index=retained_index,
                  scale_factors=scale_factors, ntc_effect=B[-1],
                  cell_ids=np.asarray(cell_ids if cell_ids is not None else np.arange(len(ti))).astype(str),
                  target_index=ti)
        report.update(status="COMPLETE", runtime_seconds=time.time()-started,
                 max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 effects_sha256=sha256(out / "effects.npz"),
                 native_unscaled_sha256=sha256(out / "native_unscaled.npz"))
        _json(out / "fit_report.json", report)
        return dict(effects=str(out / "effects.npz"), report=str(out / "fit_report.json"))
    except BaseException as exc:
        report.update(status="FAILED", error=repr(exc), traceback=traceback.format_exc(),
                      runtime_seconds=time.time()-started)
        _json(out / "fit_report.json", report)
        raise
    finally:
        spams.trainDL, spams.lasso, core.Lasso = old_train, old_lasso_spams, old_lasso_sklearn
        for handler in (fh, sh):
            logger.removeHandler(handler)
            handler.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=20)
    ap.add_argument("--spca-alpha", type=float, default=0.1)
    ap.add_argument("--lasso-alpha", type=float, default=0.0001)
    ap.add_argument("--spca-iter", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    z = np.load(args.input, allow_pickle=False)
    x = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    result = run_fr(x, z["target_index"], z["targets"], z["genes"], args.out,
         rank=args.rank, spca_alpha=args.spca_alpha, lasso_alpha=args.lasso_alpha,
         spca_iter=args.spca_iter, seed=args.seed, threads=args.threads,
         cell_ids=z["cell_ids"] if "cell_ids" in z.files else None,
         input_provenance=dict(path=str(Path(args.input).resolve()), sha256=sha256(args.input)))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
