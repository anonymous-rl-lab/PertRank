"""Audit native one-hot Lasso without replacing or tightening its solver.

Uses sklearn's documented dual-gap criterion scaled by mean squared response.
The exact one-hot solution is only an independent objective-gap diagnostic.
"""
import numpy as np

def audit_recovery(U, U_tilde, labels, alpha, dual_gap, solver_tol, convergence_warning=False):
    U=np.asarray(U,dtype=np.float64)
    y=np.asarray(U_tilde,dtype=np.float64)
    labels=np.asarray(labels,dtype=np.int64)
    # NTC occupies the last fitted coefficient, matching official design matrix.
    groups=np.where(labels==-1,U.shape[0]-1,labels)
    count=np.bincount(groups,minlength=U.shape[0]).astype(np.float64)
    if len(groups)!=len(y) or np.any(count<=0):
        raise ValueError('One-hot audit axes/counts invalid')
    sums=np.zeros_like(U)
    np.add.at(sums,groups,y)
    means=sums/count[:,None]
    exact=np.sign(means)*np.maximum(np.abs(means)-float(alpha)*len(y)/count[:,None],0)
    delta=U-exact
    # Difference of exact convex Lasso objectives; the common within-group
    # variance cancels. Native objective is ||y-Xu||^2/(2N)+alpha*||u||_1.
    gap=np.sum(count[:,None]/(2*len(y))*delta*(U+exact-2*means)
               +float(alpha)*(np.abs(U)-np.abs(exact)),axis=0)
    mean_sq=np.mean(y*y,axis=0)
    limit=float(solver_tol)*mean_sq
    eps=256*np.finfo(np.float64).eps*np.maximum(1,mean_sq)
    dual=np.asarray(dual_gap,dtype=np.float64)
    finite=bool(all(np.isfinite(a).all() for a in [U,y,exact,gap,limit,dual]))
    native_pass=bool(finite and np.all(dual<=limit+eps) and np.all(dual>=-eps))
    objective_pass=bool(finite and np.all(gap>=-eps) and np.all(gap<=limit+eps))
    dual_bound_pass=bool(finite and np.all(gap<=dual+eps))
    old_limit=2e-5*max(1.,float(np.max(np.abs(exact))))
    return dict(audit_version='native_stopping_and_exact_objective_v1',
        pass_=bool(native_pass and objective_pass and dual_bound_pass and not convergence_warning),
        n_cells=len(y),n_coefficients=U.size,
        finite=finite,native_dual_gap_pass=native_pass,
        independent_objective_gap_pass=objective_pass,
        exact_objective_within_reported_dual_gap_pass=dual_bound_pass,
        convergence_warning=bool(convergence_warning),
        native_dual_gap_tolerance=limit.tolist(),roundoff_allowance=eps.tolist(),
        independent_objective_gap=gap.tolist(),native_dual_gap=dual.tolist(),
        max_objective_gap_to_native_tolerance=float(np.max(gap/np.maximum(limit,np.finfo(float).tiny))),
        coefficient_max_absolute_error=float(np.max(np.abs(delta))),
        target_coefficient_max_absolute_error=float(np.max(np.abs(delta[:-1]))),
        ntc_coefficient_max_absolute_error=float(np.max(np.abs(delta[-1]))),
        legacy_coefficient_tolerance=old_limit,
        legacy_coefficient_gate_pass=bool(np.max(np.abs(delta))<=old_limit),
        coefficient_count_above_legacy_tolerance=int(np.count_nonzero(np.abs(delta)>old_limit)),
        note='Legacy absolute coefficient tolerance is retained only as a diagnostic; native approximate coefficients and solver parameters are unchanged.')
