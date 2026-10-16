#!/usr/bin/env python3
"""Independent explicit-pair and direct-resampling checks on small matrices."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import importlib.util
import hashlib
import itertools
import json
import numpy as np

HERE=Path(__file__).resolve().parent
SCRIPT=HERE.parent/'ntc_guide/stats_impl.py'
sp=importlib.util.spec_from_file_location('production_ntc_stats',SCRIPT)
prod=importlib.util.module_from_spec(sp);sp.loader.exec_module(prod)

def explicit_pairs(a,b):
    return float(np.mean([np.dot(a[i]-a[j],b[i]-b[j])/a.shape[1]
                          for i,j in itertools.combinations(range(len(a)),2)]))

def main():
    cases=[];maxerr=0.
    rng=np.random.default_rng(68381)
    for n,p in [(2,1),(3,4),(5,7),(9,3)]:
        a=rng.normal(size=(n,p));b=.4*a+rng.normal(size=(n,p))
        want=explicit_pairs(a,b);got=prod.pair_energy(a,b)
        assert np.isclose(want,got,rtol=1e-12,atol=1e-12)
        out,boot,null=prod.infer_ntc(a,b,repetitions=127,seed=1219)
        gen=np.random.default_rng(1219)
        direct_boot=[]
        for _ in range(127):
            ix=gen.integers(0,n,n)
            direct_boot.append(explicit_pairs(a[ix],b[ix]))
        direct_null=[]
        for _ in range(127):
            pi=gen.permutation(n)
            direct_null.append(explicit_pairs(a,b[pi]))
        assert np.allclose(boot,direct_boot,rtol=1e-12,atol=1e-12)
        assert np.allclose(null,direct_null,rtol=1e-12,atol=1e-12)
        assert np.isclose(prod.pair_energy(a+31.,b-42.),want,rtol=1e-12,atol=1e-12)
        err=max(abs(want-got),float(np.max(np.abs(boot-direct_boot))),float(np.max(np.abs(null-direct_null))))
        maxerr=max(maxerr,err)
        cases.append({'guides':n,'genes':p,'direct_pair_energy':want,'max_absolute_error':err})
    a=rng.normal(size=(8,6))
    out,bo,nu=prod.infer_ntc(a,-a,repetitions=127,seed=1219)
    assert out['pair_energy']<0 and np.all(bo<=0)
    x=rng.normal(size=(5,2,2,7))
    out_t,vals,bo_t=prod.infer_target(x,repetitions=127,seed=1219)
    manual=np.array([sum((x[t,0,0,g]-x[t,1,0,g])*(x[t,0,1,g]-x[t,1,1,g])
                         for g in range(7))/7 for t in range(5)])
    assert np.allclose(vals,manual,rtol=1e-12,atol=1e-12)
    gen=np.random.default_rng(1219)
    wantbo=np.array([manual[gen.integers(0,5,5)].mean() for _ in range(127)])
    assert np.allclose(bo_t,wantbo,rtol=1e-12,atol=1e-12)
    r={'status':'PASS','script_sha256':hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
       'explicit_pair_equivalence':cases,'max_absolute_error':maxerr,
       'paired_guide_bootstrap_equals_direct_resampling':True,
       'whole_B_spectrum_permutation_equals_direct_recomputation':True,
       'independent_half_translation_invariance':True,
       'negative_cross_energy_retained':True,
       'target_axes_and_target_unit_bootstrap_verified':True,
       'formula':'D=mean_{g<h} ((A_g-A_h) dot (B_g-B_h))/p = 2 sum_g ((A_g-Abar) dot (B_g-Bbar))/((G-1)p)',
       'scope':'Mathematics and implementation validation; no empirical conclusion from synthetic matrices.'}
    (HERE/'ntc_statistics_audit.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()
