"""Cross-half pair energy; no clipping, no shared control mean subtraction."""
import numpy as np

def pair_energy(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    assert a.shape==b.shape and a.ndim==2 and len(a)>1
    ac=a-a.mean(axis=0);bc=b-b.mean(axis=0)
    return float(2*np.sum(ac*bc)/((len(a)-1)*a.shape[1]))

def infer_ntc(a,b,repetitions=2000,seed=20261008):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    n,p=a.shape;assert b.shape==a.shape and n>1
    ac=a-a.mean(0);bc=b-b.mean(0);gram=ac@bc.T/p
    observed=2*np.trace(gram)/(n-1)
    assert np.isclose(observed,pair_energy(a,b),rtol=1e-10,atol=1e-12)
    rng=np.random.default_rng(seed);boot=np.empty(repetitions);null=np.empty(repetitions)
    diag=np.diag(gram)
    for i in range(repetitions):
        counts=np.bincount(rng.integers(0,n,n),minlength=n).astype(float)
        boot[i]=2*(counts@diag-counts@gram@counts/n)/(n-1)
    for i in range(repetitions):null[i]=2*gram[np.arange(n),rng.permutation(n)].sum()/(n-1)
    return {'n_guides':n,'n_genes':p,'pair_energy':float(observed),'centered_guide_energy':float(observed/2),
      'bootstrap_95_percentile':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_unit':'guide, paired A/B spectra resampled together',
      'permutation_p_greater':float((1+np.sum(null>=observed))/(repetitions+1)),
      'permutation_p_two_sided':float((1+np.sum(np.abs(null)>=abs(observed)))/(repetitions+1)),
      'permutation_null_95':np.quantile(null,[.025,.975]).tolist(),'repetitions':repetitions,'seed':seed,
      'permutation_scope':'permute entire B-half guide spectra, conditional guide exchangeability, not lane-adjusted'},boot,null

def infer_target(x,repetitions=2000,seed=20261008):
    x=np.asarray(x,dtype=float);assert x.ndim==4 and x.shape[1:3]==(2,2)
    d=x[:,0]-x[:,1];vals=(d[:,0]*d[:,1]).mean(axis=1);rng=np.random.default_rng(seed)
    boot=np.array([vals[rng.integers(0,len(vals),len(vals))].mean() for _ in range(repetitions)])
    return {'target_pairs':len(vals),'n_genes':x.shape[-1],'pair_energy_mean':float(vals.mean()),
      'bootstrap_95_percentile':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_unit':'target with both guides and halves retained',
      'repetitions':repetitions,'seed':seed,'interpretation':'same-target guide differences; not a between-target effect comparison'},vals,boot
