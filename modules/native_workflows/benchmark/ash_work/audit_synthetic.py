#!/usr/bin/env python3
"""Independent Bayes-formula check of R posterior transport, on synthetic data only."""
from pathlib import Path
import json
import numpy as np
from scipy.special import logsumexp,log_ndtr
from scipy.stats import norm,truncnorm

P=Path(__file__).resolve().parent
z=np.load(P/'synthetic_input.npz',allow_pickle=False)
y=z['effects'];se=z['ses'];checks=[]
for name in ['ash_default','ash_normal','ash_shrink','trade']:
    d=P/('synthetic_'+name)
    pred=np.load(d/'posterior.npz',allow_pickle=False)
    assert np.array_equal(pred['row_ids'],z['row_ids'])
    assert np.array_equal(pred['genes'],z['genes'])
    rows=[json.loads(s) for s in (d/'fit_rows.jsonl').read_text().splitlines()]
    maxerr=0
    for i,row in enumerate(rows):
        assert row['status']=='OK' and row['final_converged']
        g={k:np.asarray(v) for k,v in row['fitted_g'].items()}
        # ashr defaults prune(pi_thresh=1e-10) for posterior computation.
        keep=g['pi']>1e-10;pi=g['pi'][keep]
        Y=y[i,:,None];S=se[i,:,None]
        if 'sd' in g:
            SD=g['sd'][keep][None,:];MU=g['mean'][keep][None,:]
            marginal=norm.logpdf(Y,loc=MU,scale=np.sqrt(S*S+SD*SD))
            means=(Y*SD*SD+MU*S*S)/(S*S+SD*SD)
        else:
            a=g['a'][keep];b=g['b'][keep]
            marginal=np.zeros((len(Y),len(a)));means=np.zeros_like(marginal)
            for j,(A,B) in enumerate(zip(a,b)):
                if A==B:
                    marginal[:,j]=norm.logpdf(y[i],loc=A,scale=se[i]);means[:,j]=A
                else:
                    lo=(A-y[i])/se[i];hi=(B-y[i])/se[i]
                    L=log_ndtr(hi);R=log_ndtr(lo)
                    pos=lo>0
                    L[pos]=log_ndtr(-lo[pos]);R[pos]=log_ndtr(-hi[pos])
                    ld=L+np.log(-np.expm1(R-L))
                    marginal[:,j]=ld-np.log(B-A)
                    means[:,j]=truncnorm.mean(lo,hi,loc=y[i],scale=se[i])
        logp=marginal+np.log(pi)[None,:]
        post=(np.exp(logp-logsumexp(logp,axis=1,keepdims=True))*means).sum(axis=1)
        err=float(np.max(np.abs(post-pred['posterior'][i])))
        assert err<1e-8,(name,i,err)
        maxerr=max(maxerr,err)
    checks.append(dict(configuration=name,n_rows=len(rows),n_genes=y.shape[1],
                       all_native_converged=True,max_absolute_posterior_error=maxerr))
out=dict(status='PASS',checks=checks,reference_read=False,
         audit='Independent normal/uniform-mixture Bayes formula from exported native mixture; synthetic rows only')
(P/'SYNTHETIC_AUDIT.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
