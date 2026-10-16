"""Post-result input-side diagnostics, same 200 draws; no new scores or tuning."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from refit_cell_bootstrap import weights
from conditional_cell_bootstrap import summary
H=Path(__file__).resolve().parent;J=np.load(H/'d3_prepared/d3_fixed_control_gram.npz');G=J['gram_euclidean'];P=J['projection_centered'];ct=float(J['control_trace']);lam=J['control_projected_variances'];nc=int(J['n_controls']);nt=G.shape[0]
base=weights(np.ones((nt,20)),G,P,ct,lam,nc);rng=np.random.default_rng(202610053);q=np.array([.01,.1,.25,.5,.75,.9,.99]);fields=['mean','zero_fraction',*[f'q{x:g}' for x in q],'spearman_to_original','pearson_to_original'];vals=np.empty((200,2,len(fields)))
for b in range(200):
 m=rng.multinomial(20,np.full(20,.05),size=nt).astype(float)
 for i,w in enumerate(weights(m,G,P,ct,lam,nc)):
  vals[b,i]=[w.mean(),(w==0).mean(),*np.quantile(w,q),spearmanr(w,base[i]).statistic,np.corrcoef(w,base[i])[0,1]]
r={'identity':'post-result input-only diagnostics with exact original B200 draws; no new score or parameter selection','seed':202610053,'B':200,'original':{a:{'mean':float(w.mean()),'zero_fraction':float((w==0).mean()),'quantiles':dict(zip(map(str,q),map(float,np.quantile(w,q))))} for a,w in zip(['w0y','W1'],base)},'draws':{a:{f:summary(vals[:,i,j])for j,f in enumerate(fields)}for i,a in enumerate(['w0y','W1'])}}
np.savez_compressed(H/'refit_D3_weight_diagnostics.npz',values=vals,fields=np.array(fields),arms=np.array(['w0y','W1']));(H/'refit_D3_weight_diagnostics.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
