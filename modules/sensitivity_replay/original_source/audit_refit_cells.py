"""Post-result numerical audit and input-side weight diagnostics, no method changes.
Rebuild one complete draw directly from raw cells without the Gram representation.
"""
import json,hashlib,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from preflight_cell_states import load,S3,es
from refit_cell_bootstrap import weights
from conditional_cell_bootstrap import probs,summary
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 fz=json.loads((HERE/'REFIT_AUDIT_FREEZE.json').read_text())
 assert sha(__file__)==fz['script_sha256']
 J=np.load(HERE/'d3_prepared/d3_fixed_control_gram.npz');GE=J['gram_euclidean'];P=J['projection_centered'];ct=float(J['control_trace']);lam=J['control_projected_variances'];nc=int(J['n_controls'])
 nt=GE.shape[0];r=es.load('e4_tcell.npz');z=np.load(S3/'data/e4_w1_stage3.npz');U=z['directions'][:,:5];c=J['control_mean']
 B=200;rng=np.random.default_rng(202610053);q=np.array([.01,.1,.25,.5,.75,.9,.99]);diags=np.empty((B,2,11));base=weights(np.ones((nt,20)),GE,P,ct,lam,nc)
 first=None
 for bi in range(B):
  b=rng.multinomial(20,np.full(20,.05),size=nt).astype(float)
  wb=weights(b,GE,P,ct,lam,nc)
  if bi==0:first=(b.copy(),np.array(wb))
  for ai,w in enumerate(wb):
   diags[bi,ai]=[w.mean(),(w==0).mean(),*np.quantile(w,q),spearmanr(w,base[ai]).statistic,np.corrcoef(w,base[ai])[0,1]]
 # Keep the original repeated-cell empirical bootstrap draw, with no retuning.
 C=sparse.load_npz(HERE.parent/'d3_cell_restore/d3_full_counts.npz').tocsr();meta=np.load(HERE.parent/'d3_cell_restore/d3_full_cells_metadata.npz');scale=float(meta['scale']);tot=meta['total_counts'];b,wanted=first
 own=np.empty((2,nt));ymq=np.empty((len(r['qu']),nt));maxmoment=0.;t=time.time()
 for j in range(nt):
  sl=slice(nc+20*j,nc+20*(j+1))
  X=np.log1p(C[sl,r['keep']].toarray().astype(float)*(scale/tot[sl])[:,None])
  # This is deliberately explicit duplication; it never uses GE or P.
  Xb=np.repeat(X,b[j].astype(int),axis=0);ymq[:,j]=Xb[:,r['qu']].mean(0)
  D=Xb-c;mu=D.mean(0);a=np.sum(mu*mu);v=D.var(0,ddof=1).sum();proj=D@U
  aM=a-.9*np.sum(proj.mean(0)**2);vM=v-.9*proj.var(0,ddof=1).sum()
  own[:,j]=np.clip([1-(v/20+ct/nc)/max(a,1e-300),1-(vM/20+(ct-.9*lam.sum())/nc)/max(aM,1e-300)],0,1)
  if (j+1)%200==0:print('direct raw-cell audit',j+1,nt,round(time.time()-t,1),flush=True)
 diff=float(np.max(abs(own-wanted)));assert diff<1e-10,diff
 del C
 s,X,cs,yt,yc=load('D3');del X
 old=np.load(HERE/'refit_D3_draws.npz');scores=np.empty((2,3));maxscores=0.
 for ii,(rule,cc) in enumerate(cs.items()):
  E=ymq[cc['rows']]-cc['fixed_control_mean'][:,None];pt=probs(cc['T'],cc['sq'])
  for ai,w in enumerate([np.ones(nt),own[0],own[1]]):
   pp=probs(E*w,cc['sq']);scores[ii,ai]=(pp*pt).sum(1).mean()/25
  maxscores=max(maxscores,float(np.max(abs(scores[ii]-old['mean_scores'][0,ii]))))
 assert maxscores<1e-12,maxscores
 fields=['weight_mean','zero_fraction',*[f'q{x:g}' for x in q],'spearman_to_uniform_original','pearson_to_uniform_original']
 report={'identity':'post-result numerical audit and input-side diagnostics; no statistical or method rule changed','freeze':fz,'B':B,'seed':202610053,'full_draw_direct_raw_validation':{'targets':nt,'cells_per_target':20,'features':len(r['keep']),'explicit_repeated_cells':True,'weight_max_abs_diff':diff,'score_max_abs_diff':maxscores,'scores':scores.tolist()},'diagnostics':{a:{field:summary(diags[:,ai,fi]) for fi,field in enumerate(fields)} for ai,a in enumerate(['w0y','W1'])},'original_input_weights':{a:{'mean':float(w.mean()),'zero_fraction':float((w==0).mean()),'quantiles':dict(zip(map(str,q),map(float,np.quantile(w,q))))} for a,w in zip(['w0y','W1'],base)}}
 np.savez_compressed(HERE/'refit_D3_weight_diagnostics.npz',diagnostics=diags,fields=np.array(fields),arms=np.array(['w0y','W1']))
 (HERE/'refit_D3_direct_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
