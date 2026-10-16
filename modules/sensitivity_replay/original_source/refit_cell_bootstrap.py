import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from preflight_cell_states import load,A
from conditional_cell_bootstrap import probs,summary,sha,ARMS
HERE=Path(__file__).resolve().parent

def weights(mult,G,P,ct,lam,nc):
 n=20.; beta=.9
 # Full trace/energy units; gene-count division cancels from the reliability ratio.
 sq=np.einsum('ti,tij,tj->t',mult,G,mult,optimize=False)
 tr=np.einsum('ti,tii->t',mult,G,optimize=False)
 pp=np.einsum('ti,tik->tk',mult,P,optimize=False)
 a=sq/(n*n);v=(tr-sq/n)/(n-1)
 aM=a-beta*(pp*pp).sum(1)/(n*n)
 vM=v-beta*(np.einsum('ti,tik,tik->t',mult,P,P,optimize=False)-(pp*pp).sum(1)/n)/(n-1)
 assert np.min(a)>=-1e-9 and np.min(aM)>=-1e-9
 assert np.min(v)>=-1e-9 and np.min(vM)>=-1e-9
 w0=np.clip(1-(v/n+ct/nc)/np.maximum(a,1e-300),0,1)
 wm=np.clip(1-(vM/n+(ct-beta*lam.sum())/nc)/np.maximum(aM,1e-300),0,1)
 return w0,wm

def gram_math_check():
 rng=np.random.default_rng(404);n=20;g=37;t=6
 X=rng.normal(size=(t,n,g));c=rng.normal(size=g);U=np.linalg.qr(rng.normal(size=(g,5)))[0];D=X-c
 G=D@np.transpose(D,(0,2,1));P=D@U;ct=37.;lam=np.ones(5);nc=2000
 b=rng.multinomial(n,np.full(n,1/n),size=t).astype(float);w0,wm=weights(b,G,P,ct,lam,nc)
 worst=0.
 for j in range(t):
  Y=np.repeat(D[j],b[j].astype(int),axis=0);mean=Y.mean(0);v=Y.var(0,ddof=1).sum();p=Y@U
  ae=mean@mean;am=ae-.9*np.sum(p.mean(0)**2);vm=v-.9*p.var(0,ddof=1).sum()
  wanted=np.clip([1-(v/n+ct/nc)/max(ae,1e-300),1-(vm/n+(ct-.9*lam.sum())/nc)/max(am,1e-300)],0,1)
  got=np.array([w0[j],wm[j]]);worst=max(worst,float(np.max(abs(wanted-got))))
  assert np.max(abs(wanted-got))<=1e-10
  sq=b[j]@G[j]@b[j];tr=np.dot(b[j],np.diag(G[j]));pps=b[j]@P[j]
  direct=np.array([ae,v,am,vm]);derived=np.array([sq/n**2,(tr-sq/n)/(n-1),sq/n**2-.9*(pps@pps)/n**2,(tr-sq/n)/(n-1)-.9*(np.dot(b[j],(P[j]*P[j]).sum(1))-(pps@pps)/n)/(n-1)])
  assert np.max(abs(direct-derived)/np.maximum(1,abs(direct)))<=1e-10
 return {'synthetic_targets':t,'copied_rows_weight_max_abs_diff':worst}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gram',required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 freeze=json.loads((HERE/'REFIT_BOOTSTRAP_FREEZE.json').read_text())
 for f,h in freeze['files'].items():assert sha(HERE/f)==h,(f,'freeze changed')
 math_check=gram_math_check()
 B=2 if args.smoke else 200;J=np.load(args.gram);G=J['gram_euclidean'];P=J['projection_centered'];ct=float(J['control_trace']);lam=J['control_projected_variances'];nc=int(J['n_controls'])
 from preflight_cell_states import S3,es
 z=np.load(S3/'data/e4_w1_stage3.npz');r=es.load('e4_tcell.npz')
 assert int(J['n_genes'])==len(r['keep']) and int(J['k'])==5 and float(J['beta'])==.9
 assert str(J['input_control_state_sha256'])==sha(S3/'data/e4_w1_stage3.npz')
 assert ct==float(z['control_trace']) and np.array_equal(lam,z['eigenvalues'][:5])
 assert np.array_equal(J['control_mean'],r['cm_keep'])
 s,X,cs,yt,yc=load('D3');nt=len(s['names']);assert np.array_equal(s['cell_index'],J['cell_index']);assert np.array_equal(s['names'],J['names'])
 xt=X[nc:].reshape(nt,20,-1);del X
 w0,wm=weights(np.ones((nt,20)),G,P,ct,lam,nc)
 prep={};checks={}
 for rule,c in cs.items():
  E=yt[c['rows']]-c['fixed_control_mean'][:,None]
  ws=[np.ones(nt),w0,wm];ww=[np.ones(nt),c['w0y'],c['W1']];pt=probs(c['T'],c['sq']);orig=[];d={};origscore={}
  for a,w,old in zip(ARMS,ws,ww):
   pp=probs(E*w[None,:],c['sq']);bb=(pp*pt).sum(1).reshape(-1,2)/25;want=A.exp_scores(c['Y']*old[None,:],c)
   d[a]={'weight_max_abs_diff':float(np.max(abs(w-old))),'per_query_score_max_abs_diff':float(np.max(abs(bb-want)))}
   assert d[a]['weight_max_abs_diff']<=1e-5,(rule,a,d[a]);assert d[a]['per_query_score_max_abs_diff']<=1e-12,(rule,a,d[a])
   orig.append(pp);origscore[a]=float(bb.mean())
  checks[rule]=d;prep[rule]=(c,pt,orig,origscore)
 rng=np.random.default_rng(202610053);means=np.empty((B,2,3));ret=np.empty_like(means);wsummary=np.empty((B,4));t=time.time()
 for b in range(B):
  mult=rng.multinomial(20,np.full(20,.05),size=nt).astype(float);wb0,wbM=weights(mult,G,P,ct,lam,nc)
  ym=np.einsum('tc,tcg->tg',mult,xt,optimize=False).T/20
  for ii,(rule,(c,pt,orig,original_scores)) in enumerate(prep.items()):
   E=ym[c['rows']]-c['fixed_control_mean'][:,None]
   for ai,w in enumerate([np.ones(nt),wb0,wbM]):
    pp=probs(E*w[None,:],c['sq']);means[b,ii,ai]=float((pp*pt).sum(1).mean()/25);ret[b,ii,ai]=float((pp*orig[ai]).sum(1).mean()/25)
  wsummary[b]=[wb0.mean(),wbM.mean(),(wb0==0).mean(),(wbM==0).mean()]
  if (b+1)%20==0 or b+1==B:print('refit',b+1,B,'seconds',time.time()-t,flush=True)
 result={'identity':'development fixed-NTC-directions bootstrap with weights refitted, not full pipeline uncertainty','B':B,'smoke':args.smoke,'freeze':freeze,'gram_sha256':sha(args.gram),'gram_math_check':math_check,'checks':checks,'panels':{}}
 for ii,(rule,(_c,_pt,_orig,origscore)) in enumerate(prep.items()):
  result['panels'][rule]={'original_scores':origscore,'comparison_distributions':{f'W1 - {a}':dict(summary(means[:,ii,2]-means[:,ii,j]),original_difference=origscore['W1']-origscore[a]) for j,a in [(0,'raw'),(1,'w0y')]},'score_distributions':{a:summary(means[:,ii,j]) for j,a in enumerate(ARMS)},'overlap_with_original_list':{a:summary(ret[:,ii,j]) for j,a in enumerate(ARMS)}}
 prefix='smoke_' if args.smoke else ''
 np.savez_compressed(HERE/f'{prefix}refit_D3_draws.npz',mean_scores=means,list_overlaps=ret,weight_summaries=wsummary,panels=np.array(list(prep)),arms=np.array(ARMS))
 (HERE/f'{prefix}refit_D3_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
