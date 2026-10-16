"""Fixed-axis posthoc control replacement from archived raw sparse count rows."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
from scipy import sparse
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
import frozen_engine as fe
import stim_stats as st
RUN=Path(os.environ.get('PERT_RANK_STIM_ASSET_ROOT','/workspace/scratch/6cdbaefbed9f/stim_run_20261011'))
OUT=HERE/'control_replacement';OUT.mkdir(exist_ok=True)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
roles=np.load(RUN/'role_rows.npz'); orig=dict(np.load(RUN/'input/control_model.npz'));query=dict(np.load(RUN/'input/main_query_context.npz'))
cache={};cache_files={}
for f in (RUN/'raw_rows').glob('*.json'):
 d=json.loads(f.read_text())
 if d['role'] in ['input_union','input_control','mechanism_a','mechanism_b'] and f.with_suffix('.npz').exists():
  cache_files[f.stem]=(f.with_suffix('.npz'),d)
  for i,r in enumerate(d['rows']):cache.setdefault(int(r),(f.stem,i))
required=set(roles['main_input'].ravel())|set(roles['input_control'])|set(roles['mechanism_a'])|set(roles['mechanism_b'])
assert not required-set(cache),'missing required raw rows'
loaded={}
def raw(rows):
 groups={}
 for i,r in enumerate(rows):
  k,j=cache[int(r)];groups.setdefault(k,[]).append((i,j))
 if len(groups)==1:
  k,ij=next(iter(groups.items()));mat=sparse.load_npz(cache_files[k][0]);return mat[[j for i,j in ij]]
 out=[None]*len(rows)
 for k,ij in groups.items():
  mat=sparse.load_npz(cache_files[k][0])
  for i,j in ij:out[i]=mat[j]
 return sparse.vstack(out,format='csr')
def transform(mat,ctrl=False):
 totals=np.asarray(mat.sum(1)).ravel().astype(np.float64);z=np.log1p(mat[:,orig['keep']].toarray().astype(float)*(float(orig['scale'])/totals)[:,None])
 return z.astype(np.float32).astype(float) if ctrl else z
z0=transform(raw(roles['input_control']),True)
assert np.max(abs(z0.mean(0)-orig['cm']))<1e-14
sets={'independent_mechanism':np.r_[roles['mechanism_a'],roles['mechanism_b']], 'balanced_replacement':np.r_[roles['input_control'][:1000],roles['mechanism_a'][:500],roles['mechanism_b'][:500]]}
models={};modelinfo={}
for name,ids in sets.items():
 t=time.time();z=transform(raw(ids),True);cm,cv,U,lam=fe.geometry(z)
 lam0=np.var(z0@U,axis=0,ddof=1)
 models[name+'_direction']=(orig['cm'],orig['cv'],U,lam0)
 models[name+'_full']=(cm,cv,U,lam)
 subspace=float(np.sum(np.linalg.svd(orig['U'].T@U,compute_uv=False)**2)/5)
 np.savez_compressed(OUT/(name+'_model.npz'),control_ids=ids,cm=cm,cv=cv,U=U,lam=lam,lam_original_control=lam0)
 modelinfo[name]={'controls':len(ids),'overlap_original':int(np.isin(ids,roles['input_control']).sum()),'overlap_evaluation':int(np.isin(ids,roles['evaluation_control']).sum()),'subspace_overlap':subspace,'elapsed_seconds':time.time()-t}
 print('model',name,modelinfo[name],flush=True)
models={'original':(orig['cm'],orig['cv'],orig['U'],orig['lam']),**models}
y={name:np.empty((5,561,len(query['qu']))) for name in models};w={name:{a:np.empty((5,561)) for a in ['raw','Euclidean','PertRank']} for name in models}
maxerr={'effect':0.,'Euclidean':0.,'PertRank':0.};t=time.time()
for j in range(561):
 log=transform(raw(roles['main_input'][j]));saved=np.load(RUN/f'input/main/target_{j:04d}.npz')
 for name,(cm,cv,U,lam) in models.items():
  stat=fe.effect_stats(log,cm,U,cv,lam);y[name][:,j]=stat['y'][:,query['qu']]
  w[name]['raw'][:,j]=1;w[name]['Euclidean'][:,j]=fe.scalar(stat['aE'],stat['nuE']);w[name]['PertRank'][:,j]=fe.scalar(stat['aM'],stat['nuM'])
  if name=='original':
   maxerr['effect']=max(maxerr['effect'],float(np.max(abs(stat['y']-saved['log_y']))))
   for arm in ['Euclidean','PertRank']:maxerr[arm]=max(maxerr[arm],float(np.max(abs(w[name][arm][:,j]-saved['weight_'+arm]))))
 if (j+1)%50==0:print('targets',j+1,'seconds',time.time()-t,flush=True)
assert max(maxerr.values())<1e-12,maxerr
ref=np.load(RUN/'evaluation/main_reference.npz');registered=np.load(RUN/'evaluation/registered_scores.npz');result={'classification':'posthoc sensitivity','models':modelinfo,'baseline_reconstruction_maxabs':maxerr,'results':{}}
for name in models:
 scores,names=st.score_predictions(y[name],ref['effects'],w[name],query['query_gene_ids'],query['target_ids'])
 if name=='original':
  err=max(float(np.max(abs(scores[i]-registered['main_scores'][list(registered['main_arm_names']).index(arm)]))) for i,arm in enumerate(names));assert err<1e-14;result['baseline_score_maxabs']=err
 result['results'][name]={}
 for panel,poskey in [('responsive','responsive_positions'),('terciles','tercile_positions')]:
  ss=scores[:,:,query[poskey],:]; diff=(ss[list(names).index('PertRank')]-ss[list(names).index('Euclidean')]).mean((0,2))
  result['results'][name][panel]={'means':{a:float(ss[i].mean()) for i,a in enumerate(names)},'PertRank_minus_Euclidean':st.paired_block_interval(diff,query['blocks_'+panel]),'zero_fraction':{a:float((v==0).mean()) for a,v in w[name].items()}}
 np.savez_compressed(OUT/(name+'_scores_weights.npz'),scores=scores,arm_names=np.array(names),**{'weight_'+a:v for a,v in w[name].items()})
result['elapsed_target_seconds']=time.time()-t
result['inputs']={str(f.relative_to(RUN)):digest(f) for f in [RUN/'role_rows.npz',RUN/'input/control_model.npz',RUN/'input/main_query_context.npz',RUN/'evaluation/main_reference.npz']}
(OUT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['results'],indent=2),flush=True)
