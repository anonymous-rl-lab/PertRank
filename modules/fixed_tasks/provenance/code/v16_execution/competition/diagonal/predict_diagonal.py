#!/usr/bin/env python3
"""v16a diagonal competitor. Input-only, no evaluation/truth access.
Sequential mean/variance decompression avoids retaining both full-gene arrays.
"""
from pathlib import Path
import argparse, gc, hashlib, importlib.util, json, os, resource, time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
R20=ROOT/'v16_execution/assets/r20/PertRank_R20_20261006/new_experiment'
INV=ROOT/'v16_execution/inventory'
DONORS=('D3_Rest','D1_Rest','D2_Rest')
ARMS=('raw','Euclidean','PertRank','diagonal')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def load_original():
 p=R20/'evaluation/evaluate_real_cells.py'
 spec=importlib.util.spec_from_file_location('original_r20',p);ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev);return ev

def verify_code():
 p=HERE/'CODE_FREEZE.json';f=json.loads(p.read_text())
 for file,h in f['files'].items():
  if sha(ROOT/file)!=h:raise AssertionError('code freeze mismatch '+file)
 return sha(p)
def check(name,x,y,errors,tol=1e-9):
 if np.shape(x)!=np.shape(y):raise AssertionError(name+' shape mismatch')
 e=float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) if np.size(x) else 0.0
 errors[name]=max(errors.get(name,0.0),e)
 if not np.isfinite(e) or e>tol:raise AssertionError(name+' parity '+str(e))

def run(donor,stage,smoke):
 if donor not in DONORS:raise ValueError('only opened donors permitted')
 t=time.monotonic();fhash=verify_code()
 cp=R20/'selection'/donor/'context.npz';sp=cp.parent/'selection.npz';ap=R20/'read'/donor/'aggregate.npz'
 mp=INV/stage/donor/'full_gene_moments_smoke.npz';rp=mp.parent/'smoke_report.json';vp=INV/(donor+'_control_gene_moments.npz')
 report=json.loads(rp.read_text());assert report['status']=='PASS'
 for key,p in [('context_sha256',cp),('selection_sha256',sp),('aggregate_sha256',ap)]:assert report[key]==sha(p)
 c=np.load(cp,allow_pickle=False);s=np.load(sp,allow_pickle=False);m=np.load(mp,allow_pickle=False);v=np.load(vp,allow_pickle=False);old=np.load(ap,allow_pickle=False)
 nt=len(m['target_ids']);G=len(c['gene_ids_keep']);nc=int(c['n_control']);qu=c['qu'].astype(int)
 if not smoke and nt!=len(c['internal_ids']):raise AssertionError('full cohort required')
 assert np.array_equal(m['target_indices'],np.arange(nt))
 assert np.array_equal(m['target_ids'],c['internal_ids'][:nt])
 assert np.array_equal(m['gene_ids'],c['gene_ids_keep']) and np.array_equal(v['gene_ids'],c['gene_ids_keep'])
 assert np.array_equal(m['cell_ids'],s['cell_ids'][s['internal_positions'][:nt]])
 assert np.array_equal(v['cm'],c['cm_keep']) and np.array_equal(m['cm_keep'],c['cm_keep'])
 assert nc==int(v['n_control'])==int(m['n_control'])==2000
 cv=v['cv'];cm=c['cm_keep'];lam=float(np.median(cv));assert lam>0 and np.all(cv>=0)
 d=1/(cv+lam);energy=np.empty((nt,5));noise=np.empty_like(energy);errors={};check_energy=[]
 archived_m2=old['internal_m2'][:nt];archived_vb=old['internal_vb'][:nt];archived_yq=old['internal_Yq'][:nt]
 # Decompress and release mean before sample variance. No outcome reference is read here.
 mean=m['mean'];assert mean.shape==(nt,5,G)
 for lo in range(0,nt,32):
  hi=min(nt,lo+32);y=mean[lo:hi]-cm
  energy[lo:hi]=np.einsum('trg,g->tr',y*y,d,optimize=True)/G
  check('m2',np.mean(y*y,axis=-1),archived_m2[lo:hi],errors)
  check('query_effect',y[:,:,qu],archived_yq[lo:hi],errors)
  if lo==0:
   for ti in range(min(2,nt)):
    for rr in (0,4):check_energy.append((ti,rr,float(sum(float(dg)*float(yg)**2 for dg,yg in zip(d,y[ti,rr]))/G)))
 del mean,y;gc.collect()
 var=m['sample_variance'];assert var.shape==(nt,5,G)
 control_noise=float(np.dot(d,cv)/(G*nc));check_noise=[]
 for lo in range(0,nt,32):
  hi=min(nt,lo+32);vv=var[lo:hi]
  assert np.isfinite(vv).all() and vv.min()>=0
  noise[lo:hi]=np.einsum('trg,g->tr',vv,d,optimize=True)/(20*G)+control_noise
  check('mean_sample_variance',np.mean(vv,axis=-1),archived_vb[lo:hi],errors)
  if lo==0:
   for ti in range(min(2,nt)):
    for rr in (0,4):check_noise.append((ti,rr,float(sum(float(dg)*(float(vg)/20+float(cg)/nc) for dg,vg,cg in zip(d,vv[ti,rr],cv))/G)))
 del var,vv;gc.collect()
 for ti,rr,a in check_energy:check('independent_scalar_energy',np.array(energy[ti,rr]),np.array(a),errors)
 for ti,rr,n in check_noise:check('independent_scalar_noise',np.array(noise[ti,rr]),np.array(n),errors)
 wd=np.zeros_like(energy);positive=energy>0;wd[positive]=np.clip(1-noise[positive]/energy[positive],0,1)
 assert np.isfinite(wd).all()
 inventory=json.loads((INV/'control_moments_inventory.json').read_text())[donor]
 out={'donor':donor,'identity':'input-only diagonal prediction; no truth/reference or nomination scores read','smoke':smoke,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'code_freeze_sha256':fhash,'targets':nt,'genes':G,'cells_per_group':20,'groups':5,'lambda_median_control_variance':lam,'metric':'1/(cv+median(cv)); original-query-coordinate output','input_parity':errors,'control_trace_note':inventory,'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in (cp,sp,ap,mp,rp,vp)},'runtime_seconds':time.monotonic()-t,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 if smoke:
  dump(HERE/'smoke'/(donor+'_smoke.json'),out);print(json.dumps({'donor':donor,'smoke':'PASS','max_input_error':max(errors.values()),'performance_computed':False}));return
 ev=load_original();pred,w0,w1,b0=ev.rebuild_arms(old,'internal_',c)
 archived=np.load(R20/'predictions'/(donor+'_predictions.npz'),allow_pickle=False)
 for name,a,b in [('archived_Euclidean_weights',w0,archived['internal_w0y']),('archived_PertRank_weights',w1,archived['internal_W1']),('archived_all_three_predictions',pred,archived['internal_predictions'])]:check(name,a,b,errors,tol=0)
 assert b0==0
 diagonal=(old['internal_Yq']*wd[:,:,None]).transpose(1,2,0)
 allpred=np.concatenate([pred,diagonal[None]],axis=0)
 p=HERE/'predictions'/(donor+'_predictions.npz');p.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(p,predictions=allpred,arm_names=np.array(ARMS),w0=w0,wM=w1,wD=wd,diagonal_energy=energy,diagonal_noise=noise,target_ids=c['internal_ids'],query_ids=c['query_gene_ids'],lambda_median_control_variance=np.array(lam))
 out.update(prediction_path=str(p.relative_to(ROOT)),prediction_sha256=sha(p),context_path=str(cp.relative_to(ROOT)),context_sha256=sha(cp),runtime_seconds=time.monotonic()-t,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 dump(HERE/'predictions'/(donor+'_record.json'),out)
 print(json.dumps({'donor':donor,'prediction_lock_ready':sha(p),'seconds':out['runtime_seconds'],'peak_rss_kib':out['peak_rss_kib'],'performance_computed':False}))

def lock_all():
 verify_code();records={}
 for donor in DONORS:
  rp=HERE/'predictions'/(donor+'_record.json');r=json.loads(rp.read_text());assert r['donor']==donor and not r['smoke']
  assert r['code_freeze_sha256']==sha(HERE/'CODE_FREEZE.json')
  assert sha(ROOT/r['prediction_path'])==r['prediction_sha256']
  records[donor]={'record_path':str(rp.relative_to(ROOT)),'record_sha256':sha(rp),'prediction_path':r['prediction_path'],'prediction_sha256':r['prediction_sha256'],'context_path':r['context_path'],'context_sha256':r['context_sha256']}
 p=HERE/'ALL_PREDICTIONS_LOCK.json';assert not p.exists(),'do not overwrite an existing lock'
 dump(p,{'identity':'all three opened donor diagonal predictions locked before new scoring','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'code_freeze_sha256':sha(HERE/'CODE_FREEZE.json'),'donors':records})
 print(json.dumps({'prediction_lock_sha256':sha(p),'donors':list(records)}))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--donor',choices=DONORS);a.add_argument('--stage',default='stage1200');a.add_argument('--smoke',action='store_true');a.add_argument('--lock-all',action='store_true');args=a.parse_args()
 if args.lock_all:lock_all()
 else:run(args.donor,args.stage,args.smoke)
