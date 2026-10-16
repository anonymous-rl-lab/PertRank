#!/usr/bin/env python3
"""Post-hoc D4 TRADE scalar adapter, frozen inputs, resumable bounded batches.

This is a new recomputation from the original frozen 58,900 input cells.
It does not claim recovery of the previous interrupted TRADE run.
No scoring truth is loaded here. Source archived assets are read-only.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, signal, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import numpy as np
from scipy.sparse import csr_matrix
HERE=Path(__file__).resolve().parent
REC=HERE.parents[1]
D4=REC/'restored/v18_d4_confirmation'
DATA=D4/'data/D4_Rest'
TRADE=REC/'restored/PertRank_v17_Competition/v17_experiments/trade'
sys.path.insert(0,str(D4/'reader/vendor'))
from hdf5_chunk_index import Transport
from row_transport import plans, packs_for
STOP=False
AH='453a729283e530db8a716472172b7053982f2c8ff93da06185e2db7902556e7f'
SH='b2fc72c0e5d7c2cde9631e73e33a95cdfed23a31b3a93c5d7141332452426320'
COLS=('weight','nu','latent_mean','TI','S')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 with t.open('w') as f: json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def save(p,**x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.stem+'.tmp.npz')
 np.savez_compressed(t,**x);os.replace(t,p)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def event(**x):
 x={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**x}
 print(json.dumps(x),flush=True)
 with (HERE/'events.jsonl').open('a') as f: f.write(json.dumps(x)+'\n');f.flush()
def stopped():
 if STOP or (HERE/'STOP_REQUEST').exists():raise InterruptedError('stop requested; existing completed checkpoints remain valid')
def identity():
 assert sha(TRADE/'trade_scalar_compact.R')==AH and sha(TRADE/'R_TRADE_univariate.R')==SH
 files=[DATA/n for n in ('cohort.npz','control_model.npz','query_context.npz','input_chunk_manifest.json','indptr.npy','hdf_objects.json')]
 return {'analysis':'posthoc TRADE-core scalar adapter','origin':'fresh recomputation of original frozen D4 input cells; not recovery of old interrupted run',
   'adapter_sha256':AH,'source_sha256':SH,'inputs':{str(p.relative_to(REC)):sha(p) for p in files},'runner_sha256':sha(__file__)}
class Reader:
 def __init__(self,c,workers,gap):
  self.c=c;self.workers=workers;self.gap=gap
  self.specs=json.loads((DATA/'hdf_objects.json').read_text());self.ptr=np.load(DATA/'indptr.npy',mmap_mode='r')
  full=json.loads((DATA/'input_chunk_manifest.json').read_text());self.manifest=full['chunks'];self.expected=full['source_identity']
  self.tr=Transport(self.expected['url'],HERE/'transport_cache');assert self.tr.expected==self.expected
  assert self.expected['size']==118609904492 and self.expected['etag'].strip('"')=='27f4103f28edf9cd15e64e247cd9a507-7070'
  self.allowed=np.sort(c['input_cells'].ravel());assert len(np.unique(self.allowed))==58900
 def rows(self,rows):
  rows=np.asarray(rows,dtype=np.int64);pos=np.searchsorted(self.allowed,rows)
  assert np.all(pos<len(self.allowed)) and np.array_equal(self.allowed[pos],rows)
  pieces,ip=plans(self.manifest,rows,self.ptr,self.specs);packs=packs_for(pieces,self.gap,8<<20)
  data=np.empty(int(ip[-1]),np.uint32);indices=np.empty(int(ip[-1]),np.int32);ng=len(self.c['gene_ids_full']);t=time.monotonic()
  def fetch(pack):
   return pack,self.tr.fetch(int(pack['start']),int(pack['end']-pack['start']))
  with ThreadPoolExecutor(max_workers=self.workers) as pool:
   pending={};it=iter(packs)
   def fill():
    while len(pending)<2*self.workers:
     try:p=next(it)
     except StopIteration:return
     pending[pool.submit(fetch,p)]=p
   fill()
   while pending:
    stopped();done,_=wait(pending,return_when=FIRST_COMPLETED)
    for f in done:
     pending.pop(f);pack,body=f.result()
     for address,end,kind,dest,count in pack['pieces']:
      v=np.frombuffer(body,dtype=self.specs[kind]['dtype'],count=count,offset=address-pack['start'])
      if kind=='data':
       assert np.isfinite(v).all() and np.all(v>=0) and np.all(v==np.floor(v)) and np.all(v<=np.iinfo(np.uint32).max);data[dest:dest+count]=v
      else:assert np.all((v>=0)&(v<ng));indices[dest:dest+count]=v
     (self.tr.cache/f"{pack['start']}_{pack['end']-pack['start']}.bin").unlink(missing_ok=True)
    fill()
  raw=csr_matrix((data,indices,ip),shape=(len(rows),ng))
  assert np.array_equal(np.asarray(raw.sum(1)).ravel(),self.c['total_counts'][rows]),'source UMI totals mismatch'
  return raw,{'rows':len(rows),'packs':len(packs),'bytes':sum(p['end']-p['start'] for p in packs),'seconds':time.monotonic()-t}
def old_batch(lo,hi):
 p=DATA/f'input_predictions/targets_{lo:04d}_{hi:04d}.npz';r=json.loads(p.with_suffix('.json').read_text())
 assert sha(p)==r['sha256'];return load(p)
def moments(lo,hi,reader,c,m,q,ident):
 stopped();p=HERE/f'moments/targets_{lo:04d}_{hi:04d}.npz';jp=p.with_suffix('.json')
 if jp.exists():
  mark=json.loads(jp.read_text());assert mark['identity']==ident and sha(p)==mark['sha256'];z=load(p)
  assert np.array_equal(z['cell_ids'],c['input_cells'][lo:hi]);event(stage='moments_resume',lo=lo,hi=hi);return p,z
 cells=c['input_cells'][lo:hi];raw,transport=reader.rows(cells.ravel());old=old_batch(lo,hi)
 n=hi-lo;g=len(m['keep']);means=np.empty((n,5,g));variances=np.empty_like(means);errs={}
 spec=importlib.util.spec_from_file_location('d4_engine',D4/'prediction/engine.py');eng=importlib.util.module_from_spec(spec);spec.loader.exec_module(eng)
 for j in range(n):
  dense=raw[j*100:(j+1)*100,m['keep']].toarray().astype(float)
  log=np.log1p(dense*(float(m['scale'])/c['total_counts'][cells[j].ravel()])[:,None]);x=log.reshape(5,20,g)
  means[j]=x.mean(1);variances[j]=x.var(1,ddof=1)
  stats=eng.effect_stats(log,m['cm'],m['U'],m['cv'],m['lam'])
  new={'Y':stats['y'][:,q['qu']], 'log_aE':stats['aE'],'log_nuE':stats['nuE'],'log_aM':stats['aM'],'log_nuM':stats['nuM'],
    'log_projection_mean':stats['pmean'],'log_projection_variance':stats['pvar'],'log_trace_target':stats['trace_target']}
  for k,v in new.items():
   err=float(np.max(abs(v-old[k][j])));errs[k]=max(errs.get(k,0),err)
   assert np.allclose(v,old[k][j],atol=1e-9,rtol=1e-10),(k,lo+j,err)
  for key,computed in [('Euclidean',eng.scalar(stats['aE'],stats['nuE'])),('PertRank',eng.scalar(stats['aM'],stats['nuM']))]:
   oldw=old['weights'][j,1 if key=='Euclidean' else 2];err=float(np.max(abs(computed-oldw)));errs[key]=max(errs.get(key,0),err);assert err<1e-9
  assert np.array_equal(old['cell_ids'][j],cells[j].ravel())
  assert np.max(abs(np.mean((means[j]-m['cm'])**2,1)-stats['aE']))<1e-10
  assert np.max(abs(np.mean(variances[j]/20+m['cv']/int(m['n_control']),1)-stats['nuE']))<1e-10
 assert np.isfinite(means).all() and np.isfinite(variances).all() and variances.min()>=0
 save(p,target_indices=np.arange(lo,hi),target_ids=c['target_ids'][lo:hi],gene_ids=m['gene_ids_keep'],cell_ids=cells,mean=means,sample_variance=variances)
 dump(jp,{'identity':ident,'sha256':sha(p),'old_stat_maxabs':errs,'transport':transport,'lo':lo,'hi':hi})
 event(stage='moments_complete',lo=lo,hi=hi,transport=transport,old_stat_maxabs=errs)
 return p,load(p)
def fit(lo,hi,mp,z,c,m,ident,rscript):
 stopped();out=HERE/f'fits/targets_{lo:04d}_{hi:04d}';out.mkdir(parents=True,exist_ok=True);mark=out/'COMPLETE.json'
 def validate():
  with (out/'result_weights.tsv').open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
  assert len(rows)==(hi-lo)*5
  for r,(j,b) in zip(rows,((j,b) for j in range(hi-lo) for b in range(5))):
   assert r['target']==c['target_ids'][lo+j] and int(r['replicate'])==b and int(r['n_genes'])==len(m['keep'])
   v={k:float(r[k]) for k in COLS};assert np.isfinite(list(v.values())).all() and 0<=v['weight']<=1 and v['TI']>=-1e-12 and v['S']>=0 and v['nu']>0
   assert abs(v['S']-v['TI']-v['latent_mean']**2)<1e-10 and abs(v['weight']-v['S']/(v['S']+v['nu']))<1e-10
   assert abs(v['nu']-np.mean(z['sample_variance'][j,b]/20+m['cv']/int(m['n_control'])))<1e-12
  return rows
 if mark.exists():
  r=json.loads(mark.read_text());assert r['identity']==ident and r['moments_sha256']==sha(mp)
  for n,h in r['outputs'].items():assert sha(out/n)==h
  validate();event(stage='fits_resume',lo=lo,hi=hi,fits=(hi-lo)*5);return
 inp=out/'input.tsv';t=time.monotonic()
 with inp.open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['target','replicate','gene','effect','se'])
  for j in range(hi-lo):
   for b in range(5):
    y=z['mean'][j,b]-m['cm'];se=np.sqrt(z['sample_variance'][j,b]/20+m['cv']/int(m['n_control']))
    assert np.isfinite(se).all() and (se>0).all()
    w.writerows((str(c['target_ids'][lo+j]),b,str(g),format(e,'.17g'),format(s,'.17g')) for g,e,s in zip(m['gene_ids_keep'],y,se))
 ih=sha(inp);env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 with (out/'run.log').open('w') as f:
  p=subprocess.run([rscript,str(TRADE/'trade_scalar_compact.R'),str(inp),str(out/'result'),str(TRADE/'R_TRADE_univariate.R')],stdout=f,stderr=subprocess.STDOUT,env=env)
 if p.returncode:
  dump(out/'FAILED.json',{'returncode':p.returncode,'input_sha256':ih,'moments_sha256':sha(mp)});raise RuntimeError(f'R failure {p.returncode}: {out}/run.log')
 rows=validate();outputs={n:sha(out/n) for n in ['result_weights.tsv','result_mixtures.rds','result_runtime.json']}
 warnings=(out/'run.log').read_text();assert 'TRADE ADAPTER COMPLETED' in warnings
 dump(mark,{'identity':ident,'moments_sha256':sha(mp),'input_sha256':ih,'outputs':outputs,'fits':len(rows),'seconds':time.monotonic()-t,'warnings_present':'Warning' in warnings})
 inp.unlink();event(stage='fits_complete',lo=lo,hi=hi,fits=len(rows),seconds=time.monotonic()-t,warnings_present='Warning' in warnings)
def main():
 a=argparse.ArgumentParser();a.add_argument('--mode',choices=['read-smoke','smoke','execute'],required=True);a.add_argument('--rscript',default='Rscript');a.add_argument('--io-workers',type=int,default=8);a.add_argument('--fit-workers',type=int,default=2);a.add_argument('--gap',type=int,default=4096);args=a.parse_args()
 assert 1<=args.io_workers<=16 and 1<=args.fit_workers<=2
 ident=identity();freeze=HERE/'FREEZE.json'
 if freeze.exists():assert json.loads(freeze.read_text())==ident,'pipeline/source changed after freeze'
 else:dump(freeze,ident)
 c=load(DATA/'cohort.npz');m=load(DATA/'control_model.npz');q=load(DATA/'query_context.npz')
 assert c['input_cells'].shape==(589,5,20) and int(m['n_control'])==2000
 assert np.array_equal(c['target_ids'],q['target_ids']) and np.array_equal(c['gene_ids_full'],m['gene_ids_full'])
 reader=Reader(c,args.io_workers,args.gap);t=time.monotonic();batches=[(lo,min(lo+8,589)) for lo in range(0,589,8)]
 if args.mode!='execute':batches=batches[:1]
 event(stage='start',mode=args.mode,pid=os.getpid(),io_workers=args.io_workers,fit_workers=args.fit_workers,total_targets=sum(hi-lo for lo,hi in batches),origin=ident['origin'])
 with ThreadPoolExecutor(max_workers=args.fit_workers) as pool:
  pending={}
  for lo,hi in batches:
   stopped()
   while len(pending)>=args.fit_workers+1:
    done,_=wait(pending,return_when=FIRST_COMPLETED)
    for f in done:pending.pop(f);f.result()
   mp,z=moments(lo,hi,reader,c,m,q,ident)
   if args.mode!='read-smoke':pending[pool.submit(fit,lo,hi,mp,z,c,m,ident,args.rscript)]=(lo,hi)
  for f in pending:f.result()
 report={'status':'PASS','mode':args.mode,'targets':sum(hi-lo for lo,hi in batches),'fits':0 if args.mode=='read-smoke' else sum((hi-lo)*5 for lo,hi in batches),'elapsed_seconds':time.monotonic()-t,'source_identity':reader.expected,'network_calls':reader.tr.calls,'network_bytes':reader.tr.nbytes,'retry_events':reader.tr.retry_events,'truth_loaded':False,'identity':ident}
 dump(HERE/(args.mode.upper().replace('-','_')+'_PASS.json'),report);event(stage='complete',**{k:v for k,v in report.items() if k!='identity'})
 if args.mode=='execute':
  arrays={k:np.empty((589,5)) for k in COLS}
  for lo,hi in batches:
   out=HERE/f'fits/targets_{lo:04d}_{hi:04d}'
   with (out/'result_weights.tsv').open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
   for r,(j,b) in zip(rows,((j,b) for j in range(lo,hi) for b in range(5))):
    for k in COLS:arrays[k][j,b]=float(r[k])
  save(HERE/'D4_TRADE_weights.npz',target_ids=c['target_ids'],gene_ids=m['gene_ids_keep'],weights=arrays.pop('weight'),**arrays)
  dump(HERE/'WEIGHTS_COMPLETE.json',{'identity':ident,'sha256':sha(HERE/'D4_TRADE_weights.npz'),'fits':2945,'performance_evaluated':False})
if __name__=='__main__':
 def stop(*_):
  global STOP;STOP=True
 for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,stop)
 main()
