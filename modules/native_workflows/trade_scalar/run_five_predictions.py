#!/usr/bin/env python3
"""Five-input official-core scalar fit completion; does not load scoring truth."""
from pathlib import Path
import argparse,concurrent.futures as cf,csv,gc,hashlib,json,os,shutil,signal,subprocess,threading,time,uuid
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'v16_execution/trade';INV=ROOT/'v16_execution/inventory'
OUT=HERE/'predictions';DONORS=('D3_Rest','D1_Rest','D2_Rest');EXPECTED=(1200,1127,1200)
ADAPTER=HERE/'trade_scalar_compact.R';SOURCE=HERE/'R_TRADE_univariate.R'
AH='453a729283e530db8a716472172b7053982f2c8ff93da06185e2db7902556e7f'
SH='b2fc72c0e5d7c2cde9631e73e33a95cdfed23a31b3a93c5d7141332452426320'
COLS=('weight','nu','latent_mean','TI','S');STOP=threading.Event();WORKERS=6

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+uuid.uuid4().hex)
 with tmp.open('w') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def table(p):
 with Path(p).open() as f:return list(csv.DictReader(f,delimiter='\t'))
class Sink:
 def __init__(self):self.h=hashlib.sha256()
 def write(self,s):self.h.update(s.encode());return len(s)
def load(d):
 mp=INV/'stage1200'/d/'full_gene_moments_smoke.npz';cp=INV/(d+'_control_gene_moments.npz')
 with np.load(mp,allow_pickle=False) as z:
  c={k:z[k] for k in ['target_ids','gene_ids','mean','sample_variance','cm_keep','n_control','cell_ids','m2','vb']}
 with np.load(cp,allow_pickle=False) as z:
  assert np.array_equal(c['gene_ids'],z['gene_ids']) and int(c['n_control'])==int(z['n_control'])
  assert np.max(abs(c['cm_keep']-z['cm']))<1e-9;c['cv']=z['cv']
 n=len(c['target_ids']);g=len(c['gene_ids']);assert n==EXPECTED[DONORS.index(d)]
 assert c['mean'].shape==c['sample_variance'].shape==(n,5,g)
 assert c['cell_ids'].shape==(n,5,20)
 assert all(len(np.unique(x))==100 for x in c['cell_ids'].reshape(n,100))
 assert np.isfinite(c['mean']).all() and np.isfinite(c['sample_variance']).all() and c['sample_variance'].min()>=0
 assert np.max(abs(np.mean((c['mean']-c['cm_keep'])**2,axis=2)-c['m2']))<1e-9
 assert np.max(abs(c['sample_variance'].mean(2)-c['vb']))<1e-9
 c.update(donor=d,moments_sha256=sha(mp),control_sha256=sha(cp));return c

def write_input(f,c,pairs):
 w=csv.writer(f,delimiter='\t');w.writerow(['target','replicate','gene','effect','se'])
 for i,b in pairs:
  y=c['mean'][i,b]-c['cm_keep'];se=np.sqrt(c['sample_variance'][i,b]/20+c['cv']/int(c['n_control']))
  assert np.isfinite(se).all() and (se>0).all()
  w.writerows((str(c['target_ids'][i]),b,str(g),format(e,'.17g'),format(s,'.17g')) for g,e,s in zip(c['gene_ids'],y,se))
def validate(rows,c,pairs):
 assert len(rows)==len(pairs)
 for r,(i,b) in zip(rows,pairs):
  assert r['target']==c['target_ids'][i] and int(r['replicate'])==b and int(r['n_genes'])==len(c['gene_ids'])
  z={k:float(r[k]) for k in COLS};assert np.isfinite(list(z.values())).all()
  assert 0<=z['weight']<=1 and z['S']>=0 and z['TI']>=-1e-12 and z['nu']>0
  assert abs(z['S']-z['TI']-z['latent_mean']**2)<1e-10
  assert abs(z['weight']-z['S']/(z['S']+z['nu']))<1e-10
  assert abs(z['nu']-np.mean(c['sample_variance'][i,b]/20+c['cv']/int(c['n_control'])))<1e-12

def verify_block0(c):
 d=c['donor'];p=OLD/'lean_predictions'/d;r=json.loads((p/'DONOR_COMPLETE.json').read_text())
 assert r['moments_sha256']==c['moments_sha256'] and r['control_sha256']==c['control_sha256']
 assert r['adapter_sha256']==AH and r['source_sha256']==SH and r['performance_evaluation'] is False
 assert sha(p/'weights.npz')==r['weights_sha256']
 with np.load(p/'weights.npz',allow_pickle=False) as z:
  assert np.array_equal(z['target_ids'],c['target_ids']) and np.array_equal(z['gene_ids'],c['gene_ids'])
  arrays={k:z['weights' if k=='weight' else k][:,0].copy() for k in COLS}
 ledger=json.loads((p/'FIT_LEDGER.json').read_text());assert sha(p/'FIT_LEDGER.json')==r['fit_ledger_sha256']
 seen=set();checked={};tables={};input_batches={};proof=[]
 for e in ledger['entries']:
  i=e['target_index'];assert i not in seen and e['target']==c['target_ids'][i] and e['replicate']==0;seen.add(i)
  wp=OLD/e['weights_path'];mp=OLD/e['mixture_path']
  for path,h in [(wp,e['weights_sha256']),(mp,e['mixture_sha256'])]:
   if str(path) not in checked:checked[str(path)]=sha(path)
   assert checked[str(path)]==h
  if str(wp) not in tables:
   rows=table(wp);tables[str(wp)]=rows
   pairs=[(int(np.flatnonzero(c['target_ids']==r2['target'])[0]),int(r2['replicate'])) for r2 in rows]
   assert all(b==0 for _,b in pairs)
   sink=Sink();write_input(sink,c,pairs);input_batches[str(wp)]=sink.h.hexdigest()
   proof.append({'weights_path':str(wp),'input_sha256':sink.h.hexdigest(),'weights_sha256':sha(wp),'pairs':pairs})
  assert input_batches[str(wp)]==e['input_sha256']
  row=[x for x in tables[str(wp)] if x['target']==str(c['target_ids'][i]) and int(x['replicate'])==0][0]
  validate([row],c,[(i,0)])
  for k in COLS:assert arrays[k][i]==float(row[k])
 assert len(seen)==len(c['target_ids'])
 return arrays,proof

def verify_old_other(c):
 rows=[];proof=[]
 if c['donor']!='D3_Rest':return rows,proof
 seen=set()
 for mark in sorted((OLD/'full_predictions/D3_Rest').glob('batch_*/COMPLETE.json')):
  r=json.loads(mark.read_text());pairs=[tuple(x) for x in r['pairs']]
  assert all(0<=i<100 and 1<=b<=4 and (i,b) not in seen for i,b in pairs)
  assert r['adapter_sha256']==AH
  for name,h in r['output_hashes'].items():assert sha(mark.parent/name)==h
  runtime=json.loads((mark.parent/'result_runtime.json').read_text());assert runtime['upstream_commit']=='3432d108c43325e9df65276563d65144a26b69ce'
  sink=Sink();write_input(sink,c,pairs);assert sink.h.hexdigest()==r['input_sha256']
  rr=table(mark.parent/'result_weights.tsv');validate(rr,c,pairs)
  rows.extend(zip(pairs,rr));seen.update(pairs)
  proof.append({'completion':str(mark),'completion_sha256':sha(mark),'input_sha256':r['input_sha256'],'pairs':pairs,'output_hashes':r['output_hashes']})
 assert len(seen)==400
 return rows,proof

def check_freeze():
 assert sha(SOURCE)==SH and sha(ADAPTER)==AH
 f=json.loads((HERE/'FIT_CODE_FREEZE.json').read_text())
 for name,h in f['files'].items():assert sha(HERE/name)==h,(name,'freeze mismatch')
 return sha(HERE/'FIT_CODE_FREEZE.json')
def resources():
 if STOP.is_set() or (HERE/'STOP_REQUEST').exists():raise InterruptedError('stop requested')
 if shutil.disk_usage(HERE).free<300*1024**2:STOP.set();raise RuntimeError('free disk<300MiB')
def batch(j,pairs,c,dest,identity):
 resources();out=dest/f'batch_{j:05d}';out.mkdir(exist_ok=True);mark=out/'COMPLETE.json'
 if mark.exists():
  r=json.loads(mark.read_text());assert r['pairs']==[list(x) for x in pairs] and r['identity']==identity
  sink=Sink();write_input(sink,c,pairs);assert sink.h.hexdigest()==r['input_sha256']
  for name,h in r['outputs'].items():assert sha(out/name)==h
  rows=table(out/'result_weights.tsv');validate(rows,c,pairs);return pairs,rows,True
 inp=out/'input.tsv'
 with inp.open('w',newline='') as f:write_input(f,c,pairs)
 ih=sha(inp);t=time.monotonic();env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 try:
  with (out/'run.log').open('w') as log:
   subprocess.run([str(OLD/'r-env/bin/Rscript'),str(ADAPTER),str(inp),str(out/'result'),str(SOURCE)],check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
  rows=table(out/'result_weights.tsv');validate(rows,c,pairs)
  outputs={name:sha(out/name) for name in ['result_weights.tsv','result_mixtures.rds','result_runtime.json']}
  dump(mark,{'pairs':pairs,'identity':identity,'input_sha256':ih,'outputs':outputs,'elapsed_seconds':time.monotonic()-t})
  inp.unlink();return pairs,rows,False
 except BaseException as e:STOP.set();dump(out/'FAILED.json',{'error':repr(e),'input_sha256':ih});raise

def run(d,mode):
 t=time.monotonic();c=load(d);dest=OUT/d;dest.mkdir(parents=True,exist_ok=True)
 a0,p0=verify_block0(c);other,po=verify_old_other(c)
 proof={'donor':d,'moments_sha256':c['moments_sha256'],'control_sha256':c['control_sha256'],'block0_input_batches':p0,'other_cache':po,'block0_fits':len(c['target_ids']),'other_fits':len(other),'all_exact_input_hashes':True}
 dump(dest/'CACHE_AUDIT.json',proof)
 print(json.dumps({'donor':d,'cache_passed':True,'block0':len(c['target_ids']),'other':len(other),'seconds':time.monotonic()-t}),flush=True)
 if mode=='check-cache':return
 ident={'donor':d,'source_sha256':SH,'adapter_sha256':AH,'fit_code_freeze_sha256':check_freeze(),'moments_sha256':c['moments_sha256'],'control_sha256':c['control_sha256']}
 if mode=='smoke':
  assert d=='D3_Rest';pairs=[(i,1) for i in range(10)];res=batch(0,pairs,c,HERE/'smoke',ident)
  cache=dict(other)
  errs={k:max(abs(float(row[k])-float(cache[pair][k])) for pair,row in zip(res[0],res[1])) for k in COLS}
  assert max(errs.values())<1e-12
  dump(HERE/'SMOKE_RESULT.json',{'passed':True,'pairs':pairs,'maxabs':errs,'nomination_performance_computed':False,'runtime_seconds':time.monotonic()-t});print(json.dumps(errs),flush=True);return
 n=len(c['target_ids']);arrays={k:np.full((n,5),np.nan) for k in COLS}
 for k in COLS:arrays[k][:,0]=a0[k]
 for (i,b),r in other:
  for k in COLS:arrays[k][i,b]=float(r[k])
 pending=[(i,b) for b in range(1,5) for i in range(n) if not np.isfinite(arrays['weight'][i,b])]
 jobs=[pending[j:j+10] for j in range(0,len(pending),10)];done=0;resumed=0
 with cf.ThreadPoolExecutor(max_workers=WORKERS) as pool:
  it=iter(enumerate(jobs));active={}
  def refill():
   while len(active)<WORKERS and not STOP.is_set():
    try:j,pairs=next(it)
    except StopIteration:return
    active[pool.submit(batch,j,pairs,c,dest,ident)]=j
  refill()
  while active:
   fs,_=cf.wait(active,return_when=cf.FIRST_COMPLETED)
   for f in fs:
    j=active.pop(f)
    try:pairs,rows,was=f.result()
    except BaseException:
     STOP.set()
     for x in active:x.cancel()
     raise
    for (i,b),r in zip(pairs,rows):
     for k in COLS:arrays[k][i,b]=float(r[k])
    done+=len(pairs);resumed+=len(pairs) if was else 0
    p={'donor':d,'completed':n+len(other)+done,'total':n*5,'new_or_resumed':done,'cached_block0':n,'cached_other':len(other),'elapsed_seconds':time.monotonic()-t}
    dump(dest/'PROGRESS.json',p)
    with (OUT/'progress.jsonl').open('a') as log:log.write(json.dumps(p)+'\n')
    if done%100==0 or done==len(pending):print(json.dumps(p),flush=True)
   refill()
 if STOP.is_set():raise InterruptedError('stopped')
 assert all(np.isfinite(v).all() for v in arrays.values())
 np.savez_compressed(dest/'weights.npz',target_ids=c['target_ids'],gene_ids=c['gene_ids'],weights=arrays.pop('weight'),**arrays)
 dump(dest/'DONOR_COMPLETE.json',{**ident,'status':'complete','n_targets':n,'n_blocks':5,'total_fits':n*5,'cached_block0':n,'cached_other':len(other),'new_fits':len(pending)-resumed,'resumed_fits':resumed,'weights_sha256':sha(dest/'weights.npz'),'cache_audit_sha256':sha(dest/'CACHE_AUDIT.json'),'elapsed_seconds':time.monotonic()-t,'performance_evaluation':False})
 del c,arrays;gc.collect()

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['dry','check-cache','smoke','execute'],default='dry');p.add_argument('--donors',nargs='+',choices=DONORS,default=DONORS);a=p.parse_args()
 OUT.mkdir(exist_ok=True);check_freeze()
 if a.mode=='dry':
  print(json.dumps({'donors':DONORS,'targets':EXPECTED,'fits':17635,'old_block0':3527,'old_other':400,'new_expected':13708,'workers':WORKERS,'nomination_scoring':False},indent=2));return
 if a.mode=='smoke':(HERE/'smoke').mkdir(exist_ok=True)
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:STOP.set())
 for d in a.donors:run(d,a.mode)
 if a.mode=='execute' and tuple(a.donors)==DONORS:
  completion={d:sha(OUT/d/'DONOR_COMPLETE.json') for d in DONORS}
  dump(OUT/'ALL_PREDICTIONS_COMPLETE.json',{'status':'complete','total_fits':17635,'donor_completions':completion,'performance_evaluation':False,'fit_code_freeze_sha256':check_freeze()})
if __name__=='__main__':main()
