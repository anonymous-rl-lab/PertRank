"""Uniform posthoc EPS=1e-4 numerical sensitivity, complete cohort only."""
from pathlib import Path
import os,sys,time,json,hashlib,concurrent.futures
import numpy as np
import trade_eps_runner as tr
HERE=Path(__file__).resolve().parent
RUN=Path(os.environ.get('PERT_RANK_STIM_ASSET_ROOT','/workspace/scratch/6cdbaefbed9f/stim_run_20261011'))
WORK=Path(os.environ.get('PERT_RANK_TRADE_WORK','/workspace/scratch/6cdbaefbed9f/paper_v20_20261011/work_robustness/trade_full'))
WORK.mkdir(parents=True,exist_ok=True)
def fit_batch(bounds):
 lo,hi=bounds;zs=[dict(np.load(RUN/f'input/main/target_{i:04d}.npz')) for i in range(lo,hi)]
 r=tr.fit_trade_batch(np.stack([x['log_y'] for x in zs]),np.stack([x['trade_se'] for x in zs]),[str(x['target_id']) for x in zs],zs[0]['gene_ids_keep'],WORK/f'targets_{lo:04d}_{hi:04d}')
 return lo,hi,r['manifest']['seconds']
if __name__=='__main__':
 t=time.time();jobs=[(lo,min(lo+8,561)) for lo in range(0,561,8)];n=0;failures=[]
 print(json.dumps({'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'fits':2805,'workers':6,'protocol_sha256':tr.sha256(HERE/'TRADE_FULL_COHORT_PROTOCOL.json')}),flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=6) as ex:
  futures={ex.submit(fit_batch,b):b for b in jobs}
  for f in concurrent.futures.as_completed(futures):
   try:
    lo,hi,seconds=f.result();n+=(hi-lo)*5
    progress={'completed_fits':n,'total_fits':2805,'seconds':time.time()-t,'batch':[lo,hi],'batch_seconds':seconds,'estimated_remaining_seconds':(time.time()-t)*(2805/n-1)}
    tr.atomic_json(WORK/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
   except Exception as e:
    failures.append({'bounds':futures[f],'error':repr(e)});print('FAILED',failures[-1],flush=True)
 # collect only all complete
 status={'completed_fits':n,'planned_fits':2805,'elapsed_seconds':time.time()-t,'failures':failures,'status':'complete' if n==2805 and not failures else 'not_evaluable'}
 tr.atomic_json(WORK/'RUN_STATUS.json',status)
 if failures or n!=2805:raise SystemExit('Uniform sensitivity has failures; no scoring performed')
 print('ALL COMPLETE',flush=True)
