"""One isolated owner for one posthoc native rank or depth fit."""
from pathlib import Path
import argparse,json,os,subprocess,sys,uuid
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--kind',required=True,choices=['rank','depth']);p.add_argument('--rank',type=int)
p.add_argument('--curve');p.add_argument('--depth',type=int);p.add_argument('--input-root')
a=p.parse_args();out=HERE/'atomic_runs'/(a.kind+'_'+str(uuid.uuid4()));out.parent.mkdir(parents=True,exist_ok=True);log=out.with_suffix('.log')
request=dict(kind=a.kind,rank=a.rank,curve=a.curve,depth=a.depth,output_root=str(out),log=str(log))
(HERE/'LATEST_ONE_NATIVE_REQUEST.json').write_text(json.dumps(request,indent=2)+'\n');print(json.dumps(request),flush=True)
if a.kind=='rank':cmd=[sys.executable,str(HERE/'run_rank_boundary.py'),'--rank',str(a.rank),'--output-root',str(out)]
else:
 cmd=[sys.executable,str(HERE/'run_depth_native.py'),'--curve',a.curve,'--depth',str(a.depth),'--output-root',str(out)]
 if a.input_root:cmd+=['--input-root',a.input_root]
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMBA_NUM_THREADS='2')
with log.open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
print(json.dumps(dict(owner_exit_code=r.returncode,**request)),flush=True);raise SystemExit(r.returncode)
