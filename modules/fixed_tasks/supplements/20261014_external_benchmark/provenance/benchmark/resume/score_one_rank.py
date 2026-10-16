"""One score/audit owner, invoked only after the native owner's verified completion."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args()
owner=json.loads((a.run_root/'OWNER_COMPLETION.json').read_text());assert owner['status']=='OWNER_EXECUTION_ENDED_INTEGRITY_VERIFIED' and owner['owner_exit_code']==0
fit=Path(owner['fit_directory']);config='rank'+str(owner['rank'])+'_default_lasso';out=a.run_root/'rank_boundary_evaluation'/config
out.parent.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMBA_NUM_THREADS='2')
cmd=[sys.executable,str(ROOT/'vendor_work/run_python_atomic_npz.py'),str(ROOT/'evaluation/evaluate_six_arm.py'),'evaluate','--dataset','D3_Rest','--kind','pooled','--config-id',config,
     '--protocol',str(HERE/'POSTHOC_D3_RANK_BOUNDARY.json'),'--fr-effects',str(fit/'effects.npz'),'--out',str(out)]
with (ROOT/'heavy_memory.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (out.parent/(config+'_score.log')).open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 with (out.parent/(config+'_audit.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'audit/audit_result_reports.py'),'--reports',str(out/'evaluation_report.json'),'--out',str(out/'audit')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print(json.dumps({'score_owner_exit_code':0,'run_root':str(a.run_root),'evaluation_directory':str(out)}),flush=True)
