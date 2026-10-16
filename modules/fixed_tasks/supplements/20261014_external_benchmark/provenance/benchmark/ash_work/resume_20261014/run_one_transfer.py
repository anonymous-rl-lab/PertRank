#!/usr/bin/env python3
"""One fixed ASH task per exec session; stage privately and publish atomically."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys,uuid,zipfile
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
def sha(p):
 with open(p,'rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['D4_Rest','D4_Stim8hr','D4_Stim_independent'],required=True);ap.add_argument('--rep',type=int,required=True);a=ap.parse_args()
 assert a.rep in ([0]if a.task=='D4_Stim_independent'else range(5));prepared=P/a.task/f'rep{a.rep}';inp=prepared/'full.npz';audit=prepared/'RECOVERY_INPUT_AUDIT.json'
 av=json.loads(audit.read_text());assert sha(inp)==av['full_npz_sha256']
 with zipfile.ZipFile(inp)as z:assert z.testzip()is None
 dest=P/'D4_transfer'/a.task/f'rep{a.rep}';assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);stage=dest.with_name(dest.name+'.__staging_'+uuid.uuid4().hex[:12])
 lock=dict(status='SINGLE_TASK_ATOMIC_PUBLICATION_LOCK',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task=a.task,rep=a.rep,fixed_configuration='default_uniform_fdr',input_sha256=sha(inp),input_recovery_audit_sha256=sha(audit),source_code_sha256=sha(__file__),native_runner_sha256=sha(ROOT/'ash_work/posterior_runner.R'),sharded_runner_sha256=sha(ROOT/'ash_work/run_sharded.py'),recovery_plan_sha256=sha(P/'RECOVERY_PLAN.json'),staging_path=str(stage),publication_path=str(dest),resources={'R_streams':2,'R_threads':1},publication='Consumer must additionally wait for exec session exit_code=0 then perform independent-tool SHA/CRC verification')
 lp=dest.with_name(dest.name+'_EXECUTION_LOCK.json');assert not lp.exists();lp.write_text(json.dumps(lock,indent=2)+'\n')
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
 cmd=[sys.executable,str(ROOT/'ash_work/run_sharded.py'),'--input',str(inp),'--output',str(stage),'--shards','2'];log=dest.with_name(dest.name+'_single_task_stdout.log')
 with log.open('w')as f:rc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 assert rc==0,('native ASH failure; staging evidence retained',rc)
 ex=json.loads((stage/'execution.json').read_text());rt=json.loads((stage/'runtime.json').read_text());assert ex['returncode']==0 and rt['status']=='COMPLETE' and rt['failed_rows']==0
 assert sha(stage/'posterior.npz')==ex['posterior_npz_sha256']
 with zipfile.ZipFile(stage/'posterior.npz')as z:assert z.testzip()is None
 for file in stage.rglob('*'):
  if file.is_file():
   with file.open('rb')as f:os.fsync(f.fileno())
 (stage/'PUBLICATION.json').write_text(json.dumps({'status':'STAGED_COMPLETE_ATOMIC_RENAME','staging_directory':str(stage),'final_directory':str(dest),'execution_lock_sha256':sha(lp),'posterior_sha256':ex['posterior_npz_sha256']},indent=2)+'\n')
 stage.rename(dest);print(json.dumps({'status':'COMPLETE_ONE_TASK','task':a.task,'rep':a.rep,'directory':str(dest),'posterior_sha256':ex['posterior_npz_sha256']}),flush=True)
if __name__=='__main__':main()
