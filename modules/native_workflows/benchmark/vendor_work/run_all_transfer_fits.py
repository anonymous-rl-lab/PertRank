"""Sequential frozen transfer fits after D3 lock; prioritize available primaries.

Local ongoing execution only; no reference expression or scoring is read.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,subprocess,sys,time

def main():
    p=argparse.ArgumentParser();p.add_argument('--max-wait-seconds',type=float,default=7200);a=p.parse_args()
    here=Path(__file__).resolve().parent;root=here.parent
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    winner=json.loads((root/'FR_D3_WINNER_LOCK.json').read_text())
    assert winner['status']=='LOCKED_BEFORE_D4_SCORING'
    tasks=[('D4_Rest',0),('D4_Stim8hr',0),('D4_Stim_independent',0)]
    tasks += [(task,r) for r in range(1,5) for task in ['D4_Rest','D4_Stim8hr']]
    pending=[(arm,t,r) for arm in ['primary','sensitivity'] for t,r in tasks]
    ledger={'status':'RUNNING','winner_lock_sha256':sha(root/'FR_D3_WINNER_LOCK.json'),
      'driver_sha256':sha(__file__),'fits':[],'sensitivity_fit_wall_seconds':0,'primary_fit_wall_seconds':0}
    record=root/'fr_outputs/TRANSFER_ALL_EXECUTION.json'
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMBA_NUM_THREADS='2')
    began=time.time()
    while pending:
        selected=None
        for arm,task,rep in pending:
            source=root/'inputs'/task
            if task!='D4_Stim_independent':source=source/('rep'+str(rep))
            marker=source/('PREPARATION.json' if task=='D4_Stim_independent' else 'INPUT_AUDIT.json')
            if marker.exists():selected=(arm,task,rep,source);break
        if selected is None:
            if time.time()-began>a.max_wait_seconds:raise TimeoutError('Waiting for input audit markers exceeded limit')
            time.sleep(10);continue
        arm,task,rep,source=selected
        if arm=='sensitivity' and ledger['sensitivity_fit_wall_seconds']>=1200:
            ledger['status']='SENSITIVITY_RESOURCE_BUDGET_STOP';record.write_text(json.dumps(ledger,indent=2)+'\n')
            raise RuntimeError('Frozen sensitivity 20-minute execution target exceeded; parent review required')
        cmd=[sys.executable,str(here/'run_locked_transfer_audit_v2.py'),'--input-dir',str(source),
             '--task',task,'--rep',str(rep),'--arm',arm]
        t=time.time(); proc=subprocess.run(cmd,env=env)
        entry={'task':task,'rep':rep,'arm':arm,'returncode':proc.returncode,'fit_wall_seconds':time.time()-t}
        # Resource waiting is recorded separately from native fitting budget.
        ledger['fits'].append(entry);record.write_text(json.dumps(ledger,indent=2)+'\n')
        if proc.returncode:raise RuntimeError('Native transfer fit failed; see preserved run')
        fit=root/('fr_outputs' if arm=='primary' else 'fr_outputs_sensitivity')/task/(winner['config_id']+'_rep'+str(rep))
        fit_report=json.loads((fit/'fit_report.json').read_text())
        entry['native_fit_seconds']=fit_report['runtime_seconds']
        ledger[arm+'_fit_wall_seconds']+=entry['native_fit_seconds']
        subprocess.run([sys.executable,str(here/'add_native_audit_certificate.py'),'--fit-dir',str(fit),'--axes',str(source/'axes.npz')],env=env,check=True)
        compact=root/('fr_compact' if arm=='primary' else 'fr_compact_sensitivity')/task/fit.name
        if not compact.exists():
            with open(root/'heavy_memory.lock','a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                subprocess.run([sys.executable,str(here/'export_compact_fit.py'),'--fit-dir',str(fit),'--axes',str(source/'axes.npz'),'--out',str(compact)],env=env,check=True)
        # Preserve the fit-bound uniform certificate in the compact export.
        (compact/'NATIVE_SOLVER_OBJECTIVE_AUDIT.json').write_bytes((fit/'NATIVE_SOLVER_OBJECTIVE_AUDIT.json').read_bytes())
        entry.update(status='COMPLETE',fit_directory=str(fit),compact_directory=str(compact),
              audit_certificate_sha256=sha(fit/'NATIVE_SOLVER_OBJECTIVE_AUDIT.json'))
        record.write_text(json.dumps(ledger,indent=2)+'\n');print(json.dumps(entry),flush=True)
        pending.remove((arm,task,rep))
    ledger['status']='COMPLETE';ledger['total_wall_seconds']=time.time()-began
    record.write_text(json.dumps(ledger,indent=2)+'\n');print(json.dumps({'transfer_all_status':'COMPLETE','n_fits':len(ledger['fits'])}),flush=True)
if __name__=='__main__':main()
