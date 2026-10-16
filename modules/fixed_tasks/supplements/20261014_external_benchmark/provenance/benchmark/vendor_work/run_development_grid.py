"""Sequential native FR grid; never accepts or reads reference expression."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, time

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',required=True)
    p.add_argument('--spec',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--max-seconds',type=float,default=3600)
    a=p.parse_args()
    here=Path(__file__).resolve().parent
    source=Path(a.spec).read_bytes(); spec=json.loads(source)
    report={'spec_sha256':hashlib.sha256(source).hexdigest(),
            'input_dir':str(Path(a.input_dir).resolve()),'status':'RUNNING','runs':[],
            'total_budget_seconds':a.max_seconds}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMBA_NUM_THREADS='2')
    grid_began=time.time()
    for config in spec['fr_grid']:
        remaining=a.max_seconds-(time.time()-grid_began)
        if remaining<=0:
            report['resource_stop']='total grid budget exhausted before next configuration'
            break
        target=out/(config['id']+'_rep0')
        cmd=[sys.executable,str(here/'run_prepared_input.py'),'--input-dir',a.input_dir,
             '--out',str(target),'--rank',str(config['rank']),
             '--spca-alpha',str(config['spca_alpha']),'--lasso-alpha',str(config['lasso_alpha']),
             '--seed',str(spec['randomness']['numpy_seed_base'])]
        began=time.time(); row={'id':config['id'],'command':cmd,'status':'RUNNING'}
        report['runs'].append(row)
        (out/'GRID_EXECUTION.json').write_text(json.dumps(report,indent=2)+'\n')
        with open(out/(config['id']+'_process.log'),'w') as log:
            try:
                process=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
                row.update(returncode=process.returncode,status='COMPLETE' if process.returncode==0 else 'FAILED')
            except subprocess.TimeoutExpired:
                row.update(status='TIMEOUT',timeout_seconds=remaining)
                report['resource_stop']='total grid budget exhausted during configuration'
        row['wall_seconds']=time.time()-began
        (out/'GRID_EXECUTION.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if 'resource_stop' in report:
            break
    report['status']='COMPLETE' if len(report['runs'])==len(spec['fr_grid']) and all(r['status']=='COMPLETE' for r in report['runs']) else 'INCOMPLETE'
    report['total_wall_seconds']=time.time()-grid_began
    (out/'GRID_EXECUTION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'grid_status':report['status']}),flush=True)

if __name__=='__main__': main()
