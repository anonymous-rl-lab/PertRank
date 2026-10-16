"""Append-only identical-native reruns after pre-scoring audit correction."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
import numpy as np
from native_recovery_audit import audit_recovery
here=Path(__file__).resolve().parent; root=here.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=root/'fr_outputs/D3_Rest';out=root/'fr_outputs/D3_Rest_audit_v2'
out.mkdir(exist_ok=False)
source=json.loads((old/'GRID_EXECUTION.json').read_text())
record={k:v for k,v in source.items() if k not in ['runs','status','total_wall_seconds']}
record.update(status='RUNNING',runs=[],prior_grid_sha256=sha(old/'GRID_EXECUTION.json'),
 audit_amendment_sha256=sha(here/'NATIVE_SOLVER_AUDIT_AMENDMENT_V2.json'),
 completion_driver_sha256=sha(__file__),extra_search_configurations=0,
 prior_grid_total_wall_seconds=source['total_wall_seconds'])
labels=np.load(root/'inputs/D3_Rest/rep0/axes.npz',allow_pickle=False)['labels']
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMBA_NUM_THREADS='2')
began=time.time()
for row in source['runs']:
    name=row['id']+'_rep0';target=out/name
    current={'id':row['id'],'original_status':row['status']}
    if row['status']=='COMPLETE':
        target.symlink_to(Path('../D3_Rest')/name,target_is_directory=True)
        current.update(status='COMPLETE',mode='REUSE_ORIGINAL_UNCHANGED',wall_seconds=0)
    else:
        command=list(row['command']);command[1]=str(here/'run_prepared_input_audit_v2.py')
        command[command.index('--out')+1]=str(target)
        t=time.time()
        with open(out/(row['id']+'_process.log'),'w') as log:
            proc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        current.update(status='COMPLETE' if proc.returncode==0 else 'FAILED',
            mode='IDENTICAL_NATIVE_PARAMETERS_NEW_ACCEPTANCE_AUDIT',returncode=proc.returncode,
            command=command,wall_seconds=time.time()-t)
    record['runs'].append(current)
    (out/'GRID_EXECUTION.json').write_text(json.dumps(record,indent=2)+'\n')
    if current['status']!='COMPLETE': raise RuntimeError('Identical rerun failed '+name)
    report=json.loads((target/'fit_report.json').read_text())
    with np.load(target/'native_unscaled.npz',allow_pickle=False) as z:
        audit=audit_recovery(z['U'],z['U_tilde'],labels,report['parameters']['lasso_alpha'],
             report['recover_audit']['dual_gap'],report['recover_audit']['tolerance'],
             report['recover_audit']['convergence_warning'])
    audit.update(original_fit_status=report['status'],original_fit_report_sha256=sha(target/'fit_report.json'),
        audit_source_sha256=sha(here/'native_recovery_audit.py'))
    path=target/'NATIVE_SOLVER_OBJECTIVE_AUDIT.json'
    payload=json.dumps(audit,indent=2)+'\n'
    if path.exists():
        assert path.read_text()==payload, 'Preexisting uniform audit changed'
    else:path.write_text(payload)
    if not audit['pass_']:raise AssertionError('Uniform native/objective audit failed '+name)
    current.update(uniform_recovery_audit_pass=True,uniform_recovery_audit_sha256=sha(path))
    if row['status']!='COMPLETE':
        with np.load(old/name/'native_unscaled.npz',allow_pickle=False) as f,np.load(target/'native_unscaled.npz',allow_pickle=False) as g:
            current['unchanged_rerun_native_arrays_bitwise_equal']={k:bool(np.array_equal(f[k],g[k])) for k in ['B','U','U_tilde','W']}
        if not all(current['unchanged_rerun_native_arrays_bitwise_equal'].values()):
            raise AssertionError('Native rerun differs from preserved original')
    print(json.dumps(current),flush=True)
record.update(status='COMPLETE',uniform_recovery_audit_all_21_pass=True,
 correction_wall_seconds=time.time()-began,total_wall_seconds=source['total_wall_seconds']+time.time()-began)
(out/'GRID_EXECUTION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'effective_grid':str(out),'status':'COMPLETE','new_parameter_configurations':0}),flush=True)
