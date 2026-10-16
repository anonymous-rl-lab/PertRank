"""One frozen FR transfer fit, primary or preregistered alpha sensitivity.

This driver reads metadata and the D3 winner lock; it never reads references or
selects a configuration using D4 performance. Heavy-memory mutex is held by the
shared run_prepared_input child for every actual fit.
"""
from pathlib import Path
import argparse, json, os, subprocess, sys, time
import numpy as np
from fr_native_runner import sha256

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',required=True)
    p.add_argument('--task',required=True)
    p.add_argument('--rep',type=int,required=True)
    p.add_argument('--arm',choices=['primary','sensitivity'],required=True)
    p.add_argument('--root',default=str(Path(__file__).resolve().parent.parent))
    a=p.parse_args(); root=Path(a.root); source=Path(a.input_dir)
    lock_path=root/'FR_D3_WINNER_LOCK.json'
    lock=json.loads(lock_path.read_text())
    spec_path=root/'AMENDMENT_03_FR_SEARCH_21.json'
    if lock['status']!='LOCKED_BEFORE_D4_SCORING': raise ValueError('D3 selection is not locked')
    if lock['protocol_sha256']!=sha256(spec_path): raise AssertionError('primary protocol hash mismatch')
    sens_path=root/'AMENDMENT_04_FR_EFFECTIVE_PENALTY.json'
    sens=json.loads(sens_path.read_text())
    if sens['primary_protocol_sha256']!=sha256(spec_path): raise AssertionError('sensitivity protocol linkage mismatch')
    if a.task not in sens['input_reps'] or a.rep not in sens['input_reps'][a.task]:
        raise ValueError('task/rep not registered')
    ax=np.load(source/'axes.npz',allow_pickle=False); labels=ax['labels']; T=len(ax['target_ids'])
    if (labels==-1).sum()!=2000 or not np.all(np.bincount(labels[labels>=0],minlength=T)==20):
        raise AssertionError('cell allocation differs from fixed n20 + NTC2000')
    N=len(labels)
    if N!=sens['task_cells'][a.task]: raise AssertionError('task cell count mismatch')
    alpha=float(lock['lasso_alpha'])
    if a.arm=='sensitivity':
        # Preserve this operation order, shared with the sensitivity evaluator.
        alpha=float(lock['lasso_alpha'])*(26000/20)/(N/20)
    rank=int(lock['rank']); spca=float(lock['spca_alpha']); seed=int(lock['seed'])+a.rep
    output_root=root/('fr_outputs' if a.arm=='primary' else 'fr_outputs_sensitivity')
    out=output_root/a.task/(lock['config_id']+'_rep'+str(a.rep))
    out.parent.mkdir(parents=True,exist_ok=True)
    input_hash={'counts_sha256':sha256(source/'counts.npz'),'axes_sha256':sha256(source/'axes.npz')}
    run_record=out.parent/(out.name+'_TRANSFER_EXECUTION.json')
    rec=dict(task=a.task,rep=a.rep,arm=a.arm,primary_protocol_sha256=sha256(spec_path),
       sensitivity_protocol_sha256=sha256(sens_path),winner_lock_sha256=sha256(lock_path),
       config_id=lock['config_id'],rank=rank,spca_alpha=spca,lasso_alpha=alpha,seed=seed,
       N_task=N,n_target=20,input_hashes=input_hash,driver_sha256=sha256(__file__))
    if (out/'fit_report.json').exists():
        f=json.loads((out/'fit_report.json').read_text()); par=f['parameters']
        assert f['status']=='COMPLETE'
        assert all(par[k]==v for k,v in [('rank',rank),('spca_alpha',spca),('lasso_alpha',alpha),('seed',seed),('spca_iter',50)])
        assert all(f['input_provenance'][k]==v for k,v in input_hash.items())
        assert f['effects_sha256']==sha256(out/'effects.npz')
        rec.update(status='COMPLETE_REUSED_IDENTICAL_FIT',fit_report_sha256=sha256(out/'fit_report.json'))
        run_record.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec));return
    cmd=[sys.executable,str(Path(__file__).resolve().parent/'run_prepared_input.py'),
         '--input-dir',str(source),'--out',str(out),'--rank',str(rank),
         '--spca-alpha',str(spca),'--lasso-alpha',str(alpha),'--seed',str(seed)]
    rec.update(status='RUNNING',command=cmd);run_record.write_text(json.dumps(rec,indent=2)+'\n')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMBA_NUM_THREADS='2')
    started=time.time()
    with open(out.parent/(out.name+'_process.log'),'w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
    rec.update(returncode=result.returncode,wall_seconds=time.time()-started,
               status='COMPLETE' if result.returncode==0 else 'FAILED')
    if result.returncode==0:
        rec['fit_report_sha256']=sha256(out/'fit_report.json')
        primary=root/'fr_outputs'/a.task/out.name/'native_unscaled.npz'
        if a.arm=='sensitivity' and primary.exists():
            f=np.load(primary,allow_pickle=False);g=np.load(out/'native_unscaled.npz',allow_pickle=False)
            rec['primary_dictionary_bitwise_equal']=bool(np.array_equal(f['W'],g['W']))
            rec['primary_cell_activity_bitwise_equal']=bool(np.array_equal(f['U_tilde'],g['U_tilde']))
            if not (rec['primary_dictionary_bitwise_equal'] and rec['primary_cell_activity_bitwise_equal']):
                rec['dictionary_note']='Reported as observed; no rerun or winner selection is triggered.'
    run_record.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec))
    if result.returncode!=0: raise SystemExit(result.returncode)

if __name__=='__main__': main()
