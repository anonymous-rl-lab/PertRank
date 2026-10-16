#!/usr/bin/env python3
"""Recovery adapter: native ASH posterior input into unchanged pure score_arrays.

The original evaluator source and score_arrays/summarize function bodies are
unaltered. This private imported instance receives only four supplied effects:
raw, Euclidean weighted raw, PertRank weighted raw, and native ASH. No ASH
weighted variant is constructed. This is recovery of authorized ASH evaluation,
not a new ASH parameter search or a replacement for the original FR analysis.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('ash_original_pure_scoring',ROOT/'evaluation/evaluate_six_arm.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
# score_arrays is arm-count agnostic. These bindings adapt names/input only.
e.ARMS=['raw','euclidean','pertrank','ash']
e.CONTRASTS={'ash_minus_raw':(3,0),'ash_minus_euclidean':(3,1),'pertrank_minus_ash':(2,3),'pertrank_minus_euclidean':(2,1)}
e.PRIMARY_CONTRASTS=[]
def ash_effects(z):
    y=z['raw_scores']
    return np.stack([y,y*z['w0'][:,None],y*z['wm'][:,None],z['ash_scores']],axis=1)
e.effects_for_arms=ash_effects

def load_ash(directory, prepared, queries, targets):
    directory=Path(directory);prepared=Path(prepared)
    recovery=prepared/'RECOVERY_INPUT_AUDIT.json';audit=json.loads(recovery.read_text())
    if audit['status'] not in ('EXACT_HASH_RECOVERY_PASS','VALIDATED_NEW_FULL_STATS'):raise ValueError('ASH input recovery is not certified')
    if audit['status']=='VALIDATED_NEW_FULL_STATS' and audit['task']!='D4_Stim_independent':raise ValueError('New full statistics only authorized for existing independent task')
    source=Path(audit['source_input_directory'])
    for name in ('counts','axes'):
        if e.sha(source/(name+'.npz'))!=audit[name+'_sha256']:raise ValueError('ASH original input '+name+' changed')
    source_audit_name=audit.get('source_input_audit_filename','INPUT_AUDIT.json')
    if e.sha(source/source_audit_name)!=audit['source_input_audit_sha256']:raise ValueError('Original ASH input audit changed')
    if e.sha(prepared/'full.npz')!=audit['full_npz_sha256']:raise ValueError('Recovered ASH input changed')
    rt=json.loads((directory/'runtime.json').read_text());ex=json.loads((directory/'execution.json').read_text())
    if rt['status']!='COMPLETE' or rt['failed_rows']!=0 or ex['returncode']!=0:raise ValueError('Incomplete ASH task')
    if rt['config']['mode']!='ash' or rt['config']['ash_options']!={}:raise ValueError('ASH transfer differs from fixed default uniform/fdr')
    if rt['ashr']!='2.2-63' or not rt['R'].startswith('R version 4.4.3'):raise ValueError('Unexpected ASH runtime')
    if ex['input_sha256']!=e.sha(prepared/'full.npz'):raise ValueError('ASH source input hash mismatch')
    if ex['posterior_npz_sha256']!=e.sha(directory/'posterior.npz'):raise ValueError('ASH posterior hash mismatch')
    rows=[json.loads(x) for x in (directory/'fit_rows.jsonl').read_text().splitlines()]
    if len(rows)!=len(targets) or not all(r['status']=='OK' and r['final_converged'] for r in rows):raise ValueError('ASH failed or missing rows')
    z=e.load_npz(directory/'posterior.npz');inp=e.load_npz(prepared/'full.npz')
    if not np.array_equal(z['row_ids'],inp['row_ids']) or not np.array_equal(z['genes'],inp['genes']):raise ValueError('ASH output input axes mismatch')
    t=z['row_ids'].astype(str);g=z['genes'].astype(str)
    if len(np.unique(t))!=len(t) or len(np.unique(g))!=len(g):raise ValueError('Duplicate ASH IDs')
    if set(t)!=set(targets) or not set(queries).issubset(set(g)):raise ValueError('ASH requested axes mismatch')
    if [r['row_id'] for r in rows]!=t.tolist():raise ValueError('ASH fit diagnostics axes mismatch')
    ti={x:i for i,x in enumerate(t)};gi={x:i for i,x in enumerate(g)}
    f=z['posterior'][np.ix_([ti[x] for x in targets],[gi[x] for x in queries])].T
    e.old.finite('ASH posterior',f)
    return f,dict(directory=str(directory.resolve()),posterior_sha256=ex['posterior_npz_sha256'],
                  input_path=str((prepared/'full.npz').resolve()),input_sha256=ex['input_sha256'],
                  input_recovery_audit_path=str(recovery.resolve()),input_recovery_audit_sha256=e.sha(recovery),
                  execution_sha256=e.sha(directory/'execution.json'),runtime_sha256=e.sha(directory/'runtime.json'),
                  fit_rows_sha256=e.sha(directory/'fit_rows.jsonl'),n_rows=len(rows),n_genes=len(g),
                  warnings=rt['warnings'],mixEM_retries=sum(r['retry_mixem'] for r in rows),runtime_seconds=rt['seconds'])

def evaluate(a):
    frozen=json.loads(a.recovery_lock.read_text())
    if frozen['frozen_transfer_configuration']['ash_options']!={} or frozen['frozen_transfer_configuration']['config_id']!='default_uniform_fdr':raise ValueError('Fixed ASH configuration changed')
    confirmation=a.recovery_lock.with_name('D3_REPLAY_CONFIRMATION.json')
    if not confirmation.exists():confirmation=a.recovery_lock.with_name('D3_DEFAULT_REPLAY_CONFIRMATION.json')
    if json.loads(confirmation.read_text())['status'] not in ('REPLAY_CONFIRMED_HISTORICAL_DEFAULT','DEFAULT_REPLAY_CONFIRMED_HISTORICAL_FIXED_CHOICE'):raise ValueError('Historical fixed D3 default replay not confirmed')
    reps=list(range(1 if a.kind=='independent' else 5))
    if len(a.posteriors)!=len(reps) or len(a.prepared)!=len(reps):raise ValueError('Wrong fixed input repetition count')
    if a.kind=='pooled':
        task=ROOT/'pooled_data'/a.dataset/'frozen_task.npz';z,reference=e.assemble_pooled(task,reps)
        source={'task_path':str(task),'task_sha256':e.sha(task)}
    else:
        if a.dataset!='D4_Stim8hr':raise ValueError('Only frozen D4 Stim independent task')
        z,reference=e.assemble_independent(a.prediction,a.reference)
        source={'prediction_path':str(a.prediction),'prediction_sha256':e.sha(a.prediction),'reference_path':str(a.reference),'reference_sha256':e.sha(a.reference)}
    ashs=[];provenance=[]
    for p,i in zip(a.posteriors,a.prepared):
        f,src=load_ash(p,i,z['queries'],z['candidates']);ashs.append(f);provenance.append(src)
    z['ash_scores']=np.stack(ashs)
    if z['ash_scores'].shape!=z['raw_scores'].shape or len(z['queries'])!=300 or len(np.unique(z['blocks']))!=30:raise ValueError('Fixed task dimensions mismatch')
    a.out.mkdir(parents=True,exist_ok=False);t=time.monotonic()
    results,arrays,reference_seed=e.score_arrays(z,draws=2000,reference=reference)
    # Rename arm-specific output labels only; all underlying arrays are untouched.
    for k in (25,10):arrays[f'four_arm_p{k}']=arrays.pop(f'six_arm_p{k}');z[f'four_arm_p{k}']=arrays[f'four_arm_p{k}']
    np.savez_compressed(a.out/'compact.npz',**z);np.savez_compressed(a.out/'evaluation_arrays.npz',**arrays)
    report=dict(status='COMPLETE',created_utc=e.now(),recovery_replay=True,dataset=a.dataset,kind=a.kind,
        config_id='default_uniform_fdr',arms=e.ARMS,input_reps=reps,shape=dict(input_reps=len(reps),queries=len(z['queries']),candidates=len(z['candidates']),blocks=30),
        primary_K=25,sensitivity_K=10,bootstrap_draws=2000,query_seed=e.SEED,reference_seed=reference_seed,reference_batch_size=16 if reference_seed else None,
        results=results,elapsed_seconds=time.monotonic()-t,
        scope=e.old.SCOPE if reference_seed else 'Conditional on fixed observed cells/reference and original query panels; original query-block bootstrap, averaged over five fixed input repetitions.',
        simultaneous_intervals='ASH contrasts and intervals are descriptive; no new simultaneous family or performance-dependent choice.',
        provenance={**source,'ash':provenance,'D3_replay_confirmation_path':str(confirmation.resolve()),'D3_replay_confirmation_sha256':e.sha(confirmation),'recovery_lock_path':str(a.recovery_lock.resolve()),'recovery_lock_sha256':e.sha(a.recovery_lock),'adapter_sha256':e.sha(__file__),'unchanged_pure_evaluator_sha256':e.sha(e.__file__),'unchanged_score_primitive_sha256':e.sha(e.OLD_PATH)},
        compact_sha256=e.sha(a.out/'compact.npz'),evaluation_arrays_sha256=e.sha(a.out/'evaluation_arrays.npz'))
    e.save_json(a.out/'evaluation_report.json',report)
    print(json.dumps({'report':str(a.out/'evaluation_report.json'),'P25':results['25']}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['D4_Rest','D4_Stim8hr'],required=True);p.add_argument('--kind',choices=['pooled','independent'],default='pooled')
    p.add_argument('--posteriors',nargs='+',type=Path,required=True);p.add_argument('--prepared',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prediction',type=Path,default=ROOT/'independent/execution/D4_Stim8hr/prediction.npz');p.add_argument('--reference',type=Path,default=ROOT/'independent/execution/D4_Stim8hr/reference.npz')
    p.add_argument('--recovery-lock',type=Path,default=Path(__file__).parent/'RECOVERY_PLAN.json')
    evaluate(p.parse_args())
if __name__=='__main__':main()
