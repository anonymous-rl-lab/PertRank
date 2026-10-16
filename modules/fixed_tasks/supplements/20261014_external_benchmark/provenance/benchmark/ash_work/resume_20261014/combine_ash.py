#!/usr/bin/env python3
"""Original task-stratified bootstrap/summary math for the four ASH adapter arms."""
from pathlib import Path
import argparse,json
import numpy as np
import evaluate_ash_adapter as adapter
e=adapter.e

def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,nargs=2,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    reports=[json.loads(p.read_text())for p in a.reports]
    assert [r['dataset']for r in reports]==['D4_Rest','D4_Stim8hr']
    assert all(r['status']=='COMPLETE' and r['kind']=='pooled' and r['config_id']=='default_uniform_fdr' and r['input_reps']==list(range(5)) and r['arms']==e.ARMS for r in reports)
    assert len({r['provenance']['recovery_lock_sha256']for r in reports})==1
    compact=[];arrays=[]
    for p,r in zip(a.reports,reports):
        cp=p.with_name('compact.npz');ap=p.with_name('evaluation_arrays.npz')
        assert e.sha(cp)==r['compact_sha256'] and e.sha(ap)==r['evaluation_arrays_sha256']
        compact.append(e.load_npz(cp));arrays.append(e.load_npz(ap))
    state_seeds=np.random.SeedSequence(e.SEED).spawn(2);saved={};qweights=[]
    for i,(z,ss)in enumerate(zip(compact,state_seeds)):
        sampled,w=e.old.query_draws(z['blocks'],2000,ss);saved[f'state_{i}_sampled_blocks']=sampled;saved[f'state_{i}_query_weights']=w;qweights.append(w)
    result={}
    for k in (25,10):
        scores=[x[f'four_arm_p{k}']for x in arrays]
        boots=[w@s.mean(0).T for w,s in zip(qweights,scores)];boot=np.mean(boots,axis=0)
        result[str(k)]=e.summarize(np.concatenate(scores,axis=2),boot,prefix='task_stratified_query')
        for i,b in enumerate(boots):saved[f'state_{i}_query_bootstrap_p{k}']=b
        saved[f'equal_state_bootstrap_p{k}']=boot
    a.out.mkdir(parents=True,exist_ok=False);np.savez_compressed(a.out/'equal_state_arrays.npz',**saved)
    out=dict(status='COMPLETE',created_utc=e.now(),recovery_replay=True,config_id='default_uniform_fdr',arms=e.ARMS,datasets=['D4_Rest','D4_Stim8hr'],results=result,
        seed=e.SEED,state_seed_spawn_keys=[list(s.spawn_key)for s in state_seeds],draws=2000,
        scope='Original task-stratified query-block bootstrap: independent streams in the two original different panels, then equal-state mean. Conditional on fixed cells/reference and observed D4 donor.',
        paired_input_reps='Fixed repetition labels are equally averaged for stability; not biological donor replicates.',
        simultaneous_intervals='Descriptive ASH comparison intervals; no new multiplicity family or D4 tuning.',
        provenance=[{'path':str(p.resolve()),'sha256':e.sha(p)}for p in a.reports],
        unchanged_pure_evaluator_sha256=e.sha(e.__file__),unchanged_score_primitive_sha256=e.sha(e.OLD_PATH),adapter_sha256=e.sha(adapter.__file__),combine_adapter_sha256=e.sha(__file__),arrays_sha256=e.sha(a.out/'equal_state_arrays.npz'))
    e.save_json(a.out/'equal_state_report.json',out);print(json.dumps({'report':str(a.out/'equal_state_report.json'),'P25':result['25']}),flush=True)
if __name__=='__main__':main()
