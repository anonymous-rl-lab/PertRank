#!/usr/bin/env python3
"""Select the predeclared native ASH configuration on opened D3 only."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('six_arm',ROOT/'evaluation/evaluate_six_arm.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def score(pred_dir,z):
    rt=json.loads((pred_dir/'runtime.json').read_text());ex=json.loads((pred_dir/'execution.json').read_text())
    assert rt['status']=='COMPLETE' and rt['failed_rows']==0 and ex['returncode']==0
    assert ex['posterior_npz_sha256']==sha(pred_dir/'posterior.npz')
    rows=[json.loads(s)for s in (pred_dir/'fit_rows.jsonl').read_text().splitlines()]
    assert len(rows)==len(z['candidates']) and all(r['status']=='OK' and r['final_converged']for r in rows)
    with np.load(pred_dir/'posterior.npz',allow_pickle=False) as p:
        assert len(np.unique(p['row_ids']))==len(p['row_ids'])
        assert len(np.unique(p['genes']))==len(p['genes'])
        ti={x:i for i,x in enumerate(p['row_ids'].astype(str))};gi={x:i for i,x in enumerate(p['genes'].astype(str))}
        assert set(ti)==set(z['candidates']) and set(z['queries']).issubset(gi)
        F=p['posterior'][np.ix_([ti[x]for x in z['candidates']],[gi[x]for x in z['queries']])].T
    own=ev.old.own_mask(z['queries'],z['candidates'])
    pm=ev.old.probabilities(F[None],own,25);tm=ev.old.probabilities(z['reference_scores'][None],own,25)
    sq=ev.old.query_scores(pm,tm,25)[0,0]
    record=dict(status='COMPLETE',P25=float(sq.mean()),prediction_directory=str(pred_dir.resolve()),
                posterior_sha256=sha(pred_dir/'posterior.npz'),input_sha256=ex['input_sha256'],
                config=rt['config'],runtime_seconds=rt['seconds'],
                warning_count=rt['warnings'],mixEM_retries=sum(r['retry_mixem']for r in rows),
                method='ashr native full gene posterior means, common effect/SE scale')
    return record,sq

def main():
    p=argparse.ArgumentParser();p.add_argument('--default',type=Path,required=True)
    p.add_argument('--normal',type=Path,required=True);p.add_argument('--shrink',type=Path,required=True)
    p.add_argument('--trade',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();assert not a.output_dir.exists()
    prepared=ROOT/'ash_work/D3_Rest_input'
    validation=json.loads((prepared/'INPUT_VALIDATION.json').read_text())
    assert validation['status']=='PASS'
    assert validation['output_files']['full.npz']==sha(prepared/'full.npz')
    for source,expected in validation['input_files'].items():assert sha(Path(source))==expected
    source_dir=ROOT/'inputs/D3_Rest/rep0'
    assert sha(source_dir/'INPUT_AUDIT.json')==validation['source_input_audit_sha256']
    source_audit=validation['source_input_audit']
    assert source_audit['status']=='EXACT_RAW_R20_REPLAY_PASS'
    assert source_audit['task']=='D3_Rest' and source_audit['rep']==0
    assert source_audit['raw_totals_exact'] and source_audit['control_count']==2000
    for file,expected in source_audit['files_sha256'].items():assert sha(source_dir/file)==expected
    assert source_audit['source_selection_sha256']==sha(ROOT/'pooled_data/D3_Rest/selection.npz')
    task=ROOT/'pooled_data/D3_Rest/frozen_task.npz';z,_=ev.assemble_pooled(task,[0])
    assert len(z['candidates'])==1200 and len(z['queries'])==300 and len(np.unique(z['blocks']))==30
    records=[];sq=[]
    for name,path,opts in [('default_uniform_fdr',a.default,{}),('normal_fdr',a.normal,{'mixcompdist':'normal'}),
                           ('normal_shrink',a.shrink,{'mixcompdist':'normal','method':'shrink'})]:
        rec,s=score(path,z);assert rec['config']['mode']=='ash' and rec['config']['ash_options']==opts
        rec['config_id']=name;records.append(rec);sq.append(s)
    assert len({r['input_sha256']for r in records})==1
    assert records[0]['input_sha256']==validation['output_files']['full.npz']
    values=np.array([r['P25']for r in records]);ties=np.flatnonzero(values.max()-values<=1e-12)
    winner=int(ties[0]);trade=None
    if a.trade:
        trade,tradesq=score(a.trade,z);assert trade['config']['mode']=='trade'
    a.output_dir.mkdir(parents=True)
    np.savez_compressed(a.output_dir/'D3_ASH_query_scores.npz',scores=np.array(sq),query_ids=z['queries'],
                        target_ids=z['candidates'],blocks=z['blocks'],config_ids=np.array([r['config_id']for r in records]))
    result=dict(status='LOCKED_BEFORE_D4_SCORING',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                dataset='D3_Rest',input_rep=0,n_candidates=1200,n_queries=300,n_query_blocks=30,K=25,
                selection_rule='Highest native posterior mean P25; ties within1e-12 prefer official default then listed order',
                protocol_sha256=sha(ROOT/'EXECUTION_SPEC.json'),amendment_sha256=sha(ROOT/'AMENDMENT_01_ASH_DEFAULT.txt'),
                input_validation_sha256=sha(prepared/'INPUT_VALIDATION.json'),
                task_sha256=sha(task),full_grid_results=records,winning_config=records[winner],TRADE_development=trade,
                D4_posterior_scores_seen=False,D1_D2_Stim_expression_read=False)
    (a.output_dir/'D3_ASH_WINNER_LOCK.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
