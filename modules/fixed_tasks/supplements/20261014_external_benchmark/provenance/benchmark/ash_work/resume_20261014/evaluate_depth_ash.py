#!/usr/bin/env python3
"""Four-arm input adapter for frozen posthoc depths; unchanged pure score math."""
from pathlib import Path
import argparse,hashlib,json,time,zipfile
import numpy as np
import evaluate_ash_adapter as adapter
e=adapter.e;P=Path(__file__).resolve().parent;D=e.ROOT/'posthoc_depth_20261014'
PROTOCOL_SHA='bd529ba31eaf41cfff522716835d2f95aedc61bfd7a3fdae68e4140e7e6bb546'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--depth',type=int,choices=[20,50,100],required=True);ap.add_argument('--owner-session',required=True);ap.add_argument('--out',type=Path);a=ap.parse_args();n=a.depth
 assert e.sha(D/'POSTHOC_DEPTH_PROTOCOL.json')==PROTOCOL_SHA
 assert json.loads((P/'D3_DEFAULT_REPLAY_CONFIRMATION.json').read_text())['status']=='DEFAULT_REPLAY_CONFIRMED_HISTORICAL_FIXED_CHOICE'
 source=D/'inputs'/f'n{n}';inp=P/'depth_inputs'/f'n{n}';fit=P/'depth_transfer'/f'n{n}';out=a.out or P/'depth_evaluation'/f'n{n}'
 audit=json.loads((source/'INPUT_AUDIT.json').read_text());val=json.loads((inp/'INPUT_VALIDATION.json').read_text());ex=json.loads((fit/'execution.json').read_text());rt=json.loads((fit/'runtime.json').read_text())
 assert audit['status']=='FROZEN_POSTHOC_NESTED_DEPTH_INPUT_PASS' and audit['depth']==n and audit['protocol_sha256']==PROTOCOL_SHA
 assert e.sha(source/'INPUT_AUDIT.json')==val['source_input_audit_sha256'] and e.sha(source/'full_effect_stats.npz')==val['source_fullstats_sha256']
 assert e.sha(inp/'full.npz')==val['full_npz_sha256']==ex['input_sha256'] and e.sha(fit/'posterior.npz')==ex['posterior_npz_sha256']
 assert ex['returncode']==0 and rt['status']=='COMPLETE' and rt['failed_rows']==0 and rt['config']['mode']=='ash' and rt['config']['ash_options']=={}
 assert rt['ashr']=='2.2-63' and rt['R'].startswith('R version 4.4.3')
 with zipfile.ZipFile(fit/'posterior.npz')as archive:assert archive.testzip()is None
 rows=[json.loads(x)for x in(fit/'fit_rows.jsonl').read_text().splitlines()];assert len(rows)==250 and all(x['status']=='OK' and x['final_converged'] for x in rows)
 axes=e.load_npz(source/'axes.npz');pred=e.load_npz(fit/'posterior.npz');queries=axes['query_ids'].astype(str);targets=axes['target_ids'].astype(str)
 assert len(queries)==300 and len(targets)==250 and len(np.unique(axes['query_blocks']))==30 and np.array_equal(pred['row_ids'],targets)
 assert len(np.unique(pred['genes']))==len(pred['genes']);gi={g:i for i,g in enumerate(pred['genes'].astype(str))};assert set(queries).issubset(gi)
 ash=pred['posterior'][:,[gi[x]for x in queries]].T
 z=dict(raw_scores=axes['input_effects'].T[None],reference_scores=axes['reference_effects'].T,w0=axes['weights_euclidean'][None],wm=axes['weights_pertrank'][None],ash_scores=ash[None],queries=queries,candidates=targets,blocks=axes['query_blocks'],input_reps=np.array([0]))
 begin=time.monotonic();results,arrays,_=e.score_arrays(z,draws=2000,reference=None)
 for k in (25,10):arrays[f'four_arm_p{k}']=arrays.pop(f'six_arm_p{k}');z[f'four_arm_p{k}']=arrays[f'four_arm_p{k}']
 common={key:hashlib.sha256(np.ascontiguousarray(z[key]).tobytes()).hexdigest() for key in ['raw_scores','reference_scores','w0','wm','queries','candidates','blocks']}
 common.update({f'three_arm_query_p{k}':hashlib.sha256(np.ascontiguousarray(arrays[f'four_arm_p{k}'][:,:3]).tobytes()).hexdigest() for k in (25,10)})
 out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'compact.npz',**z);np.savez_compressed(out/'evaluation_arrays.npz',**arrays)
 report=dict(status='COMPLETE_FROZEN_POSTHOC_DEPTH_ASH',created_utc=e.now(),depth=n,main_benchmark=False,config_id='default_uniform_fdr',input_reps=[0],arms=e.ARMS,shape=dict(input_reps=1,queries=300,candidates=250,blocks=30),primary_K=25,sensitivity_K=10,bootstrap_draws=2000,query_seed=e.SEED,reference_seed=None,results=results,elapsed_seconds=time.monotonic()-begin,
  scope='Separate 250-target nested depth diagnostic. Conditional query-block bootstrap; one shared nested cell chain, fixed donor/reference/control geometry. Neither independent depth replicates nor donor-population inference; absolute scores not directly comparable to main 1200/589/561/395-target tasks.',
  simultaneous_intervals='All ASH depth contrasts descriptive; no new multiple-testing family or depth-dependent parameter choice.',common_three_arm_array_sha256=common,
  provenance=dict(protocol_sha256=PROTOCOL_SHA,owner_exec_session_id=a.owner_session,owner_exit_code=0,source_input_audit_sha256=e.sha(source/'INPUT_AUDIT.json'),input_validation_sha256=e.sha(inp/'INPUT_VALIDATION.json'),posterior_sha256=ex['posterior_npz_sha256'],execution_sha256=e.sha(fit/'execution.json'),runtime_sha256=e.sha(fit/'runtime.json'),evaluator_adapter_sha256=e.sha(__file__),four_arm_adapter_sha256=e.sha(adapter.__file__),unchanged_pure_evaluator_sha256=e.sha(e.__file__),unchanged_score_primitive_sha256=e.sha(e.OLD_PATH)),compact_sha256=e.sha(out/'compact.npz'),evaluation_arrays_sha256=e.sha(out/'evaluation_arrays.npz'))
 e.save_json(out/'evaluation_report.json',report);print(json.dumps({'report':str(out/'evaluation_report.json'),'depth':n,'P25':results['25']}),flush=True)
if __name__=='__main__':main()
