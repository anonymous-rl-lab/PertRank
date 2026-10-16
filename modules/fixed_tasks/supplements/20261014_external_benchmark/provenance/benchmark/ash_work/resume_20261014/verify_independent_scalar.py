#!/usr/bin/env python3
"""Match new ASH common arms to the existing frozen scalar independent task."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1];d=P/'D4_scores/D4_Stim8hr_independent';old=ROOT/'independent/execution/D4_Stim8hr/evaluation'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
r=json.loads((d/'evaluation_report.json').read_text());o=json.loads((old/'evaluation_report.json').read_text());assert o['formal']and o['candidate_count']==395 and o['query_count']==300 and o['block_count']==30
assert r['bootstrap_draws']==o['draws']==2000 and r['query_seed']==o['query_seed'] and r['reference_seed']==o['reference_seed'] and r['reference_batch_size']==o['batch_size']==16
for label in ['prediction','reference']:
 path=Path(r['provenance'][label+'_path']);assert sha(path)==r['provenance'][label+'_sha256']==o['provenance'][label+'_sha256']
assert sha(ROOT/'independent/execution/source_panel.npz')==o['provenance']['source_panel_sha256']
assert sha(old/'evaluation_arrays.npz')==o['evaluation_arrays_sha256'];assert sha(d/'evaluation_arrays.npz')==r['evaluation_arrays_sha256']
a=load(d/'evaluation_arrays.npz');b=load(old/'evaluation_arrays.npz');c=load(d/'compact.npz');prediction=load(Path(r['provenance']['prediction_path']));checks={}
def check(name,x,y):
 assert x.shape==y.shape and x.dtype==y.dtype and np.array_equal(x,y),name
 checks[name]=dict(exact_array_equal=True,shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest())
for new,prior in [('queries','query_ids'),('candidates','target_ids'),('blocks','query_blocks')]:check(new,c[new],b[prior])
check('raw_effects_from_frozen_prediction',c['raw_scores'][0].T,prediction['input_effects']);check('w0_from_frozen_prediction',c['w0'][0],prediction['weights_euclidean']);check('wm_from_frozen_prediction',c['wm'][0],prediction['weights_pertrank'])
check('sampled_query_blocks',a['sampled_query_blocks'],b['sampled_blocks']);check('query_weights',a['query_weights'],b['query_weights'])
for k,prior in [(25,'primary'),(10,'K10')]:
 check(f'common_query_p{k}',a[f'four_arm_p{k}'][0,:3],b[f'{prior}_scores_by_query'])
 check(f'common_joint_bootstrap_p{k}',a[f'joint_bootstrap_p{k}'][:,:3],b[f'joint_{prior}_scores'])
 for new,priorarm in [('raw','Unweighted'),('euclidean','Euclidean'),('pertrank','PertRank')]:assert r['results'][str(k)]['arms'][new]['point']==o['results'][f'K{k}']['means'][priorarm]
out=dict(status='FROZEN_SCALAR_COMMON_THREE_ARMS_BITWISE_PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),FR_rank100_comparison='NOT_AVAILABLE; not rerun solely for cross-check under user instruction.',frozen_scalar_report_path=str(old/'evaluation_report.json'),frozen_scalar_report_sha256=sha(old/'evaluation_report.json'),frozen_scalar_arrays_sha256=sha(old/'evaluation_arrays.npz'),checks=checks,exact_common_arm_points_P25_and_P10=True,frozen_input_provenance=o['provenance'],historical_quality_gates_unchanged=o['gates'],ASH_report_sha256=sha(d/'evaluation_report.json'),verification_source_sha256=sha(__file__),scope='Same frozen independent-guide 395-target task, original input/control geometry and within-lane reference-cell/query RNG. Historical scalar source only; no new FR comparison or donor-population claim.')
(d/'COMMON_THREE_ARM_MATCH.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(status=out['status'],checks=len(checks),FR_rank100_comparison='NOT_AVAILABLE')),flush=True)
