#!/usr/bin/env python3
"""Compare matched ASH and FR scoring artifacts; no rescoring or parameter changes."""
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def arrays(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--ash',type=Path,required=True);ap.add_argument('--fr',type=Path,required=True);a=ap.parse_args()
ar=json.loads((a.ash/'evaluation_report.json').read_text());fr=json.loads((a.fr/'evaluation_report.json').read_text())
assert ar['status']==fr['status']=='COMPLETE'
for k in ['dataset','kind','input_reps','bootstrap_draws','query_seed','reference_seed']:assert ar[k]==fr[k],k
assert ar['arms'][:3]==fr['arms'][:3]==['raw','euclidean','pertrank']
for d,r in [(a.ash,ar),(a.fr,fr)]:
 for n in ['compact','evaluation_arrays']:assert sha(d/(n+'.npz'))==r[n+'_sha256']
ac=arrays(a.ash/'compact.npz');fc=arrays(a.fr/'compact.npz');ae=arrays(a.ash/'evaluation_arrays.npz');fe=arrays(a.fr/'evaluation_arrays.npz');checks={}
def check(name,x,y):
 assert x.shape==y.shape and x.dtype==y.dtype and np.array_equal(x,y),name
 checks[name]=dict(exact_array_equal=True,shape=list(x.shape),dtype=str(x.dtype),sha256=ah(x))
for k in ['raw_scores','reference_scores','w0','wm','queries','candidates','blocks','input_reps']:check(k,ac[k],fc[k])
for k in ['sampled_query_blocks','query_weights']:check(k,ae[k],fe[k])
for n in [25,10]:
 check(f'first_three_query_p{n}',ae[f'four_arm_p{n}'][:,:3,:],fe[f'six_arm_p{n}'][:,:3,:])
 check(f'first_three_query_bootstrap_p{n}',ae[f'query_only_bootstrap_p{n}'][:,:3],fe[f'query_only_bootstrap_p{n}'][:,:3])
 joint=f'joint_bootstrap_p{n}'
 if joint in ae or joint in fe:check(f'first_three_{joint}',ae[joint][:,:3],fe[joint][:,:3])
report=dict(status='MATCHED_COMMON_THREE_ARMS_BITWISE_PASS',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataset=ar['dataset'],kind=ar['kind'],checks=checks,ash_report_sha256=sha(a.ash/'evaluation_report.json'),fr_report_path=str((a.fr/'evaluation_report.json').resolve()),fr_report_sha256=sha(a.fr/'evaluation_report.json'),verification_source_sha256=sha(__file__),scope='Identity check only; ASH contrasts remain descriptive and are not included in the original FR simultaneous contrast family.')
(a.ash/'COMMON_THREE_ARM_MATCH.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],dataset=ar['dataset'],kind=ar['kind'],checks=len(checks))))
