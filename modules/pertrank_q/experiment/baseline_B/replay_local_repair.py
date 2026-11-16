"""Replay the fixed post hoc diagnostic, preserving primary and recorded results."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'audit'))
import numpy as np
from rebuild_boundary import rebuild
from common import load,sha,dump,membership,SOURCES
from score_boundary import checks_before_reference,ARMS,SEEDS,K
from local_repair_probe import repair,summarize
from replay_boundary import numeric_fields

def main():
 rebuild();_,packets,_=checks_before_reference();d=ROOT/'audit/local_repair';frozen=json.loads((d/'SUMMARY.json').read_text());plan=json.loads((d/'PLAN.json').read_text())
 assert sha(ROOT/'audit/local_repair_probe_frozen.py')==plan['script_sha256']
 tasks={};sources={};arrays_checked=0
 for s in SOURCES:
  halves=[]
  for h in (0,1):
   task=f'{s}_held{h}';p,ep,_=packets[task];old=load(d/'predictions'/f'{task}.npz')
   local=np.stack([repair(x,p['base'],p['ownmask'])[0] for x in p['scores'].mean(1)])
   seed=np.stack([[repair(x,p['base'],p['ownmask'])[0] for x in arm] for arm in p['scores']])
   assert np.array_equal(local,old['local_membership']);assert np.array_equal(seed,old['seed_local_membership']);arrays_checked+=2
   e=load(ep);ref=membership(e['reference'],e['ownmask'],K)
   a={name:(old[name+'_membership']*ref).sum(-1).mean(2 if name.startswith('seed_') else 1)/K for name in ('local','seed_local','unrestricted','seed_unrestricted')}
   a['base']=(old['base_membership']*ref).sum(-1).mean(0)/K;a.update(query_ids=p['query_ids'],query_block=p['query_block']);tasks[task]=summarize(a);halves.append(a)
  a={k:np.stack([v[k] for v in halves]).mean(0) for k in ('local','seed_local','unrestricted','seed_unrestricted','base')};a.update(query_ids=halves[0]['query_ids'],query_block=halves[0]['query_block']);sources[s]=summarize(a)
 got={'tasks':tasks,'sources':sources};expected={k:frozen[k] for k in got};assert got==expected
 numbers=numeric_fields(got)
 report={'passed':True,'training_performed':False,'post_hoc_identity_preserved':True,'membership_arrays_exact':arrays_checked,'numeric_source_task_fields_exact':len(numbers),'maximum_difference':0}
 dump(ROOT/'LOCAL_REPAIR_REPLAY_CHECK.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
