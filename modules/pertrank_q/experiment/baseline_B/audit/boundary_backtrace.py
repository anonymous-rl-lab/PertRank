"""Post-score descriptive localization; never selects models or thresholds."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from common import SOURCES,load,membership,dump
from score_boundary import checks_before_reference

def run():
 plan,packets,checks=checks_before_reference()
 out={'identity':'Post-score descriptive localization, no model selection','tasks':{},'sources':{},'training':{}}
 for task,(p,ep,_) in packets.items():
  e=load(ep);own=p['ownmask'];b=membership(p['base'],own,10);ref=membership(e['reference'],own,10)
  band=(membership(p['base'],own,12)-membership(p['base'],own,8))>0
  q=len(p['query_ids']);out['tasks'][task]={}
  for ai,a in enumerate(p['arm_names']):
   pred=p['scores'][ai].mean(0);diff=membership(pred,own,10)-b;enter=np.maximum(diff,0);leave=np.maximum(-diff,0)
   parts={}
   for name,mask in [('within_fixed_local_band',band),('outside_fixed_local_band',~band)]:
    parts[name]={'entered':float((enter*mask).sum()),'exited':float((leave*mask).sum()),'entered_support':float((enter*mask*ref).sum()),'exited_support':float((leave*mask*ref).sum()),'delta_P10':float((diff*mask*ref).sum()/(2*q*10))}
   parts['total_delta_P10']=float((diff*ref).sum()/(2*q*10))
   assert abs(parts['total_delta_P10']-sum(parts[k]['delta_P10'] for k in ('within_fixed_local_band','outside_fixed_local_band')))<1e-14
   out['tasks'][task][str(a)]=parts
 for s in SOURCES:
  out['sources'][s]={}
  for a in ('base_context','boundary_context','boundary_pair'):
   z=[out['tasks'][f'{s}_held{h}'][a] for h in (0,1)]
   out['sources'][s][a]={'total_delta_P10':float(np.mean([x['total_delta_P10'] for x in z]))}
   for region in ('within_fixed_local_band','outside_fixed_local_band'):
    out['sources'][s][a][region]={k:float(np.mean([x[region][k] for x in z])) for k in z[0][region]}
 for a in ('boundary_context','boundary_pair'):
  out['training'][a]={}
  for s in SOURCES:
   records=[]
   for p in sorted((ROOT/'fits').glob(f'{s}_*{a}_*_outer/curve.json')):
    c=json.loads(p.read_text());b,e=c[0],c[-1]
    records.append({'fit':p.parent.name,'epoch':e['epoch'],'train_P10_gain':e['train']['precision']-b['train']['precision'],'rank_loss_change':e['ranking_loss']-b['ranking_loss'],'residual_penalty':e['residual_penalty'],'rank_grad':e['ranking_gradient_norm'],'penalty_grad':e['residual_gradient_norm']})
   out['training'][a][s]={'mean_train_P10_gain':float(np.mean([v['train_P10_gain'] for v in records])),'fits':records}
 return out
if __name__=='__main__':
 out=run();p=ROOT/'audit/BOUNDARY_BACKTRACE.json'
 if p.exists():assert json.loads(p.read_text())==out
 else:dump(p,out)
 for s,z in out['sources'].items():
  print(s)
  for a,v in z.items():print(a,'within',v['within_fixed_local_band']['delta_P10'],'outside',v['outside_fixed_local_band']['delta_P10'],'total',v['total_delta_P10'])
 print('training',{a:{s:v['mean_train_P10_gain'] for s,v in z.items()} for a,z in out['training'].items()})
