"""Calibration-only boundary-reference instruction probe, no held scoring."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from base_training import np
from common import load,subset,inner_ids,ref_support,dump,SOURCES
from boundary_components import build_pair_labels
out={'identity':'Calibration-only empirical reference stability; not predictive performance','pools':{}}
for s in SOURCES:
 v=load(ROOT/'inputs'/f'{s}_neural_inputs.npz')
 for h in (0,1):
  cal=subset(v,np.flatnonzero(v['targethalf']!=h));gm=inner_ids({s:cal},2026111516);g=np.array([gm[str(t)] for t in cal['target_ids']])
  for ih in (0,1):
   tr=subset(cal,np.flatnonzero(g!=ih));pi=ref_support(tr,5);z=build_pair_labels(tr,5,pi)
   q=z['q'].numpy();i=z['i'].numpy();j=z['j'].numpy();d=(z['direction'].numpy()<0).astype(int)
   diff=abs(pi[d,q,i]-pi[d,q,j]);r=dict(z['stats']);r.update(targets=len(tr['target_ids']),nonzero_support_gap_quantiles=np.quantile(diff,[.1,.5,.9]).tolist(),fraction_edges_gap_ge_half=float((diff>=.5).mean()))
   out['pools'][f'{s}_held{h}_inner{ih}']=r
p=ROOT/'audit/CALIBRATION_SUPPORT_PROBE.json'
if p.exists():assert json.loads(p.read_text())==out
else:dump(p,out)
for s in SOURCES:
 a=[v for k,v in out['pools'].items() if k.startswith(s+'_')]
 print(s,'median support gap',np.mean([x['nonzero_support_gap_quantiles'][1] for x in a]),'fraction edges gap>=.5',np.mean([x['fraction_edges_gap_ge_half'] for x in a]),'wrong confidence fraction',sum(x['wrong_confidence_weight'] for x in a)/sum(x['total_confidence_weight'] for x in a))
