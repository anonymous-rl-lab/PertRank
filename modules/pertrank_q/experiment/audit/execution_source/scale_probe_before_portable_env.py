"""Read-only scale invariance check on archived calibration fits only."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'baseline_B'
if not BASE.exists():BASE=ROOT.parent/'pertrank_boundary_learning_20261115'
sys.path.insert(0,str(BASE))
from base_training import torch,tensor,predict
from boundary_components import BoundaryNormalizer
from common import load,dump,subset,ref_support,precision,sha
from gate_model import ReliabilityGate
from ranking_ops import list_loss

def standardize(s,mask):
 count=mask.sum(1,keepdim=True);mu=(s*mask).sum(1,keepdim=True)/count
 c=torch.where(mask,s-mu,0.)
 sd=(c.square().sum(1,keepdim=True)/count).clamp_min(1e-12).sqrt()
 return c/sd

def probe(score,base,scale,own,ref,support,k):
 mask=torch.as_tensor(~own);x=tensor(score/scale[:,None]).requires_grad_()
 loss=list_loss(x,mask,tensor(support[0]),tensor(support[1]),k,.5)
 g=torch.autograd.grad(loss,x)[0].detach().numpy();valid=~own
 center=np.where(valid,x.detach().numpy()-np.sum(x.detach().numpy()*valid,1,keepdims=True)/valid.sum(1,keepdims=True),0)
 dot=(g*center).sum(1);den=(center**2).sum(1)
 radial=np.divide(dot**2,den,out=np.zeros_like(dot),where=den>0).sum()/np.sum(g**2)
 out={'joint_logscale_loss_derivative':float(dot.sum()),'gradient_squared_fraction_in_perquery_scale_direction':float(radial),'scale_conditions':{},'adaptive_scale_conditions':{}}
 for a in (.5,1.,2.):
  s=tensor(a*score/scale[:,None]);n=standardize(s,mask)
  out['scale_conditions'][str(a)]={'precision':float(precision(a*score,ref,own,k).mean()),'soft_overlap':float(-list_loss(s,mask,tensor(support[0]),tensor(support[1]),k,.5))}
  out['adaptive_scale_conditions'][str(a)]=float(-list_loss(n,mask,tensor(support[0]),tensor(support[1]),k,.5))
 assert len(set(z['precision'] for z in out['scale_conditions'].values()))==1
 out['max_adaptive_scale_difference']=max(out['adaptive_scale_conditions'].values())-min(out['adaptive_scale_conditions'].values())
 y=tensor(score/scale[:,None]).requires_grad_();ly=list_loss(standardize(y,mask),mask,tensor(support[0]),tensor(support[1]),k,.5);gy=torch.autograd.grad(ly,y)[0].detach().numpy()
 out['adaptive_logscale_gradient_dot']=float((gy*center).sum())
 # Original B correction projected onto per-query ranking-invariant affine transforms.
 meanb=np.sum(base*valid,1,keepdims=True)/valid.sum(1,keepdims=True);means=np.sum(score*valid,1,keepdims=True)/valid.sum(1,keepdims=True)
 cb=np.where(valid,base-meanb,0.);cs=np.where(valid,score-means,0.);db=(cb*cb).sum(1,keepdims=True)
 alpha=np.divide((cb*cs).sum(1,keepdims=True),db,out=np.ones_like(db),where=db>0)
 affine=means+alpha*(base-meanb);change=np.where(valid,(score-base)/scale[:,None],0.);affinechange=np.where(valid,(affine-base)/scale[:,None],0.)
 out['fitted_affine_probe']={'mean_positive_scale_slope':float(alpha.mean()),'minimum_scale_slope':float(alpha.min()),'affine_correction_energy_fraction':float(np.sum(affinechange**2)/np.sum(change**2)) if np.sum(change**2)>0 else None,
 'affine_precision':float(precision(affine,ref,own,k).mean()),'base_precision':float(precision(base,ref,own,k).mean()),
 'base_soft_overlap':float(-list_loss(tensor(base/scale[:,None]),mask,tensor(support[0]),tensor(support[1]),k,.5)),
 'affine_soft_overlap':float(-list_loss(tensor(affine/scale[:,None]),mask,tensor(support[0]),tensor(support[1]),k,.5)),
 'fitted_soft_overlap':float(-loss.detach())}
 return out

def main():
 out={'identity':'Read-only scale probe; fixed archived calibration fits; no training, no new held evaluation, no scale chosen as a predictor','conditions':{},'summary':{}}
 for source in ('D3','D1','D2'):
  for h in (0,1):
   cal=load(BASE/'views'/source/f'held{h}'/'calibration.npz');split=json.loads((BASE/'views'/source/f'held{h}'/'INNER_SPLIT.json').read_text());groups=np.array([split[str(t)] for t in cal['target_ids']])
   for stage in ('inner0','inner1','outer'):
    k=10 if stage=='outer' else 5;v=cal if stage=='outer' else subset(cal,np.flatnonzero(groups!=int(stage[-1])))
    sup=ref_support(v,k);stem=f'{source}_held{h}_boundary_context';norm=BoundaryNormalizer.from_file(BASE/'fits'/f'{stem}_seed2026111511_{stage}'/'normalization.npz')
    out['conditions'][f'{source}_held{h}_{stage}_baseline']=probe(v['base'],v['base'],norm.baseline_scale,v['ownmask'],v['reference'],sup,k)
    for seed in (2026111511,2026111512):
     fd=BASE/'fits'/f'{stem}_seed{seed}_{stage}';saved=torch.load(fd/'weights.pt',weights_only=True);m=ReliabilityGate(**saved['config']);m.load_state_dict(saved['state_dict']);n=BoundaryNormalizer.from_file(fd/'normalization.npz');score=predict(m,n.tensors(v,'boundary_context',k))
     out['conditions'][f'{source}_held{h}_{stage}_seed{seed}']=probe(score,v['base'],n.baseline_scale,v['ownmask'],v['reference'],sup,k)
  print('SOURCE_DONE',source,flush=True)
 for stage in ('inner','outer'):
  cs=[v for key,v in out['conditions'].items() if stage in key];fs=[v for key,v in out['conditions'].items() if stage in key and 'baseline' not in key]
  out['summary'][stage]={'conditions':len(cs),'precision_identical_across_scales':all(len(set(x['precision'] for x in v['scale_conditions'].values()))==1 for v in cs),
   'scale2_beats_scale1_soft_count':sum(v['scale_conditions']['2.0']['soft_overlap']>v['scale_conditions']['1.0']['soft_overlap'] for v in cs),
   'mean_soft_overlap_gain_scale2_minus1':float(np.mean([v['scale_conditions']['2.0']['soft_overlap']-v['scale_conditions']['1.0']['soft_overlap'] for v in cs])),
   'mean_gradient_squared_scale_fraction':float(np.mean([v['gradient_squared_fraction_in_perquery_scale_direction'] for v in cs])),
   'max_adaptive_scale_score_difference':float(max(v['max_adaptive_scale_difference'] for v in cs)),
   'max_abs_adaptive_logscale_gradient_dot':float(max(abs(v['adaptive_logscale_gradient_dot']) for v in cs)),
   'fitted_mean_scale_slope':float(np.mean([v['fitted_affine_probe']['mean_positive_scale_slope'] for v in fs])),
   'fitted_mean_affine_correction_energy_fraction':float(np.mean([v['fitted_affine_probe']['affine_correction_energy_fraction'] for v in fs if v['fitted_affine_probe']['affine_correction_energy_fraction'] is not None])),
   'fitted_mean_soft_gain':float(np.mean([v['fitted_affine_probe']['fitted_soft_overlap']-v['fitted_affine_probe']['base_soft_overlap'] for v in fs])),
   'affine_projection_mean_soft_gain':float(np.mean([v['fitted_affine_probe']['affine_soft_overlap']-v['fitted_affine_probe']['base_soft_overlap'] for v in fs])),
   'affine_projection_exact_precision_changed_count':sum(abs(v['fitted_affine_probe']['affine_precision']-v['fitted_affine_probe']['base_precision'])>1e-14 for v in fs)}
 checks={}
 for name,values in [('zero',[[0.,0.,0.,0.]]),('constant',[[2.,2.,2.,2.]]),('nearzero',[[0.,1e-9,-1e-9,0.]])]:
  z=tensor(values).requires_grad_();mask=torch.tensor([[True,True,True,False]])
  normalized=standardize(z,mask);ls=list_loss(normalized,mask,tensor([[1.,0.,0.,0.]]),tensor([[0.,0.,1.,0.]]),1,.5)
  g=torch.autograd.grad(ls,z)[0];assert torch.isfinite(g).all()
  checks[name]={'finite_forward_and_backward':True,'maximum_absolute_gradient':float(g.abs().max())}
 out['degenerate_sd_checks']=checks
 out['script_sha256']=sha(__file__);dump(ROOT/'audit/scale_probe.json',out);print(json.dumps(out['summary'],indent=2))
if __name__=='__main__':main()
