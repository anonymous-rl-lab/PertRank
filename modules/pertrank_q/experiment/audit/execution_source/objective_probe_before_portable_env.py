"""Read-only diagnostic: B label/objective/temperature and pool-size fidelity.
No fitting, checkpoint selection, new prediction rule, or held scoring.
"""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'baseline_B'
if not BASE.exists():BASE=ROOT.parent/'pertrank_boundary_learning_20261115'
sys.path.insert(0,str(BASE))
from base_training import torch,tensor,predict
from boundary_components import BoundaryNormalizer,make_boundary_model
from common import load,dump,subset,ref_support,membership,precision,sha
from gate_model import ReliabilityGate
from ranking_ops import soft_topk,list_loss

def cosine(a,b):
 a=np.asarray(a).ravel();b=np.asarray(b).ravel();den=np.linalg.norm(a)*np.linalg.norm(b)
 return float(a@b/den) if den else None

def grads(score,mask,labels,k,tau):
 s=tensor(score).requires_grad_();loss=list_loss(s,mask,tensor(labels[0]),tensor(labels[1]),k,tau)
 return torch.autograd.grad(loss,s)[0].detach().numpy()

def exact_expected(score,own,labels,k):
 return float((membership(score,own,k)*labels).sum(2).mean()/k)

def pool_probe(v,norm,k):
 t=norm.tensors(v,'boundary_context',k);score=v['base']/norm.baseline_scale[:,None]
 hard=membership(v['reference'],v['ownmask'],k);support=ref_support(v,k)
 pool={'n_targets':len(v['target_ids']),'k':k,'queries':len(v['query_ids']),
       'selection_fraction_range':[float((k/(~v['ownmask']).sum(1)).min()),float((k/(~v['ownmask']).sum(1)).max())],
       'bootstrap_label_mass_moved_from_nominal_topk':float(np.abs(support-hard).sum(2).mean()/(2*k)),
       'reference_hard_topk_vs_bootstrap_support_topk_membership_change':float(np.mean([np.abs(membership(support[d],v['ownmask'],k)[0]-hard[d]).sum(1).mean()/(2*k) for d in (0,1)])),
       'baseline_nominal_precision':float(precision(v['base'],v['reference'],v['ownmask'],k).mean()),
       'baseline_expected_bootstrap_precision':exact_expected(v['base'],v['ownmask'],support,k),'temperatures':{}}
 own=v['ownmask'];hm=membership(v['base'],own,k);band=membership(v['base'],own,k+2)-membership(v['base'],own,k-2)
 for tau in (.5,.25,.1):
  p=np.stack([soft_topk(tensor(sgn*score),t['mask'],k,tau).detach().numpy() for sgn in (1,-1)])
  grad_bs=grads(score,t['mask'],support,k,tau);grad_hard=grads(score,t['mask'],hard,k,tau)
  curv=p*(1-p)/tau
  pool['temperatures'][str(tau)]={'soft_vs_hard_predictor_membership_mass_moved':float(np.abs(p-hm).sum(2).mean()/(2*k)),
     'soft_nominal_precision':float((p*hard).sum(2).mean()/k),
     'soft_bootstrap_precision':float((p*support).sum(2).mean()/k),
     'gradient_cosine_bootstrap_vs_nominal_labels':cosine(grad_bs,grad_hard),
     'bootstrap_gradient_absolute_mass_with_opposite_nominal_sign':float(np.abs(grad_bs)[grad_bs*grad_hard<0].sum()/np.abs(grad_bs).sum()),
     'threshold_curvature_mass_in_rank_K_plusminus2_band':float((curv*band).sum()/curv.sum())}
 return pool,support,hard,t

def main():
 out={'identity':'Read-only exploratory training-objective probe; archived calibrations only, no held-feature/ref join and no new fit',
      'code':{n:sha(BASE/n) for n in ('train_boundary.py','ranking_ops.py','common.py','boundary_components.py')},
      'pools':{},'fits':{},'curve_audit':{}}
 for source in ('D3','D1','D2'):
  for h in (0,1):
   cal=load(BASE/'views'/source/f'held{h}'/'calibration.npz');split=json.loads((BASE/'views'/source/f'held{h}'/'INNER_SPLIT.json').read_text());g=np.array([split[str(x)] for x in cal['target_ids']])
   for stage in ('inner0','inner1','outer'):
    k=10 if stage=='outer' else 5;v=cal if stage=='outer' else subset(cal,np.flatnonzero(g!=int(stage[-1])))
    task=f'{source}_held{h}';stem=f'{task}_boundary_context';f0=BASE/'fits'/f'{stem}_seed2026111511_{stage}'
    norm=BoundaryNormalizer.from_file(f0/'normalization.npz');entry,support,hard,t=pool_probe(v,norm,k)
    pname=f'{task}_{stage}';out['pools'][pname]=entry
    for seed in (2026111511,2026111512):
     fd=BASE/'fits'/f'{stem}_seed{seed}_{stage}';saved=torch.load(fd/'weights.pt',weights_only=True);model=ReliabilityGate(**saved['config']);model.load_state_dict(saved['state_dict']);pred=predict(model,t)
     normalized=pred/norm.baseline_scale[:,None];gb=grads(normalized,t['mask'],support,k,.5);gh=grads(normalized,t['mask'],hard,k,.5)
     sb=model(t['features'],t['raw'],t['anchor'])/t['scale'];lb=list_loss(sb,t['mask'],tensor(support[0]),tensor(support[1]),k,.5);lh=list_loss(sb,t['mask'],tensor(hard[0]),tensor(hard[1]),k,.5)
     pb=torch.cat([x.ravel() for x in torch.autograd.grad(lb,tuple(model.parameters()),retain_graph=True)]).detach().numpy();ph=torch.cat([x.ravel() for x in torch.autograd.grad(lh,tuple(model.parameters()))]).detach().numpy()
     rec={'epochs':json.loads((fd/'FIT.json').read_text())['epochs'],
          'nominal_precision_gain':float((precision(pred,v['reference'],v['ownmask'],k)-precision(v['base'],v['reference'],v['ownmask'],k)).mean()),
          'expected_bootstrap_precision_gain':exact_expected(pred,v['ownmask'],support,k)-exact_expected(v['base'],v['ownmask'],support,k),
          'score_gradient_cosine_bootstrap_vs_nominal':cosine(gb,gh),'parameter_gradient_cosine_bootstrap_vs_nominal':cosine(pb,ph)}
     out['fits'][f'{pname}_seed{seed}']=rec
     curve=json.loads((fd/'curve.json').read_text());intervals=[]
     for a,b in zip(curve,curve[1:]):
      intervals.append({'epochs':[a['epoch'],b['epoch']],'soft_objective_improvement':a['ranking_loss']-b['ranking_loss'],'exact_train_precision_gain':b['train']['precision']-a['train']['precision'],
                        'exact_validation_precision_gain':None if 'validation' not in a else b['validation']['precision']-a['validation']['precision']})
     out['curve_audit'][f'{pname}_seed{seed}']=intervals
  print('SOURCE_DONE',source,flush=True)
 summary={}
 for stage in ('inner','outer'):
  pp=[p for key,p in out['pools'].items() if stage in key]
  fs=[v for key,v in out['fits'].items() if stage in key]
  curves=[z for key,v in out['curve_audit'].items() if stage in key for z in v]
  improvers=[z for z in curves if z['soft_objective_improvement']>0]
  summary[stage]={'pools':len(pp),'fits':len(fs),'mean_label_mass_moved':float(np.mean([x['bootstrap_label_mass_moved_from_nominal_topk'] for x in pp])),
   'mean_bootstrap_vs_nominal_optimal_membership_change':float(np.mean([x['reference_hard_topk_vs_bootstrap_support_topk_membership_change'] for x in pp])),
   'soft_improving_curve_steps':len(improvers),'soft_improving_but_exact_train_declines':sum(x['exact_train_precision_gain']<0 for x in improvers),
   'soft_improving_but_exact_validation_declines':sum(x['exact_validation_precision_gain'] is not None and x['exact_validation_precision_gain']<0 for x in improvers),
   'mean_fitted_parameter_gradient_cosine':float(np.mean([x['parameter_gradient_cosine_bootstrap_vs_nominal'] for x in fs])),
   'temperature_averages':{str(t):{m:float(np.mean([x['temperatures'][str(t)][m] for x in pp])) for m in pp[0]['temperatures'][str(t)]} for t in (.5,.25,.1)}}
 out['summary']=summary;out['probe_script_sha256']=sha(__file__);dump(ROOT/'audit/objective_probe.json',out);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
