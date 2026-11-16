#!/usr/bin/env python3
"""Source-specific boundary learning with an inner-only feasibility barrier."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,time
import numpy as np
from base_training import torch,tensor,predict,GateNormalizer
from gate_model import ReliabilityGate,list_loss
from boundary_components import BoundaryNormalizer,make_boundary_model,build_pair_labels,pair_loss
from common import SOURCES,load,sha,dump,subset,inner_ids,ref_support,precision
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'pertrank_geometry_gate_20261115'
ARMS=('base_context','boundary_context','boundary_pair');NEW_ARMS=ARMS[1:]
SEEDS=(2026111511,2026111512);CHECKPOINTS=(0,50,100,200,300)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def safe(relative):
 p=Path(relative);assert not p.is_absolute() and '..' not in p.parts
 return ROOT/p

def import_a():
 manifest_path=ROOT/'historical/IMPORTED_A.json'
 if manifest_path.exists():
  manifest=json.loads(manifest_path.read_text())
  for path,value in manifest['files'].items():assert sha(safe(path))==value,('changed imported A',path)
  return manifest
 files={};tasks={};prior_plan=json.loads((PREVIOUS/'PLAN_LOCK.json').read_text())
 for name,expected in prior_plan['code_hashes'].items():assert sha(PREVIOUS/name)==expected
 for source in SOURCES:
  lp=PREVIOUS/'predictions'/f'{source}_PREDICTION_LOCK.json';lock=json.loads(lp.read_text())
  assert lock['plan_sha256']==sha(PREVIOUS/'PLAN_LOCK.json')
  for h in (0,1):
   task=f'{source}_held{h}';entry=lock['tasks'][task];oldp=PREVIOUS/entry['path']
   assert sha(oldp)==entry['sha256'];a=load(oldp);ai=a['arm_names'].tolist().index('neural_no_extra_geometry')
   assert np.array_equal(a['seeds'],np.array(SEEDS))
   dest=ROOT/'historical/predictions'/f'{task}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(dest,scores=a['scores'][ai],seeds=a['seeds'],**{k:a[k] for k in ('base','raw','euclidean','ownmask','query_ids','query_block','target_ids')})
   files[str(dest.relative_to(ROOT))]=sha(dest);fits={}
   for seed in SEEDS:
    original=entry['fits'][f'neural_no_extra_geometry_seed{seed}'];oldfit=PREVIOUS/original['outer_fit']
    for n,key in [('weights.pt','weights_sha256'),('normalization.npz','normalization_sha256')]:assert sha(oldfit/n)==original[key]
    fd=ROOT/'historical/fits'/f'{task}_base_context_seed{seed}';fd.mkdir(parents=True,exist_ok=True)
    for name in ('weights.pt','normalization.npz','curve.json'):
     shutil.copy2(oldfit/name,fd/name);files[str((fd/name).relative_to(ROOT))]=sha(fd/name)
    fm=json.loads((oldfit/'FIT.json').read_text());fm.update(arm='base_context',historical_import=True,original_arm='neural_no_extra_geometry',original_weights_sha256=original['weights_sha256'],original_normalization_sha256=original['normalization_sha256'],original_fit_sha256=sha(oldfit/'FIT.json'),original_prediction_lock_sha256=sha(lp))
    dump(fd/'FIT.json',fm);files[str((fd/'FIT.json').relative_to(ROOT))]=sha(fd/'FIT.json')
    fits[f'base_context_seed{seed}']={**original,'outer_fit':str(fd.relative_to(ROOT)),'historical_import':True,'original_arm':'neural_no_extra_geometry'}
   tasks[task]=dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest),fits=fits,original_prediction_sha256=entry['sha256'],original_prediction_lock_sha256=sha(lp))
 manifest=dict(identity='All prior neural_no_extra_geometry seeds and source halves imported without refitting or selection',original_plan_sha256=sha(PREVIOUS/'PLAN_LOCK.json'),tasks=tasks,files=files)
 dump(manifest_path,manifest);return manifest

def prepare():
 if (ROOT/'PLAN_LOCK.json').exists():raise RuntimeError('Existing plan lock preserved')
 assert json.loads((ROOT/'BOUNDARY_CHECKS.json').read_text())['all_passed']
 imported=import_a();tasks={};input_hashes={}
 for s in SOURCES:
  ip=ROOT/'inputs'/f'{s}_neural_inputs.npz';v=load(ip);input_hashes[str(ip.relative_to(ROOT))]=sha(ip)
  for h in (0,1):
   task=f'{s}_held{h}';cal=subset(v,np.flatnonzero(v['targethalf']!=h));held=subset(v,np.flatnonzero(v['targethalf']==h))
   vp=ROOT/'views'/s/f'held{h}';vp.mkdir(parents=True,exist_ok=True)
   ep=ROOT/'evaluation'/s/f'held{h}.npz';ep.parent.mkdir(parents=True,exist_ok=True)
   cp=vp/'calibration.npz';hp=vp/'held_features.npz'
   np.savez_compressed(cp,**cal);np.savez_compressed(hp,**{k:a for k,a in held.items() if not k.startswith('reference')})
   np.savez_compressed(ep,**{k:held[k] for k in ('reference','reference_replicates','ownmask','query_ids','target_ids','query_block')})
   split=inner_ids({s:cal},2026111516);dump(vp/'INNER_SPLIT.json',split)
   a=load(safe(imported['tasks'][task]['path']))
   for k in ('base','raw','euclidean','ownmask','query_ids','query_block','target_ids'):assert np.array_equal(a[k],held[k]),(task,k)
   tasks[task]=dict(held=len(held['target_ids']),calibration=len(cal['target_ids']),calibration_sha256=sha(cp),held_features_sha256=sha(hp),evaluation_sha256=sha(ep),inner_split_sha256=sha(vp/'INNER_SPLIT.json'))
 names=('train_boundary.py','base_training.py','boundary_components.py','gate_model.py','ranking_ops.py','common.py','score_boundary.py','PROTOCOL.md')
 dump(ROOT/'PLAN_LOCK.json',dict(created_utc=now(),arms=list(ARMS),seeds=list(SEEDS),source_specific=True,tasks=tasks,code_hashes={n:sha(ROOT/n) for n in names},input_hashes=input_hashes,imported_a_sha256=sha(ROOT/'historical/IMPORTED_A.json'),checks_sha256=sha(ROOT/'BOUNDARY_CHECKS.json'),identity='Exploratory previously accessed data; not independent confirmation',torch_version=torch.__version__))
 print('PREPARED',flush=True)

def check_plan():
 plan=json.loads((ROOT/'PLAN_LOCK.json').read_text())
 for n,v in plan['code_hashes'].items():assert sha(safe(n))==v,('changed code',n)
 for n,v in plan['input_hashes'].items():assert sha(safe(n))==v,('changed input',n)
 assert sha(ROOT/'historical/IMPORTED_A.json')==plan['imported_a_sha256'];import_a()
 return plan

def read_calibration(plan,s,h):
 task=f'{s}_held{h}';vp=ROOT/'views'/s/f'held{h}';meta=plan['tasks'][task]
 for name,key in [('calibration.npz','calibration_sha256'),('held_features.npz','held_features_sha256'),('INNER_SPLIT.json','inner_split_sha256')]:assert sha(vp/name)==meta[key]
 return load(vp/'calibration.npz'),json.loads((vp/'INNER_SPLIT.json').read_text())

def parts(m,t,support,labels,k,arm):
 score=m(t['features'],t['raw'],t['anchor']);normalized=score/t['scale']
 rank=list_loss(normalized,t['mask'],support[0],support[1],k,.5) if arm=='boundary_context' else pair_loss(normalized,labels,.5)
 residual=.05*((score-t['base'])/t['scale'])[t['mask']].square().mean()
 ridge=1e-4*torch.cat([p.flatten() for p in m.h[-1].parameters()]).square().mean()
 return rank,residual,ridge

def gradnorm(value,m):
 gradients=torch.autograd.grad(value,list(m.parameters()),retain_graph=True,allow_unused=True)
 return float(torch.sqrt(sum((g.square().sum() for g in gradients if g is not None),start=torch.tensor(0.))))

def metrics(m,t,v,k):
 p=predict(m,t)
 with torch.no_grad():g=m.gate(t['features']).numpy()
 valid=~v['ownmask']
 return dict(precision=float(precision(p,v['reference'],v['ownmask'],k).mean()),max_abs_gate=float(abs(g[valid]).max()),fraction_saturated=float((abs(g[valid])>.99*np.log(3)).mean()),rms_score_change=float(np.sqrt(np.mean((p[valid]-v['base'][valid])**2))))

def fit(v,validation,arm,seed,path,epochs,k):
 if (path/'FIT.json').exists():
  payload=torch.load(path/'weights.pt',weights_only=True);m=ReliabilityGate(**payload['config']);m.load_state_dict(payload['state_dict'])
  return m,BoundaryNormalizer.from_file(path/'normalization.npz'),json.loads((path/'curve.json').read_text())
 if path.exists():raise RuntimeError(f'Incomplete fit preserved at {path}')
 path.mkdir(parents=True);start=time.monotonic();norm=BoundaryNormalizer(v,k);norm.save(path/'normalization.npz')
 t=norm.tensors(v,arm,k);vt=norm.tensors(validation,arm,k) if validation is not None else None
 support_np=ref_support(v,k);support=tensor(support_np);labels=build_pair_labels(v,k,support_np)
 dump(path/'SUPERVISION.json',{**labels['stats'],'support_float64_sha256':hashlib.sha256(support_np.tobytes()).hexdigest()})
 m=make_boundary_model(seed);assert np.array_equal(predict(m,t),v['base'])
 opt=torch.optim.Adam(m.parameters(),lr=.01);curve=[];grad=0.
 for epoch in range(epochs+1):
  if epoch in CHECKPOINTS or epoch==epochs:
   rank,residual,ridge=parts(m,t,support,labels,k,arm)
   rec=dict(epoch=epoch,loss=float((rank+residual+ridge).detach()),ranking_loss=float(rank.detach()),residual_penalty=float(residual.detach()),head_ridge=float(ridge.detach()),gradient_norm=grad,ranking_gradient_norm=gradnorm(rank,m),residual_gradient_norm=gradnorm(residual,m),ridge_gradient_norm=gradnorm(ridge,m),train=metrics(m,t,v,k))
   if validation is not None:rec['validation']=metrics(m,vt,validation,k)
   curve.append(rec)
  if epoch==epochs:break
  opt.zero_grad();loss=sum(parts(m,t,support,labels,k,arm))
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
  loss.backward()
  for p in m.parameters():
   if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient')
  grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),5.));opt.step()
 torch.save(dict(config=m.get_config(),state_dict=m.state_dict()),path/'weights.pt')
 dump(path/'curve.json',curve)
 dump(path/'FIT.json',dict(arm=arm,seed=seed,epochs=epochs,selected_epoch=epochs,k=k,source_specific=True,training_target_ids=v['target_ids'].tolist(),validation_target_ids=None if validation is None else validation['target_ids'].tolist(),duration_seconds=time.monotonic()-start,parameters=sum(p.numel() for p in m.parameters())))
 return m,norm,curve

def run_inner(source):
 plan=check_plan();dest=ROOT/'selections'/f'{source}.json'
 if dest.exists():raise RuntimeError('Complete selection already exists')
 tasks={}
 for h in (0,1):
  task=f'{source}_held{h}';cal,groups=read_calibration(plan,source,h);g=np.array([groups[str(t)] for t in cal['target_ids']]);task_choices={}
  for arm in NEW_ARMS:
   for seed in SEEDS:
    curves=[];inner=[]
    for ih in (0,1):
     tr=subset(cal,np.flatnonzero(g!=ih));va=subset(cal,np.flatnonzero(g==ih));fp=ROOT/'fits'/f'{task}_{arm}_seed{seed}_inner{ih}'
     _,_,curve=fit(tr,va,arm,seed,fp,300,5);curves.append(curve)
     inner.append(dict(path=str(fp.relative_to(ROOT)),fit_sha256=sha(fp/'FIT.json'),curve_sha256=sha(fp/'curve.json'),weights_sha256=sha(fp/'weights.pt'),normalization_sha256=sha(fp/'normalization.npz')))
    choices=[]
    for ci,epoch in enumerate(CHECKPOINTS):
     gains=[c[ci]['validation']['precision']-c[0]['validation']['precision'] for c in curves]
     choices.append(dict(epoch=epoch,gains=gains,mean_gain=float(np.mean(gains)),eligible=all(x>=0 for x in gains)))
    chosen=max((c for c in choices if c['eligible']),key=lambda c:(c['mean_gain'],-c['epoch']))['epoch']
    task_choices[f'{arm}_seed{seed}']=dict(selected_epoch=chosen,selection=choices,inner_fits=inner)
    print(source,h,arm,seed,'selected',chosen,flush=True)
  tasks[task]=task_choices
 dump(dest,dict(source=source,plan_sha256=sha(ROOT/'PLAN_LOCK.json'),created_utc=now(),tasks=tasks))
 print('INNER_COMPLETE',source,flush=True)

def feasibility():
 check_plan();dest=ROOT/'FEASIBILITY.json'
 if dest.exists():raise RuntimeError('Feasibility already frozen')
 selection={s:json.loads((ROOT/'selections'/f'{s}.json').read_text()) for s in SOURCES}
 for s,z in selection.items():assert z['plan_sha256']==sha(ROOT/'PLAN_LOCK.json')
 arms={}
 for arm in NEW_ARMS:
  selected=[]
  for s,z in selection.items():
   for task,values in z['tasks'].items():
    for seed in SEEDS:
     entry=values[f'{arm}_seed{seed}'];selected.append(dict(source=s,task=task,seed=seed,epoch=entry['selected_epoch']))
  nonzero=[x for x in selected if x['epoch']>0];sources=sorted(set(x['source'] for x in nonzero))
  arms[arm]=dict(total_selections=len(selected),nonzero_selections=len(nonzero),sources_with_nonzero_selection=sources,passed=len(nonzero)>=4 and len(sources)>=2,selections=selected)
 value=dict(created_utc=now(),plan_sha256=sha(ROOT/'PLAN_LOCK.json'),selection_sha256={s:sha(ROOT/'selections'/f'{s}.json') for s in SOURCES},arms=arms,proceed_to_outer=any(x['passed'] for x in arms.values()),identity='Resource feasibility from inner validation only, not external performance evidence')
 dump(dest,value);print(json.dumps(value,indent=2),flush=True)

def run_outer(source):
 plan=check_plan();feasible=json.loads((ROOT/'FEASIBILITY.json').read_text());assert feasible['proceed_to_outer']
 assert feasible['plan_sha256']==sha(ROOT/'PLAN_LOCK.json')
 for s,h in feasible['selection_sha256'].items():assert sha(ROOT/'selections'/f'{s}.json')==h
 selection=json.loads((ROOT/'selections'/f'{source}.json').read_text());imported=import_a();lp=ROOT/'predictions'/f'{source}_PREDICTION_LOCK.json'
 if lp.exists():raise RuntimeError('Completed predictions preserved')
 tasks={}
 for h in (0,1):
  task=f'{source}_held{h}';cal,_=read_calibration(plan,source,h);held=load(ROOT/'views'/source/f'held{h}'/'held_features.npz')
  prior=imported['tasks'][task];a=load(safe(prior['path']));scores=np.empty((3,2)+held['base'].shape);scores[0]=a['scores'];fits=dict(prior['fits'])
  for ai,arm in enumerate(NEW_ARMS,1):
   for si,seed in enumerate(SEEDS):
    choice=selection['tasks'][task][f'{arm}_seed{seed}'];epoch=choice['selected_epoch'];fp=ROOT/'fits'/f'{task}_{arm}_seed{seed}_outer'
    m,norm,_=fit(cal,None,arm,seed,fp,epoch,10);scores[ai,si]=predict(m,norm.tensors(held,arm,10))
    fits[f'{arm}_seed{seed}']={**choice,'outer_fit':str(fp.relative_to(ROOT)),'weights_sha256':sha(fp/'weights.pt'),'normalization_sha256':sha(fp/'normalization.npz')}
    print(source,h,arm,seed,'outer',epoch,flush=True)
  pp=ROOT/'predictions'/f'{task}.npz';pp.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(pp,scores=scores,arm_names=np.array(ARMS),seeds=np.array(SEEDS),**{k:held[k] for k in ('base','raw','euclidean','ownmask','query_ids','query_block','target_ids')})
  tasks[task]=dict(path=str(pp.relative_to(ROOT)),sha256=sha(pp),fits=fits)
 dump(lp,dict(plan_sha256=sha(ROOT/'PLAN_LOCK.json'),feasibility_sha256=sha(ROOT/'FEASIBILITY.json'),prediction_time_utc=now(),tasks=tasks));print('FROZEN',source,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
 g.add_argument('--prepare',action='store_true');g.add_argument('--inner',choices=SOURCES);g.add_argument('--feasibility',action='store_true');g.add_argument('--outer',choices=SOURCES)
 a=p.parse_args()
 if a.prepare:prepare()
 elif a.inner:run_inner(a.inner)
 elif a.feasibility:feasibility()
 else:run_outer(a.outer)
