#!/usr/bin/env python3
"""A bounded 2 x 2 B update: ranking-scale objective and ensemble selection.

No evaluation reference is opened by this program. Existing B predictions and
matching outer fits are imported, not overwritten. Inner replay is necessary
because the predecessor retained only its final inner checkpoint.
"""
from pathlib import Path
import argparse, datetime, hashlib, importlib, json, sys, time
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCES=('D3','D1','D2')
SEEDS=(2026111511,2026111512)
CHECKPOINTS=(0,50,100,200,300)
OBJECTIVES=('frozen_scale','rank_scale')
ARMS=('old_B','B_joint_selection','B_rank_scale','B_rank_scale_joint')
BASE=None

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def configure(base):
 global BASE,torch,tensor,predict,BoundaryNormalizer,make_boundary_model,list_loss,subset,ref_support,precision
 BASE=Path(base).resolve()
 for runtime in (ROOT/'.runtime',ROOT.parent/'pertrank_neural_rank_20261115'/'.runtime'):
  if runtime.is_dir():sys.path.insert(0,str(runtime));break
 sys.path.insert(0,str(BASE))
 bt=importlib.import_module('base_training');bc=importlib.import_module('boundary_components');co=importlib.import_module('common')
 torch=bt.torch;tensor=bt.tensor;predict=bt.predict
 BoundaryNormalizer=bc.BoundaryNormalizer;make_boundary_model=bc.make_boundary_model
 list_loss=importlib.import_module('gate_model').list_loss
 subset=co.subset;ref_support=co.ref_support;precision=co.precision

def ranking_scores(score,t,objective):
 if objective=='frozen_scale':return score/t['scale']
 if objective!='rank_scale':raise ValueError(objective)
 mask=t['mask'];n=mask.sum(1,keepdim=True)
 mean=torch.where(mask,score,0.).sum(1,keepdim=True)/n
 centered=score-mean
 variance=torch.where(mask,centered.square(),0.).sum(1,keepdim=True)/n
 # The floor addresses only degenerate/near-constant score rows. It is not
 # tuned, and all nondegenerate derivatives include the changing scale.
 floor=1e-6*t['scale']
 sd=torch.sqrt(torch.maximum(variance,floor.square()))
 return centered/sd

def parts(model,t,support,k,objective):
 score=model(t['features'],t['raw'],t['anchor'])
 rank=list_loss(ranking_scores(score,t,objective),t['mask'],support[0],support[1],k,.5)
 residual=.05*((score-t['base'])/t['scale'])[t['mask']].square().mean()
 ridge=1e-4*torch.cat([p.flatten() for p in model.h[-1].parameters()]).square().mean()
 return rank,residual,ridge

def metric(model,t,v,k):
 p=predict(model,t)
 with torch.no_grad():g=model.gate(t['features']).numpy()
 valid=~v['ownmask']
 return dict(precision=float(precision(p,v['reference'],v['ownmask'],k).mean()),max_abs_gate=float(abs(g[valid]).max()),fraction_saturated=float((abs(g[valid])>.99*np.log(3)).mean()),rms_score_change=float(np.sqrt(np.mean((p[valid]-v['base'][valid])**2))))

def checkpoint_path(path,epoch):
 return path/'weights.pt' if epoch==300 else path/f'checkpoint_{epoch:03d}.pt'
def save_model(model,p):torch.save(dict(config=model.get_config(),state_dict=model.state_dict()),p)
def restore_model(p):
 x=torch.load(p,map_location='cpu',weights_only=True)
 m=make_boundary_model(x['config']['seed']);m.load_state_dict(x['state_dict']);return m
def state_difference(a,b):
 return max(float(torch.max(torch.abs(a[k]-b[k]))) for k in a)

def train_fit(v,validation,objective,seed,path,epochs,k,inner=False,old_path=None):
 if (path/'FIT.json').exists():
  meta=read(path/'FIT.json')
  assert meta['objective']==objective and meta['seed']==seed and meta['epochs']==epochs
  return restore_model(path/'weights.pt'),BoundaryNormalizer.from_file(path/'normalization.npz'),read(path/'curve.json')
 if path.exists():raise RuntimeError(f'Incomplete fit preserved: {path}')
 path.mkdir(parents=True);start=time.monotonic()
 norm=BoundaryNormalizer(v,k);norm.save(path/'normalization.npz')
 t=norm.tensors(v,'boundary_context',k)
 vt=norm.tensors(validation,'boundary_context',k) if validation is not None else None
 support_np=ref_support(v,k);support=tensor(support_np)
 dump(path/'SUPERVISION.json',dict(k=k,support_float64_sha256=hashlib.sha256(support_np.tobytes()).hexdigest(),identity='Same empirical donor-bootstrap support as B; labels are unchanged'))
 m=make_boundary_model(seed);assert np.array_equal(predict(m,t),v['base'])
 opt=torch.optim.Adam(m.parameters(),lr=.01);curve=[];grad=0.
 for epoch in range(epochs+1):
  if epoch in CHECKPOINTS or epoch==epochs:
   rank,residual,ridge=parts(m,t,support,k,objective)
   rec=dict(epoch=epoch,loss=float((rank+residual+ridge).detach()),ranking_loss=float(rank.detach()),residual_penalty=float(residual.detach()),head_ridge=float(ridge.detach()),gradient_norm=grad,train=metric(m,t,v,k))
   if validation is not None:rec['validation']=metric(m,vt,validation,k)
   if objective=='rank_scale':
    with torch.no_grad():
     score=m(t['features'],t['raw'],t['anchor']);n=t['mask'].sum(1,keepdim=True)
     mean=torch.where(t['mask'],score,0.).sum(1,keepdim=True)/n
     var=torch.where(t['mask'],(score-mean).square(),0.).sum(1,keepdim=True)/n
     rec['scale_floor_rows']=int((var <= (1e-6*t['scale']).square()).sum())
   curve.append(rec)
   if inner and epoch in (50,100,200):save_model(m,checkpoint_path(path,epoch))
  if epoch==epochs:break
  opt.zero_grad();loss=sum(parts(m,t,support,k,objective))
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
  loss.backward()
  for p in m.parameters():
   if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient')
  grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),5.));opt.step()
 save_model(m,path/'weights.pt');dump(path/'curve.json',curve)
 meta=dict(objective=objective,seed=seed,epochs=epochs,k=k,source_specific=True,training_target_ids=v['target_ids'].tolist(),validation_target_ids=None if validation is None else validation['target_ids'].tolist(),duration_seconds=time.monotonic()-start,parameters=sum(p.numel() for p in m.parameters()),inner_checkpoint_epochs=[0,50,100,200,300] if inner else [epochs])
 if old_path is not None:
  assert objective=='frozen_scale' and epochs==300
  old=torch.load(old_path/'weights.pt',map_location='cpu',weights_only=True)
  error=state_difference(m.state_dict(),old['state_dict']);assert error==0.,('B inner replay changed',str(path),error)
  assert sha(path/'normalization.npz')==sha(old_path/'normalization.npz')
  oldcurve=read(old_path/'curve.json')
  curve_error=max(abs(curve[i][which]['precision']-oldcurve[i][which]['precision']) for i in range(len(curve)) for which in ('train','validation'))
  assert curve_error==0.
  meta['old_B_replay']=dict(state_max_abs_difference=error,precision_curve_max_abs_difference=curve_error,normalization_bytes_identical=True,old_fit_path=str(old_path.relative_to(BASE)),old_weights_sha256=sha(old_path/'weights.pt'))
 dump(path/'FIT.json',meta)
 return m,norm,curve

def checked_plan():
 plan=read(ROOT/'PLAN_LOCK.json')
 assert sha(BASE/'PLAN_LOCK.json')==plan['base_plan_sha256']
 for name,h in plan['code_hashes'].items():assert sha(ROOT/name)==h,('Changed update code',name)
 for name,h in plan['base_code_hashes'].items():assert sha(BASE/name)==h,('Changed base code',name)
 for name,h in plan['input_hashes'].items():assert sha(BASE/name)==h,('Changed input',name)
 for s,h in plan['base_prediction_lock_sha256'].items():assert sha(BASE/'predictions'/f'{s}_PREDICTION_LOCK.json')==h
 assert sha(ROOT/'NUMERICAL_CHECKS.json')==plan['checks_sha256']
 return plan

def read_calibration(plan,source,half):
 task=f'{source}_held{half}';p=BASE/'views'/source/f'held{half}';meta=plan['tasks'][task]
 for name,key in [('calibration.npz','calibration_sha256'),('held_features.npz','held_features_sha256'),('INNER_SPLIT.json','inner_split_sha256')]:assert sha(p/name)==meta[key]
 return load(p/'calibration.npz'),read(p/'INNER_SPLIT.json')

def inner_path(task,objective,seed,ih):return ROOT/'fits'/f'{task}_{objective}_seed{seed}_inner{ih}'

def run_inner(source):
 plan=checked_plan();dest=ROOT/'selections'/f'{source}.json'
 if dest.exists():raise RuntimeError('Existing selection preserved')
 tasks={}
 for half in (0,1):
  task=f'{source}_held{half}';cal,groups=read_calibration(plan,source,half)
  g=np.array([groups[str(x)] for x in cal['target_ids']]);chosen={}
  for objective in OBJECTIVES:
   stored={};validations={};curves={};fits={}
   for ih in (0,1):
    tr=subset(cal,np.flatnonzero(g!=ih));va=subset(cal,np.flatnonzero(g==ih));validations[ih]=va
    for seed in SEEDS:
     p=inner_path(task,objective,seed,ih)
     old=BASE/'fits'/f'{task}_boundary_context_seed{seed}_inner{ih}' if objective=='frozen_scale' else None
     _,norm,curve=train_fit(tr,va,objective,seed,p,300,5,inner=True,old_path=old)
     curves[seed,ih]=curve;vt=norm.tensors(va,'boundary_context',5)
     for epoch in CHECKPOINTS:
      m=make_boundary_model(seed) if epoch==0 else restore_model(checkpoint_path(p,epoch))
      stored[seed,ih,epoch]=predict(m,vt)
     fits[f'{seed}_inner{ih}']=dict(path=str(p.relative_to(ROOT)),fit_sha256=sha(p/'FIT.json'),normalization_sha256=sha(p/'normalization.npz'),weights_sha256=sha(p/'weights.pt'),curve_sha256=sha(p/'curve.json'),checkpoint_sha256={str(e):sha(checkpoint_path(p,e)) for e in (50,100,200,300)})
   per_seed={}
   for seed in SEEDS:
    choices=[]
    for ci,epoch in enumerate(CHECKPOINTS):
     gains=[curves[seed,ih][ci]['validation']['precision']-curves[seed,ih][0]['validation']['precision'] for ih in (0,1)]
     choices.append(dict(epoch=epoch,gains=gains,mean_gain=float(np.mean(gains)),eligible=all(x>=0 for x in gains)))
    best=max((x for x in choices if x['eligible']),key=lambda x:(x['mean_gain'],-x['epoch']))
    per_seed[str(seed)]=dict(selected_epoch=best['epoch'],selection=choices)
   choices=[]
   for epoch in CHECKPOINTS:
    gains=[];scores=[]
    for ih in (0,1):
     va=validations[ih];ensemble=np.mean([stored[seed,ih,epoch] for seed in SEEDS],axis=0)
     p=float(precision(ensemble,va['reference'],va['ownmask'],5).mean())
     p0=float(precision(va['base'],va['reference'],va['ownmask'],5).mean())
     gains.append(p-p0);scores.append(p)
    choices.append(dict(epoch=epoch,gains=gains,mean_gain=float(np.mean(gains)),precision=scores,eligible=all(x>=0 for x in gains)))
   best=max((x for x in choices if x['eligible']),key=lambda x:(x['mean_gain'],-x['epoch']))
   chosen[objective]=dict(per_seed=per_seed,ensemble=dict(selected_epoch=best['epoch'],selection=choices),inner_fits=fits)
   if objective=='frozen_scale':
    old=read(BASE/'selections'/f'{source}.json')['tasks'][task]
    for seed in SEEDS:
     assert per_seed[str(seed)]['selected_epoch']==old[f'boundary_context_seed{seed}']['selected_epoch']
   print(source,half,objective,'per_seed',[per_seed[str(s)]['selected_epoch'] for s in SEEDS],'ensemble',best['epoch'],flush=True)
  tasks[task]=chosen
 dump(dest,dict(source=source,created_utc=now(),plan_sha256=sha(ROOT/'PLAN_LOCK.json'),tasks=tasks))

def all_selection_hashes():
 out={}
 for source in SOURCES:
  p=ROOT/'selections'/f'{source}.json';d=read(p)
  assert d['plan_sha256']==sha(ROOT/'PLAN_LOCK.json');out[source]=sha(p)
 return out

def run_outer(source):
 plan=checked_plan();selection_hashes=all_selection_hashes();dest=ROOT/'predictions'/f'{source}_PREDICTION_LOCK.json'
 if dest.exists():raise RuntimeError('Existing outer predictions preserved')
 selected=read(ROOT/'selections'/f'{source}.json')['tasks'];oldlock=read(BASE/'predictions'/f'{source}_PREDICTION_LOCK.json');tasks={}
 for half in (0,1):
  task=f'{source}_held{half}';cal,_=read_calibration(plan,source,half)
  held=load(BASE/'views'/source/f'held{half}'/'held_features.npz');assert not any(k.startswith('reference') for k in held)
  op=BASE/oldlock['tasks'][task]['path'];assert sha(op)==oldlock['tasks'][task]['sha256']
  old=load(op);ai=old['arm_names'].tolist().index('boundary_context')
  for key in ('base','raw','euclidean','ownmask','query_ids','query_block','target_ids'):assert np.array_equal(old[key],held[key])
  scores=np.empty((len(ARMS),len(SEEDS))+held['base'].shape);scores[0]=old['scores'][ai];fits={}
  for si,seed in enumerate(SEEDS):
   original=oldlock['tasks'][task]['fits'][f'boundary_context_seed{seed}']
   fits[f'old_B_seed{seed}']=dict(fit_root='base',fit_path=original['outer_fit'],selected_epoch=original['selected_epoch'],weights_sha256=original['weights_sha256'],normalization_sha256=original['normalization_sha256'],historical_import=True)
  cache={}
  for ai,(arm,objective,selection) in enumerate((('B_joint_selection','frozen_scale','ensemble'),('B_rank_scale','rank_scale','per_seed'),('B_rank_scale_joint','rank_scale','ensemble')),1):
   for si,seed in enumerate(SEEDS):
    sel=selected[task][objective]
    epoch=sel['per_seed'][str(seed)]['selected_epoch'] if selection=='per_seed' else sel['ensemble']['selected_epoch']
    cache_key=(objective,seed,epoch)
    if cache_key not in cache:
     oldfit=oldlock['tasks'][task]['fits'][f'boundary_context_seed{seed}']
     if objective=='frozen_scale' and epoch==oldfit['selected_epoch']:
      fp=BASE/oldfit['outer_fit'];assert sha(fp/'weights.pt')==oldfit['weights_sha256'];assert sha(fp/'normalization.npz')==oldfit['normalization_sha256']
      m=restore_model(fp/'weights.pt');norm=BoundaryNormalizer.from_file(fp/'normalization.npz');fitroot='base';rel=oldfit['outer_fit'];imported=True
     else:
      fp=ROOT/'fits'/f'{task}_{objective}_seed{seed}_outer_e{epoch}'
      m,norm,_=train_fit(cal,None,objective,seed,fp,epoch,10)
      fitroot='new';rel=str(fp.relative_to(ROOT));imported=False
     pred=predict(m,norm.tensors(held,'boundary_context',10))
     cache[cache_key]=(pred,dict(fit_root=fitroot,fit_path=rel,selected_epoch=epoch,objective=objective,weights_sha256=sha(fp/'weights.pt'),normalization_sha256=sha(fp/'normalization.npz'),historical_import=imported))
    scores[ai,si],meta=cache[cache_key];fits[f'{arm}_seed{seed}']={**meta,'selection_rule':selection}
    print(source,half,arm,seed,epoch,flush=True)
  pp=ROOT/'predictions'/f'{task}.npz';pp.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(pp,scores=scores,arm_names=np.array(ARMS),seeds=np.array(SEEDS),**{key:held[key] for key in ('base','raw','euclidean','ownmask','query_ids','query_block','target_ids')})
  tasks[task]=dict(path=str(pp.relative_to(ROOT)),sha256=sha(pp),fits=fits)
 dump(dest,dict(plan_sha256=sha(ROOT/'PLAN_LOCK.json'),selection_sha256=selection_hashes,prediction_time_utc=now(),tasks=tasks))

def freeze():
 checked_plan();selected=all_selection_hashes();tasks={};sources={}
 for source in SOURCES:
  p=ROOT/'predictions'/f'{source}_PREDICTION_LOCK.json';d=read(p)
  assert d['plan_sha256']==sha(ROOT/'PLAN_LOCK.json') and d['selection_sha256']==selected
  for task,entry in d['tasks'].items():assert sha(ROOT/entry['path'])==entry['sha256']
  tasks.update(d['tasks']);sources[source]=sha(p)
 assert set(tasks)=={f'{s}_held{h}' for s in SOURCES for h in (0,1)}
 out=ROOT/'predictions'/'ALL_PREDICTIONS_LOCK.json'
 if out.exists():raise RuntimeError('Existing all-prediction lock preserved')
 dump(out,dict(plan_sha256=sha(ROOT/'PLAN_LOCK.json'),source_lock_sha256=sources,tasks=tasks,created_utc=now(),all_predictions_frozen_before_this_update_scoring=True,identity='Exploratory historical references; not a fresh holdout experiment'))
 print('ALL_SIX_FROZEN',flush=True)

def checks():
 rng=np.random.default_rng(621112);q,n=5,8;mask=np.ones((q,n),bool);mask[np.arange(q),np.arange(q)]=False
 score=tensor(rng.normal(size=(q,n)));score.requires_grad_()
 t=dict(mask=torch.as_tensor(mask),scale=tensor(np.ones((q,1))))
 supports=tensor(rng.uniform(size=(2,q,n)));loss=list_loss(ranking_scores(score,t,'rank_scale'),t['mask'],supports[0],supports[1],2,.5)
 result={};values=[]
 for factor in (.5,1.,2.):
  values.append(float(list_loss(ranking_scores(score*factor,t,'rank_scale'),t['mask'],supports[0],supports[1],2,.5).detach()))
 result['positive_scaling_loss_max_difference']=max(values)-min(values);assert result['positive_scaling_loss_max_difference']<1e-12
 g=torch.autograd.grad(loss,score)[0];result['radial_gradient_dot_max']=float(abs((g*score).sum(1)).max().detach());assert result['radial_gradient_dot_max']<1e-12
 changed=score.detach().clone();changed[~t['mask']]=1e8
 value=float(list_loss(ranking_scores(changed,t,'rank_scale'),t['mask'],supports[0],supports[1],2,.5));assert abs(value-values[1])<1e-12
 result['excluded_score_gradient_max']=float(abs(g[~t['mask']]).max());assert result['excluded_score_gradient_max']==0.
 perm=rng.permutation(n);tp=dict(mask=t['mask'][:,perm],scale=t['scale'])
 val=float(list_loss(ranking_scores(score[:,perm],tp,'rank_scale'),tp['mask'],supports[0][:,perm],supports[1][:,perm],2,.5).detach());assert abs(val-values[1])<1e-12
 result['candidate_permutation_loss_difference']=abs(val-values[1])
 zeros=torch.zeros_like(score,requires_grad=True)
 zloss=list_loss(ranking_scores(zeros,t,'rank_scale'),t['mask'],supports[0],supports[1],2,.5);zg=torch.autograd.grad(zloss,zeros)[0];assert torch.isfinite(zg).all();result['zero_variance_gradient_finite']=True
 for s in SOURCES:
  cal=load(BASE/'views'/s/'held0'/'calibration.npz');ix=np.arange(min(13,len(cal['target_ids'])));tr=subset(cal,ix)
  norm=BoundaryNormalizer(tr,5);tt=norm.tensors(tr,'boundary_context',5);model=make_boundary_model(SEEDS[0]);assert np.array_equal(predict(model,tt),tr['base'])
  labels=tensor(ref_support(tr,5))
  for objective in OBJECTIVES:
   model.zero_grad();ll=sum(parts(model,tt,labels,5,objective));ll.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
  altered={k:np.copy(v) if isinstance(v,np.ndarray) else v for k,v in tr.items()};altered['reference'][:]=100.;altered['reference_replicates'][:]=-100.
  other=BoundaryNormalizer(altered,5)
  assert np.array_equal(norm.tensors(tr,'boundary_context',5)['features'].numpy(),other.tensors(tr,'boundary_context',5)['features'].numpy())
 result.update(all_passed=True,training_performed=False,zero_initialization_exact_all_sources=True,reference_independent_normalization_all_sources=True,finite_initial_model_gradients_all_objectives=True,formula='centered prediction / eligible prediction population SD; differentiates through mean and SD; floor=1e-6*fixed training baseline SD')
 dump(ROOT/'NUMERICAL_CHECKS.json',result);print(json.dumps(result,indent=2))

def prepare():
 if (ROOT/'PLAN_LOCK.json').exists():raise RuntimeError('Existing plan preserved')
 assert read(ROOT/'NUMERICAL_CHECKS.json')['all_passed']
 old=read(BASE/'PLAN_LOCK.json')
 for name,h in old['code_hashes'].items():assert sha(BASE/name)==h
 names=('train_optimization.py','PROTOCOL.md','EXPERIMENT_SCOPE.md','score_optimization.py','check_scoring.py','SCORING_CHECKS.json')
 for name in names:assert (ROOT/name).is_file()
 basecodes=('base_training.py','boundary_components.py','gate_model.py','ranking_ops.py','common.py')
 plan=dict(created_utc=now(),arms=list(ARMS),seeds=list(SEEDS),objectives=list(OBJECTIVES),checkpoints=list(CHECKPOINTS),primary_arm='B_rank_scale_joint',primary_contrast='B_rank_scale_joint minus old_B',source_specific=True,tasks=old['tasks'],code_hashes={n:sha(ROOT/n) for n in names},base_code_hashes={n:sha(BASE/n) for n in basecodes},input_hashes=old['input_hashes'],base_plan_sha256=sha(BASE/'PLAN_LOCK.json'),base_prediction_lock_sha256={s:sha(BASE/'predictions'/f'{s}_PREDICTION_LOCK.json') for s in SOURCES},checks_sha256=sha(ROOT/'NUMERICAL_CHECKS.json'),identity='One exploratory 2x2 update on historically accessed references; no new-confirmation claim',no_outer_reference_read_by_training=True,torch_version=str(torch.__version__))
 dump(ROOT/'PLAN_LOCK.json',plan);print('PREPARED',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);default=ROOT/'baseline_B'
 if not default.exists():default=ROOT.parent/'pertrank_boundary_learning_20261115'
 p.add_argument('--base-root',type=Path,default=default);g=p.add_mutually_exclusive_group(required=True)
 g.add_argument('--checks',action='store_true');g.add_argument('--prepare',action='store_true');g.add_argument('--inner',choices=SOURCES);g.add_argument('--outer',choices=SOURCES);g.add_argument('--freeze',action='store_true')
 a=p.parse_args();configure(a.base_root)
 if a.checks:checks()
 elif a.prepare:prepare()
 elif a.inner:run_inner(a.inner)
 elif a.outer:run_outer(a.outer)
 else:freeze()
