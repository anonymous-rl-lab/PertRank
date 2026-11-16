#!/usr/bin/env python3
"""Phase 2: fixed 5 x 5 epoch-pair selection, reusing frozen inner fits.

This post hoc development extension preserves phase 1 and changes only the
restriction that the two deployed seeds must have the same epoch. No new
inner fitting, objective, feature, source or hyperparameter is introduced.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
O=importlib.import_module('train_optimization')
SOURCES=O.SOURCES;SEEDS=O.SEEDS;EPOCHS=O.CHECKPOINTS
ARMS=('B_free_pair','B_rank_scale_free_pair')
OBJECTIVES=O.OBJECTIVES
BASE=None
read=O.read;dump=O.dump;sha=O.sha;load=O.load
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def configure(base):
 global BASE
 BASE=Path(base).resolve();O.configure(BASE)

def choose(choices):
 return max((x for x in choices if x['eligible']),key=lambda x:(x['mean_gain'],-sum(x['epochs']),-max(x['epochs']),-x['epochs'][0],-x['epochs'][1]))

def checks():
 choices=[]
 for a in EPOCHS:
  for b in EPOCHS:
   choices.append(dict(epochs=[a,b],mean_gain=0.,eligible=True))
 assert choose(choices)['epochs']==[0,0]
 assert len(choices)==25 and sum(x['epochs'][0]==x['epochs'][1] for x in choices)==5
 tied=[dict(epochs=[50,100],mean_gain=1.,eligible=True),dict(epochs=[100,50],mean_gain=1.,eligible=True),dict(epochs=[0,200],mean_gain=1.,eligible=True)]
 assert choose(tied)['epochs']==[50,100]
 invalid=[dict(epochs=[0,0],mean_gain=0.,eligible=True),dict(epochs=[50,50],mean_gain=1.,eligible=False)]
 assert choose(invalid)['epochs']==[0,0]
 dump(ROOT/'SELECTION_CHECKS.json',dict(passed=True,number_of_pairs=25,number_of_diagonal_pairs=5,zero_pair_fallback=True,tie_order_sum_max_then_lexicographic=True,noneligible_pair_excluded=True,training_performed=False))

def prepare():
 if (ROOT/'PLAN_LOCK.json').exists():raise RuntimeError('Existing phase2 plan preserved')
 O.checked_plan();parent=read(PARENT/'PLAN_LOCK.json')
 assert read(ROOT/'SELECTION_CHECKS.json')['passed']
 parent_all=PARENT/'predictions/ALL_PREDICTIONS_LOCK.json';al=read(parent_all)
 assert al['plan_sha256']==sha(PARENT/'PLAN_LOCK.json')
 code=('update_pair_selection.py','PROTOCOL.md','score_pair_update.py','check_scoring.py','SCORING_CHECKS.json')
 for name in code:assert (ROOT/name).is_file(),name
 plan=dict(created_utc=now(),identity='Post hoc development extension after phase1 results; phase1 primary and results retained',arms=list(ARMS),objectives=list(OBJECTIVES),seeds=list(SEEDS),epochs=list(EPOCHS),primary_arm='B_free_pair',primary_contrast='B_free_pair minus old_B',pair_count=25,no_inner_training=True,no_new_hyperparameter_values=True,selection_candidates_before=5,selection_candidates_after=25,source_specific=True,tasks=parent['tasks'],code_hashes={n:sha(ROOT/n) for n in code},parent_code_hashes={'train_optimization.py':sha(PARENT/'train_optimization.py')},parent_plan_sha256=sha(PARENT/'PLAN_LOCK.json'),parent_prediction_lock_sha256={s:sha(PARENT/'predictions'/f'{s}_PREDICTION_LOCK.json') for s in SOURCES},parent_all_predictions_sha256=sha(parent_all),parent_selection_sha256={s:sha(PARENT/'selections'/f'{s}.json') for s in SOURCES},base_plan_sha256=parent['base_plan_sha256'],base_prediction_lock_sha256=parent['base_prediction_lock_sha256'],base_code_hashes=parent['base_code_hashes'],input_hashes=parent['input_hashes'],checks_sha256=sha(ROOT/'SELECTION_CHECKS.json'))
 dump(ROOT/'PLAN_LOCK.json',plan);print('PHASE2_PREPARED',flush=True)

def check_plan():
 plan=read(ROOT/'PLAN_LOCK.json')
 for n,h in plan['code_hashes'].items():assert sha(ROOT/n)==h,('phase2 code',n)
 for n,h in plan['parent_code_hashes'].items():assert sha(PARENT/n)==h
 assert sha(PARENT/'PLAN_LOCK.json')==plan['parent_plan_sha256']
 assert sha(PARENT/'predictions/ALL_PREDICTIONS_LOCK.json')==plan['parent_all_predictions_sha256']
 assert sha(BASE/'PLAN_LOCK.json')==plan['base_plan_sha256']
 for s,h in plan['parent_prediction_lock_sha256'].items():assert sha(PARENT/'predictions'/f'{s}_PREDICTION_LOCK.json')==h
 for s,h in plan['parent_selection_sha256'].items():assert sha(PARENT/'selections'/f'{s}.json')==h
 for n,h in plan['base_code_hashes'].items():assert sha(BASE/n)==h
 for n,h in plan['input_hashes'].items():assert sha(BASE/n)==h
 assert sha(ROOT/'SELECTION_CHECKS.json')==plan['checks_sha256']
 return plan

def select(source):
 plan=check_plan();dest=ROOT/'selections'/f'{source}.json'
 if dest.exists():raise RuntimeError('Existing phase2 selection preserved')
 parent_select=read(PARENT/'selections'/f'{source}.json')['tasks'];tasks={}
 for half in (0,1):
  task=f'{source}_held{half}';cal,groups=O.read_calibration(plan,source,half)
  g=np.array([groups[str(t)] for t in cal['target_ids']]);task_result={}
  for objective in OBJECTIVES:
   pred={};valid={};checkpoint_hashes={}
   old_selection=parent_select[task][objective]
   for ih in (0,1):
    va=O.subset(cal,np.flatnonzero(g==ih));valid[ih]=va
    for seed in SEEDS:
     f=old_selection['inner_fits'][f'{seed}_inner{ih}'];path=PARENT/f['path']
     assert sha(path/'normalization.npz')==f['normalization_sha256'] and sha(path/'FIT.json')==f['fit_sha256']
     norm=O.BoundaryNormalizer.from_file(path/'normalization.npz');vt=norm.tensors(va,'boundary_context',5)
     for epoch in EPOCHS:
      if epoch==0:m=O.make_boundary_model(seed)
      else:
       p=O.checkpoint_path(path,epoch);assert sha(p)==f['checkpoint_sha256'][str(epoch)];m=O.restore_model(p)
       checkpoint_hashes[str(p.relative_to(PARENT))]=sha(p)
      pred[seed,ih,epoch]=O.predict(m,vt)
   choices=[]
   for e1 in EPOCHS:
    for e2 in EPOCHS:
     gains=[];values=[]
     for ih in (0,1):
      va=valid[ih];p=.5*(pred[SEEDS[0],ih,e1]+pred[SEEDS[1],ih,e2])
      value=float(O.precision(p,va['reference'],va['ownmask'],5).mean())
      p0=float(O.precision(va['base'],va['reference'],va['ownmask'],5).mean())
      values.append(value);gains.append(value-p0)
     choices.append(dict(epochs=[e1,e2],precision=values,gains=gains,mean_gain=float(np.mean(gains)),eligible=all(x>=0 for x in gains)))
   chosen=choose(choices);diagonal=[x for x in choices if x['epochs'][0]==x['epochs'][1]];diag_best=choose(diagonal)
   assert len(choices)==25 and len(diagonal)==5 and chosen['mean_gain']>=diag_best['mean_gain']
   assert diag_best['epochs'][0]==old_selection['ensemble']['selected_epoch']
   for d,c in zip(diagonal,old_selection['ensemble']['selection']):
    assert d['epochs'][0]==c['epoch'] and d['gains']==c['gains'] and d['precision']==c['precision']
   task_result[objective]=dict(selected_epochs=chosen['epochs'],selected=chosen,all_25_choices=choices,diagonal_best=diag_best,diagonal_reproduces_phase1_exactly=True,checkpoint_hashes=checkpoint_hashes)
   print(source,half,objective,'freepair',chosen['epochs'],'diag',diag_best['epochs'],'inner_gain',chosen['mean_gain'],flush=True)
  tasks[task]=task_result
 dump(dest,dict(source=source,created_utc=now(),plan_sha256=sha(ROOT/'PLAN_LOCK.json'),no_new_inner_training=True,tasks=tasks))

def selections():
 hashes={}
 for source in SOURCES:
  p=ROOT/'selections'/f'{source}.json';x=read(p);assert x['plan_sha256']==sha(ROOT/'PLAN_LOCK.json');hashes[source]=sha(p)
 return hashes

def find_existing(task,objective,seed,epoch,parent_lock):
 # Every prior arm's frozen record remains eligible for exact reuse. This
 # compares only objective, initialization seed and selected training epoch.
 for key,f in parent_lock['tasks'][task]['fits'].items():
  if not key.endswith(f'_seed{seed}'):continue
  obj=f.get('objective','frozen_scale')
  if obj!=objective or f['selected_epoch']!=epoch:continue
  root=BASE if f['fit_root']=='base' else PARENT;p=root/f['fit_path']
  assert sha(p/'weights.pt')==f['weights_sha256'] and sha(p/'normalization.npz')==f['normalization_sha256']
  return p,('base' if root==BASE else 'parent'),f['fit_path']
 return None

def outer(source):
 plan=check_plan();sh=selections();dest=ROOT/'predictions'/f'{source}_PREDICTION_LOCK.json'
 if dest.exists():raise RuntimeError('Existing phase2 predictions preserved')
 selected=read(ROOT/'selections'/f'{source}.json')['tasks'];parent_lock=read(PARENT/'predictions'/f'{source}_PREDICTION_LOCK.json');tasks={}
 for half in (0,1):
  task=f'{source}_held{half}';cal,_=O.read_calibration(plan,source,half)
  held=load(BASE/'views'/source/f'held{half}'/'held_features.npz');assert not any(k.startswith('reference') for k in held)
  scores=np.empty((2,2)+held['base'].shape);fits={}
  for ai,(arm,objective) in enumerate(zip(ARMS,OBJECTIVES)):
   for si,seed in enumerate(SEEDS):
    epoch=selected[task][objective]['selected_epochs'][si]
    prior=find_existing(task,objective,seed,epoch,parent_lock)
    if prior is not None:
     path,fitroot,relative=prior;m=O.restore_model(path/'weights.pt');norm=O.BoundaryNormalizer.from_file(path/'normalization.npz');reused=True
    else:
     path=ROOT/'fits'/f'{task}_{objective}_seed{seed}_outer_e{epoch}'
     m,norm,_=O.train_fit(cal,None,objective,seed,path,epoch,10);fitroot='new';relative=str(path.relative_to(ROOT));reused=False
    scores[ai,si]=O.predict(m,norm.tensors(held,'boundary_context',10))
    fits[f'{arm}_seed{seed}']=dict(fit_root=fitroot,fit_path=relative,selected_epoch=epoch,objective=objective,seed=seed,weights_sha256=sha(path/'weights.pt'),normalization_sha256=sha(path/'normalization.npz'),historical_import=reused,selection_rule='fixed_25_epoch_pairs')
    print(source,half,arm,seed,epoch,'reused' if reused else 'new_fit',flush=True)
  p=ROOT/'predictions'/f'{task}.npz';p.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(p,scores=scores,arm_names=np.array(ARMS),seeds=np.array(SEEDS),**{k:held[k] for k in ('base','raw','euclidean','ownmask','query_ids','query_block','target_ids')})
  tasks[task]=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),fits=fits)
 dump(dest,dict(plan_sha256=sha(ROOT/'PLAN_LOCK.json'),selection_sha256=sh,prediction_time_utc=now(),tasks=tasks))

def freeze():
 plan=check_plan();sh=selections();tasks={};sources={}
 for source in SOURCES:
  p=ROOT/'predictions'/f'{source}_PREDICTION_LOCK.json';d=read(p)
  assert d['plan_sha256']==sha(ROOT/'PLAN_LOCK.json') and d['selection_sha256']==sh
  for task,e in d['tasks'].items():assert sha(ROOT/e['path'])==e['sha256']
  tasks.update(d['tasks']);sources[source]=sha(p)
 assert len(tasks)==6
 p=ROOT/'predictions/ALL_PREDICTIONS_LOCK.json'
 if p.exists():raise RuntimeError('Existing phase2 all-prediction lock preserved')
 dump(p,dict(plan_sha256=sha(ROOT/'PLAN_LOCK.json'),source_lock_sha256=sources,tasks=tasks,created_utc=now(),all_predictions_frozen_before_phase2_scoring=True,identity=plan['identity']))
 print('PHASE2_ALL_SIX_FROZEN',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);default=PARENT/'baseline_B'
 if not default.exists():default=PARENT.parent/'pertrank_boundary_learning_20261115'
 parser.add_argument('--base-root',type=Path,default=default);g=parser.add_mutually_exclusive_group(required=True)
 g.add_argument('--checks',action='store_true');g.add_argument('--prepare',action='store_true');g.add_argument('--select',choices=SOURCES);g.add_argument('--outer',choices=SOURCES);g.add_argument('--freeze',action='store_true')
 args=parser.parse_args();configure(args.base_root)
 if args.checks:checks()
 elif args.prepare:prepare()
 elif args.select:select(args.select)
 elif args.outer:outer(args.outer)
 else:freeze()
