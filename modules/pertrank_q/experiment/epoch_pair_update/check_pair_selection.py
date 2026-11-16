"""Independent audit of epoch-pair enumeration and target/seed alignment.
Reads calibration labels only; no outer evaluation references are consumed.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os
import numpy as np
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
spec=importlib.util.spec_from_file_location('review_pair',ROOT/'update_pair_selection.py');U=importlib.util.module_from_spec(spec);spec.loader.exec_module(U)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def independent_choice(values):
 return min((v for v in values if v['eligible']),key=lambda v:(-v['mean_gain'],sum(v['epochs']),max(v['epochs']),*v['epochs']))
def main(base,output,require_selections=False):
 U.configure(base);O=U.O;assert len(set(U.SEEDS))==2 and tuple(U.EPOCHS)==(0,50,100,200,300)
 rng=np.random.default_rng(41311);cases=0
 for _ in range(200):
  choices=[dict(epochs=[a,b],mean_gain=float(rng.integers(-2,4)),eligible=bool(rng.integers(2))) for a in U.EPOCHS for b in U.EPOCHS]
  choices[0].update(mean_gain=0.,eligible=True)
  assert U.choose(choices)==independent_choice(choices);cases+=1
 out={'passed':False,'training_performed':False,'held_evaluation_reference_read':False,'selection_count':25,'diagonal_count':5,'compared_to_common_epoch_selection_count':5,'larger_selection_search_requires_exploratory_interpretation':True,'synthetic_tie_and_eligibility_cases':cases,'tasks':{},'selection_tables_checked':0,'source_code_sha256':sha(ROOT/'update_pair_selection.py')}
 for source in U.SOURCES:
  prior=read(PARENT/'selections'/f'{source}.json')['tasks'];sp=ROOT/'selections'/f'{source}.json'
  if require_selections:assert sp.is_file(),sp
  selected=read(sp)['tasks'] if sp.exists() else None
  for half in (0,1):
   task=f'{source}_held{half}';vp=Path(base)/'views'/source/f'held{half}';cal=O.load(vp/'calibration.npz');groups=read(vp/'INNER_SPLIT.json');g=np.array([groups[str(t)] for t in cal['target_ids']])
   for objective in U.OBJECTIVES:
    predictions={};valid={};checked=[]
    for ih in (0,1):
     va=O.subset(cal,np.flatnonzero(g==ih));tr=O.subset(cal,np.flatnonzero(g!=ih));valid[ih]=va
     for seed in U.SEEDS:
      f=prior[task][objective]['inner_fits'][f'{seed}_inner{ih}'];d=PARENT/f['path'];meta=read(d/'FIT.json')
      assert meta['seed']==seed and meta['objective']==objective
      assert list(meta['training_target_ids'])==tr['target_ids'].tolist() and list(meta['validation_target_ids'])==va['target_ids'].tolist()
      assert not set(meta['training_target_ids'])&set(meta['validation_target_ids'])
      assert sha(d/'normalization.npz')==f['normalization_sha256'] and sha(d/'FIT.json')==f['fit_sha256']
      norm=O.BoundaryNormalizer.from_file(d/'normalization.npz');t=norm.tensors(va,'boundary_context',5)
      for epoch in U.EPOCHS:
       if epoch==0:model=O.make_boundary_model(seed)
       else:
        p=O.checkpoint_path(d,epoch);assert sha(p)==f['checkpoint_sha256'][str(epoch)]
        saved=O.torch.load(p,map_location='cpu',weights_only=True);assert saved['config']['seed']==seed
        model=O.restore_model(p)
       predictions[seed,ih,epoch]=O.predict(model,t)
      checked.append({'seed':seed,'inner_validation_group':ih,'training_targets':len(tr['target_ids']),'validation_targets':len(va['target_ids'])})
    choices=[]
    for a in U.EPOCHS:
     for b in U.EPOCHS:
      gains=[];values=[]
      for ih in (0,1):
       va=valid[ih];pred=np.stack([predictions[U.SEEDS[0],ih,a],predictions[U.SEEDS[1],ih,b]]).mean(axis=0)
       p=float(O.precision(pred,va['reference'],va['ownmask'],5).mean());p0=float(O.precision(va['base'],va['reference'],va['ownmask'],5).mean())
       values.append(p);gains.append(p-p0)
      choices.append({'epochs':[a,b],'precision':values,'gains':gains,'mean_gain':float(np.mean(gains)),'eligible':min(gains)>=0.})
    assert len(choices)==25 and len(set(tuple(v['epochs']) for v in choices))==25
    diag=[v for v in choices if v['epochs'][0]==v['epochs'][1]]
    for a,b in zip(diag,prior[task][objective]['ensemble']['selection']):assert a['gains']==b['gains'] and a['precision']==b['precision']
    chosen=independent_choice(choices)
    if selected is not None:
     value=selected[task][objective]
     assert value['all_25_choices']==choices
     assert value['selected']==chosen and value['selected_epochs']==chosen['epochs']
     assert value['diagonal_best']==independent_choice(diag)
     out['selection_tables_checked']+=1
    out['tasks'][f'{task}_{objective}']={'fit_role_checks':checked,'all25_recomputed':True,'all5_diagonal_choices_match_phase1':True,'chosen_epochs':chosen['epochs'],'saved_table_checked':selected is not None}
 out['passed']=True;out['script_sha256']=sha(__file__)
 Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'passed':True,'selection_tables_checked':out['selection_tables_checked'],'target_seed_fit_roles_checked':4*len(out['tasks']),'all_25_choices_recomputed':25*len(out['tasks']),'output':str(output)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();default=PARENT/'baseline_B'
 if not default.exists():default=PARENT.parent/'pertrank_boundary_learning_20261115'
 p.add_argument('--base-root',type=Path,default=default);p.add_argument('--output',type=Path,default=ROOT/'INDEPENDENT_SELECTION_REVIEW.json');p.add_argument('--require-selections',action='store_true');a=p.parse_args();main(a.base_root,a.output,a.require_selections)
