"""Read-only B checkpoint and selection audit; no fitting or outer scoring."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'pertrank_boundary_learning_20261115'
sys.path[:0]=[str(OLD),str(ROOT.parent/'pertrank_neural_rank_20261115/.runtime')]
from base_training import torch,predict,tensor
from boundary_components import BoundaryNormalizer
from gate_model import ReliabilityGate,list_loss
from common import load,subset,ref_support,precision,dump,sha

def main():
 result={'identity':'Exploratory read-only B selection audit; no new fitting or outer-reference scoring',
         'checkpoint_availability':'Inner files contain only final epoch-300 weights plus metric curves. Intermediate checkpoint scores and weights are absent. Exact intermediate ensemble selection cannot be reconstructed without rerunning training.',
         'selection_rule':'Per seed, require nonnegative exact P@5 improvement in both target exchanges; choose highest exchange-average improvement, tie earliest. Outer deployment averages physical scores from two separately selected seeds.',
         'code_hashes':{n:sha(OLD/n) for n in ('train_boundary.py','boundary_components.py','common.py')},'tasks':{}}
 allcurves=[];endpoint=[]
 for source in ('D3','D1','D2'):
  selections=json.loads((OLD/'selections'/f'{source}.json').read_text())
  for h in (0,1):
   task=f'{source}_held{h}';view=OLD/'views'/source/f'held{h}'
   cal=load(view/'calibration.npz');split=json.loads((view/'INNER_SPLIT.json').read_text())
   groups=np.array([split[str(t)] for t in cal['target_ids']]);entries=selections['tasks'][task]
   chosen=[];curves={};fitpaths={};seeds=(2026111511,2026111512)
   perseed={}
   for seed in seeds:
    e=entries[f'boundary_context_seed{seed}'];eligible=[x for x in e['selection'] if x['eligible']]
    recompute=max(eligible,key=lambda x:(x['mean_gain'],-x['epoch']))['epoch']
    assert recompute==e['selected_epoch'];chosen.append(recompute)
    curves[seed]=[];fitpaths[seed]=[]
    for i,entry in enumerate(e['inner_fits']):
     p=OLD/entry['path'];curve=json.loads((p/'curve.json').read_text())
     assert sha(p/'curve.json')==entry['curve_sha256']
     assert sha(p/'weights.pt')==entry['weights_sha256']
     curves[seed].append(curve);fitpaths[seed].append(p)
     for c,choice in zip(curve,e['selection']):
      assert c['epoch']==choice['epoch']
      assert c['validation']['precision']-curve[0]['validation']['precision']==choice['gains'][i]
     allcurves.append({'task':task,'seed':seed,'exchange':i,
       'loss_reduction':curve[0]['loss']-curve[-1]['loss'],
       'train_P5_gain':curve[-1]['train']['precision']-curve[0]['train']['precision'],
       'validation_P5_gain':curve[-1]['validation']['precision']-curve[0]['validation']['precision'],
       'loss_strictly_decreases_all_saved_intervals':all(a['loss']>b['loss'] for a,b in zip(curve,curve[1:])),
       'validation_gains_at_saved_epochs':[c['validation']['precision']-curve[0]['validation']['precision'] for c in curve],
       'selected_epoch':recompute})
    positive_mean_but_veto=[c['epoch'] for c in e['selection'] if c['mean_gain']>1e-14 and not c['eligible']]
    perseed[str(seed)]={'selected_epoch':recompute,'choices':e['selection'],'positive_mean_but_exchange_veto_epochs':positive_mean_but_veto}
   # Score only calibration-side inner validation at the archived final epoch.
   exchanges=[]
   for i in (0,1):
    tr=subset(cal,np.flatnonzero(groups!=i));va=subset(cal,np.flatnonzero(groups==i))
    pp=[];seedvals=[];val_losses=[];support=tensor(ref_support(va,5))
    for seed in seeds:
     p=fitpaths[seed][i];norm=BoundaryNormalizer.from_file(p/'normalization.npz')
     saved=torch.load(p/'weights.pt',map_location='cpu',weights_only=True)
     m=ReliabilityGate(**saved['config']).double();m.load_state_dict(saved['state_dict']);m.eval()
     t=norm.tensors(va,'boundary_context',5);score=predict(m,t);pp.append(score)
     pv=float(precision(score,va['reference'],va['ownmask'],5).mean())
     assert abs(pv-curves[seed][i][-1]['validation']['precision'])<1e-14
     seedvals.append(pv)
     with torch.no_grad():
      base_loss=float(list_loss(t['base']/t['scale'],t['mask'],support[0],support[1],5,.5))
      end_loss=float(list_loss(tensor(score)/t['scale'],t['mask'],support[0],support[1],5,.5))
     val_losses.append({'seed':seed,'baseline_validation_rank_loss':base_loss,'epoch300_validation_rank_loss':end_loss,'improvement':base_loss-end_loss})
    baseline=float(precision(va['base'],va['reference'],va['ownmask'],5).mean())
    ens=float(precision(np.mean(pp,0),va['reference'],va['ownmask'],5).mean())
    item={'exchange':i,'training_targets':len(tr['target_ids']),'validation_targets':len(va['target_ids']),
      'baseline_P5':baseline,'seed_P5':seedvals,'epoch300_ensemble_P5':ens,
      'epoch300_ensemble_gain':ens-baseline,'mean_seed_gain':float(np.mean(seedvals))-baseline,
      'ensemble_minus_mean_seed_P5':ens-float(np.mean(seedvals)),
      'validation_ranking_loss':val_losses}
    exchanges.append(item);endpoint.append({'task':task,**item})
   result['tasks'][task]={'calibration_targets':len(cal['target_ids']),'selected_epochs':chosen,'per_seed':perseed,
      'final_epoch_inner_ensemble':exchanges,
      'final_epoch_ensemble_eligible_both_exchanges':all(x['epoch300_ensemble_gain']>=0 for x in exchanges),
      'final_epoch_mean_individual_precision_eligible_both_exchanges':all(x['mean_seed_gain']>=0 for x in exchanges),
      'note':'No epoch/seed change applied. Epoch300 reuses saved final weights; middle-epoch ensemble quantities cannot be inferred from scalar precisions.'}
 result['curves']=allcurves
 result['summary']={
  'inner_fits':len(allcurves),'all_saved_intervals_loss_decrease_count':sum(x['loss_strictly_decreases_all_saved_intervals'] for x in allcurves),
  'final_train_P5_positive_count':sum(x['train_P5_gain']>0 for x in allcurves),
  'final_validation_P5_positive_count':sum(x['validation_P5_gain']>0 for x in allcurves),
  'final_validation_P5_negative_count':sum(x['validation_P5_gain']<0 for x in allcurves),
  'mean_final_train_P5_gain':float(np.mean([x['train_P5_gain'] for x in allcurves])),
  'mean_final_validation_P5_gain':float(np.mean([x['validation_P5_gain'] for x in allcurves])),
  'zero_epoch_selections':sum(c['selected_epoch']==0 for v in result['tasks'].values() for c in v['per_seed'].values()),
  'selection_count':12,
  'epoch300_ensemble_minus_mean_seed_P5_average':float(np.mean([x['ensemble_minus_mean_seed_P5'] for x in endpoint])),
  'epoch300_ensemble_minus_mean_seed_P5_positive_exchanges':sum(x['ensemble_minus_mean_seed_P5']>1e-14 for x in endpoint),
  'epoch300_ensemble_minus_mean_seed_P5_negative_exchanges':sum(x['ensemble_minus_mean_seed_P5']<-1e-14 for x in endpoint),
  'reference_access':'Only prior inner calibration labels and metric logs; no outer evaluation loaded.'}
 result['recommendation']={'one_change':'Select one common checkpoint for the deployed two-seed score ensemble, based on exact inner-validation P@5 of the ensemble itself, keeping both-exchange nonnegative eligibility and earliest ties.',
   'why':'The evaluated predictor is an average of scores, whose precision is not the average of seed precisions. Separate seed stopping changes the effective update size and fails to evaluate the object deployed.',
   'limit':'This closes selection/deployment mismatch, but does not remove target-domain bias and cannot guarantee an outer gain. All intermediate predictions must be saved during a controlled rerun; do not infer them from scalar curves.'}
 result['script_sha256']=sha(__file__)
 dump(ROOT/'audit/selection_probe.json',result)
 print(json.dumps(result['summary'],indent=2))

if __name__=='__main__':main()
