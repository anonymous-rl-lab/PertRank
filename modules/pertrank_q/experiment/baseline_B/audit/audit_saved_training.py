"""Actual saved-normalizer and training-interface audit; never reads held references."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import train_boundary as t
import numpy as np

def run():
 plan=t.check_plan();fits=[]
 for source in t.SOURCES:
  for half in (0,1):
   cal,_=t.read_calibration(plan,source,half);ids={str(x):i for i,x in enumerate(cal['target_ids'])}
   for p in sorted((ROOT/'fits').glob(f'{source}_held{half}_*/FIT.json')):
    f=json.loads(p.read_text());ix=np.array([ids[x] for x in f['training_target_ids']]);v=t.subset(cal,ix)
    if f['validation_target_ids'] is not None:assert not set(f['training_target_ids'])&set(f['validation_target_ids'])
    n=t.BoundaryNormalizer(v,f['k']);saved=t.load(p.parent/'normalization.npz')
    for key in ['source_feature_names','raw_scale','baseline_scale','se_floor','feature_mean','feature_scale','boundary_mean','boundary_scale']:
     assert np.array_equal(getattr(n,key),saved[key]),(p.name,key)
    assert int(saved['fit_k'])==f['k'];x=n.tensors(v,f['arm'],f['k'])
    assert np.all(x['features'].numpy()[:,:,8:11]==0)
    # Replace all observed training reference values; predictor-side transforms
    # must remain unchanged, including the newly added competition contexts.
    poison=dict(v);poison['reference']=np.full_like(v['reference'],98765.);poison['reference_replicates']=np.full_like(v['reference_replicates'],-12345.)
    pn=t.BoundaryNormalizer(poison,f['k']);px=pn.tensors(poison,f['arm'],f['k'])
    for key in x:assert np.array_equal(x[key].numpy(),px[key].numpy()),(p.name,key,'reference dependence')
    fits.append(dict(path=str(p.parent.relative_to(ROOT)),targets=len(ix),k=f['k'],normalizer_exact=True,reference_poison_invariant=True))
 assert len(fits)==72
 imports=t.import_a()
 for source in t.SOURCES:
  lock=json.loads((ROOT/'predictions'/f'{source}_PREDICTION_LOCK.json').read_text())
  for task,entry in lock['tasks'].items():
   p=t.load(ROOT/entry['path']);a=t.load(ROOT/imports['tasks'][task]['path'])
   assert np.array_equal(p['scores'][0],a['scores'])
 report=dict(passed=True,held_references_opened=False,training_performed=False,fits_checked=len(fits),imported_A_prediction_arrays_exact=6,fits=fits)
 t.dump(ROOT/'audit/SAVED_TRAINING_AUDIT.json',report);return report
if __name__=='__main__':
 z=run();print(json.dumps({k:v for k,v in z.items() if k!='fits'},indent=2))
