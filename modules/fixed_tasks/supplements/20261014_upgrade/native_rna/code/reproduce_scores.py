#!/usr/bin/env python3
"""Reproduce the saved scoring endpoints without refitting any native model."""
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    root=a.bundle.resolve();out=a.out.resolve();assert not out.exists();out.mkdir(parents=True)
    checks={}
    for d in ['D3_Rest','D1_Rest','D2_Rest']:
        cfg={'donor':d,'legacy_prediction':str(root/'legacy'/d/'predictions_locked.npz'),
           'reference':str(root/'reference/query_reference.npz'),'legacy_point_results':str(root/'legacy/point_results.json'),
           'reference_draws':{s:str(root/'reference'/d/('draws_'+s+'.npz')) for s in ['shared_labels','batch_separate']},
           'FR':str(root/'models'/d/'FR'),'ASH':str(root/'models'/d/'ASH')}
        cp=out/(d+'_config.json');cp.write_text(json.dumps(cfg,indent=2)+'\n')
        subprocess.run([sys.executable,str(root/'code/score_native.py'),'--config',str(cp),'--out',str(out/d)],check=True)
        old=np.load(root/'original_scores'/d/'scores.npz',allow_pickle=False);new=np.load(out/d/'scores.npz',allow_pickle=False)
        errors={}
        for k in old.files:
            if old[k].dtype.kind in 'fci':
                err=float(np.max(abs(old[k]-new[k]))) if old[k].size else 0.0;assert err<1e-12,k;errors[k]=err
            else:assert np.array_equal(old[k],new[k]),k
        checks[d]=errors;old.close();new.close()
        for scheme in ['shared_labels','batch_separate']:
            name='reference_propagation_'+scheme+'.npz'
            with np.load(root/'original_scores'/d/name,allow_pickle=False) as before, np.load(out/d/name,allow_pickle=False) as after:
                assert np.array_equal(before['arm_names'],after['arm_names'])
                np.testing.assert_allclose(before['expected_members'],after['expected_members'],rtol=0,atol=1e-12)

    subprocess.run([sys.executable,str(root/'code/combine_sources.py'),'--scores-root',str(out),'--out',str(out/'equal_source')],check=True)
    (out/'REPRODUCTION_CHECK.json').write_text(json.dumps({'status':'PASS','native_refits':0,'maxabs_checks':checks},indent=2)+'\n')
    print(json.dumps({'status':'PASS','native_refits':0,'out':str(out)},indent=2))

if __name__=='__main__':main()
