#!/usr/bin/env python3
"""Rebuild four-arm predictions from frozen weights; never fit or select."""
from pathlib import Path
import argparse,json,os,tempfile
import numpy as np
import train_optimization as tr
ROOT=Path(__file__).resolve().parent

def safe(root,value):
    p=Path(value)
    if p.is_absolute() or '..' in p.parts:raise ValueError(f'Unsafe path: {value}')
    return root/p

def main(base):
    base=Path(base).resolve();tr.configure(base);plan=tr.checked_plan()
    al=tr.read(ROOT/'predictions/ALL_PREDICTIONS_LOCK.json')
    assert al['plan_sha256']==tr.sha(ROOT/'PLAN_LOCK.json')
    records={}
    # Check all source locks, fit identities, and weight files before inference.
    for source in tr.SOURCES:
        lp=ROOT/'predictions'/f'{source}_PREDICTION_LOCK.json'
        assert tr.sha(lp)==al['source_lock_sha256'][source]
        lock=tr.read(lp);assert lock['plan_sha256']==al['plan_sha256']
        for s,h in lock['selection_sha256'].items():assert tr.sha(ROOT/'selections'/f'{s}.json')==h
        for task,entry in lock['tasks'].items():
            assert entry==al['tasks'][task]
            for key,fit in entry['fits'].items():
                assert fit['fit_root'] in ('base','new')
                fd=safe(base if fit['fit_root']=='base' else ROOT,fit['fit_path'])
                assert tr.sha(fd/'weights.pt')==fit['weights_sha256'],(task,key,'weight')
                assert tr.sha(fd/'normalization.npz')==fit['normalization_sha256'],(task,key,'norm')
            records[task]=entry
    assert set(records)=={f'{s}_held{h}' for s in tr.SOURCES for h in (0,1)}
    checked={}
    for source in tr.SOURCES:
        oldlock=tr.read(base/'predictions'/f'{source}_PREDICTION_LOCK.json')
        for half in (0,1):
            task=f'{source}_held{half}';entry=records[task];meta=plan['tasks'][task]
            vp=base/'views'/source/f'held{half}'/'held_features.npz'
            assert tr.sha(vp)==meta['held_features_sha256']
            held=tr.load(vp);assert not any(k.startswith('reference') for k in held)
            scores=np.empty((len(tr.ARMS),len(tr.SEEDS))+held['base'].shape)
            cache={}
            for ai,arm in enumerate(tr.ARMS):
                for si,seed in enumerate(tr.SEEDS):
                    fit=entry['fits'][f'{arm}_seed{seed}'];fd=safe(base if fit['fit_root']=='base' else ROOT,fit['fit_path'])
                    key=(fit['fit_root'],fit['fit_path'])
                    if key not in cache:
                        m=tr.restore_model(fd/'weights.pt');m.eval()
                        norm=tr.BoundaryNormalizer.from_file(fd/'normalization.npz')
                        cache[key]=tr.predict(m,norm.tensors(held,'boundary_context',10))
                    scores[ai,si]=cache[key]
            oldp=base/oldlock['tasks'][task]['path'];assert tr.sha(oldp)==oldlock['tasks'][task]['sha256']
            old=tr.load(oldp);idx=old['arm_names'].tolist().index('boundary_context')
            assert np.array_equal(scores[0],old['scores'][idx]),('old B rebuild changed',task)
            payload=dict(scores=scores,arm_names=np.array(tr.ARMS),seeds=np.array(tr.SEEDS),
                **{key:held[key] for key in ('base','raw','euclidean','ownmask','query_ids','query_block','target_ids')})
            path=safe(ROOT,entry['path']);path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():assert tr.sha(path)==entry['sha256'],('Existing prediction changed',task)
            handle,temp=tempfile.mkstemp(prefix='.rebuild_',suffix='.npz',dir=path.parent);os.close(handle)
            tp=Path(temp)
            try:
                np.savez_compressed(tp,**payload)
                assert tr.sha(tp)==entry['sha256'],('Prediction bytes differ',task,tr.sha(tp),entry['sha256'])
                if not path.exists():tp.replace(path)
            finally:tp.unlink(missing_ok=True)
            checked[task]=dict(prediction_sha256=tr.sha(path),bytes_match_original_lock=True,
                old_B_preserved=True,reference_free_feature_input=True,fit_count=len(entry['fits']),unique_weight_predictions=len(cache))
    result=dict(passed=True,training_performed=False,selection_performed=False,all_six_locks_verified_before_inference=True,
        tasks=checked,plan_sha256=tr.sha(ROOT/'PLAN_LOCK.json'),torch_version=str(tr.torch.__version__))
    tr.dump(ROOT/'REBUILD_OPTIMIZATION_CHECK.json',result)
    print(json.dumps(dict(passed=True,tasks=len(checked),training_performed=False),indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-root',type=Path,default=ROOT/'baseline_B')
    a=p.parse_args();main(a.base_root)
