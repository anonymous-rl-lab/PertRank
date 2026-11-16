#!/usr/bin/env python3
"""Score fixed four-arm predictions against the existing 500 reference draws."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from score_four_arm import PAIRS, load

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',type=Path,default=ROOT/'four_arm_results')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out.resolve();assert not out.exists();out.mkdir(parents=True)
    validations=json.loads((ROOT/'reference_draws/DOWNLOAD_VALIDATION.json').read_text())
    valid={v['file']:v['sha256'] for v in validations}
    summary={}
    def summarize(vals,names):
        contrasts={}
        for name,(a,b) in PAIRS.items():contrasts[name]=vals[:,names.index(a)]-vals[:,names.index(b)]
        contrasts['interaction_original_minus_compressed_geometry']=contrasts['geometry_on_original']-contrasts['geometry_on_compressed']
        return {k:{'P10_median':float(np.median(v)),'P10_empirical_2p5_97p5':np.quantile(v,[.025,.975]).tolist(),
            'fraction_positive':float(np.mean(v>1e-12)),'fraction_negative':float(np.mean(v < -1e-12)),
            'fraction_numerical_zero':float(np.mean(abs(v)<=1e-12))} for k,v in contrasts.items()}
    for scheme in ('shared_labels','batch_separate'):
        by_source={};all_values=[]
        for d in ('D3','D1','D2'):
            score=load(args.predictions/(d+'_scores.npz'))
            path=ROOT/f'reference_draws/{d}_{scheme}.npz'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==valid[path.name]
            z=load(path);q=score['query_ids'];t=score['target_ids'];names=list(score['arm_names'])
            assert np.array_equal(z['query_ids'],q) and np.array_equal(z['target_ids'],t)
            full=np.bitwise_xor.accumulate(z['definite_xor'].T,axis=0)
            partial=np.bitwise_xor.accumulate(z['fractional_xor'].T,axis=0)
            assert full.shape==partial.shape==(500,2*len(q))
            powers=np.left_shift(np.uint64(1),np.arange(len(t),dtype=np.uint64))
            vals=[]
            for i in range(500):
                a=(full[i,:,None]&powers)!=0;b=(partial[i,:,None]&powers)!=0
                count=b.sum(1);frac=np.divide(10-a.sum(1),count,out=np.zeros(len(a)),where=count>0)
                truth=a.astype(float)+b*frac[:,None]
                assert np.allclose(truth.sum(1),10,atol=1e-12,rtol=0)
                assert not truth[np.concatenate([score['own_mask'],score['own_mask']])].any()
                vals.append((score['memberships']*truth[None,:,:]).sum((1,2))/(10*2*len(q)))
            vals=np.array(vals);all_values.append(vals)
            by_source[d]=summarize(vals,names)
            np.savez_compressed(out/f'{d}_{scheme}.npz',arm_names=np.array(names),P10=vals)
        average=np.mean(all_values,axis=0)
        by_source['equal_source']=summarize(average,names)
        np.savez_compressed(out/f'equal_source_{scheme}.npz',arm_names=np.array(names),P10=average)
        summary[scheme]=by_source
    report={'status':'COMPLETE','draws_per_scheme':500,'units':'P@10 difference',
        'interpretation':'Empirical external-reference sensitivity ranges conditional on fixed predictions, not population confidence intervals. Same external experiment and draw identities are shared by three input sources. No new reference draws, candidate changes, or model fitting.',
        'results':summary}
    (out/'reference_sensitivity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({s:{d:r['readout_original_minus_compressed_at_wM'] for d,r in v.items()} for s,v in summary.items()},indent=2))

if __name__=='__main__':main()
