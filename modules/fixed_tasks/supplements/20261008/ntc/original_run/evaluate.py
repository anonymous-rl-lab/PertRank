#!/usr/bin/env python3
"""Evaluate the fixed diagnostic once all selected rows have been read."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.sparse import load_npz
from stats_impl import infer_ntc,infer_target
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def composition(s):
    lanes=s['lane_codes'];nla=len(s['lane_categories']);a=s['ntc_positions'];t=s['target_positions']
    counts=lambda pp:np.array([np.bincount(lanes[x],minlength=nla) for x in pp.reshape(-1,20)]).reshape(*pp.shape[:-1],nla)
    nc=counts(a);tc=counts(t)
    tv=lambda x,y:.5*np.abs(x-y).sum(axis=-1)
    nh=tv(nc[:,0]/20,nc[:,1]/20);th=tv(tc[:,:,0]/20,tc[:,:,1]/20)
    tg=tv(tc[:,0].sum(axis=1)/40,tc[:,1].sum(axis=1)/40)
    ngc=nc.sum(axis=1);expected=ngc.sum(1)[:,None]*ngc.sum(0)[None,:]/ngc.sum()
    chi=float(np.sum(np.divide((ngc-expected)**2,expected,out=np.zeros_like(expected,float),where=expected>0)))
    cramers=(chi/(ngc.sum()*min(len(ngc)-1,int(np.sum(expected.sum(0)>0))-1)))**.5
    summary={'ntc_half_lane_tv':{'median':float(np.median(nh)),'max':float(nh.max()),'perfect_balance_guides':int((nh==0).sum())},
      'target_guide_half_lane_tv':{'median':float(np.median(th)),'max':float(th.max())},
      'target_between_guide_lane_tv':{'median':float(np.median(tg)),'min':float(tg.min()),'max':float(tg.max())},
      'selected_ntc_guide_by_lane_cramers_v':float(cramers),'association_sample':'selected 40 cells per NTC guide',
      'interpretation':'Split balancing does not eliminate between-guide lane composition differences. Permutation assumes exchangeable guide spectra and does not adjust lane confounding.'}
    np.savez_compressed(ROOT/'lane_composition.npz',ntc_counts=nc,target_counts=tc,ntc_half_tv=nh,target_half_tv=th,target_between_guide_tv=tg,lanes=s['lane_categories'],ntc_guides=s['ntc_guides'],targets=s['target_names'])
    js(ROOT/'lane_composition.json',summary);return summary

def main():
    done=ROOT/'RESULT.json'
    if done.exists():raise RuntimeError('Final result exists; do not overwrite')
    assert json.loads((ROOT/'RAW_READ_COMPLETE.json').read_text())['status']=='RAW_READ_COMPLETE'
    freeze=json.loads((ROOT/'RUN_FREEZE.json').read_text())
    for p,want in freeze['file_sha256'].items():assert sha(ROOT/p)==want,('freeze changed',p)
    t=time.time();z=np.load(ROOT/'selection.npz',allow_pickle=False);s={k:z[k] for k in z.files};n=len(s['ntc_guides']);nt=len(s['target_names']);p=len(s['keep'])
    slots=np.full(len(s['cell_ids']),-1,np.int64)
    slots[s['ntc_positions']]=np.arange(n*2).reshape(n,2,1)
    slots[s['target_positions']]=n*2+np.arange(nt*4).reshape(nt,2,2,1)
    assert np.all(slots>=0) and np.all(np.bincount(slots)==20)
    sums=np.zeros((n*2+nt*4,p),float);seen=np.zeros(len(slots),bool)
    for rp in sorted((ROOT/'raw_batches').glob('batch_*.json')):
        rec=json.loads(rp.read_text());f=rp.with_suffix('.npz');assert sha(f)==rec['sha256'];rows=np.array(rec['rows'])
        pos=np.searchsorted(s['cell_ids'],rows);assert np.array_equal(s['cell_ids'][pos],rows) and not seen[pos].any()
        x=load_npz(f);assert np.array_equal(np.asarray(x.sum(1)).ravel(),s['total_counts'][pos])
        norm=np.log1p(x[:,s['keep']].toarray().astype(float)*(float(s['scale'])/s['total_counts'][pos])[:,None])
        np.add.at(sums,slots[pos],norm);seen[pos]=True
    assert seen.all();means=sums/20;a=means[:n*2].reshape(n,2,p);b=means[n*2:].reshape(nt,2,2,p)
    np.savez_compressed(ROOT/'half_means.npz',ntc=a,target=b,ntc_guides=s['ntc_guides'],target_names=s['target_names'],target_guides=s['target_guides'],gene_ids=s['gene_ids_full'][s['keep']],scale=s['scale'])
    lane=composition(s);ntcres,boot,null=infer_ntc(a[:,0],a[:,1]);tres,tvals,tboot=infer_target(b)
    np.savez_compressed(ROOT/'resampling.npz',ntc_bootstrap=boot,ntc_permutation=null,target_pair_energy=tvals,target_bootstrap=tboot)
    result={'status':'COMPLETE','donor':'D3_Rest','new_replanning_not_interrupted_script_replay':True,'ntc':ntcres,'target_same_gene_control':tres,'lane_composition':lane,
      'definition':'D = 2 sum_g <A_g-Abar,B_g-Bbar> / ((K-1)*13390); equal mean unordered-guide-pair cross-half difference product',
      'no_shared_control_mean_subtraction':True,'negative_estimates_preserved':True,'normalization_scale':float(s['scale']),
      'interpretation_limits':['Guide-related structure is conditional on the frozen gene axis and normalization scale; not a causal sequence effect.',
        'NTC guide bootstrap resamples the observed guides, paired halves together; it does not propagate donor, feature-axis or normalization uncertainty.',
        'B-spectrum permutation assumes guide exchangeability; lane composition remains a potential contributor.',
        'A negative or nonsignificant result is nondetection, not proof of absence.',
        'The same-target targeting control can reflect guide efficacy and guide-associated factors; it excludes between-target biological differences.'],
      'selection_sha256':sha(ROOT/'selection.npz'),'freeze_sha256':sha(ROOT/'RUN_FREEZE.json'),'half_means_sha256':sha(ROOT/'half_means.npz'),'seconds':time.time()-t}
    js(done,result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
