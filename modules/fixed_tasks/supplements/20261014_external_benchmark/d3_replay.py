#!/usr/bin/env python3
"""Reproduce existing D3 results from losslessly packed exact top-K memberships.

No model fitting, raw expression access, or new contrasts. D3 ASH reproduces
its originally reported P@25 only; no previously unreported CI is added.
"""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent/'d3'
ARMS=['raw','euclidean','pertrank','fr','fr_euclidean','fr_pertrank']
PAIRS={'fr_minus_raw':(3,0),'fr_minus_euclidean':(3,1),'pertrank_minus_fr':(2,3),'fr_pertrank_minus_fr':(5,3),'fr_pertrank_minus_fr_euclidean':(5,4),'pertrank_minus_euclidean':(2,1)}
def unpack(data,name,k):
    key=f'{name}__k{k}__'
    above=np.unpackbits(data[key+'above'],axis=-1,count=1200,bitorder='little').astype(bool)
    tied=np.unpackbits(data[key+'tied'],axis=-1,count=1200,bitorder='little').astype(bool)
    return above.astype(np.float64)+tied*((k-above.sum(-1))/tied.sum(-1))[...,None]
def weights(blocks):
    labels,encoded=np.unique(blocks,return_inverse=True)
    sampled=np.random.default_rng(20261012).integers(len(labels),size=(2000,len(labels)))
    counts=np.zeros((2000,len(labels)),dtype=np.int32)
    np.add.at(counts,(np.arange(2000)[:,None],sampled),1)
    query_multiplicities=counts[:,encoded]
    return query_multiplicities/query_multiplicities.sum(1,keepdims=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError('Choose a new output directory')
    expected=json.loads((ROOT/'expected_reports.json').read_text())
    actual={};scores_csv=[];contrasts_csv=[];checks=[]
    def check(name,value,target):
        v=np.asarray(value,dtype=float);t=np.asarray(target,dtype=float)
        error=float(np.max(np.abs(v-t)))
        checks.append({'name':name,'max_abs_error':error,'values':v.size})
        if not np.allclose(v,t,rtol=0,atol=1e-12):raise AssertionError((name,value,target))
    with np.load(ROOT/'memberships.npz',allow_pickle=False) as data:
        qw=weights(data['blocks'])
        truth={k:unpack(data,'reference',k)[None] for k in (25,10)}
        scalar={k:[unpack(data,arm,k) for arm in ARMS[:3]] for k in (25,10)}
        for config,report in expected.items():
            if config=='ash_default':continue
            result={}
            for k in (25,10):
                pred=np.stack(scalar[k]+[unpack(data,config+'__'+arm,k) for arm in ARMS[3:]])
                sq=np.einsum('asqt,bsqt->baq',pred,truth[k],optimize=True)[0]/(2*k)
                boot=qw@sq.T;point=sq.mean(1)
                res={'arms':{},'contrasts':{}}
                for i,arm in enumerate(ARMS):
                    ci=np.quantile(boot[:,i],[.025,.975]).tolist()
                    res['arms'][arm]={'point':float(point[i]),'input_rep_points':[float(point[i])],'query_only_95_interval':ci}
                    golden=report['results'][str(k)]['arms'][arm]
                    for key,val in res['arms'][arm].items():check(f'{config}/P{k}/{arm}/{key}',val,golden[key])
                    scores_csv.append([config,k,arm,float(point[i]),*ci])
                for name,(i,j) in PAIRS.items():
                    diff=boot[:,i]-boot[:,j]
                    val={'point':float(point[i]-point[j]),'input_rep_points':[float(point[i]-point[j])],'query_only_95_interval':np.quantile(diff,[.025,.975]).tolist(),'query_only_99_interval':np.quantile(diff,[.005,.995]).tolist()}
                    res['contrasts'][name]=val
                    golden=report['results'][str(k)]['contrasts'][name]
                    for key,v in val.items():check(f'{config}/P{k}/{name}/{key}',v,golden[key])
                    contrasts_csv.append([config,k,name,val['point'],*val['query_only_95_interval'],*val['query_only_99_interval']])
                result[str(k)]=res
            actual[config]=result
        p_ash=unpack(data,'ash',25)[None]
        ash_score=float((np.einsum('asqt,bsqt->baq',p_ash,truth[25],optimize=True)/(2*25))[0,0].mean())
        ash_expected=expected['ash_default']['default_result']['P25']
        check('ash_default/P25',ash_score,ash_expected)
        actual['ash_default']={'25':{'arms':{'ash':{'point':ash_score}},'scope':'Only originally reported P@25; no added CI or P@10 analysis'}}
        scores_csv.append(['ash_default',25,'ash',ash_score,'',''])
    a.output_dir.mkdir(parents=True)
    (a.output_dir/'RESULTS.json').write_text(json.dumps(actual,indent=2)+'\n')
    with (a.output_dir/'method_scores.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['configuration','K','method','P_at_K','conditional_95_lower','conditional_95_upper']);w.writerows(scores_csv)
    with (a.output_dir/'paired_contrasts.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['configuration','K','contrast','difference','conditional_95_lower','conditional_95_upper','conditional_99_lower','conditional_99_upper']);w.writerows(contrasts_csv)
    audit={'status':'PASS','fr_configurations':21,'default_ash_configs':1,'checks':len(checks),'numeric_values_checked':sum(c['values'] for c in checks),'max_abs_error':max(c['max_abs_error'] for c in checks),'atol':1e-12,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'memberships.npz',ROOT/'expected_reports.json',ROOT/'SOURCE_MANIFEST.json']},'scope':'Fixed-result numerical replay; same fitted predictions, exact ties, original top K and query-block resampling seed. No model fitting or new expression read.'}
    (a.output_dir/'REPRODUCTION_CHECK.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
