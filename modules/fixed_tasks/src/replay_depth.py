#!/usr/bin/env python3
"""Replay all archived seven-depth curves from exact top-K boundary memberships.

The packed records retain every tied candidate, not a chosen top-K list. Depth
expression magnitudes and below-cutoff ordering are intentionally omitted. D1's
archived depth experiment used the historical panel and is labeled accordingly;
it must not be substituted for the corrected D1 primary analysis.
"""
from pathlib import Path
import argparse, importlib.util, json, os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('frozen_depth_math',HERE/'vendor/depth_equivalence.py')
math=_mod=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_mod)
PANELS=(('responsive','responsive_positions','blocks_responsive'),('terciles','tercile_positions','blocks_terciles'))

def probabilities(z,prefix,index=()):
    """Return [..., query, sign, candidate] probabilities, preserving all ties."""
    nt=int(z['n_candidates'])
    a=np.unpackbits(z[prefix+'_above'][index],axis=-1,count=nt,bitorder='little').astype(float)
    t=np.unpackbits(z[prefix+'_tied'][index],axis=-1,count=nt,bitorder='little')
    n=z[prefix+'_numerator'][index];d=z[prefix+'_denominator'][index]
    if np.any(d==0) or np.any(n>d):raise AssertionError('invalid stored boundary fraction')
    p=a+t*(n/d)[...,None]
    if not np.allclose(p.sum(-1),int(z['K']),atol=1e-12,rtol=0):raise AssertionError('invalid membership mass')
    own=z['query_gene_ids'][:,None]==z['target_ids'][None,:]
    if p[...,np.arange(len(own)),:,:].shape!=p.shape:raise AssertionError('query axis mismatch')
    if np.any(p*own[:,None,:]):raise AssertionError('self-target not excluded')
    return p

def score_arrays(z):
    truth=probabilities(z,'truth')
    nq=len(z['query_gene_ids']);arms=np.empty((3,5,nq,2));raw=np.empty((7,5,nq,2))
    for a in range(3):
        for r in range(5):arms[a,r]=(probabilities(z,'arms20',(a,r))*truth).sum(-1)/25
    for d in range(7):
        for r in range(5):raw[d,r]=(probabilities(z,'raw',(d,r))*truth).sum(-1)/25
    assert np.array_equal(raw[0],arms[0])
    return raw,arms

def _compare(actual,expected,path,stats):
    if isinstance(expected,dict):
        assert set(actual)==set(expected),(path,'keys',set(actual)^set(expected))
        for key in expected:_compare(actual[key],expected[key],path+'/'+key,stats)
    elif isinstance(expected,list):
        assert len(actual)==len(expected),(path,'length')
        for i,(a,b) in enumerate(zip(actual,expected)):_compare(a,b,path+'/'+str(i),stats)
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        assert abs(float(actual)-float(expected))<=1e-12,(path,actual,expected)
        stats['numeric_values']+=1;stats['max_absolute_difference']=max(stats['max_absolute_difference'],abs(float(actual)-float(expected)))
    else:assert actual==expected,(path,actual,expected)

def replay_donor(root,donor):
    root=Path(root);z=np.load(root/'data'/donor/'depth.npz',allow_pickle=False)
    raw,arms=score_arrays(z)
    ref=json.loads((root/'reference_results'/(donor+'_depth_source.json')).read_text())
    source=np.load(root/'reference_results'/(donor+'_depth_source_arrays.npz'),allow_pickle=False)
    expected=ref['secondary_cell_equivalence'] if donor=='D4' else ref['panels']
    reports={};stats={'donor':donor,'panel_scope':str(z['panel_scope']),'numeric_values':0,'max_absolute_difference':0.0,'query_scores_bitwise_equal':True,'bootstrap_checks':{}}
    for panel,pk,bk in PANELS:
        pos=z[pk];r=np.ascontiguousarray(raw[:,:,pos]);a=np.ascontiguousarray(arms[:,:,pos])
        sr=source[panel+('_raw_depth_scores' if donor=='D4' else '_raw_scores')]
        sa=source[panel+('_scores' if donor=='D4' else '_arms20_scores')][:3]
        assert np.array_equal(r,sr),(donor,panel,'raw scores differ')
        assert np.array_equal(a,sa),(donor,panel,'arm scores differ')
        stats['numeric_values']+=r.size+a.size
        report,boots=math.summarize(r,a,z[bk])
        if donor=='D4':
            report.update(role='prespecified secondary estimation; cannot rescue primary failure',interpretation='ranking-input equivalent depth conditional on this 100-cell reservoir; not total experimental cost saving',interpolation_caveat='piecewise linear inverse may overestimate equivalent depth on a concave curve; no smoothing or extrapolation')
        _compare(report,expected[panel],donor+'/'+panel,stats)
        for key,value in boots.items():
            old=source[panel+'_'+key]
            assert np.array_equal(np.isnan(value),np.isnan(old)),(donor,panel,key,'NaN states')
            assert np.array_equal(np.isposinf(value),np.isposinf(old)),(donor,panel,key,'positive infinite states')
            assert np.array_equal(np.isneginf(value),np.isneginf(old)),(donor,panel,key,'negative infinite states')
            finite=np.isfinite(old);err=float(np.max(abs(value[finite]-old[finite]))) if finite.any() else 0.
            assert err<=1e-12,(donor,panel,key,err)
            stats['bootstrap_checks'][panel+'_'+key]={'values':int(value.size),'max_absolute_difference':err,'bitwise_equal_including_nan':bool(np.array_equal(value,old,equal_nan=True))}
            stats['numeric_values']+=value.size;stats['max_absolute_difference']=max(stats['max_absolute_difference'],err)
        reports[panel]=report
    stats['status']='PASS'
    return {'recomputed':reports,'verification':stats}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=HERE.parent);p.add_argument('--output',type=Path);p.add_argument('--donors',nargs='+',default=['D1','D2','D3','D4']);a=p.parse_args()
    out={}
    for d in a.donors:
        out[d]=replay_donor(a.root,d);print('DEPTH_PASS',d,'max_abs',out[d]['verification']['max_absolute_difference'],flush=True)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
