#!/usr/bin/env python3
"""Recalculate A19-R scores from fixed_tasks intermediates; no counts or model fitting."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
ROOT=Path(__file__).resolve().parent
DONORS=['D3_Rest','D1_Rest','D2_Rest'];ARMS=['raw','w0y','W1','deep'];K=10

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def membership(y,own):
    z=np.concatenate([y,-y],axis=0).copy();mask=np.concatenate([own,own],axis=0);z[mask]=-np.inf
    cut=np.partition(z,-K,axis=1)[:,-K];above=z>cut[:,None];tie=z==cut[:,None]
    p=above.astype(float)+tie*((K-above.sum(1))/tie.sum(1))[:,None]
    assert np.allclose(p.sum(1),K,rtol=0,atol=1e-12) and not p[mask].any()
    return p

def bootstrap(a,b,blocks):
    d=np.asarray(a)-np.asarray(b)
    if d.ndim==2:d=d.mean(1)
    rng=np.random.default_rng(2026100502);bl=np.unique(blocks);mem={v:np.flatnonzero(blocks==v) for v in bl}
    draws=np.array([d[np.concatenate([mem[v] for v in rng.choice(bl,len(bl),replace=True)])].mean() for _ in range(2000)])
    return {'point':float(d.mean()),'conditional_query_block_95_interval':np.quantile(draws,[.025,.975]).tolist(),'bootstrap_fraction_nonpositive':float(np.mean(draws<=0)),'draws':2000,'seed':2026100502}

def compare(actual,expected,path='root'):
    """Compare regenerated scientific fields; the historical scorer-error diagnostic is excluded."""
    errors=[]
    if isinstance(expected,dict):
        for k,v in expected.items():
            if k in ['score_rows_crosscheck_maxabs','reference_float64_files_sha256']:continue
            assert k in actual,(path,k,'missing')
            errors+=compare(actual[k],v,path+'/'+k)
    elif isinstance(expected,list):
        assert len(actual)==len(expected),path
        for i,(a,b) in enumerate(zip(actual,expected)):errors+=compare(a,b,path+'/'+str(i))
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        err=abs(float(actual)-float(expected));assert err<1e-12,(path,actual,expected);errors.append(err)
    else:assert actual==expected,(path,actual,expected)
    return errors

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir','--out',dest='out',type=Path,required=True)
    ap.add_argument('--fixed-root',type=Path,default=ROOT.parents[1]/'fixed_tasks');a=ap.parse_args()
    out=a.out.resolve();assert not out.exists(),f'Output exists: {out}';out.mkdir(parents=True);start=time.monotonic()
    fixed=a.fixed_root.resolve();native=fixed/'supplements/20261014_upgrade/native_rna'
    assert native.is_dir(),f'Fixed-task module absent: {native}; provide --fixed-root'
    provenance=json.loads((ROOT/'PROVENANCE.json').read_text())
    assert sha(ROOT/'diagnostic_memberships.npz')==provenance['diagnostic_memberships_sha256']
    for record in provenance['files']:assert sha(ROOT/record['path'])==record['sha256'],record['path']
    ref=load(native/'reference/query_reference.npz');extra=load(ROOT/'diagnostic_memberships.npz')
    old=json.loads((ROOT/'expected/point_results.json').read_text());oldprop=json.loads((ROOT/'expected/reference_propagation_results.json').read_text())
    qmap={x:i for i,x in enumerate(ref['query_ids'])};tmap={x:i for i,x in enumerate(ref['target_symbols'])}
    # Copy only prose/identity from the historical report; all numeric scientific fields are regenerated below.
    points={k:old[k] for k in ['identity','units','interval_scope','original_A19_comparison']};points['sources']={}
    pack={};source_hashes={};scores={}
    for d in DONORS:
        source=native/'legacy'/d/'predictions_locked.npz';z=load(source);source_hashes[str(source.relative_to(fixed))]=sha(source)
        assert np.array_equal(extra[d+'_query_ids'],z['query_ids']) and np.array_equal(extra[d+'_target_ids'],z['target_ids'])
        assert np.array_equal(z['own_mask'],z['query_ids'][:,None]==z['target_ids'][None,:])
        effects=np.stack([z['raw_effect'],z['raw_effect']*z['w0y'],z['raw_effect']*z['W1']])
        pred=np.stack([membership(e,z['own_mask']) for e in effects]+[extra[d+'_deep_membership']])
        assert np.array_equal(pred[:3],z['top10_probability']),d+' reconstructed source memberships'
        qi=np.array([qmap[g] for g in z['query_ids']]);ti=np.array([tmap[t] for t in z['target_symbols']]);truth=membership(ref['mean_effect'][np.ix_(qi,ti)],z['own_mask'])
        nq=len(qi);hit=np.einsum('arc,rc->ar',pred,truth).reshape(4,2,nq).transpose(0,2,1);scores[d]=hit
        floor=(100/(len(ti)-z['own_mask'].sum(1)))[:,None]*np.ones((1,2));b=z['query_blocks']
        r={'targets':len(ti),'queries':nq,'own_exclusions':int(z['own_mask'].sum()),'random_expected_members':float(floor.mean()),'means':{arm:float(hit[i].mean()) for i,arm in enumerate(ARMS)},'W1_minus_raw':bootstrap(hit[2],hit[0],b),'W1_minus_w0y':bootstrap(hit[2],hit[1],b),'deep_minus_random':bootstrap(hit[3],floor,b)}
        add=np.maximum(pred[2]-pred[1],0);rem=np.maximum(pred[1]-pred[2],0);turn=float(add.sum());assert np.allclose(add.sum(1),rem.sum(1))
        r['turnover_vs_w0y']={'mean_added_positions':float(add.sum(1).mean()),'added_support_rate':float((add*truth).sum()/turn) if turn else None,'removed_support_rate':float((rem*truth).sum()/turn) if turn else None,'exact_delta':float(((add-rem)*truth).sum(1).mean())}
        points['sources'][d]=r;pack[d]=(z,pred)
    points['equal_source_mean_descriptive']={arm:float(np.mean([points['sources'][d]['means'][arm] for d in DONORS])) for arm in ARMS}
    errors=compare(points,old);np.savez_compressed(out/'point_scores.npz',**scores)
    propagated={k:oldprop[k] for k in ['identity','shared_reference_per_draw','source_cells_queries_candidates_controls_fixed','draws']};propagated['schemes']={}
    prop_expected=json.loads((ROOT/'EXPECTED_PROPAGATION_ARRAYS.json').read_text());array_checks={}
    for scheme in ['shared_labels','batch_separate']:
        values=np.empty((500,3,4))
        for di,d in enumerate(DONORS):
            z,pred=pack[d];source=native/'reference'/d/('draws_'+scheme+'.npz');zz=load(source)
            source_hashes[str(source.relative_to(fixed))]=sha(source)
            assert np.array_equal(zz['query_ids'],z['query_ids']) and np.array_equal(zz['target_ids'],z['target_ids'])
            full=np.bitwise_xor.accumulate(zz['definite_xor'].T,axis=0);partial=np.bitwise_xor.accumulate(zz['fractional_xor'].T,axis=0)
            assert full.shape==partial.shape==(500,2*len(z['query_ids']))
            powers=np.left_shift(np.uint64(1),np.arange(len(z['target_ids']),dtype=np.uint64))
            for it in range(500):
                above=(full[it,:,None]&powers)!=0;tie=(partial[it,:,None]&powers)!=0;counts=tie.sum(1)
                frac=np.divide(K-above.sum(1),counts,out=np.zeros(len(above),float),where=counts>0)
                truth=above.astype(float)+tie*frac[:,None];assert np.allclose(truth.sum(1),K,rtol=0,atol=1e-12)
                values[it,di]=np.einsum('arc,rc->ar',pred,truth).mean(1)
        arrays={'source_names':np.array(DONORS),'arm_names':np.array(ARMS),'expected_members':values}
        for k,v in arrays.items():
            meta=prop_expected[scheme][k];assert list(v.shape)==meta['shape'] and str(v.dtype)==meta['dtype']
            assert hashlib.sha256(v.tobytes()).hexdigest()==meta['sha256_array_bytes'],(scheme,k,'propagation array differs')
        array_checks[scheme]='ALL_ARRAYS_BITWISE_EXACT'
        np.savez_compressed(out/('reference_propagation_'+scheme+'.npz'),**arrays)
        sr={}
        for di,d in enumerate(DONORS+['equal_source_mean']):
            v=values[:,di] if di<3 else values.mean(1);sr[d]={}
            for label,base in [('W1_minus_raw',0),('W1_minus_w0y',1)]:
                delta=v[:,2]-v[:,base];sr[d][label]={'median':float(np.median(delta)),'empirical_2p5_97p5':np.quantile(delta,[.025,.975]).tolist(),'fraction_positive':float(np.mean(delta>0))}
        propagated['schemes'][scheme]=sr
    qc={}
    for d,(z,pred) in pack.items():
        truth=extra[d+'_qc_membership'];assert np.allclose(truth.sum(1),K,rtol=0,atol=1e-12)
        v=np.einsum('arc,rc->ar',pred,truth).mean(1);qc[d]={'means':dict(zip(ARMS,map(float,v))),'W1_minus_raw':float(v[2]-v[0]),'W1_minus_w0y':float(v[2]-v[1])}
    propagated['author_bad_controls_excluded_point_sensitivity']=qc;errors+=compare(propagated,oldprop)
    for name,data in [('point_results.json',points),('reference_propagation_results.json',propagated)]:
        (out/name).write_text(json.dumps(data,indent=2)+'\n')
    report={'status':'PASS','identity':'A19-R fixed-cell development reconstruction, not original A19 replay',
        'reproduction_level':'Frozen-effect reranking plus query-block bootstrap and frozen-reference-membership replay',
        'recomputed':['3 shallow arms exact top10 ranks','4-arm point scores and deep diagnostic','2000 fixed-seed query-block bootstrap per original contrast','turnover accounting','both 500-draw reference sensitivities','author bad-control QC point scores'],
        'fixed_intermediates':['original raw effects and weights','deep-arm membership','reference means','both 500 reference membership sets','author-QC reference membership'],
        'not_recomputed':['source cell counts/moments/control directions','weights from raw counts','external normalization/reference draws from raw counts','FR/ASH model fitting','original A19 missing cell selection'],
        'reference_scope':'Three source inputs share one external experiment; draw ranges are conditional sensitivities, not population confidence intervals.',
        'numeric_fields_checked':len(errors),'max_abs_difference':max(errors,default=0.),'reference_propagation_checks':array_checks,
        'excluded_historical_diagnostics':['score_rows_crosscheck_maxabs is a past separate-scorer diagnostic','reference_float64_files_sha256 belongs to old unbundled containers'],
        'fixed_input_sha256':source_hashes,'runtime_seconds':time.monotonic()-start,'numpy_version':np.__version__,'new_experiments':0,'new_model_fits':0,'network_calls':0}
    (out/'REPRODUCTION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['status','numeric_fields_checked','max_abs_difference','runtime_seconds','reference_propagation_checks']},indent=2))
if __name__=='__main__':main()
