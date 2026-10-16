#!/usr/bin/env python3
"""Paired native external-reference comparison on exactly the original A19-R axes.

No training or method selection occurs here. Unavailable arms are explicitly
listed as unevaluated. Conditional query-block intervals are descriptive.
"""
from pathlib import Path
import argparse, csv, gzip, hashlib, json
import numpy as np

K = 10
SEED = 2026100502
B = 2000

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def arr(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def membership(y, own):
    y = np.asarray(y,dtype=float)
    assert y.ndim == 2 and y.shape == own.shape and np.isfinite(y).all()
    s = np.concatenate([y,-y],axis=0)
    mask = np.concatenate([own,own],axis=0)
    assert np.all(s.shape[1]-mask.sum(1) >= K)
    s[mask] = -np.inf
    threshold = np.partition(s,s.shape[1]-K,axis=1)[:,-K,None]
    above, tie = s>threshold, s==threshold
    p = above.astype(float)+tie*((K-above.sum(1))/tie.sum(1))[:,None]
    assert np.allclose(p.sum(1),K,rtol=0,atol=1e-12) and not p[mask].any()
    return p

def ordered_indices(have, want):
    h = np.asarray(have).astype(str); w = np.asarray(want).astype(str)
    assert len(set(h))==len(h) and len(set(w))==len(w)
    m = {v:i for i,v in enumerate(h)}
    assert set(w)<=set(h), list(set(w)-set(h))[:10]
    return np.array([m[x] for x in w],int)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); cfg=json.loads(a.config.read_text())
    assert cfg['donor'] in ('D3_Rest','D1_Rest','D2_Rest')
    assert not a.out.exists(); a.out.mkdir(parents=True)
    legacy=arr(cfg['legacy_prediction']); ref=arr(cfg['reference'])
    legacy['effects']=np.stack([legacy['raw_effect'],legacy['raw_effect']*legacy['w0y'],legacy['raw_effect']*legacy['W1']])
    provenance={p:sha(p) for p in [cfg['legacy_prediction'],cfg['reference'],str(a.config),__file__]}
    q=legacy['query_ids']; t=legacy['target_ids']; own=legacy['own_mask']; blocks=legacy['query_blocks']
    expected={'D3_Rest':(293,52),'D1_Rest':(289,59),'D2_Rest':(289,63)}[cfg['donor']]
    assert own.shape==expected and np.array_equal(own,q[:,None]==t[None,:])
    old_names=list(legacy['arm_names'].astype(str))
    predictions={new:legacy['effects'][old_names.index(old)] for old,new in
                 [('raw','raw'),('w0y','Euclidean'),('W1','PertRank')]}
    assert np.allclose(predictions['Euclidean'],predictions['raw']*legacy['w0y'],rtol=0,atol=1e-12)
    assert np.allclose(predictions['PertRank'],predictions['raw']*legacy['W1'],rtol=0,atol=1e-12)
    unevaluated={}
    for method in ('FR','ASH'):
        if not cfg.get(method):
            unevaluated[method]=cfg.get('unevaluated',{}).get(method,'Native input/output unavailable')
            continue
        root=Path(cfg[method]); owner=root/'PROJECTION.json'
        ov=json.loads(owner.read_text())
        assert ov['status']=='LOSSLESS_QUERY_PROJECTION_OF_ARCHIVED_MODEL' and ov['method']==method and ov['donor']==cfg['donor']
        name='effects.npz' if method=='FR' else 'posterior.npz'
        model=root/'native'/name
        assert sha(model)==ov['projected_sha256']
        provenance[str(model)]=sha(model);provenance[str(owner)]=sha(owner)
        z=arr(model); genes=z['genes']; targets=z['targets'] if method=='FR' else z['row_ids']
        qi=ordered_indices(genes,q);ti=ordered_indices(targets,t)
        assert len(targets)==len(t), 'No differently sized training candidate pool'
        field='effect' if method=='FR' else 'posterior'
        predictions[method]=z[field][np.ix_(ti,qi)].T
        if method=='FR':
            predictions['FR_x_Euclidean']=predictions['FR']*legacy['w0y']
            predictions['FR_x_PertRank']=predictions['FR']*legacy['W1']
    qi=ordered_indices(ref['query_ids'],q)
    ti=ordered_indices(ref['target_symbols'],legacy['target_symbols'])
    truth=ref['mean_effect'][np.ix_(qi,ti)]
    truth_p=membership(truth,own)
    names=list(predictions)
    probs={k:membership(v,own) for k,v in predictions.items()}
    scores={k:(v*truth_p).sum(1).reshape(2,len(q)).T/K for k,v in probs.items()}
    regression={}
    for old,new in [('raw','raw'),('w0y','Euclidean'),('W1','PertRank')]:
        old_prob=legacy['top10_probability'][old_names.index(old)]
        err=float(np.max(np.abs(probs[new]-old_prob)));assert err<1e-12
        regression[new]={'legacy_membership_maxabs':err,'P10':float(scores[new].mean())}
    if cfg.get('legacy_point_results'):
        old_report=json.loads(Path(cfg['legacy_point_results']).read_text())['sources'][cfg['donor']]
        provenance[cfg['legacy_point_results']]=sha(cfg['legacy_point_results'])
        for old,new in [('raw','raw'),('w0y','Euclidean'),('W1','PertRank')]:
            err=abs(K*float(scores[new].mean())-old_report['means'][old]);assert err<1e-12
            regression[new]['legacy_expected_members_error']=err
    labels=np.unique(blocks); members=[np.flatnonzero(blocks==l) for l in labels]
    rng=np.random.default_rng(SEED)
    weights=np.zeros((B,len(q)))
    for i in range(B):
        ix=np.concatenate([members[j] for j in rng.integers(len(labels),size=len(labels))])
        weights[i]=np.bincount(ix,minlength=len(q))/len(ix)
    draws={k:weights@v.mean(1) for k,v in scores.items()}
    pairs=[('PertRank','FR'),('PertRank','ASH'),('FR_x_PertRank','FR'),
           ('FR_x_PertRank','FR_x_Euclidean'),('PertRank','FR_x_PertRank'),
           ('FR','raw'),('FR','Euclidean'),('ASH','raw'),('PertRank','Euclidean'),('Euclidean','raw')]
    contrasts={}; redistribution={}
    candidate_rows=[]
    for aa,bb in pairs:
        if aa not in scores or bb not in scores:continue
        name=aa+'_minus_'+bb;delta=draws[aa]-draws[bb]
        point=float((scores[aa]-scores[bb]).mean())
        contrasts[name]={'P10_difference':point,'expected_members_difference':K*point,
             'conditional_query_block_95_interval':np.quantile(delta,[.025,.975]).tolist(),
             'conditional_query_block_99_interval':np.quantile(delta,[.005,.995]).tolist(),
             'bootstrap_draws':B,'seed':SEED,'query_blocks':len(labels),'confirmatory':False}
        d=probs[aa]-probs[bb]; added=np.maximum(d,0); removed=np.maximum(-d,0)
        assert np.allclose(added.sum(1),removed.sum(1),rtol=0,atol=1e-12)
        support_added=(added*truth_p).sum(1); support_removed=(removed*truth_p).sum(1)
        assert abs(float((support_added-support_removed).mean())-K*point)<1e-12
        redistribution[name]={'mean_slots_moved':float(added.sum(1).mean()),
          'mean_reference_members_added':float(support_added.mean()),
          'mean_reference_members_removed':float(support_removed.mean()),
          'net_mean_reference_members':float((support_added-support_removed).mean()),
          'definition':'Changes in marginal top-K probabilities; reference membership denotes ranking agreement, not validated true positives'}
        if aa=='PertRank' and bb in ['FR','ASH','raw','Euclidean']:
            for ix,j in zip(*np.nonzero(d)):
                qix=ix%len(q)
                candidate_rows.append([name,str(q[qix]),'positive' if ix<len(q) else 'negative',
                   str(t[j]),str(legacy['target_symbols'][j]),float(probs[aa][ix,j]),
                   float(probs[bb][ix,j]),float(d[ix,j]),float(truth_p[ix,j]),float(d[ix,j]*truth_p[ix,j])])
    reference_sensitivity={}
    for scheme,path in cfg.get('reference_draws',{}).items():
        assert scheme in ['shared_labels','batch_separate']
        zz=arr(path)
        assert np.array_equal(zz['query_ids'],q) and np.array_equal(zz['target_ids'],t)
        full=np.bitwise_xor.accumulate(zz['definite_xor'].T,axis=0)
        partial=np.bitwise_xor.accumulate(zz['fractional_xor'].T,axis=0)
        assert full.shape==partial.shape==(500,2*len(q))
        powers=np.left_shift(np.uint64(1),np.arange(len(t),dtype=np.uint64))
        provenance[path]=sha(path)
        rv=np.empty((500,len(names)))
        for i in range(500):
            aa=(full[i,:,None]&powers)!=0;bb=(partial[i,:,None]&powers)!=0
            counts=bb.sum(1)
            fraction=np.divide(K-aa.sum(1),counts,out=np.zeros(len(aa),dtype=float),where=counts>0)
            tp=aa.astype(float)+bb*fraction[:,None]
            assert np.allclose(tp.sum(1),K,rtol=0,atol=1e-12)
            rv[i]=[float((probs[k]*tp).sum(1).mean()) for k in names]
        sr={}
        for aa,bb in pairs:
            if aa not in names or bb not in names:continue
            delta=rv[:,names.index(aa)]-rv[:,names.index(bb)]
            sr[aa+'_minus_'+bb]={'units':'expected members per 10-name list',
                 'median':float(np.median(delta)), 'empirical_2p5_97p5':np.quantile(delta,[.025,.975]).tolist(),
                 'fraction_positive':float(np.mean(delta>0))}
        reference_sensitivity[scheme]={'scope':'empirical external-reference sensitivity range, not population confidence interval; exact original 500 shared reference draws',
            'contrasts':sr}
        np.savez_compressed(a.out/('reference_propagation_'+scheme+'.npz'),
                            arm_names=np.array(names),expected_members=rv,source=np.array(cfg['donor']))
    np.savez_compressed(a.out/'scores.npz',arm_names=np.array(names),
      predictions=np.stack([predictions[k] for k in names]),scores=np.stack([scores[k] for k in names]),
      bootstrap=np.stack([draws[k] for k in names]),query_ids=q,target_ids=t,
      target_symbols=legacy['target_symbols'],query_blocks=blocks,own_mask=own,reference_effect=truth)
    with gzip.open(a.out/'candidate_changes.tsv.gz','wt',newline='') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['contrast','query_id','direction','target_id','target_symbol','PertRank_membership','comparator_membership','delta_membership','reference_membership','delta_reference_support']);writer.writerows(candidate_rows)
    report={'status':'COMPLETE','analysis_label':'POSTHOC_DESCRIPTIVE','donor':cfg['donor'],
      'n_queries':len(q),'n_targets':len(t),'K':K,'n_directions':2,
      'scope':'Conditional on original A19-R cells, query panel, candidate pool and fixed external mean RNA reference. Sources share external reference and queries; no donor-level population or pooled-CI claim.',
      'means':{k:{'P10':float(v.mean()),'expected_members_per_10':K*float(v.mean())} for k,v in scores.items()},
      'contrasts':contrasts,'nomination_redistribution':redistribution,'reference_sensitivity':reference_sensitivity,
      'original_prediction_regression':regression,'unevaluated':unevaluated,'source_sha256':provenance,
      'output_sha256':{p.name:sha(p) for p in a.out.iterdir() if p.is_file()}}
    (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'COMPLETE','donor':cfg['donor'],'means':report['means'],'unevaluated':unevaluated},indent=2))

if __name__=='__main__':main()
