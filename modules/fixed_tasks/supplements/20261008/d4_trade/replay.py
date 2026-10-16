#!/usr/bin/env python3
"""Offline D4 post-hoc supplement replay using unchanged D1-D4 core data.

Reconstruct saved fitted-mixture moments, then the five frozen arms' signed
top-25 lists, all 30,000 panel scores and conditional query-block intervals.
This does not refit the empirical-Bayes model or download raw counts.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, sys, time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))
import score_trade_supplement as scorer
from build_support_delta import membership

def verify_manifest(root):
    m=scorer.read_json(HERE/'MANIFEST.json')
    for name,want in m['files'].items():
        assert scorer.sha(HERE/name)==want,('supplement changed',name)
    for name,want in m['base_files'].items():
        assert scorer.sha(root/name)==want,('base dependency changed',name)
    return m

def mixture_replay(weights):
    x=scorer.load_npz(HERE/'data/saved_uniform_mixtures.npz')
    ids={v:i for i,v in enumerate(weights['target_ids'])}
    checked=set();mx=dict(latent_mean=0.,TI=0.,S=0.,weights=0.)
    for row in range(2945):
        i=ids[x['target_ids'][row]];g=int(x['replicates'][row]);assert (i,g) not in checked
        checked.add((i,g));sl=slice(int(x['offsets'][row]),int(x['offsets'][row+1]))
        pi,a,b=(x[k][sl] for k in ('pi','a','b'))
        assert np.isfinite(pi).all() and (pi>=0).all() and abs(pi.sum()-1)<1e-6
        assert np.isfinite(a).all() and np.isfinite(b).all() and (a<=b).all()
        # Two-node quadrature independently integrates each uniform component.
        lo=(a+b)/2-(b-a)/(2*np.sqrt(3));hi=(a+b)/2+(b-a)/(2*np.sqrt(3))
        mu=np.sum(pi*(lo+hi)/2);s=np.sum(pi*(lo**2+hi**2)/2)
        got=dict(latent_mean=mu,TI=s-mu**2,S=s,weights=s/(s+weights['nu'][i,g]))
        for k,v in got.items():
            err=abs(float(v)-float(weights[k][i,g]));mx[k]=max(mx[k],err)
            assert err<1e-10,(row,k,err)
    assert len(checked)==2945
    # Link every saved per-fit TSV to the exact published target/group array.
    tsv_checked=0
    for d in sorted((HERE/'fits').iterdir()):
        meta=scorer.read_json(d/'COMPLETE.json')
        for name,want in meta['outputs'].items():
            assert scorer.sha(d/name)==want,(d.name,name)
        with (d/'result_weights.tsv').open() as f:
            for row in csv.DictReader(f,delimiter='\t'):
                i=ids[row['target']];g=int(row['replicate']);tsv_checked+=1
                for a,b in [('weight','weights'),('latent_mean','latent_mean'),('TI','TI'),('S','S'),('nu','nu')]:
                    assert float(row[a])==weights[b][i,g],(d.name,a)
    assert tsv_checked==2945
    return dict(fits=2945,maximum_absolute_errors=mx,refitting_performed=False,
                reconstruction='two-node Gauss-Legendre quadrature of saved uniform mixture components')

def replay(root, output_dir=None):
    """Return a verification dict; optionally write recomputed arrays and panels.

    root is the unchanged D1-D4 repository root. output_dir must be empty/new.
    """
    root=Path(root)
    if output_dir is not None:
        output_dir=Path(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(output_dir)
    manifest=verify_manifest(root)
    base=scorer.load_npz(root/'data/D4/ranking.npz')
    delta=scorer.load_npz(HERE/'data/TRADE_support_delta.npz')
    w=scorer.load_npz(HERE/'data/D4_TRADE_weights.npz')
    complete=scorer.read_json(HERE/'provenance/WEIGHTS_COMPLETE.json')
    assert complete['fits']==2945 and complete['sha256']==scorer.sha(HERE/'data/D4_TRADE_weights.npz')
    assert np.array_equal(w['target_ids'],base['target_ids'])
    gene_ids=scorer.load_npz(root/'data/D4/control_model.npz', ['gene_ids_keep'])['gene_ids_keep']
    assert np.array_equal(w['gene_ids'],gene_ids)
    for k in ('weights','nu','TI','latent_mean','S'):
        assert w[k].shape==(589,5) and np.isfinite(w[k]).all(),k
    assert (w['nu']>0).all() and (w['TI']>=0).all()
    assert ((w['weights']>=0)&(w['weights']<=1)).all()
    fit_report=mixture_replay(w)
    arms=scorer.BASE_ARMS+[scorer.TRADE]
    assert list(base['arm_names'])==scorer.BASE_ARMS
    weights=np.concatenate((base['weights'],w['weights'].T[None]))
    assert weights.shape==(5,5,589)
    reference=np.zeros((593,2,589))
    for q in range(593):
        sl=slice(int(base['truth_offsets'][q]),int(base['truth_offsets'][q+1]))
        ids=base['truth_ids'][sl].astype(int);assert int(base['own_index'][q]) not in ids
        reference[q]=membership(ids,base['truth_values'][sl],589)
    allscores=np.zeros((5,5,593,2))
    for g in range(5):
        for q in range(593):
            r=g*593+q
            bs=slice(int(base['Y_offsets'][r]),int(base['Y_offsets'][r+1]))
            ds=slice(int(delta['Y_offsets'][r]),int(delta['Y_offsets'][r+1]))
            ids=np.r_[base['Y_ids'][bs],delta['Y_ids'][ds]].astype(int)
            vals=np.r_[base['Y_values'][bs],delta['Y_values'][ds]]
            assert len(np.unique(ids))==len(ids) and int(base['own_index'][q]) not in ids
            order=np.argsort(ids);ids=ids[order];vals=vals[order]
            for arm in range(5):
                p=membership(ids,vals*weights[arm,g,ids],589)
                allscores[arm,g,q]=(p*reference[q]).sum(1)/25
    saved=scorer.load_npz(HERE/'results/query_scores.npz')
    original=scorer.load_npz(root/'reference_results/D4_query_scores.npz')
    result=scorer.read_json(HERE/'results/D4_TRADE_RESULT.json')
    oldresult=scorer.read_json(root/'reference_results/D4.json')
    panels={};count=0;produced={'arm_names':np.asarray(arms)}
    for panel,pk,bk in scorer.PANELS:
        scores=np.ascontiguousarray(allscores[:,:,base[pk]])
        assert np.array_equal(scores,saved[panel+'_scores']),panel
        assert np.array_equal(scores[:4],original[panel+'_scores']),('original scores',panel)
        assert np.array_equal(base['query_ids'][base[pk]],saved[panel+'_query_ids'])
        assert np.array_equal(base[bk],saved[panel+'_query_blocks'])
        panels[panel]=scorer.summaries(scores,arms,base[bk])
        assert panels[panel]==result['panels'][panel],('summary',panel)
        produced[panel+'_scores']=scores
        produced[panel+'_query_ids']=base['query_ids'][base[pk]]
        produced[panel+'_query_blocks']=base[bk]
        count+=scores.size
    assert count==30000
    assert result['selected_competitor_unchanged']=='PR_M'
    assert result['original_H1_H2_decision_unchanged']==oldresult['decision']
    # Invoke the original adjudicator with its original four-arm summaries only.
    spec=importlib.util.spec_from_file_location('original_frozen_summary',root/'src/vendor/frozen_summary.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    zero=np.flatnonzero(np.all(base['weights'][2]==0,axis=1)).tolist()
    assert frozen.adjudicate(oldresult['panels']['responsive'],'PR_M',zero)==oldresult['decision']
    report=dict(status='D4_SUPPLEMENT_REPLAY_PASS',query_scores_exact=count,
                original_query_scores_exact=24000,all_summary_records_exact=True,
                original_H1_H2_and_selected_PR_M_preserved=True,
                mixture_replay=fit_report,
                primary=panels['responsive']['repetition_average']['PertRank_minus_TRADE_scalar'],
                scope='post-hoc descriptive fixed TRADE scalar adaptation; conditional query-block intervals',
                base_files_verified=len(manifest['base_files']),supplement_files_verified=len(manifest['files']))
    if output_dir is not None:
        output_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output_dir/'query_scores.npz',**produced)
        scorer.dump(output_dir/'recomputed_panels.json',panels)
        scorer.dump(output_dir/'verification.json',report)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root-base-data',type=Path,required=True,help='D1-D4 repository root')
    p.add_argument('--output',type=Path,required=True,help='New verification JSON; original results are never overwritten')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    started=time.monotonic();report=replay(a.root_base_data.resolve())
    report['elapsed_seconds']=time.monotonic()-started
    scorer.dump(a.output,report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
