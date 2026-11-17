#!/usr/bin/env python3
"""Recompute fixed rvalues comparison scores from archived native memberships.

The statistical scorer is extracted from the completed native comparison.
Effect/SE arrays used for native R refitting are not duplicated in this
manuscript payload. This entry point does not refit rvalues or PertRank.
"""
from pathlib import Path
import argparse, csv, json, hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
TASKS=('External_D3_Rest','External_D1_Rest','External_D2_Rest','D4_Rest','D4_Stim8hr')
def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def score_all(pred,out):
    reports={};taskdraws={};summaries=[];contrasts=[]
    for task in TASKS:
        inp=load(ROOT/'data/metadata'/f'{task}.npz')
        e=load(ROOT/'data/evaluation'/f'{task}.npz')
        z=load(pred/f'{task}.npz');p=z['membership'];k=int(inp['K'])
        assert np.array_equal(z['query_ids'],inp['query_ids']) and np.array_equal(z['target_ids'],inp['target_ids'])
        nr,nq,nt=tuple(map(int,inp['effect_shape']))
        assert p.shape==(nr,2*nq,nt) and e['reference_membership'].shape==(2*nq,nt)
        mask=np.concatenate([inp['own_mask'],inp['own_mask']],axis=0)
        assert np.allclose(p.sum(2),k,rtol=0,atol=1e-12) and not p[:,mask].any()
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        signed=(p*e['reference_membership'][None]).sum(2).reshape(nr,2,nq).transpose(0,2,1)/k
        new=signed.mean(2)
        names=e['baseline_names'].astype(str).tolist()+['rvalues']
        ss=np.concatenate([e['baseline_query_scores'],new[:,None,:]],axis=1)
        assert ss.shape==(nr,len(names),nq)
        mq=ss.mean(0);means=mq.mean(1);boot=mq@e['bootstrap_weights'].T
        if 'baseline_bootstrap' in e:
            assert np.max(np.abs(boot[:-1]-e['baseline_bootstrap']))<1e-14
        fits=list(csv.DictReader((pred/f'{task}.fits.tsv').open(),delimiter='\t'))
        assert len(fits)==nr*2*nq
        counts={name:sum(r['status']==name for r in fits) for name in ('native_success','boundary_degenerate')}
        assert sum(counts.values())==len(fits)
        cc={}
        for m,name in enumerate(names[:-1]):
            point=float(means[m]-means[-1]);draw=boot[m]-boot[-1];ci=np.quantile(draw,[.025,.975]).tolist()
            label=name+'_minus_rvalues'
            cc[label]={'precision_difference':point,'conditional_query_block_95_interval':ci,'expected_reference_members_difference':k*point}
            contrasts.append([task,label,k,point,*ci,k*point])
        report={'task':task,'K':k,'input_reps':nr,'queries':nq,'candidates':nt,'means':dict(zip(names,map(float,means))),
                'contrasts':cc,'native_fit_status_counts':counts,'analysis_identity':'FIXED_POST_HOC_COMPARATOR_ADDITION',
                'interval_scope':'Conditional paired query-block bootstrap, 2000 draws; directions and input repetitions averaged within query before inference.'}
        reports[task]=report
        (out/f'{task}_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        np.savez_compressed(out/f'{task}_scores.npz',method_names=np.array(names),scores=ss,rvalues_direction_scores=signed,bootstrap=boot,query_ids=inp['query_ids'],query_blocks=inp['query_blocks'])
        for name,val in zip(names,means):summaries.append([task,name,k,float(val),k*float(val)])
        taskdraws[task]=(names,mq,e)
    # Match the original equal-state independent RNG-spawned query-block draws.
    ns,m1,e1=taskdraws['D4_Rest'];nt,m2,e2=taskdraws['D4_Stim8hr'];assert ns==nt
    b1=m1@e1['equal_state_bootstrap_weights'].T;b2=m2@e2['equal_state_bootstrap_weights'].T
    state_boot=(b1+b2)/2;state_means=(m1.mean(1)+m2.mean(1))/2
    d4={'means':dict(zip(ns,map(float,state_means))),'contrasts':{},'scope':'Equal-weighted two states of one D4 donor; original independently spawned state query-block resampling, conditional on measurements.'}
    for j,name in enumerate(ns[:-1]):
        point=float(state_means[j]-state_means[-1]);ci=np.quantile(state_boot[j]-state_boot[-1],[.025,.975]).tolist()
        d4['contrasts'][name+'_minus_rvalues']={'precision_difference':point,'conditional_query_block_95_interval':ci,'expected_reference_members_difference':25*point}
        contrasts.append(['D4_equal_state',name+'_minus_rvalues',25,point,*ci,25*point])
    for name,v in zip(ns,state_means):summaries.append(['D4_equal_state',name,25,float(v),25*float(v)])
    ext_tasks=TASKS[:3];ext_names=list(reports[ext_tasks[0]]['means'])
    ext_means={n:float(np.mean([reports[t]['means'][n] for t in ext_tasks])) for n in ext_names}
    ext={'means':ext_means,'contrasts':{n+'_minus_rvalues':ext_means[n]-ext_means['rvalues'] for n in ext_names[:-1]},'pooled_population_CI':None,'scope':'Equal-weighted three source means, sharing one external RNA reference experiment. No independent-source population CI is constructed.'}
    for name,v in ext_means.items():summaries.append(['External_equal_source',name,10,v,10*v])
    for name,delta in ext['contrasts'].items():contrasts.append(['External_equal_source',name,10,delta,'','',10*delta])
    full={'tasks':reports,'D4_equal_state':d4,'External_equal_source':ext,'contrast_sign':'A_minus_rvalues > 0 favors A; < 0 favors rvalues. Workflow contrasts do not isolate geometry.'}
    (out/'summary.json').write_text(json.dumps(full,indent=2,allow_nan=False)+'\n')
    for filename,header,rows in [('method_scores.csv',['task','method','K','precision','expected_reference_members'],summaries),('paired_contrasts.csv',['task','contrast','K','precision_difference','ci_low','ci_high','expected_reference_members_difference'],contrasts)]:
        with (out/filename).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(header);w.writerows(rows)
    return full

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Choose a new output directory')
    expected=json.loads((ROOT/'MANIFEST.json').read_text())
    for name,h in expected['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    a.out.mkdir(parents=True)
    summary=score_all(ROOT/'data/membership',a.out)
    names={'D4 Rest':'D4_Rest','D4 Stim':'D4_Stim8hr','External RNA D3':'External_D3_Rest','External RNA D1':'External_D1_Rest','External RNA D2':'External_D2_Rest'}
    maximum=0.0
    for row in csv.DictReader((ROOT/'expected_primary.csv').open()):
        r=summary['tasks'][names[row['task']]];c=r['contrasts']['PertRank_minus_rvalues']
        pairs=[(r['means']['PertRank'],row['PertRank']),(r['means']['rvalues'],row['rvalues']),(c['precision_difference'],row['difference']),(c['conditional_query_block_95_interval'][0],row['ci95_lower']),(c['conditional_query_block_95_interval'][1],row['ci95_upper']),(c['expected_reference_members_difference'],row['expected_member_difference'])]
        for x,y in pairs:maximum=max(maximum,abs(x-float(y)))
    assert maximum<1e-12,maximum
    (a.out/'REPLAY_CHECK.json').write_text(json.dumps({'status':'PASS','archived_primary_values_checked':30,'max_abs_error':maximum,'native_model_refitted':False},indent=2)+'\n')
    print('Recomputed five task comparisons and both aggregate summaries.')
