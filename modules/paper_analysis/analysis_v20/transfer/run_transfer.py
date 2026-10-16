#!/usr/bin/env python3
"""Frozen v20 panel-transfer analysis; compact data permit exact replay."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ARMS=('raw','Euclidean','PertRank','PR_M')

def probabilities(v,k=25):
    cut=np.partition(v,len(v)-k)[len(v)-k]
    above=v>cut; tied=v==cut
    p=above.astype(float);p[tied]=(k-above.sum())/tied.sum()
    return p

def score(Y,truth,weights,target_ids,queries):
    # arm x group x query x direction
    out=np.empty((len(ARMS),Y.shape[0],len(queries),2))
    for q,g in enumerate(queries):
        take=target_ids!=g
        for d,sign in enumerate([1,-1]):
            ref=probabilities(sign*truth[take,q])
            for r in range(Y.shape[0]):
                for a in range(len(ARMS)):
                    pred=probabilities(sign*Y[r,take,q]*weights[a,r,take])
                    out[a,r,q,d]=pred@ref/25
    return out

def export(root):
    stim=root/'stim_run_20261011';c=np.load(stim/'input/control_model.npz')
    ctx=np.load(stim/'input/main_query_context.npz')
    src=np.load(HERE/'source_panel.npz');sid=src['query_ids'];axis=c['gene_ids_keep']
    keep=np.isin(sid,axis);ids=sid[keep];blocks=src['blocks'][keep]
    union=np.unique(np.r_[ids,ctx['query_gene_ids']]);index={g:i for i,g in enumerate(axis)}
    pos=np.array([index[g] for g in union]);targets=ctx['target_ids']
    Y=np.empty((5,len(targets),len(union)));T=np.empty((len(targets),len(union)))
    W=np.empty((4,5,len(targets)));ec=np.load(stim/'evaluation/evaluation_control.npz')['mean']
    for i,t in enumerate(targets):
        with np.load(stim/f'input/main/target_{i:04d}.npz') as x:
            assert str(x['target_id'])==t
            Y[:,i]=x['log_y'][:,pos]
            for a,arm in enumerate(ARMS): W[a,:,i]=x['weight_'+arm]
        with np.load(stim/f'evaluation/main/target_{i:04d}.npz') as x:
            T[i]=x['mean'][pos]-ec[pos]
    qi={g:i for i,g in enumerate(union)}
    baseline=np.array([qi[g] for g in ctx['query_gene_ids']])
    actual=score(Y[:,:,baseline],T[:,baseline],W,targets,ctx['query_gene_ids'])
    z=np.load(stim/'evaluation/registered_scores.npz')
    saved_order=[list(z['main_arm_names']).index(a) for a in ARMS]
    err=float(np.max(abs(actual-z['main_scores'][saved_order])))
    assert err<1e-14,err
    ix=np.array([qi[g] for g in ids])
    np.savez_compressed(HERE/'transfer_inputs.npz',effects=Y[:,:,ix],truth=T[:,ix],weights=W,target_ids=targets,query_ids=ids,source_blocks=blocks)
    provenance={'source_panel_count':len(sid),'eligible_query_count':len(ids),'missing_query_ids':sid[~keep].tolist(),'target_count':len(targets),'n_groups':5,'target_planning_cells_used_for_panel':0,'retained_source_blocks':int(len(set(blocks))),'original_Stim_score_max_abs_error':err,'original_Stim_score_values_checked':int(actual.size),'transfer_input_sha256':hashlib.sha256((HERE/'transfer_inputs.npz').read_bytes()).hexdigest()}
    (HERE/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')

def analyze():
    z=np.load(HERE/'transfer_inputs.npz')
    scores=score(z['effects'],z['truth'],z['weights'],z['target_ids'],z['query_ids'])
    np.savez_compressed(HERE/'transfer_scores.npz',scores=scores,arm_names=np.array(ARMS),query_ids=z['query_ids'],source_blocks=z['source_blocks'])
    qs=scores.mean(axis=(1,3));blocks=z['source_blocks'];ub=np.unique(blocks)
    sz=np.array([(blocks==b).sum() for b in ub]);rng=np.random.default_rng(20261012)
    draw=rng.integers(len(ub),size=(10000,len(ub)))
    estimates={a:float(qs[i].mean()) for i,a in enumerate(ARMS)}
    contrasts={};block_rows=[]
    for b in ('Euclidean','raw','PR_M'):
        d=qs[2]-qs[ARMS.index(b)]
        sums=np.array([d[blocks==c].sum() for c in ub])
        boots=sums[draw].sum(1)/sz[draw].sum(1)
        ci=np.quantile(boots,[.025,.975]);loo=[]
        for c in ub:
            val=float(d[blocks!=c].mean());loo.append(val)
            block_rows.append({'contrast':'PertRank-'+b,'omitted_source_block':int(c),'omitted_queries':int((blocks==c).sum()),'delta_P25':val})
        contrasts['PertRank-'+b]={'delta_P25':float(d.mean()),'conditional_95pct_interval':ci.tolist(),'extra_members_per25':float(25*d.mean()),'members_interval':(25*ci).tolist(),'leave_one_block_out_range':[min(loo),max(loo)],'all_LOBO_positive':bool(min(loo)>0)}
    report=dict(status='post hoc fixed-source-panel application',means_P25=estimates,contrasts=contrasts,**json.loads((HERE/'PROVENANCE.json').read_text()))
    (HERE/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    with (HERE/'query_scores.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['query_id','source_block',*ARMS,'PertRank_minus_Euclidean'])
        for i,q in enumerate(z['query_ids']): w.writerow([q,int(blocks[i]),*qs[:,i],qs[2,i]-qs[1,i]])
    with (HERE/'block_influence.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=block_rows[0].keys());w.writeheader();w.writerows(block_rows)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export-from',type=Path)
    args=p.parse_args()
    if args.export_from: export(args.export_from)
    analyze()
