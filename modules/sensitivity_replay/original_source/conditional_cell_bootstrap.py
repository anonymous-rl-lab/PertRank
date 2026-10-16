import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from preflight_cell_states import load,A
HERE=Path(__file__).resolve().parent
SEEDS={'D3':202610053,'D1':202610051,'D2':202610052}
ARMS=('raw','w0y','W1')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def probs(E,sq,K=25):
    q,n=E.shape
    V=np.stack([E,-E],axis=1).reshape(2*q,n).copy()
    mask=(np.repeat(np.arange(q),2)[:,None]==sq[None,:])
    V[mask]=-np.inf
    cut=np.partition(V,n-K,axis=1)[:,n-K,None]
    above=V>cut; ties=V==cut
    P=above.astype(float)+ties*((K-above.sum(1))/ties.sum(1))[:,None]
    assert np.max(abs(P.sum(1)-K))<1e-10
    assert not np.any(P[mask])
    return P

def summary(x):
    return {'mean':float(np.mean(x)),'sd':float(np.std(x,ddof=1)),
            'empirical_q025_q975':[float(v) for v in np.quantile(x,[.025,.975])],
            'fraction_gt_zero':float(np.mean(x>0))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--donor',choices=list(SEEDS));args=ap.parse_args()
    frozen=json.loads((HERE/'CONDITIONAL_BOOTSTRAP_FREEZE.json').read_text())
    for f,h in frozen['files'].items():assert sha(HERE/f)==h,(f,'freeze changed')
    B=2 if args.smoke else 200
    result={'status':'development conditional fixed-weight sensitivity, not full-method resampling uncertainty',
            'B':B,'smoke':args.smoke,'freeze':frozen,'donors':{}}
    donorlist=[args.donor] if args.donor else ['D3','D1','D2']
    for d in donorlist:
        t=time.time();s,X,cs,yt,yc=load(d);nc=int(s['n_controls']);nt=len(s['names']);xt=X[nc:].reshape(nt,20,-1)
        del X
        prep={};checks={}; means=np.empty((B,len(cs),3));retains=np.empty_like(means)
        for ii,(rule,c) in enumerate(cs.items()):
            rows=c['rows']; base=(yt[rows]-c['fixed_control_mean'][:,None]);weights=[np.ones(nt),c['w0y'],c['W1']]
            truth=probs(c['T'],c['sq']);orig=[];base_scores={};check={}
            for name,w in zip(ARMS,weights):
                p=probs(base*w[None,:],c['sq']); orig.append(p)
                a=(p*truth).sum(1).reshape(-1,2)/25
                b=A.exp_scores(c['Y']*w[None,:],c)
                check[name]={'max_abs_per_query_score_diff':float(np.max(abs(a-b))),
                             'mean_score':float(a.mean())}
                assert check[name]['max_abs_per_query_score_diff']<=1e-12,(d,rule,name,check[name])
                base_scores[name]=float(a.mean())
            checks[rule]=check
            prep[rule]=(c,rows,weights,truth,orig,base_scores)
        rng=np.random.default_rng(SEEDS[d])
        for b in range(B):
            mult=rng.multinomial(20,np.full(20,.05),size=nt).astype(float)
            ym=np.einsum('tc,tcg->tg',mult,xt,optimize=False).T/20
            for ii,(rule,(c,rows,weights,truth,orig,base_scores)) in enumerate(prep.items()):
                E=ym[rows]-c['fixed_control_mean'][:,None]
                for ai,w in enumerate(weights):
                    pp=probs(E*w[None,:],c['sq'])
                    means[b,ii,ai]=float((pp*truth).sum(1).mean()/25)
                    retains[b,ii,ai]=float((pp*orig[ai]).sum(1).mean()/25)
            if (b+1)%20==0 or b+1==B:print(f'{d}: {b+1}/{B} draws, {time.time()-t:.1f}s',flush=True)
        rr={'checks':checks,'input_sha256':sha(next(iter(cs.values()))['count_data_state']),'seed':SEEDS[d],'n_targets':nt,'n_controls':nc,'cells_per_target':20,'panels':{}}
        for ii,(rule,(_c,_rows,_weights,_truth,_orig,base_scores)) in enumerate(prep.items()):
            rr['panels'][rule]={'original_scores':base_scores,
                'comparison_distributions':{f'W1 - {a}':dict(summary(means[:,ii,2]-means[:,ii,j]),original_difference=base_scores['W1']-base_scores[a]) for j,a in [(0,'raw'),(1,'w0y')]},
                'score_distributions':{a:summary(means[:,ii,j]) for j,a in enumerate(ARMS)},
                'overlap_with_original_list':{a:summary(retains[:,ii,j]) for j,a in enumerate(ARMS)}}
        rr['seconds']=time.time()-t;result['donors'][d]=rr
        prefix='smoke_' if args.smoke else ''
        np.savez_compressed(HERE/f'{prefix}conditional_bootstrap_{d}_draws.npz',mean_scores=means,list_overlaps=retains,panels=np.array(list(prep)),arms=np.array(ARMS))
        (HERE/f'{prefix}conditional_bootstrap_{d}.json').write_text(json.dumps(rr,indent=2))
    prefix='smoke_' if args.smoke else ''
    path=HERE/f'{prefix}conditional_bootstrap_results.json';path.write_text(json.dumps(result,indent=2));print('wrote',path,flush=True)
if __name__=='__main__':main()
