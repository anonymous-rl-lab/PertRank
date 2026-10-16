"""A19-R development reconstruction. Predict/lock first, then score all fixed sources.
Original A19 source lock unavailable; this is never an exact original A19 rerun.
"""
from pathlib import Path
import sys,json,hashlib,time,argparse
import numpy as np
from scipy.sparse import load_npz
ROOT=Path(__file__).resolve().parent;EXT=ROOT.parent;RUN=ROOT.parents[1]
sys.path.insert(0,str(EXT));from reference_sensitivity import tie_prob,BAD_CONTROLS
from validate_selection_axes import check_all
DONORS=['D3_Rest','D1_Rest','D2_Rest'];ARMS=['raw','w0y','W1','deep'];TOP=10

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def savejson(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False))
def bootstrap(a,b,blocks):
    d=np.asarray(a)-np.asarray(b)
    if d.ndim==2:d=d.mean(1)
    rng=np.random.default_rng(2026100502);bl=np.unique(blocks);mem={v:np.flatnonzero(blocks==v) for v in bl}
    draws=np.array([d[np.concatenate([mem[v] for v in rng.choice(bl,len(bl),replace=True)])].mean() for _ in range(2000)])
    return {'point':float(d.mean()),'conditional_query_block_95_interval':np.quantile(draws,[.025,.975]).tolist(),'bootstrap_fraction_nonpositive':float(np.mean(draws<=0)),'draws':2000,'seed':2026100502}
def probabilities(y,own):
    z=np.concatenate([y,-y],axis=0).copy();mask=np.concatenate([own,own],axis=0);z[mask]=-np.inf
    p=tie_prob(z,TOP);assert np.allclose(p.sum(1),TOP);assert not p[mask].any();return p

def predict():
    checks=check_all();locks={};summ={}
    for d in DONORS:
        p=ROOT/d;plan=np.load(p/'cell_plan.npz',allow_pickle=False);ctx=np.load(p/'context.npz',allow_pickle=False);X=load_npz(p/'source_counts.npz')
        assert X.shape[0]==len(plan['cell_ids']);assert np.array_equal(np.asarray(X.sum(1)).ravel(),plan['total_counts']);assert np.all(plan['total_counts']>0)
        smoke=json.loads((p/'smoke_read.json').read_text());full=json.loads((p/'full_read.json').read_text());assert len(smoke['complete_three_target_aggregation_checks'])==3 and full['smoke_rows_reproduced_exact']
        n=len(plan['target_symbols']);g=int(ctx['G']);keep=ctx['keep'];qu=ctx['qu'];U=ctx['U'];c=ctx['cm_keep'];scale=float(ctx['scale']);ep=ctx['external_query_positions'];qr=ctx['q_resp'][ep]
        qpos=np.searchsorted(qu,qr);assert np.array_equal(qu[qpos],qr)
        S=np.empty((n,g));Q=np.empty_like(S);PS=np.empty((n,U.shape[1]));PQ=np.empty_like(PS);SQ=np.empty((n,len(qu)));DQ=np.empty_like(SQ)
        for j in range(n):
            ii=np.flatnonzero((plan['target_index']==j)&(plan['role']==0));dd=np.flatnonzero((plan['target_index']==j)&(plan['role']==1));assert len(ii)==20 and len(dd)==plan['n_deep'][j] and not np.intersect1d(ii,dd).size
            xx=np.log1p(X[ii][:,keep].toarray().astype(float)*(scale/plan['total_counts'][ii])[:,None]);pp=xx@U
            S[j]=xx.sum(0);Q[j]=(xx*xx).sum(0);PS[j]=pp.sum(0);PQ[j]=(pp*pp).sum(0);SQ[j]=xx[:,qu].sum(0)
            DQ[j]=np.log1p(X[dd][:,keep[qu]].toarray().astype(float)*(scale/plan['total_counts'][dd])[:,None]).sum(0)
        mu=S/20;var=(Q-S*S/20)/19;assert var.min()>-1e-10
        m2=np.mean((mu-c)**2,axis=1);vb=var.mean(1);proj_c=c@U;PY=PS/20-proj_c;sv=(PQ-PS*PS/20)/19
        def weight(beta):
            a=m2-beta*(PY[:,:5]**2).sum(1)/g
            nu=vb/20+float(ctx['trC'])/(g*float(ctx['n_control']))-beta*(sv[:,:5]/20+ctx['lam'][:5]/float(ctx['n_control'])).sum(1)/g
            assert a.min()>-1e-10 and nu.min()>-1e-10
            return np.clip(1-nu/np.maximum(a,1e-300),0,1)
        w0=weight(0);w1=weight(.9);explicit=np.clip(1-(vb/20+float(ctx['trC'])/(g*float(ctx['n_control'])))/np.maximum(m2,1e-300),0,1)
        beta0=float(np.max(abs(w0-explicit)));assert beta0<1e-12
        y=(SQ/20-ctx['cm_q']).T[qpos];deep=(DQ/plan['n_deep'][:,None]-ctx['cm_q']).T[qpos]
        query_ids=ctx['genes_keep'][qr];target_ids=plan['target_ids'];own=query_ids[:,None]==target_ids[None,:]
        arms=np.stack([y,y*w0[None,:],y*w1[None,:],deep]);pred=np.stack([probabilities(a,own) for a in arms])
        dest=p/'predictions_locked.npz';np.savez_compressed(dest,query_ids=query_ids,target_ids=target_ids,target_symbols=plan['target_symbols'],query_blocks=ctx['blocks'][ep],own_mask=own,arm_names=np.array(ARMS),effects=arms,top10_probability=pred,w0y=w0,W1=w1)
        np.savez_compressed(p/'sufficient_state.npz',S=S,Q=Q,Psum=PS,Psumsq=PQ,query_sum=SQ,deep_query_sum=DQ,n_deep=plan['n_deep'],m2=m2,vb=vb,proj_control=proj_c,U=U,lam=ctx['lam'],trC=ctx['trC'],G=g,n_control=ctx['n_control'])
        locks[d]={'prediction_sha256':sha(dest),'cell_plan_sha256':sha(p/'cell_plan.npz'),'context_sha256':sha(p/'context.npz'),'counts_sha256':sha(p/'source_counts.npz')}
        summ[d]={'targets':n,'queries':len(qr),'own_exclusions':int(own.sum()),'beta0_vs_explicit_maxabs':beta0,'zero_fraction_w0y':float(np.mean(w0==0)),'zero_fraction_W1':float(np.mean(w1==0)),'n_deep_min':int(plan['n_deep'].min()),'n_deep_max':int(plan['n_deep'].max())}
    record={'identity':'A19-R new fixed cell seed; development reconstruction, not original A19 replication','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol_sha256':sha(ROOT/'PROTOCOL_A19R.md'),'evaluation_script_sha256':sha(__file__),'gene_axis_checks':checks,'locks':locks,'diagnostics':summ}
    savejson(ROOT/'PREDICTION_LOCK.json',record);print(json.dumps(record,indent=2),flush=True)

def qc_reference():
    import pandas as pd
    z=np.load(EXT/'results/reference_query_samples.npz',allow_pickle=False);meta=pd.read_csv(EXT/'results/sample_metadata_reconstructed.csv');x=z['sample_log2cpm'];targets=z['target_symbols'];ki=z['target_ko_sample_indices'];tb=z['target_batch']
    ref=np.load(EXT/'results/query_reference.npz',allow_pickle=False)
    assert np.array_equal(z['query_ids'],ref['query_ids']) and np.array_equal(targets,ref['target_symbols'])
    controls={}
    for b,ds in [(1,[1,3,4]),(2,[1,2,3])]:
        for donor in ds:
            ii=meta.index[(meta.batch==b)&(meta.donor==donor)&meta.is_control&~meta['sample'].isin(BAD_CONTROLS)].to_numpy();assert len(ii)>0;controls[b,donor]=x[:,ii].mean(1)
    e=np.empty((len(x),len(targets)))
    for j in range(len(targets)):
        ds=[1,3,4] if tb[j]==1 else [1,2,3];e[:,j]=np.mean([x[:,ki[j,r]]-controls[int(tb[j]),donor] for r,donor in enumerate(ds)],axis=0)
    return e

def score():
    lock=json.loads((ROOT/'PREDICTION_LOCK.json').read_text());assert lock['evaluation_script_sha256']==sha(__file__)
    ref=np.load(EXT/'results/query_reference.npz',allow_pickle=False);qmap={x:i for i,x in enumerate(ref['query_ids'])};tmap={x:i for i,x in enumerate(ref['target_symbols'])};pack={};out={'identity':'A19-R development reconstruction; original A19 exact replay still blocked','units':'expected externally supported members per 10-name list; divide by 10 for P@10','interval_scope':'query-block intervals condition on source cells, controls, candidate pool, queries and fixed external reference; external draw ranges condition on source predictions','sources':{}}
    scores={}
    for d in DONORS:
        p=ROOT/d/'predictions_locked.npz';assert sha(p)==lock['locks'][d]['prediction_sha256'];z=np.load(p,allow_pickle=False)
        qi=np.array([qmap[g] for g in z['query_ids']]);ti=np.array([tmap[t] for t in z['target_symbols']]);e=ref['mean_effect'][np.ix_(qi,ti)];tr=probabilities(e,z['own_mask']);pred=z['top10_probability'];nq=len(qi)
        hit=np.einsum('arc,rc->ar',pred,tr).reshape(4,2,nq).transpose(0,2,1);assert ((hit>=0)&(hit<=10+1e-12)).all()
        # Independent scorer comparison to the project's original exact-tie scorer.
        sys.path.insert(0,str(RUN/'late_recovery/run/code/repro_v10'));from pr10 import score_rows
        selfq=np.full(len(ti),-1,int)
        for jj in range(len(ti)):
            ix=np.flatnonzero(z['own_mask'][:,jj]);assert len(ix)<=1
            if len(ix):selfq[jj]=ix[0]
        maxerr=0.
        for a in range(4):maxerr=max(maxerr,float(np.max(abs(10*score_rows(z['effects'][a],e,selfq,K=10)['exp']-hit[a]))))
        assert maxerr<1e-12
        floor=(100/(len(ti)-z['own_mask'].sum(1)))[:,None]*np.ones((1,2));b=z['query_blocks'];r={'targets':len(ti),'queries':nq,'own_exclusions':int(z['own_mask'].sum()),'random_expected_members':float(floor.mean()),'means':{a:float(hit[i].mean()) for i,a in enumerate(ARMS)},'W1_minus_raw':bootstrap(hit[2],hit[0],b),'W1_minus_w0y':bootstrap(hit[2],hit[1],b),'deep_minus_random':bootstrap(hit[3],floor,b),'score_rows_crosscheck_maxabs':maxerr}
        # Accounting identity uses exact tie-expectation probabilities, no hard lists.
        add=np.maximum(pred[2]-pred[1],0);rem=np.maximum(pred[1]-pred[2],0);assert np.allclose(add.sum(1),rem.sum(1));turn=float(add.sum());r['turnover_vs_w0y']={'mean_added_positions':float(add.sum(1).mean()),'added_support_rate':float((add*tr).sum()/turn) if turn else None,'removed_support_rate':float((rem*tr).sum()/turn) if turn else None,'exact_delta':float(((add-rem)*tr).sum(1).mean())};assert abs(r['turnover_vs_w0y']['exact_delta']-r['W1_minus_w0y']['point'])<1e-12
        out['sources'][d]=r;scores[d]=hit;pack[d]=(z,qi,ti)
    out['equal_source_mean_descriptive']={a:float(np.mean([out['sources'][d]['means'][a] for d in DONORS])) for a in ARMS};out['original_A19_comparison']='Not an exact replay: original cell seed/source locks missing; these predictions use the new A19-R hash seed.'
    savejson(ROOT/'point_results.json',out)
    result={'identity':'A19-R external-reference uncertainty sensitivity, not population confidence intervals','shared_reference_per_draw':True,'source_cells_queries_candidates_controls_fixed':True,'draws':500,'schemes':{}}
    for scheme in ['shared_labels','batch_separate']:
        draws=np.load(EXT/f'results/reference_draws_{scheme}_float64.npz',allow_pickle=False);ee=draws['effect_draws'];assert ee.dtype==np.float64 and len(ee)==500;assert np.array_equal(draws['query_ids'],ref['query_ids']) and np.array_equal(draws['target_symbols'],ref['target_symbols'])
        values=np.empty((500,3,4))
        for it in range(500):
            # A single external realization is shared by all three source donors.
            for di,d in enumerate(DONORS):
                z,qi,ti=pack[d];tr=probabilities(ee[it][np.ix_(qi,ti)],z['own_mask']);values[it,di]=np.einsum('arc,rc->ar',z['top10_probability'],tr).mean(1)
        sr={}
        for di,d in enumerate(DONORS+['equal_source_mean']):
            v=values[:,di] if di<3 else values.mean(1);sr[d]={}
            for label,base in [('W1_minus_raw',0),('W1_minus_w0y',1)]:
                dd=v[:,2]-v[:,base];sr[d][label]={'median':float(np.median(dd)),'empirical_2p5_97p5':np.quantile(dd,[.025,.975]).tolist(),'fraction_positive':float(np.mean(dd>0))}
        np.savez_compressed(ROOT/f'reference_propagation_{scheme}.npz',source_names=np.array(DONORS),arm_names=np.array(ARMS),expected_members=values)
        result['schemes'][scheme]=sr
    eqc=qc_reference();qc={}
    for d,(z,qi,ti) in pack.items():
        tr=probabilities(eqc[np.ix_(qi,ti)],z['own_mask']);v=np.einsum('arc,rc->ar',z['top10_probability'],tr).mean(1);qc[d]={'means':dict(zip(ARMS,map(float,v))),'W1_minus_raw':float(v[2]-v[0]),'W1_minus_w0y':float(v[2]-v[1])}
    result['author_bad_controls_excluded_point_sensitivity']=qc;result['reference_float64_files_sha256']={s:sha(EXT/f'results/reference_draws_{s}_float64.npz') for s in result['schemes']};savejson(ROOT/'reference_propagation_results.json',result)
    print(json.dumps(out,indent=2),flush=True);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['predict','score'],required=True);a=p.parse_args();predict() if a.mode=='predict' else score()
