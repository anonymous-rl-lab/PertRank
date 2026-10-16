#!/usr/bin/env python3
"""Post hoc v20 biology sensitivity. Extract once, replay compact exact tie-rank inputs."""
from pathlib import Path
import argparse,csv,hashlib,json,datetime
import numpy as np

ARMS=['raw','Euclidean','PertRank'];STRATA=['own_gene_unavailable','low_control_expression','consistent_planning_suppression','expressed_suppression_unconfirmed']
PANELS=['responsive','terciles']; K=25

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvwrite(p,rows):
    if not rows:return
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ranks(a):
    # Exact order codes preserve every equality; no float rounding is used.
    return np.unique(a,return_inverse=True)[1].astype(np.int16)
def probability(scores,k=K):
    if len(scores)<k:raise ValueError('inadequate eligible pool')
    cut=np.partition(scores,-k)[-k];out=(scores>cut).astype(float)
    tie=scores==cut;out[tie]=(k-out.sum())/tie.sum();return out

def extract(root,out):
    inp=root/'input';ev=root/'evaluation'
    q=load(inp/'main_query_context.npz');pred=load(inp/'main_nontrade_predictions.npz');ref=load(ev/'main_reference.npz');reg=load(ev/'registered_scores.npz')
    ctrl=load(inp/'control_model.npz');mc=load(inp/'mechanism_controls.npz');al=load(inp/'alignment_pretruth.npz')
    target=q['target_ids'];queries=q['query_gene_ids'];T=len(target);Q=len(queries)
    for d in (pred,ref):
        assert np.array_equal(target,d['target_ids']) and np.array_equal(queries,d['query_gene_ids'])
    assert np.array_equal(mc['mechanism_a'],mc['mechanism_a'])
    assert not np.intersect1d(mc['mechanism_a'],mc['mechanism_b']).size
    for rows in [mc['mechanism_a'],mc['mechanism_b']]:assert not np.intersect1d(rows,ctrl['control_ids']).size
    gene={g:i for i,g in enumerate(ctrl['gene_ids_keep'])}
    c=np.full((T,2),np.nan);t=c.copy();available=np.zeros(T,bool);sources=[]
    for j,gid in enumerate(target):
        path=inp/'main'/f'target_{j:04d}.npz'
        with np.load(path,allow_pickle=False) as z:
            assert str(z['target_id'])==gid
            assert np.array_equal(z['gene_ids_keep'],ctrl['gene_ids_keep'])
            if gid in gene:
                gi=gene[gid];available[j]=True;t[j]=z['planning_half_mean'][:,gi];c[j]=mc['mechanism_control_mean'][:,gi]
            for rows in [mc['mechanism_a'],mc['mechanism_b']]:assert not np.intersect1d(z['planning_cell_ids'],rows).size
        sources.append({'path':str(path),'sha256':sha(path)})
    cmean=np.mean(c,axis=1);tmean=np.mean(t,axis=1);delta=t-c
    stratum=np.full(T,STRATA[0],dtype='<U40')
    low=available&(cmean<.10);expressed=available&~low
    suppression=expressed&(delta[:,0]<0)&(delta[:,1]<0)&(tmean<=.8*cmean)
    stratum[low]=STRATA[1];stratum[expressed]=STRATA[3];stratum[suppression]=STRATA[2]
    codes=np.empty((3,5,Q,T),np.int16)
    for a,arm in enumerate(ARMS):
        arr=pred['prediction_'+arm]
        for r in range(5):
            for qi in range(Q):codes[a,r,qi]=ranks(arr[r,:,qi])
    rcodes=np.stack([ranks(ref['effects'][:,qi]) for qi in range(Q)])
    ct=int(np.flatnonzero(queries=='ENSG00000163599')[0])
    selected=np.array([int(np.flatnonzero(reg['main_arm_names']==a)[0]) for a in ARMS])
    np.savez_compressed(out/'compact_inputs.npz',target_ids=target,target_names=q['target_names'],query_gene_ids=queries,
       rank_codes=codes,reference_rank_codes=rcodes,registered_scores=reg['main_scores'][selected],
       responsive_positions=q['responsive_positions'],tercile_positions=q['tercile_positions'],blocks_responsive=q['blocks_responsive'],blocks_terciles=q['blocks_terciles'],
       control_own_half_means=c,planning_own_half_means=t,available=available,stratum=stratum,
       weights=np.stack([pred['weight_'+a] for a in ARMS]),ctla4_effects=pred['original_effects'][:,:,ct],ctla4_reference_effects=ref['effects'][:,ct],
       alignment_A=al['A'],alignment_R=al['R'],alignment_d_I=al['d_I'])
    for path in [inp/'main_query_context.npz',inp/'main_nontrade_predictions.npz',ev/'main_reference.npz',ev/'registered_scores.npz',inp/'control_model.npz',inp/'mechanism_controls.npz',inp/'alignment_pretruth.npz']:
        sources.append({'path':str(path),'sha256':sha(path)})
    dump(out/'INPUT_PROVENANCE.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':sources,'compact_sha256':sha(out/'compact_inputs.npz'),'rank_codes':'monotone unique float64-score ranks preserve exact order and ties; not approximate expression','scope':'replays fixed full pool/eligibility-subset rankings and all contribution analyses, not refitting weights'})
    print(json.dumps({'extracted_strata':{s:int((stratum==s).sum()) for s in STRATA},'compact_bytes':(out/'compact_inputs.npz').stat().st_size}),flush=True)

def bootstrap(x,blocks):
    labels=np.unique(blocks);B=len(labels);assert B==30
    sums=np.stack([x[blocks==b].sum(axis=0) for b in labels]);sizes=np.array([(blocks==b).sum() for b in labels])
    rng=np.random.default_rng(2026);draw=rng.integers(0,B,size=(2000,B))
    values=sums[draw].sum(axis=1)/sizes[draw].sum(axis=1).reshape((-1,)+(1,)*(x.ndim-1))
    return np.quantile(values,[.025,.975],axis=0)

def analyze(out):
    d=load(out/'compact_inputs.npz');T=len(d['target_ids']);Q=len(d['query_gene_ids']);stratum=d['stratum'];codes=d['rank_codes'];refs=d['reference_rank_codes']
    scores=np.zeros((3,5,Q,2));contrib=np.zeros((3,Q,T));membership=np.zeros((3,Q,T));target_reference=np.zeros((Q,T))
    for qi,query in enumerate(d['query_gene_ids']):
        eligible=d['target_ids']!=query
        for si,sign in enumerate([1,-1]):
            rp=probability(sign*refs[qi,eligible]);target_reference[qi,eligible]+=rp/2
            for r in range(5):
                p=np.stack([probability(sign*codes[a,r,qi,eligible]) for a in range(3)])
                scores[:,r,qi,si]=p@rp/K
                membership[:,qi,eligible]+=p/10
                # Three ordered contrasts: P-E, E-raw, P-raw.
                dd=np.stack([p[2]-p[1],p[1]-p[0],p[2]-p[0]])
                contrib[:,qi,eligible]+=dd*rp/(K*10)
    assert np.max(np.abs(scores-d['registered_scores']))<1e-12
    assert np.max(abs(contrib.sum(axis=-1)-np.stack([scores[2].mean(axis=(0,2))-scores[1].mean(axis=(0,2)),scores[1].mean(axis=(0,2))-scores[0].mean(axis=(0,2)),scores[2].mean(axis=(0,2))-scores[0].mean(axis=(0,2))])))<1e-12
    paneldata={p:(d['responsive_positions'] if p=='responsive' else d['tercile_positions'],d['blocks_responsive'] if p=='responsive' else d['blocks_terciles']) for p in PANELS}
    contrasts=['PertRank_minus_Euclidean','Euclidean_minus_raw','PertRank_minus_raw'];decomprows=[];nomrows=[];fullrows=[];queryrows=[]
    for panel,(idx,blocks) in paneldata.items():
        sq=scores[:,:,idx,:].mean(axis=(1,3));ci=bootstrap(sq.T,blocks)
        for a,arm in enumerate(ARMS):fullrows.append({'panel':panel,'arm':arm,'P25':float(sq[a].mean()),'ci_low':float(ci[0,a]),'ci_high':float(ci[1,a])})
        for contrast,name in enumerate(contrasts):
            perquery=contrib[contrast,idx].sum(axis=-1);fullci=bootstrap(perquery,blocks)
            decomprows.append({'panel':panel,'contrast':name,'stratum':'ALL','targets':T,'delta_P25':float(perquery.mean()),'ci_low':float(fullci[0]),'ci_high':float(fullci[1]),'extra_members':float(K*perquery.mean())})
            for s in STRATA:
                mask=stratum==s;x=contrib[contrast,idx][:,mask].sum(axis=-1);bci=bootstrap(x,blocks)
                decomprows.append({'panel':panel,'contrast':name,'stratum':s,'targets':int(mask.sum()),'delta_P25':float(x.mean()),'ci_low':float(bci[0]),'ci_high':float(bci[1]),'extra_members':float(K*x.mean())})
        for a,arm in enumerate(ARMS):
            for s in STRATA:
                mask=stratum==s;x=membership[a,idx][:,mask].sum(axis=-1)/K;bci=bootstrap(x,blocks)
                nomrows.append({'panel':panel,'arm':arm,'stratum':s,'targets':int(mask.sum()),'nomination_share':float(x.mean()),'ci_low':float(bci[0]),'ci_high':float(bci[1])})
        for i,qi in enumerate(idx):
            row={'panel':panel,'query_gene_id':str(d['query_gene_ids'][qi]),'block':int(blocks[i])}
            for a,arm in enumerate(ARMS):row[arm+'_P25']=float(sq[a,i])
            for s in STRATA:row[s+'_metric_contribution']=float(contrib[0,qi,stratum==s].sum())
            queryrows.append(row)
    c=d['control_own_half_means'];t=d['planning_own_half_means'];cm=c.mean(axis=1);tm=t.mean(axis=1);rows=[]
    for j in range(T):
        row={'target_id':str(d['target_ids'][j]),'target_name':str(d['target_names'][j]),'stratum':str(stratum[j]),'control_mean_log1p':float(cm[j]),'planning_mean_log1p':float(tm[j]),'planning_half_A_contrast':float(t[j,0]-c[j,0]),'planning_half_B_contrast':float(t[j,1]-c[j,1]),'mean_log1p_relative_suppression':float(1-tm[j]/cm[j]) if cm[j]>0 else float('nan'),'alignment_A':float(d['alignment_A'][j]),'alignment_R':float(d['alignment_R'][j]),'signal_energy_dI':float(d['alignment_d_I'][j])}
        for a,arm in enumerate(ARMS):row[arm+'_mean_weight']=float(d['weights'][a,:,j].mean())
        for panel,(idx,_) in paneldata.items():
            row[panel+'_metric_contribution']=float(contrib[0,idx,j].mean())
            row[panel+'_euclidean_contribution']=float(contrib[1,idx,j].mean())
            row[panel+'_workflow_contribution']=float(contrib[2,idx,j].mean())
            row[panel+'_reference_membership']=float(target_reference[idx,j].mean())
            for a,arm in enumerate(ARMS):row[panel+'_'+arm+'_mean_membership']=float(membership[a,idx,j].mean())
        rows.append(row)
    subsetrows=[];subscore={};subsetstatuses={}
    masks={'expressed_all':np.isin(stratum,STRATA[2:]),'consistent_suppression':stratum==STRATA[2]}
    for subset,mask in masks.items():
        n=int(mask.sum());subsetstatuses[subset]={'targets':n,'evaluable':n>=100}
        if n<100:continue
        sc=np.zeros_like(scores)
        for qi,query in enumerate(d['query_gene_ids']):
            eligible=mask&(d['target_ids']!=query)
            for si,sign in enumerate([1,-1]):
                rp=probability(sign*refs[qi,eligible])
                for r in range(5):
                    for a in range(3):sc[a,r,qi,si]=probability(sign*codes[a,r,qi,eligible])@rp/K
        subscore[subset]=sc
        for panel,(idx,blocks) in paneldata.items():
            sq=sc[:,:,idx,:].mean(axis=(1,3));diff=np.stack([sq[2]-sq[1],sq[1]-sq[0],sq[2]-sq[0]])
            for ci,name in enumerate(contrasts):
                lo,hi=bootstrap(diff[ci],blocks)
                subsetrows.append({'subset':subset,'targets':n,'panel':panel,'contrast':name,'delta_P25':float(diff[ci].mean()),'ci_low':float(lo),'ci_high':float(hi),'extra_members':float(K*diff[ci].mean()),'raw_P25':float(sq[0].mean()),'Euclidean_P25':float(sq[1].mean()),'PertRank_P25':float(sq[2].mean())})
    flag_names=['SFTPA1','DEFA6','LYPD3','CLDN5','DLG4'];qi=int(np.flatnonzero(d['query_gene_ids']=='ENSG00000163599')[0]);eligible=d['target_ids']!=d['query_gene_ids'][qi];flags=[]
    for j,name in enumerate(d['target_names']):
        if name not in flag_names:continue
        for r in range(5):
            for sign in [1,-1]:
                indices=np.flatnonzero(eligible);jj=int(np.flatnonzero(indices==j)[0]);rp=probability(sign*refs[qi,eligible]);p=np.stack([probability(sign*codes[a,r,qi,eligible]) for a in range(3)])
                e,ppr=float(p[1,jj]),float(p[2,jj]);status='shared' if e>0 and ppr>0 else 'in' if ppr>0 else 'out' if e>0 else 'unselected'
                row={'target_name':str(name),'target_id':str(d['target_ids'][j]),'stratum':str(stratum[j]),'control_mean_log1p':float(cm[j]),'planning_mean_log1p':float(tm[j]),'planning_half_A_contrast':float(t[j,0]-c[j,0]),'planning_half_B_contrast':float(t[j,1]-c[j,1]),'input_group_zero_based':r,'direction':'up' if sign>0 else 'down','membership_change':status,'original_CTLA4_effect':float(d['ctla4_effects'][r,j]),'reference_CTLA4_effect':float(d['ctla4_reference_effects'][j]),'reference_inclusion':float(rp[jj]),'metric_contribution_P25':float((ppr-e)*rp[jj]/K)}
                for a,arm in enumerate(ARMS):row[arm+'_weight']=float(d['weights'][a,r,j]);row[arm+'_inclusion']=float(p[a,jj])
                flags.append(row)
    csvwrite(out/'target_biology.csv',rows);csvwrite(out/'full_pool_contributions.csv',decomprows);csvwrite(out/'nomination_shares.csv',nomrows);csvwrite(out/'full_pool_scores.csv',fullrows);csvwrite(out/'query_contributions.csv',queryrows);csvwrite(out/'subset_comparisons.csv',subsetrows);csvwrite(out/'CTLA4_flags.csv',flags)
    np.savez_compressed(out/'query_scores.npz',full_pool_scores=scores,**{'subset_'+k:v for k,v in subscore.items()})
    summary={'posthoc':True,'protocol_sha256':sha(out/'POSTHOC_PROTOCOL.md'),'targets':T,'strata_counts':{s:int((stratum==s).sum()) for s in STRATA},'subset_status':subsetstatuses,'original_score_max_abs_difference':float(np.max(abs(scores-d['registered_scores']))),'full_pool_contrasts':[r for r in decomprows if r['stratum']=='ALL'],'full_pool_metric_contributions':[r for r in decomprows if r['contrast']==contrasts[0]],'subset_contrasts':subsetrows,'scope':'Fixed observed pools, controls, panels, planning indices, and references; nominal paired query-block sensitivity intervals; no biological-negative classification, no detection rates or knockdown p-values'}
    dump(out/'RESULTS.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--extract-run',type=Path);ap.add_argument('--out',type=Path,default=Path(__file__).parent);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.extract_run:extract(a.extract_run,a.out)
    analyze(a.out)
