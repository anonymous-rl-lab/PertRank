#!/usr/bin/env python3
"""Offline v20 posthoc S3 accounting, from frozen compact sufficient inputs."""
from pathlib import Path
import argparse, csv, hashlib, json, os, sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen'))
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import compact_support as support

ALIGN=['low','middle','high','unclassified']
TRANS=['both_zero','Euclidean_zero_PertRank_positive','Euclidean_positive_PertRank_zero',
       'both_positive_PertRank_lower','both_positive_equal','both_positive_PertRank_higher']
METRICS=['supported_gain','supported_loss','net_supported','added_slot_mass','removed_slot_mass']
K=25

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_csv(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def serialize(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(str(type(x)))
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=serialize,allow_nan=False)+'\n')

def bootstrap_weights(blocks):
    labels=np.unique(blocks);rng=np.random.default_rng(2026)
    out=np.empty((2000,len(blocks)))
    for draw in range(2000):
        picked=rng.choice(labels,len(labels),replace=True)
        weights=np.sum(blocks[None,:]==picked[:,None],axis=0)
        out[draw]=weights/weights.sum()
    return out

def summarize(x,w):
    x=np.asarray(x);d=w@x
    return {'estimate':float(x.mean()),'lower':float(np.quantile(d,.025)),
            'upper':float(np.quantile(d,.975))}

def transition_ids(we,wp):
    r=np.full(we.shape,-1,int)
    r[(we==0)&(wp==0)]=0
    r[(we==0)&(wp>0)]=1
    r[(we>0)&(wp==0)]=2
    positive=(we>0)&(wp>0)
    r[positive&(wp<we)]=3;r[positive&(wp==we)]=4;r[positive&(wp>we)]=5
    assert np.all(r>=0)
    return r

def fit(y,design,masks):
    z=np.empty((len(masks),3))
    for q,m in enumerate(masks):
        z[q],_,rank,_=np.linalg.lstsq(design[m],y[m,q],rcond=None)
        assert rank==3
    return z

def run(out):
    if out.exists():raise FileExistsError('Refusing to overwrite '+str(out))
    lock=json.loads((ROOT/'PROTOCOL_LOCK.json').read_text())
    assert sha(ROOT/'POSTHOC_PROTOCOL.md')==lock['protocol_sha256']
    for rel,h in lock['files'].items():assert sha(ROOT/rel)==h,rel
    code_lock=json.loads((ROOT/'CODE_LOCK.json').read_text())
    assert sha(Path(__file__))==code_lock['script_sha256']
    assert sha(ROOT/'inputs/planning_responsive_effects.npz')==code_lock['planning_sha256']
    data=support.load_npz(ROOT/'inputs/main_support.npz')
    context=support.load_npz(ROOT/'inputs/main_query_context.npz')
    align=support.load_npz(ROOT/'inputs/alignment_pretruth.npz')
    original=support.load_npz(ROOT/'inputs/original_registered_scores.npz')
    original_report=json.loads((ROOT/'inputs/original_evaluation_report.json').read_text())
    original_audit=json.loads((ROOT/'inputs/original_membership_audit.json').read_text())
    nt,nq=support.validate_support(data);assert nt==561
    names=tuple(data['arm_names'].astype(str));weights=support.reconstruct_weights(data)
    rp=context['responsive_positions'];qids=data['query_gene_ids'][rp];Q=len(rp)
    assert Q==300
    assert np.array_equal(align['design_target_ids'],data['target_ids'])
    assert np.array_equal(align['design_query_ids'],qids)
    ix=np.concatenate([rp,nq+rp])
    n_eligible=nt-(qids[:,None]==data['target_ids'][None,:]).sum(1)
    eligible=align['eligible'];ar=align['rank_A']
    bins=np.full(nt,3,int)
    bins[eligible&(ar<=1/3)]=0
    bins[eligible&(ar>1/3)&(ar<=2/3)]=1
    bins[eligible&(ar>2/3)]=2
    assert eligible.sum()==539 and np.sum(~eligible)==22
    transitions=transition_ids(weights['Euclidean'],weights['PertRank'])
    truth=np.zeros((2*nq,nt),np.float64,order='F')
    for q in range(nq):
        lo,hi=data['truth_offsets'][q:q+2]
        ids=data['truth_ids'][lo:hi].astype(int)
        p=support.probability_row(data['truth_values'][lo:hi])
        truth[q,ids],truth[nq+q,ids]=p[0],p[1]
    assert hashlib.sha256(truth.tobytes()).hexdigest()==original_audit['truth_membership_sha256']
    t=truth[ix].reshape(2,Q,nt)
    observed=np.zeros((nt,Q));neutral=np.zeros((nt,Q))
    # [query, alignment, transition, metric]; all are per-original-list members.
    accounting=np.zeros((Q,len(ALIGN),len(TRANS),len(METRICS)))
    scores=np.empty((len(names),5,nq,2));digest=hashlib.sha256()
    for group in range(5):
        pm={}
        for name in names:
            p=np.zeros_like(truth)
            for q in range(nq):
                row=group*nq+q;lo,hi=data['Y_offsets'][row:row+2]
                ids=data['Y_ids'][lo:hi].astype(int)
                pp=support.probability_row(data['Y_values'][lo:hi]*weights[name][group,ids])
                p[q,ids],p[nq+q,ids]=pp[0],pp[1]
            assert np.max(np.abs(p.sum(1)-K))<1e-12
            digest.update(p.tobytes())
            scores[names.index(name),group]=(p*truth).sum(1).reshape(2,nq).T/K
            if name in ('Euclidean','PertRank'):pm[name]=p[ix].reshape(2,Q,nt)
        delta=pm['PertRank']-pm['Euclidean']
        observed+=((delta*t).sum(axis=0).T)/(5*2*K)
        neutral+=(delta/n_eligible[None,:,None]).sum(axis=0).T/(5*2)
        plus=np.maximum(delta,0);minus=np.maximum(-delta,0)
        for a in range(len(ALIGN)):
            for c in range(len(TRANS)):
                mask=(bins==a)&(transitions[group]==c)
                if not mask.any():continue
                gain=(plus[:,:,mask]*t[:,:,mask]).sum((0,2))/10
                loss=(minus[:,:,mask]*t[:,:,mask]).sum((0,2))/10
                accounting[:,a,c,0]+=gain;accounting[:,a,c,1]+=loss
                accounting[:,a,c,2]+=gain-loss
                accounting[:,a,c,3]+=plus[:,:,mask].sum((0,2))/10
                accounting[:,a,c,4]+=minus[:,:,mask].sum((0,2))/10
    assert digest.hexdigest()==original_audit['prediction_membership_sha256']
    assert np.array_equal(scores,original['main_scores'])
    assert np.array_equal(observed,original['alignment_contributions'])
    co=fit(observed,align['design_design'],align['design_query_target_masks'])
    cn=fit(neutral,align['design_design'],align['design_query_target_masks'])
    assert np.allclose(co,original['alignment_coefficients'],rtol=0,atol=1e-17)
    ip,ie=names.index('PertRank'),names.index('Euclidean')
    query_delta=(scores[ip,:,rp,:]-scores[ie,:,rp,:]).mean((1,2))
    # Advanced indexing puts query first; verify against target decomposition.
    assert query_delta.shape==(Q,)
    assert np.max(np.abs(query_delta-observed.sum(0)))<2e-15
    by_align=accounting.sum(2);by_trans=accounting.sum(1)
    algebra=np.max(np.abs(accounting[:,:,:,2].sum((1,2))-K*query_delta))
    assert algebra<2e-13
    assert np.max(np.abs(neutral.sum(0)))<2e-17
    blocks=context['blocks_responsive'];wb=bootstrap_weights(blocks)
    coeffs={'observed':co[:,1],'reference_neutral':cn[:,1],
            'observed_minus_neutral':co[:,1]-cn[:,1]}
    coeff_summary=[dict(series=k,**summarize(v,wb)) for k,v in coeffs.items()]
    assert abs(coeff_summary[0]['estimate']-original_report['S3_alignment']['mean_rank_A_coefficient'])<1e-17
    assert np.max(np.abs(np.array([coeff_summary[0]['lower'],coeff_summary[0]['upper']])-original_report['S3_alignment']['nominal_95_interval']))<1e-17
    group_rows=[];transition_rows=[];cross_rows=[]
    def measures(arr):
        row={}
        for k,name in enumerate(METRICS):
            st=summarize(arr[:,k],wb)
            row[name]=st['estimate'];row[name+'_lower']=st['lower'];row[name+'_upper']=st['upper']
        row['support_per_added_slot']=row['supported_gain']/row['added_slot_mass'] if row['added_slot_mass'] else None
        row['support_per_removed_slot']=row['supported_loss']/row['removed_slot_mass'] if row['removed_slot_mass'] else None
        return row
    for a,label in enumerate(ALIGN):
        m=bins==a;rr=align['rank_R'][m&eligible]
        group_rows.append(dict(group=label,target_count=int(m.sum()),rank_R_mean=float(rr.mean()) if rr.size else None,
                               rank_R_median=float(np.median(rr)) if rr.size else None,**measures(by_align[:,a])))
        for c,tlabel in enumerate(TRANS):
            cross_rows.append(dict(alignment_group=label,transition=tlabel,
              target_input_count=int(np.sum((bins[None,:]==a)&(transitions==c))),**measures(accounting[:,a,c])))
    for c,label in enumerate(TRANS):
        m=transitions==c
        transition_rows.append(dict(transition=label,target_input_count=int(m.sum()),unique_targets=int(m.any(0).sum()),
          mean_Euclidean_weight=float(weights['Euclidean'][m].mean()) if m.any() else None,
          mean_PertRank_weight=float(weights['PertRank'][m].mean()) if m.any() else None,**measures(by_trans[:,c])))
    all_series={'primary_P_at_25':query_delta,**coeffs}
    all_series.update({a+'_net_members':by_align[:,i,2] for i,a in enumerate(ALIGN)})
    loo=[]
    for label in np.unique(blocks):
        keep=blocks!=label
        loo.append(dict(omitted_block=int(label),omitted_queries=int((~keep).sum()),
                        **{name:float(v[keep].mean()) for name,v in all_series.items()}))
    planning=support.load_npz(ROOT/'inputs/planning_responsive_effects.npz')
    assert np.array_equal(planning['query_gene_ids'],qids)
    rows=planning['planning_effects'];centered=rows-rows.mean(1,keepdims=True)
    centered/=np.maximum(np.linalg.norm(centered,axis=1,keepdims=True),1e-9)
    D=1-centered@centered.T;np.fill_diagonal(D,0)
    tree=linkage(squareform(D,checks=False),'complete')
    derived=fcluster(tree,30,'maxclust')
    assert np.array_equal(derived[:,None]==derived[None,:],blocks[:,None]==blocks[None,:])
    block_rows=[];block_sets={}
    for requested in (20,30,40):
        b=blocks if requested==30 else fcluster(tree,requested,'maxclust')
        block_sets[str(requested)]=b
        w=bootstrap_weights(b)
        counts=np.unique(b,return_counts=True)[1]
        for name,v in all_series.items():
            block_rows.append(dict(requested_blocks=requested,actual_blocks=len(counts),largest_block_queries=int(counts.max()),
                series=name,**summarize(v,w)))
    out.mkdir(parents=True)
    write_csv(out/'s3_alignment_groups.csv',group_rows)
    write_csv(out/'s3_weight_transitions.csv',transition_rows)
    write_csv(out/'s3_alignment_transitions.csv',cross_rows)
    write_csv(out/'s3_coefficients_summary.csv',coeff_summary)
    write_csv(out/'leave_one_block_out.csv',loo)
    write_csv(out/'block_partition_sensitivity.csv',block_rows)
    query_rows=[]
    for q,qid in enumerate(qids):
        row=dict(query_gene_id=str(qid),block=int(blocks[q]),primary_P_at_25=float(query_delta[q]))
        row.update({name:float(v[q]) for name,v in coeffs.items()})
        for a,name in enumerate(ALIGN):
            for m,metric in enumerate(METRICS):row[name+'_'+metric]=float(by_align[q,a,m])
        query_rows.append(row)
    write_csv(out/'query_level_results.csv',query_rows)
    target_rows=[]
    for j,tid in enumerate(data['target_ids']):
        target_rows.append(dict(target_id=str(tid),alignment_group=ALIGN[bins[j]],
          rank_A=float(ar[j]) if eligible[j] else None,rank_R=float(align['rank_R'][j]) if eligible[j] else None,
          net_supported_members=float(observed[j].mean()*K)))
    write_csv(out/'target_groups.csv',target_rows)
    np.savez_compressed(out/'replay_statistics.npz',query_gene_ids=qids,target_ids=data['target_ids'],
        observed_contributions=observed,neutral_contributions=neutral,
        observed_coefficients=co,neutral_coefficients=cn,accounting=accounting,
        alignment_bin=bins,weight_transition=transitions,primary_query_delta=query_delta,
        **{'blocks_'+k:v for k,v in block_sets.items()})
    report={'status':'PASS','analysis_identity':'v20 posthoc; original registered endpoints unchanged',
      'protocol_sha256':sha(ROOT/'POSTHOC_PROTOCOL.md'),'code_sha256':sha(Path(__file__)),
      'source_memberships_exact':True,'source_scores_bitwise_exact':True,'source_S3_contributions_bitwise_exact':True,
      'source_S3_coefficient_max_error':float(np.max(np.abs(co-original['alignment_coefficients']))),
      'counts':{'targets':nt,'queries':Q,'input_groups':5,'directions':2,'eligible_alignment':539,'unclassified':22},
      'primary_P_at_25':summarize(query_delta,wb),
      'net_reference_members':summarize(K*query_delta,wb),
      'alignment_groups':group_rows,'weight_transitions':transition_rows,'coefficients':coeff_summary,
      'decomposition_max_query_error':float(algebra),'neutral_sum_max_query_error':float(np.max(np.abs(neutral.sum(0)))),
      'leave_one_block_out':{name:{'minimum':min(x[name] for x in loo),'maximum':max(x[name] for x in loo)} for name in all_series},
      'block_sensitivity':block_rows,'planning_30_block_reconstructed_exact_partition':True,
      'bootstrap':{'draws':2000,'seed':2026,'unit':'planning-derived query blocks',
       'scope':'nominal conditional resampling ranges over represented query modules; no donor, control, index, or full-experiment uncertainty'},
      'raw_cell_reads':0,'network_requests':0,'new_parameters_or_methods':False}
    dump(out/'RESULT.json',report)
    methods='''# v20 post hoc S3 explanation\n\n## Methods\n\nAll rankings used the original 561-candidate pool, own-target exclusions, exact boundary-tie probabilities, five 20-cell inputs and both directions. The 539 valid frozen alignment ranks were split at 1/3 and 2/3; the 22 invalid indices remained a fourth group, and no subgroup was reranked. For each group, positive and negative changes in membership probability were multiplied by the original reference membership and summed to give supported gains and losses per 25-slot list. Their difference is net contribution; the four groups sum to the original metric gain. Six mutually exclusive exact weight-transition categories partition target-by-input combinations using the same accounting. They describe attribution under joint reranking, not isolated interventions.\n\nThe analytical reference-neutral diagnostic replaces each candidate's reference membership by K/N_q, retaining every prediction, original S3 design, covariate rank and query-specific exclusion. Observed, neutral and paired-difference rank_A slopes are reported. This reference-agnostic comparator detects structural association; it is not a biological null and does not establish causality.\n\nNominal conditional ranges use the same 2,000 paired query-block draws (seed 2026), equal-query weighting and 30 original blocks. They quantify sensitivity across represented query modules, not donor or full-experiment uncertainty. A fixed leave-one-block-out analysis and prespecified-in-this-posthoc-protocol 20/40 planning-derived complete-linkage partitions assess block dependence; the archived 30-block partition was exactly reconstructed before these alternatives were used. None replaces any registered decision.\n\n## Results\n\n'''
    methods+=f"The metric gain reproduced exactly: {report['primary_P_at_25']['estimate']:.10f} P@25, or {report['net_reference_members']['estimate']:.6f} reference members per 25 nominations.\n\n"
    methods+='| Alignment group | Targets | Gained | Lost | Net [conditional range] |\n|---|---:|---:|---:|---:|\n'
    for z in group_rows:methods+=f"| {z['group']} | {z['target_count']} | {z['supported_gain']:.6f} | {z['supported_loss']:.6f} | {z['net_supported']:.6f} [{z['net_supported_lower']:.6f}, {z['net_supported_upper']:.6f}] |\n"
    methods+='\n| Alignment slope | Mean [conditional range] |\n|---|---:|\n'
    for z in coeff_summary:methods+=f"| {z['series']} | {z['estimate']:.10f} [{z['lower']:.10f}, {z['upper']:.10f}] |\n"
    methods+='\nWeight-transition results, exact per-query results and all fixed block sensitivities accompany this report. The observed-minus-neutral sign must be interpreted jointly with actual gains and losses; it is not itself a benefit score. No simulation-derived tercile values are substituted for observed quantities.\n'
    (out/'Methods_Results.md').write_text(methods)
    dump(out/'MANIFEST_SHA256.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({'status':report['status'],'net_reference_members':report['net_reference_members'],
          'alignment_groups':[{k:z[k] for k in ('group','target_count','supported_gain','supported_loss','net_supported','net_supported_lower','net_supported_upper')} for z in group_rows],
          'coefficients':coeff_summary,'weight_transitions':[{k:z[k] for k in ('transition','target_input_count','net_supported','net_supported_lower','net_supported_upper')} for z in transition_rows],
          'leave_one_block_out':report['leave_one_block_out']},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'results')
    run(p.parse_args().output_dir.resolve())
