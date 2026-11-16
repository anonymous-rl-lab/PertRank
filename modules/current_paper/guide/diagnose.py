#!/usr/bin/env python3
"""Fixed-prediction retained-guide diagnostics; no new rankings or model fits."""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np
from scipy.stats import spearmanr

EXPECTED_PROTOCOL='ac4b8728b29dfec4ce80c6051ce80280ecf5067dcfc84eb7376f99592f3da54b'
METRICS=['I','P','M']
QUANTITIES=['E1','E2','C','D','C_A','C_B','Sref','Sref_minus_C','guide_difference_component','imbalance_component']

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def finite(x):
    if isinstance(x,dict): return {str(k):finite(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [finite(v) for v in x]
    if isinstance(x,np.ndarray): return finite(x.tolist())
    if isinstance(x,np.bool_): return bool(x)
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    return x

def split(ids,guides,lanes,guide_order,source,target,repeat,seed):
    out=np.full(len(ids),-1,dtype=np.int8)
    for g in guide_order:
        sizes=[0,0]
        for lane in sorted(np.unique(lanes[guides==g])):
            inds=np.flatnonzero((guides==g)&(lanes==lane))
            key=lambda i: hashlib.sha256(f'{seed}|{source}|{target}|{g}|{lane}|{repeat}|{int(ids[i])}'.encode()).digest()
            for i in sorted(inds,key=key):
                half=0 if sizes[0]<=sizes[1] else 1
                out[i]=half; sizes[half]+=1
            assert abs(np.sum(out[inds]==0)-np.sum(out[inds]==1))<=1
        assert abs(sizes[0]-sizes[1])<=1
    return out

def metrics(gram,proj,beta):
    pp=proj@proj.T
    return np.stack([gram,pp,gram-beta*pp])
def rho(x,y):
    ok=np.isfinite(x)&np.isfinite(y); n=int(ok.sum())
    if n<3 or len(np.unique(x[ok]))<2 or len(np.unique(y[ok]))<2: return {'n':n,'spearman_rho':None}
    return {'n':n,'spearman_rho':float(spearmanr(x[ok],y[ok]).statistic)}

def summarize(rows,ar):
    eligible=np.array([r['eligible_primary'] for r in rows])
    zero=np.array([r['cross_M_is_zero'] and r['input_guide_count']==2 for r in rows])
    new=np.array([r['newly_zero'] for r in rows])
    groups={'all_primary':eligible,'cross_M_zero':eligible&zero,'cross_M_positive':eligible&~zero,'newly_zero':eligible&new,'already_zero':eligible&zero&~new}
    for b in ['1','2-3','>=4']: groups['input_min_'+b]=eligible&np.array([r['input_coverage_bin']==b for r in rows])
    for b in ['4-9','>=10']: groups['reference_min_'+b]=eligible&np.array([r['reference_coverage_bin']==b for r in rows])
    out={'counts':{'all_source_target_entries':len(rows),'original_two_guide_entries':sum(r['input_guide_count']==2 for r in rows),'primary_eligible':int(eligible.sum()),'cross_M_zero_total_two_guide':int(zero.sum()),'cross_M_zero_primary':int((zero&eligible).sum()),'newly_zero_primary':int((new&eligible).sum()),'single_guide_fallback':sum(r['input_guide_count']==1 for r in rows),'reference_two_guides_with_shared_lane':sum(r['reference_guide_count']==2 and r['shared_lane_count']>0 for r in rows)},'groups':{}}
    qi={v:i for i,v in enumerate(QUANTITIES)}
    for name,mask in groups.items():
        res={'n':int(mask.sum()),'metrics':{}}
        if mask.any():
            for m,mname in enumerate(METRICS):
                a=ar['quantities'][mask,:,m,:]; p=a[:,0,:]
                cf=ar['C_full'][mask,m]; di=ar['D_input_ref'][mask,m]; ci=ar['C_input'][mask,m]
                counts={f'{k}_positive':int((p[:,qi[k]]>0).sum()) for k in ['E1','E2','C','D','Sref']}
                counts.update(C_full_positive=int((cf>0).sum()),D_input_ref_positive=int((di>0).sum()),both_half_C_positive=int(((p[:,qi['C_A']]>0)&(p[:,qi['C_B']]>0)).sum()),D_and_C_positive=int(((p[:,qi['D']]>0)&(p[:,qi['C']]>0)).sum()))
                res['metrics'][mname]={'primary_median':{k:float(np.median(p[:,i])) for i,k in enumerate(QUANTITIES)},'primary_counts':counts,'C_full_median':float(np.median(cf)),'C_input_median':float(np.median(ci)),'D_input_ref_median':float(np.median(di)),'repeat_D_positive_fraction_target_median':float(np.median((a[:,:,qi['D']]>0).mean(axis=1))),'repeat_C_positive_fraction_target_median':float(np.median((a[:,:,qi['C']]>0).mean(axis=1))),'repeat_all20_D_positive_targets':int(np.all(a[:,:,qi['D']]>0,axis=1).sum()),'repeat_all20_C_positive_targets':int(np.all(a[:,:,qi['C']]>0,axis=1).sum()),'correlations':{'C_input_vs_C_full':rho(ci,cf),'C_input_vs_C_primary':rho(ci,p[:,qi['C']]),'D_input_ref_vs_D_primary_shared_reference':rho(di,p[:,qi['D']])}}
        out['groups'][name]=res
    return finite(out)

def run(args):
    assert digest(args.protocol)==EXPECTED_PROTOCOL
    protocol=json.loads(args.protocol.read_text())
    lock=json.loads(args.protocol_lock.read_text())
    assert EXPECTED_PROTOCOL in json.dumps(lock)
    assert protocol['split']['repeats_total']==20 and protocol['split']['primary_repeat']==0
    recovery=json.loads((args.inputs/'RECOVERY.json').read_text())
    assert recovery['protocol_sha256']==EXPECTED_PROTOCOL
    args.out.mkdir(parents=True,exist_ok=True)
    allrows=[]; allarrays=[]; sources={}; provenance={}
    errors={k:0.0 for k in ['D_identity','imbalance_identity','metric_linearity','input_cross_signal_match','old_all_pair_signal_weight_match']}
    for source in protocol['sources']:
        rp=args.inputs/f'{source}_reference_cells.npz'; sp=args.shallow/f'{source}_shallow_cells.npz'; pp=args.predictions/f'{source}.npz'; mp=args.metadata/f'{source}_guide_metadata.npz'
        assert digest(rp)==recovery['sources'][source]['output_sha256']
        ref,sh,pred,md=map(load,[rp,sp,pp,mp]); n=len(sh['target_ids'])
        for x in [ref,pred,md]: assert np.array_equal(sh['target_ids'],x['target_ids'])
        assert np.array_equal(sh['cell_ids'],ref['shallow_cell_ids']) and np.array_equal(sh['cell_ids'],md['cell_ids'])
        assert float(sh['beta'])==float(ref['beta'])==0.9
        G=int(sh['G']); beta=float(sh['beta'])
        nu=np.array([float(sh['trC']),float(sh['lam'].sum()),float(sh['trC'])-beta*float(sh['lam'].sum())])/(G*int(sh['n_control']))
        values=np.full((n,20,3,len(QUANTITIES)),np.nan)
        cf=np.full((n,3),np.nan); di=np.full((n,3),np.nan); ci=np.full((n,3),np.nan); sa=np.full((n,3),np.nan)
        assignments=np.full((20,len(ref['reference_cell_ids'])),-1,dtype=np.int8); rows=[]
        oldidx=int(np.flatnonzero(pred['arm_names']=='original_wM')[0]); newidx=int(np.flatnonzero(pred['arm_names']=='crossguide_wM')[0])
        for j,target in enumerate(sh['target_ids']):
            sl=slice(int(ref['reference_offsets'][j]),int(ref['reference_offsets'][j+1]))
            gids=ref['reference_guide_ids'][sl]; lanes=ref['reference_lane_ids'][sl]; ids=ref['reference_cell_ids'][sl]
            ig,ni=np.unique(md['guide_ids'][j],return_counts=True); rg,nr=np.unique(gids,return_counts=True)
            nrmap=dict(zip(rg,map(int,nr))); matched=np.array([nrmap.get(g,0) for g in ig])
            nmin=int(min(ni)); rmin=int(min(matched)) if len(ig)==2 else None
            eligible=len(ig)==2 and min(matched)>=4
            oldw=float(pred['weights'][oldidx,j]); neww=float(pred['weights'][newidx,j])
            shared=len(set(lanes[gids==rg[0]])&set(lanes[gids==rg[1]])) if len(rg)==2 else 0
            row={'source':source,'target_id':str(target),'target_symbol':str(sh['target_symbols'][j]),'eligible_primary':bool(eligible),'input_guide_count':len(ig),'reference_guide_count':len(rg),'input_min_guide_cells':nmin,'reference_min_matched_guide_cells':rmin,'input_coverage_bin':'1' if nmin==1 else ('2-3' if nmin<4 else '>=4'),'reference_coverage_bin':'NA' if rmin is None else ('0' if rmin==0 else ('1' if rmin==1 else ('2-3' if rmin<4 else ('4-9' if rmin<10 else '>=10')))),'input_guide_counts':json.dumps(dict(zip(ig,map(int,ni))),sort_keys=True),'reference_guide_counts':json.dumps(nrmap,sort_keys=True),'shared_lane_count':shared,'original_M_weight':oldw,'cross_M_weight':neww,'cross_M_is_zero':neww==0.0,'newly_zero':oldw>0 and neww==0.0,'unavailable_reason':'' if eligible else ('single_guide_input_fallback' if len(ig)<2 else 'fewer_than_four_reference_cells_per_matched_guide')}
            hs=metrics(sh['gram'][j],sh['projected'][j],beta)
            for m in range(3): sa[j,m]=(hs[m].sum()-np.trace(hs[m]))/(20*19*G)-nu[m]
            for m,oldname in [(0,'original_w0'),(2,'original_wM')]:
                area=hs[m].sum()/(400*G); w=np.clip(sa[j,m]/max(area,1e-300),0,1); oi=int(np.flatnonzero(pred['arm_names']==oldname)[0])
                errors['old_all_pair_signal_weight_match']=max(errors['old_all_pair_signal_weight_match'],abs(w-float(pred['weights'][oi,j])))
            if len(ig)==2:
                si=[np.flatnonzero(md['guide_ids'][j]==g) for g in ig]
                for m in range(3): ci[j,m]=hs[m][np.ix_(si[0],si[1])].mean()/G-nu[m]
                errors['input_cross_signal_match']=max(errors['input_cross_signal_match'],float(np.max(np.abs(ci[j,[0,2]]-pred['cross_signal'][:,j]))))
            if eligible:
                assert np.array_equal(ig,ref['shallow_guide_order'][j]) and np.array_equal(ni,ref['shallow_guide_n'][j])
                assert set(rg)==set(ig),'Unexpected third reference guide.'
                nn=len(ids); h=np.zeros((nn,nn)); tri=np.triu_indices(nn)
                h[tri]=ref['reference_gram_upper'][int(ref['gram_offsets'][j]):int(ref['gram_offsets'][j+1])]; h=h+h.T-np.diag(np.diag(h))
                hh=metrics(h,ref['reference_projected'][sl],beta)
                fw=np.stack([(gids==g).astype(float)/np.sum(gids==g) for g in ig])
                for m in range(3): cf[j,m]=fw[0]@hh[m]@fw[1]/G-nu[m]
                cross=ref['shallow_guide_reference_dot'][:,sl]; rd=fw[0]-fw[1]
                diffI=(cross[0]-cross[1])@rd/G
                diffP=(ref['shallow_guide_projected'][j,0]-ref['shallow_guide_projected'][j,1])@(rd@ref['reference_projected'][sl])/G
                di[j]=[diffI,diffP,diffI-beta*diffP]
                n1,n2=map(int,ni); l1=n1*(n1-1)/380; l2=n2*(n2-1)/380; lc=2*n1*n2/380
                assert abs(l1+l2+lc-1)<1e-14
                row.update(lambda1=l1,lambda2=l2,lambdac=lc)
                for r in range(20):
                    half=split(ids,gids,lanes,ig,source,str(target),r,protocol['split']['seed']); assignments[r,sl]=half
                    ws=[]
                    for g in ig:
                        for a in [0,1]:
                            mask=(gids==g)&(half==a); assert mask.sum()>=2; ws.append(mask.astype(float)/mask.sum())
                    ww=np.stack(ws)
                    for m in range(3):
                        b=ww@hh[m]@ww.T/G
                        e1=b[0,1]-nu[m]; e2=b[2,3]-nu[m]; c=(b[0,3]+b[1,2])/2-nu[m]
                        d=b[0,1]-b[0,3]-b[2,1]+b[2,3]; ca=b[0,2]-nu[m]; cb=b[1,3]-nu[m]
                        sr=l1*e1+l2*e2+lc*c; gd=(l1+l2)*d/2; im=(l1-l2)*(e1-e2)/2
                        values[j,r,m]=[e1,e2,c,d,ca,cb,sr,sr-c,gd,im]
                        errors['D_identity']=max(errors['D_identity'],abs(d-e1-e2+2*c))
                        errors['imbalance_identity']=max(errors['imbalance_identity'],abs(sr-c-gd-im))
                errors['metric_linearity']=max(errors['metric_linearity'],float(np.max(np.abs(values[j,:,2]-(values[j,:,0]-beta*values[j,:,1])))))
                for m,mname in enumerate(METRICS):
                    for i,q in enumerate(QUANTITIES): row[f'{mname}_{q}_primary']=float(values[j,0,m,i])
                    row[f'{mname}_C_full']=float(cf[j,m]); row[f'{mname}_D_input_ref']=float(di[j,m])
                    for q,ix in [('D',3),('C',2)]:
                        row[f'{mname}_{q}_positive_repeat_fraction']=float(np.mean(values[j,:,m,ix]>0)); row[f'{mname}_{q}_repeat_min']=float(np.min(values[j,:,m,ix])); row[f'{mname}_{q}_repeat_max']=float(np.max(values[j,:,m,ix]))
                    row[f'{mname}_both_half_C_positive_repeat_fraction']=float(np.mean((values[j,:,m,4]>0)&(values[j,:,m,5]>0)))
            for m,mname in enumerate(METRICS): row[f'{mname}_C_input']=float(ci[j,m]); row[f'{mname}_S_all_input']=float(sa[j,m])
            rows.append(row)
        ar={'quantities':values,'C_full':cf,'D_input_ref':di,'C_input':ci,'S_all_input':sa}
        np.savez_compressed(args.out/f'{source}_diagnostic_arrays.npz',**ar,metrics=np.array(METRICS),quantity_names=np.array(QUANTITIES),eligible_primary=np.array([r['eligible_primary'] for r in rows]),target_ids=sh['target_ids'],target_symbols=sh['target_symbols'],reference_half_assignment=assignments,reference_offsets=ref['reference_offsets'],reference_cell_ids=ref['reference_cell_ids'],nu_c=nu)
        sources[source]=summarize(rows,ar); allrows.extend(rows); allarrays.append(ar); provenance[source]={p.name:digest(p) for p in [rp,sp,pp,mp]}
    pooled={k:np.concatenate([x[k] for x in allarrays],axis=0) for k in allarrays[0]}
    fields=list(dict.fromkeys(k for row in allrows for k in row))
    with (args.out/'target_diagnostics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for row in allrows: writer.writerow({k:('' if finite(v) is None else finite(v)) for k,v in row.items()})
    assert len(allrows)==174 and sum(r['eligible_primary'] for r in allrows)==123
    assert max(errors.values())<1e-10,errors
    result={'protocol_sha256':EXPECTED_PROTOCOL,'protocol_lock_sha256':digest(args.protocol_lock),'script_sha256':digest(__file__),'status':'Completed fixed-prediction diagnostic; no new predictor or ranking.','identity':protocol['identity'],'primary_repeat':0,'repeat_count':20,'repeat_interpretation':protocol['split']['interpretation'],'uncertainty':'No inferential CI or p-value. Signs and partition stability are descriptive; shared controls/M remain plug-in.','correlation_interpretation':'Signed Spearman correlations retain negative measurements. D_input_ref versus D_primary shares retained cells; descriptive, not independent replication.','sources':sources,'pooled_descriptive':summarize(allrows,pooled),'algebraic_audit_max_abs_errors':errors,'input_sha256':provenance}
    (args.out/'summary.json').write_text(json.dumps(finite(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS','counts':result['pooled_descriptive']['counts'],'errors':errors},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for arg in ['inputs','shallow','metadata','predictions','protocol','protocol-lock','out']: parser.add_argument('--'+arg,type=Path,required=True)
    run(parser.parse_args())
