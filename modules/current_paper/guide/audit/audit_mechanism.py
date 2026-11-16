#!/usr/bin/env python3
"""Independent Gram-block reconstruction; never imports production functions."""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np
from scipy.stats import spearmanr

FIELDS=['E1','E2','C','D','C_full','C_half_A','C_half_B','D_input_ref','Sref','Sref_minus_C','D_component','imbalance_component']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def independently_split(ids,labels,lanes,source,target,guide,repeat):
    groups=[[],[]]
    for lane in sorted(set(lanes[labels==guide])):
        rows=np.flatnonzero((labels==guide)&(lanes==lane)).tolist()
        rows.sort(key=lambda row:hashlib.sha256(f'2026111404|{source}|{target}|{guide}|{lane}|{repeat}|{int(ids[row])}'.encode()).hexdigest())
        lane_assignment=[[],[]]
        for row in rows:
            half=int(len(groups[0])>len(groups[1]))
            groups[half].append(row);lane_assignment[half].append(row)
        assert abs(len(lane_assignment[0])-len(lane_assignment[1]))<=1
    assert abs(len(groups[0])-len(groups[1]))<=1
    assert not set(groups[0])&set(groups[1])
    assert set(groups[0]+groups[1])==set(np.flatnonzero(labels==guide))
    return [np.array(g,dtype=int) for g in groups]

def reconstruct(args):
    result=[]
    report={'status':'INDEPENDENT_RECONSTRUCTION_COMPLETE','sources':{},'max_identity_error':0.0,
            'max_imbalance_identity_error':0.0,'max_metric_identity_error':0.0,'audit_script_sha256':digest(__file__)}
    for source in ['D3','D1','D2']:
        x=load(args.inputs/f'{source}_reference_cells.npz')
        p=load(args.predictions/f'{source}.npz')
        s=load(args.shallow/f'{source}_shallow_cells.npz')
        md=load(args.metadata/f'{source}_guide_metadata.npz')
        nt=len(x['target_ids'])
        ss={'targets':nt,'two_input_guides':0,'primary_eligible':0,'old_cross_weight_zero_two_guide':0,
            'primary_zero':0,'primary_positive':0,'primary_zero_Dpositive':0,'primary_zero_Cpositive':0,
            'primary_zero_both_half_Cpositive':0,'primary_positive_Dpositive':0,
            'primary_positive_Cpositive':0,'primary_positive_both_half_Cpositive':0}
        assert np.array_equal(x['target_ids'],p['target_ids'])
        for k in ['target_ids','target_symbols']:
            assert np.array_equal(x[k],s[k]) and np.array_equal(x[k],md[k])
        assert np.array_equal(x['shallow_cell_ids'],s['cell_ids'])
        assert np.array_equal(x['shallow_cell_ids'],md['cell_ids'])
        assert np.array_equal(p['weights'][0],s['expected_w0'])
        assert np.array_equal(p['weights'][1],s['expected_wM'])
        oldm=p['weights'][1];crossm=p['weights'][3]
        for j,target in enumerate(x['target_ids']):
            n=int(x['n_reference'][j]);lo,hi=x['reference_offsets'][j:j+2];gl,gh=x['gram_offsets'][j:j+2]
            rr,cc=np.triu_indices(n);H=np.empty((n,n));H[rr,cc]=x['reference_gram_upper'][gl:gh];H[cc,rr]=H[rr,cc]
            P=x['reference_projected'][lo:hi];labels=x['reference_guide_ids'][lo:hi]
            lanes=x['reference_lane_ids'][lo:hi];ids=x['reference_cell_ids'][lo:hi]
            assert len(set(ids.tolist()))==n
            assert not np.intersect1d(ids,x['shallow_cell_ids'][j]).size
            archived=(md['all_archived_target_index']==j)&(md['all_archived_role']==1)
            assert np.array_equal(ids,md['all_archived_cell_ids'][archived])
            assert np.array_equal(labels,md['all_archived_guide_ids'][archived])
            assert np.array_equal(lanes,md['all_archived_lane_ids'][archived])
            guides=x['shallow_guide_order'][j]
            two=bool(np.all(x['shallow_guide_n'][j]>0))
            ng=[int(np.sum(labels==g)) for g in guides] if two else [0,0]
            primary=two and min(ng)>=4;has_split=two and min(ng)>=2;zero=two and crossm[j]==0
            ss['two_input_guides']+=int(two);ss['primary_eligible']+=int(primary)
            ss['old_cross_weight_zero_two_guide']+=int(zero);ss['primary_zero']+=int(primary and zero)
            ss['primary_positive']+=int(primary and not zero)
            G=float(x['G']);vcI=float(x['trC'])/(float(x['n_control'])*G)
            vcP=float(np.sum(x['lam']))/(float(x['n_control'])*G)
            gfull=[np.flatnonzero(labels==g) for g in guides]
            ns=x['shallow_guide_n'][j].astype(float)
            lam=[ns[0]*(ns[0]-1)/380,ns[1]*(ns[1]-1)/380,2*ns[0]*ns[1]/380]
            if two:assert abs(sum(lam)-1)<1e-15
            for repeat in range(20):
                if has_split:
                    g1a,g1b=independently_split(ids,labels,lanes,source,str(target),str(guides[0]),repeat)
                    g2a,g2b=independently_split(ids,labels,lanes,source,str(target),str(guides[1]),repeat)
                    grouping=[g1a,g1b,g2a,g2b]
                metrics={}
                for metric in ['I','P','M']:
                    if metric=='I':kernel,vc=H,vcI
                    elif metric=='P':kernel,vc=P@P.T,vcP
                    else:kernel,vc=H-.9*(P@P.T),vcI-.9*vcP
                    val={k:float('nan') for k in FIELDS}
                    if two and min(ng)>=1:
                        a,b=gfull;val['C_full']=float(kernel[np.ix_(a,b)].mean()/G-vc)
                        rawcross=x['shallow_guide_reference_dot'][:,lo:hi];sp=x['shallow_guide_projected'][j]
                        pcross=sp@P.T;cross=rawcross if metric=='I' else pcross if metric=='P' else rawcross-.9*pcross
                        sd=cross[0]-cross[1];val['D_input_ref']=float((sd[a].mean()-sd[b].mean())/G)
                    if has_split:
                        q=np.array([[float(kernel[np.ix_(a,b)].mean())/G for b in grouping] for a in grouping])
                        E1,E2=q[0,1]-vc,q[2,3]-vc;C=(q[0,3]+q[1,2])/2-vc
                        da=np.zeros(n);db=np.zeros(n)
                        da[g1a]=1/len(g1a);da[g2a]=-1/len(g2a);db[g1b]=1/len(g1b);db[g2b]=-1/len(g2b)
                        D=float(da@kernel@db/G)
                        report['max_identity_error']=max(report['max_identity_error'],abs(D-E1-E2+2*C))
                        Sref=lam[0]*E1+lam[1]*E2+lam[2]*C;dc=(lam[0]+lam[1])*D/2;ic=(lam[0]-lam[1])*(E1-E2)/2
                        report['max_imbalance_identity_error']=max(report['max_imbalance_identity_error'],abs(Sref-C-dc-ic))
                        val.update(E1=E1,E2=E2,C=C,D=D,C_half_A=q[0,2]-vc,C_half_B=q[1,3]-vc,
                                   Sref=Sref,Sref_minus_C=Sref-C,D_component=dc,imbalance_component=ic)
                    metrics[metric]=val
                    result.append({'source':source,'target_id':str(target),'target_symbol':str(x['target_symbols'][j]),
                                   'repeat':repeat,'metric':metric,'two_input_guides':two,
                                   'reference_guide_1_n':ng[0],'reference_guide_2_n':ng[1],
                                   'primary_eligible':primary,'old_cross_weight_zero':zero,
                                   'newly_zero':bool(zero and oldm[j]>0),
                                   'input_min_guide_cells':int(ns.min()) if two else 20,
                                   'reference_min_matched_guide_cells':min(ng) if two else None,
                                   'reference_guide_count':len(np.unique(labels)),
                                   'shared_lane_count':len(set(lanes[labels==np.unique(labels)[0]])&set(lanes[labels==np.unique(labels)[1]])) if len(np.unique(labels))==2 else 0,
                                   **val})
                for f in FIELDS:
                    if np.isfinite(metrics['M'][f]):
                        report['max_metric_identity_error']=max(report['max_metric_identity_error'],abs(metrics['M'][f]-metrics['I'][f]+.9*metrics['P'][f]))
                if repeat==0 and primary:
                    group='primary_zero' if zero else 'primary_positive'
                    ss[group+'_Dpositive']+=int(metrics['M']['D']>0);ss[group+'_Cpositive']+=int(metrics['M']['C']>0)
                    ss[group+'_both_half_Cpositive']+=int(metrics['M']['C_half_A']>0 and metrics['M']['C_half_B']>0)
        report['sources'][source]=ss
    assert len(result)==174*20*3
    return result,report

def compare_production(args,rows,report):
    lookup={(r['source'],r['target_id'],r['repeat'],r['metric']):r for r in rows}
    qmap={'C_A':'C_half_A','C_B':'C_half_B','guide_difference_component':'D_component'}
    maxima={key:0.0 for key in ['all_primary_quantities','C_full','D_input_ref','C_input','S_all_input','summary_values','target_csv_values']}
    counts={'primary_target_metric_repeats':0,'reference_assignment_values':0,'summary_numeric_values':0,'csv_numeric_values':0}
    summary=json.loads((args.results/'summary.json').read_text())
    assert summary['protocol_sha256']==digest(args.protocol)
    candidates=[args.production_script] if args.production_script else [args.results.parent/'diagnose.py']
    production_script=next((path for path in candidates if path.is_file() and digest(path)==summary['script_sha256']),None)
    assert production_script is not None,'No matching production script alongside results or in package root'
    def eq(a,b,key,tol=1e-10):
        if a is None or b is None:
            assert a is None and b is None,(a,b,key)
            return
        a=np.asarray(a);b=np.asarray(b)
        assert a.shape==b.shape,(key,a.shape,b.shape)
        assert np.array_equal(np.isnan(a),np.isnan(b)),(key,'missingness')
        ok=np.isfinite(a)
        if np.any(ok):
            err=float(np.max(np.abs(a[ok]-b[ok])));maxima[key]=max(maxima[key],err)
            assert err<=tol,(key,err,tol)
    source_arrays={}
    base_records=[]
    for source in ['D3','D1','D2']:
        x=load(args.inputs/f'{source}_reference_cells.npz');s=load(args.shallow/f'{source}_shallow_cells.npz')
        md=load(args.metadata/f'{source}_guide_metadata.npz');a=load(args.results/f'{source}_diagnostic_arrays.npz')
        assert np.array_equal(a['target_ids'],x['target_ids'])
        assert np.array_equal(a['reference_cell_ids'],x['reference_cell_ids'])
        assert np.array_equal(a['reference_offsets'],x['reference_offsets'])
        vals=np.full_like(a['quantities'],np.nan);cf=np.full_like(a['C_full'],np.nan);di=np.full_like(a['D_input_ref'],np.nan)
        ci=np.full_like(a['C_input'],np.nan);sa=np.full_like(a['S_all_input'],np.nan)
        nuI=float(s['trC'])/(float(s['G'])*int(s['n_control']));nuP=float(np.sum(s['lam']))/(float(s['G'])*int(s['n_control']))
        nu=np.array([nuI,nuP,nuI-.9*nuP]);eq(a['nu_c'],nu,'summary_values')
        for j,target in enumerate(x['target_ids']):
            b=lookup[(source,str(target),0,'M')];base_records.append(b)
            primary=b['primary_eligible'];assert bool(a['eligible_primary'][j])==primary
            lo,hi=x['reference_offsets'][j:j+2]
            guides=x['shallow_guide_order'][j]
            if primary:
                for repeat in range(20):
                    assignments=np.full(int(hi-lo),-1,dtype=np.int8)
                    for guide in guides:
                        ga,gb=independently_split(x['reference_cell_ids'][lo:hi],x['reference_guide_ids'][lo:hi],x['reference_lane_ids'][lo:hi],source,str(target),str(guide),repeat)
                        assignments[ga]=0;assignments[gb]=1
                    assert np.array_equal(assignments,a['reference_half_assignment'][repeat,lo:hi])
                    counts['reference_assignment_values']+=len(assignments)
                    for m,metric in enumerate(a['metrics']):
                        r=lookup[(source,str(target),repeat,str(metric))]
                        vals[j,repeat,m]=[r[qmap.get(str(q),str(q))] for q in a['quantity_names']]
                        counts['primary_target_metric_repeats']+=1
                for m,metric in enumerate(a['metrics']):
                    r=lookup[(source,str(target),0,str(metric))];cf[j,m]=r['C_full'];di[j,m]=r['D_input_ref']
            else:
                assert np.all(a['reference_half_assignment'][:,lo:hi]==-1)
            H=s['gram'][j];P=s['projected'][j];HP=P@P.T
            gids=np.unique(md['guide_ids'][j])
            kernels=[H,HP,H-.9*HP]
            for m,K in enumerate(kernels):
                # Explicit off-diagonal upper triangle sum, independent of production sum-minus-trace.
                sa[j,m]=2*np.triu(K,k=1).sum()/(20*19*float(s['G']))-nu[m]
                if len(gids)==2:
                    g1=np.flatnonzero(md['guide_ids'][j]==gids[0]);g2=np.flatnonzero(md['guide_ids'][j]==gids[1])
                    ci[j,m]=K[np.ix_(g1,g2)].mean()/float(s['G'])-nu[m]
            for m,metric in enumerate(a['metrics']):
                for repeat in range(20):
                    lookup[(source,str(target),repeat,str(metric))]['C_input']=ci[j,m]
        eq(vals,a['quantities'],'all_primary_quantities');eq(cf,a['C_full'],'C_full');eq(di,a['D_input_ref'],'D_input_ref')
        eq(ci,a['C_input'],'C_input');eq(sa,a['S_all_input'],'S_all_input')
        source_arrays[source]={'quantities':vals,'C_full':cf,'D_input_ref':di,'C_input':ci,'S_all_input':sa,'quantity_names':a['quantity_names']}
    # All main CSV metadata and every numeric diagnostic value are checked.
    with (args.results/'target_diagnostics.csv').open() as f: csvrows=list(csv.DictReader(f))
    assert len(csvrows)==174
    for pr in csvrows:
        b=lookup[(pr['source'],pr['target_id'],0,'M')]
        assert (pr['eligible_primary']=='True')==b['primary_eligible']
        assert int(pr['input_guide_count'])==(2 if b['two_input_guides'] else 1)
        assert int(pr['shared_lane_count'])==b['shared_lane_count']
        assert (pr['cross_M_is_zero']=='True')==b['old_cross_weight_zero']
        for metric in ['I','P','M']:
            r=lookup[(pr['source'],pr['target_id'],0,metric)]
            for key in ['E1','E2','C','D','C_A','C_B','Sref','Sref_minus_C','guide_difference_component','imbalance_component']:
                column=metric+'_'+key+'_primary'
                if b['primary_eligible']:
                    eq(float(pr[column]),r[qmap.get(key,key)],'target_csv_values');counts['csv_numeric_values']+=1
                else:assert not pr[column]
            for key in ['C_full','D_input_ref']:
                if b['primary_eligible']:
                    eq(float(pr[metric+'_'+key]),r[key],'target_csv_values');counts['csv_numeric_values']+=1
                else:assert not pr[metric+'_'+key]
    # Independently aggregate all reported groups and metrics, including medians,
    # sign counts, all-20 repeat stability, and signed correlations.
    for source in ['D3','D1','D2','pooled_descriptive']:
        selected=[r for r in base_records if source=='pooled_descriptive' or r['source']==source]
        actual=summary['pooled_descriptive'] if source=='pooled_descriptive' else summary['sources'][source]
        expected_counts={
            'all_source_target_entries':len(selected),
            'original_two_guide_entries':sum(r['two_input_guides'] for r in selected),
            'primary_eligible':sum(r['primary_eligible'] for r in selected),
            'cross_M_zero_total_two_guide':sum(r['old_cross_weight_zero'] for r in selected),
            'cross_M_zero_primary':sum(r['old_cross_weight_zero'] and r['primary_eligible'] for r in selected),
            'newly_zero_primary':sum(r['newly_zero'] and r['primary_eligible'] for r in selected),
            'single_guide_fallback':sum(not r['two_input_guides'] for r in selected),
            'reference_two_guides_with_shared_lane':sum(r['reference_guide_count']==2 and r['shared_lane_count']>0 for r in selected)}
        assert expected_counts==actual['counts'],(source,expected_counts,actual['counts'])
        def in_group(r,name):
            if not r['primary_eligible']:return False
            if name=='all_primary':return True
            if name=='cross_M_zero':return r['old_cross_weight_zero']
            if name=='cross_M_positive':return not r['old_cross_weight_zero']
            if name=='newly_zero':return r['newly_zero']
            if name=='already_zero':return r['old_cross_weight_zero'] and not r['newly_zero']
            if name.startswith('input_min_'):
                n=r['input_min_guide_cells'];return name==('input_min_1' if n==1 else 'input_min_2-3' if n<4 else 'input_min_>=4')
            if name.startswith('reference_min_'):
                return name==('reference_min_4-9' if r['reference_min_matched_guide_cells']<10 else 'reference_min_>=10')
            raise AssertionError(name)
        for name,group in actual['groups'].items():
            members=[r for r in selected if in_group(r,name)]
            assert len(members)==group['n'],(source,name)
            if not members:assert not group['metrics'];continue
            for metric,observed in group['metrics'].items():
                records=[[lookup[(b['source'],b['target_id'],repeat,metric)] for repeat in range(20)] for b in members]
                prim=[r[0] for r in records]
                for key,value in observed['primary_median'].items():
                    expected=float(np.median([r[qmap.get(key,key)] for r in prim]));eq(expected,value,'summary_values');counts['summary_numeric_values']+=1
                expected_counts={key+'_positive':sum(r[key]>0 for r in prim) for key in ['E1','E2','C','D','Sref','C_full','D_input_ref']}
                expected_counts.update(both_half_C_positive=sum(r['C_half_A']>0 and r['C_half_B']>0 for r in prim),D_and_C_positive=sum(r['D']>0 and r['C']>0 for r in prim))
                assert expected_counts==observed['primary_counts'],(source,name,metric)
                for key in ['C_full','C_input','D_input_ref']:
                    eq(np.median([r[key] for r in prim]),observed[key+'_median'],'summary_values');counts['summary_numeric_values']+=1
                for key in ['D','C']:
                    positive=np.array([[r[key]>0 for r in records_i] for records_i in records])
                    eq(np.median(positive.mean(axis=1)),observed['repeat_'+key+'_positive_fraction_target_median'],'summary_values')
                    assert int(np.all(positive,axis=1).sum())==observed['repeat_all20_'+key+'_positive_targets']
                pairs={'C_input_vs_C_full':('C_input','C_full'),'C_input_vs_C_primary':('C_input','C'),'D_input_ref_vs_D_primary_shared_reference':('D_input_ref','D')}
                for cname,(kx,ky) in pairs.items():
                    xx=np.array([r[kx] for r in prim]);yy=np.array([r[ky] for r in prim]);ok=np.isfinite(xx)&np.isfinite(yy)
                    assert int(ok.sum())==observed['correlations'][cname]['n']
                    corr=None if ok.sum()<3 or len(np.unique(xx[ok]))<2 or len(np.unique(yy[ok]))<2 else float(spearmanr(xx[ok],yy[ok]).statistic)
                    eq(corr,observed['correlations'][cname]['spearman_rho'],'summary_values')
    report.update(status='PASS',production_script_sha256=digest(production_script),production_summary_sha256=digest(args.results/'summary.json'),comparison_max_errors=maxima,checks=counts)
    report['scope']='Independent fixed hash splits, all 174 entries, all 123 primary eligible targets x 20 repartitions x 3 metrics, all formula and decomposition quantities, missingness, source/pooled group counts, medians, sign stability and signed correlations. No production functions imported; no raw counts re-read.'
    report['limits']='Arithmetic and provenance verification only. Does not establish new biological replication, target-level significance, or predictive improvement.'
    return report

def main():
    p=argparse.ArgumentParser()
    for key in ['inputs','shallow','metadata','predictions','results','protocol','protocol-lock','out']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--production-script',type=Path)
    args=p.parse_args()
    output_file=args.out if args.out.suffix=='.json' else args.out/'INDEPENDENT_AUDIT.json'
    output_file.parent.mkdir(parents=True,exist_ok=True)
    lock=json.loads(args.protocol_lock.read_text())
    assert digest(args.protocol)==lock['protocol_sha256']
    rows,report=reconstruct(args)
    report['protocol_sha256']=digest(args.protocol)
    report=compare_production(args,rows,report)
    output_file.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
