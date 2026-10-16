#!/usr/bin/env python3
"""Metadata-only new selection. This does not reconstruct the interrupted script."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
REC=ROOT.parents[1]
OLD=None # original source identity retained in frozen_assets provenance
SALT='pertrank-ntc-guide-replan-20261008|D3_Rest|'
def h(s):return hashlib.sha256(s.encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def savejson(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')

def split(cells,guide,lanes):
    """40 cells: prefer 20 within-lane pairs; unmatched singleton pairs only if needed."""
    paired=[];single=[]
    for lane in sorted(set(lanes[cells].tolist())):
        cc=[int(c) for c in cells if lanes[c]==lane]
        cc.sort(key=lambda c:h(SALT+'cell|'+guide+'|'+str(lane)+'|'+str(c)))
        for i in range(0,len(cc)-1,2):paired.append((cc[i],cc[i+1]))
        if len(cc)%2:single.append(cc[-1])
    paired.sort(key=lambda pair:h(SALT+'pair|'+guide+'|'+str(pair)))
    single.sort(key=lambda c:h(SALT+'single|'+guide+'|'+str(c)))
    pairs=paired[:20]
    if len(pairs)<20:pairs += [(single[i],single[i+1]) for i in range(0,2*(20-len(pairs)),2)]
    assert len(pairs)==20
    oriented=[]
    for pair in pairs:
        oriented.append(pair if int(h(SALT+'orientation|'+guide+'|'+str(pair))[-1],16)%2==0 else pair[::-1])
    out=np.array(oriented,np.int64).T
    assert out.shape==(2,20) and len(np.unique(out))==40
    return out

def main():
    if (ROOT/'selection_report.json').exists():raise RuntimeError('selection exists; immutable')
    z=np.load(ROOT/'metadata/obs_metadata.npz',allow_pickle=False);m={k:z[k] for k in z.files}
    types=m['type_categories'].astype(str);gc=m['gene_categories'].astype(str);guides=m['guide_categories'].astype(str)
    ok=(~m['low_quality'])&(m['gene_codes']>=0)&(m['type_codes']>=0)
    nt=ok&(m['type_codes']==np.flatnonzero(types=='non-targeting')[0])
    ntr=np.flatnonzero(nt);assert np.all(m['guide_codes'][ntr]>=0) and np.all(m['lane_codes'][ntr]>=0)
    ntby={int(g):ntr[m['guide_codes'][ntr]==g] for g in np.unique(m['guide_codes'][ntr])}
    eligible=sorted([g for g,c in ntby.items() if len(c)>=40],key=lambda g:guides[g])
    ntcells=np.array([split(ntby[g],guides[g],m['lane_codes']) for g in eligible])
    bad=np.array([('multi' in g.lower())or g in ('NTC','non-targeting','nan') for g in gc])
    targ=ok&(m['type_codes']==np.flatnonzero(types=='targeting')[0])&(~bad[np.clip(m['gene_codes'],0,len(gc)-1)])
    rows=np.flatnonzero(targ);key=m['gene_codes'][rows].astype(np.int64)*len(guides)+m['guide_codes'][rows]
    order=np.argsort(key,kind='stable');kk=key[order];edges=np.flatnonzero(np.r_[True,kk[1:]!=kk[:-1]])
    by={int(kk[a]):rows[order[a:b]] for a,b in zip(edges,np.r_[edges[1:],len(kk)])}
    cohortp=ROOT/'frozen_assets/A15_D3_cohort.npz'
    names=np.load(cohortp,allow_pickle=False)['names'].astype(str);geneindex={s:i for i,s in enumerate(gc)}
    a15=[]
    for target in names:
        gi=geneindex[target];cand=[(len(c),int(k%len(guides)),c) for k,c in by.items() if k//len(guides)==gi]
        cand.sort(key=lambda x:-x[0])
        if len(cand)<2 or min(x[0] for x in cand[:2])<100:continue
        pair=sorted(cand[:2],key=lambda x:h('pertrescue-a15|'+guides[x[1]]))
        a15.append((target,pair))
    assert len(a15)==441,('historical A15 cohort differs',len(a15))
    a15.sort(key=lambda x:h(SALT+'target-control|'+x[0]));a15=a15[:64]
    tcells=np.array([[split(c,guides[g],m['lane_codes']) for _,g,c in pair] for target,pair in a15])
    ids=np.unique(np.r_[ntcells.ravel(),tcells.ravel()]);assert len(ids)==ntcells.size+tcells.size
    orig=ROOT/'frozen_assets/ntc_original/original_input_ntc_metadata.npz'
    c=np.load(orig,allow_pickle=False);assert len(c['keep'])==13390
    assert np.array_equal(c['gene_ids_full'].astype(str),m['gene_ids_full'].astype(str))
    ng=np.array([guides[g] for g in eligible]);tg=np.array([[guides[g] for _,g,_ in pair] for _,pair in a15]);tn=np.array([s for s,_ in a15])
    lanes=m['lane_codes'][ntcells];nla=len(m['lane_categories']);lc=np.array([[np.bincount(half,minlength=nla) for half in pair] for pair in lanes])
    np.savez_compressed(ROOT/'selection.npz',cell_ids=ids,total_counts=m['total_counts'][ids],ntc_cells=ntcells,ntc_guides=ng,ntc_counts=np.array([len(ntby[g]) for g in eligible]),
      ntc_positions=np.searchsorted(ids,ntcells),target_cells=tcells,target_positions=np.searchsorted(ids,tcells),target_names=tn,target_guides=tg,
      lane_codes=m['lane_codes'][ids],lane_categories=m['lane_categories'],ntc_lane_counts=lc,keep=c['keep'],scale=c['scale'],gene_ids_full=c['gene_ids_full'])
    recs=[{'guide':str(g),'eligible_cells':len(ntby[i]),'cells_half_A':ntcells[j,0].tolist(),'cells_half_B':ntcells[j,1].tolist(),
      'lane_counts_A':lc[j,0].tolist(),'lane_counts_B':lc[j,1].tolist()} for j,(i,g) in enumerate(zip(eligible,ng))]
    savejson(ROOT/'ntc_selected_cells.json',recs)
    savejson(ROOT/'target_selected_cells.json',[{'target':str(t),'guides':tg[j].tolist(),'cells':tcells[j].tolist()} for j,t in enumerate(tn)])
    mr=json.loads((ROOT/'metadata/metadata_report.json').read_text())
    report={'status':'METADATA_SELECTION_COMPLETE','new_replanning_not_interrupted_script_replay':True,'donor':'D3_Rest','raw_expression_read':False,
      'source_identity':{'url':mr['source_url'],'etag':mr['etag'],'size':mr['source_size']},'all_ntc_guides':len(ntby),'all_ntc_cells':len(ntr),
      'eligible_ntc_guides':len(eligible),'ntc_cells':ntcells.size,'target_pairs':len(a15),'target_cells':tcells.size,'total_selected_cells':len(ids),
      'historical_A15_qualifying_target_pairs':441,'genes':len(c['keep']),'scale':float(c['scale']),'salt':SALT,'split':'20 within-lane pairs if available; otherwise minimally necessary cross-lane singleton pairs',
      'perfect_ntc_half_lane_balance':int(np.all(lc[:,0]==lc[:,1],axis=1).sum()),'ntc_half_max_lane_difference':int(np.abs(lc[:,0]-lc[:,1]).max()),
      'metadata_sha256':sha(ROOT/'metadata/obs_metadata.npz'),'selection_sha256':sha(ROOT/'selection.npz'),'script_sha256':sha(__file__),
      'old_context_sha256':sha(orig),'A15_cohort_source_sha256':sha(cohortp),'ntc_original_control_overlap':int(np.intersect1d(ntcells,c['cell_ids']).size),
      'original_control_reuse_note':'No control mean is subtracted; fixed keep and scale are conditioned upon.'}
    savejson(ROOT/'selection_report.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
