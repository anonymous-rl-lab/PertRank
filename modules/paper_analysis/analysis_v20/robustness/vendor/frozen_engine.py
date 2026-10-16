#!/usr/bin/env python3
"""Input-only D4 prediction primitives. This module has no network or truth API.

Log control rows deliberately use A8 float32 storage followed by float64
moments; target log rows and every Pearson operation use float64. All weights
multiply the same original log effect. Reader owns access/phase enforcement.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import hashlib,json
from pathlib import Path
import numpy as np
from scipy import linalg
from scipy.cluster.hierarchy import fcluster,linkage
from scipy.spatial.distance import squareform

NC=2000; K=5; BETA=.9; THETA=100.; CLIP=float(np.sqrt(NC))
DEPTHS=np.arange(20,51,5)
ELIGIBLE_COMPETITORS=('Euclidean','diagonal','TRADE_scalar','PR_E','PR_M')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,**arrays)

def require_counts(raw,totals):
    if raw.ndim!=2 or len(totals)!=raw.shape[0]:raise ValueError('raw/totals shape')
    if not np.array_equal(np.asarray(raw.sum(1)).ravel(),totals):raise ValueError('library totals must use full gene axis')
    if np.any(np.asarray(totals)<=0):raise ValueError('nonpositive total')
    vals=raw.data if hasattr(raw,'toarray') else raw
    if not np.isfinite(vals).all() or np.any(vals<0) or np.any(vals!=np.floor(vals)):
        raise ValueError('invalid raw UMI')

def dense_columns(raw,keep):
    x=raw[:,keep]
    return x.toarray().astype(float) if hasattr(x,'toarray') else np.asarray(x,float)

def residual(dense,totals,p):
    mu=np.asarray(totals,float)[:,None]*p[None,:]
    if np.any(mu<=0):raise ValueError('zero NTC-derived expected count on keep axis')
    return np.clip((dense-mu)/np.sqrt(mu+mu*mu/THETA),-CLIP,CLIP)

def geometry(z):
    z=np.asarray(z,float);n,g=z.shape
    if n!=NC or min(n,g)<=K:raise ValueError('control dimensions')
    mean=z.mean(0);zc=z-mean
    ev,left=linalg.eigh(zc@zc.T/(n-1),subset_by_index=[n-K,n-1],driver='evr',check_finite=False)
    ev=ev[::-1];left=left[:,::-1]
    if ev[-1]<=0:raise ValueError('fewer than five positive covariance directions')
    U=(zc.T@left)/np.sqrt((n-1)*ev)[None,:]
    for k in range(K):
        if U[np.argmax(abs(U[:,k])),k]<0:U[:,k]*=-1
    if np.max(abs(U.T@U-np.eye(K)))>1e-10:raise ValueError('nonorthogonal control directions')
    variance=(zc*zc).sum(0)/(n-1)
    lam=np.var(z@U,axis=0,ddof=1)
    if np.max(abs(lam-ev))>1e-8:raise ValueError('eigenvalue/projection variance mismatch')
    return mean,variance,U,lam

def build_control_model(raw,totals,gene_ids,scale,control_ids=None,output_path=None):
    require_counts(raw,totals)
    gene_ids=np.asarray(gene_ids).astype(str);totals=np.asarray(totals,float)
    if raw.shape!=(NC,len(gene_ids)) or len(np.unique(gene_ids))!=len(gene_ids):raise ValueError('control gene/cell axis')
    # A8 keeps every gene detected in the input NTC; evaluation controls unused.
    keep=np.flatnonzero(np.asarray(raw.sum(0)).ravel()>0)
    if len(keep)<300:raise ValueError('too few weight genes for fixed query rules')
    dense=dense_columns(raw,keep)
    log=np.log1p(dense*(float(scale)/totals)[:,None]).astype(np.float32).astype(float)
    cm,cv,U,lam=geometry(log)
    if not np.all(cm>0):raise ValueError('positive-count and positive-log-mean keep disagree')
    p=np.asarray(raw[:,keep].sum(0)).ravel().astype(float)/totals.sum()
    z=residual(dense,totals,p)
    pm,pv,pU,plam=geometry(z)
    model=dict(keep=keep,gene_ids_full=gene_ids,gene_ids_keep=gene_ids[keep],scale=np.array(float(scale)),
      n_control=np.array(NC),cm=cm,cv=cv,U=U,lam=lam,trC=np.array(cv.sum()),
      proj_control=cm@U,pr_p=p,pr_mean=pm,pr_var=pv,pr_U=pU,pr_lam=plam,
      pr_proj_control=pm@pU,control_ids=np.array([] if control_ids is None else control_ids,dtype=np.int64),
      theta=np.array(THETA),clip=np.array(CLIP),k=np.array(K),beta=np.array(BETA),
      log_control_precision=np.array('float32_per_row_then_float64_moments'),
      target_log_precision=np.array('float64'),pr_precision=np.array('float64'),
      control_total_umi=np.array(totals.sum()))
    if control_ids is not None and (len(control_ids)!=NC or len(np.unique(control_ids))!=NC):raise ValueError('control IDs')
    if output_path is not None:save_npz(output_path,**model)
    return model

def make_blocks(rows):
    rows=np.asarray(rows,float)
    A=rows-rows.mean(1,keepdims=True)
    A=A/np.maximum(np.linalg.norm(A,axis=1,keepdims=True),1e-9)
    D=1-A@A.T;np.fill_diagonal(D,0)
    blocks=fcluster(linkage(squareform(D,checks=False),'complete'),30,'maxclust')
    # As in A8, maxclust can have fewer labels under exact ties; record actual count.
    return blocks

def build_query_context(planning_sum,model,target_ids,target_names=None,output_path=None):
    target_ids=np.asarray(target_ids).astype(str);n=len(target_ids)
    g=model['gene_ids_keep'];planning_sum=np.asarray(planning_sum,float)
    if planning_sum.shape!=(n,len(g)) or len(np.unique(target_ids))!=n:raise ValueError('planning target/gene axis')
    if not np.isfinite(planning_sum).all():raise ValueError('nonfinite planning sum')
    pool=(planning_sum/100-model['cm']).T
    g2k={v:i for i,v in enumerate(g.astype(str))}
    own=np.array([g2k.get(v,-1) for v in target_ids],int)
    qt=[]
    for part in np.array_split(np.argsort(model['cm'],kind='stable'),3):
        order=sorted((hashlib.sha256(('tcell-probe|query|'+str(g[i])).encode()).hexdigest(),int(i)) for i in part)
        qt.extend(i for _,i in order[:100])
    qt=np.asarray(qt,int)
    corrected=pool.copy()
    for j,k in enumerate(own):
        if k>=0:corrected[k,j]=0.
    variances=corrected.var(1)
    qr=np.argsort(-variances)[:300]
    qu=np.union1d(qt,qr)
    q=dict(target_ids=target_ids,target_names=np.asarray(target_ids if target_names is None else target_names).astype(str),
      query_gene_ids=g[qu],qu=qu,q_terc=qt,q_resp=qr,query_keep_positions=qu,
      responsive_positions=np.searchsorted(qu,qr),tercile_positions=np.searchsorted(qu,qt),
      blocks_responsive=make_blocks(pool[qr]),blocks_terciles=make_blocks(pool[qt]),self_row=own,
      planning_corrected_variance=variances,planning_cells_per_target=np.array(100),
      query_salt=np.array('tcell-probe|query|'))
    q['n_blocks_responsive']=np.array(len(np.unique(q['blocks_responsive'])))
    q['n_blocks_terciles']=np.array(len(np.unique(q['blocks_terciles'])))
    if output_path is not None:save_npz(output_path,**q)
    return q

def scalar(energy,noise):
    energy=np.asarray(energy,float);noise=np.asarray(noise,float)
    if not np.isfinite(energy).all() or not np.isfinite(noise).all() or energy.min()<-1e-10 or noise.min()<-1e-10:
        raise ValueError('nonfinite/negative PSD energy or noise')
    return np.clip(1-noise/np.maximum(energy,1e-300),0,1)

def effect_stats(rows,control_mean,U,control_var,lam):
    x=rows.reshape(5,20,-1);g=x.shape[-1]
    sums=x.sum(1);sumsq=(x*x).sum((1,2))
    y=sums/20-control_mean
    traces=(sumsq-(sums*sums).sum(1)/20)/19
    ps=(rows@U).reshape(5,20,K);pmean=ps.mean(1)-control_mean@U;pvar=ps.var(1,ddof=1)
    aE=np.mean(y*y,1);nuE=(traces/20+np.sum(control_var)/NC)/g
    aM=aE-BETA*np.sum(pmean*pmean,1)/g
    nuM=nuE-BETA*np.sum(pvar/20+lam/NC,1)/g
    return dict(y=y,aE=aE,nuE=nuE,aM=aM,nuM=nuM,pmean=pmean,pvar=pvar,trace_target=traces)

def process_target(raw,totals,cell_ids,target_id,donor,model,query,*,selected_competitor,trade_adapter=None):
    """Raw rows exactly the target's permuted positions100:200, grouped5×20."""
    if selected_competitor not in ELIGIBLE_COMPETITORS:raise ValueError('unknown frozen competitor')
    require_counts(raw,totals);cell_ids=np.asarray(cell_ids,np.int64)
    if raw.shape!=(100,len(model['gene_ids_full'])) or cell_ids.shape!=(100,) or len(np.unique(cell_ids))!=100:
        raise ValueError('target must have100 unique cells on full gene axis')
    keep=model['keep'];qu=query['qu'];dense=dense_columns(raw,keep)
    log=np.log1p(dense*(float(model['scale'])/np.asarray(totals,float))[:,None])
    ls=effect_stats(log,model['cm'],model['U'],model['cv'],model['lam'])
    pr=residual(dense,totals,model['pr_p'])
    rs=effect_stats(pr,model['pr_mean'],model['pr_U'],model['pr_var'],model['pr_lam'])
    wE=scalar(ls['aE'],ls['nuE']);wM=scalar(ls['aM'],ls['nuM'])
    if selected_competitor=='Euclidean':wc=wE
    elif selected_competitor=='PR_E':wc=scalar(rs['aE'],rs['nuE'])
    elif selected_competitor=='PR_M':wc=scalar(rs['aM'],rs['nuM'])
    elif selected_competitor=='diagonal':
        d=1/(model['cv']+np.median(model['cv']))
        v=log.reshape(5,20,-1).var(1,ddof=1)
        wc=scalar(np.mean(d*ls['y']**2,axis=1),np.mean(d*(v/20+model['cv']/NC),axis=1))
    else:
        # Caller must supply the locked v17 official-core scalar adapter, never an approximation.
        if trade_adapter is None:raise ValueError('TRADE selected: frozen official-core adapter required')
        v=log.reshape(5,20,-1).var(1,ddof=1)
        se=np.sqrt(v/20+model['cv']/NC)
        wc=np.asarray(trade_adapter(ls['y'],se,target_id),float)
        if wc.shape!=(5,) or not np.isfinite(wc).all() or np.any((wc<0)|(wc>1)):raise ValueError('TRADE adapter weight contract')
    weights=np.stack([np.ones(5),wE,wM,wc])
    Y=ls['y'][:,qu]
    # Cell-equivalence depth20 shares original groups; >20 adds fixed hash-order complement.
    raw_depth=np.empty((7,5,len(qu)));raw_depth[0]=Y
    additions=np.empty((5,30),np.int16)
    sums20=log.reshape(5,20,-1)[:,:,qu].sum(1)
    for r in range(5):
        complement=[j for j in range(100) if not 20*r<=j<20*r+20]
        def key(j):
            st=json.dumps(['pertrank-celleq-v16b',str(donor),str(target_id),r,int(cell_ids[j]),2026],separators=(',',':'))
            return hashlib.sha256(st.encode()).digest(),int(cell_ids[j])
        additions[r]=sorted(complement,key=key)[:30]
        csum=np.cumsum(log[additions[r]][:,qu],axis=0)
        for di,n in enumerate(DEPTHS[1:],1):raw_depth[di,r]=(sums20[r]+csum[n-21])/(float(n))-model['cm'][qu]
    result=dict(weights=weights,Y=Y,predictions=weights[:,:,None]*Y[None,:,:],raw_depth_Y=raw_depth,
      additions=additions,log_aE=ls['aE'],log_nuE=ls['nuE'],log_aM=ls['aM'],log_nuM=ls['nuM'],
      log_projection_mean=ls['pmean'],log_projection_variance=ls['pvar'],
      pr_aE=rs['aE'],pr_nuE=rs['nuE'],pr_aM=rs['aM'],pr_nuM=rs['nuM'],
      pr_projection_mean=rs['pmean'],pr_projection_variance=rs['pvar'],
      cell_ids=cell_ids,query_sums20=sums20,
      log_trace_target=ls['trace_target'],pr_trace_target=rs['trace_target'])
    for k,v in result.items():
        if not np.isfinite(v).all():raise ValueError('nonfinite target output '+k)
    return result
