#!/usr/bin/env python3
"""Batch-read full-axis raw-count AnnData to PertRank nominations.

X contains raw integer counts; obs columns are target_ids, target_gene_ids,
batch_ids and is_control; obs_names and var_names are unique canonical IDs.
All supplied cells are prespecified inputs from one matching stratum.
"""
import argparse,json,time,sys
try:
    import resource
except ImportError:
    resource=None
from pathlib import Path
import numpy as np
from pertrank.core import _strings,_controls,_directions,reliability_weights
from run_counts import save_outputs

def _read(adata,rows):
    x=adata.X[rows,:]
    if hasattr(x,'toarray'): x=x.toarray()
    x=np.asarray(x,dtype=np.float64)
    if not np.isfinite(x).all() or (x<0).any() or (x!=np.floor(x)).any():
        raise ValueError('AnnData X must contain finite nonnegative integer raw counts')
    return x

def fit_anndata(adata,query_ids,batch_rows=512):
    n,g=adata.shape
    if n==0 or g==0 or batch_rows<1: raise ValueError('Nonempty input and positive batch_rows required')
    genes=_strings(adata.var_names,g,'var_names');cells=_strings(adata.obs_names,n,'obs_names')
    if len(set(genes))!=g or len(set(cells))!=n: raise ValueError('Unique gene and cell IDs required')
    for key in ('target_ids','target_gene_ids','batch_ids','is_control'):
        if key not in adata.obs: raise ValueError('Missing obs column '+key)
    targets=_strings(adata.obs['target_ids'].to_numpy(),n,'target_ids')
    own=_strings(adata.obs['target_gene_ids'].to_numpy(),n,'target_gene_ids')
    batches=_strings(adata.obs['batch_ids'].to_numpy(),n,'batch_ids')
    if len(set(batches))!=1: raise ValueError('Select one prespecified control-matching stratum')
    ctrl=_controls(adata.obs['is_control'].to_numpy(),n)
    if ctrl.sum()<6: raise ValueError('At least six matched controls required')
    names=np.unique(targets[~ctrl])
    if len(names)==0 or set(names)&set(targets[ctrl]): raise ValueError('Distinct target and control IDs required')
    # First pass sees every full-axis count. Filtering occurs after full totals.
    totals=np.zeros(n);csums=np.zeros(g)
    for start in range(0,n,batch_rows):
        stop=min(start+batch_rows,n);x=_read(adata,slice(start,stop))
        totals[start:stop]=x.sum(1);csums+=x[ctrl[start:stop]].sum(0)
    if (totals<=0).any(): raise ValueError('Zero-library cells must be QC-filtered before input')
    keep=np.flatnonzero(csums>0);queries=_strings(query_ids,len(query_ids),'query_ids')
    if not len(queries) or len(set(queries))!=len(queries): raise ValueError('Unique nonempty queries required')
    if set(queries)-set(genes[keep]): raise ValueError('Queries must occur on the input-control gene axis')
    scale=float(np.median(totals[ctrl]));ci=np.flatnonzero(ctrl)
    xc=np.empty((len(ci),len(keep)),dtype=float)
    for start in range(0,len(ci),batch_rows):
        ix=ci[start:start+batch_rows];x=_read(adata,ix)
        xc[start:start+len(ix)]=np.log1p(x[:,keep]*(scale/totals[ix])[:,None]).astype(np.float32).astype(float)
    cm=xc.mean(0);cv=xc.var(0,ddof=1);u,lam=_directions(xc-cm)
    del xc
    model=dict(directions=u,eigenvalues=lam,mean=cm,variance=cv,n_control=len(ci))
    stats=[]
    # Each target is read separately; target cells need never form one dense cohort.
    for name in names:
        ix=np.flatnonzero((~ctrl)&(targets==name));gs=np.unique(own[ix])
        if len(ix)<2 or len(gs)!=1: raise ValueError('Each target needs >=2 cells and one target_gene_id')
        xt=np.empty((len(ix),len(keep)),dtype=float)
        for start in range(0,len(ix),batch_rows):
            rows=ix[start:start+batch_rows];x=_read(adata,rows)
            xt[start:start+len(rows)]=np.log1p(x[:,keep]*(scale/totals[rows])[:,None])
        y=xt.mean(0)-cm;var=xt.var(0,ddof=1);py=y@u;pv=(xt@u).var(0,ddof=1);b=len(ix)
        stats.append((y,float(y@y),float(var.sum()/b+cv.sum()/len(ci)),py*py,pv/b+lam/len(ci),b,str(batches[0]),str(gs[0])))
    result=dict(gene_ids=genes[keep],target_ids=names,effects=np.array([s[0] for s in stats]),
        energy=np.array([s[1] for s in stats]),noise_trace=np.array([s[2] for s in stats]),
        direction_energy=np.array([s[3] for s in stats]),direction_noise=np.array([s[4] for s in stats]),
        n_cells=np.array([s[5] for s in stats]),batch_ids=np.array([s[6] for s in stats]),
        target_gene_ids=np.array([s[7] for s in stats]),control_models={str(batches[0]):model})
    result['weights']=reliability_weights(result['energy'],result['noise_trace'],result['direction_energy'],result['direction_noise'])
    metadata=dict(input_format='AnnData X raw counts; batched full-axis access',batch_rows=batch_rows,
        normalization_scales={str(batches[0]):scale},weight_gene_positions=keep.tolist(),full_gene_count=g,
        weight_gene_count=len(keep),n_input_controls=len(ci),n_input_target_cells=int((~ctrl).sum()),
        target_depths=sorted(set(result['n_cells'].tolist())),normalization='log1p(count * median full-library input-control total / full-library total)',
        control_precision='float32 per log value, then float64 moments',target_precision='float64')
    return result,metadata

def run(path,query_path,out,top_k=25,seed=2026,batch_rows=512):
    import anndata as ad
    if Path(out).exists(): raise FileExistsError('Choose a new output directory')
    t0=time.perf_counter();adata=ad.read_h5ad(path,backed='r')
    try:
        queries=np.array([line.strip() for line in Path(query_path).read_text().splitlines() if line.strip()])
        result,metadata=fit_anndata(adata,queries,batch_rows)
    finally: adata.file.close()
    metadata['peak_process_rss_MiB_before_output']=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024 if sys.platform=='darwin' else 1024)) if resource is not None else None
    return save_outputs(result,metadata,queries,path,out,top_k,seed,t0)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path)
    p.add_argument('--queries',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--top-k',type=int,default=25);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--batch-rows',type=int,default=512);a=p.parse_args()
    print(json.dumps(run(a.input,a.queries,a.out,a.top_k,a.seed,a.batch_rows),indent=2))
