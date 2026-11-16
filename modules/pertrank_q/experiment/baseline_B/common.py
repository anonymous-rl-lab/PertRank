"""Exact scoring and train-pool empirical reference uncertainty utilities."""
from pathlib import Path
import hashlib, itertools, math, json
import numpy as np
SOURCES=('D3','D1','D2')
ARMS=('linear_mse','conditional_mse','linear_list','conditional_list')
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def membership(scores,own,k):
    scores=np.asarray(scores,float); own=np.asarray(own,bool)
    assert scores.shape==own.shape and np.isfinite(scores).all()
    out=np.zeros((2,)+scores.shape)
    for d,sign in enumerate((1.,-1.)):
        v=np.where(~own,sign*scores,-np.inf)
        assert np.all((~own).sum(1)>=k)
        cut=np.sort(v,axis=1)[:,-k,None]
        above=(v>cut)&~own; equal=(v==cut)&~own
        out[d]=above+equal*(k-above.sum(1))[:,None]/equal.sum(1)[:,None]
    assert np.max(np.abs(out.sum(2)-k))<1e-10
    return out
def precision(scores,reference,own,k):
    a=membership(scores,own,k); b=membership(reference,own,k)
    return (a*b).sum(2).mean(0)/k
def ref_support(v,k):
    """Exact empirical donor-bootstrap support, not latent-truth probability.
    Counts shared across targets/queries within batch; batches sampled
    independently because cross-batch donor identity is unestablished.
    """
    e=v['reference_replicates']; batches=v['reference_target_batch']
    assert e.shape[2]==3
    choices=[]
    for a in range(4):
        for b in range(4-a):
            c=3-a-b
            w=math.factorial(3)/(math.factorial(a)*math.factorial(b)*math.factorial(c))/27
            choices.append((np.array([a,b,c],float)/3,w))
    batch_ids=np.unique(batches); labels=v['reference_donor_labels']
    for b in batch_ids:
        rows=np.flatnonzero(batches==b)
        assert np.all(labels[rows]==labels[rows[0]])
    out=np.zeros((2,)+v['reference'].shape); total=0.
    for cc in itertools.product(choices,repeat=len(batch_ids)):
        y=np.zeros_like(v['reference']); w=1.
        for b,(counts,prob) in zip(batch_ids,cc):
            take=batches==b; y[:,take]=(e[:,take]*counts).sum(2);w*=prob
        out+=w*membership(y,v['ownmask'],k);total+=w
    assert abs(total-1)<1e-12 and np.max(np.abs(out.sum(2)-k))<1e-10
    return out
def subset(v,ix):
    out={}
    qt={'X','base','raw','euclidean','reference','ownmask'}
    t={'target_ids','target_symbols','targethalf','reference_target_batch','reference_donor_labels','reference_control_group','guide_counts','input_cell_ids'}
    for key,a in v.items():
        if key in qt or key=='reference_replicates':out[key]=a[:,ix]
        elif key in t:out[key]=a[ix]
        else:out[key]=a
    return out
def inner_ids(vs,seed):
    ids=sorted(set(str(i) for v in vs.values() for i in v['target_ids']))
    order=sorted(ids,key=lambda t:hashlib.sha256(f'{seed}|{t}'.encode()).hexdigest())
    return {t:i%2 for i,t in enumerate(order)}
def block_interval(diff,blocks,draws=2000):
    u=np.unique(blocks); ss=np.array([diff[blocks==b].sum() for b in u]);nn=np.array([(blocks==b).sum() for b in u])
    ix=np.random.default_rng(2026111518).integers(len(u),size=(draws,len(u)))
    return np.quantile(ss[ix].sum(1)/nn[ix].sum(1),[.025,.975]).tolist()
