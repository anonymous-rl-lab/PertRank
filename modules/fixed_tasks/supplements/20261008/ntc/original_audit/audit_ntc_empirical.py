#!/usr/bin/env python3
"""Offline independent reconstruction of complete NTC empirical results.

Refuses partial results. Reads archived selected rows only; never downloads.
Uses sparse count transformation instead of the production dense accumulator.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.sparse import load_npz

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'ntc_guide'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def main():
    # Check all completion markers before loading any empirical spectra/results.
    for name in ('RESULT.json','RAW_READ_COMPLETE.json','RUN_FREEZE.json',
                 'half_means.npz','resampling.npz'):
        assert (ROOT/name).is_file(),f'Empirical audit requires complete artifact: {name}'
    result=read(ROOT/'RESULT.json');rawdone=read(ROOT/'RAW_READ_COMPLETE.json')
    assert result['status']=='COMPLETE' and rawdone['status']=='RAW_READ_COMPLETE'
    freeze=read(ROOT/'RUN_FREEZE.json')
    for name,digest in freeze['file_sha256'].items():assert sha(ROOT/name)==digest
    assert result['half_means_sha256']==sha(ROOT/'half_means.npz')
    assert result['selection_sha256']==sha(ROOT/'selection.npz')
    s=npz(ROOT/'selection.npz');saved=npz(ROOT/'half_means.npz');r=npz(ROOT/'resampling.npz')
    g=len(s['ntc_guides']);t=len(s['target_names']);p=len(s['keep'])
    ids=s['cell_ids'];nslots=2*g+4*t
    slots=np.full(len(ids),-1,dtype=int)
    for guide in range(g):
        for half in range(2):slots[s['ntc_positions'][guide,half]]=2*guide+half
    for target in range(t):
        for guide in range(2):
            for half in range(2):slots[s['target_positions'][target,guide,half]]=2*g+4*target+2*guide+half
    assert np.all(np.bincount(slots)==20)
    sums=np.zeros((nslots,p));seen=np.zeros(len(ids),bool)
    batches=sorted((ROOT/'raw_batches').glob('batch_*.json'))
    assert len(batches)==rawdone['batches']
    for meta in batches:
        m=read(meta);f=meta.with_suffix('.npz');assert sha(f)==m['sha256']
        rows=np.asarray(m['rows']);pos=np.searchsorted(ids,rows)
        assert np.unique(rows).size==rows.size
        assert np.array_equal(ids[pos],rows) and not seen[pos].any()
        x=load_npz(f)
        assert np.array_equal(np.asarray(x.sum(axis=1)).ravel(),s['total_counts'][pos])
        y=x[:,s['keep']].tocsr().astype(np.float64)
        # Sparse zeros transform to zero, so only nonzeros need normalization.
        factors=np.repeat(float(s['scale'])/s['total_counts'][pos],np.diff(y.indptr))
        y.data=np.log1p(y.data*factors)
        yy=y.tocoo()
        np.add.at(sums,(slots[pos][yy.row],yy.col),yy.data)
        seen[pos]=True
    assert seen.all() and rawdone['rows']==len(ids)
    means=sums/20
    ntc=means[:2*g].reshape(g,2,p);target=means[2*g:].reshape(t,2,2,p)
    errors={}
    def check(key,x,y,atol=1e-12):
        error=float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
        errors[key]=error
        assert np.allclose(x,y,rtol=1e-11,atol=atol),(key,error)
    check('ntc_sparse_half_reconstruction',ntc,saved['ntc'],1e-13)
    check('target_sparse_half_reconstruction',target,saved['target'],1e-13)
    assert np.array_equal(saved['ntc_guides'],s['ntc_guides'])
    assert np.array_equal(saved['target_names'],s['target_names'])
    assert np.array_equal(saved['target_guides'],s['target_guides'])
    assert np.array_equal(saved['gene_ids'],s['gene_ids_full'][s['keep']])
    a,b=ntc[:,0],ntc[:,1]
    # Uncentered Gram contraction is independent of production's centered Gram.
    gram=a@b.T/p
    d=2*(np.trace(gram)-gram.sum()/g)/(g-1)
    check('NTC_point',d,result['ntc']['pair_energy'])
    nrep=result['ntc']['repetitions'];seed=result['ntc']['seed']
    assert nrep==2000 and seed==20261008
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,g,size=(nrep,g))
    counts=np.zeros((nrep,g),dtype=np.int64)
    np.add.at(counts,(np.arange(nrep)[:,None],draws),1)
    boot=2*(counts@np.diag(gram)-np.einsum('ij,ij->i',counts@gram,counts)/g)/(g-1)
    null=np.array([2*(gram[np.arange(g),rng.permutation(g)].sum()-gram.sum()/g)/(g-1)
                   for _ in range(nrep)])
    check('NTC_bootstrap_all_draws',boot,r['ntc_bootstrap'])
    check('NTC_permutation_all_draws',null,r['ntc_permutation'])
    check('NTC_bootstrap_interval',np.quantile(boot,[.025,.975]),result['ntc']['bootstrap_95_percentile'])
    check('NTC_permutation_interval',np.quantile(null,[.025,.975]),result['ntc']['permutation_null_95'])
    pgreater=(1+(null>=d).sum())/(nrep+1)
    ptwo=(1+(abs(null)>=abs(d)).sum())/(nrep+1)
    check('NTC_p_greater',pgreater,result['ntc']['permutation_p_greater'],0.)
    check('NTC_p_two_sided',ptwo,result['ntc']['permutation_p_two_sided'],0.)
    vals=np.array([np.dot(target[i,0,0]-target[i,1,0],target[i,0,1]-target[i,1,1])/p for i in range(t)])
    check('target_pair_energies',vals,r['target_pair_energy'])
    tr=result['target_same_gene_control'];assert tr['repetitions']==2000 and tr['seed']==20261008
    rt=np.random.default_rng(tr['seed'])
    tboot=vals[rt.integers(0,t,size=(2000,t))].mean(1)
    check('target_bootstrap_all_draws',tboot,r['target_bootstrap'])
    check('target_point',vals.mean(),tr['pair_energy_mean'])
    check('target_interval',np.quantile(tboot,[.025,.975]),tr['bootstrap_95_percentile'])
    report={'status':'PASS','raw_rows_reconstructed':len(ids),'ntc_guides':g,'target_pairs':t,
        'genes':p,'bootstrap_draws':nrep,'permutation_draws':nrep,
        'errors':errors,'method':'independent sparse raw-count normalization and uncentered Gram resampling',
        'result_sha256':sha(ROOT/'RESULT.json'),'half_means_sha256':sha(ROOT/'half_means.npz'),
        'network_reads':0,'model_fits':0}
    (HERE/'ntc_empirical_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
