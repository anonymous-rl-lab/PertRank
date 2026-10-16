#!/usr/bin/env python3
"""Build native full-gene FR inputs from exact A19-R shallow cells and original NTCs."""
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np
import scipy.sparse as sp

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--donor',required=True)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--controls',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert a.donor in ['D3_Rest','D1_Rest','D2_Rest'] and not a.out.exists()
    ctx=load(a.source/'context.npz');plan=load(a.source/'cell_plan.npz')
    original=load(a.source/'predictions_locked.npz');control=load(a.controls/'metadata.npz')
    cx=sp.load_npz(a.controls/'counts.npz');sx=sp.load_npz(a.source/'source_counts.npz')
    assert np.array_equal(control['gene_ids'],ctx['gene_ids_full'])
    assert np.array_equal(original['target_ids'],plan['target_ids'])
    assert len(control['cell_ids'])==int(ctx['n_control'])==2000
    assert float(control['scale'])==float(ctx['scale'])
    assert np.array_equal(np.asarray(cx.sum(1)).ravel(),control['total_counts'])
    assert np.array_equal(np.asarray(sx.sum(1)).ravel(),plan['total_counts'])
    shallow=np.flatnonzero(plan['role']==0)
    idx=plan['target_index'][shallow]
    expected={'D3_Rest':52,'D1_Rest':59,'D2_Rest':63}[a.donor]
    assert len(plan['target_ids'])==expected and np.array_equal(np.bincount(idx),np.full(expected,20))
    cells=np.r_[control['cell_ids'],plan['cell_ids'][shallow]]
    assert len(np.unique(cells))==len(cells)
    assert not np.intersect1d(cells,plan['cell_ids'][plan['role']==1]).size
    counts=sp.vstack([cx,sx[shallow]],format='csr')
    assert counts.shape==(2000+20*expected,len(ctx['gene_ids_full']))
    assert np.isfinite(counts.data).all() and np.all(counts.data>=0) and np.all(counts.data==np.floor(counts.data))
    qm={str(g):i for i,g in enumerate(ctx['gene_ids_full'])}
    qcols=np.array([qm[str(q)] for q in original['query_ids']])
    # Independently bind cell-level raw inputs to the immutable query predictions.
    qkeep=np.searchsorted(ctx['keep'],qcols);assert np.array_equal(ctx['keep'][qkeep],qcols)
    rebuilt=[]
    for j in range(expected):
        ix=np.flatnonzero(idx==j)
        tt=plan['total_counts'][shallow[ix]]
        y=np.log1p(sx[shallow[ix]][:,qcols].toarray().astype(float)*(float(ctx['scale'])/tt)[:,None]).mean(0)-ctx['cm_keep'][qkeep]
        rebuilt.append(y)
    error=float(np.max(abs(np.array(rebuilt).T-original['effects'][0])))
    assert error<1e-12,error
    a.out.mkdir(parents=True)
    sp.save_npz(a.out/'counts.npz',counts)
    np.savez_compressed(a.out/'axes.npz',labels=np.r_[np.full(2000,-1,dtype=np.int32),idx],
            target_ids=plan['target_ids'],gene_ids=ctx['gene_ids_full'],cell_ids=cells,
            query_ids=original['query_ids'],query_columns=qcols,
            input_control_count=np.array(2000),target_cells_per_group=np.array(20))
    files={n:{'path':str((a.out/n).resolve()),'sha256':sha(a.out/n)} for n in ['counts.npz','axes.npz']}
    prep={'status':'COMPLETE','task':a.donor,'rep':0,'files_sha256':{n:v['sha256'] for n,v in files.items()},
          'input_scope':'exact original A19-R shallow cells; no source deep or reference cells','n_targets':expected,'n_cells':len(cells),'query_raw_maxabs':error}
    (a.out/'INPUT_AUDIT.json').write_text(json.dumps(prep,indent=2)+'\n')
    sources=[a.source/x for x in ['context.npz','cell_plan.npz','source_counts.npz','predictions_locked.npz']]+[a.controls/x for x in ['metadata.npz','counts.npz','CONTROL_RECOVERY.json']]+[Path(__file__)]
    lock={'status':'READY_EXACT_A19R_INPUT','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'donor':a.donor,'methods':{'FR':{'files':files,'input_directory':str(a.out.resolve())}},
       'checks':prep,'source_sha256':{str(p.resolve()):sha(p) for p in sources},'reference_read':False,
       'configuration':'official FR300 alpha.1 lasso1e-4 iter50 seed20261012; full raw gene/cell input'}
    (a.out/'INPUT_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n')
    print(json.dumps({'status':lock['status'],'donor':a.donor,'checks':prep},indent=2))

if __name__=='__main__':main()
