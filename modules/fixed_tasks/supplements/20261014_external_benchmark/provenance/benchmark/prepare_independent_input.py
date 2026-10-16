from pathlib import Path
import sys,json,hashlib,time
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'independent/execution'
sys.path.insert(0,str(SRC))
from role_loader import load_role
def sha(p):
    with open(p,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    t=time.time(); out=ROOT/'inputs/D4_Stim_independent';out.mkdir(parents=True,exist_ok=True)
    sel=dict(np.load(SRC/'selection_D4_Stim8hr.npz',allow_pickle=False))
    p=dict(np.load(SRC/'D4_Stim8hr/prediction.npz',allow_pickle=False))
    m=dict(np.load(SRC/'D4_Stim8hr/control_model.npz',allow_pickle=False))
    raw=SRC/'D4_Stim8hr/raw'
    c,ct=load_role(raw,sel,'NTC_input'); a,at=load_role(raw,sel,'A_input')
    counts=sparse.vstack([c,a],format='csr')
    labels=np.r_[np.full(c.shape[0],-1,np.int32),np.repeat(np.arange(len(sel['target_ids']),dtype=np.int32),20)]
    qix=np.array([np.flatnonzero(sel['gene_ids_full']==g)[0] for g in sel['query_ids']])
    cq=np.log1p(c[:,qix].toarray().astype(float)*(float(m['scale'])/ct)[:,None]).astype(np.float32).astype(float).mean(0)
    aq=np.log1p(a[:,qix].toarray().astype(float)*(float(m['scale'])/at)[:,None]).reshape(len(sel['target_ids']),20,300).mean(1)-cq
    err=float(np.max(np.abs(aq-p['input_effects'])))
    assert err<1e-10,err
    sparse.save_npz(out/'counts.npz',counts)
    np.savez_compressed(out/'axes.npz',labels=labels,gene_ids=sel['gene_ids_full'],target_ids=sel['target_ids'],target_names=sel['target_names'],cell_ids=np.r_[sel['NTC_input'],sel['A_input'].ravel()],query_ids=sel['query_ids'],query_blocks=sel['query_blocks'],query_columns=qix,keep=m['keep'],scale=m['scale'],input_effects=p['input_effects'],weights_euclidean=p['weights_euclidean'],weights_pertrank=p['weights_pertrank'])
    result={'status':'PASS','shape':list(counts.shape),'nnz':int(counts.nnz),'effect_replay_max_error':err,'source_selection_sha256':sha(SRC/'selection_D4_Stim8hr.npz'),'source_prediction_sha256':sha(SRC/'D4_Stim8hr/prediction.npz'),'counts_sha256':sha(out/'counts.npz'),'axes_sha256':sha(out/'axes.npz'),'seconds':time.time()-t,'D1_D2_Stim_expression_read':False}
    (out/'PREPARATION.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
