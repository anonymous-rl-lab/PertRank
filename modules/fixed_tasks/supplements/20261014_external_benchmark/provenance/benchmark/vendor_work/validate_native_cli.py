"""Compare adapter against original CLI, intercepting only full-precision export."""
from pathlib import Path
import sys, json, random, runpy
HERE=Path(__file__).resolve().parent
adapter_directory=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'synthetic_smoke_03'
sys.path[:0]=[str(HERE/'deps'),str(HERE/'FR-Perturb')]
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata
import spams
from threadpoolctl import threadpool_limits
import FR_Perturb.utils as utils

z=np.load(HERE/'synthetic_input.npz')
x=sp.csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(z['shape']))
dat=anndata.AnnData(x)
dat.var_names=z['genes']; dat.obs_names=np.array([f'cell{i}' for i in range(x.shape[0])])
dat.write_h5ad(HERE/'synthetic_cli.h5ad')
labels=np.where(z['target_index']==-1,len(z['targets']),z['target_index'])
design=np.zeros((len(labels),len(z['targets'])+1),int);design[np.arange(len(labels)),labels]=1
pd.DataFrame(design,index=dat.obs_names,columns=np.concatenate([z['targets'],['__NTC__']])).to_csv(HERE/'synthetic_design.txt',sep='\t')
original_signif=utils.signif
def capture(v,n):
    v.to_csv(HERE/'synthetic_cli_fullprecision.tsv',sep='\t')
    return original_signif(v,n)
utils.signif=capture
old_train=spams.trainDL; old_project=spams.lasso
def train(*a,**kw):
    return old_train(*a,numThreads=2,**kw)
def project(*a,**kw):
    return old_project(*a,numThreads=2,**kw)
spams.trainDL=train;spams.lasso=project
np.random.seed(20261012); random.seed(20261012)
sys.argv=['run_FR_Perturb.py','--input-h5ad',str(HERE/'synthetic_cli.h5ad'),
    '--input-perturbation-matrix',str(HERE/'synthetic_design.txt'),
    '--control-perturbation-name','__NTC__','--out',str(HERE/'synthetic_official')]
with threadpool_limits(limits=2):
    runpy.run_path(str(HERE/'FR-Perturb/run_FR_Perturb.py'),run_name='__main__')
f=pd.read_csv(HERE/'synthetic_cli_fullprecision.tsv',sep='\t',index_col=0)
a=np.load(adapter_directory/'effects.npz')
exact=f[z['targets']].loc[z['genes']].values.T
maxdiff=float(np.max(np.abs(exact-a['effect'])))
result={'official_cli_to_adapter_max_absolute_error':maxdiff,
        'allclose_1e-12':bool(np.allclose(exact,a['effect'],rtol=1e-12,atol=1e-12)),
        'fullprecision_intercept_only':True,'spams_threads':2}
(HERE/'NATIVE_CLI_IDENTITY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
assert result['allclose_1e-12']
