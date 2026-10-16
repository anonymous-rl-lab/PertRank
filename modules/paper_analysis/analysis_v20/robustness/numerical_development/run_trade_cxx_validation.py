from pathlib import Path
import numpy as np,json,time
import trade_cxx_validation_runner as tr
base=Path('/workspace/scratch/6cdbaefbed9f/stim_run_20261011/input/main')
p=Path(__file__).resolve().parent
idx=[0,1,8]; z=[dict(np.load(base/f'target_{i:04d}.npz')) for i in idx]
t=time.time();r=tr.fit_trade_batch(np.stack([x['log_y'] for x in z]),np.stack([x['trade_se'] for x in z]),[str(x['target_id']) for x in z],z[0]['gene_ids_keep'],p/'trade_smoke_cxx_checked')
print(json.dumps({'seconds':time.time()-t,'weights':r['weights'].tolist()}),flush=True)
