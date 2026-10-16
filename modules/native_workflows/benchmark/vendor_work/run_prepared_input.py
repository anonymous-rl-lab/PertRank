"""Run a prepared counts.npz/axes.npz shallow input through native FR."""
import argparse
from pathlib import Path
import json, fcntl, os, time
import numpy as np
import scipy.sparse as sparse
from fr_native_runner import run_fr, sha256

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--rank',type=int,default=20)
    p.add_argument('--spca-alpha',type=float,default=.1)
    p.add_argument('--lasso-alpha',type=float,default=.0001)
    p.add_argument('--spca-iter',type=int,default=50)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=20261012)
    a=p.parse_args(); root=Path(a.input_dir)
    # Shared resource mutex with raw-input export. The cgroup has 8 GiB RAM;
    # holding this for the full child process avoids overlapping dense phases.
    lock_path=Path(__file__).resolve().parent.parent/'heavy_memory.lock'
    memory_lock=open(lock_path,'a+')
    waited=time.time()
    print(json.dumps({'event':'waiting_for_heavy_memory_lock','pid':os.getpid(),'path':str(lock_path)}),flush=True)
    fcntl.flock(memory_lock.fileno(),fcntl.LOCK_EX)
    lock_wait=time.time()-waited
    print(json.dumps({'event':'acquired_heavy_memory_lock','pid':os.getpid(),'wait_seconds':lock_wait}),flush=True)
    x=sparse.load_npz(root/'counts.npz')
    z=np.load(root/'axes.npz',allow_pickle=False)
    result=run_fr(x,z['labels'],z['target_ids'],z['gene_ids'],a.out,
        rank=a.rank,spca_alpha=a.spca_alpha,lasso_alpha=a.lasso_alpha,
        spca_iter=a.spca_iter,seed=a.seed,threads=a.threads,cell_ids=z['cell_ids'],
        input_provenance={'input_directory':str(root.resolve()),
            'counts_sha256':sha256(root/'counts.npz'), 'axes_sha256':sha256(root/'axes.npz'),
            'memory_lock_path':str(lock_path),'memory_lock_wait_seconds':lock_wait})
    print(json.dumps(result))
    fcntl.flock(memory_lock.fileno(),fcntl.LOCK_UN)
    memory_lock.close()

if __name__=='__main__':
    main()
