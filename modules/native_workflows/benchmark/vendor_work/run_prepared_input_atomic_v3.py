"""Unmodified native audit-v2 inference with atomic NPZ and input stability guards."""
from pathlib import Path
import argparse,fcntl,json,os,time
import numpy as np
import scipy.sparse as sparse
from fr_native_runner_audit_v2 import run_fr,sha256
import atomic_npz

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',required=True);p.add_argument('--out',required=True)
    p.add_argument('--rank',type=int,default=20);p.add_argument('--spca-alpha',type=float,default=.1)
    p.add_argument('--lasso-alpha',type=float,default=.0001);p.add_argument('--spca-iter',type=int,default=50)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=20261012)
    a=p.parse_args();source=Path(a.input_dir);out=Path(a.out);here=Path(__file__).resolve().parent
    lock_path=here.parent/'heavy_memory.lock'
    with lock_path.open('a') as lock:
        began=time.time();print(json.dumps({'event':'waiting_for_heavy_memory_lock','path':str(lock_path)}),flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        waited=time.time()-began;print(json.dumps({'event':'acquired_heavy_memory_lock','wait_seconds':waited}),flush=True)
        before={n:atomic_npz.validate(source/n) for n in ['counts.npz','axes.npz']}
        marker=source/('INPUT_AUDIT.json' if (source/'INPUT_AUDIT.json').exists() else 'PREPARATION.json')
        if marker.exists():
            audit=json.loads(marker.read_text());expected=audit.get('files_sha256',{})
            if not expected:expected={'counts.npz':audit.get('counts_sha256'),'axes.npz':audit.get('axes_sha256')}
            for n,h in expected.items():
                if n in before and h is not None and before[n]!=h:raise AssertionError('Input differs from ready-marker SHA: '+n)
        x=sparse.load_npz(source/'counts.npz')
        z=np.load(source/'axes.npz',allow_pickle=False)
        labels=z['labels'];targets=z['target_ids'];genes=z['gene_ids'];cells=z['cell_ids']
        assert before=={n:sha256(source/n) for n in before},'Input changed while loading'
        atomic_npz.install()
        result=run_fr(x,labels,targets,genes,out,rank=a.rank,spca_alpha=a.spca_alpha,lasso_alpha=a.lasso_alpha,
                      spca_iter=a.spca_iter,seed=a.seed,threads=a.threads,cell_ids=cells,
                      input_provenance={'input_directory':str(source.resolve()),'counts_sha256':before['counts.npz'],
                          'axes_sha256':before['axes.npz'],'memory_lock_path':str(lock_path),'memory_lock_wait_seconds':waited,
                          'serialization_wrapper_sha256':sha256(__file__),'atomic_npz_helper_sha256':sha256(here/'atomic_npz.py'),
                          'scientific_runner_unchanged':'fr_native_runner_audit_v2.py'})
        after={n:sha256(source/n) for n in before}
        state={'status':'PASS' if before==after else 'FAILED_INPUT_CHANGED','before':before,'after':after,
               'serialization_events':atomic_npz.EVENTS,'wrapper_sha256':sha256(__file__),'helper_sha256':sha256(here/'atomic_npz.py')}
        (out/'ATOMIC_SERIALIZATION_INPUT_STABILITY.json').write_text(json.dumps(state,indent=2)+'\n')
        if before!=after:raise AssertionError('Input changed during native fitting')
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()
