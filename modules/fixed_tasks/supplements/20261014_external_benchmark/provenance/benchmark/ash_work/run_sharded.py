#!/usr/bin/env python3
"""Native ASH/TRADE row-independent fits in two fixed contiguous target shards."""
from pathlib import Path
import argparse,concurrent.futures,gc,hashlib,json,os,subprocess,time
import numpy as np

P=Path(__file__).resolve().parent
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--input',type=Path,required=True)
    pa.add_argument('--output',type=Path,required=True);pa.add_argument('--mode',choices=['ash','trade'],default='ash')
    pa.add_argument('--mixcompdist');pa.add_argument('--method');pa.add_argument('--shards',type=int,default=2)
    pa.add_argument('--seed',type=int,default=20261012);a=pa.parse_args()
    assert a.shards in (2,3) and not a.output.exists()
    t=time.monotonic();a.output.mkdir(parents=True)
    with np.load(a.input,allow_pickle=False) as z:
        effects=z['effects'];ses=z['ses'];ids=z['row_ids'];genes=z['genes']
    assert effects.shape==ses.shape and effects.shape==(len(ids),len(genes))
    indices=np.array_split(np.arange(len(ids)),a.shards);paths=[]
    for i,ix in enumerate(indices):
        f=a.output/f'shard_input_{i:02d}.npz'
        np.savez_compressed(f,effects=effects[ix],ses=ses[ix],row_ids=ids[ix],genes=genes);paths.append(f)
    del effects,ses;gc.collect()
    live_KB=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')/1024
    options=['--mode',a.mode,'--retry-mixem','--seed',str(a.seed)]
    if a.mixcompdist:options+=['--mixcompdist',a.mixcompdist]
    if a.method:options+=['--method',a.method]
    def run(item):
        i,f=item;d=a.output/f'shard_{i:02d}'
        cmd=['python',str(P/'run_posterior.py'),'--input',str(f),'--output',str(d),*options]
        with (a.output/f'shard_{i:02d}_stdout.log').open('w') as log:
            r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        return i,d,r.returncode
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.shards) as ex:
        result=list(ex.map(run,enumerate(paths)))
    status=[dict(shard=i,directory=str(d),returncode=rc)for i,d,rc in result]
    (a.output/'SHARD_PROCESS_RESULTS.json').write_text(json.dumps(status,indent=2))
    records=[];pieces=[];runtimes=[];execs=[]
    for i,d,rc in sorted(result):
        if not (d/'runtime.json').exists():raise RuntimeError(f'Shard {i} exited {rc} without a complete runtime record')
        rt=json.loads((d/'runtime.json').read_text());ee=json.loads((d/'execution.json').read_text());execs.append(ee);runtimes.append(rt)
        with np.load(d/'posterior.npz',allow_pickle=False) as z:
            assert np.array_equal(z['row_ids'],ids[indices[i]]) and np.array_equal(z['genes'],genes)
            pieces.append(z['posterior'])
        rows=[json.loads(line)for line in (d/'fit_rows.jsonl').read_text().splitlines()]
        assert len(rows)==len(indices[i])
        for j,r in enumerate(rows):
            assert r['row_id']==ids[indices[i][j]] and r['row']==j+1
            r['shard']=i;r['row_local']=r['row'];r['row']=int(indices[i][j])+1;records.append(r)
    assert len({json.dumps(r['config'],sort_keys=True)for r in runtimes})==1
    pred=np.concatenate(pieces);assert pred.shape==(len(ids),len(genes))
    np.savez_compressed(a.output/'posterior.npz',posterior=pred,row_ids=ids,genes=genes)
    (a.output/'fit_rows.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':'))for r in records)+'\n')
    failed=sum(r['failed_rows']for r in runtimes);wall=time.monotonic()-t
    rt=dict(runtimes[0]);rt.update(status='COMPLETE'if failed==0 else'INCOMPLETE_FAILED_FITS',
        n_rows=len(ids),failed_rows=failed,warnings=sum(r['warnings']for r in runtimes),seconds=wall,
        sharded=True,n_shards=a.shards,shard_rows=[len(ix)for ix in indices],
        parent_resident_during_shards_KB=live_KB,
        conservative_combined_fit_peak_RSS_KB=live_KB+sum(e['peak_R_child_RSS_KB']+e['parent_resident_before_R_KB']for e in execs),
        shard_runtime_files=[str(d/'runtime.json')for _,d,_ in sorted(result)])
    (a.output/'runtime.json').write_text(json.dumps(rt,indent=2))
    execution=dict(input_sha256=sha(a.input),posterior_npz_sha256=sha(a.output/'posterior.npz'),returncode=0 if failed==0 else 2,
        seconds=wall,runner_sha256=sha(P/'posterior_runner.R'),sharded_runner_sha256=sha(Path(__file__)),config=rt['config'],
        finite_rows=int(np.isfinite(pred).all(axis=1).sum()),shard_input_sha256=[sha(f)for f in paths],
        shard_execution_sha256=[sha(d/'execution.json')for _,d,_ in sorted(result)])
    (a.output/'execution.json').write_text(json.dumps(execution,indent=2));print(json.dumps(rt,indent=2),flush=True)
    if failed:raise SystemExit(2)

if __name__=='__main__':main()
