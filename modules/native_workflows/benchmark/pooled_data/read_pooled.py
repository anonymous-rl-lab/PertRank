#!/usr/bin/env python3
"""Use audited exact CSR transport for original pooled input rows only."""
from pathlib import Path
import sys, argparse, json, time, fcntl
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'independent/execution'))
import extract_selected_rows_fast as fast
import extract_selected_rows as base
import numpy as np

ROLES=('NTC_input','A_input','input_group1','input_group2','input_group3','input_group4')

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','plan','smoke','extract']);p.add_argument('--task',choices=['D3_Rest','D4_Rest','D4_Stim8hr'],required=True);p.add_argument('--roles',nargs='+',choices=ROLES);p.add_argument('--workers',type=int,default=64)
    a=p.parse_args();d=Path(__file__).parent/a.task
    if a.mode=='extract' and a.task.startswith('D4_'):
        subsequent=any(x.startswith('input_group') for x in (a.roles or []))
        prerequisite=None
        if a.task=='D4_Rest' and not subsequent:prerequisite=('D4_Stim8hr',0)
        elif a.task=='D4_Stim8hr' and subsequent:prerequisite=('D4_Rest',0)
        elif a.task=='D4_Rest' and subsequent:prerequisite=('D4_Stim8hr',4)
        if prerequisite:
            ready=ROOT/'inputs'/prerequisite[0]/f'rep{prerequisite[1]}'/'INPUT_AUDIT.json';started=time.monotonic()
            if not ready.exists():print(json.dumps({'phase':'WAIT_FOR_SEQUENTIAL_SOURCE','task':a.task,'prerequisite':str(ready.relative_to(ROOT))}),flush=True)
            while not ready.exists():
                if time.monotonic()-started>7200:raise TimeoutError('Sequential source predecessor has not completed')
                time.sleep(5)
            assert json.loads(ready.read_text())['status']=='EXACT_RAW_REPLAY_PASS'
    if a.mode=='extract' and a.task.startswith('D4_') and any(x.startswith('input_group') for x in (a.roles or [])):
        ready=ROOT/'inputs/D3_Rest/rep0/INPUT_AUDIT.json';start=time.monotonic()
        if not ready.exists():print(json.dumps({'phase':'WAIT_FOR_PRIORITY_D3_INPUT','task':a.task}),flush=True)
        while not ready.exists():
            if time.monotonic()-start>2400:raise TimeoutError('D3 priority input not ready within bounded wait')
            time.sleep(5)
        assert json.loads(ready.read_text())['status']=='EXACT_RAW_R20_REPLAY_PASS'
    a.selection=str(d/'selection.npz');a.lock=str(d/'selection.lock.json');a.output=str(d/'raw');a.shard_rows=512;a.gap=4096;a.max_pack_bytes=8<<20;a.max_raw_gb=6
    base.ROLES=ROLES;base.fetched=fast.persistent_fetched
    t=time.monotonic()
    with ExitStack() as stack:
        lock=stack.enter_context((ROOT/'pooled_data/GLOBAL_RANGE_READER.lock').open('a'))
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        stack.enter_context(fast.persistent_pool(a.workers))
        r=fast.FastReader(a);assert a.task in r.identity['url']
        if a.task=='D4_Stim8hr' and not (r.out/'selected_chunk_manifest.json').exists():
            specs=json.loads((r.out/'hdf_objects.json').read_text());ptr=np.load(r.out/'indptr.npy',mmap_mode='r')
            old=json.loads((ROOT/'independent/execution/D4_Stim8hr/raw/selected_chunk_manifest.json').read_text())
            assert old['input_identity']['source_identity']==r.identity
            chunks={};missing={}
            for kind in ('data','indices'):
                s=specs[kind];need=sorted({c for row in r.rows_all for c in range(int(ptr[row])//s['chunk_size']*s['chunk_size'],int(ptr[row+1]),s['chunk_size'])})
                chunks[kind]={str(c):old['chunks'][kind][str(c)] for c in need if str(c) in old['chunks'][kind]}
                missing[kind]=[c for c in need if str(c) not in chunks[kind]]
            active={k:specs[k] for k in missing if missing[k]}
            audit={'reused_chunk_coordinates':{k:len(v) for k,v in chunks.items()},'missing_chunk_coordinates':{k:len(v) for k,v in missing.items()},'structural_only':True}
            if active:
                got,extra=base.chunk_manifest(r.transport('index'),active,{k:missing[k] for k in active},a.workers)
                for k in got:chunks[k].update({str(c):v for c,v in got[k].items()})
                audit['missing_index_audit']=extra
            base.js(r.out/'selected_chunk_manifest.json',{'input_identity':r.input_identity,'chunks':chunks,'audit':audit})
        r.prepare()
        if a.mode=='plan':r.cost()
        elif a.mode=='smoke':r.smoke()
        elif a.mode=='extract':
            assert a.roles;r.extract(a.roles)
    print(json.dumps({'task':a.task,'mode':a.mode,'seconds':time.monotonic()-t}),flush=True)
if __name__=='__main__':main()
