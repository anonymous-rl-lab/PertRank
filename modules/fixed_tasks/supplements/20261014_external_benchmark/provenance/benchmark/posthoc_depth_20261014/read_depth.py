#!/usr/bin/env python3
"""Frozen D3 depth input via unchanged audited32worker range transport."""
from pathlib import Path
import argparse,fcntl,json,sys,time
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'independent/execution'))
import extract_selected_rows as base
import extract_selected_rows_fast as fast

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','plan','smoke','extract']);a=p.parse_args()
 protocol=json.loads((HERE/'POSTHOC_DEPTH_PROTOCOL.json').read_text())
 assert base.sha(HERE/'selection.npz')==protocol['files_sha256']['selection.npz']
 assert base.sha(HERE/'selection.lock.json')==protocol['files_sha256']['selection.lock.json']
 assert (ROOT/'pooled_data/D4_Stim8hr/raw/raw_batches/input_group4/COMPLETE.json').exists(),'D4 pooled raw recovery must finish first'
 a.selection=str(HERE/'selection.npz');a.lock=str(HERE/'selection.lock.json');a.output=str(HERE/'raw');a.workers=32;a.shard_rows=512;a.gap=4096;a.max_pack_bytes=8<<20;a.max_raw_gb=6;a.roles=['NTC_input','A_input']
 base.fetched=fast.persistent_fetched
 with ExitStack()as stack:
  lock=stack.enter_context((ROOT/'pooled_data/GLOBAL_RANGE_READER.lock').open('a'));fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
  stack.enter_context(fast.persistent_pool(a.workers));r=fast.FastReader(a)
  assert r.identity['url'].endswith('/D3_Rest.assigned_guide.h5ad')
  r.prepare()
  if a.mode=='plan':r.cost()
  elif a.mode=='smoke':r.smoke()
  elif a.mode=='extract':r.extract(a.roles)
 print(json.dumps({'status':'DEPTH_READER_MODE_COMPLETE','mode':a.mode,'workers':32,'time':time.time()}),flush=True)
if __name__=='__main__':main()
