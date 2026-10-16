#!/usr/bin/env python3
"""Read D3 Rest obs and matrix layout only, verify historical source identity."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import importlib.util,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REC=ROOT.parents[1]
VENDOR=ROOT/'vendor/metadata_source.py'
s=importlib.util.spec_from_file_location('metadata_source',VENDOR)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
source=ROOT/'frozen_assets/ntc_original/selection_check.json'
identity=json.loads(source.read_text())['source_identity']
out=ROOT/'metadata';out.mkdir(exist_ok=True)
t=time.time()
z,r=m.metadata(identity['url'],out)
assert r['etag']==identity['etag'] and r['source_size']==identity['size']
print(json.dumps({'status':'METADATA_PASS','source_identity':identity,'seconds':time.time()-t,'fields':z.files}),flush=True)
