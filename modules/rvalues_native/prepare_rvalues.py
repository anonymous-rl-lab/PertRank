#!/usr/bin/env python3
"""Restore and verify all native-rvalues fitting materials without precision changes."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent

def digest(data):
    return hashlib.sha256(data).hexdigest()

def safe_relative(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or p.as_posix() != name:
        raise ValueError('Unsafe stored path: ' + name)
    return p

def required_shared(relative):
    path = ROOT.parent/relative
    if not path.is_file():
        raise ValueError('Shared input is not materialized: ' + relative +
                         '. Run the package-root command: python reproduce.py --only rvalues_native --output-dir /new/output')
    return path

def shared_effect(spec, blobs):
    encoding = spec['encoding']
    if encoding == 'shared_external_effect':
        path = required_shared('fixed_tasks/supplements/20261014_upgrade/native_rna/legacy/' + spec['donor'] + '_Rest/predictions_locked.npz')
        with np.load(path,allow_pickle=False) as z:
            return z['raw_effect'][None].tobytes()
    if encoding == 'shared_rest_effect':
        path = required_shared('fixed_tasks/data/D4/ranking.npz')
        with np.load(path,allow_pickle=False) as sp:
            shape = (int(sp['n_groups']),int(sp['n_queries']),int(sp['n_candidates']))
            mask = np.zeros((shape[0]*shape[1],shape[2]),bool)
            values = np.zeros_like(mask,dtype='<f8')
            rows = np.repeat(np.arange(mask.shape[0]),np.diff(sp['Y_offsets']))
            mask[rows,sp['Y_ids']] = True
            values[rows,sp['Y_ids']] = sp['Y_values']
            qi = {x:i for i,x in enumerate(sp['query_ids'])}
            ti = {x:i for i,x in enumerate(sp['target_ids'])}
            selq = [qi[x] for x in spec['query_ids']]
            selt = [ti[x] for x in spec['target_ids']]
            mask = mask.reshape(shape)[:,selq,:][:,:,selt]
            values = values.reshape(shape)[:,selq,:][:,:,selt]
        if mask.sum() != spec['shared_values']:
            raise ValueError('Unexpected Rest shared support')
        remainder = np.frombuffer(blobs[spec['remaining_data']],dtype=np.uint8).reshape(8,-1).T.tobytes()
        values[~mask] = np.frombuffer(remainder,dtype='<f8')
        return values.tobytes()
    if encoding == 'shared_stim_effect':
        wanted = {'trade_effects_missing/array.npy','trade_other/query_gene_ids.npy','trade_other/target_ids.npy'}
        data = {}
        with tarfile.open(required_shared('paper_analysis/array_payload.tar.xz'),'r:xz') as archive:
            for member in archive:
                if member.name in wanted:
                    data[member.name] = np.load(io.BytesIO(archive.extractfile(member).read()),allow_pickle=False)
        if set(data) != wanted:
            raise ValueError('Incomplete shared Stim effect inputs')
        with np.load(required_shared('fixed_tasks/supplements/20261011_stim/data/main_support.npz'),allow_pickle=False) as sp:
            queries = data['trade_other/query_gene_ids.npy']
            targets = data['trade_other/target_ids.npy']
            shape = (5,len(queries),len(targets))
            mask = np.zeros((shape[0]*shape[1],shape[2]),bool)
            rows = np.repeat(np.arange(mask.shape[0]),np.diff(sp['Y_offsets']))
            mask[rows,sp['Y_ids']] = True
            values = np.empty(mask.shape,dtype='<f8')
            values[mask] = sp['Y_values']
            values[~mask] = data['trade_effects_missing/array.npy']
        qi = {x:i for i,x in enumerate(queries)}
        ti = {x:i for i,x in enumerate(targets)}
        values = values.reshape(shape)[:,[qi[x] for x in spec['query_ids']],:][:,:,[ti[x] for x in spec['target_ids']]]
        return values.tobytes()
    raise ValueError('Unsupported shared effect recipe')

def restore(out):
    index = json.loads((ROOT/'NATIVE_INDEX.json').read_text())
    if index.get('schema') != 1:
        raise ValueError('Unsupported native store schema')
    payload = ROOT/'native_data.tar.xz'
    if digest(payload.read_bytes()) != index['payload_sha256']:
        raise ValueError('Native store checksum mismatch')
    if out.exists():
        raise ValueError('Output must name a new directory')
    blobs = {}
    with tarfile.open(payload, 'r:xz') as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith('blobs/'):
                raise ValueError('Unexpected store member')
            key = member.name.removeprefix('blobs/')
            if key not in index['blobs'] or key in blobs:
                raise ValueError('Unexpected or duplicated store blob')
            raw = archive.extractfile(member).read()
            if len(raw) != index['blobs'][key]['bytes'] or digest(raw) != key:
                raise ValueError('Native blob checksum mismatch')
            blobs[key] = raw
    if set(blobs) != set(index['blobs']):
        raise ValueError('Incomplete native store')
    out.mkdir(parents=True)
    root = out/'native_rvalues'
    root.mkdir()
    restored = []
    for item in index['files']:
        path = root/safe_relative(item['path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        if item['kind'] == 'bytes':
            raw = blobs[item['blob']]
        elif item['kind'] == 'npz':
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
                for a in item['arrays']:
                    head = blobs[a['header']]
                    if a['encoding'].startswith('shared_'):
                        data = shared_effect(a, blobs)
                    elif a['encoding'] == 'dictionary':
                        table = np.frombuffer(blobs[a['dictionary']], dtype=f"V{a['itemsize']}")
                        codes = np.frombuffer(blobs[a['codes']], dtype=a['code_dtype'])
                        if len(codes) != a['count'] or (len(codes) and codes.max() >= len(table)):
                            raise ValueError('Invalid exact dictionary indices')
                        data = table[codes].tobytes()
                    elif a['encoding'] == 'shuffle':
                        data = np.frombuffer(blobs[a['data']], dtype=np.uint8).reshape(a['itemsize'], -1).T.tobytes()
                    elif a['encoding'] == 'bytes':
                        data = blobs[a['data']]
                    else:
                        raise ValueError('Unsupported array encoding')
                    npy = head + data
                    if len(npy) != a['bytes'] or digest(npy) != a['sha256']:
                        raise ValueError('Restored array checksum mismatch: ' + item['path'] + '/' + a['name'])
                    with z.open(a['name']+'.npy', 'w', force_zip64=True) as entry:
                        entry.write(npy)
            raw = buffer.getvalue()
        else:
            raise ValueError('Unsupported native file kind')
        if len(raw) != item['bytes'] or digest(raw) != item['sha256']:
            raise ValueError('Restored native file checksum mismatch: ' + item['path'])
        path.write_bytes(raw)
        restored.append({'path':item['path'],'sha256':item['sha256'],'bytes':item['bytes']})
    subprocess.run([sys.executable, '-B', str(root/'run.py'), '--mode', 'verify'], check=True)
    receipt = {'status':'PASS','files_restored':len(restored),'array_precision':'All original NPY bytes and all original file SHA256 values verified.',
               'native_fits_run':0,'payload_sha256':index['payload_sha256'],'files':restored}
    (out/'RESTORE_REPORT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return root

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True,help='New output directory outside the repository')
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    repository = ROOT.parents[1]
    if out == repository or repository in out.parents:
        parser.error('Output must be outside the repository')
    try:
        root = restore(out)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print('Native fitting package:', root)
    print('Install native dependency: R CMD INSTALL',root/'vendor/rvalues_0.7.1.tar.gz')
    print('Fit: python',root/'run.py','--mode fit --out /absolute/path/to/new_refit --workers 3')

if __name__ == '__main__':
    main()
