#!/usr/bin/env python3
"""Materialize losslessly compacted arrays, then run unchanged scientific replay."""
from pathlib import Path
import hashlib,io,json,os,shutil,subprocess,sys,tarfile
import numpy as np
ROOT=Path(__file__).resolve().parent

def materialize(dest):
    shutil.copytree(ROOT,dest,ignore=shutil.ignore_patterns('array_payload.tar.xz','__pycache__','*.pyc'))
    data={}
    with tarfile.open(ROOT/'array_payload.tar.xz','r:xz') as tar:
        for m in tar:
            group,key=m.name.split('/');data.setdefault(group,{})[key[:-4]]=np.load(io.BytesIO(tar.extractfile(m).read()),allow_pickle=False)
    layout=json.loads((ROOT/'COMPACT_LAYOUT.json').read_text())
    core=Path(os.environ.get('PERTRANK_FIXED_TASKS',str(ROOT.parent/'fixed_tasks')))
    trade=None
    for rel,record in layout['records'].items():
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
        kind=record['kind']
        if kind=='core':
            p=core/record['path'];assert p.is_file(),f'Required original v1.4 fixed_tasks input missing: {p}'
            assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
            shutil.copyfile(p,target);continue
        if kind=='trade':
            p=core/'supplements/20261011_stim/data/main_support.npz'
            with np.load(p,allow_pickle=False) as sp:
                z=dict(data['trade_other'])
                for field,prefix,shape in [('effects','Y',(2970,561)),('reference','truth',(594,561))]:
                    mask=np.zeros(shape,bool);idx=np.repeat(np.arange(shape[0]),np.diff(sp[prefix+'_offsets']));mask[idx,sp[prefix+'_ids']]=True
                    a=np.empty(shape);a[mask]=sp[prefix+'_values'];a[~mask]=data['trade_'+field+'_missing']['array']
                    z[field]=a.reshape(5,594,561).transpose(0,2,1) if field=='effects' else a.T
            trade=z
        elif kind=='biology':
            z=dict(data['biology']);ranks=lambda v:np.unique(v,return_inverse=True)[1].astype(np.int16)
            codes=np.empty((3,5,594,561),np.int16)
            for a,arm in enumerate(['raw','Euclidean','PertRank']):
                for r in range(5):
                    for qi in range(594):codes[a,r,qi]=ranks(trade['effects'][r,:,qi]*trade['weight_'+arm][r])
            z['rank_codes']=codes;z['reference_rank_codes']=np.stack([ranks(trade['reference'][:,i]) for i in range(594)])
        elif kind=='transfer':
            z=dict(data['transfer']);ix=z.pop('trade_index');missing=ix<0
            eff=np.empty((5,561,300));ref=np.empty((561,300));eff[:,:,missing]=z.pop('effects_missing');ref[:,missing]=z.pop('truth_missing')
            eff[:,:,~missing]=trade['effects'][:,:,ix[~missing]];ref[:,~missing]=trade['reference'][:,ix[~missing]];z.update(effects=eff,truth=ref)
        else:z=data[record['key']]
        assert set(z)==set(record['arrays']),rel
        for key,a in z.items():assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==record['arrays'][key],(rel,key,'lossless restoration failed')
        z={key:np.array(z[key],copy=True,order=record['array_orders'][key]) for key in record['arrays']}
        np.savez_compressed(target,**z)
    return {'status':'PASS','restored_array_files':sum(r['kind']!='core' for r in layout['records'].values()),'all_array_bytes_match':True,'precision_changed':False,'core_files_verified':sum(r['kind']=='core' for r in layout['records'].values())}

def main():
    args=sys.argv[1:];flag='--out' if json.loads((ROOT/'MODULE.json').read_text())['name']=='paper_analysis' else '--output-dir'
    if flag not in args:
        subprocess.run([sys.executable,str(ROOT/'replay_original.py'),*args],check=True);return
    i=args.index(flag);out=Path(args[i+1]).resolve()
    if out.exists():raise FileExistsError('Choose a new output directory: '+str(out))
    out.mkdir(parents=True);restored=out/'restored_inputs';check=materialize(restored)
    (out/'LOSSLESS_RESTORATION.json').write_text(json.dumps(check,indent=2)+'\n')
    args[i+1]=str(out/'replay')
    result=subprocess.run([sys.executable,str(restored/'replay_original.py'),*args])
    receipt=out/'replay/REPLAY_REPORT.json'
    if receipt.exists():
        report=json.loads(receipt.read_text());report['lossless_restoration']=check;report['replay_output_subdirectory']='replay';(out/'REPLAY_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    sys.exit(result.returncode)
if __name__=='__main__':main()
