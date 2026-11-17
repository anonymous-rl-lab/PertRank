#!/usr/bin/env python3
"""Recompute the two completed PertRank studies from compact retained inputs."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent

def verify():
    record=json.loads((ROOT/'MANIFEST_SHA256.json').read_text())
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=ROOT/'MANIFEST_SHA256.json'}
    if actual!=set(record['files']):
        raise ValueError({'missing':sorted(set(record['files'])-actual),'extra':sorted(actual-set(record['files']))})
    for n,row in record['files'].items():
        rel=Path(n)
        if rel.is_absolute() or '..' in rel.parts or rel.as_posix()!=n:
            raise ValueError('Invalid manifest path: '+n)
        p=ROOT/n
        if p.is_symlink():raise ValueError('Unexpected symlink: '+n)
        if p.stat().st_size!=row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('File integrity mismatch: '+n)
    return len(actual)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['core','exploratory','all','verify'])
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    count=verify()
    if args.mode=='verify':
        print(json.dumps({'status':'PASS','files_verified':count}));return
    if args.out is None:ap.error('--out is required for a replay')
    out=args.out.resolve()
    if out==ROOT or ROOT in out.parents:ap.error('Use an output directory outside the delivered module')
    if out.exists():raise FileExistsError('Choose a new output directory: '+str(out))
    out.mkdir(parents=True)
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    commands=[]
    if args.mode in ['core','all']:
        commands.append(('rvalues',[ROOT/'rvalues/run.py','--out',out/'rvalues']))
    commands.append(('program',[ROOT/'program/run.py',args.mode,'--out',out/'program']))
    reports={}
    for name,command in commands:
        with (out/(name+'.log')).open('w') as log:
            p=subprocess.run([sys.executable,'-B',*map(str,command)],stdout=log,stderr=subprocess.STDOUT,env=env)
        if p.returncode:raise RuntimeError(f'{name} failed; see {out/(name+".log")}')
        report=out/name/('REPLAY_CHECK.json' if name=='rvalues' else 'REPORT.json')
        reports[name]=json.loads(report.read_text())
        if reports[name].get('status')!='PASS':raise ValueError(reports[name])
        print(name+' PASS',flush=True)
    summary={'status':'PASS','mode':args.mode,'files_verified':count,'model_refitting':False,
        'scope':'The fixed rvalues comparison and predefined-program studies, with exploratory direction adjustments kept separate. Baseline native-model fitting and raw-count recovery are outside this compact replay.',
        'reports':reports}
    (out/'REPLAY_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('PASS: '+str(out/'REPLAY_SUMMARY.json'))

if __name__=='__main__':main()
