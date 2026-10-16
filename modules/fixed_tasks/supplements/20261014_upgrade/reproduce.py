#!/usr/bin/env python3
"""Offline reproduction of the completed three-source RNA and interaction results."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args()
    if sys.version_info<(3,11) or not __debug__:p.error('Python >=3.11 without -O required; recorded environment is Python 3.12.')
    out=a.output_dir.resolve()
    if out.exists() or out.is_relative_to(ROOT):p.error('Choose a new output directory outside the supplement')
    manifest=json.loads((ROOT/'SUPPLEMENT_MANIFEST.json').read_text())
    for name,h in manifest['files'].items():
        path=ROOT/name
        with path.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=h:raise RuntimeError('Integrity mismatch: '+name)
    out.mkdir(parents=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.run([sys.executable,str(ROOT/'native_rna/code/reproduce_scores.py'),'--bundle',str(ROOT/'native_rna'),'--out',str(out/'native_rna')],check=True,env=env)
    subprocess.run([sys.executable,str(ROOT/'interactions/replay.py'),'--output-dir',str(out/'interactions')],check=True,env=env)
    report={'status':'UPGRADE_FIXED_TASK_REPRODUCTION_PASS','model_refits':0,'new_expression_downloads':0,'terminated_stim_executed':False,'scope':'Completed seven-arm independent-RNA comparison including reference sensitivity; geometry and depth interactions.'}
    (out/'SUPPLEMENT_REPRODUCTION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
