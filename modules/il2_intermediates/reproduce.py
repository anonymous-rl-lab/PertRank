#!/usr/bin/env python3
"""Replay fixed IL2 comparisons from archived guide readouts and MAGeCK outputs."""
from pathlib import Path
import argparse,json,subprocess,sys,os,time,shutil,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent

def compare(a,b,path='',checks=None):
    if checks is None:checks=[]
    if isinstance(a,dict):
        assert set(a)==set(b),path
        for key in a:compare(a[key],b[key],path+'/'+key,checks)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y)in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i),checks)
    elif isinstance(a,(int,float))and not isinstance(a,bool):
        assert np.isclose(a,b,rtol=1e-11,atol=1e-12),(path,a,b)
        checks.append(abs(a-b))
    else:assert a==b,(path,a,b)
    return checks

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();out=a.output_dir.resolve()
    if out==ROOT or ROOT in out.parents:raise ValueError('Output must be outside module')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);start=time.monotonic();work=out/'work';shutil.copytree(ROOT,work)
    report={'status':'RUNNING','reproduction_level':'readout_score_to_comparison','raw_counts_read':False,'guide_overlap_reclassified':False,'MAGeCK_refit':False,'bootstrap_draws':20000,'bootstrap_seed':20261011}
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    try:
        for script,extra in [('analyze_il2_intermediates.py',['--out',str(work/'results')]),('score_mageck.py',['--root',str(work),'--out',str(work/'results')])]:
            with (out/(script+'.log')).open('w')as log:subprocess.run([sys.executable,'-B',str(work/script),*extra],check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
        expected=json.loads((ROOT/'EXPECTED_RESULTS.json').read_text());primary=json.loads((work/'results/PRIMARY_RESULTS.json').read_text());secondary=json.loads((work/'results/MAGECK_SUPPORT_RESULTS.json').read_text())
        checks=compare(expected['primary_scorecards'],primary['scorecards'])+compare(expected['mageck_scorecards'],secondary['scorecards'])
        report.update(status='PASS',scientific_numeric_values_checked=len(checks),maximum_absolute_difference=max(checks),primary_scorecards=primary['scorecards'],mageck_scorecards=secondary['scorecards'])
    except Exception as exc:
        report.update(status='FAIL',error=repr(exc));raise
    finally:
        report['seconds']=time.monotonic()-start;(out/'REPRODUCTION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k]for k in ['status','seconds','scientific_numeric_values_checked','maximum_absolute_difference']}))
if __name__=='__main__':main()
