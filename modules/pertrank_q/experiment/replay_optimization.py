#!/usr/bin/env python3
"""Rebuild predecessor and updated predictions, rescore, compare all results.

All writes of generated predictions and scores occur in a temporary directory.
The original results are unchanged. Only this verification report and log are
written to --output-root. No training, model selection, or parameter search.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def numbers(x,key=''):
    if isinstance(x,bool):return {}
    if isinstance(x,(int,float)):return {key:x}
    if isinstance(x,dict):return {kk:vv for k,v in x.items() for kk,vv in numbers(v,key+'/'+k).items()}
    if isinstance(x,list):return {kk:vv for i,v in enumerate(x) for kk,vv in numbers(v,key+'/'+str(i)).items()}
    return {}
def copy_assets(src,dest,subdirs):
    dest.mkdir()
    for p in src.iterdir():
        if p.is_file() and p.suffix in ('.py','.json','.txt','.md'):shutil.copy2(p,dest/p.name)
    for name in subdirs:
        if (src/name).is_dir():shutil.copytree(src/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (dest/'predictions').mkdir()
    for p in (src/'predictions').glob('*LOCK.json'):shutil.copy2(p,dest/'predictions'/p.name)

def main(base,output):
    base=Path(base).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    rp=out/'REPLAY_OPTIMIZATION_CHECK.json';lp=out/'replay_optimization.log'
    if rp.exists() or lp.exists():raise RuntimeError('Existing verification preserved; choose an empty output directory')
    result=dict(passed=False,training_performed=False,model_selection_performed=False,formula_search_performed=False)
    logs=[];env=os.environ.copy()
    for runtime in (ROOT/'.runtime',ROOT.parent/'pertrank_neural_rank_20261115'/'.runtime'):
        if runtime.is_dir():env['PYTHONPATH']=str(runtime)+os.pathsep+env.get('PYTHONPATH','');break
    try:
        expected=read(ROOT/'results/SUMMARY.json')
        with tempfile.TemporaryDirectory(prefix='pertrank_b_optimization_replay_') as td:
            temp=Path(td);bp=temp/'baseline_B';np=temp/'update'
            copy_assets(base,bp,('inputs','fits','historical','selections'))
            copy_assets(ROOT,np,('fits','selections','audit'))
            def run(cmd):
                p=subprocess.run(cmd,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
                logs.append('$ '+' '.join(map(str,cmd))+'\n'+p.stdout)
            run([sys.executable,str(bp/'rebuild_boundary.py')])
            result['baseline_rebuild']=read(bp/'REBUILD_CHECK.json');assert result['baseline_rebuild']['passed']
            run([sys.executable,str(np/'rebuild_optimization.py'),'--base-root',str(bp)])
            result['updated_rebuild']=read(np/'REBUILD_OPTIMIZATION_CHECK.json');assert result['updated_rebuild']['passed']
            run([sys.executable,str(np/'score_optimization.py'),'--base-root',str(bp),'--output-dir',str(np/'results_replay')])
            actual=read(np/'results_replay/SUMMARY.json')
            a=numbers(actual);e=numbers(expected);assert a.keys()==e.keys(),'Numeric field names differ'
            diff={k:abs(a[k]-v) for k,v in e.items()};assert max(diff.values(),default=0)==0,('Numeric results differ',diff)
            ignored={'scoring_lock_sha256'}
            assert {k:v for k,v in actual.items() if k not in ignored}=={k:v for k,v in expected.items() if k not in ignored},'Result content differs'
            arrays={}
            for path,h in expected['files'].items():
                assert sha(np/'results_replay'/path)==h,('Scored array hash differs',path)
                arrays[path]=h
            result.update(numeric_fields_compared=len(e),maximum_numeric_difference=max(diff.values(),default=0),
                all_reported_content_matches_except_scoring_timestamp_hash=True,scored_array_hashes=arrays,
                original_summary_sha256=sha(ROOT/'results/SUMMARY.json'),replayed_summary_sha256=sha(np/'results_replay/SUMMARY.json'),
                primary_gates=actual['primary_gates'])
        result.update(passed=True,temporary_workspace_removed=True,temporary_prediction_and_scoring_arrays_removed=True,
                      replay_script_sha256=sha(__file__),rebuild_script_sha256=sha(ROOT/'rebuild_optimization.py'))
    except Exception as error:
        result['error']=repr(error)
        if isinstance(error,subprocess.CalledProcessError):logs.append(error.stdout or '')
        dump(rp,result);lp.write_text('\n'.join(logs));raise
    dump(rp,result);lp.write_text('\n'.join(logs))
    print(json.dumps(dict(passed=result['passed'],numeric_fields_compared=result['numeric_fields_compared'],maximum_numeric_difference=result['maximum_numeric_difference'],predictions_rebuilt=len(result['updated_rebuild']['tasks']),scored_arrays_checked=len(result['scored_array_hashes']),report=str(rp)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-root',type=Path,default=ROOT/'baseline_B')
    p.add_argument('--output-root',type=Path,default=Path('.'))
    a=p.parse_args();main(a.base_root,a.output_root)
