#!/usr/bin/env python3
"""Replay both frozen B development stages from weights without training."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from replay_optimization import copy_assets,numbers

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main(base,output):
    base=Path(base).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    rp=out/'REPLAY_PAIR_CHECK.json';lp=out/'replay_pair.log'
    if rp.exists() or lp.exists():raise RuntimeError('Preserve existing verification; choose a new output directory')
    result=dict(passed=False,training_performed=False,model_selection_performed=False,post_score_retuning_performed=False)
    logs=[];env=os.environ.copy()
    for runtime in (ROOT.parent/'.runtime',ROOT.parent.parent/'pertrank_neural_rank_20261115'/'.runtime'):
        if runtime.is_dir():env['PYTHONPATH']=str(runtime)+os.pathsep+env.get('PYTHONPATH','');break
    try:
        expected=read(ROOT/'results/SUMMARY.json')
        with tempfile.TemporaryDirectory(prefix='pertrank_free_pair_replay_') as td:
            temp=Path(td);bp=temp/'baseline_B';parent=temp/'update';p2=parent/'epoch_pair_update'
            copy_assets(base,bp,('inputs','fits','historical','selections'))
            copy_assets(ROOT.parent,parent,('fits','selections','audit'))
            copy_assets(ROOT,p2,('fits','selections'))
            def run(cmd):
                p=subprocess.run(cmd,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
                logs.append('$ '+' '.join(map(str,cmd))+'\n'+p.stdout)
            run([sys.executable,str(bp/'rebuild_boundary.py')])
            result['baseline_rebuild']=read(bp/'REBUILD_CHECK.json')
            run([sys.executable,str(parent/'rebuild_optimization.py'),'--base-root',str(bp)])
            result['phase1_rebuild']=read(parent/'REBUILD_OPTIMIZATION_CHECK.json')
            run([sys.executable,str(p2/'rebuild_pair_update.py'),'--base-root',str(bp)])
            result['phase2_rebuild']=read(p2/'REBUILD_PAIR_CHECK.json')
            for k in ('baseline_rebuild','phase1_rebuild','phase2_rebuild'):assert result[k]['passed']
            run([sys.executable,str(p2/'score_pair_update.py'),'--base-root',str(bp),'--output-dir',str(p2/'results_replay')])
            actual=read(p2/'results_replay/SUMMARY.json');a=numbers(actual);e=numbers(expected)
            assert a.keys()==e.keys();diff={k:abs(a[k]-v) for k,v in e.items()}
            assert max(diff.values(),default=0)==0,('Numeric results differ',diff)
            ignored={'scoring_lock_sha256'}
            assert {k:v for k,v in actual.items() if k not in ignored}=={k:v for k,v in expected.items() if k not in ignored}
            arrays={}
            for path,h in expected['files'].items():
                assert sha(p2/'results_replay'/path)==h,('Scored array hash differs',path);arrays[path]=h
            result.update(numeric_fields_compared=len(e),maximum_numeric_difference=max(diff.values(),default=0),
                all_reported_content_matches_except_scoring_timestamp_hash=True,scored_array_hashes=arrays,
                original_summary_sha256=sha(ROOT/'results/SUMMARY.json'),primary_gates=actual['primary_gates'])
        result.update(passed=True,temporary_workspace_removed=True,
                      replay_script_sha256=sha(__file__),rebuild_script_sha256=sha(ROOT/'rebuild_pair_update.py'))
    except Exception as error:
        result['error']=repr(error)
        if isinstance(error,subprocess.CalledProcessError):logs.append(error.stdout or '')
        dump(rp,result);lp.write_text('\n'.join(logs));raise
    dump(rp,result);lp.write_text('\n'.join(logs))
    print(json.dumps(dict(passed=result['passed'],numeric_fields_compared=result['numeric_fields_compared'],maximum_numeric_difference=result['maximum_numeric_difference'],new_predictions_rebuilt=len(result['phase2_rebuild']['tasks']),scored_arrays_checked=len(result['scored_array_hashes']),report=str(rp)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-root',type=Path,default=ROOT.parent/'baseline_B')
    p.add_argument('--output-root',type=Path,default=Path('.'))
    a=p.parse_args();main(a.base_root,a.output_root)
