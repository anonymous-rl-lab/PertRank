#!/usr/bin/env python3
"""Recompute archived R20 means, contrasts and conditional intervals from query scores."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
import hashlib
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
DONORS = ('D3_Rest','D1_Rest','D2_Rest')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def compare(a, b, path=''):
    if isinstance(a, dict):
        if set(a) != set(b): raise AssertionError('Keys differ: '+path)
        for k in a: compare(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        if len(a) != len(b): raise AssertionError('Lengths differ: '+path)
        for i,(x,y) in enumerate(zip(a,b)): compare(x,y,path+'/'+str(i))
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        if not math.isclose(a,b,rel_tol=0,abs_tol=1e-12):
            raise AssertionError(f'Numeric mismatch {path}: {a} != {b}')
    elif a != b:
        raise AssertionError('Value differs: '+path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    out=parser.parse_args().output_dir.resolve()
    if out==ROOT or ROOT in out.parents: raise SystemExit('Output must be outside the module')
    if out.exists() and any(out.iterdir()): raise SystemExit('Output directory must be new or empty')
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for f in manifest['files']:
        if sha(ROOT/f['path']) != f['sha256']: raise AssertionError('Payload hash mismatch: '+f['path'])
    provenance=json.loads((ROOT/'provenance/EXPORT_PROVENANCE.json').read_text())
    code=ROOT/'original_code/evaluate_real_cells.py'
    if sha(code)!=provenance['statistical_code_sha256']: raise AssertionError('Frozen code changed')
    spec=importlib.util.spec_from_file_location('r20_frozen',code)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.BOOTSTRAPS==2000 and module.BOOTSTRAP_SEED==20261006
    rebuilt=[];checks={};maxdiff=0.0
    for donor in DONORS:
        score=np.load(ROOT/'data'/f'{donor}_query_scores.npz',allow_pickle=False)
        definitions=np.load(ROOT/'data'/f'{donor}_query_definitions.npz',allow_pickle=False)
        if list(definitions['arm_names'])!=list(module.ARMS): raise AssertionError('Method order changed')
        old=json.loads((ROOT/'expected'/f'{donor}_results.json').read_text())
        result={'donor':donor,'internal':{},'external':{}}
        checks[donor]={}
        for task,k,reps in [('internal_responsive',25,5),('internal_terciles',25,5),('external',10,2)]:
            s=score[task+'_scores'];blocks=definitions[task+'_blocks'];ids=definitions[task+'_query_ids']
            if s.shape!=(3,reps,len(blocks),2) or len(ids)!=len(blocks): raise AssertionError('Query/repetition axes differ')
            if not np.isfinite(s).all(): raise AssertionError('Nonfinite query scores')
            got=module.summarize_scores(s,blocks,k)
            expected=old['external'] if task=='external' else old['internal'][task.removeprefix('internal_')]
            target={key:expected[key] for key in ('repetitions','repetition_average')}
            compare(target,got,donor+'/'+task)
            if task=='external': result['external']=got
            else: result['internal'][task.removeprefix('internal_')]=got
            checks[donor][task]={'status':'PASS','input_shape':list(s.shape),'query_blocks':len(np.unique(blocks)),
                                  'means_contrasts_intervals':'all historical fields matched','absolute_tolerance':1e-12}
        path=out/(donor+'_results.json');path.write_text(json.dumps(result,indent=2)+'\n');rebuilt.append(path)
    class Args: pass
    args=Args();args.results=rebuilt;args.out=out/'descriptive_summary.json'
    module.summary(args)
    compare(json.loads((ROOT/'expected/descriptive_summary.json').read_text()),
            json.loads(args.out.read_text()),'cross_source_summary')
    receipt={'status':'PASS','module':'r20_intermediates','replay_level':'per_query_scores_to_means_contrasts_intervals',
             'new_experiment':False,'predictions_rebuilt':False,'weights_refitted':False,'raw_cells_used':False,
             'historical_D1_query_panel_retained':True,'bootstrap_draws':2000,'seed':20261006,
             'checks':checks,'cross_source_descriptive_summary':'PASS',
             'scope':'Three source donors, five responsive and five tercile within-screen input groups and two external RNA input groups each; conditional query-block intervals. No pooled donor CI.',
             'not_replayed':'Cell selection, moment estimation, weight fitting, ranking, list membership and nomination redistribution.',
             'seconds':time.monotonic()-start}
    (out/'REPRODUCTION_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':'PASS','seconds':receipt['seconds'],'replay_level':receipt['replay_level']},indent=2))

if __name__=='__main__': main()
