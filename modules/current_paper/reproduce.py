#!/usr/bin/env python3
"""Replay the current submission four-arm/guide additions and regenerate current display figures.

No refitting, raw-cell processing, altered statistical rules, or new experiment.
"""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys, time
import numpy as np
sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parent
REPOSITORY=ROOT.parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jread(p):return json.loads(p.read_text())
def write(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def nested_check(actual,expected,path='',errors=None):
    if errors is None:errors=[]
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and set(actual)==set(expected),path
        for k in expected:nested_check(actual[k],expected[k],path+'/'+str(k),errors)
    elif isinstance(expected,list):
        assert isinstance(actual,list) and len(actual)==len(expected),path
        for i,(a,b) in enumerate(zip(actual,expected)):nested_check(a,b,path+'/'+str(i),errors)
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        diff=abs(actual-expected);assert diff<=1e-11,(path,actual,expected,diff);errors.append(diff)
    else:assert actual==expected,(path,actual,expected)
    return max(errors,default=0.)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--skip-figures',action='store_true',help='Run numeric checks without regenerating display figures.')
    args=ap.parse_args();out=args.output_dir.expanduser().resolve()
    if out.exists() or out==REPOSITORY or REPOSITORY in out.parents:
        raise SystemExit('Use a new output directory outside the repository.')
    out.mkdir(parents=True);started=time.monotonic();work=out/'runtime'
    shutil.copytree(ROOT,work,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    reused=[]
    for item in jread(ROOT/'SHARED_ASSETS.json'):
        src=(REPOSITORY/item['source']).resolve();dst=(work/item['destination']).resolve()
        assert src.is_relative_to(REPOSITORY) and dst.is_relative_to(work)
        assert src.is_file() and sha(src)==item['sha256'],item['source']
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);reused.append(item['source'])
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(out/'matplotlib_cache'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    logs=out/'logs';logs.mkdir()
    def run(name,cmd,cwd=None):
        with (logs/(name+'.log')).open('w') as f:
            p=subprocess.run([sys.executable,'-B',*map(str,cmd)],stdout=f,stderr=subprocess.STDOUT,env=env,cwd=cwd)
        if p.returncode:raise RuntimeError(name+' failed; inspect '+str(logs/(name+'.log')))
        print(name+': PASS',flush=True)
    f=work/'four_arm'
    run('four_arm_baseline',[f/'run_replay.py','--out',out/'baseline'])
    run('four_arm_ranking',[f/'score_four_arm.py','--out',out/'four_arm'])
    run('four_arm_reference_sensitivity',[f/'score_reference_sensitivity.py','--predictions',out/'four_arm','--out',out/'reference_sensitivity'])
    actual=jread(out/'four_arm/summary.json');expected=jread(f/'expected/summary.json')
    comparisons={}
    for s in ['D3','D1','D2']:
        for key in ['means','contrasts','nomination_accounting']:
            comparisons[s+'/'+key]=nested_check(actual['sources'][s][key],expected['sources'][s][key])
    comparisons['equal_source']=nested_check(actual['equal_source_descriptive_mean'],expected['equal_source_descriptive_mean'])
    comparisons['reference_sensitivity']=nested_check(jread(out/'reference_sensitivity/reference_sensitivity.json'),jread(f/'expected/reference_sensitivity.json'))
    g=work/'guide';shallow=g/'prior/crossfit_pilot/inputs';pred=g/'predictions';metadata=g/'probe/metadata'
    common=['--inputs',g/'inputs','--shallow',shallow,'--metadata',metadata,'--predictions',pred,'--protocol',g/'PROTOCOL.json','--protocol-lock',g/'PROTOCOL_LOCK.json']
    run('guide_mechanism',[g/'diagnose.py',*common,'--out',out/'mechanism'])
    run('guide_list_accounting',[g/'account_fixed_lists.py','--probe-root',g/'probe','--prior',g/'prior','--predictions',pred,'--protocol',g/'PROTOCOL.json','--protocol-lock',g/'PROTOCOL_LOCK.json','--out',out/'accounting'])
    run('guide_independent_audit',[g/'audit/audit_mechanism.py',*common,'--results',out/'mechanism','--production-script',g/'diagnose.py','--out',out/'guide_audit'])
    assert jread(out/'guide_audit/INDEPENDENT_AUDIT.json')['status']=='PASS'
    guide_checks={}
    for rel,entry in jread(g/'EXPECTED_OUTPUTS.json').items():
        p=out/rel
        if entry['type']=='file':assert sha(p)==entry['sha256'],rel
        else:
            with np.load(p,allow_pickle=False) as z:
                assert set(z.files)==set(entry['arrays']),rel
                for key,meta in entry['arrays'].items():
                    a=z[key];assert a.dtype.str==meta['dtype'] and list(a.shape)==meta['shape'],(rel,key)
                    assert hashlib.sha256(a.tobytes()).hexdigest()==meta['sha256'],(rel,key)
        guide_checks[rel]='EXACT'
    rendered=[]
    if not args.skip_figures:
        d=work/'display';run('current_figure_rendering',[d/'figure_code/draw_main_figures.py'],d)
        run('supplement_figures_1_to_4',[d/'figure_code/plot_supplement.py'],d)
        run('supplement_figure_5',[d/'figure_code/plot_design_attenuation.py'],d)
        expected_figures=[f'fig{i}' for i in range(1,7)]+['figS1_refit_sensitivity','figS2_mixscale_adaptation','figS3_mechanism','figS4_cell_equivalence','figS5_design_attenuation','figS6_guide_response','figS7_query_refinement']
        for stem in expected_figures:
            for ext in (['pdf','png'] if stem.startswith(('figS1_','figS2_','figS3_','figS4_')) else ['pdf','png','svg']):
                p=d/'figures'/(stem+'.'+ext);assert p.is_file() and p.stat().st_size>0;p2=out/'figures'/p.name;p2.parent.mkdir(exist_ok=True);shutil.copy2(p,p2);rendered.append(p.name)
        # Rendering is explicitly separate from numerical experiment replay.
        comparisons['plotted_contrasts']= 'identical_table_text' if (d/'figure_code/plotted_new_contrasts.csv').read_text()==(ROOT/'display/figure_code/plotted_new_contrasts.csv').read_text() else 'changed'
        assert comparisons['plotted_contrasts']=='identical_table_text'
    report={'status':'PASS','paper_version':'current_submission','scope':'Four-arm candidate rankings, original paired intervals and existing reference-draw sensitivity; retained-cell guide diagnostics and fixed-list accounting; current-source figure rendering. No raw-cell preprocessing or model training.','four_arm_numeric_checks':comparisons,'guide_exact_checks':guide_checks,'guide_audit':'guide_audit/INDEPENDENT_AUDIT.json','shared_assets_verified':len(reused),'rendered_figures':rendered,'new_model_fits':0,'seconds':time.monotonic()-started}
    write(out/'REPRODUCTION_REPORT.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
