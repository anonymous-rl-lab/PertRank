"""Replay frozen scoring into new recovery directories; do not refit or relock."""
from pathlib import Path
import fcntl, json, os, subprocess, sys, time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENV = os.environ.copy()
ENV.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
ENV['PYTHONPATH'] = str(ROOT/'vendor_work/deps') + os.pathsep + ENV.get('PYTHONPATH','')
PROTO = ROOT/'AMENDMENT_03_FR_SEARCH_21.json'
LOCK = ROOT/'FR_D3_WINNER_LOCK.json'
WINNER = json.loads(LOCK.read_text())['config_id']

def call(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(ROOT/'heavy_memory.lock', 'a') as lock, open(log, 'w') as output:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run([sys.executable, *map(str,cmd)], env=ENV, stdout=output, stderr=subprocess.STDOUT, check=True)

def complete(path):
    return path.exists() and json.loads(path.read_text()).get('status') == 'COMPLETE'

tasks = ['D4_Rest','D4_Stim8hr','D4_Stim_independent']
pending = [(arm, task) for arm in ['primary','sensitivity'] for task in tasks]
start=time.time()
while pending:
    progressed=False
    for arm,task in list(pending):
        out=HERE/('evaluation' if arm=='primary' else 'evaluation_sensitivity')/task
        report=out/'evaluation_report.json'
        if complete(report):
            pending.remove((arm,task)); continue
        reps=range(1 if task.endswith('independent') else 5)
        fits=[HERE/('fr_outputs' if arm=='primary' else 'fr_outputs_sensitivity')/task/(WINNER+'_rep'+str(i)) for i in reps]
        if arm=='primary' and task=='D4_Rest':
            fits[2]=HERE/'replay_retry1/fr_outputs/D4_Rest'/(WINNER+'_rep2')
        if not all(complete(f/'fit_report.json') and (f/'NATIVE_SOLVER_OBJECTIVE_AUDIT.json').exists() for f in fits): continue
        dataset='D4_Stim8hr' if task.endswith('independent') else task
        args=['evaluate','--dataset',dataset,'--kind','independent' if task.endswith('independent') else 'pooled',
              '--protocol',PROTO,'--winner-lock',LOCK,'--fr-effects',*[f/'effects.npz' for f in fits],'--out',out]
        if arm=='sensitivity':
            primary=HERE/'evaluation'/task/'evaluation_report.json'
            if not complete(primary):continue
            args+=['--primary-report',primary]
        scorer=ROOT/'evaluation'/('evaluate_six_arm.py' if arm=='primary' else 'evaluate_nscaled.py')
        call([scorer,*args],out.parent/(task+'_score.log'))
        call([ROOT/'audit/audit_result_reports.py','--reports',report,'--out',out/'audit'],out.parent/(task+'_audit.log'))
        pending.remove((arm,task)); progressed=True
        r=json.loads(report.read_text())
        print(json.dumps({'status':'SCORED_AND_AUDITED','arm':arm,'task':task,'P25':{k:v['point'] for k,v in r['results']['25']['arms'].items()}}),flush=True)
    if time.time()-start>21600:raise TimeoutError('Recovery scoring inputs incomplete after six hours')
    if not progressed:time.sleep(10)

for arm in ['primary','sensitivity']:
    base=HERE/('evaluation' if arm=='primary' else 'evaluation_sensitivity')
    reports=[base/t/'evaluation_report.json' for t in tasks[:2]]
    out=base/'equal_state'
    if not complete(out/'equal_state_report.json'):
        scorer=ROOT/'evaluation'/('evaluate_six_arm.py' if arm=='primary' else 'evaluate_nscaled.py')
        call([scorer,'combine','--reports',*reports,'--protocol',PROTO,'--winner-lock',LOCK,'--out',out],base/'combine.log')
        call([ROOT/'audit/audit_result_reports.py','--reports',*reports,'--combined',out/'equal_state_report.json','--out',base/'combined_audit'],base/'combined_audit.log')
    print(json.dumps({'status':'COMBINED_AND_AUDITED','arm':arm,'report':str(out/'equal_state_report.json')}),flush=True)
