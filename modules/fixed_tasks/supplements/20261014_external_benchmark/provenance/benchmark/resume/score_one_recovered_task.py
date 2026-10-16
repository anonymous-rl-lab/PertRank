"""One completed, owner-certified FR task per process; no polling or refitting."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_record(record):
    fit = Path(record['fit_directory'])
    owner_path = Path(record['owner_completion'])
    owner = json.loads(owner_path.read_text())
    assert owner['owner_exit_code'] == 0
    assert owner['status'] == 'OWNER_EXECUTION_ENDED_INTEGRITY_VERIFIED'
    assert owner['task'] == record['task'] and owner['rep'] == record['rep']
    effect = fit / 'effects.npz'
    assert sha(effect) == owner['npz_files']['effects.npz']['sha256']
    with zipfile.ZipFile(effect) as z:
        assert z.testzip() is None
    assert sha(fit / 'fit_report.json') == owner['fit_report_sha256']
    assert sha(fit / 'NATIVE_SOLVER_OBJECTIVE_AUDIT.json') == owner['native_solver_objective_audit_sha256']
    return effect


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['D4_Stim8hr', 'D4_Rest', 'D4_Stim_independent'], required=True)
    p.add_argument('--arm', choices=['primary', 'sensitivity'], required=True)
    a = p.parse_args()
    mapping = json.loads((HERE / 'RECOVERED_OUTPUT_PATH_MAP.json').read_text())
    records = sorted([r for r in mapping['records'] if r['task'] == a.task and r['arm'] == a.arm], key=lambda r:r['rep'])
    count = 1 if a.task.endswith('independent') else 5
    assert [r['rep'] for r in records] == list(range(count))
    effects = [verify_record(r) for r in records]
    before = {str(f):sha(f) for f in effects}
    base = HERE / ('evaluation' if a.arm == 'primary' else 'evaluation_sensitivity')
    out = base / a.task
    assert not out.exists(), f'Refusing to overwrite existing output: {out}'
    base.mkdir(parents=True, exist_ok=True)
    args = ['evaluate', '--dataset', 'D4_Stim8hr' if count == 1 else a.task,
            '--kind', 'independent' if count == 1 else 'pooled',
            '--protocol', ROOT / 'AMENDMENT_03_FR_SEARCH_21.json',
            '--winner-lock', ROOT / 'FR_D3_WINNER_LOCK.json',
            '--fr-effects', *effects, '--out', out]
    if a.arm == 'sensitivity':
        primary = HERE / 'evaluation' / a.task / 'evaluation_report.json'
        assert json.loads(primary.read_text())['status'] == 'COMPLETE'
        args += ['--primary-report', primary]
    scorer = ROOT / 'evaluation' / ('evaluate_six_arm.py' if a.arm == 'primary' else 'evaluate_nscaled.py')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env['PYTHONPATH'] = str(ROOT / 'vendor_work/deps') + os.pathsep + env.get('PYTHONPATH', '')
    def call(cmd, log_name):
        with open(base / (a.task + '_' + log_name), 'w') as log:
            subprocess.run([sys.executable, *map(str, cmd)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with open(ROOT / 'heavy_memory.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        call([scorer, *args], 'score.log')
        call([ROOT / 'audit/audit_result_reports.py', '--reports', out / 'evaluation_report.json', '--out', out / 'audit'], 'audit.log')
        call([HERE / 'add_requested_contrasts.py', out / 'evaluation_report.json'], 'extra_contrasts.log')
    assert before == {str(f):sha(f) for f in effects}, 'Input effects changed during scoring'
    report = json.loads((out / 'evaluation_report.json').read_text())
    assert json.loads((out / 'audit/AUDIT_RESULT.json').read_text())['status'] == 'PASS'
    files = sorted(f for f in out.rglob('*') if f.is_file())
    pending = {'status':'AWAITING_SEPARATE_OWNER_EXIT_VERIFICATION', 'task':a.task, 'arm':a.arm,
               'inputs':before, 'outputs':{str(f.relative_to(out)):sha(f) for f in files}}
    (out / 'PENDING_OWNER_VERIFICATION.json').write_text(json.dumps(pending, indent=2)+'\n')
    print(json.dumps({'status':'SCORED_AUDITED_AWAITING_OWNER_EXIT', 'report':str(out / 'evaluation_report.json'),
                     'P25':{k:v['point'] for k,v in report['results']['25']['arms'].items()}}), flush=True)


if __name__ == '__main__':
    main()
