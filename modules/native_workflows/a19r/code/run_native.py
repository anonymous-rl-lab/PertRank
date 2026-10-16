#!/usr/bin/env python3
"""Run one locked native external-reference baseline, never reading its reference.

Input lock is created by the adapter after original-task identity verification.
Every native attempt has a unique directory; a failed fit is not silently retried.
Run --verify only in a separate tool invocation after the owner returns exit zero.
"""
from pathlib import Path
import argparse, datetime, fcntl, hashlib, json, os, signal, subprocess, sys, time, zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write(p, x):
    p = Path(p)
    payload = json.dumps(x, indent=2, allow_nan=False) + '\n'
    if p.exists():
        assert p.read_text() == payload, str(p)
        return
    with p.open('x') as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())

def crc(p):
    before = sha(p)
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
    assert sha(p) == before
    return {'sha256': before, 'CRC': 'PASS', 'bytes': Path(p).stat().st_size}

def process_rss(pid):
    """Sum resident memory of a process and its live descendants."""
    stats = {}
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():
            continue
        try:
            s = (d/'stat').read_text().rsplit(')', 1)[1].split()
            stats[int(d.name)] = (int(s[1]), int(s[21])*os.sysconf('SC_PAGE_SIZE'))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    related = {pid}
    while True:
        new = {p for p, (parent, _) in stats.items() if parent in related}
        if new <= related:
            break
        related |= new
    return sum(stats[p][1] for p in related if p in stats)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-lock', required=True, type=Path)
    ap.add_argument('--method', required=True, choices=['FR', 'ASH'])
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--timeout', type=int, default=600)
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--owner-session-id')
    a = ap.parse_args()
    lock = json.loads(a.input_lock.read_text())
    assert lock['status'] == 'READY_EXACT_A19R_INPUT'
    assert lock['donor'] in ['D3_Rest','D1_Rest','D2_Rest']
    assert 0 < a.timeout <= 1200
    required = lock['methods'][a.method]
    for name, record in required['files'].items():
        assert sha(record['path']) == record['sha256'], name
    if a.verify:
        assert a.owner_session_id
        r = json.loads((a.out/'RUN_RECORD.json').read_text())
        assert r['returncode'] == 0 and r['input_lock_sha256'] == sha(a.input_lock)
        native = a.out/'native'
        if a.method == 'FR':
            fit = json.loads((native/'fit_report.json').read_text())
            assert fit['status'] == 'COMPLETE'
            certificate = json.loads((native/'NATIVE_SOLVER_OBJECTIVE_AUDIT.json').read_text())
            assert certificate['pass_'] is True
            archives = {n: crc(native/n) for n in ['effects.npz','native_unscaled.npz']}
            assert archives['effects.npz']['sha256'] == fit['effects_sha256']
        else:
            rt = json.loads((native/'runtime.json').read_text())
            ee = json.loads((native/'execution.json').read_text())
            assert rt['status'] == 'COMPLETE' and rt['failed_rows'] == 0 and ee['returncode'] == 0
            archives = {'posterior.npz': crc(native/'posterior.npz')}
            assert archives['posterior.npz']['sha256'] == ee['posterior_npz_sha256']
        write(a.out/'OWNER_COMPLETION.json', {'status':'VERIFIED_AFTER_OWNER_EXIT',
              'owner_session_id':str(a.owner_session_id), 'owner_exit_code':0,
              'run_record_sha256':sha(a.out/'RUN_RECORD.json'),'input_lock_sha256':sha(a.input_lock),
              'method':a.method,'donor':lock['donor'],'archives':archives})
        print(json.dumps({'status':'VERIFIED','method':a.method,'donor':lock['donor'],'path':str(a.out)}))
        return
    assert not a.out.exists(), 'Preserve every attempt; use a reviewed new attempt label after failure'
    a.out.mkdir(parents=True)
    native = a.out/'native'
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='2' if a.method=='FR' else '1',
               OMP_NUM_THREADS='2' if a.method=='FR' else '1', MKL_NUM_THREADS='2' if a.method=='FR' else '1',
               NUMBA_NUM_THREADS='2')
    if a.method == 'FR':
        cmd = [sys.executable, str(HERE/'guarded_native.py'),'--record',str(a.out/'NATIVE_SELF_GUARD.json'),
               '--rss-gib','5.5','--seconds',str(a.timeout),str(ROOT/'vendor_work/run_prepared_input_atomic_v3.py'),
               '--input-dir',required['input_directory'],'--out',str(native),
               '--rank','300','--spca-alpha','0.1','--lasso-alpha','0.0001',
               '--spca-iter','50','--threads','2','--seed','20261012']
    else:
        cmd = [sys.executable, str(ROOT/'ash_work/run_posterior.py'),
               '--input',required['files']['full.npz']['path'],'--output',str(native),
               '--mode','ash','--retry-mixem','--seed','20261012']
    record = {'status':'RUNNING','donor':lock['donor'],'method':a.method,
              'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'command':cmd,'input_lock_sha256':sha(a.input_lock),
              'wrapper_sha256':sha(__file__),'timeout_seconds':a.timeout,
              'RSS_limit_GiB':5.5,'reference_read':False,'score_computed':False}
    write(a.out/'START.json', record)
    # The FR wrapper acquires the same lock itself; ASH needs its own holder.
    handle = (ROOT/'heavy_memory.lock').open('a')
    if a.method == 'ASH':
        fcntl.flock(handle,fcntl.LOCK_EX)
    began = time.monotonic(); peak = 0; reason = None
    with (a.out/'stdout.log').open('x') as out:
        proc = subprocess.Popen(cmd,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                rss = process_rss(proc.pid); peak = max(peak,rss)
                if time.monotonic()-began > a.timeout+30:
                    reason = 'PREDECLARED_TIME_GUARD'; break
                time.sleep(.5)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGKILL)
            code = proc.wait()
    if a.method == 'ASH':
        fcntl.flock(handle,fcntl.LOCK_UN)
    if code == 0 and a.method == 'FR':
        audit = [sys.executable,str(ROOT/'vendor_work/add_native_audit_certificate.py'),
                 '--fit-dir',str(native),'--axes',required['files']['axes.npz']['path']]
        with (a.out/'audit_stdout.log').open('x') as out:
            code = subprocess.run(audit,env=env,stdout=out,stderr=subprocess.STDOUT).returncode
    native_guard = a.out/'NATIVE_SELF_GUARD.json'
    record.update(status='COMPLETE_PENDING_SEPARATE_VERIFICATION' if code==0 else 'FAILED',
                  returncode=code,stop_reason=reason,wall_seconds=time.monotonic()-began,
                  process_tree_peak_RSS_bytes=peak,
                  parent_PID_RSS_note='Transport can expose proxy PIDs; parent process-tree RSS is not a valid native RSS estimate here. Use native self guard or native R child resource record.',
                  native_self_guard=json.loads(native_guard.read_text()) if native_guard.exists() else None)
    write(a.out/'RUN_RECORD.json', record)
    print(json.dumps(record,indent=2))
    raise SystemExit(code)

if __name__ == '__main__':
    main()
