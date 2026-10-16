#!/usr/bin/env python3
"""Offline reproduction of the archived external-baseline evidence report."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True

def verify():
    meta = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text())
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
              if p.is_file() and p.name != 'PACKAGE_MANIFEST.json'
              and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if actual != set(meta['files']):
        raise AssertionError({'missing': sorted(set(meta['files']) - actual),
                              'extra': sorted(actual - set(meta['files']))})
    for name, digest in meta['files'].items():
        p = (ROOT / name).resolve()
        if not p.is_relative_to(ROOT):
            raise ValueError('Unsafe manifest path')
        with p.open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != digest:
                raise AssertionError('File hash mismatch: ' + name)
    return len(actual)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT.parent / 'PertRank_external_benchmark_reproduced')
    p.add_argument('--verify-only', action='store_true', help='Check all file hashes without numerical replay')
    a = p.parse_args()
    n = verify()
    if a.verify_only:
        print(json.dumps({'status': 'MANIFEST_PASS', 'files': n}))
        return
    out = a.output_dir.resolve()
    if out.is_relative_to(ROOT) or out.exists():
        p.error('Use a new output directory outside this package')
    out.mkdir(parents=True)
    start = time.monotonic()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    receipts = {}
    for script, folder, receipt in [
        ('replay.py', 'numerical', 'REPRODUCTION_CHECK.json'),
        ('d3_replay.py', 'd3_development', 'REPRODUCTION_CHECK.json'),
        ('d3_reconciliation.py', 'd3_reconciliation', 'D3_RECONCILIATION_REPLAY.json'),
    ]:
        subprocess.run([sys.executable, str(ROOT / script), '--output-dir', str(out / folder)],
                       check=True, env=env)
        receipts[folder] = json.loads((out / folder / receipt).read_text())
    result = {'status': 'EXTERNAL_BENCHMARK_FIXED_RESULT_REPRODUCTION_PASS',
              'package_version': '1.4.0', 'manifest_files': n,
              'new_model_fits': 0, 'expression_matrices_read': False,
              'network_required': False, 'results': receipts,
              'elapsed_seconds': time.monotonic() - start}
    (out / 'BENCHMARK_REPRODUCTION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'])

if __name__ == '__main__':
    main()
