"""Run Stim in a separate offline interpreter to isolate its frozen modules."""
import json
import os
from pathlib import Path
import subprocess
import sys


def run(root: Path, output: Path) -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError('Use a new output directory outside the repository')
    script = root / 'supplements' / '20261011_stim' / 'replay.py'
    environment = dict(os.environ)
    environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, '-B', str(script), '--output-dir', str(output)],
                   cwd=root, env=environment, check=True)
    result = json.loads((output / 'VERIFICATION.json').read_text())
    if result.get('status') != 'STIM_EXACT_REPRODUCTION_PASS':
        raise AssertionError(('Stim verification failed', result.get('status')))
    print('STIM_SUPPLEMENT_PASS', flush=True)
    return result
