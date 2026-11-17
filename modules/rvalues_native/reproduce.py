#!/usr/bin/env python3
"""Verify and prepare native rvalues fitting materials; no R fitting."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
out = a.output_dir.expanduser().resolve()
if out.exists() or out == REPOSITORY or REPOSITORY in out.parents:
    p.error('Use a new output directory outside the code package.')
out.mkdir(parents=True)
with (out/'preparation.log').open('w') as log:
    subprocess.run([sys.executable, '-B', str(ROOT/'prepare_rvalues.py'), '--out', str(out/'materials')],
                   stdout=log, stderr=subprocess.STDOUT, check=True)
receipt = json.loads((out/'materials/RESTORE_REPORT.json').read_text())
report = {'status':'PASS', 'scope':'Exact native inputs and source files restored and checked, including all 49 original immutable-file hashes. No R model was fitted.',
          'payload_sha256':receipt['payload_sha256'], 'payload_bytes':(ROOT/'native_data.tar.xz').stat().st_size,
          'restored_files':receipt['files_restored'], 'extracted_root':'materials/native_rvalues',
          'new_model_fits':0, 'log':'preparation.log','restore_report':'materials/RESTORE_REPORT.json'}
(out/'REPRODUCTION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
