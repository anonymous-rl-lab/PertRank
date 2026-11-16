#!/usr/bin/env python3
"""Reproduce original A19-R three-arm endpoints; no new Ay result is generated."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True, help='New output directory')
    args = ap.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit('Output directory must be new.')
    manifest = json.loads((root/'DOWNLOADED_INPUT_SHA256.json').read_text())
    for name, entry in manifest.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == entry['sha256'], name
    out.mkdir(parents=True)
    summary = {}
    for donor in ('D3', 'D1', 'D2'):
        config = {
            'donor': donor+'_Rest',
            'legacy_prediction': str(root/f'predictions_{donor}.npz'),
            'reference': str(root/'query_reference.npz'),
            'legacy_point_results': str(root/'point_results.json'),
        }
        cp = out/(donor+'_config.json')
        cp.write_text(json.dumps(config, indent=2)+'\n')
        subprocess.run([sys.executable, str(root/'score_native.py'), '--config', str(cp), '--out', str(out/donor)], check=True)
        report = json.loads((out/donor/'report.json').read_text())
        summary[donor] = {
            'means': report['means'],
            'geometry': report['contrasts']['PertRank_minus_Euclidean'],
            'checks': report['original_prediction_regression'],
        }
    summary['_status'] = 'BASELINE_REPLAY_PASS'
    summary['_scope'] = 'Previously analyzed independent RNA reference; conditional replay only. FR/ASH are outside this replay, not unavailable from the original repository.'
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
