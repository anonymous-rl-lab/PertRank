#!/usr/bin/env python3
"""Verify released depth scoring after its exec owner has exited successfully.

The caller supplies a completed owner session ID; this script never polls live
outputs and does not infer owner completion from file existence.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import zipfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score-dir', type=Path, required=True)
    ap.add_argument('--owner-session-id', type=int, required=True)
    ap.add_argument('--completed-owner-exit-code', type=int, required=True)
    a = ap.parse_args()
    assert a.completed_owner_exit_code == 0
    report = json.loads((a.score_dir / 'evaluation_report.json').read_text())
    assert report['all_intervals_descriptive'] is True
    assert report['confirmatory_main_result'] is False
    for audit in report['independent_audit'].values():
        assert audit['status'] == 'PASS' and audit['check_count'] == 17
        assert audit['same_sampled_query_blocks'] is True
        assert audit['per_query_max_abs_error'] < 1e-12
        assert audit['bootstrap_max_abs_error'] < 1e-12
    for name, h in report['files_sha256'].items():
        assert sha(a.score_dir / name) == h, name
    files = [p for p in a.score_dir.iterdir() if p.suffix in ('.json', '.npz')
             and p.name not in ('SCORE_OWNER_COMPLETION.json', 'descriptive_relative_to_raw.json')]
    for p in files:
        if p.suffix == '.npz':
            with zipfile.ZipFile(p) as z:
                assert z.testzip() is None, p.name
    out = a.score_dir / 'SCORE_OWNER_COMPLETION.json'
    assert not out.exists(), 'Never overwrite prior owner completion'
    out.write_text(json.dumps({
        'status': 'SCORE_OWNER_EXECUTION_ENDED_INTEGRITY_VERIFIED',
        'owner_session_id': a.owner_session_id,
        'owner_exit_code': a.completed_owner_exit_code,
        'npz_crc': 'PASS',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'files_sha256': {p.name: sha(p) for p in files}
    }, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'depth': report['depth'],
                      'curve': report['curve'], 'owner_session_id': a.owner_session_id,
                      'P25': {k: v['point'] for k, v in report['results']['25']['arms'].items()}}))


if __name__ == '__main__':
    main()
