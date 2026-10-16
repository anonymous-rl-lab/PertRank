"""Descriptive paired depth contrasts from completed, owner-verified scores."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make(report_path):
    report_path = Path(report_path)
    r = json.loads(report_path.read_text())
    owner_path = report_path.with_name('SCORE_OWNER_COMPLETION.json')
    owner = json.loads(owner_path.read_text())
    assert owner.get('owner_exit_code') == 0
    assert owner['status'] == 'SCORE_OWNER_EXECUTION_ENDED_INTEGRITY_VERIFIED'
    assert sha(report_path) == owner['files_sha256']['evaluation_report.json']
    assert r['status'] == 'COMPLETE_INDEPENDENT_AUDIT_PASS'
    array_path = report_path.with_name('evaluation_arrays.npz')
    assert sha(array_path) == r['files_sha256']['evaluation_arrays.npz']
    arms = r['arms']
    base = arms.index('raw')
    output = {'status':'COMPLETE', 'analysis_label':'Post hoc descriptive paired contrasts',
              'report_path':str(report_path.resolve()), 'report_sha256':sha(report_path),
              'arrays_sha256':sha(array_path), 'owner_sha256':sha(owner_path),
              'inference_scope':'One nested depth chain; conditional query-block intervals. Not donor-population inference and not an additional corrected testing family.',
              'results':{}}
    with np.load(array_path, allow_pickle=False) as z:
        for k in ['25','10']:
            b = z['query_only_bootstrap_p'+k]
            assert b.shape == (2000, len(arms))
            output['results'][k] = {}
            for i, arm in enumerate(arms):
                if arm == 'raw':
                    continue
                arm_mean = r['results'][k]['arms'][arm]['point']
                raw_mean = r['results'][k]['arms']['raw']['point']
                diff = b[:,i] - b[:,base]
                output['results'][k][arm+'_minus_raw'] = {
                    'point':arm_mean-raw_mean,
                    'query_only_95_interval':np.quantile(diff,[.025,.975]).tolist(),
                    'query_only_99_interval':np.quantile(diff,[.005,.995]).tolist(),
                    'descriptive':True}
    destination = report_path.with_name('descriptive_relative_to_raw.json')
    destination.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'path':str(destination),'pertrank_minus_raw_P25':output['results']['25']['pertrank_minus_raw']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('reports',type=Path,nargs='+')
    for report in p.parse_args().reports:
        make(report)
