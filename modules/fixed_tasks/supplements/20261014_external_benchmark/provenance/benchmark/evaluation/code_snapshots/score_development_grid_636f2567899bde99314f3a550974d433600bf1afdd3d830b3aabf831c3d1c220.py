#!/usr/bin/env python3
"""Execute locked D3 score/selection only after every frozen FR fit is complete."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import evaluate_six_arm as e


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=e.ROOT/'AMENDMENT_03_FR_SEARCH_21.json')
    p.add_argument('--fit-root',type=Path,default=e.ROOT/'fr_outputs/D3_Rest')
    p.add_argument('--out-root',type=Path,default=e.ROOT/'evaluation/D3_Rest')
    a=p.parse_args()
    grid=json.loads(a.protocol.read_text())['fr_grid']
    fitted=json.loads((a.fit_root/'GRID_EXECUTION.json').read_text())
    if fitted['status']!='COMPLETE' or len(fitted['runs'])!=len(grid) or any(r['status']!='COMPLETE' for r in fitted['runs']):
        raise ValueError('All frozen FR grid runs must complete before scoring/selection')
    if fitted['spec_sha256']!=e.sha(a.protocol):
        raise ValueError('FR fit-grid specification hash differs from scoring protocol')
    code_lock=json.loads((e.ROOT/'SCORING_CODE_AMENDMENT_03_INPUT_FINGERPRINT.json').read_text())
    if code_lock['current_sha256']!=e.sha(e.__file__):
        raise ValueError('Scoring source changed after latest declared metadata-only amendment')
    a.out_root.mkdir(parents=True,exist_ok=True)
    reports=[]
    began=time.monotonic()
    for config in grid:
        target=a.out_root/config['id']
        output=target/'evaluation_report.json'
        if target.exists():
            raise FileExistsError('Refusing to overwrite prior or partial scoring directory: '+str(target))
        command=[sys.executable,str(Path(e.__file__)), 'evaluate', '--dataset','D3_Rest',
            '--config-id',config['id'],'--protocol',str(a.protocol),
            '--fr-effects',str(a.fit_root/(config['id']+'_rep0')/'effects.npz'),'--out',str(target)]
        with (a.out_root/(config['id']+'_score_process.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        reports.append(output)
        report=json.loads(output.read_text())
        print(json.dumps({'config_id':config['id'],'native_fr_P25':report['results']['25']['arms']['fr']['point']}),flush=True)
    command=[sys.executable,str(Path(e.__file__)),'select-winner','--protocol',str(a.protocol),'--reports',*[str(p) for p in reports]]
    with (a.out_root/'select_winner_process.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    lock=json.loads((e.ROOT/'FR_D3_WINNER_LOCK.json').read_text())
    summary=dict(status='COMPLETE',reports=[str(p.resolve()) for p in reports],
        winner_config_id=lock['config_id'],winner_P25=lock['winner_P25'],
        winner_lock_sha256=e.sha(e.ROOT/'FR_D3_WINNER_LOCK.json'),elapsed_seconds=time.monotonic()-began,
        score_driver_sha256=e.sha(__file__),protocol_sha256=e.sha(a.protocol))
    e.save_json(a.out_root/'GRID_SCORING.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
