#!/usr/bin/env python3
"""Freeze all six verified native fits together before any new external scoring."""
from pathlib import Path
import datetime,hashlib,json

HERE=Path(__file__).resolve().parent

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    selected=json.loads((HERE/'SELECTED_RUNS.json').read_text());records={}
    for d,methods in selected['runs'].items():
        records[d]={}
        for method,path in methods.items():
            p=Path(path);owner=json.loads((p/'OWNER_COMPLETION.json').read_text())
            assert owner['status']=='VERIFIED_AFTER_OWNER_EXIT' and owner['owner_exit_code']==0
            assert owner['donor']==d and owner['method']==method
            assert sha(p/'RUN_RECORD.json')==owner['run_record_sha256']
            r=json.loads((p/'RUN_RECORD.json').read_text());assert r['returncode']==0
            for f,h in owner['archives'].items():assert sha(p/'native'/f)==h['sha256']
            if method=='FR':
                fit=json.loads((p/'native/fit_report.json').read_text())
                assert fit['status']=='COMPLETE'
                assert fit['parameters']=={'rank':300,'spca_alpha':0.1,'lasso_alpha':0.0001,'spca_iter':50,'seed':20261012,'threads':2}
            else:
                fit=json.loads((p/'native/runtime.json').read_text())
                assert fit['status']=='COMPLETE' and fit['failed_rows']==0 and fit['config']['ash_options']=={}
            records[d][method]={'directory':str(p),'owner_sha256':sha(p/'OWNER_COMPLETION.json'),'archives':owner['archives']}
    file=HERE/'ALL_PREDICTIONS_LOCK.json';assert not file.exists()
    payload={'status':'ALL_SIX_VERIFIED_NATIVE_FITS_FROZEN_BEFORE_NEW_SCORE',
      'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':records,
      'selected_runs_sha256':sha(HERE/'SELECTED_RUNS.json'),
      'scoring_source_sha256':{n:sha(HERE/n) for n in ['score_native.py','combine_sources.py']},
      'config_sha256':{d:sha(HERE/'configs'/(d+'_final.json')) for d in records},
      'analysis_status':'Posthoc expanded comparison on a previously opened external biological reference; no prospective confirmation claim.',
      'model_selection_by_new_scores':False}
    file.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'status':payload['status'],'sha256':sha(file)}))

if __name__=='__main__':main()
