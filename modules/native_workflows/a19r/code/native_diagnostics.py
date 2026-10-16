#!/usr/bin/env python3
"""Describe native convergence and shrinkage without making mechanism claims."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
    results={}
    for d in ['D3_Rest','D1_Rest','D2_Rest']:
        selected=json.loads((a.root/'SELECTED_RUNS.json').read_text())['runs'][d]
        r={};ash=Path(selected['ASH'])/'native';fr=Path(selected['FR'])/'native'
        if ash.exists() and (ash/'runtime.json').exists():
            rt=json.loads((ash/'runtime.json').read_text());ex=json.loads((ash/'execution.json').read_text())
            rows=[json.loads(x) for x in (ash/'fit_rows.jsonl').read_text().splitlines()]
            with np.load(a.root/'inputs'/d/'ASH/full.npz',allow_pickle=False) as z:y=z['effects'];ids=z['row_ids'];genes=z['genes']
            with np.load(ash/'posterior.npz',allow_pickle=False) as z:
                assert np.array_equal(z['genes'],genes) and np.array_equal(z['row_ids'],ids);pm=z['posterior']
            ratio=np.linalg.norm(pm,axis=1)/np.linalg.norm(y,axis=1)
            r['ASH']={'status':rt['status'],'targets':rt['n_rows'],'genes':rt['n_genes'],
              'failed':rt['failed_rows'],'warnings':rt['warnings'],'native_mixEM_retries':sum(x['retry_mixem'] for x in rows),
              'native_R_peak_RSS_KiB':ex['peak_R_child_RSS_KB'],'seconds':ex['seconds'],
              'posterior_L2_over_input_L2_target_quantiles':np.quantile(ratio,[0,.25,.5,.75,1]).tolist(),
              'exact_posterior_zero_fraction':float(np.mean(pm==0)),
              'nonzero_input_sign_flip_fraction':float(np.mean((y*pm)<0)),
              'fit_record_sha256':sha(ash/'fit_rows.jsonl'),'interpretation':'Describes native posterior regularization; it does not establish why ranking changes.'}
        if fr.exists() and (fr/'fit_report.json').exists():
            rt=json.loads((fr/'fit_report.json').read_text());c=json.loads((fr/'NATIVE_SOLVER_OBJECTIVE_AUDIT.json').read_text())
            r['FR']={'status':rt['status'],'parameters':rt['parameters'],'input':rt['input'],
               'warnings':rt['warnings'],'runtime_seconds':rt['runtime_seconds'],'max_rss_kib':rt['max_rss_kib'],
               'factor_diagnostics':rt['factor_diagnostics'],'recover_audit':rt['recover_audit'],
               'scaling_order_audit':rt['scaling_order_audit'],'native_solver_certificate_pass':c['pass_'],
               'dictionary_convergence':'fixed50iteration budget; no native dictionary convergence criterion or full-rank optimality claim',
               'fit_report_sha256':sha(fr/'fit_report.json')}
        results[d]=r
    a.out.write_text(json.dumps({'status':'COMPLETE','sources':results,'reference_scores_read':False},indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'COMPLETE','out':str(a.out)}))

if __name__=='__main__':main()
