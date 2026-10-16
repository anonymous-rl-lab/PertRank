"""Descriptive contrasts requested after seeing D3; original reports untouched."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def make(report):
    r=json.loads(report.read_text())
    combined=report.name=='equal_state_report.json'
    ap=report.with_name('equal_state_arrays.npz' if combined else 'evaluation_arrays.npz')
    assert sha(ap)==r['arrays_sha256' if combined else 'evaluation_arrays_sha256']
    out={'status':'COMPLETE','label':'Post hoc descriptive contrast, excluded from original five-contrast family',
         'interpretation':'No preregistered superiority, noninferiority or equivalence test is claimed.',
         'source':str(report.resolve()),'source_sha256':sha(report),'arrays_sha256':sha(ap),'results':{}}
    with np.load(ap,allow_pickle=False) as a:
        for k in [25,10]:
            x=r['results'][str(k)]['arms']
            item={'point':x['pertrank']['point']-x['fr_pertrank']['point'],
                  'input_rep_points':(np.array(x['pertrank']['input_rep_points'])-np.array(x['fr_pertrank']['input_rep_points'])).tolist()}
            keys=[('task_stratified_query','equal_state_bootstrap')] if combined else [('query_only','query_only_bootstrap')]
            if 'joint_bootstrap_p'+str(k) in a:keys.append(('joint','joint_bootstrap'))
            for prefix,key in keys:
                b=a[key+'_p'+str(k)]
                diff=b[:,2]-b[:,5]
                item[prefix+'_95_interval']=np.quantile(diff,[.025,.975]).tolist()
                item[prefix+'_99_interval']=np.quantile(diff,[.005,.995]).tolist()
            out['results'][str(k)]={'pertrank_minus_fr_pertrank':item}
    dest=report.with_name('requested_extra_contrasts.json')
    dest.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    return str(dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reports',nargs='+',type=Path)
    for report in p.parse_args().reports:print(make(report))
