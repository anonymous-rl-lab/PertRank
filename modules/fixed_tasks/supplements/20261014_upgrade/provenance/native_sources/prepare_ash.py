#!/usr/bin/env python3
"""Bind exact A19-R sufficient statistics to the matching original control variance."""
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--donor',required=True)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--control-stats',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert a.donor in ['D3_Rest','D1_Rest','D2_Rest'] and not a.out.exists()
    with np.load(a.source/'context.npz',allow_pickle=False) as z:c={k:z[k] for k in z.files}
    with np.load(a.source/'sufficient_state.npz',allow_pickle=False) as z:
        S=z['S'];Q=z['Q'];m2=z['m2'];vb=z['vb']
    with np.load(a.source/'predictions_locked.npz',allow_pickle=False) as z:
        query_ids=z['query_ids'];targets=z['target_ids'];old_raw=z['effects'][0]
    with np.load(a.control_stats,allow_pickle=False) as z:
        genes=z['gene_ids'];cm=z['control_mean'];cv=z['control_var'];n=int(z['n_control']);scale=float(z['scale'])
    assert np.array_equal(genes,c['genes_keep']) and np.array_equal(cm,c['cm_keep'])
    assert n==int(c['n_control']) and scale==float(c['scale'])
    assert abs(float(cv.sum())-float(c['trC'])) < 1e-9
    effects=S/20-cm
    variance=(Q-S*S/20)/19
    assert np.isfinite(variance).all() and variance.min()>=0, 'No silent variance repair or SE floor'
    ses=np.sqrt(variance/20+cv/n)
    assert np.isfinite(ses).all() and ses.min()>0
    query_map={str(g):i for i,g in enumerate(genes)};qi=np.array([query_map[str(g)] for g in query_ids])
    query_error=float(np.max(np.abs(effects[:,qi].T-old_raw)))
    m2_error=float(np.max(np.abs(np.mean(effects*effects,axis=1)-m2)))
    vb_error=float(np.max(np.abs(variance.mean(1)-vb)))
    assert query_error<1e-12 and m2_error<1e-12 and vb_error<1e-12
    expected={'D3_Rest':(52,293),'D1_Rest':(59,289),'D2_Rest':(63,289)}[a.donor]
    assert (len(targets),len(query_ids))==expected
    a.out.mkdir(parents=True)
    f=a.out/'full.npz';np.savez_compressed(f,effects=effects,ses=ses,row_ids=targets,genes=genes)
    sources=[a.source/x for x in ['context.npz','sufficient_state.npz','predictions_locked.npz']]+[a.control_stats,Path(__file__)]
    checks={'same_control_mean_exact':True,'same_control_gene_axis_exact':True,'same_scale_ncontrol_exact':True,
            'control_trace_error':float(cv.sum())-float(c['trC']),'query_effect_maxabs':query_error,
            'm2_maxabs':m2_error,'mean_target_variance_maxabs':vb_error,'min_SE':float(ses.min()),
            'min_target_variance':float(variance.min()),'targets':len(targets),'genes':len(genes),'queries':len(query_ids)}
    lock={'status':'READY_EXACT_A19R_INPUT','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'donor':a.donor,'methods':{'ASH':{'files':{'full.npz':{'path':str(f.resolve()),'sha256':sha(f)}}}},
      'checks':checks,'source_sha256':{str(p.resolve()):sha(p) for p in sources},
      'new_benchmark_reference_read':False,'no_perturbation_refit':True,
      'configuration':'ashr2.2-63 default uniform/fdr; full original control-expressed gene axis; n_target20'}
    (a.out/'INPUT_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n')
    print(json.dumps({'status':lock['status'],'donor':a.donor,'checks':checks,'input':str(f)},indent=2))

if __name__=='__main__':main()
