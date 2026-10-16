#!/usr/bin/env python3
"""Validate frozen-scale sufficient statistics, then prepare native ashr input.

No comparison scores are evaluated. ASH/TRADE share identical effect and SE arrays.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--D3-weight-stats',type=Path)
    a=p.parse_args();src=a.input_dir/'full_effect_stats.npz';axes=a.input_dir/'axes.npz'
    with np.load(src,allow_pickle=False) as z:
        y=z['effects'];s=z['standard_errors'];genes=z['gene_ids'];targets=z['target_ids']
        qi=z['query_keep_columns'];cm=z['control_mean'];cv=z['control_var'];scale=float(z['scale'])
    assert y.ndim==2 and s.shape==y.shape and genes.shape==(y.shape[1],) and targets.shape==(y.shape[0],)
    assert np.isfinite(y).all() and np.isfinite(s).all() and np.isfinite(cv).all()
    assert len(np.unique(genes))==len(genes) and len(np.unique(targets))==len(targets)
    if not np.all(s>0):
        bad=np.argwhere(s<=0)
        raise ValueError(f'Nonpositive SE: {len(bad)} coordinates, first={bad[:10].tolist()}; no automatic floor')
    with np.load(axes,allow_pickle=False) as z:
        assert np.array_equal(targets,z['target_ids'])
        assert np.array_equal(genes[qi],z['query_ids'])
        qerror=float(np.max(np.abs(y[:,qi]-z['input_effects'])))
        assert qerror<1e-12,qerror
        assert scale==float(z['scale'])
        nc=int(z['input_control_count']);nt=int(z['target_cells_per_group'])
    assert nc==2000 and nt==20
    checks=dict(query_effect_max_abs_error=qerror,minimum_SE=float(s.min()),maximum_SE=float(s.max()),
                nonpositive_SE_count=int(np.sum(s<=0)),n_control=nc,n_target=nt,
                min_control_variance=float(cv.min()),scale=scale)
    if a.D3_weight_stats:
        with np.load(a.D3_weight_stats,allow_pickle=False) as w:
            e1=float(np.max(np.abs(np.mean(y*y,axis=1)-w['log_m2'][:,0])))
            e2=float(np.max(np.abs(np.mean(s*s,axis=1)-cv.mean()/nc-w['log_vb'][:,0]/nt)))
        assert e1<1e-10,e1
        assert e2<1e-12,e2
        checks.update(frozen_m2_max_error=e1,variance_identity_max_error=e2)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    for name,idx in [('full',np.arange(len(targets))),('smoke32',np.arange(min(32,len(targets))))]:
        np.savez_compressed(a.output_dir/(name+'.npz'),effects=y[idx],ses=s[idx],row_ids=targets[idx],genes=genes)
    audit=json.loads((a.input_dir/'INPUT_AUDIT.json').read_text())
    report=dict(status='PASS',input_files={str(src):sha(src),str(axes):sha(axes)},
                source_input_audit_sha256=sha(a.input_dir/'INPUT_AUDIT.json'),
                output_files={p.name:sha(p)for p in a.output_dir.glob('*.npz')},checks=checks,
                effect_center='frozen historical control center; see original INPUT_AUDIT for floating-point accumulation difference',
                standard_errors='sqrt(fresh target sample variance/20 + fresh control sample variance/2000), ddof=1',
                source_input_audit=audit,reference_score_computed=False)
    (a.output_dir/'INPUT_VALIDATION.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
