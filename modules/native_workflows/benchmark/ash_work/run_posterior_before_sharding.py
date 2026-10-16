#!/usr/bin/env python3
"""Write ordered full-gene effect/SE input and invoke the isolated official R core.

Input NPZ keys: effects, ses (rows x genes); row_ids and genes (unicode strings).
This script never reads deep reference matrices and never computes benchmark scores.
Output retains all row/column axes; failed native fits remain NaN, never omitted.
"""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, time
import numpy as np

HERE=Path(__file__).resolve().parent
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20), b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['ash','trade'],default='ash')
    p.add_argument('--mixcompdist',choices=['uniform','normal','halfuniform'],default=None)
    p.add_argument('--method',choices=['fdr','shrink'],default=None)
    p.add_argument('--optmethod',choices=['mixSQP','mixEM'],default=None)
    p.add_argument('--seed',type=int,default=20261012)
    p.add_argument('--retry-mixem',action='store_true',help='Predeclared native mixEM fallback only after explicit mixSQP failure/nonconvergence; maxiter=10000')
    p.add_argument('--effect-scale',default='normalized log1p expression mean difference')
    p.add_argument('--trade-source',type=Path,default=HERE/'R_TRADE_univariate.R')
    a=p.parse_args()
    if a.mode=='trade' and any([a.mixcompdist,a.method,a.optmethod]):
        p.error('Immutable TRADE fit_ash does not accept ash option overrides')
    if a.output.exists():p.error('Output directory must not already exist')
    with np.load(a.input,allow_pickle=False) as z:
        y=np.asarray(z['effects'],dtype='<f8');s=np.asarray(z['ses'],dtype='<f8')
        ids=z['row_ids'].astype(str);genes=z['genes'].astype(str)
    assert y.ndim==2 and s.shape==y.shape
    assert ids.shape==(y.shape[0],) and genes.shape==(y.shape[1],)
    assert np.isfinite(y).all() and np.isfinite(s).all() and (s>0).all(), 'Require finite effects and strictly positive SE; no silent floor'
    assert len(set(ids))==len(ids) and len(set(genes))==len(genes)
    assert not any('\n' in x or '\r' in x for x in [*ids,*genes])
    a.output.mkdir(parents=True);inp=a.output/'input';inp.mkdir()
    y.tofile(inp/'effects.f64');s.tofile(inp/'ses.f64')
    (inp/'row_ids.txt').write_text('\n'.join(ids)+'\n')
    (inp/'genes.txt').write_text('\n'.join(genes)+'\n')
    (inp/'metadata.json').write_text(json.dumps(dict(n_rows=y.shape[0],n_genes=y.shape[1],effect_scale=a.effect_scale,input_npz_sha256=digest(a.input)),indent=2))
    cfg=dict(mode=a.mode,seed=a.seed,ash_options={},trade_source=str(a.trade_source.resolve()),retry_mixem=a.retry_mixem)
    for opt in ['mixcompdist','method','optmethod']:
        if getattr(a,opt) is not None:cfg['ash_options'][opt]=getattr(a,opt)
    if a.mode=='trade':cfg['trade_source_sha256']=digest(a.trade_source)
    config=a.output/'config.json';config.write_text(json.dumps(cfg,indent=2))
    launcher=HERE/'environment/bin/R'
    cmd=[str(launcher),'--vanilla','--slave','--file='+str(HERE/'posterior_runner.R'),'--args',str(inp.resolve()),str(a.output.resolve()),str(config.resolve())]
    t=time.monotonic()
    with (a.output/'run.log').open('w') as f:
        result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    actual=a.output/'posterior.f64'
    manifest=dict(input_sha256=digest(a.input),command=cmd,returncode=result.returncode,seconds=time.monotonic()-t,runner_sha256=digest(HERE/'posterior_runner.R'),config=cfg)
    if actual.exists():
        pred=np.fromfile(actual,dtype='<f8')
        if pred.size==y.size:
            pred=pred.reshape(y.shape)
            np.savez_compressed(a.output/'posterior.npz',posterior=pred,row_ids=ids,genes=genes)
            manifest.update(posterior_npz_sha256=digest(a.output/'posterior.npz'),finite_rows=int(np.isfinite(pred).all(axis=1).sum()))
    (a.output/'execution.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
