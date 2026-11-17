#!/usr/bin/env python3
"""Lossless program-experiment reconstruction and independent scientific replay."""
from pathlib import Path
import argparse, hashlib, io, json, os, subprocess, sys, tarfile, time
import numpy as np
from export_semantics import normalize, assert_preserved
ROOT=Path(__file__).resolve().parent
sha=lambda x:hashlib.sha256(x).hexdigest()
def require(ok,message):
    if not ok:raise RuntimeError(message)

def restore(destination):
    index=json.loads((ROOT/'INDEX.json').read_text())
    stored={}
    with tarfile.open(ROOT/'program_data.tar.xz','r:xz') as tf:
        for member in tf:
            require(member.isfile() and member.name in index['blobs'], 'Unexpected archive entry: '+member.name)
            b=tf.extractfile(member).read();meta=index['blobs'][member.name];size=meta['shuffle'];n=len(b)//size*size
            raw=np.frombuffer(b[:n],dtype=np.uint8).reshape(size,-1).T.copy().tobytes()+b[n:]
            require(sha(raw)==member.name, 'Lossless blob hash mismatch: '+member.name)
            stored[member.name]=raw
    arrays={}
    def arr(h):
        if h in arrays:return arrays[h]
        if h in stored:
            a=np.load(io.BytesIO(stored[h]),allow_pickle=False)
        else:
            recipe=index['recipes'][h];args=[arr(v) for v in recipe['inputs']];op=recipe['op']
            if op=='center_scale':a=(args[0]-args[1])/args[2]
            elif op=='divide':a=args[0]/args[1]
            elif op=='six_scores':
                r,p,w0,wm=args;a=np.stack([r,r*w0[:,None],r*wm[:,None],p,p*w0[:,None],p*wm[:,None]])
            elif op=='plate_effect':
                means,plates,candidateplates,centers=args;lookup={str(p):i for i,p in enumerate(plates)};ix=np.array([lookup[str(p)] for p in candidateplates]);a=means-centers[ix]
            elif op=='plate_se':
                variances,n,plates,candidateplates,cv,cn=args;lookup={str(p):i for i,p in enumerate(plates)};ix=np.array([lookup[str(p)] for p in candidateplates]);a=np.sqrt(variances/int(n)+cv[ix]/cn[ix,None])
            else:raise ValueError(op)
            b=io.BytesIO();np.save(b,a,allow_pickle=False);require(sha(b.getvalue())==h, 'Derived array fingerprint mismatch: '+h)
        arrays[h]=a;return a
    count=0
    for rel,record in index['files'].items():
        if rel in index.get('file_recipes',{}):continue
        p=destination/rel;p.parent.mkdir(parents=True,exist_ok=True)
        if record['kind']=='file':p.write_bytes(stored[record['blob']])
        else:np.savez_compressed(p,**{a['key']:arr(a['blob']) for a in record['arrays']})
        require(p.stat().st_size==record['bytes'] and sha(p.read_bytes())==record['sha256'], 'Restored file mismatch: '+rel)
        count+=1
    for rel,recipe in index.get('file_recipes',{}).items():
        require(recipe['op']=='ash_tsv','Unexpected file recipe');rows=['fit_id\tcandidate_id\teffect\tse\n']
        fits=json.loads((destination/'external/predictions/ASH_FITS.json').read_text())['fits']
        for f in fits:
            with np.load(destination/'external/predictions/input_tasks'/f"{f['task_id']}.npz",allow_pickle=False) as d:
                k=f['program_index']
                for j,c in enumerate(d['candidate_ids']):rows.append(f"{f['fit_id']}\t{c}\t{d['effects'][j,k]:.17g}\t{d['standard_errors'][j,k]:.17g}\n")
        p=destination/rel;p.write_text(''.join(rows));record=index['files'][rel];require(sha(p.read_bytes())==record['sha256'], 'Derived TSV fingerprint mismatch');count+=1
    return {'status':'PASS','restored_files':count,'unique_lossless_blobs':len(stored),'derived_arrays_verified_bitwise':len(index['recipes']),'dtype_precision_preserved':True,'no_reference_cells_discarded':True}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['core','exploratory','all']);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=args.out.resolve();require(not out.exists(),'Choose a new output directory');out.mkdir(parents=True)
    t=time.time();materialized=out/'materials';materialized.mkdir();restoration=restore(materialized)
    (out/'RESTORATION.json').write_text(json.dumps(restoration,indent=2)+'\n')
    modules=['mechanism','external'] if args.action=='core' else ['exploratory'] if args.action=='exploratory' else ['mechanism','external','exploratory']
    reports={};results={}
    for name in modules:
        root=materialized/name;dest=out/name
        flag='--output' if name=='external' else '--out'
        cmd=[sys.executable,str(root/'run.py'),'replay',flag,str(dest)]
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONOPTIMIZE='0')
        print('Recomputing '+name,flush=True)
        with (out/(name+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env=env)
        reportfile={'mechanism':'REPLAY_VERIFICATION.json','external':'REPLAY_CHECK.json','exploratory':'REPLAY_AUDIT.json'}[name]
        reports[name]=json.loads((dest/reportfile).read_text())
        r=reports[name]
        if name=='mechanism':
            require(r.get('passed') is True and r.get('fixed_arms')==54 and r.get('reference_bootstrap_draws')==4000 and r.get('permutations_per_mode')==1000, 'D3 verification did not pass completely')
            require(r.get('array_groups_bitwise_identical')=={'input_statistics':7,'original_predictions':6,'fixed_predictions':20,'all_predictions':4}, 'D3 array verification incomplete')
            require(sum(x['numeric_values_checked'] for x in r['summary_comparisons'].values())==2399 and all(x['maximum_absolute_difference']<=1e-10 for x in r['summary_comparisons'].values()), 'D3 numerical verification incomplete')
        elif name=='external':
            require(r.get('status')=='PASS' and r.get('summary_numeric_fields')==11619 and r.get('numeric_arrays')==40 and r.get('summary_max_abs_error',1)<1e-12 and r.get('array_max_abs_error',1)<1e-12, 'External verification incomplete')
        else:
            require(r.get('status')=='PASS' and r.get('all_original_and_repair_arms_recomputed') is True and r.get('bootstrap_draws_per_source')==4000 and r.get('bitwise_equal_arrays',0)>0 and r.get('matched_summary_numbers',0)>0, 'Exploratory verification incomplete')
        if name=='mechanism':results[name]={p:json.loads((dest/'results'/p/'summary.json').read_text()) for p in ['fixed','permutation','interaction']}
        elif name=='external':results[name]=json.loads((dest/'summary.json').read_text())
        else:results[name]={p:json.loads((dest/p/'summary.json').read_text()) for p in ['external','d3']}
    export_checks={}
    if 'external' in results:
        old=results['external'];new=normalize(old);export_checks['external']=assert_preserved(old,new);results['external']=new
    if 'exploratory' in results:
        old=results['exploratory']['external'];new=normalize(old);export_checks['exploratory_external']=assert_preserved(old,new);results['exploratory']['external']=new
    report={'status':'PASS','export_numeric_checks':export_checks,'action':args.action,'seconds':time.time()-t,'restoration':restoration,'experiments':reports,'native_ASH_refitted':False,'precision':'Original float64 arrays; byte-lossless transforms only; exact moments preserve the original reduction order.'}
    (out/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');(out/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
