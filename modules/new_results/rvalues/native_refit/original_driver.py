#!/usr/bin/env python3
"""Native-rvalues refit or exact membership replay on fixed PertRank tasks.

No reference information enters export_inputs() or run_native().
"""
from pathlib import Path
import argparse, csv, hashlib, json, shutil, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

ROOT = Path(__file__).resolve().parent
TASKS = ('External_D3_Rest','External_D1_Rest','External_D2_Rest','D4_Rest','D4_Stim8hr')

def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_files():
    p=ROOT/'FILE_SHA256.json'
    if not p.exists():
        raise RuntimeError('FILE_SHA256.json missing; package assembly is incomplete')
    j=json.loads(p.read_text())
    for name,expected in j['sha256'].items():
        if sha(ROOT/name)!=expected:
            raise RuntimeError('Hash mismatch: '+name)
    return len(j['sha256'])

def membership(score, mask, k):
    s=np.asarray(score,dtype=np.float64).copy()
    assert s.ndim==2 and s.shape==mask.shape
    assert np.isfinite(s[~mask]).all()
    assert np.all(s.shape[1]-mask.sum(1)>=k)
    s[mask]=-np.inf
    th=np.partition(s,s.shape[1]-k,axis=1)[:,-k,None]
    above=s>th; tie=s==th
    p=above.astype(float)+tie*((k-above.sum(1))/tie.sum(1))[:,None]
    assert np.allclose(p.sum(1),k,rtol=0,atol=1e-12)
    assert not p[mask].any()
    return p

def export_inputs(task, rep, dest):
    z=load(ROOT/'data/inputs'/f'{task}.npz')
    effect=z['effect'][rep]; se=z['se'][rep]; q=z['query_ids']; t=z['target_ids']; own=z['own_mask']
    assert effect.shape==se.shape==own.shape==(len(q),len(t))
    assert np.array_equal(own,q[:,None]==t[None,:])
    assert np.isfinite(effect[~own]).all() and np.isfinite(se[~own]).all() and np.all(se[~own]>0)
    with dest.open('w',newline='') as f:
        w=csv.writer(f,delimiter='\t')
        w.writerow(['task','input_id','query_index','query_id','direction','target_index','target_id','effect','se'])
        for qi,gene in enumerate(q):
            for direction in (1,-1):
                for ti,target in enumerate(t):
                    if own[qi,ti]: continue
                    w.writerow([task,rep,qi,gene,direction,ti,target,format(float(effect[qi,ti]),'.17g'),format(float(se[qi,ti]),'.17g')])

def read_native(task, rep, path):
    z=load(ROOT/'data/inputs'/f'{task}.npz'); q=z['query_ids']; t=z['target_ids']; own=z['own_mask']
    nq=len(q); mask=np.concatenate([own,own],axis=0)
    s=np.full(mask.shape,np.nan);seen=np.zeros(mask.shape,dtype=np.uint8)
    with path.open() as f:
        for row in csv.DictReader(f,delimiter='\t'):
            assert row['task']==task and int(row['input_id'])==rep
            qi=int(row['query_index']);ti=int(row['target_index']);di=int(row['direction'])
            assert di in (-1,1) and row['query_id']==q[qi] and row['target_id']==t[ti]
            rr=qi+(nq if di==-1 else 0)
            assert not mask[rr,ti] and seen[rr,ti]==0
            assert row['fit_status'] in ('native_success','boundary_degenerate'), row
            s[rr,ti]=float(row['score']);seen[rr,ti]+=1
    assert np.array_equal(seen,(~mask).astype(np.uint8))
    fits=[]
    with Path(str(path)+'.fits.tsv').open() as f:
        fits=list(csv.DictReader(f,delimiter='\t'))
    assert len(fits)==2*nq
    assert all(r['status'] in ('native_success','boundary_degenerate') for r in fits)
    p=membership(s,mask,int(z['K']))
    return p,fits

def run_native(task,rep,work,rscript):
    d=work/task/f'rep{rep}';d.mkdir(parents=True)
    inp=d/'input.tsv';out=d/'predictions.tsv';export_inputs(task,rep,inp)
    with (d/'native.log').open('w') as log:
        proc=subprocess.run([rscript,'--vanilla',str(ROOT/'code/run_rvalues_precise.R'),str(inp),str(out)],stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f'{task} input {rep} failed; see {d / "native.log"}')
    p,fits=read_native(task,rep,out)
    return task,rep,p,fits

def fit_all(out,rscript,workers):
    work=out/'native_runs';work.mkdir()
    allp={};allfits={};jobs=[]
    for task in TASKS:
        z=load(ROOT/'data/inputs'/f'{task}.npz');nr,nq,nt=z['effect'].shape
        allp[task]=np.empty((nr,2*nq,nt));allfits[task]=[]
        jobs.extend((task,rep) for rep in range(nr))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_native,t,r,work,rscript) for t,r in jobs]
        for f in as_completed(futures):
            task,rep,p,fits=f.result();allp[task][rep]=p;allfits[task].extend(fits)
            print(f'Completed native fit: {task} input {rep}',flush=True)
    pred=out/'membership';pred.mkdir()
    for task,p in allp.items():
        z=load(ROOT/'data/inputs'/f'{task}.npz')
        np.savez_compressed(pred/f'{task}.npz',membership=p,query_ids=z['query_ids'],target_ids=z['target_ids'],input_reps=z['input_reps'],K=z['K'])
        fits=sorted(allfits[task],key=lambda r:(int(r['input_id']),int(r['query_index']),int(r['direction'])))
        with (pred/f'{task}.fits.tsv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fits[0].keys(),delimiter='\t');w.writeheader();w.writerows(fits)
    return pred

def score_all(pred,out):
    reports={};taskdraws={};summaries=[];contrasts=[]
    for task in TASKS:
        inp=load(ROOT/'data/inputs'/f'{task}.npz')
        e=load(ROOT/'data/evaluation'/f'{task}.npz')
        z=load(pred/f'{task}.npz');p=z['membership'];k=int(inp['K'])
        assert np.array_equal(z['query_ids'],inp['query_ids']) and np.array_equal(z['target_ids'],inp['target_ids'])
        nr,nq,nt=inp['effect'].shape
        assert p.shape==(nr,2*nq,nt) and e['reference_membership'].shape==(2*nq,nt)
        mask=np.concatenate([inp['own_mask'],inp['own_mask']],axis=0)
        assert np.allclose(p.sum(2),k,rtol=0,atol=1e-12) and not p[:,mask].any()
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        signed=(p*e['reference_membership'][None]).sum(2).reshape(nr,2,nq).transpose(0,2,1)/k
        new=signed.mean(2)
        names=e['baseline_names'].astype(str).tolist()+['rvalues']
        ss=np.concatenate([e['baseline_query_scores'],new[:,None,:]],axis=1)
        assert ss.shape==(nr,len(names),nq)
        mq=ss.mean(0);means=mq.mean(1);boot=mq@e['bootstrap_weights'].T
        if 'baseline_bootstrap' in e:
            assert np.max(np.abs(boot[:-1]-e['baseline_bootstrap']))<1e-14
        fits=list(csv.DictReader((pred/f'{task}.fits.tsv').open(),delimiter='\t'))
        assert len(fits)==nr*2*nq
        counts={name:sum(r['status']==name for r in fits) for name in ('native_success','boundary_degenerate')}
        assert sum(counts.values())==len(fits)
        cc={}
        for m,name in enumerate(names[:-1]):
            point=float(means[m]-means[-1]);draw=boot[m]-boot[-1];ci=np.quantile(draw,[.025,.975]).tolist()
            label=name+'_minus_rvalues'
            cc[label]={'precision_difference':point,'conditional_query_block_95_interval':ci,'expected_reference_members_difference':k*point}
            contrasts.append([task,label,k,point,*ci,k*point])
        report={'task':task,'K':k,'input_reps':nr,'queries':nq,'candidates':nt,'means':dict(zip(names,map(float,means))),
                'contrasts':cc,'native_fit_status_counts':counts,'analysis_identity':'FIXED_POST_HOC_COMPARATOR_ADDITION',
                'interval_scope':'Conditional paired query-block bootstrap, 2000 draws; directions and input repetitions averaged within query before inference.'}
        reports[task]=report
        (out/f'{task}_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        np.savez_compressed(out/f'{task}_scores.npz',method_names=np.array(names),scores=ss,rvalues_direction_scores=signed,bootstrap=boot,query_ids=inp['query_ids'],query_blocks=inp['query_blocks'])
        for name,val in zip(names,means):summaries.append([task,name,k,float(val),k*float(val)])
        taskdraws[task]=(names,mq,e)
    # Match the original equal-state independent RNG-spawned query-block draws.
    ns,m1,e1=taskdraws['D4_Rest'];nt,m2,e2=taskdraws['D4_Stim8hr'];assert ns==nt
    b1=m1@e1['equal_state_bootstrap_weights'].T;b2=m2@e2['equal_state_bootstrap_weights'].T
    state_boot=(b1+b2)/2;state_means=(m1.mean(1)+m2.mean(1))/2
    d4={'means':dict(zip(ns,map(float,state_means))),'contrasts':{},'scope':'Equal-weighted two states of one D4 donor; original independently spawned state query-block resampling, conditional on measurements.'}
    for j,name in enumerate(ns[:-1]):
        point=float(state_means[j]-state_means[-1]);ci=np.quantile(state_boot[j]-state_boot[-1],[.025,.975]).tolist()
        d4['contrasts'][name+'_minus_rvalues']={'precision_difference':point,'conditional_query_block_95_interval':ci,'expected_reference_members_difference':25*point}
        contrasts.append(['D4_equal_state',name+'_minus_rvalues',25,point,*ci,25*point])
    for name,v in zip(ns,state_means):summaries.append(['D4_equal_state',name,25,float(v),25*float(v)])
    ext_tasks=TASKS[:3];ext_names=list(reports[ext_tasks[0]]['means'])
    ext_means={n:float(np.mean([reports[t]['means'][n] for t in ext_tasks])) for n in ext_names}
    ext={'means':ext_means,'contrasts':{n+'_minus_rvalues':ext_means[n]-ext_means['rvalues'] for n in ext_names[:-1]},'pooled_population_CI':None,'scope':'Equal-weighted three source means, sharing one external RNA reference experiment. No independent-source population CI is constructed.'}
    for name,v in ext_means.items():summaries.append(['External_equal_source',name,10,v,10*v])
    for name,delta in ext['contrasts'].items():contrasts.append(['External_equal_source',name,10,delta,'','',10*delta])
    full={'tasks':reports,'D4_equal_state':d4,'External_equal_source':ext,'contrast_sign':'A_minus_rvalues > 0 favors A; < 0 favors rvalues. Workflow contrasts do not isolate geometry.'}
    (out/'summary.json').write_text(json.dumps(full,indent=2,allow_nan=False)+'\n')
    for filename,header,rows in [('method_scores.csv',['task','method','K','precision','expected_reference_members'],summaries),('paired_contrasts.csv',['task','contrast','K','precision_difference','ci_low','ci_high','expected_reference_members_difference'],contrasts)]:
        with (out/filename).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(header);w.writerows(rows)
    return full

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--mode',choices=['verify','replay','fit'],default='replay')
    a.add_argument('--out',type=Path)
    a.add_argument('--rscript',default='Rscript')
    a.add_argument('--workers',type=int,default=3)
    x=a.parse_args();count=verify_files();print(f'Verified {count} package file hashes',flush=True)
    if x.mode=='verify':return
    if x.out is None or x.out.exists():raise ValueError('--out must name a new output directory')
    x.out.mkdir(parents=True)
    if x.mode=='fit':
        if x.workers<1:raise ValueError('workers must be positive')
        pred=fit_all(x.out,x.rscript,x.workers)
    else:pred=ROOT/'data/membership'
    results=x.out/'results';results.mkdir()
    full=score_all(pred,results)
    print(json.dumps({'D4_equal_state':full['D4_equal_state']['means'],'External_equal_source':full['External_equal_source']['means']},indent=2))

if __name__=='__main__':main()
