#!/usr/bin/env python3
"""Native GPerturb implementation smoke and opt-in synthetic full-shape pilot.

Real input requires a separately frozen D3-only manifest and an explicit flag.
Small smoke invokes the unchanged official training helper; full-shape pilots
invoke its exact loss on 100-cell batches and the entire 1/8 validation subset.
No pilot is a performance benchmark.
"""
from pathlib import Path
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
import signal
import threading

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=HERE/'gperturb_source/GPerturb/GPerturb_model.py'
SOURCE_SHA='29ab3ef8f27e883d46e0a8e54b4783c6ea7fe677d73e04b873d74de90bcd100d'
SEED=20261014

def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def process_tree_rss(pid):
    """Include runtime-launcher descendants, not just the small launcher PID."""
    stack=[pid];seen=set();total=0
    while stack:
        current=stack.pop()
        if current in seen:continue
        seen.add(current)
        try:
            status=Path(f'/proc/{current}/status').read_text()
            total+=int(next(x.split()[1] for x in status.splitlines() if x.startswith('VmRSS:')))*1024
            stack.extend(int(x) for x in Path(f'/proc/{current}/task/{current}/children').read_text().split())
        except (FileNotFoundError,ProcessLookupError,StopIteration):pass
    return total,len(seen)

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def real_tensors(args,np,torch):
    """Only already-opened, frozen D3 rep0; no new cell/reference access."""
    from scipy.sparse import load_npz
    import zipfile
    manifest=json.loads(Path(args.real_input_manifest).read_text())
    if manifest['status']!='FROZEN_REAL_D3_RESOURCE_PILOT_INPUTS' or manifest['dataset']!='D3_Rest' or manifest['rep']!=0:
        raise ValueError('Real pilot requires a frozen D3 rep0 manifest')
    expected={'counts.npz':'d349da549202ba1e4618b684b6d940aaf3eedfa3be18b97ba5472bece830c0d7',
              'axes.npz':'752d9f232b7c4a2ad2dd53f4fcbe12d03f0efb5c3ee40e1cc0a0102096aabb59'}
    paths={}
    for name,item in manifest['files'].items():
        path=Path(item['path']).resolve()
        if file_sha(path)!=item['sha256']:raise ValueError(f'Real pilot source hash: {name}')
        if name in expected and item['sha256']!=expected[name]:raise ValueError('Not the original D3 rep0 raw input')
        if path.suffix=='.npz':
            with zipfile.ZipFile(path) as z:
                if z.testzip() is not None:raise ValueError('Real pilot NPZ CRC failure')
        paths[name]=path
    with np.load(paths['axes.npz'],allow_pickle=False) as z:
        axes={k:z[k] for k in ['labels','gene_ids','target_ids','query_ids','cell_ids','scale','task','rep']}
    with np.load(paths['covariates.npz'],allow_pickle=False) as z:
        cv={k:z[k] for k in z.files}
    required=['log_total_umi','detected_features_per_umi','mitochondrial_fraction']
    if list(cv['covariate_names'].astype(str)[:3])!=required or cv['values'].shape[1]!=26:
        raise ValueError('Require three validated technical covariates plus 23 lane indicators')
    if not np.array_equal(cv['cell_ids'],axes['cell_ids']) or not np.isfinite(cv['values']).all():
        raise ValueError('Real covariate alignment/finite check failed')
    if manifest.get('covariates_metadata_verified') is not True or manifest.get('reference_cells_used') is not False:
        raise ValueError('Real covariate provenance incomplete')
    counts=load_npz(paths['counts.npz']).tocsr()
    if counts.shape!=(26000,18130) or len(axes['target_ids'])!=1200 or str(axes['task'])!='D3_Rest':
        raise ValueError('Not complete frozen D3 development task')
    totals=np.asarray(counts.sum(1)).ravel();detected=np.asarray((counts>0).sum(1)).ravel()
    if not np.allclose(cv['values'][:,0],np.log(totals),rtol=1e-6,atol=1e-6) or not np.allclose(cv['values'][:,1],detected/totals,rtol=1e-6,atol=1e-7):
        raise ValueError('Covariates disagree with frozen raw counts')
    keep=np.flatnonzero(np.asarray((counts>0).sum(0)).ravel())
    genes=axes['gene_ids'][keep]
    if len(keep)!=15616 or not set(axes['query_ids']).issubset(set(genes)):
        raise ValueError('Complete input-positive gene axis differs or loses a query')
    obs=torch.empty((26000,len(keep)),dtype=torch.float32)
    for start in range(0,26000,100):
        dense=counts[start:start+100,keep].toarray().astype(np.float32)
        dense*=np.float32(axes['scale'])/totals[start:start+100,None].astype(np.float32)
        np.log1p(dense,out=dense);obs[start:start+100]=torch.from_numpy(dense)
    del counts,dense
    labels=axes['labels'];cond=torch.zeros((26000,1200),dtype=torch.float32)
    rows=np.flatnonzero(labels>=0);cond[torch.from_numpy(rows),torch.from_numpy(labels[rows].astype(np.int64))]=1.
    if np.count_nonzero(labels==-1)!=2000 or not np.array_equal(np.bincount(labels[rows],minlength=1200),np.full(1200,20)):
        raise ValueError('Real shallow cells/control count changed')
    cov=torch.from_numpy(cv['values'].astype(np.float32))
    return obs,cond,cov,{'manifest_sha256':file_sha(args.real_input_manifest),
        'input_files_sha256':{k:v['sha256'] for k,v in manifest['files'].items()},
        'genes':len(genes),'all_queries_preserved':True,'normalization_scale':float(axes['scale']),
        'preprocessing':'FP32 log1p(raw counts * frozen scale / total UMI), no centering',
        'reference_expression_read':False,'prediction_readout_or_performance_score_saved':False}

def worker(args):
    # This runtime can launch Python through a detached execution proxy. /proc
    # of the supervisor's direct child then measures the proxy, not this worker.
    # Enforce time/RSS inside the worker and publish its own measured high-water
    # mark; retain external monitoring only as a second, nonauthoritative check.
    guard_started=time.monotonic()
    guard_budget=600 if args.worker=='smoke' else 3600
    guard_memory=int((1.25 if args.worker=='smoke' else 6.5)*2**30)
    guard_done=threading.Event()
    telemetry=Path(args.output).with_name(Path(args.output).stem+'_WORKER_GUARD.json')
    def guard():
        while not guard_done.is_set():
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            elapsed=time.monotonic()-guard_started
            state={'status':'RUNNING','worker_pid':os.getpid(),'worker_max_rss_bytes':rss,
                   'elapsed_seconds':elapsed,'memory_guard_bytes':guard_memory,'budget_seconds':guard_budget}
            if elapsed>guard_budget or rss>guard_memory:
                state['status']='BUDGET_EXHAUSTED' if elapsed>guard_budget else 'RESOURCE_MEMORY_GUARD'
                save(telemetry,state);os._exit(124 if elapsed>guard_budget else 125)
            save(telemetry,state)
            guard_done.wait(.1)
    monitor=threading.Thread(target=guard,daemon=True);monitor.start()
    # Imports and allocations happen in the budget-monitored child process.
    import numpy as np
    import torch
    import importlib.metadata as meta
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError('Official source differs from pinned hash')
    spec=importlib.util.spec_from_file_location('pinned_native_gperturb',SOURCE)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.manual_seed(SEED);np.random.seed(SEED)
    synthetic=args.worker=='smoke';real=args.worker=='full-shape-real'
    input_provenance=None
    if real:
        obs,cond,cov,input_provenance=real_tensors(args,np,torch)
        N,G=obs.shape;P=cond.shape[1];B=cov.shape[1]
    else:
        N,G,P,B=(256,64,8,4) if synthetic else (26000,15616,1200,26)
        # Moderate synthetic inputs, not an approximation to the real biology.
        obs=torch.empty((N,G),dtype=torch.float32)
        for start in range(0,N,100):obs[start:start+100].normal_(0.2,0.1)
        cond=torch.zeros((N,P),dtype=torch.float32)
        ntc=32 if synthetic else 2000
        target_indices=torch.arange(N-ntc)%P
        cond[torch.arange(ntc,N),target_indices]=1.
        cov=torch.randn((N,B),dtype=torch.float32)*0.1
    model=native.GPerturb_Gaussian(P,G,B,N,args.width,4,4,torch.tensor(1.))
    report={'status':'RUNNING','mode':args.worker,'created_utc':datetime.now(timezone.utc).isoformat(),
            'official_source_sha256':SOURCE_SHA,'commit':'05d5a431413ff9dc443c82b994b571ecc563426d',
            'model_class':'GPerturb_Gaussian','shapes':{'N':N,'G':G,'P':P,'B':B,'width':args.width},
            'versions':{p:meta.version(p) for p in ['torch','numpy','pandas','matplotlib','sympy','networkx','filelock','jinja2','fsspec']},
            'cpu_threads':args.threads,'cuda_available':torch.cuda.is_available(),
            'parameter_count':sum(p.numel() for p in model.parameters()),
            'real_expression_read':real,'performance_score_computed':False,'checks':{},
            'input_provenance':input_provenance}
    checks=report['checks'];t=time.monotonic()
    def loss(m,o,c,v,test=False):
        # Native defaults copied explicitly; no objective/scaling changes.
        return m.normal_loss(o,c,v,1.,.1,1.,.1,1.,.1,test=test)
    if synthetic:
        other=copy.deepcopy(model)
        optimizer1=torch.optim.Adam(model.parameters(),lr=.001)
        optimizer2=torch.optim.Adam(other.parameters(),lr=.001)
        state=torch.get_rng_state()
        l1=loss(model,obs[:100],cond[:100],cov[:100]);optimizer1.zero_grad();l1.backward()
        grad1=[p.grad.clone() for p in model.parameters()]
        torch.set_rng_state(state)
        l2=other.normal_loss(observation=obs[:100],conditioner=cond[:100],cell_info=cov[:100],
                            nu_1=1.,nu_2=.1,nu_3=1.,nu_4=.1,nu_5=1.,nu_6=.1)
        optimizer2.zero_grad();l2.backward()
        errors=[float((a-p.grad).abs().max()) for a,p in zip(grad1,other.parameters())]
        optimizer1.step();optimizer2.step()
        pe=max(float((a-b).abs().max()) for a,b in zip(model.parameters(),other.parameters()))
        checks['native_loss_and_adapter_loss_same_rng']={'pass':float(l1)==float(l2),'value':float(l1)}
        checks['native_gradient_adapter_equivalence']={'pass':max(errors)==0.,'max_abs_error':max(errors)}
        checks['one_Adam_update_equivalence']={'pass':pe==0.,'max_abs_error':pe}
        del other,optimizer1,optimizer2,grad1,l1,l2
        # An instrumented subclass is unnecessary: register a temporary instance
        # wrapper that only records the exact scalar returned by the native loss.
        # Its objective and official training/validation calls are unchanged.
        history=[];original_loss=model.normal_loss
        def record_loss(*a,**kw):
            value=original_loss(*a,**kw)
            history.append({'validation':bool(kw.get('test',False)),
                            'batch_cells':len(kw.get('observation',a[0] if a else [])),
                            'loss':float(value.detach())})
            return value
        model.normal_loss=record_loss
        np.random.seed(SEED);torch.manual_seed(SEED)
        result=model.GPerturb_train(epoch=2,observation=obs,cell_info=cov,perturbation=cond,
                                   lr=.001,device='cpu')
        model.normal_loss=original_loss
        checks['official_training_helper_completed']={'pass':result is None,'epochs':2,
            'validation_cells':len(model.test_id),'native_Gaussian_return_is_None':result is None}
        checks['finite_all_native_training_validation_losses']={'pass':all(np.isfinite(v['loss']) for v in history)}
        checks['native_validation_is_entire_one_eighth_holdout']={'pass':all(v['batch_cells']==N//8 for v in history if v['validation'])}
        checks['native_training_batch_max100']={'pass':max(v['batch_cells'] for v in history if not v['validation'])==100}
        report['native_loss_history']=history
        model.eval()
        with torch.no_grad():
            design=torch.cat([torch.zeros((1,P)),torch.eye(P)])
            mu,_,_,_,logit,_,_=model(design,None)
            continuous=mu*torch.sigmoid(logit);sparse=mu*(torch.sigmoid(logit)>.5)
            checks['non_targeting_effect_exact_zero']={'pass':bool((continuous[0]==0).all() and (sparse[0]==0).all())}
            checks['native_readouts_finite_shape']={'pass':tuple(continuous.shape)==(P+1,G) and bool(torch.isfinite(continuous).all() and torch.isfinite(sparse).all())}
            # Explicit identity adapter round trip. Effects are target x gene;
            # benchmark output is query x target. No sign or mean-centering change.
            gene_ids=np.array([f'gene_{i}' for i in range(G)]);target_ids=np.array([f'target_{i}' for i in range(P)])
            query_ids=gene_ids[np.arange(0,G,3)[::-1]];candidate_ids=target_ids[::-1]
            gm={s:i for i,s in enumerate(gene_ids)};tm={s:i for i,s in enumerate(target_ids)}
            effect=continuous[1:].numpy();ordered=effect[np.ix_([tm[x] for x in candidate_ids],[gm[x] for x in query_ids])].T
            checks['stable_identity_gene_target_alignment']={'pass':all(ordered[q,t]==effect[tm[tid],gm[qid]] for q,qid in enumerate(query_ids) for t,tid in enumerate(candidate_ids)),
                'returned_axis_order':'query,target','self_exclusion':'deferred to unchanged benchmark scorer by exact gene ID; synthetic target names differ from gene IDs'}
            checks['no_query_or_candidate_dropped']={'pass':ordered.shape==(len(query_ids),P)}
        report['convergence_claim']=False
        report['convergence_note']='Two epochs are an implementation smoke, never evidence of adequate training.'
    else:
        rng=np.random.RandomState(SEED)
        test=sorted(rng.choice(N,size=N//8,replace=False).tolist())
        train=np.array(sorted(set(range(N))-set(test)));rng.shuffle(train)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        batch_times=[]
        for i in range(20):
            ix=train[i*100:(i+1)*100];start=time.monotonic()
            value=loss(model,obs[ix],cond[ix],cov[ix]);optimizer.zero_grad();value.backward();optimizer.step()
            if not bool(torch.isfinite(value)) or not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError('Native pilot numerical failure')
            batch_times.append(time.monotonic()-start)
        model.eval();start=time.monotonic()
        with torch.no_grad():
            # Entire native validation split in one call; never approximated by
            # a sum of mini-batch KL terms.
            value=loss(model,obs[test],cond[test],cov[test],test=True)
        validation_seconds=time.monotonic()-start
        epoch_est=np.mean(batch_times)*int(np.ceil(len(train)/100))+validation_seconds
        report['resource_timing']={'batch_seconds':batch_times,'full_validation_seconds':validation_seconds,
            'estimated_epoch_seconds':float(epoch_est),'estimated_250_epoch_hours':float(epoch_est*250/3600),
            'estimated_500_epoch_hours':float(epoch_est*500/3600),'native_validation_loss':float(value)}
        checks['finite_native_full_validation']={'pass':bool(torch.isfinite(value))}
        report['convergence_claim']=False
    if not all(v['pass'] for v in checks.values()):
        raise AssertionError('Implementation check failed')
    report.update(status='IMPLEMENTATION_SMOKE_PASS' if synthetic else ('REAL_FULL_SHAPE_RESOURCE_PILOT_COMPLETE' if real else 'SYNTHETIC_FULL_SHAPE_RESOURCE_PILOT_COMPLETE'),
                  elapsed_seconds=time.monotonic()-t,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    guard_done.set();monitor.join(timeout=1)
    save(telemetry,{'status':'WORKER_COMPLETED_WITHIN_BUDGET','worker_pid':os.getpid(),
        'worker_max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'elapsed_seconds':time.monotonic()-guard_started,'budget_seconds':guard_budget,'memory_guard_bytes':guard_memory})
    save(args.output,report)

def supervise(args):
    output=Path(args.output)
    if output.exists():raise FileExistsError('Use a new output path; do not overwrite a prior run')
    budget=600 if args.mode=='smoke' else 3600
    memlimit=int((1.25 if args.mode=='smoke' else 6.5)*2**30)
    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',args.mode,'--width',str(args.width),
         '--threads',str(args.threads),'--output',str(output)]
    if args.real_input_manifest:cmd+=['--real-input-manifest',str(args.real_input_manifest)]
    output.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();peak=0;reason=None;descendant_count=1
    with output.with_suffix('.log').open('x') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while p.poll() is None:
            elapsed=time.monotonic()-start
            rss,count=process_tree_rss(p.pid)
            peak=max(peak,rss);descendant_count=max(descendant_count,count)
            if elapsed>budget or rss>memlimit:
                reason='BUDGET_EXHAUSTED' if elapsed>budget else 'RESOURCE_MEMORY_GUARD'
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                break
            time.sleep(.25)
    result={'status':'SUPERVISOR_COMPLETE' if p.returncode==0 and reason is None else 'PILOT_NOT_ELIGIBLE',
            'child_exit_code':p.returncode,'reason':reason,'wall_seconds':time.monotonic()-start,
            'peak_polled_rss_bytes':peak,'budget_seconds':budget,'memory_guard_bytes':memlimit,
            'rss_accounting':'Sum of native launcher process and all descendants; max RSS of worker also recorded separately.',
            'max_process_tree_size':descendant_count,
            'method_defeat_claim':False,'real_expression_read':args.mode=='full-shape-real'}
    telemetry=output.with_name(output.stem+'_WORKER_GUARD.json')
    if telemetry.exists():
        result['authoritative_worker_guard']=json.loads(telemetry.read_text())
        result['proxy_rss_is_not_worker_rss']=True
    save(output.with_name(output.stem+'_SUPERVISOR.json'),result)
    print(json.dumps(result,indent=2))
    if p.returncode!=0:raise SystemExit(1)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['smoke','full-shape-synthetic','full-shape-real'],default='smoke')
    ap.add_argument('--worker',choices=['smoke','full-shape-synthetic','full-shape-real'])
    ap.add_argument('--width',type=int,choices=[700,1000],default=700)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--allow-heavy-synthetic-pilot',action='store_true')
    ap.add_argument('--allow-real-pilot',action='store_true')
    ap.add_argument('--real-input-manifest',type=Path)
    args=ap.parse_args()
    if args.worker:worker(args);return
    if args.mode!='smoke':
        if args.mode=='full-shape-synthetic' and not args.allow_heavy_synthetic_pilot:ap.error('Explicit --allow-heavy-synthetic-pilot required; not part of implementation smoke')
        if args.mode=='full-shape-real' and (not args.allow_real_pilot or args.real_input_manifest is None):ap.error('Explicit --allow-real-pilot and frozen --real-input-manifest required')
        with (ROOT/'heavy_memory.lock').open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX);supervise(args)
    else:supervise(args)

if __name__=='__main__':main()
