#!/usr/bin/env python3
"""Audit exact pooled-cell replay and export a common external-method input."""
from pathlib import Path
import argparse,hashlib,json,time,fcntl
import numpy as np
from scipy.sparse import load_npz,save_npz,vstack

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def role(d,name,ordered):
    folder=d/'raw/raw_batches'/name;done=json.loads((folder/'COMPLETE.json').read_text());xs=[];ids=[]
    for fname,want in done['batch_files'].items():
        p=folder/fname;assert sha(p)==want;j=json.loads(p.with_suffix('.json').read_text());assert j['sha256']==want
        xs.append(load_npz(p));ids.extend(j['cell_ids'])
    x=vstack(xs,format='csr');ids=np.array(ids);pos=np.searchsorted(ids,ordered);assert np.array_equal(ids[pos],ordered)
    return x[pos]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['D4_Rest','D4_Stim8hr'],required=True);ap.add_argument('--rep',type=int,choices=range(5),default=0);a=ap.parse_args();t=time.monotonic()
    d=Path(__file__).parent/a.task;r=a.rep;out=ROOT/'inputs'/a.task/f'rep{r}';out.mkdir(parents=True,exist_ok=True)
    # The downloader and streaming exporter may reach the same completed role.
    # Serialize each output and reuse only a fully audited, hash-identical export.
    output_lock=(out/'.EXPORT.lock').open('a');fcntl.flock(output_lock.fileno(),fcntl.LOCK_EX)
    if (out/'INPUT_AUDIT.json').exists():
        done=json.loads((out/'INPUT_AUDIT.json').read_text())
        assert done['status']=='EXACT_RAW_REPLAY_PASS' and done['task']==a.task and done['rep']==r
        assert done['source_selection_sha256']==sha(d/'selection.npz')
        assert all(sha(out/name)==expected for name,expected in done['files_sha256'].items())
        print(json.dumps({'status':'REUSED_VERIFIED_EXPORT','task':a.task,'rep':r}),flush=True)
        return
    # A fit launched before shared-lock adoption must finish first. The FR
    # coordinator creates this marker only after that legacy process exits.
    coordination_ready=ROOT/'HEAVY_MEMORY_COORDINATED.json';coordination_start=time.monotonic()
    if not coordination_ready.exists():
        print(json.dumps({'phase':'WAIT_FOR_SHARED_MEMORY_COORDINATION','task':a.task,'rep':r}),flush=True)
    while not coordination_ready.exists():
        if time.monotonic()-coordination_start>10800:raise TimeoutError('FR/export memory coordination not established')
        time.sleep(5)
    heavy_lock=(ROOT/'heavy_memory.lock').open('a')
    print(json.dumps({'phase':'WAIT_FOR_HEAVY_MEMORY_LOCK','task':a.task,'rep':r}),flush=True)
    fcntl.flock(heavy_lock.fileno(),fcntl.LOCK_EX)
    # Resource gate only: retain room for the approximately 1.1 GiB export
    # working set while FR and bounded ASH workers use the same cgroup.
    memory_root=Path('/sys/fs/cgroup');limit_text=(memory_root/'memory.max').read_text().strip()
    if limit_text!='max':
        memory_limit=int(limit_text);minimum_headroom=int(1.6*(1<<30));wait_start=time.monotonic();announced=False
        while memory_limit-int((memory_root/'memory.current').read_text())<minimum_headroom:
            if not announced:
                print(json.dumps({'phase':'WAIT_FOR_MEMORY_HEADROOM','task':a.task,'rep':r,'minimum_bytes':minimum_headroom}),flush=True);announced=True
            if time.monotonic()-wait_start>10800:raise TimeoutError('Insufficient cgroup headroom for input export')
            time.sleep(5)
        print(json.dumps({'phase':'EXPORT_MEMORY_GATE_PASS','task':a.task,'rep':r,'current_bytes':int((memory_root/'memory.current').read_text()),'limit_bytes':memory_limit}),flush=True)
    s=load(d/'selection.npz');m=load(d/'control_model.npz');q=load(d/'query_context.npz');f=load(d/'frozen_task.npz')
    ntc=s['NTC_input'];tr=s['input_cells'][:,r,:];cn=role(d,'NTC_input',ntc);tx=role(d,'A_input' if r==0 else f'input_group{r}',tr.ravel());raw=vstack([cn,tx],format='csr')
    ids=np.r_[ntc,tr.ravel()];tot=np.asarray(raw.sum(1)).ravel();sp=np.searchsorted(s['all_selected_rows'],ids);assert np.array_equal(s['all_selected_rows'][sp],ids);assert np.array_equal(s['selected_total_counts'][sp],tot)
    assert np.array_equal(np.flatnonzero(np.asarray(cn.sum(0)).ravel()>0),m['keep'])
    c=np.log1p(cn[:,m['keep']].toarray().astype(float)*(float(m['scale'])/tot[:len(ntc)])[:,None]).astype(np.float32).astype(float)
    cm=c.mean(0);cv=c.var(0,ddof=1);cmerr=float(np.max(np.abs(cm-m['cm'])));cverr=float(np.max(np.abs(cv-m['cv'])));assert cmerr<1e-12 and cverr<1e-12
    del c
    n=len(tr);g=len(m['keep']);effects=np.empty((n,g));se=np.empty((n,g));target_var=np.empty((n,g));we=np.empty(n);wm=np.empty(n)
    for j in range(n):
        dense=tx[j*20:(j+1)*20,m['keep']].toarray().astype(float);totj=tot[len(ntc)+j*20:len(ntc)+(j+1)*20]
        x=np.log1p(dense*(float(m['scale'])/totj)[:,None]);sums=x.sum(0);y=sums/20-m['cm'];v=x.var(0,ddof=1)
        trace=((x*x).sum()-(sums*sums).sum()/20)/19
        ps=x@m['U'];pm=ps.mean(0)-m['cm']@m['U'];pv=ps.var(0,ddof=1)
        ae=np.mean(y*y);ne=(trace/20+m['cv'].sum()/2000)/g
        am=ae-.9*np.sum(pm*pm)/g;nm=ne-.9*np.sum(pv/20+m['lam']/2000)/g
        we[j]=np.clip(1-ne/max(ae,1e-300),0,1);wm[j]=np.clip(1-nm/max(am,1e-300),0,1)
        effects[j]=y;se[j]=np.sqrt(v/20+m['cv']/2000);target_var[j]=v
    y=effects[:,q['qu']];yerr=float(np.max(np.abs(y-f['original_effects'][r])));weerr=float(np.max(np.abs(we-f['weight_Euclidean'][r])));wmerr=float(np.max(np.abs(wm-f['weight_PertRank'][r])))
    assert yerr<1e-12 and weerr<1e-12 and wmerr<1e-12,(yerr,weerr,wmerr)
    out=ROOT/'inputs'/a.task/f'rep{r}';out.mkdir(parents=True,exist_ok=True);save_npz(out/'counts.npz',raw)
    pos=q['responsive_positions'];columns=m['keep'][q['qu'][pos]]
    np.savez_compressed(out/'axes.npz',labels=np.r_[np.full(len(ntc),-1,dtype=np.int32),np.repeat(np.arange(n,dtype=np.int32),20)],gene_ids=s['gene_ids_full'],target_ids=f['target_ids'],target_names=f['target_names'],query_ids=f['query_gene_ids'][pos],query_blocks=q['blocks_responsive'],query_columns=columns,input_effects=y[:,pos],weights_euclidean=f['weight_Euclidean'][r],weights_pertrank=f['weight_PertRank'][r],cell_ids=ids,reference_effects=f['truth'][:,pos],rep=np.array(r),task=np.array(a.task),scale=m['scale'],input_control_count=np.array(len(ntc)),target_cells_per_group=np.array(20),all_query_ids=f['query_gene_ids'],all_query_columns=m['keep'][q['qu']],all_input_effects=y,all_reference_effects=f['truth'],responsive_positions=pos,tercile_positions=q['tercile_positions'],blocks_terciles=q['blocks_terciles'])
    np.savez_compressed(out/'full_effect_stats.npz',effects=effects,standard_errors=se,target_var=target_var,gene_ids=m['gene_ids_keep'],target_ids=f['target_ids'],query_keep_columns=q['qu'][pos],control_mean=m['cm'],control_mean_raw=cm,control_var=m['cv'],scale=m['scale'],n_control=np.array(2000),n_target=np.array(20),variance_ddof=np.array(1))
    report={'status':'EXACT_RAW_REPLAY_PASS','task':a.task,'rep':r,'targets':n,'cells':len(ids),'control_count':len(ntc),'genes':raw.shape[1],'weight_genes':g,'queries':len(pos),'raw_totals_exact':True,'full_gene_axis_exact':True,'max_abs_error':{'control_mean':cmerr,'control_var':cverr,'query_effects':yerr,'euclidean_weights':weerr,'pertrank_weights':wmerr},'source_selection_sha256':sha(d/'selection.npz'),'files_sha256':{p.name:sha(p)for p in out.glob('*.npz')},'seconds':time.monotonic()-t,'note':'Identical cells for every method; original five input groups on already-opened data. No independent-donor confirmation claim.'}
    report['standard_errors']={'min':float(se.min()),'max':float(se.max()),'zero_count':int(np.count_nonzero(se==0)),'ddof':1,'target_n':20,'control_n':2000}
    (out/'INPUT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
