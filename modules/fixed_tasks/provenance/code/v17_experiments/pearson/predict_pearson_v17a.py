#!/usr/bin/env python3
"""Input-only frozen control-fitted Pearson scalar predictions; no ranking/truth.

Read original 2,000 NTC and frozen R20 target rows. Only original query-coordinate
outputs are retained. Sequential donors, bounded cell chunks, no target covariance.
"""
from pathlib import Path
import argparse, gc, hashlib, json, os, resource, time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
from scipy import linalg
from scipy.sparse import load_npz

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
R20=ROOT/'v16_execution/assets/r20/PertRank_R20_20261006/new_experiment'
INV=ROOT/'v16_execution/inventory'
PROTOCOL=ROOT/'v17_experiments/protocol/EXECUTION_PROTOCOL_v17_COMPETITION.md'
PROTOCOL_SHA='9f0963a2206bac8b84b5aa6e8ecde249ce0afa5f07de886b00cf691fc328e784'
CODE_FREEZE_PATH=HERE/'CODE_FREEZE_v17a_RUN.json'
DONORS=('D3_Rest','D1_Rest','D2_Rest')
THETA=100.; NC=2000; K=5; BETA=.9; CLIP=np.sqrt(NC)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def check(name,x,y,errors,tol=1e-9):
    assert np.shape(x)==np.shape(y),(name,np.shape(x),np.shape(y))
    e=float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) if np.size(x) else 0.
    errors[name]=max(errors.get(name,0.),e)
    assert np.isfinite(e) and e<=tol,(name,e,tol)

def residual(counts,total,p):
    """All cells share NTC proportions, theta and clipping bound."""
    mu=np.asarray(total,dtype=np.float64)[:,None]*p[None,:]
    assert np.all(mu>0)
    out=(np.asarray(counts,dtype=np.float64)-mu)/np.sqrt(mu+mu*mu/THETA)
    return np.clip(out,-CLIP,CLIP)

def scalar(a,nu):
    assert np.isfinite(a).all() and np.isfinite(nu).all()
    assert np.min(a)>=-1e-10 and np.min(nu)>=-1e-10
    a=np.maximum(a,0);nu=np.maximum(nu,0)
    w=np.zeros_like(a);pos=a>0
    w[pos]=np.clip(1-nu[pos]/a[pos],0,1)
    return w

def geometry(z):
    """Exact top-five of the sample covariance via the small NTC Gram matrix."""
    n,g=z.shape;mean=z.mean(0);zc=z-mean
    gram=zc@zc.T/(n-1)
    ev,left=linalg.eigh(gram,subset_by_index=[n-K,n-1],driver='evr',check_finite=False)
    ev=ev[::-1];left=left[:,::-1]
    assert ev[-1]>0
    u=(zc.T@left)/np.sqrt((n-1)*ev)[None,:]
    for k in range(K):
        if u[np.argmax(np.abs(u[:,k])),k]<0:u[:,k]*=-1
    err={}
    check('U_orthogonality',u.T@u,np.eye(K),err,tol=1e-10)
    covariance_u=zc.T@(zc@u)/(n-1)
    eigerr=float(np.linalg.norm(covariance_u-u*ev)/max(1.,np.linalg.norm(u*ev)))
    assert eigerr<1e-10;err['relative_eigen_residual']=eigerr
    variance=np.sum(zc*zc,axis=0)/(n-1)
    projection_variance=np.var(z@u,axis=0,ddof=1)
    check('empirical_eigenvalue',projection_variance,ev,err,tol=1e-8)
    return mean,variance,u,ev,err

def freeze():
    assert sha(PROTOCOL)==PROTOCOL_SHA
    f=CODE_FREEZE_PATH;assert not f.exists(),'never overwrite code freeze'
    dump(f,{'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'protocol_sha256':PROTOCOL_SHA,'files':{str(Path(__file__).relative_to(ROOT)):sha(__file__)}})
    print('CODE_FROZEN',sha(f),flush=True)

def verify_freeze():
    f=CODE_FREEZE_PATH;r=json.loads(f.read_text())
    assert sha(PROTOCOL)==r['protocol_sha256']==PROTOCOL_SHA
    for p,h in r['files'].items():assert sha(ROOT/p)==h,('code changed',p)
    return sha(f)

def legacy_control_identity(donor,x,m,oldlog,c,v,errors):
    """Reproduce archived precision conventions, not a different-control proxy.

    D3's old cell-range worker sums were float32; D1/D2 retained float32 rows
    then computed float64 moments. Pearson itself always uses float64 raw rows.
    """
    if donor=='D3_Rest':
        ip=np.load(R20/'read'/donor/'indptr.npy',mmap_mode='r')
        bounds=np.linspace(0,len(ip)-1,5).astype(np.int64)
        assert np.array_equal(bounds,np.array([0,694631,1389262,2083893,2778524]))
        ss=np.zeros((4,x.shape[1]),np.float32);qq=np.zeros_like(ss)
        assert np.all(np.diff(m['cell_ids'])>0)
        for row,cell in enumerate(m['cell_ids']):
            w=int(np.searchsorted(bounds,cell,side='right')-1)
            lo,hi=x.indptr[row:row+2];gi=x.indices[lo:hi]
            z=np.log1p(x.data[lo:hi].astype(float)*(float(m['scale'])/float(m['total_counts'][row])))
            ss[w,gi]+=z;qq[w,gi]+=z*z
        s=np.zeros(x.shape[1]);q=np.zeros_like(s)
        for w in range(4):s+=ss[w];q+=qq[w]
        z=np.load(ROOT/'v16_execution/assets/canonical/Pertrescue_v6/code/repro_v10/data/e4_tcell.npz',allow_pickle=False)
        check('legacy_control_full_sum',s,z['audit_ctrl_s'][0],errors,tol=0)
        check('legacy_control_full_sumsq',q,z['audit_ctrl_q'][0],errors,tol=0)
        check('legacy_control_mean',(s/NC)[c['keep']],c['cm_keep'],errors,tol=0)
        check('legacy_control_variance',((q-s*s/NC)/(NC-1))[c['keep']],v['cv'],errors,tol=0)
    else:
        legacy=oldlog.astype(np.float32).astype(np.float64)
        check('legacy_control_mean',legacy.mean(0),c['cm_keep'],errors,tol=1e-12)
        check('legacy_control_variance',legacy.var(0,ddof=1),v['cv'],errors,tol=1e-12)

def controls(donor,ntc_dir,c,errors):
    rp=ntc_dir/'original_input_ntc_raw.npz'
    mp=ntc_dir/'original_input_ntc_metadata.npz'
    rr=json.loads((ntc_dir/'FULL_READ_PASS.json').read_text())
    assert rr['status']=='FULL_READ_PASS' and rr['donor']==donor
    assert rr['raw_sha256']==sha(rp) and rr['metadata_sha256']==sha(mp)
    x=load_npz(rp);m=np.load(mp,allow_pickle=False)
    assert x.shape==(NC,len(c['gene_ids_full']))
    assert str(m['donor'])==donor
    assert str(m['context_sha256'])==sha(R20/'selection'/donor/'context.npz')
    assert str(m['control_ids_sha256'])==hashlib.sha256(m['cell_ids'].tobytes()).hexdigest()
    assert float(m['scale'])==float(c['scale'])
    assert np.array_equal(m['gene_ids_full'],c['gene_ids_full'])
    keep=c['keep'];assert np.array_equal(m['keep'],keep)
    assert len(np.unique(m['cell_ids']))==NC
    assert np.all(x.data>=0) and np.all(x.data==np.floor(x.data))
    total=np.asarray(x.sum(1)).ravel().astype(float)
    assert np.array_equal(total,m['total_counts']) and np.all(total>0)
    p=np.asarray(x[:,keep].sum(0)).ravel().astype(float)/total.sum()
    assert np.all(p>0),'unexpected zero NTC proportion on frozen positive-expression axis'
    dense=x[:,keep].toarray().astype(float)
    oldlog=np.log1p(dense*(float(c['scale'])/total)[:,None])
    v=np.load(INV/(donor+'_control_gene_moments.npz'),allow_pickle=False)
    legacy_control_identity(donor,x,m,oldlog,c,v,errors)
    errors['descriptive_float64_vs_archived_control_mean']=float(np.max(abs(oldlog.mean(0)-c['cm_keep'])))
    errors['descriptive_float64_vs_archived_control_variance']=float(np.max(abs(oldlog.var(0,ddof=1)-v['cv'])))
    del oldlog
    z=residual(dense,total,p)
    mean,var,u,ev,err=geometry(z);errors.update(err)
    # Direct slow formula on selected entries does not share vectorized implementation.
    slow=[];fast=[]
    for i in (0,17,1999):
        for g in (0,len(keep)//2,len(keep)-1):
            mu=float(total[i])*float(p[g])
            rr=(float(dense[i,g])-mu)/np.sqrt(mu+mu*mu/100.)
            slow.append(min(CLIP,max(-CLIP,rr)));fast.append(z[i,g])
    check('independent_control_PR_formula',np.array(slow),np.array(fast),errors,tol=1e-12)
    out={'p':p,'mean':mean,'variance':var,'U':u,'eigenvalues':ev,
         'control_ids':m['cell_ids'],'control_projection_mean':mean@u,
         'clip_fraction':float(np.mean(np.abs(z)>=CLIP)),
         'total_UMI_sum':float(total.sum()),'ntc_raw_sha256':sha(rp),'ntc_metadata_sha256':sha(mp)}
    del dense,z,x;gc.collect()
    return out

def run(donor,ntc_dir,targets,cell_chunk):
    start=time.monotonic();fhash=verify_freeze();errors={}
    cp=R20/'selection'/donor/'context.npz';sp=cp.parent/'selection.npz'
    ap=R20/'read'/donor/'aggregate.npz'
    c=np.load(cp,allow_pickle=False);s=np.load(sp,allow_pickle=False);old=np.load(ap,allow_pickle=False)
    assert str(c['donor'])==donor and int(c['n_control'])==NC
    assert str(old['context_sha256'])==sha(cp) and str(old['selection_sha256'])==sha(sp)
    ctrl=controls(donor,ntc_dir,c,errors)
    keep=c['keep'];g=len(keep);nall=len(c['internal_ids']);nt=nall if targets==0 else min(targets,nall)
    idx=s['internal_positions'][:nt];assert idx.shape==(nt,5,20)
    assert len(np.unique(idx))==nt*100
    assert len(np.intersect1d(s['cell_ids'][idx.ravel()],ctrl['control_ids']))==0
    group=np.full(len(s['cell_ids']),-1,np.int32)
    group[idx.ravel()]=np.repeat(np.arange(nt*5),20)
    seen=np.zeros(len(group),bool);n=np.zeros(nt*5,int)
    # Two sums, not four full gene-moment arrays. Squared norms suffice for trace.
    sums=np.zeros((nt*5,g));oldsum=np.zeros_like(sums)
    normsq=np.zeros(nt*5);oldnormsq=np.zeros(nt*5)
    ps=np.zeros((nt*5,K));pq=np.zeros_like(ps);oldps=np.zeros_like(ps);oldpq=np.zeros_like(ps)
    hashes=[];clipped=0;values=0
    sharddir=R20/'read'/donor/'raw_shards'
    for mp in sorted(sharddir.glob('rows_*_metadata.npz')):
        meta=np.load(mp,allow_pickle=False);positions=meta['plan_positions']
        loc=np.flatnonzero(group[positions]>=0)
        if not len(loc):continue
        xp=mp.with_name(mp.name.replace('_metadata.npz','.npz'))
        h=sha(xp);rec=json.loads(xp.with_suffix('.json').read_text());assert h==rec['counts_sha256']
        hashes.append({'file':str(xp.relative_to(ROOT)),'sha256':h})
        assert np.array_equal(meta['cell_ids'],s['cell_ids'][positions])
        pos=positions[loc];assert not seen[pos].any();seen[pos]=True
        x=load_npz(xp)[loc];total=s['total_counts'][pos]
        assert np.array_equal(np.asarray(x.sum(1)).ravel(),total)
        assert np.all(x.data>=0) and np.all(x.data==np.floor(x.data))
        for lo in range(0,len(loc),cell_chunk):
            hi=min(lo+cell_chunk,len(loc));gg=group[pos[lo:hi]]
            dense=x[lo:hi,keep].toarray().astype(float)
            z=residual(dense,total[lo:hi],ctrl['p']);pr=z@ctrl['U']
            np.add.at(sums,gg,z);np.add.at(normsq,gg,np.einsum('ij,ij->i',z,z))
            np.add.at(ps,gg,pr);np.add.at(pq,gg,pr*pr)
            clipped+=int(np.sum(np.abs(z)>=CLIP));values+=z.size
            zl=np.log1p(dense*(float(c['scale'])/total[lo:hi])[:,None]);pl=zl@c['U'][:,:K]
            np.add.at(oldsum,gg,zl);np.add.at(oldnormsq,gg,np.einsum('ij,ij->i',zl,zl))
            np.add.at(oldps,gg,pl);np.add.at(oldpq,gg,pl*pl);np.add.at(n,gg,1)
        print(json.dumps({'donor':donor,'targets':nt,'shards':len(hashes),'cells':int(n.sum()),'seconds':time.monotonic()-start}),flush=True)
    assert np.array_equal(seen,group>=0) and np.all(n==20)
    qu=c['qu'];og={};means=oldsum/20
    og['vb']=(oldnormsq-np.einsum('ij,ij->i',oldsum,oldsum)/20)/(19*g)
    means-=c['cm_keep'];og['m2']=np.mean(means*means,axis=1);og['Yq']=means[:,qu].copy()
    og['pmean']=oldps/20-c['proj_control'][:K];og['pvar']=(oldpq-oldps*oldps/20)/19
    for name,v in og.items():check('old_target_'+name,v,old['internal_'+name][:nt].reshape(v.shape),errors)
    oldYq=og['Yq'].reshape(nt,5,len(qu));del means,oldsum;gc.collect()
    trace_target=(normsq-np.einsum('ij,ij->i',sums,sums)/20)/19
    means=sums/20;means-=ctrl['mean'];aE=np.mean(means*means,axis=1)
    qmean=ps/20-ctrl['control_projection_mean'];qvar=(pq-ps*ps/20)/19
    check('projection_of_effect_mean',means@ctrl['U'],qmean,errors,tol=1e-9)
    assert trace_target.min()>-1e-7 and qvar.min()>-1e-8
    trace_target=np.maximum(trace_target,0);qvar=np.maximum(qvar,0)
    nuE=(trace_target/20+ctrl['variance'].sum()/NC)/g
    directional_a=np.sum(qmean*qmean,axis=1)/g
    directional_nu=np.sum(qvar/20+ctrl['eigenvalues']/NC,axis=1)/g
    aM=aE-BETA*directional_a;nuM=nuE-BETA*directional_nu
    wE=scalar(aE,nuE);wM=scalar(aM,nuM)
    check('beta0_PR_E',scalar(aE-0*directional_a,nuE-0*directional_nu),wE,errors,tol=0)
    # Dense square-root transform identity without constructing a G by G matrix.
    probe=means[:10];a=1-np.sqrt(1-BETA);az=probe-a*(probe@ctrl['U'])@ctrl['U'].T
    check('M_energy_square_root',np.mean(az*az,axis=1),aM[:10],errors,tol=1e-9)
    del means,sums;gc.collect()
    report={'donor':donor,'identity':'input-only; no reference/truth arrays, ranking, or performance scores read',
      'targets':nt,'full_cohort':nt==nall,'genes':g,'groups':5,'cells_per_group':20,
      'theta':THETA,'clip':float(CLIP),'k':K,'beta':BETA,'n_control':NC,
      'protocol_sha256':PROTOCOL_SHA,'code_freeze_sha256':fhash,'input_parity':errors,
      'input_hashes':{'context':sha(cp),'selection':sha(sp),'aggregate':sha(ap),
                      'ntc_raw':ctrl['ntc_raw_sha256'],'ntc_metadata':ctrl['ntc_metadata_sha256']},
      'raw_shards':hashes,'runtime_seconds':time.monotonic()-start,
      'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'control_residual_clip_fraction':ctrl['clip_fraction'],
      'target_residual_clip_fraction':clipped/values,
      'control_top5_trace_fraction':float(ctrl['eigenvalues'].sum()/ctrl['variance'].sum()),
      'control_total_UMI_sum':ctrl['total_UMI_sum']}
    outdir=HERE/'predictions' if targets==0 else HERE/'smoke'/str(targets)
    outdir.mkdir(parents=True,exist_ok=True)
    p=outdir/(donor+'_predictions.npz');assert not p.exists(),'prediction already exists'
    wE=wE.reshape(nt,5).T;wM=wM.reshape(nt,5).T
    pred=np.array([(oldYq*w.T[:,:,None]).transpose(1,2,0) for w in (wE,wM)])
    np.savez_compressed(p,arm_names=np.array(['PR_E','PR_M']),w_PR_E=wE,w_PR_M=wM,
        predictions=pred,target_ids=c['internal_ids'][:nt],query_ids=c['query_gene_ids'],
        a_E=aE.reshape(nt,5).T,nu_E=nuE.reshape(nt,5).T,
        a_M=aM.reshape(nt,5).T,nu_M=nuM.reshape(nt,5).T,
        control_U=ctrl['U'],control_eigenvalues=ctrl['eigenvalues'],
        control_residual_mean=ctrl['mean'],control_residual_variance=ctrl['variance'],
        control_p=ctrl['p'],gene_ids=c['gene_ids_keep'],control_ids=ctrl['control_ids'])
    report.update(prediction_path=str(p.relative_to(ROOT)),prediction_sha256=sha(p),
                  context_path=str(cp.relative_to(ROOT)),context_sha256=sha(cp))
    dump(outdir/(donor+'_record.json'),report)
    print(json.dumps({'status':'PREDICTIONS_READY' if targets==0 else 'SMOKE_PASS','donor':donor,
          'targets':nt,'prediction_sha256':sha(p),'seconds':report['runtime_seconds'],
          'maximum_input_error':max(errors.values()),'performance_computed':False}),flush=True)

def selftest():
    rng=np.random.default_rng(2026);c=rng.poisson(np.linspace(.2,8.,19),(30,19));L=c.sum(1)
    p=c.sum(0)/L.sum();r=residual(c,L,p)
    for i in range(30):
        for j in range(19):
            mu=float(L[i])*p[j];v=(c[i,j]-mu)/np.sqrt(mu+mu*mu/100)
            assert abs(r[i,j]-np.clip(v,-CLIP,CLIP))<1e-12
    z=rng.normal(size=(30,19));m,cv,u,ev,e=geometry(z)
    check('full_covariance',(np.cov(z,rowvar=False)@u),u*ev,e,tol=1e-10)
    target=rng.normal(size=(20,19));v=np.cov(target,rowvar=False)/20+np.cov(z,rowvar=False)/30
    y=target.mean(0)-m;M=np.eye(19)-.9*u@u.T
    tr=np.var(target,axis=0,ddof=1).sum()/20+cv.sum()/30
    pn=np.var(target@u,axis=0,ddof=1)/20+ev/30
    check('direct_projected_noise',np.array(tr-.9*pn.sum()),np.array(np.trace(M@v)),e,tol=1e-10)
    check('direct_projected_energy',np.array(y@y-.9*np.sum((y@u)**2)),np.array(y@M@y),e,tol=1e-10)
    check('beta0',scalar(np.array([y@y]),np.array([tr])),scalar(np.array([y@y-0]),np.array([tr-0])),e,tol=0)
    dump(HERE/'SYNTHETIC_SMOKE.json',{'status':'PASS','no_real_weights_or_scores':True,'errors':e,'script_sha256':sha(__file__)})
    print('SYNTHETIC_SMOKE_PASS',json.dumps(e),flush=True)

def lockall():
    fh=verify_freeze();records={}
    for d in DONORS:
        rp=HERE/'predictions'/(d+'_record.json');r=json.loads(rp.read_text())
        assert r['full_cohort'] and r['code_freeze_sha256']==fh
        assert sha(ROOT/r['prediction_path'])==r['prediction_sha256']
        records[d]={k:r[k] for k in ['prediction_path','prediction_sha256','context_path','context_sha256']}
        records[d].update(record_path=str(rp.relative_to(ROOT)),record_sha256=sha(rp))
    p=HERE/'ALL_PREDICTIONS_LOCK.json';assert not p.exists()
    dump(p,{'identity':'three opened donors Pearson common-output predictions before new score evaluation',
            'code_freeze_sha256':fh,'protocol_sha256':PROTOCOL_SHA,'donors':records,
            'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    print('ALL_PREDICTIONS_LOCKED',sha(p),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--freeze',action='store_true');ap.add_argument('--lock-all',action='store_true')
    ap.add_argument('--donor',choices=DONORS);ap.add_argument('--ntc-dir',type=Path);ap.add_argument('--targets',type=int,default=0);ap.add_argument('--cell-chunk',type=int,default=128)
    a=ap.parse_args()
    if a.selftest:selftest()
    elif a.freeze:freeze()
    elif a.lock_all:lockall()
    else:
        assert a.donor and a.ntc_dir
        run(a.donor,a.ntc_dir,a.targets,a.cell_chunk)
