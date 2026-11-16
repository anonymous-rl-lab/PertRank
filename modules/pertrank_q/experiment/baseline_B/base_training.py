#!/usr/bin/env python3
"""Source-specific reliability-gate development. No held references in fit/predict."""
from pathlib import Path
import sys, json, time, datetime, argparse
ROOT=Path(__file__).resolve().parent
LOCAL_RUNTIME=ROOT.parent/'pertrank_neural_rank_20261115'/'.runtime'
if LOCAL_RUNTIME.is_dir(): sys.path.insert(0,str(LOCAL_RUNTIME))
import numpy as np
import torch
from common import load,sha,dump,subset,inner_ids,ref_support,precision,SOURCES
from gate_model import ReliabilityGate,list_loss
ARMS=('linear_full','neural_full','neural_no_extra_geometry')
SEEDS=(2026111511,2026111512)
CHECKPOINTS=(0,50,100,200,300)
FEATURES=('w0','log_energy_I','absolute_query','log_conditional_SNR','query_SE','guide_absolute_contrast','guide_available','guide_min_fraction','wM','log_energy_M','control_projection_fraction')
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def tensor(a): return torch.as_tensor(a,dtype=torch.float64)

class GateNormalizer:
    def __init__(self,v):
        self.source_feature_names=v['feature_names']
        mask=~v['ownmask']
        self.raw_scale=np.array([max(np.std(row[m]),1e-6) for row,m in zip(v['raw'],mask)])
        self.baseline_scale=np.array([max(np.std(row[m]),1e-6) for row,m in zip(v['base'],mask)])
        se=v['X'][:,:,3]
        self.se_floor=np.array([max(.1*np.median(row[m]),1e-6) for row,m in zip(se,mask)])
        f=self.features(v)
        self.feature_mean=f[mask].mean(0)
        self.feature_scale=np.maximum(f[mask].std(0),1e-6)
    def features(self,v):
        assert np.array_equal(v['feature_names'],self.source_feature_names)
        x=v['X']; raw=np.abs(v['raw']); sd=self.raw_scale[:,None]
        den=np.maximum(x[:,:,3],self.se_floor[:,None])
        return np.stack((x[:,:,5],np.log1p(x[:,:,7]),raw/sd,np.log1p(raw/den),x[:,:,3]/sd,x[:,:,10]/sd,x[:,:,11],x[:,:,12],x[:,:,4],np.log1p(x[:,:,8]),x[:,:,9]),axis=2)
    def tensors(self,v,arm):
        f=(self.features(v)-self.feature_mean)/self.feature_scale
        if arm=='neural_no_extra_geometry': f[:,:,8:]=0
        return dict(features=tensor(f),raw=tensor(v['raw']),anchor=tensor(v['X'][:,:,4]),base=tensor(v['base']),mask=torch.as_tensor(~v['ownmask'],dtype=torch.bool),scale=tensor(self.baseline_scale[:,None]))
    def save(self,p):
        np.savez_compressed(p,source_feature_names=self.source_feature_names,feature_names=np.array(FEATURES),raw_scale=self.raw_scale,baseline_scale=self.baseline_scale,se_floor=self.se_floor,feature_mean=self.feature_mean,feature_scale=self.feature_scale)
    @classmethod
    def from_file(cls,p):
        obj=cls.__new__(cls)
        for k,v in load(p).items(): setattr(obj,k,v)
        return obj

def predict(m,t):
    with torch.no_grad(): return m(t['features'],t['raw'],t['anchor']).numpy()
def objective(m,t,support,k):
    score=m(t['features'],t['raw'],t['anchor'])
    rank=list_loss(score/t['scale'],t['mask'],support[0],support[1],k,.5)
    residual=((score-t['base'])/t['scale'])[t['mask']].square().mean()
    pars=list(m.h.parameters()) if m.mode=='linear' else list(m.h[-1].parameters())
    reg=torch.cat([p.flatten() for p in pars]).square().mean()
    return rank+.05*residual+1e-4*reg

def metrics(m,t,v,k):
    p=predict(m,t)
    with torch.no_grad(): g=m.gate(t['features']).numpy()
    valid=~v['ownmask']
    return dict(precision=float(precision(p,v['reference'],v['ownmask'],k).mean()),max_abs_gate=float(abs(g[valid]).max()),fraction_saturated=float((abs(g[valid])>.99*np.log(3)).mean()),rms_score_change=float(np.sqrt(np.mean((p[valid]-v['base'][valid])**2))))

def fit(v,validation,arm,seed,path,epochs,k):
    if (path/'FIT.json').exists():
        payload=torch.load(path/'weights.pt',weights_only=True)
        m=ReliabilityGate(**payload['config']);m.load_state_dict(payload['state_dict'])
        return m,GateNormalizer.from_file(path/'normalization.npz'),json.loads((path/'curve.json').read_text())
    if path.exists(): raise RuntimeError(f'Incomplete existing fit: {path}; preserve before retry')
    path.mkdir(parents=True)
    start=time.monotonic(); norm=GateNormalizer(v);norm.save(path/'normalization.npz')
    t=norm.tensors(v,arm); vt=norm.tensors(validation,arm) if validation is not None else None
    support=tensor(ref_support(v,k))
    m=ReliabilityGate(11,'linear' if arm=='linear_full' else 'neural',hidden=8,seed=seed)
    assert np.array_equal(predict(m,t),v['base']), 'Initialization is not exact original baseline'
    opt=torch.optim.Adam(m.parameters(),lr=.01)
    curve=[]; grad=0.
    for epoch in range(epochs+1):
        if epoch in CHECKPOINTS or epoch==epochs:
            rec=dict(epoch=epoch,loss=float(objective(m,t,support,k).detach()),gradient_norm=grad,train=metrics(m,t,v,k))
            if validation is not None: rec['validation']=metrics(m,vt,validation,k)
            curve.append(rec)
        if epoch==epochs: break
        opt.zero_grad(); loss=objective(m,t,support,k)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite objective')
        loss.backward()
        for p in m.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite gradient')
        grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),5.));opt.step()
    torch.save(dict(config=m.get_config(),state_dict=m.state_dict()),path/'weights.pt')
    dump(path/'curve.json',curve)
    dump(path/'FIT.json',dict(arm=arm,seed=seed,epochs=epochs,selected_epoch=epochs,k=k,source_specific=True,training_target_ids=v['target_ids'].tolist(),validation_target_ids=None if validation is None else validation['target_ids'].tolist(),duration_seconds=time.monotonic()-start,parameters=sum(p.numel() for p in m.parameters())))
    return m,norm,curve

def prepare():
    if (ROOT/'PLAN_LOCK.json').exists(): raise RuntimeError('Existing plan lock')
    assert json.loads((ROOT/'MODEL_CHECKS.json').read_text())['all_passed']
    tasks={}
    for s in SOURCES:
        v=load(ROOT/'inputs'/f'{s}_neural_inputs.npz')
        for h in (0,1):
            cal=subset(v,np.flatnonzero(v['targethalf']!=h)); held=subset(v,np.flatnonzero(v['targethalf']==h))
            vp=ROOT/'views'/s/f'held{h}';vp.mkdir(parents=True,exist_ok=True)
            ep=ROOT/'evaluation'/s/f'held{h}.npz';ep.parent.mkdir(parents=True,exist_ok=True)
            cp=vp/'calibration.npz';hp=vp/'held_features.npz'
            np.savez_compressed(cp,**cal)
            np.savez_compressed(hp,**{k:a for k,a in held.items() if not k.startswith('reference')})
            np.savez_compressed(ep,**{k:held[k] for k in ('reference','reference_replicates','ownmask','query_ids','target_ids','query_block')})
            groups=inner_ids({s:cal},2026111516);dump(vp/'INNER_SPLIT.json',groups)
            tasks[f'{s}_held{h}']=dict(held=len(held['target_ids']),calibration=len(cal['target_ids']),calibration_sha256=sha(cp),held_features_sha256=sha(hp),evaluation_sha256=sha(ep),inner_split_sha256=sha(vp/'INNER_SPLIT.json'))
    names=['train_gate.py','gate_model.py','ranking_ops.py','common.py','PROTOCOL.md','score_gate.py']
    plan=dict(created_utc=now(),arms=list(ARMS),seeds=list(SEEDS),source_specific=True,tasks=tasks,code_hashes={n:sha(ROOT/n) for n in names},identity='Exploratory previously accessed references; no independent-confirmation claim',torch_version=torch.__version__)
    dump(ROOT/'PLAN_LOCK.json',plan)
    print('Prepared and locked',flush=True)

def run(source):
    plan=json.loads((ROOT/'PLAN_LOCK.json').read_text())
    for n,h in plan['code_hashes'].items(): assert sha(ROOT/n)==h, n
    lockpath=ROOT/'predictions'/f'{source}_PREDICTION_LOCK.json'
    if lockpath.exists(): raise RuntimeError('Source already complete')
    tasks={}
    for h in (0,1):
        task=f'{source}_held{h}';vp=ROOT/'views'/source/f'held{h}'
        cal=load(vp/'calibration.npz');held=load(vp/'held_features.npz')
        groups=json.loads((vp/'INNER_SPLIT.json').read_text())
        assert sha(vp/'INNER_SPLIT.json')==plan['tasks'][task]['inner_split_sha256']
        g=np.array([groups[str(t)] for t in cal['target_ids']]); scores=np.empty((3,2)+held['base'].shape); fits={}
        for ai,arm in enumerate(ARMS):
            for si,seed in enumerate(SEEDS):
                curves=[]
                for ih in (0,1):
                    tr=subset(cal,np.flatnonzero(g!=ih));va=subset(cal,np.flatnonzero(g==ih))
                    fp=ROOT/'fits'/f'{task}_{arm}_seed{seed}_inner{ih}'
                    _,_,curve=fit(tr,va,arm,seed,fp,300,5);curves.append(curve)
                choices=[]
                for ci,epoch in enumerate(CHECKPOINTS):
                    gains=[c[ci]['validation']['precision']-c[0]['validation']['precision'] for c in curves]
                    choices.append(dict(epoch=epoch,gains=gains,mean_gain=float(np.mean(gains)),eligible=all(x>=0 for x in gains)))
                chosen=max((c for c in choices if c['eligible']),key=lambda c:(c['mean_gain'],-c['epoch']))['epoch']
                fp=ROOT/'fits'/f'{task}_{arm}_seed{seed}_outer'
                m,norm,_=fit(cal,None,arm,seed,fp,chosen,10)
                scores[ai,si]=predict(m,norm.tensors(held,arm))
                fits[f'{arm}_seed{seed}']=dict(selected_epoch=chosen,selection=choices,outer_fit=str(fp.relative_to(ROOT)),weights_sha256=sha(fp/'weights.pt'),normalization_sha256=sha(fp/'normalization.npz'))
                print(source,h,arm,seed,'epoch',chosen,flush=True)
        pp=ROOT/'predictions'/f'{task}.npz';pp.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(pp,scores=scores,arm_names=np.array(ARMS),seeds=np.array(SEEDS),**{k:held[k] for k in ('base','raw','euclidean','ownmask','query_ids','query_block','target_ids')})
        tasks[task]=dict(path=str(pp.relative_to(ROOT)),sha256=sha(pp),fits=fits)
    dump(lockpath,dict(plan_sha256=sha(ROOT/'PLAN_LOCK.json'),prediction_time_utc=now(),tasks=tasks))
    print('FROZEN',source,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--source',choices=SOURCES);a=p.parse_args()
    if a.prepare: prepare()
    elif a.source: run(a.source)
    else: p.error('Choose --prepare or --source')
