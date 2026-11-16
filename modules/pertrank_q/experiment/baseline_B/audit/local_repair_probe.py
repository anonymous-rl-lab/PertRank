#!/usr/bin/env python3
"""Fixed post-score local-list repair; no training, window scan or primary replacement."""
from pathlib import Path
import sys,json,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common import load,sha,dump,membership,block_interval,SOURCES
from score_boundary import checks_before_reference,ARMS,SEEDS,K
M=2

def capacity_select(score,capacity,budget):
    """Descending selection; exact-score ties get proportional capacity."""
    score=np.asarray(score,float);capacity=np.asarray(capacity,float)
    assert score.shape==capacity.shape and score.ndim==2
    assert np.isfinite(score).all() and np.isfinite(capacity).all()
    assert capacity.min()>=-1e-14 and np.min(capacity.sum(1))>=budget-1e-12
    out=np.zeros_like(capacity)
    for q in range(len(score)):
        remaining=float(budget);valid=capacity[q]>0
        for value in np.unique(score[q,valid])[::-1]:
            take=valid&(score[q]==value);mass=capacity[q,take].sum();used=min(remaining,mass)
            if used>0:out[q,take]=capacity[q,take]*(used/mass)
            remaining-=used
            if remaining<=0:break
        assert abs(remaining)<1e-10
    assert np.max(abs(out.sum(1)-budget))<1e-10
    assert np.min(out)>=0 and np.max(out-capacity)<1e-10
    return out

def repair(scores,base,own,k=K,m=M):
    assert 0<m<k and np.min((~own).sum(1))>=k+m
    core=membership(base,own,k-m);cap=membership(base,own,k+m)-core
    assert cap.min()>=-1e-14 and np.max(abs(cap.sum(2)-2*m))<1e-10
    out=np.stack([core[d]+capacity_select(sign*scores,cap[d],m) for d,sign in enumerate((1.,-1.))])
    assert np.isfinite(out).all() and np.max(abs(out.sum(2)-k))<1e-10
    assert out.min()>=0 and out.max()<=1+1e-12 and np.all(out[:,own]==0)
    assert np.min(out-core)>=-1e-12 and np.max(out-core-cap)<=1e-12
    return out,core,cap

def synthetic_checks():
    rng=np.random.default_rng(2026111527);cases=0;err=0.
    for k,m,t in [(3,1,9),(5,2,15),(10,2,27)]:
        for _ in range(20):
            q=5;base=rng.integers(-3,4,(q,t)).astype(float);scores=rng.integers(-3,4,(q,t)).astype(float)
            own=np.zeros((q,t),bool);own[np.arange(q),rng.integers(t,size=q)]=True
            identity,_,_=repair(base,base,own,k,m);error=np.max(abs(identity-membership(base,own,k)));err=max(err,float(error));assert error<1e-12
            out,_,_=repair(scores,base,own,k,m);perm=rng.permutation(t)
            permuted,_,_=repair(scores[:,perm],base[:,perm],own[:,perm],k,m)
            assert np.allclose(out[:,:,perm],permuted,rtol=0,atol=1e-12)
            flip,_,_=repair(-scores,-base,own,k,m);assert np.allclose(out[::-1],flip,rtol=0,atol=1e-12);cases+=1
    base=np.zeros((2,15));own=np.zeros_like(base,bool);own[0,0]=True
    out,_,_=repair(base,base,own,5,2);expect=(~own)*5/(~own).sum(1)[:,None]
    assert np.allclose(out,np.stack([expect,expect]),atol=1e-12,rtol=0)
    c=np.array([[.2,.3,.5,1.,0.]]);assert np.array_equal(capacity_select(np.zeros_like(c),c,1.),c/2)
    return dict(passed=True,random_tied_cases=cases,baseline_identity_max_error=err,all_tie_case=True,
        proportional_capacity_tie=True,permutation_invariant=True,sign_flip_equivariant=True,
        identity_note='Mathematically exact baseline membership; floating fractional arithmetic checked to1e-12.')

def paired(x,blocks):return dict(delta=float(x.mean()),conditional95=block_interval(x,blocks))
def summarize(a):
    pq,spq,uq,suq,baseq=[a[k] for k in ('local','seed_local','unrestricted','seed_unrestricted','base')];blocks=a['query_block']
    z=dict(original_precision=float(baseq.mean()),arms={},local_contrasts={},seed_local_contrasts={})
    for i,arm in enumerate(ARMS):
        z['arms'][arm]=dict(local_precision=float(pq[i].mean()),unrestricted_precision=float(uq[i].mean()),
            local_vs_original=paired(pq[i]-baseq,blocks),local_vs_unrestricted=paired(pq[i]-uq[i],blocks),seeds={})
        for j,seed in enumerate(SEEDS):
            z['arms'][arm]['seeds'][str(seed)]=dict(local_precision=float(spq[i,j].mean()),unrestricted_precision=float(suq[i,j].mean()),
                local_vs_original=paired(spq[i,j]-baseq,blocks),local_vs_unrestricted=paired(spq[i,j]-suq[i,j],blocks))
    for i in (1,2):
        name=ARMS[i]+'_minus_'+ARMS[0];z['local_contrasts'][name]=paired(pq[i]-pq[0],blocks)
        z['seed_local_contrasts'][name]={str(seed):paired(spq[i,j]-spq[0,j],blocks) for j,seed in enumerate(SEEDS)}
    return z

def main():
    out=ROOT/'audit/local_repair'
    if out.exists():raise RuntimeError('Preserve prior diagnostic; no retuning or window search')
    checks=synthetic_checks();primary_files={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'results').rglob('*') if p.is_file()}
    plan,packets,inputchecks=checks_before_reference();out.mkdir();(out/'predictions').mkdir()
    dump(out/'PLAN.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        identity='POST HOC fixed local-list repair after reference-aware backtrace; not a new primary model or confirmation',
        rule='core=membership(base,K-2); cap=membership(base,K+2)-core; select residual2 by modelscore with fractional caps and proportional allocation inside exact ties',
        K=K,m=M,arms=list(ARMS),seeds=list(SEEDS),ensemble='average seed scores before local membership repair',
        constraints='preserve core and far exclusion; sumK; each membership<=1; own excluded',
        no_window_search=True,no_model_selection=True,no_training=True,
        score_representation='two local directional memberships, not a single scalar full ranking',primary_C_result_remains_failed=True,
        script_sha256=sha(__file__),primary_plan_sha256=sha(ROOT/'PLAN_LOCK.json'),primary_result_hashes=primary_files,
        synthetic_checks=checks,input_checks=inputchecks))
    preds={}
    # Predictors only until all six membership outputs are frozen.
    for task,(p,ep,expected) in packets.items():
        bmem=membership(p['base'],p['ownmask'],K);identity,core,cap=repair(p['base'],p['base'],p['ownmask'])
        assert np.allclose(identity,bmem,atol=1e-12,rtol=0)
        ensemble=p['scores'].mean(1)
        local=np.stack([repair(x,p['base'],p['ownmask'])[0] for x in ensemble])
        seedlocal=np.stack([[repair(x,p['base'],p['ownmask'])[0] for x in arm] for arm in p['scores']])
        unrestricted=np.stack([membership(x,p['ownmask'],K) for x in ensemble])
        seedunrestricted=np.stack([[membership(x,p['ownmask'],K) for x in arm] for arm in p['scores']])
        dest=out/'predictions'/f'{task}.npz'
        np.savez_compressed(dest,arm_names=np.array(ARMS),seeds=np.array(SEEDS),K=np.array(K),m=np.array(M),
            base_membership=bmem,core_membership=core,cap_membership=cap,local_membership=local,seed_local_membership=seedlocal,
            unrestricted_membership=unrestricted,seed_unrestricted_membership=seedunrestricted,
            **{k:p[k] for k in ('query_ids','query_block','target_ids','ownmask')})
        preds[task]=dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest),evaluation_sha256=expected)
    lock=out/'PREDICTION_LOCK.json'
    dump(lock,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(out/'PLAN.json'),
        tasks=preds,all_six_frozen_before_reference=True))
    assert len(preds)==6 and lock.exists()
    # This reference read occurs only after the additional local prediction barrier.
    tasks={};sources={};files={};(out/'per_source').mkdir()
    for source in SOURCES:
        halves=[]
        for h in (0,1):
            task=f'{source}_held{h}';info=preds[task];p=load(ROOT/info['path']);assert sha(ROOT/info['path'])==info['sha256']
            ep=packets[task][1];assert sha(ep)==info['evaluation_sha256'];e=load(ep)
            for key in ('query_ids','target_ids','ownmask','query_block'):assert np.array_equal(p[key],e[key])
            ref=membership(e['reference'],e['ownmask'],K)
            a={name:(p[name+'_membership']*ref).sum(-1).mean(2 if name.startswith('seed_') else 1)/K for name in ('local','seed_local','unrestricted','seed_unrestricted')}
            a['base']=(p['base_membership']*ref).sum(-1).mean(0)/K
            a.update(query_ids=p['query_ids'],query_block=p['query_block']);tasks[task]=summarize(a);halves.append(a)
        for key in ('query_ids','query_block'):assert np.array_equal(halves[0][key],halves[1][key])
        a={k:np.stack([v[k] for v in halves]).mean(0) for k in ('local','seed_local','unrestricted','seed_unrestricted','base')}
        a.update(query_ids=halves[0]['query_ids'],query_block=halves[0]['query_block']);sources[source]=summarize(a)
        dest=out/'per_source'/f'{source}.npz';np.savez_compressed(dest,**a);files[str(dest.relative_to(ROOT))]=sha(dest)
    aggregate={arm:dict(local_precision=float(np.mean([sources[s]['arms'][arm]['local_precision'] for s in SOURCES])),
        unrestricted_precision=float(np.mean([sources[s]['arms'][arm]['unrestricted_precision'] for s in SOURCES])),
        delta_vs_original=float(np.mean([sources[s]['arms'][arm]['local_vs_original']['delta'] for s in SOURCES])),
        delta_vs_unrestricted=float(np.mean([sources[s]['arms'][arm]['local_vs_unrestricted']['delta'] for s in SOURCES]))) for arm in ARMS}
    contrasts={name:float(np.mean([sources[s]['local_contrasts'][name]['delta'] for s in SOURCES])) for name in sources[SOURCES[0]]['local_contrasts']}
    assert primary_files=={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'results').rglob('*') if p.is_file()}
    result=dict(identity='Post hoc fixed local membership repair; all arms reported; no training/window scan/primary replacement',
        sources=sources,tasks=tasks,equal_source_aggregate=aggregate,equal_source_local_contrasts=contrasts,
        synthetic_checks=checks,prediction_lock_sha256=sha(lock),plan_sha256=sha(out/'PLAN.json'),
        interval='2000 paired query-block bootstrap; conditional on fixed data/models/shared assay; no multiplicity correction or independent-study aggregate interval',
        primary_results_unchanged=True,declared_primary_C_still_failed=True,files=files)
    dump(out/'SUMMARY.json',result);write_report(out,result)
    print(json.dumps(dict(equal_source_aggregate=aggregate,equal_source_local_contrasts=contrasts),indent=2))

def write_report(out,r):
    def fmt(x):return f"{x['delta']:+.6f} [{x['conditional95'][0]:+.6f}, {x['conditional95'][1]:+.6f}]"
    lines=['# 后验诊断：固定局部名单调整','',
        '**本项是看到前轮回溯后固定的一次诊断，不是原主方法，也不是独立确认。原主方法C的失败结果保持不变。**','',
        'K=10、m=2；保留原名单前8名的分数化归属，只在原排名9–12位对应的分数化容量内选2位。三臂、两种子及先平均分数的集成统一适用；精确并列按容量比例分配。没有调窗口或择优来源。','',
        '全部六个局部归属预测先冻结，之后才读取本项评价参考。输出为两条方向性的局部名单，不宣称等价于一个完整标量排序。','',
        '| 来源 | 方法 | 局部P@10 | Δ原PertRank [条件95%] | Δ同臂原输出 [条件95%] |','|---|---|---:|---|---|']
    for s in SOURCES:
        for a,z in r['sources'][s]['arms'].items():lines.append(f"| {s} | {a} | {z['local_precision']:.6f} | {fmt(z['local_vs_original'])} | {fmt(z['local_vs_unrestricted'])} |")
    lines+=['','| 方法 | 等来源局部P@10 | Δ原PertRank | Δ同臂原输出 |','|---|---:|---:|---:|']
    for a,z in r['equal_source_aggregate'].items():lines.append(f"| {a} | {z['local_precision']:.6f} | {z['delta_vs_original']:+.6f} | {z['delta_vs_unrestricted']:+.6f} |")
    lines+=['','## 局部调整后的B/C对A','', '| 来源 | 对比 | ΔP@10 [条件95%] |','|---|---|---|']
    for s in SOURCES:
        for a,z in r['sources'][s]['local_contrasts'].items():lines.append(f'| {s} | {a} | {fmt(z)} |')
    lines+=['','全部种子、半池和来源的区间保存在SUMMARY.json；预测归属和逐查询精度随包保存。正向诊断只能作为后续方法开发线索，不替代新的确认。','']
    (out/'REPORT_zh.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
