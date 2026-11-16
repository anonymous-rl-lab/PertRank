#!/usr/bin/env python3
"""Score the locked 2x2 B update, preserving all arms and seed diagnostics.

All six prediction packets and fit hashes are checked before reading an
evaluation file. Previously accessed references retain their exploratory
identity. No fitting, model selection, source selection, or retuning occurs.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCES = ('D3', 'D1', 'D2')
ARMS = ('old_B', 'B_joint_selection', 'B_rank_scale', 'B_rank_scale_joint')
SEEDS = (2026111511, 2026111512)
METHODS = ('original_PertRank', 'Unweighted', 'Euclidean') + ARMS
PRIMARY = 'B_rank_scale_joint'
COMPARATORS = ('original_PertRank', 'old_B')
K = 10
TURN_KEYS = ('entered', 'removed', 'entered_supported', 'removed_supported',
             'entered_unsupported', 'removed_unsupported')
EFFECT_NAMES = ('selection_frozen_scale', 'selection_rank_scale',
                'rank_scale_individual_selection', 'rank_scale_joint_selection',
                'interaction')

def load(path):
    with np.load(path, allow_pickle=False) as z: return {k:z[k] for k in z.files}
def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def safe(root,value):
    p=Path(value)
    if p.is_absolute() or '..' in p.parts: raise ValueError(f'Unsafe relative path: {value}')
    return root/p
def default_base():
    return ROOT/'baseline_B' if (ROOT/'baseline_B').is_dir() else ROOT.parent/'pertrank_boundary_learning_20261115'

def membership(scores,own,k):
    """Independent exact fractional boundary ties, in both directions."""
    scores=np.asarray(scores,float);own=np.asarray(own,bool)
    assert scores.shape==own.shape and np.isfinite(scores).all()
    assert np.all((~own).sum(1)>=k)
    out=np.zeros((2,)+scores.shape)
    for d,sgn in enumerate((1.,-1.)):
        s=np.where(own,-np.inf,sgn*scores)
        boundary=np.sort(s,axis=1)[:,-k,None]
        strict=(s>boundary)&~own;tied=(s==boundary)&~own
        out[d]=strict+tied*((k-strict.sum(1))/tied.sum(1))[:,None]
    assert np.max(abs(out.sum(-1)-k))<1e-10
    return out

def pairwise_error(scores,reference,own):
    i,j=np.triu_indices(reference.shape[1],1)
    truth=reference[:,i]-reference[:,j];delta=scores[:,i]-scores[:,j]
    valid=(~own[:,i])&(~own[:,j])&(truth!=0);n=valid.sum(1)
    loss=((np.sign(delta)!=np.sign(truth))&(delta!=0)).astype(float)+.5*(delta==0)
    ans=np.full(len(scores),.5);np.divide((loss*valid).sum(1),n,out=ans,where=n>0)
    return ans,n

def interval(diff,blocks):
    labels=np.unique(blocks)
    mass=np.array([diff[blocks==b].sum() for b in labels]);n=np.array([(blocks==b).sum() for b in labels])
    ix=np.random.default_rng(2026111518).integers(len(labels),size=(2000,len(labels)))
    return np.quantile(mass[ix].sum(1)/n[ix].sum(1),[.025,.975]).tolist()
def paired(diff,blocks):
    return dict(delta=float(np.mean(diff)),conditional95=interval(diff,blocks),
                query_win=int((diff>1e-12).sum()),query_tie=int((abs(diff)<=1e-12).sum()),
                query_loss=int((diff < -1e-12).sum()))

def barrier(base):
    """No evaluation file is accessed or hashed inside this function."""
    plan=read(ROOT/'PLAN_LOCK.json');ph=sha(ROOT/'PLAN_LOCK.json')
    assert plan['arms']==list(ARMS) and plan['seeds']==list(SEEDS)
    assert set(plan['tasks'])=={f'{s}_held{h}' for s in SOURCES for h in (0,1)}
    for n,h in plan['code_hashes'].items(): assert sha(safe(ROOT,n))==h,('new code',n)
    for n,h in plan.get('base_code_hashes',{}).items(): assert sha(safe(base,n))==h,('base code',n)
    assert sha(base/'PLAN_LOCK.json')==plan['base_plan_sha256']
    baseplan=read(base/'PLAN_LOCK.json')
    for s,h in plan['base_prediction_lock_sha256'].items():
        assert sha(base/'predictions'/f'{s}_PREDICTION_LOCK.json')==h
    all_lock=read(ROOT/'predictions/ALL_PREDICTIONS_LOCK.json')
    assert all_lock['plan_sha256']==ph
    packets={};audit={};global_halves={}
    for source in SOURCES:
        lp=ROOT/'predictions'/f'{source}_PREDICTION_LOCK.json'
        assert sha(lp)==all_lock['source_lock_sha256'][source]
        sl=read(lp);oldlock=read(base/'predictions'/f'{source}_PREDICTION_LOCK.json')
        assert sl['plan_sha256']==ph
        for s,h in sl['selection_sha256'].items(): assert sha(ROOT/'selections'/f'{s}.json')==h
        audit[source]={'lock_sha256':sha(lp),'tasks':{}}
        for half in (0,1):
            task=f'{source}_held{half}';entry=sl['tasks'][task];meta=plan['tasks'][task]
            assert entry==all_lock['tasks'][task]
            assert entry['path']==f'predictions/{task}.npz'
            pp=safe(ROOT,entry['path']);assert sha(pp)==entry['sha256']
            p=load(pp);q,t=p['base'].shape
            assert p['arm_names'].tolist()==list(ARMS) and p['seeds'].tolist()==list(SEEDS)
            assert p['scores'].shape==(4,2,q,t) and np.isfinite(p['scores']).all()
            assert np.array_equal(p['ownmask'],p['query_ids'][:,None]==p['target_ids'][None,:])
            assert np.all((~p['ownmask']).sum(1)>=K)
            vp=base/'views'/source/f'held{half}'
            assert sha(vp/'held_features.npz')==meta['held_features_sha256']
            assert sha(vp/'calibration.npz')==meta['calibration_sha256']
            v=load(vp/'held_features.npz')
            assert not any(k.startswith('reference') for k in v)
            for name in ('base','raw','euclidean','ownmask','query_ids','query_block','target_ids'):
                assert np.array_equal(p[name],v[name]),(task,name)
            with np.load(vp/'calibration.npz',allow_pickle=False) as z:cal_ids=z['target_ids']
            assert not set(cal_ids)&set(p['target_ids'])
            assert t==meta['held'] and len(cal_ids)==meta['calibration']
            oldp=base/oldlock['tasks'][task]['path']
            assert sha(oldp)==oldlock['tasks'][task]['sha256']
            old=load(oldp);bi=old['arm_names'].tolist().index('boundary_context')
            assert np.array_equal(p['scores'][0],old['scores'][bi]),('old B changed',task)
            assert set(entry['fits'])=={f'{a}_seed{seed}' for a in ARMS for seed in SEEDS}
            for key,f in entry['fits'].items():
                fd=safe(base if f['fit_root']=='base' else ROOT,f['fit_path'])
                assert f['fit_root'] in ('base','new')
                assert sha(fd/'weights.pt')==f['weights_sha256'],(task,key,'weight')
                assert sha(fd/'normalization.npz')==f['normalization_sha256'],(task,key,'norm')
                fm=read(fd/'FIT.json')
                assert fm['training_target_ids']==cal_ids.tolist(),(task,key,'fit targets')
                assert fm.get('validation_target_ids') is None,(task,key,'outer validation')
                assert fm.get('selected_epoch',fm.get('epochs'))==f['selected_epoch']
                assert f['selected_epoch'] in (0,50,100,200,300)
            for tid in p['target_ids']:
                tid=str(tid)
                if tid in global_halves:assert global_halves[tid]==half
                global_halves[tid]=half
            assert meta['evaluation_sha256']==baseplan['tasks'][task]['evaluation_sha256']
            packets[task]=(p,base/'evaluation'/source/f'held{half}.npz',meta['evaluation_sha256'])
            audit[source]['tasks'][task]={'prediction_sha256':entry['sha256'],'queries':q,'candidates':t,
                'fit_hashes_checked':True,'old_B_bytewise_preserved':True,
                'selected_epochs':{k:f['selected_epoch'] for k,f in entry['fits'].items()}}
        a,b=[packets[f'{source}_held{h}'][0] for h in (0,1)]
        for k in ('query_ids','query_block'):assert np.array_equal(a[k],b[k])
        assert not set(a['target_ids'])&set(b['target_ids'])
    assert len(global_halves)==75
    return plan,packets,audit

def turnover(a,b,ref):
    entered=np.maximum(a-b,0);removed=np.maximum(b-a,0)
    assert np.max(abs(entered.sum(-1)-removed.sum(-1)))<1e-10
    return dict(entered=entered,removed=removed,entered_supported=entered*ref,
        removed_supported=removed*ref,entered_unsupported=entered*(1-ref),removed_unsupported=removed*(1-ref))
def mass_summary(turn,delta):
    r={k+'_mass':float(turn[k].sum()) for k in TURN_KEYS}
    net=r['entered_supported_mass']-r['removed_supported_mass']
    r.update(net_supported_mass=net,delta_precision=float(delta.mean()),net_over_2QK=net/(2*len(delta)*K))
    assert abs(r['net_over_2QK']-r['delta_precision'])<1e-12
    return r

def task_arrays(p,e):
    for k in ('ownmask','query_ids','query_block','target_ids'):assert np.array_equal(p[k],e[k]),k
    assert np.array_equal(e['reference_replicates'].mean(-1),e['reference'])
    score=np.concatenate([np.stack([p['base'],p['raw'],p['euclidean']]),p['scores'].mean(1)])
    member=np.stack([membership(x,p['ownmask'],K) for x in score]);ref=membership(e['reference'],p['ownmask'],K)
    directional=(member*ref).sum(-1)/K;pq=directional.mean(1)
    sm=np.stack([[membership(x,p['ownmask'],K) for x in arm] for arm in p['scores']])
    seedpq=(sm*ref).sum(-1).mean(2)/K
    err,n=zip(*(pairwise_error(x,e['reference'],p['ownmask']) for x in score))
    for nn in n[1:]:assert np.array_equal(nn,n[0])
    seederr=np.stack([[pairwise_error(x,e['reference'],p['ownmask'])[0] for x in arm] for arm in p['scores']])
    ix={m:METHODS.index(m) for m in METHODS};b,j,s,c=[pq[ix[a]] for a in ARMS]
    effects=np.stack([j-b,c-s,s-b,c-j,c-s-j+b])
    turns=[turnover(member,member[ix[m]],ref) for m in COMPARATORS]
    a=dict(method_names=np.array(METHODS),arm_names=np.array(ARMS),seeds=np.array(SEEDS),K=np.array(K),
        comparator_names=np.array(COMPARATORS),effect_names=np.array(EFFECT_NAMES),scores=score,membership=member,
        reference_membership=ref,precision_per_query=pq,precision_by_direction=directional,
        seed_precision_per_query=seedpq,seed_membership=sm,pairwise_error_per_query=np.stack(err),
        seed_pairwise_error_per_query=seederr,pairwise_valid_pair_count=n[0],effects_per_query=effects,
        query_ids=p['query_ids'],query_block=p['query_block'],target_ids=p['target_ids'],ownmask=p['ownmask'])
    for key in TURN_KEYS:a[key+'_membership']=np.stack([v[key] for v in turns])
    result=dict(queries=len(p['query_ids']),candidates=len(p['target_ids']),means={m:float(pq[i].mean()) for i,m in enumerate(METHODS)},
        seed_means={arm:{str(seed):float(seedpq[i,z].mean()) for z,seed in enumerate(SEEDS)} for i,arm in enumerate(ARMS)},
        effects={k:float(effects[i].mean()) for i,k in enumerate(EFFECT_NAMES)},backtrace={})
    for ci,comp in enumerate(COMPARATORS):
        result['backtrace'][comp]={m:mass_summary({k:turns[ci][k][i] for k in TURN_KEYS},pq[i]-pq[ix[comp]]) for i,m in enumerate(METHODS)}
    return a,result

def source_arrays(halves):
    a,b=halves
    for k in ('query_ids','query_block'):assert np.array_equal(a[k],b[k])
    out={k:a[k] for k in ('method_names','arm_names','seeds','K','comparator_names','effect_names','query_ids','query_block')}
    for k in ('precision_per_query','precision_by_direction','seed_precision_per_query','pairwise_error_per_query','seed_pairwise_error_per_query','effects_per_query'):
        out[k+'_by_half']=np.stack([x[k] for x in halves]);out[k]=out[k+'_by_half'].mean(0)
    for k in TURN_KEYS:
        key=k+'_mass_by_direction_query'
        out[key+'_by_half']=np.stack([x[k+'_membership'].sum(-1) for x in halves]);out[key]=out[key+'_by_half'].mean(0)
    pq=out['precision_per_query'];spq=out['seed_precision_per_query'];err=out['pairwise_error_per_query'];blocks=out['query_block']
    r=dict(queries=len(blocks),query_blocks=len(np.unique(blocks)),means={},contrasts={},seed_means={},seed_contrasts={},effects={},backtrace={},pairwise_error={})
    for i,m in enumerate(METHODS):r['means'][m]=float(pq[i].mean())
    for ci,comp in enumerate(COMPARATORS):
        index=METHODS.index(comp)
        r['contrasts'][comp]={m:paired(pq[i]-pq[index],blocks) for i,m in enumerate(METHODS)}
        r['backtrace'][comp]={m:mass_summary({k:out[k+'_mass_by_direction_query'][ci,i] for k in TURN_KEYS},pq[i]-pq[index]) for i,m in enumerate(METHODS)}
        r['pairwise_error'][comp]={m:dict(error=float(err[i].mean()),**paired(err[i]-err[index],blocks)) for i,m in enumerate(METHODS)}
        r['seed_contrasts'][comp]={arm:{str(seed):paired(spq[i,z]-(pq[index] if comp=='original_PertRank' else spq[0,z]),blocks) for z,seed in enumerate(SEEDS)} for i,arm in enumerate(ARMS)}
    for i,arm in enumerate(ARMS):r['seed_means'][arm]={str(seed):float(spq[i,z].mean()) for z,seed in enumerate(SEEDS)}
    for i,name in enumerate(EFFECT_NAMES):r['effects'][name]=paired(out['effects_per_query'][i],blocks)
    return out,r

def report_text(summary):
    r=summary;s=r['sources'];agg=r['equal_source_aggregate']
    lines=['# B 分支两因素优化：完整结果','',
        '本轮为历史外部参考上的探索性开发，所有臂和来源均保留。主臂预先指定为 B_rank_scale_joint；没有依据外层成绩重新选择主臂。','',
        '四臂：old_B=旧B；B_joint_selection=保持旧训练目标，按两个种子物理分数集成共同选择轮次；B_rank_scale=只将训练SoftTopK中的分数除以当前预测的每查询候选标准差；B_rank_scale_joint=同时作两项改动。原几何锚点、参考支持标签与真实P@10评价均保持。','',
        '| 方法 | D3 Δ原方法 | D1 Δ原方法 | D2 Δ原方法 | 等来源 P@10 | 等来源 Δ原方法 | 等来源 Δ旧B |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        lines.append('| '+m+' | '+' | '.join(f"{s[x]['contrasts']['original_PertRank'][m]['delta']:+.6f}" for x in SOURCES)+f" | {agg[m]['precision']:.6f} | {agg[m]['delta_original']:+.6f} | {agg[m]['delta_old_B']:+.6f} |")
    lines+=['','## 各来源相对旧B','', '| 方法 | D3 Δ旧B | D1 Δ旧B | D2 Δ旧B |','|---|---:|---:|---:|']
    for m in ARMS:
        lines.append('| '+m+' | '+' | '.join(f"{s[x]['contrasts']['old_B'][m]['delta']:+.6f}" for x in SOURCES)+' |')
    lines+=['','## 主臂条件区间','', '| 来源 | 主臂−原方法 | 主臂−旧B |','|---|---|---|']
    for source in SOURCES:
        vals=[]
        for comp in COMPARATORS:
            v=s[source]['contrasts'][comp][PRIMARY];lo,hi=v['conditional95'];vals.append(f"{v['delta']:+.6f} [{lo:+.6f}, {hi:+.6f}]")
        lines.append('| '+source+' | '+' | '.join(vals)+' |')
    lines+=['',f"开发推进门：{r['primary_gates']['development_passed']}；稳定升级门：{r['primary_gates']['robustness_passed']}。",'',
        '区间为固定候选、输入、拟合模型和外部参考下的2000次查询块重抽样；没有跨独立外部研究的总体区间，没有多重搜索校正。三个输入来源共享同一个外部RNA实验。','',
        '## 两因素归因（等来源精度差）','', '| 对比 | ΔP@10 |','|---|---:|']
    for name,value in r['equal_source_effects'].items():lines.append(f'| {name} | {value:+.6f} |')
    lines+=['','这些差异比较本轮受控算法改动，不把相关性诊断当作独立生物学因果证明。逐来源区间与全部种子结果在 SUMMARY.json 中。','',
        '## 完整名单换人','', '| 来源 | 比较对象 | 进入 | 移出 | 进入支持 | 移出支持 | 净支持 |','|---|---|---:|---:|---:|---:|---:|']
    for source in SOURCES:
        for comp in COMPARATORS:
            b=s[source]['backtrace'][comp][PRIMARY]
            lines.append(f"| {source} | {comp} | {b['entered_mass']:.3f} | {b['removed_mass']:.3f} | {b['entered_supported_mass']:.3f} | {b['removed_supported_mass']:.3f} | {b['net_supported_mass']:+.3f} |")
    lines+=['','## 主臂全部种子与半池','', '| 来源 | 种子1 Δ原方法 | 种子2 Δ原方法 | 半池0集成 Δ原方法 | 半池1集成 Δ原方法 |','|---|---:|---:|---:|---:|']
    for source in SOURCES:
        seedvals=[s[source]['seed_contrasts']['original_PertRank'][PRIMARY][str(seed)]['delta'] for seed in SEEDS]
        halfvals=[r['tasks'][f'{source}_held{h}']['means'][PRIMARY]-r['tasks'][f'{source}_held{h}']['means']['original_PertRank'] for h in (0,1)]
        lines.append('| '+source+' | '+' | '.join(f'{x:+.6f}' for x in seedvals+halfvals)+' |')
    lines+=['','支持指固定外部参考名单的分数化归属，不是独立湿实验命中。各来源换人质量先平均两个半池；净支持/(2×查询数×10)=ΔP@10 已逐任务和逐来源核验。','',
        '当前评价在各自的留出候选池内完成，不拼接跨折模型分数；绝对精度不能与论文完整原候选池的P@10直接比较。原参考已经在此前反复用于开发，本次锁定仅保证本轮预测在本轮评分之前完成。','']
    return '\n'.join(lines)

def main(base,out):
    base=Path(base).resolve();out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise RuntimeError('Preserve existing results; use a new empty output directory')
    plan,packets,audit=barrier(base)
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'SCORING_LOCK.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        score_code_sha256=sha(__file__),plan_sha256=sha(ROOT/'PLAN_LOCK.json'),all_six_predictions_verified_before_reference=True,audit=audit))
    tasks={};sources={};files={}
    for source in SOURCES:
        halves=[]
        for half in (0,1):
            task=f'{source}_held{half}';p,ep,eh=packets[task]
            assert sha(ep)==eh,('reference changed',task)
            arrays,res=task_arrays(p,load(ep));tasks[task]=res;halves.append(arrays)
            path=out/'per_task'/f'{task}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**arrays)
            files[str(path.relative_to(out))]=sha(path)
        arrays,res=source_arrays(halves);sources[source]=res
        path=out/'per_source'/f'{source}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**arrays)
        files[str(path.relative_to(out))]=sha(path)
    aggregate={m:dict(precision=float(np.mean([sources[s]['means'][m] for s in SOURCES])),
        delta_original=float(np.mean([sources[s]['contrasts']['original_PertRank'][m]['delta'] for s in SOURCES])),
        delta_old_B=float(np.mean([sources[s]['contrasts']['old_B'][m]['delta'] for s in SOURCES]))) for m in METHODS}
    seedmean={a:{str(seed):float(np.mean([sources[s]['seed_contrasts']['original_PertRank'][a][str(seed)]['delta'] for s in SOURCES])) for seed in SEEDS} for a in ARMS}
    gains=[sources[s]['contrasts']['original_PertRank'][PRIMARY]['delta'] for s in SOURCES]
    low=[s for s in SOURCES if sources[s]['contrasts']['original_PertRank'][PRIMARY]['conditional95'][0]>1e-12]
    develop=aggregate[PRIMARY]['delta_old_B']>0 and all(x>=0 for x in gains)
    robust=develop and len(low)>=2 and all(x>0 for x in seedmean[PRIMARY].values())
    summary=dict(identity='Exploratory 2x2 B optimization on previously accessed external references; no post-score retuning',
        K=K,methods=list(METHODS),arms=list(ARMS),primary_arm=PRIMARY,seeds=list(SEEDS),
        sources=sources,tasks=tasks,equal_source_aggregate=aggregate,equal_source_seed_deltas_original=seedmean,
        equal_source_effects={name:float(np.mean([sources[s]['effects'][name]['delta'] for s in SOURCES])) for name in EFFECT_NAMES},
        primary_gates=dict(development_passed=develop,robustness_passed=robust,
            equal_source_vs_old_B_positive=aggregate[PRIMARY]['delta_old_B']>0,every_source_vs_original_nonnegative=all(x>=0 for x in gains),
            sources_with_positive_conditional_lower_limit=low,both_seed_equal_source_gains_positive=all(x>0 for x in seedmean[PRIMARY].values())),
        gate_definition='Development: equal-source primary-oldB>0 and each primary-original>=0. Robustness additionally requires >=2 source conditional 95% lower limits>1e-12 and positive equal-source primary-original for both seeds.',
        aggregation='Average physical scores across seeds within each held pool, then up/down precision, half precision, query mean, equal source mean. Never join cross-fold scores.',
        intervals='2000 paired query-block percentile draws, seed2026111518; conditional on fixed inputs, candidates, models and shared reference. No independent-study aggregate CI or multiplicity correction.',
        attribution='2x2 contrasts use exact precision after fixed selection. Interaction=joint_scale-joint_only-scale_only+old_B. All contrasts retained, not post-score selection.',
        reference_identity='Existing external RNA mean of three replicate effects, unchanged. Training label remains empirical donor-bootstrap membership support.',
        scoring_lock_sha256=sha(out/'SCORING_LOCK.json'),files=files)
    dump(out/'SUMMARY.json',summary);(out/'REPORT_zh.md').write_text(report_text(summary))
    print(json.dumps(dict(equal_source_aggregate=aggregate,primary_gates=summary['primary_gates']),indent=2))
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root',type=Path,default=default_base())
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    args=parser.parse_args();main(args.base_root,args.output_dir)
