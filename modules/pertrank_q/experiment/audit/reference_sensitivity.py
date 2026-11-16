#!/usr/bin/env python3
"""Exact empirical external-reference resampling of every frozen method.

This is a post hoc sensitivity analysis, not a confidence interval, posterior,
new independent experiment, or model-selection step. No predictions are fitted
or changed. All nine methods and all 100 reference count cases are retained.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib,json,math,os,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
DEFAULT_EXPERIMENT=ROOT.parent

def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def count_choices():
    choices=[]
    for a in range(4):
        for b in range(4-a):
            c=3-a-b
            choices.append(([a,b,c],math.factorial(3)/(math.factorial(a)*math.factorial(b)*math.factorial(c))/27))
    return choices
def weighted_quantile(values,weights,prob):
    # Left-inverse discrete empirical CDF; no interpolation and no CI claim.
    order=np.argsort(values,kind='stable');v=values[order];w=weights[order]
    return float(v[min(int(np.searchsorted(np.cumsum(w),prob,side='left')),len(v)-1)])
def describe(delta,weights,nominal):
    return dict(nominal_delta=float(delta[nominal]),weighted_mean_delta=float(np.dot(weights,delta)),
        weighted_empirical_2p5_97p5=[weighted_quantile(delta,weights,p) for p in (.025,.975)],
        weighted_mass_positive=float(weights[delta>1e-12].sum()),
        weighted_mass_zero=float(weights[abs(delta)<=1e-12].sum()),
        weighted_mass_negative=float(weights[delta < -1e-12].sum()),
        minimum=float(delta.min()),maximum=float(delta.max()))

def main(base,experiment,out,rebuild=False):
    base=Path(base).resolve();experiment=Path(experiment).resolve();out=Path(out).resolve()
    phase2=experiment/'epoch_pair_update'
    if out.exists() and any(out.iterdir()):raise RuntimeError('Preserve existing sensitivity results; use a fresh output directory')
    if rebuild:
        env=os.environ.copy()
        for runtime in (experiment/'.runtime',experiment.parent/'pertrank_neural_rank_20261115'/'.runtime'):
            if runtime.is_dir():env['PYTHONPATH']=str(runtime)+os.pathsep+env.get('PYTHONPATH','');break
        for script,extra in [(base/'rebuild_boundary.py',[]),(experiment/'rebuild_optimization.py',['--base-root',str(base)]),
                             (phase2/'rebuild_pair_update.py',['--base-root',str(base)])]:
            subprocess.run([sys.executable,str(script),*extra],env=env,check=True,stdout=subprocess.DEVNULL)
    sys.path.insert(0,str(phase2));sc=importlib.import_module('score_pair_update')
    assert Path(sc.__file__).resolve()==phase2/'score_pair_update.py'
    plan,packets,prediction_audit=sc.barrier(base)
    choices=count_choices();cases=[(a[0],b[0]) for a in choices for b in choices]
    weights=np.array([a[1]*b[1] for a in choices for b in choices])
    assert len(cases)==100 and abs(weights.sum()-1)<1e-12
    nominal=cases.index(([1,1,1],[1,1,1]))
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        identity='Post hoc empirical reference-resampling sensitivity; not CI, posterior, new experiment or selection',
        methods=list(sc.METHODS),source_order=list(sc.SOURCES),k=10,
        algorithm='Each of two original reference batches has three replicates. Enumerate all ten n=3 multinomial count vectors per batch; independent between batches, identical counts shared across every source and half within each batch. Ref=sum(replicate_effect*integer_count)/3.',
        expected_batches=2,replicates_per_batch=3,reference_cases=100,
        cases=cases,case_probabilities=weights.tolist(),nominal_case_index=nominal,
        aggregation='Frozen physical score means over seeds, exact fractional Top10 in each held pool. Average up/down precision, queries within source-half, two halves, then equal three sources.',
        quantiles='Weighted inverse empirical CDF at 0.025 and 0.975 with no interpolation; this is a reference sensitivity range, not a confidence interval.',
        positive_threshold=1e-12,comparators=['original_PertRank','old_B'],
        changes_forbidden='No weight update, no model/source/seed selection, no retuning and no change to either stage main outcome or gate.',
        prediction_lock_sha256={'phase1':sha(experiment/'predictions/ALL_PREDICTIONS_LOCK.json'),'phase2':sha(phase2/'predictions/ALL_PREDICTIONS_LOCK.json')},
        code_sha256=sha(__file__),scoring_code_sha256=sha(phase2/'score_pair_update.py'),
        input_sha256={s:sha(base/'inputs'/f'{s}_neural_inputs.npz') for s in sc.SOURCES},
        complete_prediction_barrier_checked=True)
    dump(out/'REFERENCE_SENSITIVITY_PROTOCOL.json',protocol)
    # First reference loading in this analysis occurs after the above freeze.
    inputs={s:load(base/'inputs'/f'{s}_neural_inputs.npz') for s in sc.SOURCES}
    batch_ids=sorted(set(v for s in sc.SOURCES for v in inputs[s]['reference_target_batch'].tolist()))
    assert len(batch_ids)==2,'Unexpected batch count; stop instead of adapting cases'
    donor_map={};reference_identity={};members={};held_refs={};feature_checks={}
    for si,source in enumerate(sc.SOURCES):
        full=inputs[source];assert full['reference_replicates'].shape[-1]==3
        for batch in batch_ids:
            rows=np.flatnonzero(full['reference_target_batch']==batch)
            if not len(rows):continue
            labels=full['reference_donor_labels'][rows]
            assert np.all(labels==labels[0]),('Within-batch donor ordering differs',source,batch)
            bkey=str(batch);current=labels[0].tolist()
            if bkey in donor_map:assert donor_map[bkey]==current,('Across-source donor ordering differs',source,batch)
            else:donor_map[bkey]=current
        for qi,qid in enumerate(full['query_ids']):
            for ti,tid in enumerate(full['target_ids']):
                key=(str(qid),str(tid))
                values=full['reference_replicates'][qi,ti]
                if key in reference_identity:assert np.array_equal(reference_identity[key],values),('Shared query-target reference differs',key)
                else:reference_identity[key]=values.copy()
        for half in (0,1):
            task=f'{source}_held{half}';p,ep,eh=packets[task]
            assert sha(ep)==eh;e=load(ep)
            ix=np.flatnonzero(full['targethalf']==half)
            assert np.array_equal(full['target_ids'][ix],p['target_ids'])
            assert np.array_equal(full['query_ids'],p['query_ids'])
            reps=full['reference_replicates'][:,ix]
            assert np.array_equal(reps,e['reference_replicates'])
            assert np.array_equal(reps.mean(2),e['reference'])
            batch=full['reference_target_batch'][ix]
            scores=np.concatenate([np.stack([p['base'],p['raw'],p['euclidean']]),p['scores'].mean(1)])
            members[task]=np.stack([sc.membership(x,p['ownmask'],10) for x in scores])
            held_refs[task]=(reps,batch,p['ownmask'])
            feature_checks[task]=dict(queries=len(p['query_ids']),candidates=len(p['target_ids']),reference_replicates_match_frozen_evaluator=True)
    assert set(donor_map)==set(map(str,batch_ids))
    points=np.empty((100,len(sc.SOURCES),2,len(sc.METHODS)))
    for ci,counts in enumerate(cases):
        for si,source in enumerate(sc.SOURCES):
            for half in (0,1):
                task=f'{source}_held{half}';reps,batch,own=held_refs[task]
                ref=np.empty(reps.shape[:2])
                for bi,b in enumerate(batch_ids):
                    ix=np.flatnonzero(batch==b)
                    ref[:,ix]=(reps[:,ix]*np.asarray(counts[bi],dtype=np.int64)).sum(2)/3
                rm=sc.membership(ref,own,10)
                # Same arithmetic order as main ensemble scoring.
                pq=(members[task]*rm).sum(-1)/10
                points[ci,si,half]=pq.mean(1).mean(1)
    per_source=points.mean(2);aggregate=per_source.mean(1)
    original=read(phase2/'results/SUMMARY.json');max_nominal_error=0.
    for si,source in enumerate(sc.SOURCES):
        for mi,method in enumerate(sc.METHODS):
            expected=original['sources'][source]['means'][method]
            error=abs(per_source[nominal,si,mi]-expected);max_nominal_error=max(error,max_nominal_error)
            assert error<2e-15,('Nominal reference did not reproduce main result',source,method,error)
    for mi,method in enumerate(sc.METHODS):
        error=abs(aggregate[nominal,mi]-original['equal_source_aggregate'][method]['precision'])
        max_nominal_error=max(max_nominal_error,error);assert error<2e-15
    comparisons={};source_comparisons={}
    for comparator in protocol['comparators']:
        refidx=sc.METHODS.index(comparator)
        comparisons[comparator]={method:describe(aggregate[:,mi]-aggregate[:,refidx],weights,nominal) for mi,method in enumerate(sc.METHODS)}
        source_comparisons[comparator]={source:{method:describe(per_source[:,si,mi]-per_source[:,si,refidx],weights,nominal) for mi,method in enumerate(sc.METHODS)} for si,source in enumerate(sc.SOURCES)}
    array_path=out/'REFERENCE_CASES.npz'
    np.savez_compressed(array_path,case_counts=np.array(cases,dtype=np.int64),case_probability=weights,
        method_names=np.array(sc.METHODS),source_names=np.array(sc.SOURCES),batch_ids=np.array(batch_ids),
        precision_by_case_source_half_method=points,precision_by_case_source_method=per_source,
        precision_by_case_method=aggregate,nominal_case_index=np.array(nominal))
    result=dict(identity=protocol['identity'],cases=100,probability_sum=float(weights.sum()),
        batch_ids=[str(b) for b in batch_ids],replicate_labels_by_batch=donor_map,
        same_batch_counts_shared_across_all_sources_and_halves=True,
        batches_independently_reweighted=True,shared_query_target_reference_entries_checked=len(reference_identity),
        nominal_reference_maximum_precision_difference=max_nominal_error,
        methods=list(sc.METHODS),aggregate_comparisons=comparisons,source_comparisons=source_comparisons,
        feature_checks=feature_checks,protocol_sha256=sha(out/'REFERENCE_SENSITIVITY_PROTOCOL.json'),
        arrays_sha256=sha(array_path),model_or_prediction_changed=False,
        original_phase2_primary_gates_preserved=original['primary_gates'])
    dump(out/'REFERENCE_SENSITIVITY_RESULTS.json',result)
    lines=['# 外部 RNA 参考重采样敏感性','',
        '这是全部固定模型名单对现有参考重复重加权的敏感性诊断。下列范围不是置信区间或后验概率，也不是新增独立验证。两批参考分别重加权；同批计数在所有来源和半池完全共享。','',
        '| 方法 | 名义 Δ旧B | 重采样加权均值 Δ旧B | 加权经验2.5%—97.5%范围 | 正差异的权重质量 |','|---|---:|---:|---|---:|']
    for method in sc.METHODS:
        v=comparisons['old_B'][method];lo,hi=v['weighted_empirical_2p5_97p5']
        lines.append(f"| {method} | {v['nominal_delta']:+.6f} | {v['weighted_mean_delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {v['weighted_mass_positive']:.4f} |")
    lines+=['','## 预设次要臂逐来源','', '| 来源 | 名义 Δ旧B | 加权均值 Δ旧B | 加权经验范围 | 正差异权重质量 |','|---|---:|---:|---|---:|']
    for source in sc.SOURCES:
        v=source_comparisons['old_B'][source]['B_rank_scale_free_pair'];lo,hi=v['weighted_empirical_2p5_97p5']
        lines.append(f"| {source} | {v['nominal_delta']:+.6f} | {v['weighted_mean_delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {v['weighted_mass_positive']:.4f} |")
    lines+=['','全部方法相对原PertRank的对应诊断也在JSON中。此诊断不参与选择、不更新权重、不更改任何阶段主臂及通过状态。三重复经验分布不能代表所有未测实验条件；跨批 donor 身份未假设一致。','']
    (out/'REPORT_zh.md').write_text('\n'.join(lines))
    print(json.dumps(dict(aggregate_comparisons=comparisons,nominal_reference_maximum_precision_difference=max_nominal_error),indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root',type=Path,default=DEFAULT_EXPERIMENT)
    p.add_argument('--base-root',type=Path)
    p.add_argument('--output-root',type=Path,default=ROOT/'reference_sensitivity')
    p.add_argument('--rebuild',action='store_true',help='Reconstruct missing derived views/predictions from archived weights; never train')
    a=p.parse_args();base=a.base_root or (a.experiment_root/'baseline_B' if (a.experiment_root/'baseline_B').is_dir() else a.experiment_root.parent/'pertrank_boundary_learning_20261115')
    main(base,a.experiment_root,a.output_root,a.rebuild)
