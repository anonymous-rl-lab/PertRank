#!/usr/bin/env python3
"""Score frozen boundary-learning predictions; never fit or select a model.

All source prediction locks are required and verified before any evaluation
file is opened or hashed. The primary method and contrasts are fixed here.
The reference is the existing assay, not a new blind validation.
"""
from pathlib import Path
import argparse
import datetime
import json
import numpy as np
from common import SOURCES, load, sha, dump, membership, precision, block_interval

ROOT = Path(__file__).resolve().parent
K = 10
ARMS = ('base_context', 'boundary_context', 'boundary_pair')
SEEDS = (2026111511, 2026111512)
METHODS = ('original_PertRank', 'raw', 'euclidean') + ARMS
PRIMARY = 'boundary_pair'
SECONDARY = ('boundary_context', 'base_context')
COMPARATORS = ('original_PertRank', 'base_context', 'boundary_context')
PI = METHODS.index(PRIMARY)
CI = [METHODS.index(m) for m in COMPARATORS]
TURNOVER_KEYS = ('entered', 'removed', 'entered_supported', 'removed_supported',
                 'entered_unsupported', 'removed_unsupported')


def equal(a, b, label):
    assert np.array_equal(a, b), label


def pairwise_error(scores, reference, own):
    """Unordered-pair error: exclude reference ties, score ties cost one half."""
    q, t = reference.shape
    i, j = np.triu_indices(t, k=1)
    truth = reference[:, i] - reference[:, j]
    valid = (~own[:, i]) & (~own[:, j]) & (truth != 0)
    count = valid.sum(1)
    delta = scores[:, i] - scores[:, j]
    wrong = ((np.sign(delta) != np.sign(truth)) & (delta != 0)).astype(float)
    wrong += .5 * (delta == 0)
    result = np.full(q, .5)
    np.divide((wrong * valid).sum(1), count, out=result, where=count > 0)
    return result, count


def _relative_path(value):
    p = Path(value)
    assert not p.is_absolute() and '..' not in p.parts, ('unsafe relative path', value)
    return ROOT / p


def checks_before_reference():
    """Verify the prediction barrier before touching the evaluation directory."""
    paths = {s: ROOT / 'predictions' / f'{s}_PREDICTION_LOCK.json' for s in SOURCES}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise RuntimeError('All source prediction locks required: ' + ', '.join(missing))
    pp = ROOT / 'PLAN_LOCK.json'
    plan = json.loads(pp.read_text())
    ph = sha(pp)
    assert plan['arms'] == list(ARMS)
    assert plan['seeds'] == list(SEEDS)
    assert plan['source_specific'] is True
    assert set(plan['tasks']) == {f'{s}_held{h}' for s in SOURCES for h in (0, 1)}
    for name, expected in plan['code_hashes'].items():
        assert sha(_relative_path(name)) == expected, ('changed locked code', name)
    packets, audit, global_half = {}, {}, {}
    # This pass is exclusively prediction, calibration-feature and fit checks.
    for s in SOURCES:
        lock = json.loads(paths[s].read_text())
        assert lock['plan_sha256'] == ph, ('plan hash', s)
        assert set(lock['tasks']) == {f'{s}_held0', f'{s}_held1'}
        if 'prediction_time_utc' in lock and 'created_utc' in plan:
            assert datetime.datetime.fromisoformat(lock['prediction_time_utc']) >= datetime.datetime.fromisoformat(plan['created_utc'])
        audit[s] = dict(prediction_lock_sha256=sha(paths[s]),
                        prediction_time_utc=lock.get('prediction_time_utc'), tasks={})
        for h in (0, 1):
            task = f'{s}_held{h}'
            entry, meta = lock['tasks'][task], plan['tasks'][task]
            assert entry['path'] == f'predictions/{task}.npz'
            path = _relative_path(entry['path'])
            assert sha(path) == entry['sha256'], ('prediction hash', task)
            p = load(path)
            equal(p['arm_names'], np.array(ARMS), (task, 'arms'))
            equal(p['seeds'], np.array(SEEDS), (task, 'seeds'))
            q, t = p['base'].shape
            assert p['scores'].shape == (len(ARMS), len(SEEDS), q, t)
            assert np.isfinite(p['scores']).all()
            for name in ('base', 'raw', 'euclidean'):
                assert p[name].shape == (q, t) and np.isfinite(p[name]).all()
            assert len(p['query_ids']) == q and len(p['target_ids']) == t
            assert len(set(p['query_ids'])) == q and len(set(p['target_ids'])) == t
            equal(p['ownmask'], p['query_ids'][:, None] == p['target_ids'][None, :], (task, 'own identity'))
            assert np.min((~p['ownmask']).sum(1)) >= K
            # The plan records the two disjoint views and the separate evaluator.
            vp = ROOT / 'views' / s / f'held{h}'
            for name, key in [('calibration.npz', 'calibration_sha256'), ('held_features.npz', 'held_features_sha256')]:
                assert sha(vp / name) == meta[key], (task, key)
            v = load(vp / 'held_features.npz')
            assert not any(k.startswith('reference') for k in v), ('reference in held feature view', task)
            for key in ('base', 'raw', 'euclidean', 'ownmask', 'query_ids', 'target_ids', 'query_block'):
                equal(p[key], v[key], (task, key))
            assert np.all(v['targethalf'] == h)
            with np.load(vp / 'calibration.npz', allow_pickle=False) as z:
                cal_ids = z['target_ids']
            assert not set(cal_ids) & set(p['target_ids'])
            assert t == meta['held'] and len(cal_ids) == meta['calibration']
            fits = entry['fits']
            assert set(fits) == {f'{a}_seed{seed}' for a in ARMS for seed in SEEDS}
            for arm in ARMS:
                for seed in SEEDS:
                    f = fits[f'{arm}_seed{seed}']
                    fd = _relative_path(f['outer_fit'])
                    assert f['selected_epoch'] in (0, 50, 100, 200, 300)
                    assert sha(fd / 'normalization.npz') == f['normalization_sha256']
                    assert sha(fd / 'weights.pt') == f['weights_sha256']
                    fm = json.loads((fd / 'FIT.json').read_text())
                    assert fm['arm'] == arm and fm['seed'] == seed
                    assert fm['selected_epoch'] == f['selected_epoch']
                    assert fm['training_target_ids'] == cal_ids.tolist()
                    assert fm.get('validation_target_ids') is None
                    if arm == 'base_context' and fm.get('historical_import'):
                        assert fm['original_arm'] == 'neural_no_extra_geometry'
                        assert fm['original_weights_sha256'] == f['weights_sha256']
                        assert fm['original_normalization_sha256'] == f['normalization_sha256']
            for tid in p['target_ids']:
                if str(tid) in global_half:
                    assert global_half[str(tid)] == h
                global_half[str(tid)] = h
            packets[task] = (p, ROOT / 'evaluation' / s / f'held{h}.npz', meta['evaluation_sha256'])
            audit[s]['tasks'][task] = dict(prediction_sha256=entry['sha256'],
                evaluation_sha256=meta['evaluation_sha256'], calibration_sha256=meta['calibration_sha256'],
                held_features_sha256=meta['held_features_sha256'], queries=q, held_candidates=t,
                fit_hashes_verified=True, selected_epochs={k: f['selected_epoch'] for k, f in fits.items()})
        a, b = (packets[f'{s}_held{h}'][0] for h in (0, 1))
        for key in ('query_ids', 'query_block'):
            equal(a[key], b[key], (s, 'half alignment', key))
        assert not set(a['target_ids']) & set(b['target_ids'])
    assert len(global_half) == 75
    return plan, packets, audit


def turnover(member, comparator_member, reference_member):
    """Candidate-resolved fractional entry/exit and reference-supported mass."""
    entered = np.maximum(member - comparator_member, 0)
    removed = np.maximum(comparator_member - member, 0)
    assert np.max(np.abs(entered.sum(-1) - removed.sum(-1))) < 1e-10
    return dict(entered=entered, removed=removed,
        entered_supported=entered * reference_member, removed_supported=removed * reference_member,
        entered_unsupported=entered * (1-reference_member), removed_unsupported=removed * (1-reference_member))


def mass_summary(arrays, delta):
    result = {key + '_mass': float(arrays[key].sum()) for key in TURNOVER_KEYS}
    net = result['entered_supported_mass'] - result['removed_supported_mass']
    q = len(delta)
    result.update(net_supported_mass=net, delta_precision=float(np.mean(delta)),
                  net_support_over_2QK=float(net / (2*q*K)), identity_verified=True)
    assert abs(result['net_support_over_2QK'] - result['delta_precision']) < 1e-12
    return result


def score_task(p, e):
    for key in ('ownmask', 'query_ids', 'target_ids', 'query_block'):
        equal(p[key], e[key], ('held reference alignment', key))
    equal(e['reference_replicates'].mean(2), e['reference'], 'frozen donor mean')
    assert np.isfinite(e['reference']).all()
    scores = np.concatenate([np.stack([p['base'], p['raw'], p['euclidean']]), p['scores'].mean(1)])
    member = np.stack([membership(x, p['ownmask'], K) for x in scores])
    refm = membership(e['reference'], e['ownmask'], K)
    direction = (member * refm[None]).sum(3) / K
    pq = direction.mean(1)
    seedmember = np.stack([[membership(x, p['ownmask'], K) for x in arm] for arm in p['scores']])
    seed_direction = (seedmember * refm[None, None]).sum(-1) / K
    seedpq = seed_direction.mean(2)
    errors, counts = zip(*(pairwise_error(x, e['reference'], p['ownmask']) for x in scores))
    for n in counts[1:]: equal(n, counts[0], 'eligible pair count')
    seederr = np.stack([[pairwise_error(x, e['reference'], p['ownmask'])[0] for x in arm] for arm in p['scores']])
    base_turn = turnover(member, member[0], refm)
    contrast_turn = turnover(member[PI][None], member[CI], refm)
    bi, ai = (METHODS.index(x) for x in SECONDARY)
    secondary_turn = turnover(member[bi], member[ai], refm)
    # Compare each neural seed to the same seed of its learned comparators.
    primary_seed = seedpq[ARMS.index(PRIMARY)]
    seedcontrast = np.stack([primary_seed - pq[0][None],
        primary_seed - seedpq[ARMS.index('base_context')],
        primary_seed - seedpq[ARMS.index('boundary_context')]])
    arrays = dict(method_names=np.array(METHODS), arm_names=np.array(ARMS), seeds=p['seeds'], K=np.array(K),
        comparator_names=np.array(COMPARATORS), scores=scores, membership=member, reference_membership=refm,
        precision_per_query=pq, precision_by_direction=direction, seed_precision_per_query=seedpq,
        seed_precision_by_direction=seed_direction, seed_membership=seedmember,
        pairwise_error_per_query=np.stack(errors), seed_pairwise_error_per_query=seederr,
        pairwise_valid_pair_count=counts[0], primary_delta_per_query=pq[PI][None] - pq[CI],
        primary_seed_delta_per_query=seedcontrast,
        secondary_context_delta_per_query=pq[bi]-pq[ai],
        secondary_context_seed_delta_per_query=seedpq[ARMS.index(SECONDARY[0])]-seedpq[ARMS.index(SECONDARY[1])],
        query_ids=p['query_ids'], query_block=p['query_block'], target_ids=p['target_ids'], ownmask=p['ownmask'])
    for key in TURNOVER_KEYS:
        arrays[key + '_membership'] = base_turn[key]
        arrays['primary_' + key + '_membership'] = contrast_turn[key]
        arrays['secondary_context_' + key + '_membership'] = secondary_turn[key]
    summary = dict(queries=len(p['query_ids']), candidates=len(p['target_ids']),
        eligible_candidate_min=int((~p['ownmask']).sum(1).min()),
        means={m: float(pq[i].mean()) for i, m in enumerate(METHODS)},
        seed_means={a: {str(seed): float(seedpq[i, j].mean()) for j, seed in enumerate(SEEDS)} for i, a in enumerate(ARMS)},
        pairwise_ordering_error={m: float(np.mean(errors[i])) for i, m in enumerate(METHODS)},
        seed_pairwise_ordering_error={a: {str(seed): float(seederr[i,j].mean()) for j, seed in enumerate(SEEDS)} for i,a in enumerate(ARMS)},
        zero_decisive_pair_queries=int((counts[0] == 0).sum()), backtrace={}, primary_backtrace={},
        secondary_context_delta=float((pq[bi]-pq[ai]).mean()),
        secondary_context_backtrace=mass_summary(secondary_turn,pq[bi]-pq[ai]))
    for i,m in enumerate(METHODS):
        summary['backtrace'][m] = mass_summary({key: value[i] for key,value in base_turn.items()}, pq[i]-pq[0])
    for i,m in enumerate(COMPARATORS):
        summary['primary_backtrace'][m] = mass_summary({key: value[i] for key,value in contrast_turn.items()}, pq[PI]-pq[CI[i]])
    return arrays, summary


def paired_summary(diff, blocks):
    return dict(delta=float(diff.mean()), conditional95=block_interval(diff, blocks))


def score_source(halves):
    a,b = halves
    for key in ('query_ids', 'query_block'): equal(a[key], b[key], ('source alignment',key))
    arrays = dict(method_names=np.array(METHODS), arm_names=np.array(ARMS), seeds=np.array(SEEDS), K=np.array(K),
                  comparator_names=np.array(COMPARATORS), query_ids=a['query_ids'], query_block=a['query_block'])
    for key in ('precision_per_query','seed_precision_per_query','pairwise_error_per_query','seed_pairwise_error_per_query',
                'primary_delta_per_query','primary_seed_delta_per_query',
                'secondary_context_delta_per_query','secondary_context_seed_delta_per_query'):
        arrays[key + '_by_half'] = np.stack([x[key] for x in halves])
        arrays[key] = arrays[key + '_by_half'].mean(0)
    for prefix in ('', 'primary_', 'secondary_context_'):
        for key in TURNOVER_KEYS:
            name = prefix + key + '_mass_by_direction_query'
            arrays[name + '_by_half'] = np.stack([x[prefix + key + '_membership'].sum(-1) for x in halves])
            arrays[name] = arrays[name + '_by_half'].mean(0)
    pq, spq, err = (arrays[k] for k in ('precision_per_query','seed_precision_per_query','pairwise_error_per_query'))
    blocks = arrays['query_block']
    summary = dict(queries=len(a['query_ids']), query_blocks=len(np.unique(blocks)), means={}, seed_means={},
                   seed_deltas_vs_original={}, primary_contrasts={}, primary_seed_contrasts={},
                   pairwise_ordering_error={}, seed_pairwise_ordering_error={}, backtrace={}, primary_backtrace={},
                   secondary_context_contrast=paired_summary(arrays['secondary_context_delta_per_query'],blocks),
                   secondary_context_seed_contrasts={str(seed):paired_summary(arrays['secondary_context_seed_delta_per_query'][j],blocks) for j,seed in enumerate(SEEDS)},
                   secondary_context_backtrace=mass_summary({key:arrays['secondary_context_'+key+'_mass_by_direction_query'] for key in TURNOVER_KEYS},arrays['secondary_context_delta_per_query']))
    for i,m in enumerate(METHODS):
        summary['means'][m] = dict(precision=float(pq[i].mean()), **paired_summary(pq[i]-pq[0],blocks))
        summary['pairwise_ordering_error'][m] = dict(error=float(err[i].mean()), **paired_summary(err[i]-err[0],blocks))
        summary['backtrace'][m] = mass_summary({key: arrays[key+'_mass_by_direction_query'][i] for key in TURNOVER_KEYS},pq[i]-pq[0])
    for i,m in enumerate(COMPARATORS):
        summary['primary_contrasts'][m] = paired_summary(arrays['primary_delta_per_query'][i],blocks)
        summary['primary_seed_contrasts'][m] = {str(seed): paired_summary(arrays['primary_seed_delta_per_query'][i,j],blocks) for j,seed in enumerate(SEEDS)}
        summary['primary_backtrace'][m] = mass_summary({key: arrays['primary_'+key+'_mass_by_direction_query'][i] for key in TURNOVER_KEYS},arrays['primary_delta_per_query'][i])
    for i,arm in enumerate(ARMS):
        summary['seed_means'][arm] = {str(seed): float(spq[i,j].mean()) for j,seed in enumerate(SEEDS)}
        summary['seed_deltas_vs_original'][arm] = {str(seed): paired_summary(spq[i,j]-pq[0],blocks) for j,seed in enumerate(SEEDS)}
        summary['seed_pairwise_ordering_error'][arm] = {str(seed): dict(error=float(arrays['seed_pairwise_error_per_query'][i,j].mean()),
            **paired_summary(arrays['seed_pairwise_error_per_query'][i,j]-err[0],blocks)) for j,seed in enumerate(SEEDS)}
    return arrays, summary


def development_gate(contrasts):
    mean = float(np.mean([contrasts[s]['delta'] for s in SOURCES]))
    positive_lower = [s for s in SOURCES if contrasts[s]['conditional95'][0] > 0]
    checks = dict(positive_equal_source_delta=mean > 0,
                  every_source_delta_nonnegative=all(contrasts[s]['delta'] >= 0 for s in SOURCES),
                  at_least_two_source_conditional_lower_limits_positive=len(positive_lower) >= 2)
    return dict(equal_source_delta=mean, positive_lower_limit_sources=positive_lower,
                source_deltas={s: contrasts[s]['delta'] for s in SOURCES}, **checks, passed=all(checks.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='results', help='New relative directory, for deterministic scoring replay.')
    args = parser.parse_args()
    out = _relative_path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Output is nonempty; preserve the frozen run and use a new replay directory.')
    plan, packets, checks = checks_before_reference()
    out.mkdir(parents=True, exist_ok=True)
    (out/'per_task').mkdir(); (out/'per_source').mkdir()
    dump(out/'SCORING_LOCK.json', dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        score_code_sha256=sha(__file__), common_code_sha256=sha(ROOT/'common.py'), plan_sha256=sha(ROOT/'PLAN_LOCK.json'),
        all_predictions_verified_before_reference_load=True, checks=checks))
    tasks, sources, files = {}, {}, {}
    for s in SOURCES:
        halves = []
        for h in (0,1):
            task = f'{s}_held{h}'
            p, ep, expected = packets[task]
            assert sha(ep) == expected, ('evaluation hash', task)
            arrays,result = score_task(p,load(ep))
            halves.append(arrays); tasks[task]=result
            dest=out/'per_task'/f'{task}.npz'; np.savez_compressed(dest,**arrays)
            files[str(dest.relative_to(ROOT))]=sha(dest)
        arrays,result=score_source(halves); sources[s]=result
        dest=out/'per_source'/f'{s}.npz'; np.savez_compressed(dest,**arrays)
        files[str(dest.relative_to(ROOT))]=sha(dest)
    aggregate={m:dict(precision=float(np.mean([sources[s]['means'][m]['precision'] for s in SOURCES])),
                     delta_vs_original=float(np.mean([sources[s]['means'][m]['delta'] for s in SOURCES])),
                     pairwise_error=float(np.mean([sources[s]['pairwise_ordering_error'][m]['error'] for s in SOURCES])),
                     pairwise_error_delta_vs_original=float(np.mean([sources[s]['pairwise_ordering_error'][m]['delta'] for s in SOURCES]))) for m in METHODS}
    seed_aggregate={a:{str(seed):float(np.mean([sources[s]['seed_means'][a][str(seed)] for s in SOURCES])) for seed in SEEDS} for a in ARMS}
    gates={m:development_gate({s:sources[s]['primary_contrasts'][m] for s in SOURCES}) for m in COMPARATORS}
    summary=dict(identity='Frozen exploratory boundary-learning evaluation on previously accessed references; A is imported historical output; no post-score retuning or arm selection',
        K=K, methods=list(METHODS), primary_arm=PRIMARY, comparator_names=list(COMPARATORS), seeds=list(SEEDS),
        sources=sources, tasks=tasks, equal_source_aggregate=aggregate, equal_source_seed_means=seed_aggregate,
        primary_contrast_gates=gates, primary_development_gate=gates['original_PertRank'],
        secondary_context_comparison=list(SECONDARY),
        secondary_context_equal_source_delta=float(np.mean([sources[s]['secondary_context_contrast']['delta'] for s in SOURCES])),
        aggregation='Average two seed SCORES before ranking each separate held pool; average up/down per query; average two half precisions per query; average queries per source; equal-source mean. Single seeds are diagnostics, not candidates for selection.',
        interval='Paired query-block bootstrap: 2000 draws, seed 2026111518; conditional on fixed candidates, inputs, fitted models and shared reference. No aggregate independent-study population CI; no multiplicity adjustment.',
        gate='Primary boundary_pair versus original: mean delta > 0, every source delta >= 0, positive conditional lower limits in at least two sources. A and B contrasts receive separate identically defined descriptive gates, not a post-hoc joint gate.',
        pairwise_error='Eligible unordered pairs; reference ties excluded; score ties cost .5; no decisive pair fallback .5 with counts retained. Negative delta improves.',
        exchange='All fractional entered/removed, supported/unsupported masses saved versus original and primary comparators. Net supported mass/(2*Q*10)=delta. Source masses average across halves.',
        history='A=base_context reuses the prior neural_no_extra_geometry weights and outputs; it is a historical comparator selected during exploratory development, not a newly confirmed method. B adds boundary context with the old SoftTopK objective; C changes B to support-gap-weighted local boundary pair loss. All arms retain the original wM geometric anchor.',
        scoring_lock_sha256=sha(out/'SCORING_LOCK.json'),files=files)
    dump(out/'SUMMARY.json',summary)
    write_report(out,summary)
    print(json.dumps(dict(equal_source_aggregate=aggregate,primary_contrast_gates=gates),indent=2))


def _interval(z):
    lo,hi=z['conditional95']
    return f"{z['delta']:+.6f} [{lo:+.6f}, {hi:+.6f}]"


def write_report(out,r):
    a,s,g=r['equal_source_aggregate'],r['sources'],r['primary_development_gate']
    lines=['# 名单边界学习：冻结探索性评估','',
        f"预先指定主方法 boundary_pair 对原 PertRank 的发展门槛：**{'通过' if g['passed'] else '未通过'}**。等来源 ΔP@10={g['equal_source_delta']:+.6f}；条件95%区间下限大于零的来源：{', '.join(g['positive_lower_limit_sources']) or '无'}。",'',
        '原始参考在先前实验中已使用；本次不是独立盲测。三个来源共享外部实验，来源间不能当作独立研究。全部臂、种子和来源均报告；未选最好模型、种子、来源或半池。','',
        'A=base_context：复用上一轮不额外输入几何摘要的神经门控；B=boundary_context：加入当前候选池的边界上下文，保留原SoftTopK损失；C=boundary_pair（主方法）：同B输入，改用参考支持差加权的局部边界成对损失。全部臂保留原几何权重。','',
        '先平均两个种子的分数，在各自 held 候选池内计算双方向 P@10；逐查询平均两方向、两半，再来源内平均查询，最后三个来源等权。精确边界并列使用分数化归属，排除 query-self target，不拼接两半。','',
        '| 方法 | '+ ' | '.join(SOURCES)+' | 等来源 P@10 | Δ 原方法 |',
        '|---|'+'---:|'*(len(SOURCES)+2)]
    for m in METHODS:
        vals=' | '.join(f"{s[d]['means'][m]['precision']:.6f}" for d in SOURCES)
        lines.append(f"| {m} | {vals} | {a[m]['precision']:.6f} | {a[m]['delta_vs_original']:+.6f} |")
    lines+=['','## 预先指定主比较','',
        '以下为 boundary_pair 减比较方法的逐查询配对差异。区间来自2000次 query-block bootstrap，固定候选、输入、模型和共享参考；没有全局独立研究总体区间，也没有多重比较调整。','',
        '| 比较方法 | '+' | '.join(SOURCES)+' | 等来源 Δ | 描述性门槛 |',
        '|---|'+'---|'*(len(SOURCES)+2)]
    for m in COMPARATORS:
        vals=' | '.join(_interval(s[d]['primary_contrasts'][m]) for d in SOURCES)
        gate=r['primary_contrast_gates'][m]
        lines.append(f"| {m} | {vals} | {gate['equal_source_delta']:+.6f} | {'通过' if gate['passed'] else '未通过'} |")
    lines+=['','主发展门槛仅对应原 PertRank：等来源平均差异严格为正、每来源差异非负、至少两个来源的条件95%下限严格为正。C−A与C−B使用相同条件单独描述。A复用上一轮预设消融臂，属于历史开发对照；B−A检验边界上下文扩展，C−B检验局部边界训练目标。所有臂仍保留原wM锚点。','',
        '## 各臂相对原方法的条件区间','', '| 来源 | 方法 | ΔP@10 [条件95%] |','|---|---|---|']
    for d in SOURCES:
        for m in ARMS:lines.append(f"| {d} | {m} | {_interval(s[d]['means'][m])} |")
    lines+=['','## 边界上下文的配对对照：B−A','',
        '这是预设次要对照。它比较边界上下文扩展及对应学习结果；不将结果归为单独、普适的生物学机制。','',
        '| 来源 | B−A ΔP@10 [条件95%] |','|---|---|']
    for d in SOURCES:lines.append(f"| {d} | {_interval(s[d]['secondary_context_contrast'])} |")
    lines.append(f"\n等来源B−A={r['secondary_context_equal_source_delta']:+.6f}；没有独立研究层面总体区间。各种子和完整交换质量保存在SUMMARY.json。")
    lines+=['','## 各半池','', '| 任务 | '+' | '.join(METHODS)+' |', '|---|'+'---:|'*len(METHODS)]
    for task,z in r['tasks'].items():
        lines.append('| '+task+' | '+' | '.join(f"{z['means'][m]:.6f}" for m in METHODS)+' |')
    lines+=['','## 每个种子（全部保留）','', '| 方法 | 种子 | '+' | '.join(SOURCES)+' | 等来源 |','|---|---:|'+'---:|'*(len(SOURCES)+1)]
    for m in ARMS:
        for seed in SEEDS:
            key=str(seed);vals=' | '.join(f"{s[d]['seed_means'][m][key]:.6f}" for d in SOURCES)
            lines.append(f"| {m} | {seed} | {vals} | {r['equal_source_seed_means'][m][key]:.6f} |")
    lines+=['','每个种子的主方法三项配对差异及区间、各臂对原方法的区间均保存在 SUMMARY.json；种子没有参与评估后选择。','',
        '## 完整交换回溯与排序错误','',
        '归属变化以分数质量计；支持质量乘固定参考 Top10 归属，不支持质量乘其补数。下表为各来源两半平均质量。进入=移出，净支持/(2Q×10)=ΔP@10 已逐任务及逐来源核验。','',
        '| 来源 | 方法 | 进入/移出 | 进入支持 | 移出支持 | 进入不支持 | 移出不支持 | 净支持 | Δ排序错误 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in SOURCES:
        for m in METHODS:
            b=s[d]['backtrace'][m];e=s[d]['pairwise_ordering_error'][m]
            lines.append(f"| {d} | {m} | {b['entered_mass']:.3f}/{b['removed_mass']:.3f} | {b['entered_supported_mass']:.3f} | {b['removed_supported_mass']:.3f} | {b['entered_unsupported_mass']:.3f} | {b['removed_unsupported_mass']:.3f} | {b['net_supported_mass']:+.3f} | {e['delta']:+.6f} |")
    lines+=['','排序错误使用所有合格无序候选对；排除参考并列，预测并列记0.5；没有决定性对时记0.5并保留计数。误差差异为负表示改善。每来源条件区间、每种子误差以及主方法对A/B的完整交换也在机器可读结果中。','',
        '## 冻结与可复核性','',
        '评分器先核验全部来源预测锁、计划/代码/预测/拟合/归一化及特征视图哈希，再访问独立 evaluation 文件。逐候选归属、进入/移出支持及不支持、逐查询 P@10、种子结果和排序误差均保存在 per_task；两半平均逐查询结果保存在 per_source。SUMMARY.json 保存全部数值，SCORING_LOCK.json 保存评分输入核验记录。','',
        '该门槛只是开发约定。A为上一轮历史模型复用，其小正号是本次开发依据，不是新确认；本轮不得把A选取或任何臂的后验择优当作独立证据。预测后不再调整参数、门槛、候选恢复规则或选择有利来源。','']
    (out/'REPORT_zh.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
