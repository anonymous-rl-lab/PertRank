#!/usr/bin/env python3
"""R20: fixed-rule real 20-cell replacements, NOT a refit of duplicated old cells.

Only --regression may be run before the new experiment's final freeze. Evaluation
requires --freeze-record and verifies this script and the selected fixed context.
Input state comes from the R20 reader; matrices use [target, repetition, ...].
No donor is treated as an independent copy of a shared external reference. The
cross-donor summary contains point estimates only, never averaged CI endpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_CODE = PROJECT / 'restored/restored/working_v9_with_v6_inputs/code/repro_v10'
REFERENCE = HERE / 'reference/external_uncertainty'
K_DIRECTION, BETA, DEPTH = 5, 0.9, 20
BOOTSTRAPS, BOOTSTRAP_SEED = 2000, 20261006
ARMS = ('raw', 'w0y', 'W1')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, x):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def finite(name, x):
    if not np.isfinite(x).all():
        raise ValueError(f'{name} contains nonfinite values')


def own_mask(query_ids, target_ids):
    q = np.asarray(query_ids).astype(str); t = np.asarray(target_ids).astype(str)
    if len(np.unique(q)) != len(q) or len(np.unique(t)) != len(t):
        raise ValueError('query and target identifiers must be unique on their axes')
    return q[:, None] == t[None, :]


def probabilities(effects, own, k):
    """Exact independent tie membership, signed row order + queries then - queries."""
    y = np.asarray(effects, dtype=np.float64)
    if y.ndim != 2 or own.shape != y.shape:
        raise ValueError('effects/own mask shape mismatch')
    finite('effects', y)
    if np.any(y.shape[1]-own.sum(1) < k):
        raise ValueError('fewer than K eligible candidates')
    s = np.concatenate([y, -y], axis=0)
    mask = np.concatenate([own, own], axis=0)
    s[mask] = -np.inf
    kth = np.partition(s, s.shape[1]-k, axis=1)[:, -k, None]
    above, tied = s > kth, s == kth
    slots = k-above.sum(1)
    p = above.astype(np.float64) + tied*(slots/tied.sum(1))[:, None]
    if not np.allclose(p.sum(1), k, rtol=0, atol=1e-12) or p[mask].any():
        raise AssertionError('invalid top-K membership')
    return p


def score(effects, truth, own, k):
    q = effects.shape[0]
    p = probabilities(effects, own, k)
    t = probabilities(truth, own, k)
    return (p*t).sum(1).reshape(2, q).T/k


def query_bootstrap(a, b, blocks, seed=BOOTSTRAP_SEED):
    """Conditional paired query-block interval; cell groups first averaged by caller."""
    d = np.asarray(a)-np.asarray(b)
    if d.ndim != 2 or d.shape[1] != 2:
        raise ValueError('query bootstrap expects [query, two directions]')
    d = d.mean(1)
    blocks = np.asarray(blocks)
    if len(blocks) != len(d):
        raise ValueError('query blocks length mismatch')
    labels = np.unique(blocks)
    members = {v: np.flatnonzero(blocks == v) for v in labels}
    rng = np.random.default_rng(seed)
    draws = np.array([d[np.concatenate([members[v] for v in rng.choice(labels, len(labels), replace=True)])].mean()
                      for _ in range(BOOTSTRAPS)])
    nonpositive = int(np.count_nonzero(draws <= 0))
    return {'difference': float(d.mean()), 'conditional_query_block_95_interval': np.quantile(draws, [.025, .975]).tolist(),
            'bootstrap_nonpositive_count': nonpositive, 'bootstrap_fraction_nonpositive': nonpositive/BOOTSTRAPS,
            'draws': BOOTSTRAPS, 'seed': seed, 'query_blocks': len(labels),
            'scope': 'conditional on the realized input cell groups, fixed controls, query panel, candidate pool and reference; not a population or cell-sampling confidence interval'}


def weights(m2, vb, pmean, pvar, lam, trc, nc, g, beta=BETA):
    """Target means centered on fixed controls; projection sample variance uses ddof=1."""
    m2 = np.asarray(m2, float); vb = np.asarray(vb, float)
    pm = np.asarray(pmean, float)[..., :K_DIRECTION]
    pv = np.asarray(pvar, float)[..., :K_DIRECTION]
    la = np.asarray(lam, float)[:K_DIRECTION]
    if m2.shape != vb.shape or pm.shape != m2.shape+(K_DIRECTION,) or pv.shape != pm.shape:
        raise ValueError('weight sufficient-statistic shape mismatch')
    for name, x in [('m2', m2), ('vb', vb), ('pmean', pm), ('pvar', pv)]:
        finite(name, x)
    if m2.min() < -1e-10 or vb.min() < -1e-10 or pv.min() < -1e-10:
        raise AssertionError('negative energy/variance beyond roundoff')
    a = m2-beta*(pm*pm).sum(-1)/g
    nu = vb/DEPTH+trc/(g*nc)-beta*(pv/DEPTH+la/nc).sum(-1)/g
    if a.min() < -1e-10 or nu.min() < -1e-10:
        raise AssertionError('metric energy/noise violates PSD check')
    return np.clip(1-nu/np.maximum(a, 1e-300), 0, 1)


def rebuild_arms(stats, prefix, context):
    m2, vb = stats[prefix+'m2'], stats[prefix+'vb']
    pm, pv, y = stats[prefix+'pmean'], stats[prefix+'pvar'], stats[prefix+'Yq']
    g = len(context['gene_ids_keep']); nc = float(context['n_control']); trc = float(context['trC'])
    w0 = weights(m2, vb, pm, pv, context['lam'], trc, nc, g, beta=0)
    explicit = np.clip(1-(vb/DEPTH+trc/(g*nc))/np.maximum(m2, 1e-300), 0, 1)
    err = float(np.max(np.abs(w0-explicit)))
    if err > 1e-12:
        raise AssertionError('beta=0 failed to reproduce Euclidean weighting')
    w1 = weights(m2, vb, pm, pv, context['lam'], trc, nc, g)
    if y.shape[:2] != m2.shape or y.ndim != 3 or y.shape[2] != len(context['query_gene_ids']):
        raise ValueError('Yq must have [target, repetition, union-query] shape')
    finite('Yq', y)
    # arm, repetition, union-query, target
    pred = np.stack([y, y*w0[..., None], y*w1[..., None]]).transpose(0, 2, 3, 1)
    return pred, w0, w1, err


def summarize_scores(all_scores, blocks, k):
    """all_scores = arm, repetition, query, direction; never pool repetitions as queries."""
    _, r, _, _ = all_scores.shape
    output = {'repetitions': [], 'repetition_average': {}}
    pairs = [('W1_minus_w0y', 2, 1), ('W1_minus_raw', 2, 0), ('w0y_minus_raw', 1, 0)]
    for i in range(r):
        s = all_scores[:, i]
        item = {'repetition': i, 'means': dict(zip(ARMS, map(float, s.mean((1, 2)))))}
        for label, a, b in pairs:
            item[label] = query_bootstrap(s[a], s[b], blocks)
            item[label]['extra_expected_members_per_list'] = k*item[label]['difference']
        output['repetitions'].append(item)
    ave = all_scores.mean(1)
    average = {'means': dict(zip(ARMS, map(float, ave.mean((1, 2)))))}
    for label, a, b in pairs:
        diffs = (all_scores[a]-all_scores[b]).mean((1, 2))
        average[label] = query_bootstrap(ave[a], ave[b], blocks)
        average[label].update({'extra_expected_members_per_list': k*average[label]['difference'],
                              'realized_group_differences': diffs.tolist(),
                              'realized_group_min_max': [float(diffs.min()), float(diffs.max())],
                              'realized_groups_positive': int(np.sum(diffs > 0)),
                              'realized_groups_negative': int(np.sum(diffs < 0)),
                              'realized_groups_zero': int(np.sum(diffs == 0))})
    output['repetition_average'] = average
    return output


def nomination_redistribution(predictions, own, k):
    """Changes in marginal membership probabilities, not realized independently tied lists."""
    pp=[[probabilities(predictions[a,r],own,k) for r in range(predictions.shape[1])] for a in range(3)]
    out={'definition':'For each signed query, sum max(p_W1-p_comparator,0); equal to removed probability mass. This is movement of marginal top-K nomination probability, not expected turnover of two independently tie-broken lists.'}
    for label,base in [('W1_vs_w0y',1),('W1_vs_raw',0)]:
        vals=[]
        for r in range(predictions.shape[1]):
            d=pp[2][r]-pp[base][r];add=np.maximum(d,0).sum(1);rem=np.maximum(-d,0).sum(1)
            if not np.allclose(add,rem,rtol=0,atol=1e-12):raise AssertionError('nomination probability balance failed')
            vals.append(float(add.mean()))
        out[label]={'mean_probability_slots_moved_by_repetition':vals,
                    'repetition_average_probability_slots_moved':float(np.mean(vals)),
                    'repetition_average_fraction_of_K':float(np.mean(vals)/k)}
    return out


def external_fixed(context, donor):
    """Use exactly the previously frozen A19-R candidate and query order."""
    legacy = np.load(REFERENCE/'A19R'/donor/'predictions_locked.npz', allow_pickle=False)
    if not np.array_equal(context['external_names'].astype(str), legacy['target_symbols'].astype(str)):
        raise AssertionError('external candidates changed from A19R')
    if not np.array_equal(context['external_ids'].astype(str), legacy['target_ids'].astype(str)):
        raise AssertionError('external target identifiers changed from A19R')
    qmap = {str(g): i for i, g in enumerate(context['query_gene_ids'])}
    pos = np.array([qmap[str(g)] for g in legacy['query_ids']], int)
    ref = np.load(REFERENCE/'results/query_reference.npz', allow_pickle=False)
    rq = {str(g): i for i, g in enumerate(ref['query_ids'])}
    rt = {str(g): i for i, g in enumerate(ref['target_symbols'])}
    qi = [rq[str(g)] for g in legacy['query_ids']]
    ti = [rt[str(g)] for g in legacy['target_symbols']]
    t = ref['mean_effect'][np.ix_(qi, ti)]
    mask = own_mask(legacy['query_ids'], legacy['target_ids'])
    if not np.array_equal(mask, legacy['own_mask']):
        raise AssertionError('external own-gene mask mismatch')
    return pos, t, mask, legacy['query_blocks']


def parse_freeze(record):
    p = Path(record).resolve(); out = {}
    for line in p.read_text().splitlines():
        sp = line.split(maxsplit=1)
        if len(sp) == 2 and len(sp[0]) == 64:
            n = sp[1].lstrip('*'); q = Path(n)
            if not q.is_absolute(): q = p.parent/q
            out[q.resolve()] = sp[0]
    return out


def verify_freeze(record, required):
    entries = parse_freeze(record)
    for p in required:
        p = Path(p).resolve()
        if p not in entries or entries[p] != sha(p):
            raise AssertionError(f'not in matching final freeze: {p}')
    return {str(p): sha(p) for p in required}


def predict(args):
    """Input-only phase: lazy NPZ access NEVER indexes truth or a reference effect."""
    cpath, spath = Path(args.context).resolve(), Path(args.state).resolve()
    context = np.load(cpath, allow_pickle=False); state = np.load(spath, allow_pickle=False)
    donor = args.donor
    if str(state['context_sha256']) != sha(cpath):
        raise AssertionError('aggregate is bound to a different context')
    selection_path=cpath.parent/'selection.npz'
    if str(state['selection_sha256']) != sha(selection_path):
        raise AssertionError('aggregate is bound to a different cell selection')
    if str(state['donor']) != donor or str(context['donor']) != donor:
        raise AssertionError('donor identity mismatch')
    if int(state['variance_ddof']) != 1 or 'proj_control' not in str(state['projection_mean_centering']):
        raise AssertionError('aggregate does not use R20b fixed projection centering/ddof1')
    for prefix, expected_repetitions in [('internal',5),('external',2)]:
        counts=state[prefix+'_n']
        if counts.ndim!=2 or counts.shape[1]!=expected_repetitions or not np.all(counts==20):
            raise AssertionError(f'{prefix} incomplete or wrong-sized real-cell groups')
    if donor not in ('D3_Rest','D1_Rest','D2_Rest'):
        raise ValueError('only previously opened donors permitted')
    locked = [Path(__file__).resolve(), cpath, selection_path,
              REFERENCE/'results/query_reference.npz', REFERENCE/'A19R'/donor/'predictions_locked.npz']
    verified = verify_freeze(args.freeze_record, locked)
    for key in ['internal_names','internal_ids','external_names','external_ids']:
        if not np.array_equal(state[key].astype(str), context[key].astype(str)):
            raise AssertionError(f'state/context identity mismatch: {key}')
    inp, w0, w1, b0 = rebuild_arms(state, 'internal_', context)
    ext, ew0, ew1, eb0 = rebuild_arms(state, 'external_', context)
    if inp.shape[-1]!=len(context['internal_names']) or ext.shape[-1]!=len(context['external_names']):
        raise AssertionError('prediction candidate axis mismatch')
    if inp.shape[1] != 5 or ext.shape[1] != 2:
        raise AssertionError('all five internal and two external groups must be available')
    out = {'identity':'R20 input-only prediction, all three donors must be locked before new scoring',
           'donor': donor, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'state_path':str(spath), 'state_sha256':sha(spath), 'context_path':str(cpath), 'context_sha256':sha(cpath),
           'freeze_record_path':str(Path(args.freeze_record).resolve()), 'freeze_record_sha256':sha(args.freeze_record),
           'script_sha256':sha(__file__), 'verified_frozen_inputs':verified,
           'parameters':{'directions':5,'beta':.9,'cells_per_target_per_repetition':20,'bootstrap_seed':BOOTSTRAP_SEED,'bootstrap_draws':BOOTSTRAPS},
           'weight_checks':{'internal_beta0_maxabs':b0,'external_beta0_maxabs':eb0}}
    for tag,ws,bs in [('internal',w1,w0),('external',ew1,ew0)]:
        out['weight_checks'][tag+'_W1_zero_fraction_by_repetition'] = np.mean(ws==0,axis=0).tolist()
        out['weight_checks'][tag+'_w0y_zero_fraction_by_repetition'] = np.mean(bs==0,axis=0).tolist()
        out['weight_checks'][tag+'_W1_mean_by_repetition'] = ws.mean(0).tolist()
        out['weight_checks'][tag+'_degenerate_W1_repetitions'] = [bool(np.ptp(ws[:,i])==0) for i in range(ws.shape[1])]
    dest=Path(args.outdir).resolve();dest.mkdir(parents=True,exist_ok=True)
    pp=dest/(donor+'_predictions.npz')
    np.savez_compressed(pp,internal_predictions=inp,external_predictions=ext,
                        internal_w0y=w0,internal_W1=w1,external_w0y=ew0,external_W1=ew1,
                        internal_names=context['internal_names'],internal_ids=context['internal_ids'],
                        external_names=context['external_names'],external_ids=context['external_ids'],
                        query_gene_ids=context['query_gene_ids'],arm_names=np.array(ARMS))
    out['prediction_path']=str(pp);out['prediction_sha256']=sha(pp)
    dump(dest/(donor+'_prediction_record.json'),out)
    print(json.dumps({'donor':donor,'input_only_predictions_sha256':out['prediction_sha256'],'weight_checks':out['weight_checks']},indent=2))


def lock_predictions(args):
    path=Path(args.prediction_dir).resolve(); records={}
    for donor in ('D3_Rest','D1_Rest','D2_Rest'):
        record=path/(donor+'_prediction_record.json');r=json.loads(record.read_text())
        if r['donor']!=donor or r['script_sha256']!=sha(__file__):
            raise AssertionError('prediction record mismatch')
        for filekey,hashkey in [('prediction_path','prediction_sha256'),('context_path','context_sha256'),('state_path','state_sha256'),('freeze_record_path','freeze_record_sha256')]:
            if sha(r[filekey])!=r[hashkey]:raise AssertionError(f'prediction lock hash mismatch: {donor} {filekey}')
        records[donor]={'record_path':str(record),'record_sha256':sha(record),**r}
    lock={'identity':'R20 all-source prediction lock, before new scoring',
          'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'script_sha256':sha(__file__),'donors':records}
    dump(args.out,lock);print(json.dumps({'prediction_lock_sha256':sha(args.out),'donors':list(records)},indent=2))


def evaluate(args):
    lock=json.loads(Path(args.prediction_lock).read_text())
    if lock['script_sha256']!=sha(__file__) or set(lock['donors'])!={'D3_Rest','D1_Rest','D2_Rest'}:
        raise AssertionError('all three donors must be locked under this exact evaluation script before scoring')
    for donor,r in lock['donors'].items():
        for filekey,hashkey in [('prediction_path','prediction_sha256'),('record_path','record_sha256'),('context_path','context_sha256')]:
            if sha(r[filekey])!=r[hashkey]:raise AssertionError(f'all-source lock changed: {donor} {filekey}')
        verify_freeze(r['freeze_record_path'],[Path(__file__).resolve(),Path(r['context_path']),Path(r['context_path']).parent/'selection.npz',REFERENCE/'results/query_reference.npz',REFERENCE/'A19R'/donor/'predictions_locked.npz'])
    donor=args.donor;r=lock['donors'][donor]
    context=np.load(r['context_path'],allow_pickle=False);pred=np.load(r['prediction_path'],allow_pickle=False)
    inp,ext=pred['internal_predictions'],pred['external_predictions']
    out={'identity':'R20 development stability analysis on newly selected real cell groups in opened donors',
         'donor':donor,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         'prediction_lock_sha256':sha(args.prediction_lock),'prediction_record':r,
         'scope':'new target-cell replacement with freshly estimated target weights; fixed original controls and metric. Shared queries/reference/donor are not independent replicates.',
         'internal':{},'external':{}}
    arrays={}
    truth=context['truth'];ids=context['internal_ids'];qids=context['query_gene_ids']
    for rule,pk,bk in [('responsive','responsive_positions','blocks_responsive'),('terciles','tercile_positions','blocks_terciles')]:
        pos=context[pk].astype(int);mask=own_mask(qids[pos],ids)
        scores=np.array([[score(arm[rep,pos],truth[pos],mask,25) for rep in range(5)] for arm in inp])
        item=summarize_scores(scores,context[bk],25)
        item.update({'role':'primary' if rule=='responsive' else 'secondary','top_k':25,'queries':len(pos),'targets':len(ids),'own_exclusions':int(mask.sum()),
                     'random_precision':float(np.mean(25/(len(ids)-mask.sum(1))))})
        item['nomination_probability_redistribution']=nomination_redistribution(inp[:,:,pos,:],mask,25)
        out['internal'][rule]=item;arrays['internal_'+rule+'_scores']=scores
    pos,t,mask,blocks=external_fixed(context,donor)
    esc=np.array([[score(arm[rep,pos],t,mask,10) for rep in range(2)] for arm in ext])
    out['external']=summarize_scores(esc,blocks,10)
    out['external'].update({'top_k':10,'queries':len(pos),'targets':mask.shape[1],'own_exclusions':int(mask.sum()),
                            'random_precision':float(np.mean(10/(mask.shape[1]-mask.sum(1)))),
                            'reference':'fixed separately generated arrayed-knockout RNA reference, same A19R pool; not a new independent reference',
                            'reference_sha256':sha(REFERENCE/'results/query_reference.npz')})
    out['external']['nomination_probability_redistribution']=nomination_redistribution(ext[:,:,pos,:],mask,10)
    arrays['external_scores']=esc
    dest=Path(args.outdir);dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/(donor+'_query_scores.npz'),**arrays)
    out['query_scores_sha256']=sha(dest/(donor+'_query_scores.npz'))
    dump(dest/(donor+'_results.json'),out)
    print(json.dumps({'donor':donor,'internal_primary':out['internal']['responsive']['repetition_average'],'external':out['external']['repetition_average']},indent=2))


def archived_main(code, donor):
    stage=code/'same_budget_weight/stage3'
    if donor=='D3_Rest':
        z=np.load(code/'data/e4_tcell.npz',allow_pickle=False); w=np.load(stage/'data/e4_w1_stage3.npz',allow_pickle=False)
        mp=np.load(code/'same_budget_weight/inputs/own_gene_map.npz',allow_pickle=False)
        cmq,ceq=z['cm_q'],z['ce_q'];ps=w['proj_sum'][:,:5];pq=w['proj_sumsq'][:,:5]
        m2,vb=w['m2_20'],w['varbar_20'];g=int(w['n_genes'])
        expected=json.loads((stage/'results/stage3_w1_e4.json').read_text())['cells']
        ids=mp['ids']
    else:
        file='d1_confirm_state.npz' if donor=='D1_Rest' else 'd2_prospective_state.npz'
        z=np.load(stage/'data'/file,allow_pickle=False);w=z
        cmq,ceq=z['cm'][z['qu']],z['ce'][z['qu']];ps=z['proj_sum'][0,:,:5];pq=z['proj_sumsq'][0,:,:5]
        m2,vb=z['m2_20'],z['vb_20'];g=len(z['genes_keep'])
        if donor=='D1_Rest':
            mp=np.load(stage/'data/d1_own_gene_map.npz',allow_pickle=False);ids=mp['ids']
            expected=json.loads((stage/'results/d1_self_gene_correction.json').read_text())['cells']['corrected']
        else:
            ids=z['ids'];expected=json.loads((stage/'results/b7_d2_prospective.json').read_text())['test1']['cells']
    y=z['S0_q']/20-cmq[:,None];t=z['S4_q']/100-ceq[:,None]
    pm=ps/20-w['proj_control'][None,:5];pv=(pq-ps*ps/20)/19
    w1=weights(m2,vb,pm,pv,w['eigenvalues'],float(w['control_trace']),float(w['n_control']),g)
    # D3's historical Euclidean baseline was computed from stored reduced float32
    # energies; retain it in the regression, while all new-group arms use the same statistics.
    w0=np.clip(1-(z['vb_20']/20+float(z['cvm'])/float(w['n_control']))/z['m2_20'],0,1)
    return z,ids,y,t,w0,w1,expected


def regression(args):
    """Only already-scored historical arrays, with independent scorer crosschecks."""
    code=Path(args.code).resolve();sys.path.insert(0,str(code));from pr10 import score_rows
    report={'identity':'offline regression against existing opened/frozen results only; no new cell scores', 'sources':{},'external':{}}
    for donor in ('D3_Rest','D1_Rest','D2_Rest'):
        z,ids,y,t,w0,w1,expected=archived_main(code,donor)
        report['sources'][donor]={}
        for rule,qkey in [('responsive','q_resp'),('terciles','q_terc')]:
            pos=np.searchsorted(z['qu'],z[qkey]);qids=z['genes_keep'][z[qkey]];mask=own_mask(qids,ids)
            selfq=np.array([next(iter(np.flatnonzero(mask[:,j])),-1) for j in range(len(ids))],int)
            got={};err=0
            for arm,weight in zip(ARMS,(np.ones(len(ids)),w0,w1)):
                v=score(y[pos]*weight[None,:],t[pos],mask,25)
                archived=score_rows(y[pos]*weight[None,:],t[pos],selfq,K=25)['exp']
                err=max(err,float(np.max(abs(v-archived))));got[arm]=float(v.mean())
                if abs(got[arm]-expected[rule+' 20'][arm])>1e-12:
                    raise AssertionError(f'{donor} {rule} {arm} old result mismatch {got[arm]} vs {expected[rule+" 20"][arm]}')
            if err>1e-12:raise AssertionError('independent scorer disagreement')
            report['sources'][donor][rule]={'means':got,'W1_minus_w0y':got['W1']-got['w0y'],'score_rows_maxabs':err}
        old=np.load(REFERENCE/'A19R'/donor/'predictions_locked.npz',allow_pickle=False)
        ref=np.load(REFERENCE/'results/query_reference.npz',allow_pickle=False)
        qi={str(g):i for i,g in enumerate(ref['query_ids'])};ti={str(g):i for i,g in enumerate(ref['target_symbols'])}
        tt=ref['mean_effect'][np.ix_([qi[str(g)] for g in old['query_ids']],[ti[str(g)] for g in old['target_symbols']])]
        got={str(a):float(10*score(old['effects'][i],tt,old['own_mask'],10).mean()) for i,a in enumerate(old['arm_names'])}
        expectedext=json.loads((REFERENCE/'A19R/point_results.json').read_text())['sources'][donor]['means']
        err=max(abs(got[k]-expectedext[k]) for k in got)
        if err>1e-12:raise AssertionError(f'external regression mismatch {donor} {err}')
        report['external'][donor]={'expected_members_per_10':got,'archived_maxabs':err}
    const=np.zeros((2,31));own=np.zeros_like(const,dtype=bool);own[0,0]=True
    s=score(const,const,own,10)
    assert np.allclose(s[:,0],[10/30,10/31]) and np.allclose(s[:,0],s[:,1])
    rng=np.random.default_rng(17);u=np.linalg.qr(rng.normal(size=(9,5)))[0]
    x=rng.normal(size=(4,2,20,9))+rng.normal(size=(4,2,1,9));cm=rng.normal(size=9)
    mu=x.mean(2);vb=x.var(2,ddof=1).mean(2);m2=((mu-cm)**2).mean(2)
    pp=x@u;pm=pp.mean(2)-cm@u;pv=pp.var(2,ddof=1)
    ww=weights(m2,vb,pm,pv,np.ones(5),9,2000,9)
    mm=np.eye(9)-BETA*u@u.T;direct=np.empty((4,2))
    for j in range(4):
        for r in range(2):
            yy=mu[j,r]-cm;v=np.cov(x[j,r],rowvar=False,ddof=1)/20+np.eye(9)/2000
            direct[j,r]=np.clip(1-np.trace(mm@v)/(yy@mm@yy),0,1)
    metric_error=float(np.max(abs(ww-direct)))
    if metric_error>1e-12:raise AssertionError('projection statistics do not match full covariance formula')
    class InputOnlyContext(dict):
        def __getitem__(self,key):
            if key=='truth':raise AssertionError('prediction must never access truth')
            return super().__getitem__(key)
    cx=InputOnlyContext(gene_ids_keep=np.arange(9),query_gene_ids=np.arange(9),lam=np.ones(5),trC=9,n_control=2000)
    st={'internal_m2':m2,'internal_vb':vb,'internal_Yq':mu-cm,'internal_pmean':pm,'internal_pvar':pv}
    pred,w0test,w1test,beta0=rebuild_arms(st,'internal_',cx)
    assert np.allclose(pred[2],((mu-cm)*ww[...,None]).transpose(1,2,0),rtol=0,atol=1e-12)
    report['tests']={'constant_ties_and_self_exclusion':'PASS','historical_results_and_independent_scorer':'PASS',
                     'projection_formula_vs_full_covariance_maxabs':metric_error,'reader_contract_axes_and_input_only_prediction':'PASS'}
    dump(args.out,report);print(json.dumps(report,indent=2))


def summary(args):
    result=[json.loads(Path(p).read_text()) for p in args.results]
    if len(result)!=3 or {x['donor'] for x in result}!={'D3_Rest','D1_Rest','D2_Rest'}:
        raise ValueError('summary requires all three fixed donors exactly once')
    out={'identity':'descriptive equal-source summary; donors share queries and external reference, no pooled CI', 'donors':[x['donor'] for x in result]}
    for label,sel in [('responsive',lambda x:x['internal']['responsive']['repetition_average']),('terciles',lambda x:x['internal']['terciles']['repetition_average']),('external',lambda x:x['external']['repetition_average'])]:
        out[label]={key:float(np.mean([sel(x)[key]['difference'] for x in result])) for key in ('W1_minus_w0y','W1_minus_raw','w0y_minus_raw')}
    dump(args.out,out);print(json.dumps(out,indent=2))


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    r=sub.add_parser('regression');r.add_argument('--code',default=str(DEFAULT_CODE));r.add_argument('--out',default=str(HERE/'historical_regression.json'))
    e=sub.add_parser('predict');e.add_argument('--donor',required=True);e.add_argument('--context',required=True);e.add_argument('--state',required=True);e.add_argument('--freeze-record',required=True);e.add_argument('--outdir',required=True)
    l=sub.add_parser('lock-predictions');l.add_argument('--prediction-dir',required=True);l.add_argument('--out',required=True)
    e=sub.add_parser('evaluate');e.add_argument('--donor',required=True);e.add_argument('--prediction-lock',required=True);e.add_argument('--outdir',required=True)
    s=sub.add_parser('summary');s.add_argument('--results',nargs=3,required=True);s.add_argument('--out',required=True)
    a=p.parse_args();{'regression':regression,'predict':predict,'lock-predictions':lock_predictions,'evaluate':evaluate,'summary':summary}[a.mode](a)

if __name__=='__main__': main()
