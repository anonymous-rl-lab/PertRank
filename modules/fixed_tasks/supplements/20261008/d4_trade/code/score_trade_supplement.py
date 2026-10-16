#!/usr/bin/env python3
"""Post-hoc D4 TRADE-scalar supplement; original H1/H2 remain unchanged.

baseline reads only already-published D4 assets. score requires a root-issued
release bound to the complete weight file, baseline inputs, and scoring code.
replay independently recomputes archived compact query scores without refitting.
"""
from pathlib import Path
import argparse, csv, hashlib, itertools, json, os, time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import numpy as np
import exact_score as exact

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[1] / 'restored/v18_d4_confirmation'
BASE_ARMS = ['raw', 'Euclidean', 'PertRank', 'PR_M']
TRADE = 'TRADE_scalar'
K, SEED, DRAWS, MARGIN = 25, 2026, 2000, .005
PANELS = [('responsive','responsive_positions','blocks_responsive'),
          ('terciles','tercile_positions','blocks_terciles')]

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda:f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def utc(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
def read_json(p): return json.loads(Path(p).read_text())
def dump(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def load_npz(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in (z.files if keys is None else keys)}

def source_paths(root):
    data = root / 'data/D4_Rest'
    return {'predictions': data/'predictions.npz', 'truth':data/'truth.npz',
            'context':data/'query_context.npz', 'control':data/'control_model.npz',
            'cohort':data/'cohort.npz',
            'old_scores':root/'results/query_scores_and_depth_bootstrap.npz',
            'old_result':root/'results/D4_RESULT.json',
            'old_prediction_lock':root/'protocol/PREDICTION_LOCK.json',
            'old_scorer':root/'evaluation/score_d4.py',
            'old_exact_scorer':root/'evaluation/vendor/exact_score.py'}

def load_original(root):
    paths = source_paths(root)
    p = load_npz(paths['predictions'], ['predictions','weights','Y','arm_names','target_ids','query_gene_ids','donor'])
    t = load_npz(paths['truth'], ['truth','target_ids','query_gene_ids','donor'])
    c = load_npz(paths['context'])
    assert list(p['arm_names']) == BASE_ARMS
    assert str(p['donor']) == str(t['donor']) == 'D4_Rest'
    assert len(c['target_ids']) == 589 and len(c['query_gene_ids']) == 593
    for state in (p,t):
        for key in ('target_ids','query_gene_ids'):
            assert np.array_equal(state[key], c[key]), key
    assert p['predictions'].shape == (4,5,593,589)
    assert p['weights'].shape == (4,5,589)
    assert p['Y'].shape == (5,593,589) and t['truth'].shape == (593,589)
    assert np.array_equal(p['Y'], p['predictions'][0])
    assert np.array_equal(p['predictions'], p['Y'][None] * p['weights'][:,:,None,:])
    assert np.array_equal(p['weights'][0], np.ones((5,589)))
    for x in (p['predictions'],p['weights'],t['truth']): assert np.isfinite(x).all()
    assert np.all((p['weights'] >= 0) & (p['weights'] <= 1))
    for _,pk,bk in PANELS:
        assert len(c[pk]) == len(np.unique(c[pk])) == 300
        assert len(c[bk]) == 300 and len(np.unique(c[bk])) == 30
    assert sha(HERE/'exact_score.py') == sha(paths['old_exact_scorer'])
    return p,t,c,paths

def fullsort_score(y, t, mask):
    """Independent full-sort implementation: eligible candidates only, per row."""
    out = np.empty((len(y),2), dtype=np.float64)
    for q in range(len(y)):
        use = np.flatnonzero(~mask[q])
        assert len(use) >= K
        for sign_index, sign in enumerate((1,-1)):
            memberships=[]
            for row in (y[q,use]*sign,t[q,use]*sign):
                threshold = np.sort(row)[-K]
                above = row > threshold
                at = row == threshold
                membership = np.zeros(len(use), dtype=float)
                membership[above] = 1.
                membership[at] = (K-int(above.sum()))/int(at.sum())
                assert abs(membership.sum()-K) <= 1e-12
                memberships.append(membership)
            out[q,sign_index] = np.dot(*memberships)/K
    return out

def score_all(y, truth, context):
    out, audit = {}, {}
    for panel,pk,bk in PANELS:
        pos = context[pk]; mask = exact.own_mask(context['query_gene_ids'][pos],context['target_ids'])
        scores = np.empty((len(y),5,300,2), dtype=float)
        max_error = 0.
        for arm in range(len(y)):
            for group in range(5):
                scores[arm,group] = exact.score(y[arm,group,pos],truth[pos],mask,K)
                independent = fullsort_score(y[arm,group,pos],truth[pos],mask)
                err = float(np.max(np.abs(independent-scores[arm,group])))
                max_error = max(max_error,err)
                assert err <= 1e-12, ('independent fullsort mismatch',panel,arm,group,err)
        out[panel+'_scores'] = scores
        out[panel+'_query_ids'] = context['query_gene_ids'][pos]
        out[panel+'_query_blocks'] = context[bk]
        audit[panel] = {'scores_checked':int(scores.size), 'max_abs_difference':max_error,
                        'own_exclusions':int(mask.sum()), 'all_scores_passed':True}
    return out,audit

def pair(a,b,blocks):
    z = exact.query_bootstrap(a,b,blocks,seed=SEED)
    assert z['draws'] == DRAWS
    lo,hi = z['conditional_query_block_95_interval']
    z.update(statistical_direction='superiority' if lo>0 else ('inferiority' if hi<0 else 'unresolved'),
             practical_equivalence='established_within_0.005' if lo>=-MARGIN and hi<=MARGIN else 'not_established',
             practical_equivalence_margin=MARGIN, extra_reference_members_per_25=K*z['difference'])
    return z

def summaries(scores, arms, blocks):
    order = ['raw','Euclidean','PR_M'] + ([TRADE] if TRADE in arms else []) + ['PertRank']
    contrasts = [(hi+'_minus_'+lo,arms.index(hi),arms.index(lo)) for lo,hi in itertools.combinations(order,2)]
    result={'repetitions':[]}
    for g in range(5):
        r = {'repetition':g,'means':{a:float(scores[i,g].mean()) for i,a in enumerate(arms)}}
        for name,i,j in contrasts: r[name] = pair(scores[i,g],scores[j,g],blocks)
        result['repetitions'].append(r)
    # First average five input groups within each query/sign. pair averages signs
    # within each query and resamples paired query blocks (never input groups).
    averaged = scores.mean(1)
    r = {'means':{a:float(averaged[i].mean()) for i,a in enumerate(arms)}}
    for name,i,j in contrasts:
        z = pair(averaged[i],averaged[j],blocks)
        d = (scores[i]-scores[j]).mean((1,2))
        z.update(real_group_differences=d.tolist(),real_group_min_max=[float(d.min()),float(d.max())],
                 real_groups_positive=int((d>0).sum()),real_groups_negative=int((d<0).sum()),real_groups_zero=int((d==0).sum()))
        r[name]=z
    result['repetition_average']=r
    return result

def verify_old(scores, original, archived):
    check={}
    for panel,_,bk in PANELS:
        s=scores[panel+'_scores'][:4]
        assert np.array_equal(s,archived[panel+'_scores']), ('old score changed',panel)
        check[panel]={'score_values_exact':int(s.size)}
        # Compare point and CI records to the original published report.
        for group,record in [('average',original['panels'][panel]['repetition_average'])]:
            for arm,val in record['means'].items():
                actual=float(s.mean(1)[BASE_ARMS.index(arm)].mean())
                assert actual==val,(panel,arm,actual,val)
            for name,val in record.items():
                if '_minus_' not in name: continue
                a,b=name.split('_minus_')
                actual=pair(s[BASE_ARMS.index(a)].mean(0),s[BASE_ARMS.index(b)].mean(0),scores[panel+'_query_blocks'])
                for key in ('difference','conditional_query_block_95_interval','bootstrap_nonpositive_count',
                            'statistical_direction','practical_equivalence'):
                    assert actual[key]==val[key],(panel,name,key)
        for g in range(5):
            for arm,val in original['panels'][panel]['repetitions'][g]['means'].items():
                assert float(s[BASE_ARMS.index(arm),g].mean())==val
        check[panel]['original_means_and_aggregate_intervals_exact']=True
    return check

def do_baseline(root):
    p,t,c,paths=load_original(root)
    scores,audit=score_all(p['predictions'],t['truth'],c)
    verified=verify_old(scores,read_json(paths['old_result']),load_npz(paths['old_scores']))
    record={'status':'pass','created_utc':utc(),'identity':'baseline only; no new TRADE weights read',
            'source_files':{k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()},
            'baseline_verification':verified,'independent_fullsort':audit,
            'score_code_sha256':sha(__file__),'exact_score_sha256':sha(HERE/'exact_score.py')}
    path=HERE/'BASELINE_REPLAY_LOCKED.json'
    if path.exists(): raise FileExistsError(path)
    dump(path,record)
    print(json.dumps(record,ensure_ascii=False),flush=True)

def score_authorize(a):
    release=read_json(a.release); baseline=read_json(HERE/'BASELINE_REPLAY_LOCKED.json')
    assert release['scoring_authorized'] is True
    assert release['weights_sha256']==sha(a.weights)
    assert release['weights_complete_sha256']==sha(a.weights_complete)
    assert release['score_code_sha256']==sha(__file__)
    assert release['baseline_replay_sha256']==sha(HERE/'BASELINE_REPLAY_LOCKED.json')
    assert baseline['status']=='pass' and baseline['score_code_sha256']==sha(__file__)
    assert baseline['exact_score_sha256']==sha(HERE/'exact_score.py')
    paths=source_paths(a.root)
    for key,path in paths.items():
        assert sha(path)==baseline['source_files'][key]['sha256'], ('baseline source changed',key)
    complete=read_json(a.weights_complete)
    assert complete['sha256']==sha(a.weights)
    assert complete['fits']==2945
    return {'release':release,'baseline_replay_sha256':sha(HERE/'BASELINE_REPLAY_LOCKED.json'),
            'weights_complete':complete,'source_files':baseline['source_files']}

def write_csv(path,rows):
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def do_score(a):
    # The new weights and truth arrays are not opened until the release and
    # complete-file hashes have passed; a partial fit file cannot be scored.
    auth=score_authorize(a)
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(a.output)
    p,t,c,paths=load_original(a.root)
    trade=load_npz(a.weights)
    assert np.array_equal(trade['target_ids'],c['target_ids'])
    control=load_npz(paths['control'],['gene_ids_keep'])
    assert np.array_equal(trade['gene_ids'],control['gene_ids_keep'])
    for key in ('weights','nu','TI','latent_mean','S'):
        assert trade[key].shape==(589,5) and np.isfinite(trade[key]).all(),key
    assert np.all(trade['nu']>0) and np.all(trade['TI']>=0) and np.all(trade['S']>=0)
    assert np.all((trade['weights']>=0)&(trade['weights']<=1))
    assert np.allclose(trade['S'],trade['TI']+trade['latent_mean']**2,rtol=1e-12,atol=1e-14)
    assert np.allclose(trade['weights'],trade['S']/(trade['S']+trade['nu']),rtol=1e-12,atol=1e-14)
    arms=BASE_ARMS+[TRADE]
    weights=np.concatenate([p['weights'],trade['weights'].T[None]],axis=0)
    y=np.concatenate([p['predictions'],(p['Y']*trade['weights'].T[:,None,:])[None]],axis=0)
    scores,audit=score_all(y,t['truth'],c)
    original=read_json(paths['old_result'])
    verified=verify_old(scores,original,load_npz(paths['old_scores']))
    panels={panel:summaries(scores[panel+'_scores'],arms,c[bk]) for panel,_,bk in PANELS}
    result={'created_utc':utc(),'donor':'D4_Rest','identity':'post-hoc descriptive fixed TRADE-scalar adaptation comparison',
            'primary_descriptive_contrast':'responsive PertRank_minus_TRADE_scalar',
            'arms':arms,'candidates':589,'input_groups':5,'cells_per_target_per_group':20,
            'group_interpretation':'five realized input cell groups within one donor, not five independent donors',
            'selected_competitor_unchanged':'PR_M','original_H1_H2_decision_unchanged':original['decision'],
            'panels':panels,'baseline_verification':verified,'independent_fullsort':audit,'authorization':auth,
            'bootstrap':{'draws':DRAWS,'seed':SEED,'blocks':30,'averaging':'input groups within query/sign, then signs within query; paired query-block resampling',
                         'scope':'conditional on the five realized D4 input groups, fixed controls, planning panels, 589 candidates and deep reference; not a donor-population interval',
                         'labels':'statistical direction and practical equivalence within +/-0.005 are separate, possibly overlapping labels',
                         'multiplicity':'nominal descriptive intervals, no new prospective confirmation or multiplicity-adjusted claim'},
            'method_scope':'TRADE-derived scalar weights applied to the same original log effects; not a full native-TRADE comparison',
            'weights':{arm:{'mean_by_group':weights[i].mean(1).tolist(),'zero_fraction_by_group':(weights[i]==0).mean(1).tolist()} for i,arm in enumerate(arms)},
            'scoring_code_sha256':sha(__file__),'exact_score_sha256':sha(HERE/'exact_score.py'),
            'new_weights_sha256':sha(a.weights)}
    a.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output/'query_scores.npz',arm_names=np.array(arms),**scores)
    compact={k:c[k] for k in ('target_ids','query_gene_ids','responsive_positions','tercile_positions','blocks_responsive','blocks_terciles')}
    np.savez_compressed(a.output/'scoring_inputs.npz',arm_names=np.array(arms),raw_Y=p['Y'],truth=t['truth'],weights=weights,**compact)
    dump(a.output/'D4_TRADE_RESULT.json',result)
    means,contrasts=[],[]
    for panel,report in panels.items():
        for rep,r in [('average',report['repetition_average'])]+[(str(g),v) for g,v in enumerate(report['repetitions'])]:
            means += [{'panel':panel,'input_group':rep,'arm':arm,'P_at_25':val,'reference_members_per_25':K*val} for arm,val in r['means'].items()]
            for name,v in r.items():
                if '_minus_' not in name: continue
                lo,hi=v['conditional_query_block_95_interval']
                contrasts.append({'panel':panel,'input_group':rep,'contrast':name,'difference':v['difference'],'lower95':lo,'upper95':hi,
                                  'statistical_direction':v['statistical_direction'],'practical_equivalence':v['practical_equivalence']})
    write_csv(a.output/'ALL_ARM_MEANS.csv',means);write_csv(a.output/'ALL_CONTRASTS.csv',contrasts)
    lines=['# D4 fixed TRADE-scalar supplementary comparison','',
           'Post-hoc descriptive comparison. The original D4 H1/H2 decisions and selected PR_M competitor remain unchanged.',
           'Five input groups come from one donor; they are not five donors. TRADE_scalar is an adaptation that weights the original log-effect output, not the full native TRADE workflow.','',
           '| Panel | Arm | P@25 | Reference members per 25 |','|---|---|---:|---:|']
    for panel,report in panels.items():
        for arm,val in report['repetition_average']['means'].items(): lines.append(f'| {panel} | {arm} | {val:.8f} | {K*val:.5f} |')
    lines += ['','| Panel | PertRank minus TRADE_scalar [conditional 95% interval] | Statistical direction | Practical equivalence ±0.005 |',
              '|---|---|---|---|']
    for panel,report in panels.items():
        z=report['repetition_average']['PertRank_minus_TRADE_scalar'];lo,hi=z['conditional_query_block_95_interval']
        lines.append(f'| {panel} | {z["difference"]:+.8f} [{lo:+.8f}, {hi:+.8f}] | {z["statistical_direction"]} | {z["practical_equivalence"]} |')
    lines += ['', 'Exact independent boundary-tie expectation, own-target exclusion, two signs, original 300-query panels, 30 query blocks, 2,000 paired bootstrap draws, seed 2026.',
              'Groups are averaged within each query/sign before signs are averaged within query and query blocks are resampled. Intervals condition on inputs, controls, panel, candidates and deep reference. Labels can overlap; statistical superiority does not by itself establish a practically large difference.',
              'All old query scores and aggregate reported intervals reproduce exactly. Every new and old query score is cross-checked by an independent full-sort scorer (tolerance 1e-12).',
              'ALL_ARM_MEANS.csv and ALL_CONTRASTS.csv retain each of the five input groups and every method pair. D4_TRADE_RESULT.json records original source paths/hashes, weight provenance and unchanged original decisions.',
              'Compact replay: python score_trade_supplement.py replay --replay-dir RESULTS_DIRECTORY','']
    (a.output/'REPORT.md').write_text('\n'.join(lines))
    dump(a.output/'SCORING_COMPLETE.json',{'status':'complete','created_utc':utc(),'fits':2945,'new_weights_sha256':sha(a.weights),
        'scoring_code_sha256':sha(__file__),'file_sha256':{f.name:sha(f) for f in a.output.iterdir() if f.is_file()}})
    print(json.dumps({'status':'complete','primary':panels['responsive']['repetition_average']['PertRank_minus_TRADE_scalar'],'output':str(a.output)},ensure_ascii=False),flush=True)

def do_replay(path):
    complete=read_json(path/'SCORING_COMPLETE.json')
    for name,digest in complete['file_sha256'].items(): assert sha(path/name)==digest,name
    assert complete['scoring_code_sha256']==sha(__file__)
    result=read_json(path/'D4_TRADE_RESULT.json')
    assert result['exact_score_sha256']==sha(HERE/'exact_score.py')
    x=load_npz(path/'scoring_inputs.npz');saved=load_npz(path/'query_scores.npz')
    y=x['raw_Y'][None]*x['weights'][:,:,None,:]
    scores,audit=score_all(y,x['truth'],x)
    for k,v in scores.items(): assert np.array_equal(v,saved[k]),k
    for panel,_,bk in PANELS:
        report=summaries(scores[panel+'_scores'],list(x['arm_names']),x[bk])
        assert report==result['panels'][panel],panel
    print(json.dumps({'status':'pass','compact_replay':'all query scores and summary records exact','fullsort':audit}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['baseline','score','replay'])
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--weights',type=Path);p.add_argument('--weights-complete',type=Path)
    p.add_argument('--release',type=Path);p.add_argument('--output',type=Path,default=HERE/'results')
    p.add_argument('--replay-dir',type=Path)
    a=p.parse_args()
    if a.mode=='baseline':do_baseline(a.root.resolve())
    elif a.mode=='score':
        for key in ('weights','weights_complete','release'): assert getattr(a,key) is not None,key
        do_score(a)
    else:
        assert a.replay_dir is not None
        do_replay(a.replay_dir)

if __name__=='__main__':main()
