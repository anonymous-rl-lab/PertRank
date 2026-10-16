"""Frozen common-coordinate scorer infrastructure; no new methods are fit here."""
from pathlib import Path
import hashlib, importlib.util, itertools, json, os, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R20 = ROOT/'v16_execution/assets/r20/PertRank_R20_20261006/new_experiment'
DIAG = ROOT/'v16_execution/competition/diagonal'
OLD_TRADE = ROOT/'v16_execution/trade/lean_evaluation'
PROTOCOL = ROOT/'v17_experiments/protocol/EXECUTION_PROTOCOL_v17_COMPETITION.md'
DONORS = ('D3_Rest', 'D1_Rest', 'D2_Rest')
BASE_ARMS = ('raw', 'Euclidean', 'PertRank', 'diagonal')
NEW_ARMS = {'trade': ('TRADE_scalar',), 'pearson': ('PR_E', 'PR_M')}
PANELS = (('responsive', 'responsive_positions', 'blocks_responsive'),
          ('terciles', 'tercile_positions', 'blocks_terciles'))
SEED, DRAWS, K, MARGIN = 2026, 2000, 25, .005

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def rel(path): return str(Path(path).resolve().relative_to(ROOT))
def utc(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def load_original():
    path = R20/'evaluation/evaluate_real_cells.py'
    spec = importlib.util.spec_from_file_location('original_r20', path)
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    assert ev.BOOTSTRAPS == DRAWS
    return ev

def verify_code():
    p = HERE/'SCORING_CODE_FREEZE.json'; f = json.loads(p.read_text())
    for name, digest in f['files'].items():
        assert sha(ROOT/name) == digest, 'Changed frozen file: '+name
    return sha(p)

def context_path(donor): return R20/'selection'/donor/'context.npz'
def base_path(donor): return DIAG/'predictions'/(donor+'_predictions.npz')
def old_trade_path(donor): return OLD_TRADE/'predictions'/(donor+'_predictions.npz')

def base_weights(donor):
    with np.load(base_path(donor), allow_pickle=False) as p:
        return np.stack([np.ones_like(p['w0']).T, p['w0'].T, p['wM'].T, p['wD'].T])

def all_contrasts(arms):
    # Every pair is reported once. PertRank has the positive-side orientation.
    priority = ['raw', 'Euclidean', 'diagonal', 'TRADE_scalar', 'PR_E', 'PR_M', 'PertRank']
    ordered = [a for a in priority if a in arms]
    return [(high+'_minus_'+low, arms.index(high), arms.index(low))
            for low, high in itertools.combinations(ordered, 2)]

def dual_labels(z):
    lo, hi = z['conditional_query_block_95_interval']
    z.update(statistical_direction='superiority' if lo > 0 else ('inferiority' if hi < 0 else 'unresolved'),
             practical_equivalence='established_within_0.005' if lo >= -MARGIN and hi <= MARGIN else 'not_established',
             practical_equivalence_margin=MARGIN,
             extra_reference_members_per_25=K*z['difference'])
    return z

def summarize(scores, arms, blocks, ev):
    assert scores.shape[:2] == (len(arms), 5)
    pairs = all_contrasts(list(arms)); out = {'repetitions': []}
    for group in range(5):
        ss = scores[:, group]
        row = {'repetition': group, 'means': {a: float(ss[i].mean()) for i,a in enumerate(arms)}}
        for name, i, j in pairs:
            row[name] = dual_labels(ev.query_bootstrap(ss[i], ss[j], blocks, seed=SEED))
        out['repetitions'].append(row)
    ave = scores.mean(1)
    row = {'means': {a: float(ave[i].mean()) for i,a in enumerate(arms)}}
    for name, i, j in pairs:
        z = dual_labels(ev.query_bootstrap(ave[i], ave[j], blocks, seed=SEED))
        diff = (scores[i]-scores[j]).mean((1,2))
        z.update(real_group_differences=diff.tolist(), real_group_min_max=[float(diff.min()),float(diff.max())],
                 real_groups_positive=int((diff > 0).sum()), real_groups_negative=int((diff < 0).sum()),
                 real_groups_zero=int((diff == 0).sum()))
        row[name] = z
    out['repetition_average'] = row
    return out

def check_equal(name, x, y):
    assert np.array_equal(x, y), (name, np.shape(x), np.shape(y),
                                float(np.max(np.abs(x-y))) if np.shape(x)==np.shape(y) else None)

def legacy_regression(donor, ev):
    """Only old arms. Proves all archived per-query scores exactly, not just means."""
    with np.load(context_path(donor), allow_pickle=False) as c, \
         np.load(base_path(donor), allow_pickle=False) as p, \
         np.load(old_trade_path(donor), allow_pickle=False) as tp, \
         np.load(R20/'results'/(donor+'_query_scores.npz'), allow_pickle=False) as rscore, \
         np.load(DIAG/'results'/(donor+'_query_scores.npz'), allow_pickle=False) as dscore, \
         np.load(OLD_TRADE/'results'/(donor+'_query_scores.npz'), allow_pickle=False) as tscore:
        assert tuple(p['arm_names']) == BASE_ARMS
        check_equal('base targets', p['target_ids'], c['internal_ids'])
        check_equal('base queries', p['query_ids'], c['query_gene_ids'])
        check_equal('old TRADE targets', tp['target_ids'], c['internal_ids'])
        pred = p['predictions']; weights = base_weights(donor)
        for i in range(4):
            check_equal('base scalar rebuild '+BASE_ARMS[i], pred[i], pred[0]*weights[i,:,None,:])
        out, arrays = {}, {}
        for panel, pk, bk in PANELS:
            pos = c[pk].astype(int); blocks = c[bk]
            assert len(np.unique(blocks)) == 30
            own = ev.own_mask(c['query_gene_ids'][pos], c['internal_ids'])
            scores = np.array([[ev.score(pred[i,g,pos],c['truth'][pos],own,K) for g in range(5)] for i in range(4)])
            check_equal(panel+' R20 all groups', scores[:3], rscore['internal_'+panel+'_scores'])
            check_equal(panel+' diagonal all groups', scores, dscore[panel+'_scores'])
            t0 = ev.score(tp['predictions'][4,pos],c['truth'][pos],own,K)
            check_equal(panel+' TRADE block0', t0, tscore[panel+'_scores'][4])
            check_equal(panel+' block0 baseline', scores[:,0], tscore[panel+'_scores'][:4])
            arrays[panel+'_scores'] = scores
            arrays[panel+'_trade_block0'] = t0
            out[panel] = {'R20_three_arms_all_five_groups_exact': True, 'diagonal_all_five_groups_exact': True,
                          'TRADE_block0_exact': True, 'maximum_absolute_score_error': 0.0,
                          'queries': len(pos), 'targets': len(c['internal_ids']), 'own_exclusions': int(own.sum())}
        return out, arrays

def verify_family_lock(family):
    path = HERE/'prepared'/family/'ALL_DONORS_LOCK.json'
    lock = json.loads(path.read_text()); assert set(lock['donors']) == set(DONORS)
    assert lock['scoring_code_freeze_sha256'] == verify_code()
    assert lock['protocol_sha256'] == sha(PROTOCOL)
    assert sha(ROOT/lock['upstream_lock_path']) == lock['upstream_lock_sha256']
    for donor, rec in lock['donors'].items():
        for pk,hk in [('weights_path','weights_sha256'), ('context_path','context_sha256'),
                      ('base_prediction_path','base_prediction_sha256'),('source_weights_path','source_weights_sha256')]:
            assert sha(ROOT/rec[pk]) == rec[hk], (donor,pk)
    return lock
