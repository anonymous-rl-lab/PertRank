"""Original scientific summary/decision functions, extracted verbatim.

Only filesystem-dependent orchestration has been omitted. See SOURCE_PROVENANCE.json.
"""
import itertools
import numpy as np
SEED=2026
DRAWS=2000
K=25
MARGIN=.005

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

def adjudicate(responsive,competitor,allzero_pertrank_groups=()):
    r=responsive['repetition_average'];main=r['PertRank_minus_Euclidean']
    lo=main['conditional_query_block_95_interval'][0]
    numerical_confirmation=bool(main['difference']>=.01 and lo>0)
    valid=not len(allzero_pertrank_groups)
    confirmation=bool(valid and numerical_confirmation)
    improvement=bool(valid and main['difference']>=.02 and lo>0)
    z=r['PertRank_minus_'+competitor]
    return {'primary_endpoint':'responsive PertRank minus Euclidean, five groups averaged within query',
        'confirmation':confirmation,'improvement_grade':improvement,
        'numerical_primary_threshold_pass':numerical_confirmation,
        'method_valid_for_confirmation':valid,'allzero_pertrank_groups':list(allzero_pertrank_groups),
        'confirmation_rule':'no allzero PertRank input group AND point >=0.01 AND lower nominal two-sided 95% paired query-block percentile bound >0',
        'improvement_rule':'no allzero PertRank input group AND point >=0.02 AND the same lower95 bound >0',
        'competitive_gate_open':confirmation,'selected_competitor':competitor,
        'formal_competitive_statistical_direction':z['statistical_direction'] if confirmation else 'not_activated',
        'formal_competitive_practical_equivalence':z['practical_equivalence'] if confirmation else 'not_activated',
        'descriptive_competitive_statistical_direction':z['statistical_direction'],
        'descriptive_competitive_practical_equivalence':z['practical_equivalence'],
        'closed_gate_rule':'all contrasts still reported descriptively; neither raw comparisons, terciles nor cell equivalents rescue a failed primary gate'}
