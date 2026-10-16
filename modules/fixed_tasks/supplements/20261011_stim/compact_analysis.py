"""Result orchestration adapted solely to consume exact reconstructed memberships.
Numerical decision/bootstrap/regression implementations are unchanged.
"""
import numpy as np
import stim_stats as stats
import stim_secondary as secondary
from stats_frozen_summary import all_contrasts

def full_arm_table(summary, unavailable):
    """Keep every registered arm/pair visible, including unavailable contrasts."""
    available = summary['repetition_average']
    means = {arm: available['means'][arm] if arm in available['means'] else
             {'status': 'not_evaluable', 'reason': unavailable[arm]} for arm in stats.ARMS}
    contrasts = {}
    for name, i, j in all_contrasts(stats.ARMS):
        if name in available:
            contrasts[name] = available[name]
        else:
            affected = [arm for arm in (stats.ARMS[i], stats.ARMS[j]) if arm in unavailable]
            if not affected:
                raise ValueError('Missing contrast without declared unavailable arm')
            contrasts[name] = {'status': 'not_evaluable', 'formal_superiority': False,
                               'unavailable_arms': {arm: unavailable[arm] for arm in affected}}
    return {'means': means, 'all_ten_pairwise_contrasts': contrasts}

def analyze(main, query, unavailable, *, cross=None, alignment=None):
    """Pure numerical join, also used by synthetic verification without I/O."""
    weights = main['weights']
    scores, names = main['scores'], main['arm_names']
    rp, tp = query['responsive_positions'], query['tercile_positions']
    responsive = np.take(scores, rp, axis=2)
    terciles = np.take(scores, tp, axis=2)
    primary = stats.adjudicate_main(responsive, names, query['blocks_responsive'],
                                   weights['PertRank'], unavailable_arms=unavailable)
    tercile_summary = stats.summarize_scores(terciles, names, query['blocks_terciles'])
    ip, ie = names.index('PertRank'), names.index('Euclidean')
    difference_by_query = (scores[ip] - scores[ie]).mean(axis=(0, 2))
    panel = secondary.union_panel_bootstrap(
        difference_by_query[rp], difference_by_query[tp],
        main['query_gene_ids'][rp], main['query_gene_ids'][tp],
        main['query_gene_ids'], query['blocks_union'])
    equivalence_key = 'PertRank_minus_PR_M'
    equivalence = tercile_summary['repetition_average'].get(equivalence_key,
        {'status': 'not_evaluable', 'reason': unavailable.get('PR_M')})
    output_arrays = {'main_scores': scores, 'main_arm_names': np.asarray(names),
                     'main_query_gene_ids': main['query_gene_ids'],
                     'main_target_ids': main['target_ids']}
    report = {
        'H1_H2': primary,
        'responsive_all_arms': full_arm_table(primary['summary'], unavailable),
        'tercile_summary': tercile_summary,
        'tercile_all_arms': full_arm_table(tercile_summary, unavailable),
        'S1_panel_interaction': panel,
        'S1_tercile_PertRank_PR_M_approximate_equivalence': equivalence,
        'unavailable_arms_frozen_before_truth': unavailable,
        'zero_weight_fractions': {arm: {'overall': float(np.mean(w == 0)),
            'per_input_group': np.mean(w == 0, axis=1).tolist()} for arm, w in weights.items()},
        'main_counts': {'targets': len(main['target_ids']), 'groups': 5, 'signs': 2,
                        'responsive_queries': len(rp), 'tercile_queries': len(tp)},
        'interpretation': 'S1-S3 are nominal conditional secondary estimates and cannot rescue H1.'}
    if cross is None or cross['status'] != 'ok':
        report['S2_crossguide'] = cross or {'status': 'not_evaluable', 'reason': 'pretruth_task_unavailable'}
    else:
        xp, xq = cross['predictions'], cross['query']
        xw = xp['weights']
        xs, xn = xp['scores'], xp['arm_names']
        pos = xq['responsive_positions']
        report['S2_crossguide'] = {'status': 'ok', 'targets': len(xp['target_ids']),
            'summary': stats.summarize_scores(np.take(xs, pos, axis=2), xn, xq['blocks_responsive']),
            'scope': 'Independent-guide task with its own pool/panel; no artifact fraction is inferred.'}
        output_arrays.update(crossguide_scores=xs, crossguide_arm_names=np.asarray(xn),
                             crossguide_query_gene_ids=xp['query_gene_ids'], crossguide_target_ids=xp['target_ids'])
    if alignment is None or alignment['status'] != 'ok':
        report['S3_alignment'] = alignment or {'status': 'not_evaluable', 'reason': 'pretruth_design_unavailable'}
    else:
        contributions = main['alignment_contributions']
        regression = secondary.alignment_regression(contributions, alignment,
            query['blocks_responsive'], target_ids=main['target_ids'], query_ids=main['query_gene_ids'][rp])
        output_arrays['alignment_contributions'] = contributions
        if 'coefficients' in regression:
            output_arrays['alignment_coefficients'] = regression.pop('coefficients')
        report['S3_alignment'] = regression
    return report, output_arrays
