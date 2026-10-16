#!/usr/bin/env python3
"""Score the frozen IL2 lists with the prespecified MAGeCK secondary endpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SIGN = {'increasing_expression': 1, 'decreasing_expression': -1}
DENOMINATOR = 250


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def subset_counts(block, weight_column=None):
    weights = block[weight_column] if weight_column else pd.Series(1., index=block.index)
    present = weights > 0
    observed = present & block.observed
    supported = observed & block.mageck_support
    return {
        'nomination_occurrences': float(weights.sum()),
        'observed_occurrences': float(weights[observed].sum()),
        'unknown_occurrences': float(weights[present & ~block.observed].sum()),
        'supported_occurrences': float(weights[supported].sum()),
        'unique_targets': int(block.loc[present, 'target_id'].nunique()),
        'unique_observed_targets': int(block.loc[observed, 'target_id'].nunique()),
        'unique_supported_targets': int(block.loc[supported, 'target_id'].nunique()),
        'unique_supported_target_direction_pairs': int(block.loc[supported, ['target_id', 'direction']].drop_duplicates().shape[0]),
    }


def comparison(block, denominator):
    net = float((block.membership_delta * block.mageck_support.astype(int)).sum())
    unknown = block.loc[~block.observed]
    lower = float(unknown.loc[unknown.membership_delta < 0, 'membership_delta'].sum())
    upper = float(unknown.loc[unknown.membership_delta > 0, 'membership_delta'].sum())
    return {
        'fixed_denominator': denominator,
        'observed_net_supported_occurrences': net,
        'observed_delta_support_fraction': net / denominator,
        'unknown_net_contribution_bounds_occurrences': [lower, upper],
        'complete_delta_support_fraction_bounds': [(net + lower) / denominator, (net + upper) / denominator],
        'unknown_changed_targets': sorted(unknown.loc[unknown.membership_delta != 0, 'gene_symbol'].unique().tolist()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = args.root
    out = args.out or root / 'results'
    out.mkdir(parents=True, exist_ok=True)
    plan = root / 'ANALYSIS_PLAN_v1.2.0.md'
    plan_hash = hashlib.sha256(plan.read_bytes()).hexdigest()
    assert 'pos|fdr <0.05' in plan.read_text()
    assert 'neg|lfc <-0.5' in plan.read_text()
    mapping = pd.read_csv(root / 'data/candidate_mapping.tsv', sep='\t', keep_default_na=False)
    nominations = pd.read_csv(root / 'data/il2_all_nominations.tsv.gz', sep='\t', keep_default_na=False)
    effects = pd.read_csv(root / 'results/target_external_effects.tsv', sep='\t', keep_default_na=False)
    assert len(mapping) == 561 and mapping.target_id.is_unique
    assert set(nominations['query']) == {'IL2'} and set(nominations.direction) == set(SIGN)
    assert not nominations.duplicated(['input_group', 'direction', 'target_id']).any()
    assert len(nominations.groupby(['input_group', 'direction'])) == 10
    assert set(nominations.membership_delta) <= {-1., 0., 1.}
    for _, block in nominations.groupby(['input_group', 'direction']):
        assert block.euclidean_membership.sum() == 25 and block.pertrank_membership.sum() == 25
        assert np.allclose(block.membership_delta, block.pertrank_membership - block.euclidean_membership)
    assert nominations.euclidean_membership.sum() == DENOMINATOR
    assert nominations.pertrank_membership.sum() == DENOMINATOR
    modes = [('independent', 'nonoverlap', [1, 2]), ('all_guides', 'all_guides', [1])]
    all_rows, list_rows, cards, input_hashes, audits = [], [], {}, {}, {}
    for mode, effect_mode, minimums in modes:
        summary_file = root / 'mageck' / mode / 'result.gene_summary.txt'
        manifest = json.loads((summary_file.parent / 'RUN_MANIFEST.json').read_text())
        assert manifest['analysis_plan_sha256'] == plan_hash
        scores = pd.read_csv(summary_file, sep='\t', keep_default_na=False)
        assert scores.id.is_unique
        assert not scores.id.str.upper().str.contains('NO-TARGET|NONTARGET|NON-TARGET', regex=True).any()
        scores = scores.rename(columns={'id': 'external_design_label', 'num': 'mageck_num',
            'pos|fdr': 'pos_fdr', 'neg|fdr': 'neg_fdr', 'pos|lfc': 'pos_lfc', 'neg|lfc': 'neg_lfc'})
        score_columns = ['external_design_label', 'mageck_num', 'pos_fdr', 'neg_fdr', 'pos_lfc', 'neg_lfc']
        ef = effects.loc[effects['mode'] == effect_mode, ['target_id', 'n_guides', 'D1_lfc', 'D2_lfc']].copy()
        for col in ['D1_lfc', 'D2_lfc']:
            ef[col] = pd.to_numeric(ef[col], errors='coerce')
        frame = nominations.merge(mapping[['target_id', 'external_design_label']], on='target_id', validate='many_to_one')
        frame = frame.merge(ef, on='target_id', validate='many_to_one')
        frame = frame.merge(scores[score_columns], on='external_design_label', how='left', validate='many_to_one')
        frame['score_available'] = frame[['pos_fdr', 'neg_fdr', 'pos_lfc', 'neg_lfc']].notna().all(axis=1)
        available = frame.loc[frame.score_available]
        assert np.allclose(available.mageck_num, 2 * available.n_guides), 'Paired MAGeCK count != 2 x eligible guide identities'
        assert available[['pos_fdr', 'neg_fdr']].ge(0).all().all() and available[['pos_fdr', 'neg_fdr']].le(1).all().all()
        frame['sign'] = frame.direction.map(SIGN)
        frame['D1_oriented_ntc_centered_lfc'] = frame.sign * frame.D1_lfc
        frame['D2_oriented_ntc_centered_lfc'] = frame.sign * frame.D2_lfc
        frame['donor_direction_consistent'] = (frame.D1_oriented_ntc_centered_lfc > 0) & (frame.D2_oriented_ntc_centered_lfc > 0)
        frame['donor_both_effect_at_least_0_5'] = (frame.D1_oriented_ntc_centered_lfc >= .5) & (frame.D2_oriented_ntc_centered_lfc >= .5)
        for minimum in minimums:
            analysis = f'{mode}_min{minimum}'
            rows = frame.copy()
            rows['analysis'] = analysis
            rows['observed'] = rows.score_available & (rows.n_guides >= minimum)
            rows['mageck_support'] = rows.observed & (
                ((rows.sign == 1) & (rows.pos_lfc > .5) & (rows.pos_fdr < .05)) |
                ((rows.sign == -1) & (rows.neg_lfc < -.5) & (rows.neg_fdr < .05)))
            rows['binary_net_contribution'] = rows.membership_delta * rows.mageck_support.astype(int)
            all_rows.extend(rows.to_dict('records'))
            card = comparison(rows, DENOMINATOR)
            card['observed_net_supported_per_list'] = card['observed_net_supported_occurrences'] / 10
            card['full_lists'] = {method: subset_counts(rows, col) for method, col in
                [('Euclidean', 'euclidean_membership'), ('PertRank', 'pertrank_membership')]}
            card['changes'] = {kind: subset_counts(block) for kind, block in rows.groupby('change')}
            changed = rows.loc[rows.membership_delta != 0]
            card['unique_changed_targets_total'] = int(changed.target_id.nunique())
            card['unique_changed_targets_observed'] = int(changed.loc[changed.observed, 'target_id'].nunique())
            card['unknown_full_list_targets'] = sorted(rows.loc[~rows.observed, 'gene_symbol'].unique().tolist())
            card['directions'] = {direction: comparison(block, 125) for direction, block in rows.groupby('direction')}
            supported_unique = rows.loc[rows.mageck_support, ['target_id', 'gene_symbol', 'direction', 'donor_direction_consistent', 'donor_both_effect_at_least_0_5']].drop_duplicates()
            card['supported_target_directions'] = supported_unique.to_dict('records')
            card['donor_consistency_is_filter'] = False
            for (group, direction), block in rows.groupby(['input_group', 'direction'], sort=True):
                comp = comparison(block, 25)
                for method, col in [('Euclidean', 'euclidean_membership'), ('PertRank', 'pertrank_membership')]:
                    counts = subset_counts(block, col)
                    list_rows.append(dict(analysis=analysis, input_group=int(group), direction=direction, method=method,
                        **counts, supported_fraction_observed=counts['supported_occurrences'] / 25,
                        delta_supported_occurrences=comp['observed_net_supported_occurrences'],
                        delta_support_fraction_observed=comp['observed_delta_support_fraction'],
                        delta_support_fraction_lower=comp['complete_delta_support_fraction_bounds'][0],
                        delta_support_fraction_upper=comp['complete_delta_support_fraction_bounds'][1]))
            assert card['changes']['swap_in']['supported_occurrences'] - card['changes']['swap_out']['supported_occurrences'] == card['observed_net_supported_occurrences']
            assert card['full_lists']['PertRank']['supported_occurrences'] - card['full_lists']['Euclidean']['supported_occurrences'] == card['observed_net_supported_occurrences']
            cards[analysis] = card
        audits[mode] = {'gene_summary_rows': len(scores), 'ntc_gene_summary_rows': 0,
            'score_available_unique_nominated_targets': int(available.target_id.nunique()),
            'paired_num_equals_two_times_guide_count': True,
            'manifest_plan_hash_matches': True}
        input_hashes[f'mageck/{mode}/result.gene_summary.txt'] = hashlib.sha256(summary_file.read_bytes()).hexdigest()
    evidence = pd.DataFrame(all_rows)
    evidence.to_csv(out / 'mageck_nomination_evidence.tsv', sep='\t', index=False)
    pd.DataFrame(list_rows).to_csv(out / 'mageck_list_support.tsv', sep='\t', index=False)
    for name in ['ANALYSIS_PLAN_v1.2.0.md', 'data/candidate_mapping.tsv', 'data/il2_all_nominations.tsv.gz', 'results/target_external_effects.tsv']:
        input_hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    result = {'version': '1.2.0', 'endpoint': 'prespecified_secondary_MAGeCK_direction_matched_support',
        'fixed_denominator': DENOMINATOR, 'number_of_lists': 10, 'external_biological_donors': 2,
        'positive_effect_interpretation': 'CRISPRi increases IL2 protein; high/low is not sign reversed',
        'thresholds': {'positive_lfc_strictly_greater': .5, 'negative_lfc_strictly_less': -.5, 'directional_fdr_strictly_less': .05},
        'scores_are_bootstrapped': False, 'missing_is_observed_failure': False,
        'bounds_interpretation': 'Binary missing-contribution bounds with fixed 250 denominator, not confidence intervals.',
        'scorecards': cards, 'audits': audits, 'inputs_sha256': input_hashes}
    (out / 'MAGECK_SUPPORT_RESULTS.json').write_text(json.dumps(clean(result), indent=2) + '\n')
    print(json.dumps(clean({key: {k:v for k,v in card.items() if k in ['observed_net_supported_occurrences', 'observed_net_supported_per_list', 'observed_delta_support_fraction', 'complete_delta_support_fraction_bounds', 'unknown_changed_targets', 'changes', 'full_lists']} for key,card in cards.items()}), indent=2))


if __name__ == '__main__':
    main()
