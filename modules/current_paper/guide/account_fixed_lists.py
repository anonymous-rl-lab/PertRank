#!/usr/bin/env python3
"""Exact attribution of changes between two previously locked candidate lists.

This script never estimates weights or forms new prediction rankings. It uses
the existing fractional membership arrays and constructs only fixed reference
memberships. All candidates and query blocks are retained.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_membership(effect, own, k=10):
    """Independent exact-tie implementation based on strictly larger ranks."""
    out = np.zeros((2,) + effect.shape)
    for d, sign in enumerate((1., -1.)):
        for q in range(effect.shape[0]):
            eligible = np.flatnonzero(~own[q])
            values = sign * effect[q, eligible]
            greater = (values[:, None] < values[None, :]).sum(1)
            equal = (values[:, None] == values[None, :]).sum(1)
            out[d, q, eligible] = np.clip((k - greater) / equal, 0., 1.)
    assert np.allclose(out.sum(-1), k, rtol=0, atol=1e-12)
    return out


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def concentration(values, labels, sign):
    mass = np.maximum(sign * values, 0.)
    # Numerical zero only: block sums of cancelling rational tie shares can
    # leave ~1e-19 residue. This never changes scores or any candidate list.
    mass[np.abs(values) <= 1e-14] = 0.
    order = np.argsort(-mass, kind='stable')
    order = order[mass[order] > 0]
    total = float(mass.sum())
    top = order[:5]
    return {
        'counting_zero_tolerance': 1e-14,
        'positive_mass' if sign > 0 else 'absolute_negative_mass': total,
        'nonzero_count': int(len(order)),
        'top5_mass': float(mass[top].sum()),
        'top5_fraction_of_same_sign_mass': float(mass[top].sum() / total) if total else None,
        'top5': [{'label': str(labels[i]), 'signed_P10_contribution': float(values[i])}
                 for i in top],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe-root', type=Path, required=True)
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--protocol-lock', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    lock = json.loads(args.protocol_lock.read_text())
    assert lock['protocol_sha256'] == sha(args.protocol)
    assert not args.out.exists(), 'Use a new results directory.'
    args.out.mkdir(parents=True)
    old = json.loads((args.probe_root / 'observed/RESULTS.json').read_text())
    old_lock = json.loads((args.probe_root / 'observed/PREDICTION_LOCK.json').read_text())
    external = load(args.prior / 'crossfit_pilot/evaluation_inputs/reference.npz')
    qi = {str(v): i for i, v in enumerate(external['query_ids'])}
    ti = {str(v): i for i, v in enumerate(external['target_symbols'])}
    targets, blocks, queries = [], [], []
    summary = {
        'analysis_identity': 'Post hoc exact accounting of previously locked predictions; no new model, ranking or performance selection',
        'protocol_sha256': sha(args.protocol),
        'script_sha256': sha(__file__),
        'formula': '(new_membership-old_membership)*reference_membership/(2*Q*10)',
        'uncertainty': 'Arithmetic accounting only. No confidence interval or new significance test.',
        'sources': {},
        'checks': {},
    }
    for source in ('D3', 'D1', 'D2'):
        pp = args.predictions / f'{source}.npz'
        p = load(pp)
        md = load(args.probe_root / 'metadata' / f'{source}_guide_metadata.npz')
        x = load(args.prior / 'crossfit_pilot/inputs' / f'{source}_shallow_cells.npz')
        assert np.array_equal(p['target_ids'], md['target_ids'])
        assert np.array_equal(p['query_ids'], x['query_ids'])
        assert np.array_equal(p['target_ids'], x['target_ids'])
        assert np.array_equal(p['query_blocks'], x['query_blocks'])
        arm = {str(v): i for i, v in enumerate(p['arm_names'])}
        old_i, new_i = arm['original_wM'], arm['crossguide_wM']
        before, after = p['memberships'][old_i], p['memberships'][new_i]
        assert np.allclose(before.sum(-1), 10, rtol=0, atol=1e-12)
        assert np.allclose(after.sum(-1), 10, rtol=0, atol=1e-12)
        delta = after - before
        Q = before.shape[1]
        denom = 2 * Q * 10
        reference_effect = external['mean_effect'][np.ix_(
            [qi[str(v)] for v in x['query_ids']],
            [ti[str(v)] for v in x['target_symbols']])]
        ref_external = reference_membership(reference_effect, x['own_mask'])
        deep = load(args.prior / 'inputs' / f'{source}_same_screen_reference.npz')
        ref_deep = deep['reference_membership']
        assert np.max(np.abs(ref_deep - reference_membership(deep['deep_effect'], deep['own_mask']))) < 1e-12
        assert np.array_equal(deep['target_ids'], p['target_ids'])
        guide_counts = [np.unique(row, return_counts=True)[1] for row in md['guide_ids']]
        fallback = np.array([len(v) == 1 for v in guide_counts])
        nmin = np.array([int(v.min()) if len(v) > 1 else -1 for v in guide_counts])
        old_zero = p['weights'][old_i] == 0
        new_zero = p['weights'][new_i] == 0
        zero_group = np.where(old_zero, 'originally_zero',
                             np.where(new_zero, 'newly_zero', 'positive_after'))
        coverage = np.where(fallback, 'single_guide_fallback',
                            np.where(nmin == 1, 'n_min_1',
                                     np.where(nmin <= 3, 'n_min_2_3', 'n_min_ge4')))
        summary['sources'][source] = {}
        source_error = 0.
        for rname, ref in [('external_RNA', ref_external), ('same_screen_cells', ref_deep)]:
            con = delta * ref / denom
            added = np.maximum(delta, 0.) * ref / denom
            lost = np.maximum(-delta, 0.) * ref / denom
            per_target = con.sum((0, 1))
            target_added, target_lost = added.sum((0, 1)), lost.sum((0, 1))
            per_query = con.sum((0, 2)) * Q
            old_mean = float((before * ref).sum() / denom)
            new_mean = float((after * ref).sum() / denom)
            net = float(con.sum())
            expected = old['sources'][source][rname]['contrasts']['cross_M_minus_original_M']['mean']
            source_error = max(source_error, abs(net - expected))
            assert abs(net - expected) < 1e-12
            assert abs(net - (new_mean - old_mean)) < 1e-12
            assert abs(net - (added.sum() - lost.sum())) < 1e-12
            assert abs(per_query.mean() - net) < 1e-12
            groups = {}
            for group_name, values in [('zero_status', zero_group), ('input_coverage', coverage),
                                       ('locked_cross_M_zero', np.where(new_zero, 'zero', 'nonzero'))]:
                groups[group_name] = {}
                for label in sorted(set(values)):
                    sel = values == label
                    groups[group_name][str(label)] = {
                        'targets': int(sel.sum()), 'net_P10': float(per_target[sel].sum()),
                        'added_supported_P10': float(target_added[sel].sum()),
                        'lost_supported_P10': float(target_lost[sel].sum()),
                        'net_reference_members_per10': float(10 * per_target[sel].sum()),
                    }
                assert abs(sum(v['net_P10'] for v in groups[group_name].values()) - net) < 1e-12
            block_labels = np.unique(p['query_blocks'])
            block_values = np.array([con[:, p['query_blocks'] == b, :].sum() for b in block_labels])
            summary['sources'][source][rname] = {
                'original_wM_P10': old_mean, 'crossguide_wM_P10': new_mean,
                'net_P10': net, 'net_reference_members_per10': 10 * net,
                'added_supported_P10': float(added.sum()), 'lost_supported_P10': float(lost.sum()),
                'added_reference_members_per10': float(10 * added.sum()),
                'lost_reference_members_per10': float(10 * lost.sum()),
                'average_replaced_positions_per10': float(np.maximum(delta, 0.).sum() / (2 * Q)),
                'group_totals': groups,
                'target_positive_concentration': concentration(per_target, md['target_symbols'], 1),
                'target_negative_concentration': concentration(per_target, md['target_symbols'], -1),
                'block_positive_concentration': concentration(block_values, block_labels, 1),
                'block_negative_concentration': concentration(block_values, block_labels, -1),
            }
            for j, target in enumerate(md['target_ids']):
                targets.append({
                    'source': source, 'reference': rname, 'target_id': str(target),
                    'target_symbol': str(md['target_symbols'][j]),
                    'input_guide_count': int(len(guide_counts[j])),
                    'minor_guide_cell_count': int(nmin[j]) if nmin[j] >= 0 else '',
                    'input_coverage': str(coverage[j]), 'zero_status': str(zero_group[j]),
                    'locked_original_M_weight': float(p['weights'][old_i, j]),
                    'locked_cross_M_weight': float(p['weights'][new_i, j]),
                    'net_P10_contribution': float(per_target[j]),
                    'added_supported_P10_contribution': float(target_added[j]),
                    'lost_supported_P10_contribution': float(target_lost[j]),
                    'net_reference_members_per10_contribution': float(10 * per_target[j]),
                })
            for b, bv in zip(block_labels, block_values):
                sel = p['query_blocks'] == b
                blocks.append({
                    'source': source, 'reference': rname, 'query_block': str(b),
                    'queries': int(sel.sum()), 'net_P10_contribution': float(bv),
                    'within_block_mean_P10_difference': float(per_query[sel].mean()),
                    'added_supported_P10_contribution': float(added[:, sel, :].sum()),
                    'lost_supported_P10_contribution': float(lost[:, sel, :].sum()),
                })
            for q in range(Q):
                queries.append({
                    'source': source, 'reference': rname, 'query_id': str(p['query_ids'][q]),
                    'query_block': str(p['query_blocks'][q]), 'P10_difference': float(per_query[q]),
                    'P10_contribution_to_source_mean': float(per_query[q] / Q),
                    'added_supported_P10': float(added[:, q, :].sum() * Q),
                    'lost_supported_P10': float(lost[:, q, :].sum() * Q),
                })
        summary['checks'][source] = {
            'prediction_sha256': sha(pp),
            'recorded_prediction_bytes_match': sha(pp) == old_lock['prediction_sha256'][source],
            'max_net_difference_from_archived_results': float(source_error),
            'all_candidate_rows_retained': int(len(md['target_ids'])),
            'all_query_blocks_retained': int(len(np.unique(p['query_blocks']))),
        }
        assert summary['checks'][source]['recorded_prediction_bytes_match'], 'Use exact locked predictions.'
    write_csv(args.out / 'target_accounting.csv', targets)
    write_csv(args.out / 'block_accounting.csv', blocks)
    write_csv(args.out / 'query_accounting.csv', queries)
    summary['equal_source_descriptive'] = {
        ref: {key: float(np.mean([summary['sources'][s][ref][key] for s in ('D3', 'D1', 'D2')]))
              for key in ('net_P10', 'net_reference_members_per10',
                          'added_reference_members_per10', 'lost_reference_members_per10')}
        for ref in ('external_RNA', 'same_screen_cells')}
    (args.out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS', 'checks': summary['checks'],
                      'equal_source_descriptive': summary['equal_source_descriptive']}, indent=2))


if __name__ == '__main__':
    main()
