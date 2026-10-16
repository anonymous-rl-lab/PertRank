#!/usr/bin/env python3
"""Construct exact fixed-method top-25 sufficient inputs from full frozen states.

This retrospective compactor never changes scientific inputs, weights or results.
It preserves the original float64 raw effects on the union of every frozen arm's
signed top-25 candidates, including every boundary tie. Both deep-reference tails
are also preserved. The original full states are needed to run this builder;
the public replay needs only its much smaller output.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np

K = 25
PANEL_KEYS = ('responsive_positions', 'tercile_positions',
              'blocks_responsive', 'blocks_terciles')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def content_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def membership(values):
    """Both signed tails, independent tie expectation, in supplied ID order."""
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and len(values) >= K and np.isfinite(values).all()
    out = []
    for sign in (1., -1.):
        signed = sign * values
        threshold = np.partition(signed, len(signed) - K)[-K]
        above, tied = signed > threshold, signed == threshold
        p = above.astype(np.float64) + tied * ((K - above.sum()) / tied.sum())
        out.append(p)
    return np.asarray(out)


def full_membership(values, own):
    valid = np.arange(len(values)) != own
    p = np.zeros((2, len(values)), np.float64)
    p[:, valid] = membership(values[valid])
    return p


def full_sort_membership(values, own):
    """Independent full-sort implementation for compaction validation."""
    ids = np.flatnonzero(np.arange(len(values)) != own)
    p = np.zeros((2, len(values)), np.float64)
    for si, sign in enumerate((1., -1.)):
        signed = sign * values[ids]
        boundary = np.sort(signed)[-K]
        above, tied = signed > boundary, signed == boundary
        p[si, ids[above]] = 1.
        p[si, ids[tied]] = (K - np.count_nonzero(above)) / np.count_nonzero(tied)
    return p


def sources(root, donor):
    if donor == 'D4':
        base = root / 'PertRank_D4_OSF_core_under60MB_20261008/result/data'
        core, context, truth = [load(base / n) for n in
                                ('prediction_core.npz', 'query_context.npz', 'truth.npz')]
        out = {k: context[k] for k in PANEL_KEYS}
        out.update(original_log_effects=core['Y'], weights=core['weights'],
                   truth=truth['truth'], arm_names=core['arm_names'],
                   query_ids=core['query_gene_ids'], target_ids=core['target_ids'])
        return out, [base / n for n in
                     ('prediction_core.npz', 'query_context.npz', 'truth.npz')]
    path = root / ('v18_d4_confirmation/d1_correction/corrected_panel_predictions_locked.npz'
                   if donor == 'D1' else
                   f'v17_experiments/delivery/compact/{donor}_Rest_state.npz')
    return load(path), [path]


def build(root, destination, donor):
    start = time.monotonic()
    s, source_files = sources(root, donor)
    y, weights, truth = [s[k] for k in ('original_log_effects', 'weights', 'truth')]
    nq, nt = truth.shape
    assert y.shape == (5, nq, nt)
    assert weights.shape == (len(s['arm_names']), 5, nt)
    assert y.dtype == weights.dtype == truth.dtype == np.float64
    assert nt < 65536
    assert np.isfinite(y).all() and np.isfinite(weights).all() and np.isfinite(truth).all()
    targets = s['target_ids'].astype(str)
    queries = s['query_ids'].astype(str)
    assert len(set(targets)) == nt and len(set(queries)) == nq
    lookup = {g: j for j, g in enumerate(targets)}
    own = np.asarray([lookup.get(g, -1) for g in queries], np.int32)
    full_truth_p = np.asarray([full_membership(truth[q], own[q]) for q in range(nq)])
    truth_ids, truth_values, truth_offsets = [], [], [0]
    truth_membership_comparisons = 0
    for q in range(nq):
        ids = np.flatnonzero(np.any(full_truth_p[q] > 0, axis=0))
        values = truth[q, ids]
        sparse_p = np.zeros((2, nt), np.float64)
        sparse_p[:, ids] = membership(values)
        assert np.array_equal(sparse_p, full_truth_p[q])
        assert np.array_equal(sparse_p, full_sort_membership(truth[q], own[q]))
        truth_membership_comparisons += sparse_p.size
        truth_ids.extend(ids)
        truth_values.extend(values)
        truth_offsets.append(len(truth_ids))
    Y_ids, Y_values, Y_offsets = [], [], [0]
    row_counts = []
    prediction_membership_comparisons = 0
    full_membership_hash = hashlib.sha256()
    sparse_membership_hash = hashlib.sha256()
    scores = np.zeros((len(weights), 5, nq, 2), np.float64)
    for group in range(5):
        for q in range(nq):
            predictions = y[group, q][None, :] * weights[:, group, :]
            full = np.asarray([full_membership(v, own[q]) for v in predictions])
            keep = np.any(full > 0, axis=(0, 1)) | np.any(full_truth_p[q] > 0, axis=0)
            ids = np.flatnonzero(keep)
            values = y[group, q, ids]
            assert own[q] < 0 or not keep[own[q]]
            sparse_predictions = values[None, :] * weights[:, group, ids]
            assert sparse_predictions.shape == (len(weights), len(ids))
            for arm in range(len(weights)):
                sparse = np.zeros((2, nt), np.float64)
                sparse[:, ids] = membership(sparse_predictions[arm])
                independent = full_sort_membership(predictions[arm], own[q])
                assert np.array_equal(sparse, full[arm]), (donor, group, q, arm)
                assert np.array_equal(sparse, independent), (donor, 'independent', group, q, arm)
                scores[arm, group, q] = (sparse * full_truth_p[q]).sum(1) / K
                assert np.array_equal(scores[arm, group, q],
                                      (full[arm] * full_truth_p[q]).sum(1) / K)
                prediction_membership_comparisons += sparse.size
                full_membership_hash.update(full[arm].tobytes())
                sparse_membership_hash.update(sparse.tobytes())
            Y_ids.extend(ids)
            Y_values.extend(values)
            Y_offsets.append(len(Y_ids))
            row_counts.append(len(ids))
    arrays = dict(
        Y_offsets=np.asarray(Y_offsets, np.uint32), Y_ids=np.asarray(Y_ids, np.uint16),
        Y_values=np.asarray(Y_values, np.float64),
        truth_offsets=np.asarray(truth_offsets, np.uint32),
        truth_ids=np.asarray(truth_ids, np.uint16), truth_values=np.asarray(truth_values, np.float64),
        weights=weights, arm_names=s['arm_names'], query_ids=s['query_ids'], target_ids=s['target_ids'],
        own_index=own, n_candidates=np.asarray(nt, np.int32), n_queries=np.asarray(nq, np.int32),
        n_groups=np.asarray(5, np.int32), k=np.asarray(K, np.int32), donor=np.asarray(donor),
        schema_version=np.asarray(1, np.int32), **{k: s[k] for k in PANEL_KEYS})
    dest = destination / 'data' / donor
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'ranking.npz'
    np.savez_compressed(path, **arrays)
    reference_path = (root / 'v18_d4_confirmation/d1_correction/D1_corrected_query_scores.npz'
                      if donor == 'D1' else
                      root / 'PertRank_D4_OSF_core_under60MB_20261008/result/v18_d4_confirmation/results/query_scores_and_depth_bootstrap.npz'
                      if donor == 'D4' else
                      root / f'v17_experiments/evaluation/results/combined/{donor}_Rest_query_scores.npz')
    ref = load(reference_path)
    reference_arm_ids = [list(s['arm_names'].astype(str)).index(str(a)) for a in ref['arm_names']]
    official_exact = {}
    for panel, key in (('responsive', 'responsive_positions'), ('terciles', 'tercile_positions')):
        actual = scores[reference_arm_ids][:, :, s[key], :]
        expected = ref[panel + '_scores']
        assert np.array_equal(actual, expected), (donor, panel, float(np.max(abs(actual - expected))))
        official_exact[panel] = dict(values=int(actual.size), maximum_absolute_difference=0.)
    report = dict(
        schema_version=1, donor=donor,
        identity='retrospective exact fixed-method top-25 sufficient-input compaction',
        source_files={str(p.relative_to(root)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                      for p in source_files + [reference_path]},
        output_file='ranking.npz', output_sha256=sha(path), output_bytes=path.stat().st_size,
        k=K, groups=5, queries=nq, candidates=nt, arms=list(s['arm_names'].astype(str)),
        full_Y_entries=int(y.size), retained_Y_entries=len(Y_values),
        retained_fraction=len(Y_values)/y.size,
        retained_candidates_per_row_percentiles=np.percentile(row_counts, [0, 25, 50, 75, 100]).tolist(),
        full_source_weights_content_sha256=content_sha(weights),
        full_source_Y_content_sha256=content_sha(y), full_source_truth_content_sha256=content_sha(truth),
        prediction_membership_values_compared=prediction_membership_comparisons,
        truth_membership_values_compared=truth_membership_comparisons,
        all_memberships_exact=True, independent_full_sort_memberships_exact=True,
        full_prediction_membership_stream_sha256=full_membership_hash.hexdigest(),
        compact_prediction_membership_stream_sha256=sparse_membership_hash.hexdigest(),
        original_official_query_scores_exact=official_exact,
        source_arrays_modified=False, float_quantization=False,
        supported_scope='Rebuild frozen scalar weights from separately retained statistics; rerank retained original float64 raw effects using those exact weights; reproduce every fixed signed top-25 membership, query score and subsequent paired bootstrap. All tied candidates at every frozen boundary are retained.',
        unsupported_scope='The omitted interior candidates cannot support arbitrary new weights, a changed K, a changed candidate/query pool, control refitting or raw UMI preprocessing. The compaction is post-result and is not a replacement prospective lock.',
        elapsed_seconds=time.monotonic()-start)
    (dest / 'RANKING_COMPACTION.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'donor': donor, 'bytes': path.stat().st_size,
                      'retained_fraction': report['retained_fraction'],
                      'membership_values_exact': prediction_membership_comparisons + truth_membership_comparisons,
                      'official_scores_exact': True}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument('--destination', type=Path, default=Path(__file__).resolve().parent / 'repo')
    p.add_argument('--donors', nargs='+', default=['D1', 'D2', 'D3', 'D4'])
    args = p.parse_args()
    reports = {d: build(args.source_root.resolve(), args.destination.resolve(), d) for d in args.donors}
    path = args.destination / 'audit' / 'RANKING_COMPACTION_AUDIT.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'status': 'PASS', 'donors': reports}, indent=2) + '\n')


if __name__ == '__main__':
    main()
