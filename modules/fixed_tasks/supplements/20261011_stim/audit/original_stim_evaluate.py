#!/usr/bin/env python3
"""Investigator-directed evaluation with explicit E4-deviation reporting.

This separate CLI requires the investigator-instruction release, not a claim
that the second registration is verified. Frozen estimators, predictions,
roles and scientific decisions are unchanged. No release is created here.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

import numpy as np

import stim_numeric as numeric
import stim_secondary as secondary
import stim_stats as stats
from stats_frozen_summary import all_contrasts
from stim_input_pipeline import array_digest, immutable_json, sha_file, utcnow


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class FrozenAssets:
    """Read only exact files bound by the released prediction manifest."""
    def __init__(self, root, manifest):
        self.root = Path(root).resolve()
        self.manifest = manifest
        self.hashes = manifest['file_sha256']

    def path(self, relative):
        path = (self.root / relative).resolve()
        if self.root not in path.parents or relative not in self.hashes:
            raise ValueError('Artifact absent from prediction lock: ' + relative)
        if sha_file(path) != self.hashes[relative]:
            raise ValueError('Prediction artifact hash changed: ' + relative)
        return path

    def npz(self, relative):
        with np.load(self.path(relative), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def json(self, relative):
        return json.loads(self.path(relative).read_text())


def declared_unavailable(manifest):
    out = {}
    for arm, declaration in manifest.get('unavailable_arms', {}).items():
        if arm not in ('PR_M', 'TRADE_scalar'):
            raise ValueError('Required H1/raw arm cannot be silently omitted')
        if isinstance(declaration, str):
            reason = declaration
        elif isinstance(declaration, dict) and declaration.get('status') == 'not_evaluable':
            reason = declaration.get('reason')
        else:
            raise ValueError('Malformed pre-truth unavailable-arm declaration')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Missing pre-truth reason for unavailable ' + arm)
        out[arm] = reason
    return out


def weights_from_lock(predictions, unavailable, *, crossguide=False):
    wanted = numeric.CROSSGUIDE_ARMS if crossguide else stats.ARMS
    weights = {}
    effects = np.asarray(predictions['original_effects'], float)
    for arm in wanted:
        if arm in unavailable:
            continue
        key = 'weight_' + arm
        if key not in predictions:
            raise ValueError('Undeclared missing arm: ' + arm)
        w = np.asarray(predictions[key], float)
        if w.shape != effects.shape[:2] or not np.isfinite(w).all() or np.any((w < 0) | (w > 1)):
            raise ValueError('Invalid locked weights: ' + arm)
        pkey = 'prediction_' + arm
        if pkey not in predictions or not np.array_equal(predictions[pkey], w[:, :, None] * effects):
            raise ValueError('Saved prediction differs from frozen effect/weight: ' + arm)
        weights[arm] = w
    return weights


def validate_axes(predictions, query, target_ids, model, *, main):
    tids = np.asarray(target_ids).astype(str)
    qu = np.asarray(query['qu'])
    gids = np.asarray(model['gene_ids_keep']).astype(str)
    qids = np.asarray(predictions['query_gene_ids']).astype(str)
    if (qu.ndim != 1 or qu.dtype.kind not in 'iu' or len(np.unique(qu)) != len(qu)
            or np.any(qu < 0) or np.any(qu >= len(gids))):
        raise ValueError('Invalid frozen query projection')
    if not (np.array_equal(predictions['target_ids'], tids)
            and np.array_equal(query['target_ids'], tids)
            and np.array_equal(query['query_gene_ids'], qids)
            and np.array_equal(qids, gids[qu])):
        raise ValueError('Frozen target/query/gene axes disagree')
    if np.asarray(predictions['original_effects']).shape != (5, len(tids), len(qu)):
        raise ValueError('Expected five input groups on locked target/query axes')
    panels = [('responsive', 'responsive_positions')]
    if main:
        panels.append(('terciles', 'tercile_positions'))
    for panel, key in panels:
        positions = np.asarray(query[key])
        if (positions.shape != (300,) or positions.dtype.kind not in 'iu'
                or len(np.unique(positions)) != 300 or np.any(positions < 0)
                or np.any(positions >= len(qu)) or np.asarray(query['blocks_' + panel]).shape != (300,)):
            raise ValueError('Invalid frozen 300-query panel: ' + panel)
    if main and np.asarray(query['blocks_union']).shape != qu.shape:
        raise ValueError('Union blocks not aligned to frozen union queries')


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


def score_and_analyze(main, query, truth, unavailable, *, cross=None, alignment=None):
    """Pure numerical join, also used by synthetic verification without I/O."""
    weights = weights_from_lock(main, unavailable)
    scores, names = stats.score_predictions(main['original_effects'], truth, weights,
                                            main['query_gene_ids'], main['target_ids'])
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
        xp, xq, xt = cross['predictions'], cross['query'], cross['truth']
        xw = weights_from_lock(xp, {}, crossguide=True)
        xs, xn = stats.score_predictions(xp['original_effects'], xt, xw,
                                         xp['query_gene_ids'], xp['target_ids'])
        pos = xq['responsive_positions']
        report['S2_crossguide'] = {'status': 'ok', 'targets': len(xp['target_ids']),
            'summary': stats.summarize_scores(np.take(xs, pos, axis=2), xn, xq['blocks_responsive']),
            'scope': 'Independent-guide task with its own pool/panel; no artifact fraction is inferred.'}
        output_arrays.update(crossguide_scores=xs, crossguide_arm_names=np.asarray(xn),
                             crossguide_query_gene_ids=xp['query_gene_ids'], crossguide_target_ids=xp['target_ids'])
    if alignment is None or alignment['status'] != 'ok':
        report['S3_alignment'] = alignment or {'status': 'not_evaluable', 'reason': 'pretruth_design_unavailable'}
    else:
        contributions = secondary.nomination_contributions(
            main['original_effects'][:, :, rp], truth[:, rp], weights['PertRank'],
            weights['Euclidean'], main['query_gene_ids'][rp], main['target_ids'])
        regression = secondary.alignment_regression(contributions, alignment,
            query['blocks_responsive'], target_ids=main['target_ids'], query_ids=main['query_gene_ids'][rp])
        output_arrays['alignment_contributions'] = contributions
        if 'coefficients' in regression:
            output_arrays['alignment_coefficients'] = regression.pop('coefficients')
        report['S3_alignment'] = regression
    return report, output_arrays


class TruthResults:
    """Durable reference moments; exact lock-bound resume, never overwrite."""
    def __init__(self, root, release):
        self.root = Path(root)
        self.out = self.root / 'evaluation'
        self.out.mkdir(exist_ok=True)
        self.binding = {k: release[k] for k in ('release_sha256', 'prediction_manifest_sha256',
                                               'evaluation_manifest_sha256')}

    def load(self, name, identity):
        p = self.out / name
        marker = p.with_suffix(p.suffix + '.json')
        if not p.exists() and not marker.exists():
            return None
        if not p.exists() or not marker.exists():
            raise ValueError('Incomplete evaluation checkpoint; documented recovery required')
        rec = json.loads(marker.read_text())
        if rec['binding'] != self.binding or rec['identity'] != identity or rec['sha256'] != sha_file(p):
            raise ValueError('Evaluation checkpoint identity/digest mismatch')
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def save(self, name, values, identity):
        old = self.load(name, identity)
        if old is not None:
            if array_digest(old) != array_digest(values):
                raise ValueError('Refusing changed evaluation output')
            return self.out / name
        p = self.out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if any(np.asarray(v).dtype.hasobject for v in values.values()):
            raise ValueError('Object dtype in evaluation checkpoint')
        tmp = p.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, **values)
        os.replace(tmp, p)
        immutable_json(p.with_suffix(p.suffix + '.json'),
                       {'binding': self.binding, 'identity': identity, 'sha256': sha_file(p)})
        return p


def reference_mean(reader, store, model, rows, role, identity, name):
    """Exact registered transform; caller supplies frozen reserved row IDs."""
    found = store.load(name, identity)
    if found is not None:
        if not np.array_equal(found['row_ids'], rows):
            raise ValueError('Reference checkpoint row identity changed')
        return found['mean']
    expected = 2000 if role == 'evaluation_control' else 100
    if len(rows) != expected:
        raise ValueError('Invalid registered reference cell count')
    raw = reader.rows(rows, role=role)
    mean = numeric.log_rows(raw, model, control_precision=(role == 'evaluation_control')).mean(axis=0)
    if not np.isfinite(mean).all():
        raise ValueError('Nonfinite reference mean')
    path = store.save(name, {'mean': mean, 'row_ids': np.asarray(rows)}, identity)
    # The optional truth-specific eviction method must enforce the same gate.
    if hasattr(reader, 'prune_truth_checkpoint'):
        reader.prune_truth_checkpoint(rows, role, {'path': str(path.relative_to(store.root)), 'sha256': sha_file(path)})
    return mean


def evaluate(root, workers=8):
    # This is the first statement that can authorize actual truth evaluation.
    from stim_truth_reader import TruthReader, validate_investigator_directed_release
    release = validate_investigator_directed_release(root)
    assets = FrozenAssets(root, release['prediction_manifest'])
    unavailable = declared_unavailable(release['prediction_manifest'])
    model = assets.npz('input/control_model.npz')
    cohort = assets.npz('cohort_roles.npz')
    query = assets.npz('input/main_query_context.npz')
    if 'TRADE_scalar' in unavailable:
        if release['prediction_manifest'].get('trade_status') != 'not_evaluable':
            raise ValueError('TRADE omission requires explicit pretruth task status')
        assets.json('input/TRADE_NOT_EVALUABLE.json')
        main = assets.npz('input/main_nontrade_predictions.npz')
    else:
        main = assets.npz('input/main_allarm_predictions.npz')
    validate_axes(main, query, cohort['target_ids'], model, main=True)
    weights_from_lock(main, unavailable)
    cross_status = assets.json('input/crossguide_pretruth_report.json')
    alignment_report = assets.json('input/alignment_pretruth_report.json')
    for value in (cross_status, alignment_report):
        if value.get('status') not in ('ok', 'not_evaluable'):
            raise ValueError('Secondary endpoint lacks frozen feasibility status')
    cross = dict(cross_status)
    if cross['status'] == 'ok':
        cross['predictions'] = assets.npz('input/xguide_nontrade_predictions.npz')
        cross['query'] = assets.npz('input/xguide_query_context.npz')
        validate_axes(cross['predictions'], cross['query'], cohort['xguide_target_ids'], model, main=False)
        weights_from_lock(cross['predictions'], {}, crossguide=True)
    alignment = dict(alignment_report)
    if alignment['status'] == 'ok':
        frozen = assets.npz('input/alignment_pretruth.npz')
        alignment.update({k[len('design_'):]: v for k, v in frozen.items() if k.startswith('design_')})
        if (not np.array_equal(alignment['target_ids'], main['target_ids']) or
                not np.array_equal(alignment['query_ids'], main['query_gene_ids'][query['responsive_positions']])):
            raise ValueError('Frozen alignment design identifiers changed')
    reader = TruthReader(root, workers=workers)
    try:
        return _evaluate_references(root, release, assets, model, cohort, query, main,
                                    unavailable, cross, alignment, alignment_report, reader)
    finally:
        # New transport-only Reader versions may retain a network thread pool.
        # The evaluator owns it through success and every exceptional exit.
        close = getattr(reader, 'close', None)
        if callable(close):
            close()


def _evaluate_references(root, release, assets, model, cohort, query, main,
                         unavailable, cross, alignment, alignment_report, reader):
    store = TruthResults(root, release)
    ctrl_rows = cohort['role__evaluation_control']
    control = reference_mean(reader, store, model, ctrl_rows, 'evaluation_control',
                             {'role': 'evaluation_control'}, 'evaluation_control.npz')
    print(json.dumps({'utc': utcnow(), 'stage': 'evaluation_control_complete',
                      'protocol_status': 'investigator_directed_deviation'}), flush=True)
    truths = {}
    tasks = [('main', main, query)]
    if cross['status'] == 'ok':
        tasks.append(('xguide', cross['predictions'], cross['query']))
    for task, pred, context in tasks:
        rows = cohort['role__' + task + '_truth']
        if rows.shape != (len(pred['target_ids']), 100):
            raise ValueError('Truth target/row axis mismatch')
        reference = np.empty((len(rows), len(context['qu'])))
        for j, reserved in enumerate(rows):
            identity = {'role': task + '_truth', 'target_index': j, 'target_id': str(pred['target_ids'][j])}
            mean = reference_mean(reader, store, model, reserved, task + '_truth', identity,
                                  f'{task}/target_{j:04d}.npz')
            reference[j] = (mean - control)[context['qu']]
            if (j + 1) % 8 == 0 or j + 1 == len(rows):
                print(json.dumps({'utc': utcnow(), 'stage': task + '_reference_progress',
                                  'completed': j + 1, 'total': len(rows)}), flush=True)
        truths[task] = reference
        store.save(task + '_reference.npz', {'effects': reference, 'target_ids': pred['target_ids'],
                   'query_gene_ids': pred['query_gene_ids']}, {'task': task})
    if cross['status'] == 'ok':
        cross['truth'] = truths['xguide']
    report, arrays = score_and_analyze(main, query, truths['main'], unavailable,
                                      cross=cross, alignment=alignment)
    report['S3_frozen_pretruth_report'] = alignment_report
    if 'input/alignment_pretruth.npz' in assets.hashes:
        # Full index distributions/ranks and exclusions stay in their exact
        # pre-truth archive, including when the regression is not estimable.
        report['S3_frozen_indices'] = {'path': 'input/alignment_pretruth.npz',
            'sha256': assets.hashes['input/alignment_pretruth.npz'],
            'interpretation': 'Original pre-truth indices and exclusions; no re-estimation after truth.'}
    scores_path = store.save('registered_scores.npz', arrays, {'stage': 'all_registered_scores'})
    report = {
        'protocol_status': 'investigator_directed_deviation',
        'independent_e4_registration_verified': False,
        'e4_archive_approval_status_before_evaluation': 'not_independently_verified',
        'protocol_deviation': 'Evaluation proceeded under explicit investigator direction while the separate E4 prediction-lock archive and approval were not independently verified. The statistical rules were kept fixed; full preregistration compliance is not claimed.',
        **report,
    }
    report.update(condition='D4_Stim8hr', status='EVALUATION_COMPLETE',
                  prediction_manifest_sha256=release['prediction_manifest_sha256'],
                  evaluation_manifest_sha256=release['evaluation_manifest_sha256'],
                  truth_release_sha256=release['release_sha256'], scores_sha256=sha_file(scores_path))
    # Deterministic reruns must match the originally saved report exactly.
    serial = json.loads(json.dumps(report, default=jsonable, allow_nan=False))
    immutable_json(store.out / 'EVALUATION_REPORT.json', serial)
    return serial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    root = args.run_dir.resolve()
    # Separate from the input lock; root must first confirm input has ended.
    with (root / 'EVALUATION_OWNER.lock').open('a+') as owner:
        try:
            fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another evaluation owner is active') from error
        result = evaluate(root, workers=args.workers)
        print(json.dumps(result, default=jsonable, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
