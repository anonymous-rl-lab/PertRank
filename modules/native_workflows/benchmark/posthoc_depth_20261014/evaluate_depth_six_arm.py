#!/usr/bin/env python3
"""Post-hoc 250-target depth adapter; unchanged primary scorer and independent audit.

No polling: an explicit owner completion attestation is required. The owner must
have observed its execution session exit before issuing the attestation. New data
are not read by --check-code-only. Each curve/depth is a separate descriptive task.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
FROZEN = {
    'posthoc_depth_20261014/POSTHOC_DEPTH_PROTOCOL.json': 'bd529ba31eaf41cfff522716835d2f95aedc61bfd7a3fdae68e4140e7e6bb546',
    'evaluation/evaluate_six_arm.py': '4acd8cebce3162b25874518942dbed4fae8299cd8fead65da2ef1b789dd95237',
    'audit/audit_six_arm.py': '44b5840c8b60ae209fbbc4c7e686c8574ba16d6d3a42fd9bda4713867b6c367a',
    'independent/execution/evaluate_independent.py': '21cfcbe73404d7560922069de96b25fa706ac75900afb014a8aed1880fbc05cf',
    'AMENDMENT_03_FR_SEARCH_21.json': 'edd9b8580c27bdc951938146fe90227aff8f54e9835569ce0122d2bfe0432ab6',
}
CURVES = ['A_native_default_penalty', 'B_effective_penalty_sensitivity']
SCOPE = ('Post-hoc descriptive diagnostic on 250 metadata-selected D3 targets, '
         'one nested 20/50/100-cell chain, fixed query panel/reference/controls. '
         'Query-block intervals describe conditional query heterogeneity only. '
         'Depths are dependent and are not biological replicates. Absolute levels '
         'must not be compared directly with the 1,200-target main benchmark. '
         'No study-wide or confirmatory multiplicity claim is made.')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def check_npz(path):
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad is not None:
            raise ValueError(f'NPZ CRC failure: {path}: {bad}')

def get_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def check_code():
    for name, expected in FROZEN.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Frozen protocol/source changed: {name}')
    return FROZEN.copy()

def check_owner(path, curve, n, output_dir):
    """An owner-issued attestation supplements local CRC/hash checks, not a poll."""
    owner = read_json(path)
    if owner.get('status') != 'OWNER_EXECUTION_ENDED_INTEGRITY_VERIFIED':
        raise ValueError('Owner execution has not been explicitly released')
    if owner.get('owner_exit_code') != 0 or not owner.get('owner_session_id'):
        raise ValueError('Owner must record the completed execution session and exit code')
    if owner.get('curve') != curve or owner.get('depth') != n:
        raise ValueError('Owner attestation identifies a different task')
    if owner.get('npz_crc') != 'PASS':
        raise ValueError('Owner has not certified NPZ CRC')
    files = owner.get('files_sha256', {})
    for name in ['effects.npz', 'fit_report.json', 'NATIVE_SOLVER_OBJECTIVE_AUDIT.json']:
        if name not in files or sha(output_dir / name) != files[name]:
            raise ValueError(f'Owner file identity mismatch: {name}')
    check_npz(output_dir / 'effects.npz')
    return owner

def run(args):
    check_code()
    n, curve = args.depth, args.curve
    protocol_path = HERE / 'POSTHOC_DEPTH_PROTOCOL.json'
    protocol = read_json(protocol_path)
    if protocol['status'] != 'FROZEN_POSTHOC_DIAGNOSTIC_BEFORE_NEW_DEPTH_PAYLOAD':
        raise ValueError('Unrecognized depth protocol status')
    inputs = HERE / 'inputs' / f'n{n}'
    fitdir = (args.fr_output_dir or HERE / 'native_fr_outputs' / curve / f'n{n}').resolve()
    # Owner gate comes before opening axes/expression data.
    owner = check_owner(args.owner_completion, curve, n, fitdir)
    src = read_json(ROOT / 'AMENDMENT_03_FR_SEARCH_21.json')
    config = dict(rank=protocol['FR']['rank'], spca_alpha=protocol['FR']['spca_alpha'],
                  lasso_alpha=protocol['FR']['curves'][curve]['lasso_alpha_by_depth'][str(n)])
    if protocol['FR']['seed'] != 20261012:
        raise ValueError('Depth seed no longer matches unmodified evaluator')
    selection_path = HERE / 'selection.npz'
    for name in ['selection.npz', 'selection.lock.json']:
        if sha(HERE / name) != protocol['files_sha256'][name]:
            raise ValueError(f'Depth selection identity mismatch: {name}')
    ia = read_json(inputs / 'INPUT_AUDIT.json')
    if (ia['status'] != 'FROZEN_POSTHOC_NESTED_DEPTH_INPUT_PASS'
            or ia['depth'] != n or ia['targets'] != 250
            or ia['cells'] != 2000 + 250*n
            or ia['selection_sha256'] != sha(selection_path)
            or ia['protocol_sha256'] != sha(protocol_path)):
        raise ValueError('Input audit does not bind this frozen depth task')
    for key in ['same_original_2000_NTC', 'raw_totals_exact', 'full_gene_axis_exact',
                'same_query_panel', 'same_archived_reference_subset', 'weights_recomputed_with_this_n']:
        if ia.get(key) is not True:
            raise ValueError(f'Input audit missing a passing check: {key}')
    for name, expected in ia['files_sha256'].items():
        path = inputs / name
        if sha(path) != expected:
            raise ValueError(f'Input source hash mismatch: {name}')
        if path.suffix == '.npz':
            check_npz(path)
    for required in ['axes.npz', 'counts.npz', 'full_effect_stats.npz']:
        if required not in ia['files_sha256']:
            raise ValueError(f'Incomplete input audit: {required}')
    if n == 20 and (not ia['n20_frozen_replay_errors'] or max(ia['n20_frozen_replay_errors'].values()) >= 1e-10):
        raise ValueError('n20 did not reproduce the frozen D3 subset')
    selection = load_npz(selection_path)
    axes = load_npz(inputs / 'axes.npz')
    context = load_npz(ROOT / 'pooled_data/D3_Rest/r20_context.npz')
    target_cells = selection['input_cells'][:, :n]
    expected_ids = np.r_[selection['NTC_input'], target_cells.ravel()]
    expected_labels = np.r_[np.full(2000, -1, np.int32), np.repeat(np.arange(250), n)]
    checks = {}
    def exact(name, a, b):
        checks[name] = bool(np.array_equal(a, b))
        if not checks[name]:
            raise ValueError(f'Depth identity mismatch: {name}')
    exact('ordered_input_cells', axes['cell_ids'], expected_ids)
    exact('ordered_input_labels', axes['labels'], expected_labels)
    exact('target_identity', axes['target_ids'], selection['target_ids'])
    exact('full_gene_identity', axes['gene_ids'], selection['gene_ids_full'])
    exact('full_gene_identity_original_context', axes['gene_ids'], context['gene_ids_full'])
    ix, pos = selection['target_indices'], context['responsive_positions']
    exact('metadata_selected_candidates', axes['target_ids'], context['target_ids'][ix])
    exact('responsive_query_ids', axes['query_ids'], context['query_gene_ids'][pos])
    exact('query_blocks', axes['query_blocks'], context['blocks_responsive'])
    exact('responsive_gene_columns', axes['query_columns'], context['keep'][context['qu'][pos]])
    exact('same_archived_reference', axes['reference_effects'], context['truth'].T[ix][:, pos])
    exact('target_count_metadata', axes['target_cells_per_group'], np.asarray(n))
    exact('control_count_metadata', axes['input_control_count'], np.asarray(2000))
    exact('input_rep_metadata', axes['rep'], np.asarray(0))
    exact('depth_task_metadata', axes['task'], np.asarray(f'D3_DEPTH250_n{n}'))
    fit = read_json(fitdir / 'fit_report.json')
    if fit['parameters']['spca_iter'] != protocol['FR']['spca_iterations']:
        raise ValueError('Native FR iteration budget changed')
    if fit['input_provenance']['counts_sha256'] != ia['files_sha256']['counts.npz'] or fit['input_provenance']['axes_sha256'] != ia['files_sha256']['axes.npz']:
        raise ValueError('FR was fit on a different depth input')
    if (fit['input']['n_cells'] != 2000+250*n or fit['input']['n_targets'] != 250
            or fit['input']['n_ntc'] != 2000 or fit['input']['n_genes'] != 18130
            or fit['input']['cells_per_target'] != [n]*250):
        raise ValueError('Native FR input dimensions are not the locked depth task')
    primary = get_module('depth_frozen_primary', ROOT / 'evaluation/evaluate_six_arm.py')
    audit = get_module('depth_independent_audit', ROOT / 'audit/audit_six_arm.py')
    z = dict(raw_scores=axes['input_effects'].T[None].astype(np.float64),
             reference_scores=axes['reference_effects'].T.astype(np.float64),
             w0=axes['weights_euclidean'][None].astype(np.float64),
             wm=axes['weights_pertrank'][None].astype(np.float64),
             queries=axes['query_ids'].astype(str), candidates=axes['target_ids'].astype(str),
             blocks=axes['query_blocks'], input_reps=np.array([0]))
    native_selection = dict(target_ids=selection['target_ids'], A_input=target_cells,
                            NTC_input=selection['NTC_input'], gene_ids_full=selection['gene_ids_full'])
    effect, native_provenance = primary.native_fr(fitdir / 'effects.npz', z['candidates'],
        z['queries'], 0, config, {'fr_commit': src['fr_commit']}, selection=native_selection)
    z['fr_scores'] = effect[None]
    # D3 pooled validation checks allow exactly one repetition and >=250 targets;
    # they are shape safeguards, not a claim that this diagnostic is confirmatory.
    primary.validate_compact(z, 'D3_Rest', 'pooled')
    started = time.monotonic()
    result, arrays, reference_seed = primary.score_arrays(z, draws=2000)
    if reference_seed is not None:
        raise ValueError('Unexpected reference-cell bootstrap in depth diagnostic')
    for k in (25, 10):
        # The unchanged scorer carries primary-task family labels; explicitly
        # replace their interpretation for these post-hoc diagnostic outputs.
        for item in result[str(k)]['contrasts'].values():
            item['in_prespecified_five_contrast_family'] = False
            item['inference_role'] = 'posthoc_descriptive_only'
        point = arrays[f'six_arm_p{k}'].mean((0, 2))
        boot = arrays[f'query_only_bootstrap_p{k}']
        draw = boot[:, 2] - boot[:, 5]
        result[str(k)]['contrasts']['pertrank_minus_fr_pertrank'] = {
            'point': float(point[2]-point[5]), 'input_rep_points': [float(point[2]-point[5])],
            'query_only_95_interval': np.quantile(draw, [.025, .975]).tolist(),
            'query_only_99_interval': np.quantile(draw, [.005, .995]).tolist(),
            'in_prespecified_five_contrast_family': False, 'inference_role': 'posthoc_descriptive_only'}
        z[f'six_arm_p{k}'] = arrays[f'six_arm_p{k}']
    destination = HERE / 'evaluation' / curve / f'n{n}'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite an existing depth result: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.n{n}_incomplete_', dir=destination.parent))
    # Publish only a completely verified folder; leave failed staging evidence.
    np.savez_compressed(stage / 'compact.npz', **z)
    np.savez_compressed(stage / 'evaluation_arrays.npz', **arrays)
    independent_checks = {}
    for k in (25, 10):
        ap = stage / f'INDEPENDENT_AUDIT_P{k}.json'
        ar = audit.run([stage/'compact.npz'], ap, k, 2000, primary.SEED, True, expected_reps=1)
        az = load_npz(ap.with_suffix('.npz'))
        err = float(np.max(np.abs(az[f'condition_0_six_arm_p{k}']-arrays[f'six_arm_p{k}'])))
        berr = float(np.max(np.abs(az['condition_0_bootstrap_means']-arrays[f'query_only_bootstrap_p{k}'])))
        if err > 1e-12 or berr > 1e-12:
            raise ValueError(f'Independent P{k} scoring/bootstrap mismatch')
        if not np.array_equal(az['condition_0_bootstrap_sampled_block_indices'], arrays['sampled_query_blocks']):
            raise ValueError('Independent bootstrap block draws differ')
        for name, pair in audit.CONTRASTS.items():
            expected = result[str(k)]['contrasts'][name]
            observed = ar['conditions'][0]['contrasts'][name]
            if max(abs(expected['point']-observed['mean']),
                   np.max(np.abs(np.array(expected['query_only_95_interval'])-observed['conditional_query_block_ci95'])),
                   np.max(np.abs(np.array(expected['query_only_99_interval'])-observed['conditional_query_block_ci99_bonferroni_five']))) > 1e-12:
                raise ValueError(f'Independent P{k} contrast mismatch: {name}')
        independent_checks[str(k)] = dict(status='PASS', per_query_max_abs_error=err,
            bootstrap_max_abs_error=berr, same_sampled_query_blocks=True,
            check_count=len(ar['conditions'][0]['checks']),
            scope_override=SCOPE,
            note='The unchanged audit uses generic family-label field names; all intervals in this post-hoc depth adapter are descriptive.')
        # Atomic publication relocates the same bytes; bind the audit to the
        # final compact path, keeping its recomputed SHA and numbers unchanged.
        ar['conditions'][0]['source'] = str((destination/'compact.npz').resolve())
        save_json(ap, ar)
    # Common-arm fingerprint lets root verify identity before joining ASH tables.
    common = {key: array_sha(z[key]) for key in ['raw_scores','reference_scores','w0','wm','queries','candidates','blocks']}
    common.update({f'three_arm_query_p{k}': array_sha(arrays[f'six_arm_p{k}'][:, :3]) for k in (25,10)})
    report = dict(status='COMPLETE_INDEPENDENT_AUDIT_PASS', created_utc=datetime.now(timezone.utc).isoformat(),
        task=f'D3_DEPTH250_n{n}', dataset='D3_Rest', kind='pooled_posthoc_depth', depth=n,
        curve=curve, curve_role=protocol['FR']['curves'][curve]['role'], config=config,
        input_reps=[0], arms=primary.ARMS, shape=dict(input_reps=1, queries=300, candidates=250, blocks=30),
        primary_K=25, sensitivity_K=10, bootstrap_draws=2000, query_seed=primary.SEED,
        results=result, scope=SCOPE, all_intervals_descriptive=True, confirmatory_main_result=False,
        depth_inference='Nested cells: no independent-depth or biological-replicate interpretation.',
        source_checks=checks, independent_audit=independent_checks, common_three_arm_array_sha256=common,
        provenance=dict(protocol_path=str(protocol_path), protocol_sha256=sha(protocol_path),
            selection_path=str(selection_path), selection_sha256=sha(selection_path),
            input_audit_path=str(inputs/'INPUT_AUDIT.json'), input_audit_sha256=sha(inputs/'INPUT_AUDIT.json'),
            input_file_sha256=ia['files_sha256'], owner_completion_path=str(args.owner_completion.resolve()),
            owner_completion_sha256=sha(args.owner_completion), owner=owner,
            adapter_sha256=sha(__file__), frozen_source_sha256=FROZEN, fr=native_provenance),
        elapsed_scoring_and_audit_seconds=time.monotonic()-started)
    for p in stage.glob('*.npz'):
        check_npz(p)
    report['files_sha256'] = {p.name: sha(p) for p in stage.iterdir() if p.is_file()}
    save_json(stage / 'evaluation_report.json', report)
    os.rename(stage, destination)
    print(json.dumps({'status': report['status'], 'report': str(destination/'evaluation_report.json'),
                      'P25': {a: result['25']['arms'][a]['point'] for a in primary.ARMS}}, indent=2))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-code-only', action='store_true')
    ap.add_argument('--depth', type=int, choices=[20,50,100])
    ap.add_argument('--curve', choices=CURVES)
    ap.add_argument('--fr-output-dir', type=Path, help='Owner-released unique output directory; defaults to the conventional depth folder')
    ap.add_argument('--owner-completion', type=Path)
    args = ap.parse_args()
    if args.check_code_only:
        print(json.dumps({'status':'FROZEN_SOURCE_CHECK_PASS_NO_DATA_READ', 'hashes':check_code()}, indent=2))
        return
    if args.depth is None or args.curve is None or args.owner_completion is None:
        ap.error('--depth, --curve and --owner-completion are required; no automatic polling')
    with (ROOT/'heavy_memory.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        run(args)

if __name__ == '__main__':
    main()
