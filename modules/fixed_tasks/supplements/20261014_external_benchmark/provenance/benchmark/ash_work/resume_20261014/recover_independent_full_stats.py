#!/usr/bin/env python3
"""Derive full-gene statistics from the locked independent D4 input only.

The frozen control model and input predictions are verified before use. No
reference or QC expression arrays are opened. These sufficient statistics have
no historical file hash: validation is against the original input predictions.
Source inputs and historical audits are never modified.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import threading
import time

import numpy as np
from scipy.sparse import load_npz


ROOT = Path(__file__).resolve().parents[2]
CGROUP = Path('/sys/fs/cgroup')
TASK = 'D4_Stim_independent'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_keys(path, keys):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in keys}


def emit(**entry):
    print(json.dumps(entry, sort_keys=True), flush=True)


def current_memory():
    return int((CGROUP / 'memory.current').read_text())


def memory_headroom(limit):
    stat = dict(line.split() for line in (CGROUP / 'memory.stat').read_text().splitlines())
    current = current_memory()
    inactive_file = int(stat['inactive_file'])
    dirty_and_writeback = int(stat['file_dirty']) + int(stat['file_writeback'])
    clean_inactive_file = max(0, inactive_file - dirty_and_writeback)
    return dict(current_bytes=current, limit_bytes=limit,
                inactive_file_bytes=inactive_file,
                dirty_and_writeback_bytes=dirty_and_writeback,
                reclaimable_clean_inactive_file_bytes=clean_inactive_file,
                effective_headroom_bytes=None if limit is None else limit - current + clean_inactive_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument('--minimum-headroom-gib', type=float, default=1.6)
    parser.add_argument('--source-input-dir', type=Path, default=ROOT / 'inputs' / TASK)
    args = parser.parse_args()
    if args.minimum_headroom_gib <= 0:
        parser.error('--minimum-headroom-gib must be positive')
    start = time.monotonic()
    source = args.source_input_dir.resolve()
    execution = ROOT / 'independent' / 'execution'
    context = execution / 'D4_Stim8hr'
    output = args.output_root / TASK / 'rep0'
    output.mkdir(parents=True, exist_ok=False)

    audit_path = source / 'PREPARATION.json'
    audit_sha = sha(audit_path)
    source_audit = json.loads(audit_path.read_text())
    assert source_audit['status'] == 'PASS'
    assert source_audit['shape'] == [9900, 18130]
    lock_path = context / 'PREDICTION_LOCK.json'
    prediction_lock = json.loads(lock_path.read_text())
    assert prediction_lock['status'] == 'PREDICTION_LOCKED'
    assert prediction_lock['condition'] == 'D4_Stim8hr'
    assert prediction_lock['targets'] == 395
    assert prediction_lock['queries'] == 300
    assert prediction_lock['input_depth'] == 20
    assert prediction_lock['reference_roles_opened'] == []
    selection_path = execution / 'selection_D4_Stim8hr.npz'
    selection_lock_path = execution / 'selection_D4_Stim8hr.lock.json'
    model_path = context / 'control_model.npz'
    prediction_path = context / 'prediction.npz'
    source_hashes = {str(audit_path): audit_sha, str(lock_path): sha(lock_path)}

    def verify(path, *expected):
        actual = sha(path)
        assert all(actual == value for value in expected), (str(path), actual, expected)
        source_hashes[str(path)] = actual

    verify(source / 'counts.npz', source_audit['counts_sha256'])
    verify(source / 'axes.npz', source_audit['axes_sha256'])
    verify(selection_path, source_audit['source_selection_sha256'],
           prediction_lock['selection_sha256'])
    verify(selection_lock_path, prediction_lock['selection_lock_sha256'])
    selection_lock = json.loads(selection_lock_path.read_text())
    assert selection_lock['status'] == 'METADATA_SELECTION_LOCKED'
    assert selection_lock['condition'] == 'D4_Stim8hr'
    assert selection_lock['selection_sha256'] == source_hashes[str(selection_path)]
    verify(model_path, prediction_lock['control_model_sha256'])
    verify(prediction_path, source_audit['source_prediction_sha256'],
           prediction_lock['prediction_sha256'])

    memory_samples = [current_memory()]
    stopped = threading.Event()

    def monitor():
        while not stopped.wait(0.5):
            memory_samples.append(current_memory())

    sampler = threading.Thread(target=monitor, daemon=True)
    sampler.start()
    lock = (ROOT / 'heavy_memory.lock').open('a')
    emit(phase='WAIT_FOR_HEAVY_MEMORY_LOCK', task=TASK, rep=0)
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    lock_acquired = time.monotonic()
    limit_text = (CGROUP / 'memory.max').read_text().strip()
    limit = None if limit_text == 'max' else int(limit_text)
    minimum = int(args.minimum_headroom_gib * (1 << 30))
    with (source / 'counts.npz').open('rb') as handle:
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    gate = memory_headroom(limit)
    while limit is not None and gate['effective_headroom_bytes'] < minimum:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        emit(phase='WAIT_FOR_MEMORY_HEADROOM_LOCK_RELEASED',
             minimum_headroom=minimum, **gate)
        if time.monotonic() - lock_acquired > 10800:
            raise TimeoutError('Insufficient memory headroom after three hours')
        time.sleep(5)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        gate = memory_headroom(limit)
    compute_start = time.monotonic()
    emit(phase='REBUILD_START', task=TASK, rep=0,
         cgroup_current_bytes=current_memory(), cgroup_limit_bytes=limit)

    axes = load_keys(source / 'axes.npz', (
        'labels', 'gene_ids', 'target_ids', 'target_names', 'cell_ids',
        'query_ids', 'query_blocks', 'query_columns', 'keep', 'scale',
        'input_effects', 'weights_euclidean', 'weights_pertrank'))
    selection = load_keys(selection_path, (
        'NTC_input', 'A_input', 'target_ids', 'target_names', 'gene_ids_full',
        'query_ids', 'query_blocks', 'all_selected_rows', 'selected_total_counts'))
    model = load_keys(model_path, (
        'keep', 'gene_ids_full', 'gene_ids_keep', 'scale', 'cm', 'cv',
        'trC', 'n_control'))
    prediction = load_keys(prediction_path, (
        'input_effects', 'weights_euclidean', 'weights_pertrank', 'energy',
        'noise_trace', 'target_ids', 'target_names', 'query_ids', 'query_blocks',
        'A_input_cell_ids'))
    keep = model['keep']
    genes = model['gene_ids_keep']
    targets = selection['target_ids']
    cm = model['cm']
    cv = model['cv']
    scale = model['scale']
    ntc = selection['NTC_input']
    tr = selection['A_input']
    n, g = len(targets), len(genes)
    assert n == 395 and g == prediction_lock['weight_genes'] == 13327
    assert ntc.shape == (2000,) and tr.shape == (n, 20)
    assert int(model['n_control']) == 2000
    assert float(scale) == prediction_lock['normalization_scale'] == 9051.0
    assert cm.shape == cv.shape == (g,)
    assert np.isfinite(cm).all() and np.isfinite(cv).all() and np.all(cv >= 0)
    assert len(np.unique(genes)) == g and len(np.unique(targets)) == n
    assert np.array_equal(model['gene_ids_full'], selection['gene_ids_full'])
    assert np.array_equal(genes, selection['gene_ids_full'][keep])
    gene_positions = {str(gene): i for i, gene in enumerate(genes)}
    qi = np.array([gene_positions[str(gene)] for gene in selection['query_ids']], dtype=np.int64)
    assert qi.shape == (300,) and len(np.unique(qi)) == 300
    expected_labels = np.r_[np.full(2000, -1, dtype=np.int32),
                            np.repeat(np.arange(n, dtype=np.int32), 20)]
    assert axes['labels'].dtype == expected_labels.dtype
    assert np.array_equal(axes['labels'], expected_labels)
    assert np.array_equal(axes['cell_ids'], np.r_[ntc, tr.ravel()])
    assert np.array_equal(axes['gene_ids'], selection['gene_ids_full'])
    assert np.array_equal(axes['keep'], keep)
    assert float(axes['scale']) == float(scale)
    assert np.array_equal(axes['query_columns'], keep[qi])
    for key in ('target_ids', 'target_names', 'query_ids', 'query_blocks'):
        assert np.array_equal(axes[key], selection[key]), key
        assert np.array_equal(prediction[key], selection[key]), key
    assert np.array_equal(prediction['A_input_cell_ids'], tr)
    for key in ('input_effects', 'weights_euclidean', 'weights_pertrank'):
        assert np.array_equal(axes[key], prediction[key]), key
    assert prediction['input_effects'].shape == (n, 300)
    assert prediction['energy'].shape == prediction['noise_trace'].shape == (n,)

    raw = load_npz(source / 'counts.npz')
    assert raw.shape == tuple(source_audit['shape']) == (2000 + n * 20, len(axes['gene_ids']))
    assert int(raw.nnz) == source_audit['nnz']
    tot = np.asarray(raw.sum(1)).ravel()
    sp = np.searchsorted(selection['all_selected_rows'], axes['cell_ids'])
    assert np.array_equal(selection['all_selected_rows'][sp], axes['cell_ids'])
    assert np.array_equal(selection['selected_total_counts'][sp], tot)
    assert np.isfinite(tot).all() and np.all(tot > 0)
    cn = raw[:2000]
    assert np.array_equal(np.flatnonzero(np.asarray(cn.sum(0)).ravel() > 0), keep)
    assert float(np.median(tot[:2000])) == float(scale)
    cc = np.log1p(cn[:, keep].toarray().astype(float) *
                  (float(scale) / tot[:2000])[:, None]).astype(np.float32).astype(float)
    cm_raw = cc.mean(0)
    centered = cc - cm_raw
    cv_raw = (centered * centered).sum(0) / 1999
    center_delta = float(np.max(np.abs(cm_raw - cm)))
    variance_delta = float(np.max(np.abs(cv_raw - cv)))
    trace_delta = abs(float(cv.sum()) - float(model['trC']))
    assert center_delta < 1e-12 and variance_delta < 1e-12
    assert trace_delta < 1e-10
    del cn, cc, centered, cv_raw

    effects = np.empty((n, g))
    se = np.empty((n, g))
    target_var = np.empty((n, g))
    energy = np.empty(n)
    noise_trace = np.empty(n)
    for j in range(n):
        lo = 2000 + j * 20
        dense = raw[lo:lo + 20, keep].toarray().astype(float)
        x = np.log1p(dense * (float(scale) / tot[lo:lo + 20])[:, None])
        y = x.mean(0) - cm
        v = x.var(0, ddof=1)
        effects[j] = y
        target_var[j] = v
        se[j] = np.sqrt(v / 20 + cv / 2000)
        energy[j] = y @ y
        noise_trace[j] = v.sum() / 20 + float(model['trC']) / 2000
        if (j + 1) % 200 == 0 or j + 1 == n:
            emit(phase='TARGETS_REBUILT', completed=j + 1, total=n)
    del raw, dense, x, y, v
    assert np.isfinite(effects).all() and np.isfinite(se).all() and np.all(se > 0)
    qerror = float(np.max(np.abs(effects[:, qi] - prediction['input_effects'])))
    energy_error = float(np.max(np.abs(energy - prediction['energy'])))
    noise_error = float(np.max(np.abs(noise_trace - prediction['noise_trace'])))
    assert qerror < 1e-12, qerror
    np.testing.assert_allclose(energy, prediction['energy'], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(noise_trace, prediction['noise_trace'], rtol=1e-12, atol=1e-10)
    checks = dict(query_effect_max_abs_error=qerror,
                  all_query_effect_max_abs_error=qerror,
                  input_prediction_energy_max_abs_error=energy_error,
                  input_prediction_noise_trace_max_abs_error=noise_error,
                  input_prediction_moment_tolerance=dict(rtol=1e-12, atol=1e-10),
                  minimum_SE=float(se.min()), maximum_SE=float(se.max()),
                  nonpositive_SE_count=int(np.sum(se <= 0)),
                  n_control=2000, n_target=20, variance_ddof=1,
                  min_control_variance=float(cv.min()), scale=float(scale),
                  raw_totals_exact=True, full_gene_axis_exact=True,
                  control_mean_max_abs_error=center_delta,
                  control_variance_max_abs_error=variance_delta,
                  control_trace_max_abs_error=trace_delta,
                  prediction_axes_exact=True)

    np.savez_compressed(output / 'full_effect_stats.npz',
        effects=effects, standard_errors=se, target_var=target_var,
        gene_ids=genes, target_ids=targets, query_keep_columns=qi,
        control_mean=cm, control_mean_raw=cm_raw, control_var=cv, scale=scale,
        n_control=np.array(2000), n_target=np.array(20), variance_ddof=np.array(1))
    for name, idx in [('full', np.arange(n)), ('smoke32', np.arange(min(32, n)))]:
        np.savez_compressed(output / (name + '.npz'), effects=effects[idx],
                            ses=se[idx], row_ids=targets[idx], genes=genes)
    output_hashes = {p.name: sha(p) for p in output.glob('*.npz')}
    for path, expected in source_hashes.items():
        assert sha(path) == expected, ('Source changed during derivation', path)
    stopped.set()
    sampler.join()
    memory_samples.append(current_memory())
    report = dict(status='VALIDATED_NEW_FULL_STATS', task=TASK, rep=0,
                  targets=n, genes=g, output_directory=str(output),
                  source_input_directory=str(source),
                  counts_sha256=source_hashes[str(source / 'counts.npz')],
                  axes_sha256=source_hashes[str(source / 'axes.npz')],
                  full_npz_sha256=output_hashes['full.npz'],
                  source_input_audit_filename='PREPARATION.json',
                  source_input_audit_sha256=audit_sha,
                  source_files_sha256=source_hashes, output_files_sha256=output_hashes,
                  historical_expected_sha256={}, historical_hash_matches={},
                  historical_full_stats_hash_available=False,
                  historical_stats_hash_note='No legacy independent full_effect_stats.npz or ASH full.npz hash exists in the restored inputs; these are newly derived statistics validated against the frozen input prediction.',
                  checks=checks, source_audit_unchanged=True,
                  source_files_unchanged=True, source_input_audit=source_audit,
                  reference_score_computed=False, reference_expression_read=False,
                  qc_expression_read=False, D1_D2_expression_read=False,
                  arithmetic_sources=['independent/execution/fit_prediction.py',
                                      'independent/execution/vendor/frozen_v20.py',
                                      'prepare_independent_input.py',
                                      'ash_work/prepare_full_stats.py'],
                  effect_center='Frozen independent-guide input-control center, with original float32 control rounding retained.',
                  standard_errors='sqrt(target sample variance/20 + frozen input-control sample variance/2000), ddof=1',
                  resources=dict(wall_seconds=time.monotonic() - start,
                                 lock_wait_seconds=lock_acquired - start,
                                 headroom_wait_seconds=compute_start - lock_acquired,
                                 compute_and_save_seconds=time.monotonic() - compute_start,
                                 process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                 cgroup_peak_observed_bytes=max(memory_samples),
                                 cgroup_limit_bytes=limit,
                                 memory_gate=gate,
                                 memory_gate_basis='limit - current + max(0, inactive_file - file_dirty - file_writeback)',
                                 heavy_memory_lock=str(ROOT / 'heavy_memory.lock'),
                                 minimum_headroom_bytes=minimum, numerical_threads=1))
    (output / 'RECOVERY_INPUT_AUDIT.json').write_text(json.dumps(report, indent=2) + '\n')
    # Release clean input/output page cache before the next worker's gate.
    for path in [source / 'counts.npz', *output.glob('*.npz')]:
        with path.open('rb') as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    emit(phase='RECOVERY_COMPLETE', **{k: report[k] for k in
         ['status', 'task', 'rep', 'output_directory', 'historical_hash_matches', 'resources']})


if __name__ == '__main__':
    main()
