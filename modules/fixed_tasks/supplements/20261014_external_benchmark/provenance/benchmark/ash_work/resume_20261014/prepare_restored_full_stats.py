#!/usr/bin/env python3
"""Verify and reuse restored D4 Stim sufficient statistics for ASH inputs.

All three source NPZ hashes and the frozen selection hash are checked against
the original input audit. Counts are hashed but never loaded or recomputed.
The source full-statistics file is copied unchanged; ASH inputs use the original
prepare_full_stats.py key order and row indexing. Sources remain read-only.
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
import shutil
import threading
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CGROUP = Path('/sys/fs/cgroup')
FULL_STATS_KEYS = (
    'effects', 'standard_errors', 'target_var', 'gene_ids', 'target_ids',
    'query_keep_columns', 'control_mean', 'control_mean_raw', 'control_var',
    'scale', 'n_control', 'n_target', 'variance_ddof')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def emit(**entry):
    print(json.dumps(entry, sort_keys=True), flush=True)


def current_memory():
    return int((CGROUP / 'memory.current').read_text())


def memory_headroom(limit):
    stat = dict(line.split() for line in (CGROUP / 'memory.stat').read_text().splitlines())
    current = current_memory()
    inactive_file = int(stat['inactive_file'])
    dirty = int(stat['file_dirty'])
    writeback = int(stat['file_writeback'])
    clean_inactive_file = max(0, inactive_file - dirty - writeback)
    raw_headroom = None if limit is None else limit - current
    return dict(current_bytes=current, limit_bytes=limit,
                raw_headroom_bytes=raw_headroom,
                inactive_file_bytes=inactive_file, file_dirty_bytes=dirty,
                file_writeback_bytes=writeback,
                dirty_and_writeback_bytes=dirty + writeback,
                reclaimable_clean_inactive_file_bytes=clean_inactive_file,
                effective_headroom_bytes=None if limit is None else raw_headroom + clean_inactive_file)


def release_cache(path):
    with Path(path).open('rb') as handle:
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['D4_Stim8hr'], required=True)
    parser.add_argument('--rep', type=int, choices=range(5), required=True)
    parser.add_argument('--output-root', type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument('--minimum-headroom-gib', type=float, default=1.6)
    args = parser.parse_args()
    if args.minimum_headroom_gib <= 0:
        parser.error('--minimum-headroom-gib must be positive')
    start = time.monotonic()
    source = ROOT / 'inputs' / args.task / f'rep{args.rep}'
    output = args.output_root / args.task / f'rep{args.rep}'
    output.mkdir(parents=True, exist_ok=False)
    audit_path = source / 'INPUT_AUDIT.json'
    audit_sha = sha(audit_path)
    source_audit = json.loads(audit_path.read_text())
    assert source_audit['status'] == 'EXACT_RAW_REPLAY_PASS'
    assert source_audit['task'] == args.task and source_audit['rep'] == args.rep
    assert source_audit['control_count'] == 2000
    source_hashes = {str(audit_path): audit_sha}
    input_names = ('counts.npz', 'axes.npz', 'full_effect_stats.npz')
    assert set(source_audit['files_sha256']) == set(input_names)
    for name in input_names:
        actual = sha(source / name)
        assert actual == source_audit['files_sha256'][name], (name, actual)
        source_hashes[str(source / name)] = actual
    # Counts will not be loaded; release their clean verification cache now.
    release_cache(source / 'counts.npz')
    selection_path = ROOT / 'pooled_data' / args.task / 'selection.npz'
    selection_sha = sha(selection_path)
    assert selection_sha == source_audit['source_selection_sha256']
    source_hashes[str(selection_path)] = selection_sha

    memory_samples = [current_memory()]
    stopped = threading.Event()

    def monitor():
        while not stopped.wait(0.5):
            memory_samples.append(current_memory())

    sampler = threading.Thread(target=monitor, daemon=True)
    sampler.start()
    lock = (ROOT / 'heavy_memory.lock').open('a')
    emit(phase='WAIT_FOR_HEAVY_MEMORY_LOCK', task=args.task, rep=args.rep)
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    lock_acquired = time.monotonic()
    limit_text = (CGROUP / 'memory.max').read_text().strip()
    limit = None if limit_text == 'max' else int(limit_text)
    minimum = int(args.minimum_headroom_gib * (1 << 30))
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
    emit(phase='REUSE_FULL_STATS_START', task=args.task, rep=args.rep,
         cgroup_current_bytes=current_memory(), cgroup_limit_bytes=limit,
         reuse_existing_full_stats=True, raw_counts_recomputed=False)

    stats_path = source / 'full_effect_stats.npz'
    with np.load(stats_path, allow_pickle=False) as archive:
        assert tuple(archive.files) == FULL_STATS_KEYS, archive.files
        stats = {name: archive[name] for name in FULL_STATS_KEYS}
    with np.load(source / 'axes.npz', allow_pickle=False) as archive:
        axes = {name: archive[name] for name in (
            'labels', 'gene_ids', 'target_ids', 'cell_ids', 'query_ids',
            'query_columns', 'input_effects', 'rep', 'task', 'scale',
            'input_control_count', 'target_cells_per_group', 'all_query_ids',
            'all_query_columns', 'all_input_effects')}
    # Read only selection identities, never any expression or reference arrays.
    with np.load(selection_path, allow_pickle=False) as archive:
        selection = {name: archive[name] for name in (
            'gene_ids_full', 'target_ids', 'NTC_input', 'input_cells')}

    y = stats['effects']
    se = stats['standard_errors']
    target_var = stats['target_var']
    genes = stats['gene_ids']
    targets = stats['target_ids']
    qi = stats['query_keep_columns']
    cm = stats['control_mean']
    cm_raw = stats['control_mean_raw']
    cv = stats['control_var']
    assert y.ndim == 2
    n, g = y.shape
    assert se.shape == target_var.shape == y.shape
    assert genes.shape == (g,) and targets.shape == (n,)
    assert cm.shape == cm_raw.shape == cv.shape == (g,)
    for name in ('effects', 'standard_errors', 'target_var', 'control_mean',
                 'control_mean_raw', 'control_var', 'scale'):
        assert stats[name].dtype == np.dtype('float64'), (name, stats[name].dtype)
        assert np.isfinite(stats[name]).all(), name
    assert np.all(se > 0), 'Nonpositive SE; no automatic floor'
    assert np.all(target_var >= 0) and np.all(cv >= 0)
    assert len(np.unique(genes)) == g and len(np.unique(targets)) == n
    assert genes.dtype.kind == targets.dtype.kind == 'U'
    for name, value in (('n_control', 2000), ('n_target', 20), ('variance_ddof', 1)):
        assert stats[name].shape == () and stats[name].dtype == np.dtype('int64')
        assert int(stats[name]) == value
    assert stats['scale'].shape == () and float(stats['scale']) > 0
    scale = float(stats['scale'])
    assert qi.ndim == 1 and qi.dtype == np.dtype('int64')
    assert np.all(qi >= 0) and np.all(qi < g) and len(np.unique(qi)) == len(qi)
    assert str(axes['task']) == args.task and int(axes['rep']) == args.rep
    assert int(axes['input_control_count']) == 2000
    assert int(axes['target_cells_per_group']) == 20
    assert float(axes['scale']) == scale
    assert n == source_audit['targets'] and g == source_audit['weight_genes']
    assert len(qi) == source_audit['queries']
    assert len(axes['gene_ids']) == source_audit['genes']
    assert np.array_equal(targets, axes['target_ids'])
    assert np.array_equal(targets, selection['target_ids'])
    assert np.array_equal(axes['gene_ids'], selection['gene_ids_full'])
    assert len(np.unique(axes['gene_ids'])) == len(axes['gene_ids'])
    full_gene_positions = {str(gene): i for i, gene in enumerate(axes['gene_ids'])}
    keep = np.array([full_gene_positions[str(gene)] for gene in genes], dtype=np.int64)
    assert np.all(np.diff(keep) > 0)
    assert np.array_equal(genes[qi], axes['query_ids'])
    assert np.array_equal(keep[qi], axes['query_columns'])
    kept_gene_positions = {str(gene): i for i, gene in enumerate(genes)}
    all_qi = np.array([kept_gene_positions[str(gene)]
                       for gene in axes['all_query_ids']], dtype=np.int64)
    assert np.array_equal(keep[all_qi], axes['all_query_columns'])
    assert axes['input_effects'].shape == (n, len(qi))
    assert axes['all_input_effects'].shape == (n, len(all_qi))
    qerror = float(np.max(np.abs(y[:, qi] - axes['input_effects'])))
    aqerror = float(np.max(np.abs(y[:, all_qi] - axes['all_input_effects'])))
    assert qerror < 1e-12 and aqerror < 1e-12, (qerror, aqerror)
    expected_labels = np.r_[np.full(2000, -1, dtype=np.int32),
                            np.repeat(np.arange(n, dtype=np.int32), 20)]
    assert np.array_equal(axes['labels'], expected_labels)
    assert selection['NTC_input'].shape == (2000,)
    assert selection['input_cells'].shape == (n, 5, 20)
    assert np.array_equal(axes['cell_ids'], np.r_[selection['NTC_input'],
                          selection['input_cells'][:, args.rep, :].ravel()])
    assert len(axes['cell_ids']) == source_audit['cells'] == 2000 + n * 20
    se_identity_error = float(np.max(np.abs(se - np.sqrt(target_var / 20 + cv / 2000))))
    assert se_identity_error < 1e-12, se_identity_error
    checks = dict(query_effect_max_abs_error=qerror,
                  all_query_effect_max_abs_error=aqerror,
                  minimum_SE=float(se.min()), maximum_SE=float(se.max()),
                  nonpositive_SE_count=int(np.sum(se <= 0)),
                  n_control=2000, n_target=20, variance_ddof=1,
                  min_control_variance=float(cv.min()), scale=scale,
                  standard_error_identity_max_abs_error=se_identity_error,
                  full_gene_axis_exact=True, target_axis_exact=True,
                  query_axis_exact=True, cell_axis_exact=True,
                  original_full_stats_sha256_verified=True,
                  raw_totals_exact=source_audit['raw_totals_exact'],
                  raw_totals_validation_source='Original hash-verified INPUT_AUDIT.json; no raw recomputation')

    shutil.copyfile(stats_path, output / 'full_effect_stats.npz')
    for name, idx in [('full', np.arange(n)), ('smoke32', np.arange(min(32, n)))]:
        np.savez_compressed(output / (name + '.npz'), effects=y[idx],
                            ses=se[idx], row_ids=targets[idx], genes=genes)
    output_hashes = {p.name: sha(p) for p in output.glob('*.npz')}
    expected = {'full_effect_stats.npz': source_audit['files_sha256']['full_effect_stats.npz']}
    exact = {name: output_hashes[name] == value for name, value in expected.items()}
    for path, value in source_hashes.items():
        assert sha(path) == value, ('Source changed during preparation', path)
    stopped.set()
    sampler.join()
    memory_samples.append(current_memory())
    report = dict(status='EXACT_HASH_RECOVERY_PASS' if all(exact.values()) else 'HASH_MISMATCH',
                  task=args.task, rep=args.rep, targets=n, genes=g,
                  output_directory=str(output), source_input_directory=str(source),
                  counts_sha256=source_hashes[str(source / 'counts.npz')],
                  axes_sha256=source_hashes[str(source / 'axes.npz')],
                  full_npz_sha256=output_hashes['full.npz'],
                  source_input_audit_filename='INPUT_AUDIT.json',
                  source_input_audit_sha256=audit_sha,
                  source_files_sha256=source_hashes, output_files_sha256=output_hashes,
                  historical_expected_sha256=expected, historical_hash_matches=exact,
                  reuse_existing_full_stats=True, raw_counts_recomputed=False,
                  checks=checks, source_audit_unchanged=True, source_files_unchanged=True,
                  source_input_audit=source_audit, reference_score_computed=False,
                  reference_expression_read=False, D1_D2_expression_read=False,
                  arithmetic_sources=['pooled_data/export_inputs.py',
                                      'ash_work/prepare_full_stats.py'],
                  full_stats_source='Hash-verified original full_effect_stats.npz copied byte-for-byte; no raw statistics recomputed.',
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
    for path in [source / name for name in input_names] + list(output.glob('*.npz')):
        release_cache(path)
    emit(phase='RECOVERY_COMPLETE', **{k: report[k] for k in
         ['status', 'task', 'rep', 'output_directory', 'historical_hash_matches', 'resources']})
    assert all(exact.values()), exact


if __name__ == '__main__':
    main()
