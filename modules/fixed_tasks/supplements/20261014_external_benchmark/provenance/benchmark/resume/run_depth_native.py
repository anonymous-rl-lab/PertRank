"""One native FR fit for the separately frozen reduced-target depth diagnostic.

Run once per tool execution with a fresh UUID output root. After this owner exits
zero, a separate tool execution must check NPZ CRCs and SHA256 values and write
OWNER_COMPLETION.json before any downstream scoring or publication.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VENDOR = ROOT / 'vendor_work'
DEPTH = ROOT / 'posthoc_depth_20261014'
PROTOCOL_SHA = 'bd529ba31eaf41cfff522716835d2f95aedc61bfd7a3fdae68e4140e7e6bb546'
ENV = os.environ.copy()
ENV.update(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMBA_NUM_THREADS='2')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_record(path, ledger):
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(ledger, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DEPTH / 'inputs')
    parser.add_argument('--curve', required=True,
                        choices=('A_native_default_penalty', 'B_effective_penalty_sensitivity'))
    parser.add_argument('--depth', type=int, choices=(20, 50, 100), required=True)
    parser.add_argument('--output-root', type=Path, required=True,
                        help='New directory whose basename ends with a canonical UUID; must not exist.')
    args = parser.parse_args()
    supplied_root = args.output_root.expanduser()
    if supplied_root.exists() or supplied_root.is_symlink():
        parser.error('--output-root already exists; use a fresh UUID output root')
    suffix = supplied_root.name[-36:]
    try:
        parsed_uuid = uuid.UUID(suffix)
    except ValueError:
        parser.error('--output-root basename must end with a canonical UUID')
    if str(parsed_uuid) != suffix:
        parser.error('--output-root basename must end with a canonical lowercase UUID')
    args.output_root = supplied_root.resolve()
    args.input_root = args.input_root.expanduser().resolve()
    args.run_uuid = str(parsed_uuid)
    return args


def main():
    args = parse_args()
    protocol_path = DEPTH / 'POSTHOC_DEPTH_PROTOCOL.json'
    if sha256(protocol_path) != PROTOCOL_SHA:
        raise AssertionError('Frozen depth protocol SHA256 changed')
    protocol = json.loads(protocol_path.read_text())
    if args.curve not in protocol['FR']['curves'] or args.depth not in protocol['depths']:
        raise ValueError('Requested curve and depth must belong to the frozen protocol')
    winner_lock = ROOT / 'FR_D3_WINNER_LOCK.json'
    winner_sha256 = sha256(winner_lock)
    settings = protocol['FR']['curves'][args.curve]
    source = args.input_root / ('n' + str(args.depth))
    alpha = settings['lasso_alpha_by_depth'][str(args.depth)]
    output_root = args.output_root
    # Atomic directory creation also rejects an output root created after parsing.
    output_root.mkdir(parents=True, exist_ok=False)
    fit = output_root / 'native_fr_outputs' / args.curve / ('n' + str(args.depth))
    compact = output_root / 'native_fr_compact' / args.curve / ('n' + str(args.depth))
    record = output_root / 'NATIVE_FR_EXECUTION.json'
    process_log = fit.parent / ('n' + str(args.depth) + '_process.log')
    audit_log = fit.parent / ('n' + str(args.depth) + '_audit.log')
    compact_log = fit.parent / ('n' + str(args.depth) + '_compact.log')
    cmd = [sys.executable, str(VENDOR / 'run_prepared_input_atomic_v3.py'),
           '--input-dir', str(source), '--out', str(fit), '--rank', str(protocol['FR']['rank']),
           '--spca-alpha', str(protocol['FR']['spca_alpha']), '--lasso-alpha', str(alpha),
           '--spca-iter', str(protocol['FR']['spca_iterations']), '--threads', '2',
           '--seed', str(protocol['FR']['seed'])]
    audit_cmd = [sys.executable, str(VENDOR / 'add_native_audit_certificate.py'),
                 '--fit-dir', str(fit), '--axes', str(source / 'axes.npz')]
    compact_cmd = [sys.executable, str(VENDOR / 'export_compact_fit_atomic_v3.py'),
                   '--fit-dir', str(fit), '--axes', str(source / 'axes.npz'), '--out', str(compact)]
    entry = dict(curve=args.curve, depth=args.depth, lasso_alpha=alpha, status='RUNNING',
                 stage='INPUT_AXIS_CHECK', command=cmd, audit_command=audit_cmd, compact_command=compact_cmd,
                 start_unix=time.time(), input_directory=str(source), fit_directory=str(fit),
                 compact_directory=str(compact), process_log=str(process_log), audit_log=str(audit_log),
                 compact_log=str(compact_log), timeout_per_fit_seconds=protocol['execution']['max_fit_seconds'])
    ledger = dict(schema_version=1, status='RUNNING', analysis='POSTHOC_REDUCED_TARGET_DEPTH_DIAGNOSTIC',
                  original_winner_unchanged=True, protocol_path=str(protocol_path), protocol_sha256=PROTOCOL_SHA,
                  winner_lock_path=str(winner_lock), winner_lock_sha256=winner_sha256,
                  driver_path=str(Path(__file__).resolve()), driver_sha256=sha256(__file__),
                  output_root=str(output_root), run_uuid=args.run_uuid, owner_pid=os.getpid(),
                  start_unix=entry['start_unix'], fits=[entry],
                  owner_verification=dict(status='PENDING',
                      owner_completion_path=str(output_root / 'OWNER_COMPLETION.json'),
                      separate_exec_after_owner_exit_zero=True,
                      required_before_scoring_or_publication=True,
                      required_checks=['NPZ_CRC', 'SHA256']),
                  inference_scope=protocol['scope'] + '; ' + protocol['inference_limits'])
    write_record(record, ledger)
    try:
        with np.load(source / 'axes.npz', allow_pickle=False) as axes:
            labels = axes['labels']
            targets = axes['target_ids']
            genes = axes['gene_ids']
        if len(targets) != 250 or len(genes) != 18130:
            raise AssertionError('Frozen depth target or gene axis size differs')
        if len(labels) != 250 * args.depth + 2000 or (labels == -1).sum() != 2000:
            raise AssertionError('Frozen depth input or control count differs')
        if not np.all(np.bincount(labels[labels >= 0], minlength=250) == args.depth):
            raise AssertionError('Input target depth differs from requested frozen depth')
        fit.parent.mkdir(parents=True, exist_ok=True)
        entry['stage'] = 'NATIVE_FIT'
        write_record(record, ledger)
        with process_log.open('w') as log:
            subprocess.run(cmd, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=entry['timeout_per_fit_seconds'])
        entry['stage'] = 'NATIVE_AUDIT_AND_COMPACT'
        write_record(record, ledger)
        with (ROOT / 'heavy_memory.lock').open('a') as memory_lock:
            fcntl.flock(memory_lock, fcntl.LOCK_EX)
            with audit_log.open('w') as log:
                subprocess.run(audit_cmd, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
            with compact_log.open('w') as log:
                subprocess.run(compact_cmd, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
            shutil.copyfile(fit / 'NATIVE_SOLVER_OBJECTIVE_AUDIT.json',
                            compact / 'NATIVE_SOLVER_OBJECTIVE_AUDIT.json')
        report = json.loads((fit / 'fit_report.json').read_text())
        manifest = json.loads((compact / 'COMPACT_MANIFEST.json').read_text())
        certificate = json.loads((fit / 'ATOMIC_SERIALIZATION_INPUT_STABILITY.json').read_text())
        if sha256(winner_lock) != winner_sha256:
            raise AssertionError('Immutable D3 winner lock changed during execution')
        if sha256(protocol_path) != PROTOCOL_SHA:
            raise AssertionError('Frozen depth protocol changed during execution')
        artifact_hashes = {
            str(source / name): digest for name, digest in certificate['after'].items()
        }
        artifact_hashes.update({
            str(fit / 'effects.npz'): report['effects_sha256'],
            str(fit / 'native_unscaled.npz'): report['native_unscaled_sha256'],
            str(compact / 'compact.npz'): manifest['compact_sha256'],
        })
        for directory, names in [(fit, ['fit_report.json', 'NATIVE_SOLVER_OBJECTIVE_AUDIT.json',
                                        'ATOMIC_SERIALIZATION_INPUT_STABILITY.json']),
                                 (compact, ['fit_report.json', 'COMPACT_MANIFEST.json',
                                            'NATIVE_SOLVER_OBJECTIVE_AUDIT.json',
                                            'ATOMIC_SERIALIZATION_INPUT_STABILITY.json'])]:
            for name in names:
                artifact_hashes[str(directory / name)] = sha256(directory / name)
        entry.update(status='FIT_AUDIT_COMPACT_COMPLETE', stage='AWAITING_OWNER_VERIFICATION',
                     compact_sha256=manifest['compact_sha256'], artifact_sha256=artifact_hashes,
                     runtime_seconds=report['runtime_seconds'], max_rss_kib=report['max_rss_kib'],
                     audit_pass=report['recover_audit']['acceptance_audit']['pass_'],
                     wall_seconds=time.time() - entry['start_unix'])
        ledger.update(status='AWAITING_OWNER_VERIFICATION', completed_unix=time.time())
        write_record(record, ledger)
    except BaseException as exc:
        entry.update(status='RESOURCE_OR_EXECUTION_FAILURE_NOT_SCORED', error=repr(exc),
                     wall_seconds=time.time() - entry['start_unix'])
        ledger.update(status='INCOMPLETE', completed_unix=time.time())
        write_record(record, ledger)
        raise
    print('DEPTH_NATIVE_FIT_AUDIT_COMPACT_COMPLETE ' + json.dumps({
        'curve': args.curve, 'depth': args.depth, 'status': ledger['status'],
        'execution_metadata': str(record), 'output_root': str(output_root),
        'owner_verification_required': True}), flush=True)


if __name__ == '__main__':
    main()
