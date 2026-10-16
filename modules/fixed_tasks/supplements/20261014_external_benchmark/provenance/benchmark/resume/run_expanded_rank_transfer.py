#!/usr/bin/env python
"""Conditional posthoc D4 transfer; never score D4 or replace the primary lock.

Default invocation only records the selection; it never fits. --task/--rep runs
exactly one registered fit in its own unique run directory. The parent must wait
for that execution session to exit successfully, then invoke --verify-run in a
later tool call before publishing its artifacts. Existing/failed fit directories
are never overwritten or automatically retried.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VENDOR = ROOT / "vendor_work"
OUTPUT = HERE / "expanded_rank_transfer"
MEMORY_LOCK = ROOT / "heavy_memory.lock"
TOLERANCE = 1e-12
RANKS = [150, 200, 300]
TASKS = ["D4_Rest", "D4_Stim8hr"]
REPS = [0, 1, 2, 3, 4]
RULE_SHA256 = "9925d185212de1d4b71a07d6cfb50b2574f168abe5aa2444e75d2cb017e1bd82"
ENV = os.environ.copy()
ENV.update(OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2",
           MKL_NUM_THREADS="2", NUMBA_NUM_THREADS="2")


class NotReady(RuntimeError):
    """Incomplete D3 sensitivity is not an eligible selection set."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    """Create, or verify identical contents; never replace an existing artifact."""
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        require(path.read_text() == payload, "Existing immutable artifact differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def append_event(event):
    event = {"time_unix": time.time(), **event}
    with (OUTPUT / "EXPANDED_TRANSFER_EXECUTION.jsonl").open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(event, sort_keys=True), flush=True)


def check_immutable():
    rule_path = HERE / "POSTHOC_EXPANDED_RANK_TRANSFER_RULE.json"
    require(sha(rule_path) == RULE_SHA256, "Expanded-transfer rule changed")
    require((HERE / "POSTHOC_EXPANDED_RANK_TRANSFER_RULE.sha256").read_text().split()[0]
            == RULE_SHA256, "Expanded-transfer rule sidecar changed")
    rule = read_json(rule_path)
    protocol_path = HERE / "POSTHOC_D3_RANK_BOUNDARY.json"
    protocol_hash = sha(protocol_path)
    require(protocol_hash == rule["original_rank_boundary_protocol_sha256"],
            "Rank-boundary protocol linkage mismatch")
    require((HERE / "POSTHOC_D3_RANK_BOUNDARY.sha256").read_text().split()[0]
            == protocol_hash, "Rank-boundary sidecar mismatch")
    protocol = read_json(protocol_path)
    winner_hash = rule["original_winner_lock_sha256"]
    for path in (ROOT / "FR_D3_WINNER_LOCK.json", HERE / "FR_D3_WINNER_LOCK.json"):
        require(sha(path) == winner_hash, "Original winner lock changed: " + str(path))
    require(protocol["immutable_winner_lock_sha256"] == winner_hash,
            "Boundary protocol has a different original winner")
    winner = read_json(HERE / "FR_D3_WINNER_LOCK.json")
    require(winner["rank"] == 100 and winner["winner_P25"]
            == rule["original_winner"]["native_fr_P25"], "Original score mismatch")
    require(rule["candidate_ranks"] == RANKS, "Unregistered candidate ranks")
    expected_grid = [dict(id=f"rank{k}_default_lasso", rank=k,
                         spca_alpha=0.1, lasso_alpha=0.0001) for k in RANKS]
    require(protocol["fr_grid"] == expected_grid, "Expanded D3 grid mismatch")
    transfer = rule["transfer"]
    expected = dict(datasets=TASKS, input_reps=REPS, n_additional_fits_if_triggered=10,
                    spca_alpha=0.1, lasso_alpha=0.0001, spca_iter=50,
                    seed_base=20261012, threads=2, independent_guide_extra_fits=False,
                    Nscaled_extra_fits=False)
    require(all(transfer.get(k) == v for k, v in expected.items()),
            "Unregistered transfer settings")
    require(rule["resource_budget"]["per_fit_seconds"] == 900
            and rule["resource_budget"]["maximum_additional_fits"] == 10
            and Path(rule["resource_budget"]["same_heavy_memory_lock"]) == MEMORY_LOCK,
            "Resource budget mismatch")
    return rule, protocol, protocol_hash


def select_candidate(candidates, original_score):
    """Apply the frozen strict trigger, then the smallest-K near-maximum tie rule."""
    require(sorted(x["rank"] for x in candidates) == RANKS, "Need exactly K150/200/300")
    require(all(math.isfinite(x["native_fr_P25"]) for x in candidates), "Nonfinite D3 score")
    maximum = max(x["native_fr_P25"] for x in candidates)
    best = min((x for x in candidates if maximum - x["native_fr_P25"] <= TOLERANCE),
               key=lambda x: x["rank"])
    return best, maximum > original_score + TOLERANCE, maximum


def build_decision():
    rule, protocol, protocol_hash = check_immutable()
    record = HERE / "RANK_BOUNDARY_EXECUTION.json"
    if not record.exists():
        raise NotReady("RANK_BOUNDARY_EXECUTION.json does not yet exist; no winner or fits")
    execution = read_json(record)
    if execution.get("status") != "COMPLETE":
        raise NotReady("All three D3 scores/audits must be COMPLETE; no winner or fits")
    require(execution.get("protocol_sha256") == protocol_hash, "D3 execution protocol mismatch")
    entries = execution.get("fits", [])
    require(len(entries) == 3, "Need exactly three D3 execution entries")
    candidates = []
    for config in protocol["fr_grid"]:
        matches = [entry for entry in entries if entry.get("config") == config]
        require(len(matches) == 1, "Missing/duplicate D3 config: " + config["id"])
        entry = matches[0]
        if entry.get("status") != "COMPLETE" or entry.get("audit_pass") is not True:
            raise NotReady("Incomplete or rejected D3 fit: " + config["id"])
        command = entry.get("command", [])
        default_fit = HERE / "rank_boundary_outputs" / (config["id"] + "_rep0")
        command_fit = command[command.index("--out") + 1] if "--out" in command else default_fit
        fit = Path(entry.get("fit_directory", command_fit)).resolve()
        compact = Path(entry["compact_directory"]).resolve()
        evaluation = Path(entry["evaluation_directory"]).resolve()
        require(all(path.is_relative_to(HERE) for path in (fit, compact, evaluation)),
                "D3 evidence must belong to this append-only continuation")
        report_path = fit / "fit_report.json"
        certificate_path = fit / "NATIVE_SOLVER_OBJECTIVE_AUDIT.json"
        score_path = evaluation / "evaluation_report.json"
        report, certificate, score = map(read_json, (report_path, certificate_path, score_path))
        parameters = dict(rank=config["rank"], spca_alpha=0.1, lasso_alpha=0.0001,
                          spca_iter=50, seed=20261012, threads=2)
        require(report["status"] == "COMPLETE" and report["parameters"] == parameters,
                "D3 native fit incomplete or parameter mismatch")
        d3_inputs = input_hashes("D3_Rest", 0, Path(report["input_provenance"]["input_directory"]))
        require(report["input_provenance"]["counts_sha256"] == d3_inputs["counts.npz"]
                and report["input_provenance"]["axes_sha256"] == d3_inputs["axes.npz"],
                "D3 fit input does not match its verified ready marker")
        require(report["official_commit"] == rule["transfer"]["native_core_commit"],
                "D3 native core commit mismatch")
        require(report["recover_audit"]["acceptance_audit"]["pass_"] is True
                and certificate.get("pass_") is True
                and certificate["original_fit_report_sha256"] == sha(report_path),
                "D3 fit-bound native audit rejected or mismatched")
        require(score["status"] == "COMPLETE" and score["dataset"] == "D3_Rest"
                and score["kind"] == "pooled" and score["config_id"] == config["id"]
                and score["config"] == config and score["input_reps"] == [0],
                "D3 score is not the registered Rest rep0 fit")
        provenance = score["provenance"]
        require(provenance["protocol_sha256"] == protocol_hash, "D3 score protocol mismatch")
        require(len(provenance["fr"]) == 1, "D3 score contains extra FR inputs")
        scored_fit = provenance["fr"][0]
        require(scored_fit["fit_report_sha256"] == sha(report_path)
                and scored_fit["solver_audit_sha256"] == sha(certificate_path)
                and scored_fit["effects_sha256"] == report["effects_sha256"]
                and Path(scored_fit["fit_report_path"]).resolve() == report_path,
                "D3 score is not bound to the audited native fit")
        p25 = score["results"]["25"]["arms"]["fr"]["point"]
        p10 = score["results"]["10"]["arms"]["fr"]["point"]
        require(p25 == entry["native_fr_P25"] and p10 == entry["native_fr_P10"]
                and math.isfinite(p25) and math.isfinite(p10), "D3 ledger score mismatch")
        compact_hash = sha(compact / "compact.npz")
        manifest = read_json(compact / "COMPACT_MANIFEST.json")
        require(compact_hash == entry.get("compact_sha256", compact_hash) == manifest["compact_sha256"],
                "D3 compact hash mismatch")
        require(manifest["all_compact_arrays_bitwise_verified"] is True
                and manifest["source_effects_sha256"] == report["effects_sha256"]
                and sha(compact / "NATIVE_SOLVER_OBJECTIVE_AUDIT.json") == sha(certificate_path),
                "D3 compact lacks validated native provenance")
        owner_path = Path(entry["owner_completion"]).resolve()
        require(owner_path.is_relative_to(HERE), "Owner completion belongs outside this continuation")
        owner = read_json(owner_path)
        require(owner["status"] == "OWNER_EXECUTION_ENDED_INTEGRITY_VERIFIED"
                and owner["owner_exit_code"] == 0 and bool(owner["owner_session_id"])
                and Path(owner["fit_directory"]).resolve() == fit
                and Path(owner["compact_directory"]).resolve() == compact
                and owner["fit_report_sha256"] == sha(report_path)
                and owner["native_solver_objective_audit_sha256"] == sha(certificate_path)
                and owner["input_stability_sha256"] == sha(fit / "ATOMIC_SERIALIZATION_INPUT_STABILITY.json"),
                "D3 owner completion is missing or not bound to this native fit")
        expected_npz = {"effects.npz": report["effects_sha256"],
                        "native_unscaled.npz": report["native_unscaled_sha256"],
                        "compact.npz": compact_hash}
        require(all(owner["npz_files"][name]["sha256"] == digest
                    and owner["npz_files"][name]["npz_crc_pass"] is True
                    for name, digest in expected_npz.items()), "D3 owner NPZ integrity checks mismatch")
        candidates.append(dict(config_id=config["id"], rank=config["rank"],
            native_fr_P25=p25, native_fr_P10=p10, audit_pass=True,
            score_file=str(score_path), score_sha256=sha(score_path),
            fit_report_sha256=sha(report_path), certificate_sha256=sha(certificate_path),
            compact_sha256=compact_hash, owner_completion=str(owner_path),
            owner_completion_sha256=sha(owner_path)))
    original = rule["original_winner"]["native_fr_P25"]
    best, triggered, maximum = select_candidate(candidates, original)
    return dict(status="LOCKED_POSTHOC_TRANSFER" if triggered else "NO_RUN_TRIGGER_FALSE",
                analysis_label=rule["analysis_label"], transfer_triggered=triggered,
                config_id=best["config_id"], rank=best["rank"], spca_alpha=0.1,
                lasso_alpha=0.0001, spca_iter=50, seed_base=20261012, threads=2,
                winner_P25=best["native_fr_P25"], maximum_expanded_P25=maximum,
                original_rank100_P25=original, strict_trigger_tolerance=TOLERANCE,
                tie_rule="Within 1e-12 of maximum, choose smallest K",
                expected_fits=10 if triggered else 0, candidate_evidence=candidates,
                rule_sha256=RULE_SHA256, boundary_protocol_sha256=protocol_hash,
                boundary_execution_sha256=sha(record),
                original_winner_lock_sha256=rule["original_winner_lock_sha256"],
                original_primary_unchanged=True, D4_scores_read=False,
                inference_scope=rule["inference"])


def save_decision(decision):
    path = OUTPUT / "POSTHOC_EXPANDED_RANK_WINNER.json"
    write_once(path, decision)
    sidecar = path.with_suffix(".sha256")
    payload = sha(path) + "  " + path.name + "\n"
    if sidecar.exists():
        require(sidecar.read_text() == payload, "Expanded winner hash mismatch")
    else:
        with sidecar.open("x") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    return sha(path)


def sources():
    paths = [Path(__file__), Path(sys.executable).resolve(),
             VENDOR / "run_prepared_input_audit_v2.py",
             VENDOR / "run_prepared_input_atomic_v3.py", VENDOR / "atomic_npz.py",
             VENDOR / "fr_native_runner_audit_v2.py", VENDOR / "native_recovery_audit.py",
             VENDOR / "add_native_audit_certificate.py", VENDOR / "export_compact_fit.py",
             VENDOR / "export_compact_fit_atomic_v3.py",
             VENDOR / "fr_native_runner.py", VENDOR / "official_manifest.json"]
    manifest = read_json(VENDOR / "official_manifest.json")
    require(manifest["commit"] == "24e653b6a5c050ab17f3600ebaf222bef32e2d25",
            "Native manifest commit mismatch")
    for relative, expected in manifest["files"].items():
        path = VENDOR / "FR-Perturb" / relative
        require(sha(path) == expected, "Native source changed: " + relative)
        paths.append(path)
    return {str(path): sha(path) for path in paths}


def input_hashes(task, rep, source=None):
    source = Path(source or ROOT / "inputs" / task / f"rep{rep}").resolve()
    require(source.is_relative_to(ROOT), "Prepared inputs must belong to this benchmark")
    audit_path = source / "INPUT_AUDIT.json"
    if not audit_path.exists():
        audit_path = source / "PREPARATION.json"
    audit = read_json(audit_path)
    allowed_status = {"EXACT_RAW_REPLAY_PASS", "COMPLETE", "PASS"}
    if task == "D3_Rest":
        allowed_status.add("EXACT_RAW_R20_REPLAY_PASS")
    require(audit.get("status", "COMPLETE") in allowed_status
            and audit.get("task", audit.get("dataset")) == task and audit["rep"] == rep
            and audit.get("control_count", 2000) == 2000,
            "Frozen input audit mismatch")
    hashes = {name: sha(source / name) for name in ("counts.npz", "axes.npz")}
    expected = audit.get("files_sha256") or {"counts.npz": audit.get("counts_sha256"),
                                            "axes.npz": audit.get("axes_sha256")}
    require(all(expected[name] == digest for name, digest in hashes.items()),
            "Frozen input bytes changed: " + str(source))
    hashes[audit_path.name] = sha(audit_path)
    return hashes


def run_process(command, log_path, timeout=900):
    """Bound the child and its descendants; keep the log and partial files on failure."""
    started = time.monotonic()
    with Path(log_path).open("x") as log:
        process = subprocess.Popen(command, env=ENV, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)
    return time.monotonic() - started


def verify_complete(fit, compact, parameters, inputs):
    report = read_json(fit / "fit_report.json")
    certificate = read_json(fit / "NATIVE_SOLVER_OBJECTIVE_AUDIT.json")
    manifest = read_json(compact / "COMPACT_MANIFEST.json")
    stability = read_json(fit / "ATOMIC_SERIALIZATION_INPUT_STABILITY.json")
    expected_inputs = {name: inputs[name] for name in ("counts.npz", "axes.npz")}
    require(stability["status"] == "PASS" and stability["before"] == expected_inputs
            and stability["after"] == expected_inputs
            and stability["wrapper_sha256"] == sha(VENDOR / "run_prepared_input_atomic_v3.py")
            and stability["helper_sha256"] == sha(VENDOR / "atomic_npz.py"),
            "Atomic serialization/input stability guard failed or changed")
    require(report["status"] == "COMPLETE" and report["parameters"] == parameters,
            "Transfer report status/parameters mismatch")
    require(report["official_commit"] == "24e653b6a5c050ab17f3600ebaf222bef32e2d25"
            and report["runner_sha256"] == sha(VENDOR / "fr_native_runner_audit_v2.py"),
            "Transfer did not use the unchanged native audit-v2 runner")
    require(report["input_provenance"]["counts_sha256"] == inputs["counts.npz"]
            and report["input_provenance"]["axes_sha256"] == inputs["axes.npz"],
            "Transfer input hash mismatch")
    require(report["recover_audit"]["acceptance_audit"]["pass_"] is True
            and certificate["pass_"] is True
            and certificate["original_fit_report_sha256"] == sha(fit / "fit_report.json")
            and certificate["audit_source_sha256"] == sha(VENDOR / "native_recovery_audit.py"),
            "Transfer native audit rejected/mismatched")
    require(sha(fit / "effects.npz") == report["effects_sha256"]
            and sha(fit / "native_unscaled.npz") == report["native_unscaled_sha256"],
            "Native fit output hash mismatch")
    require(sha(compact / "compact.npz") == manifest["compact_sha256"]
            and manifest["source_effects_sha256"] == report["effects_sha256"]
            and manifest["source_native_unscaled_sha256"] == report["native_unscaled_sha256"]
            and manifest["source_axes_sha256"] == inputs["axes.npz"]
            and manifest["all_compact_arrays_bitwise_verified"] is True,
            "Transfer compact provenance mismatch")
    require(sha(compact / "NATIVE_SOLVER_OBJECTIVE_AUDIT.json")
            == sha(fit / "NATIVE_SOLVER_OBJECTIVE_AUDIT.json"), "Compact certificate mismatch")
    return dict(runtime_seconds=report["runtime_seconds"], max_rss_kib=report["max_rss_kib"],
                fit_report_sha256=sha(fit / "fit_report.json"),
                effects_sha256=report["effects_sha256"],
                native_unscaled_sha256=report["native_unscaled_sha256"],
                compact_sha256=manifest["compact_sha256"],
                certificate_sha256=sha(fit / "NATIVE_SOLVER_OBJECTIVE_AUDIT.json"),
                serialization_stability_sha256=sha(fit / "ATOMIC_SERIALIZATION_INPUT_STABILITY.json"),
                audit_pass=True, versions=report["versions"],
                official_source_sha256=report["official_source_sha256"])


def run_job(task, rep, decision, winner_hash, source_hashes, inputs, run_id, source):
    name = decision["config_id"] + f"_rep{rep}"
    run_root = OUTPUT / "runs" / run_id
    fit = run_root / "fr_outputs" / task / name
    compact = run_root / "fr_compact" / task / name
    record = run_root / "records" / task / name
    final_path = record / "FINAL.json"
    parameters = dict(rank=decision["rank"], spca_alpha=0.1, lasso_alpha=0.0001,
                      spca_iter=50, seed=20261012 + rep, threads=2)
    if final_path.exists():
        final = read_json(final_path)
        require(final["expanded_winner_sha256"] == winner_hash
                and final["input_hashes"] == inputs, "Existing job provenance mismatch")
        if final["status"] == "COMPLETE_PENDING_EXTERNAL_VERIFICATION":
            verified = verify_complete(fit, compact, parameters, inputs)
            require(all(final[key] == value for key, value in verified.items()),
                    "Existing complete job changed")
        append_event(dict(event="PRESERVED_EXISTING_JOB", task=task, rep=rep,
                          status=final["status"], final_sha256=sha(final_path)))
        return final
    command = [sys.executable, str(VENDOR / "run_prepared_input_atomic_v3.py"),
               "--input-dir", str(source),
               "--out", str(fit), "--rank", str(parameters["rank"]),
               "--spca-alpha", "0.1", "--lasso-alpha", "0.0001", "--spca-iter", "50",
               "--threads", "2", "--seed", str(parameters["seed"])]
    entry = dict(task=task, rep=rep, run_id=run_id, analysis_label=decision["analysis_label"],
                 config_id=decision["config_id"], parameters=parameters,
                 input_hashes=inputs, source_sha256=source_hashes,
                 input_directory=str(source),
                 expanded_winner_sha256=winner_hash, rule_sha256=RULE_SHA256,
                 fit_directory=str(fit), compact_directory=str(compact), command=command,
                 timeout_per_fit_seconds=900, timeout_includes_child_lock_wait=True,
                 heavy_memory_lock=str(MEMORY_LOCK), D4_scoring_performed=False)
    if record.exists() or fit.exists() or compact.exists():
        entry.update(status="INCOMPLETE_PRESERVED_INTERRUPTED_JOB",
                     error="Existing output/start record has no final record; no automatic refit")
        write_once(final_path, entry)
        append_event(dict(event="JOB_INCOMPLETE", task=task, rep=rep, status=entry["status"]))
        return entry
    write_once(record / "START.json", {**entry, "status": "RUNNING", "start_unix": time.time()})
    append_event(dict(event="JOB_STARTED", task=task, rep=rep, command=command))
    started = time.monotonic()
    phase = "native_fit"
    try:
        check_immutable()
        require(sources() == source_hashes, "Native source/runtime changed since preflight")
        require(input_hashes(task, rep, source) == inputs, "Input changed since preflight")
        fit.parent.mkdir(parents=True, exist_ok=True)
        entry["native_child_wall_seconds"] = run_process(command, record / "native_process.log")
        phase = "certificate_and_compact"
        # The native wrapper holds this lock itself. Acquire only after that child
        # exits, then hold continuously across certificate and compact export.
        with MEMORY_LOCK.open("a") as memory_lock:
            lock_started = time.monotonic()
            fcntl.flock(memory_lock, fcntl.LOCK_EX)
            entry["postprocess_lock_wait_seconds"] = time.monotonic() - lock_started
            axes = source / "axes.npz"
            certify = [sys.executable, str(VENDOR / "add_native_audit_certificate.py"),
                       "--fit-dir", str(fit), "--axes", str(axes)]
            export = [sys.executable, str(VENDOR / "export_compact_fit_atomic_v3.py"),
                      "--fit-dir", str(fit), "--axes", str(axes), "--out", str(compact)]
            entry["certificate_wall_seconds"] = run_process(certify, record / "certificate_process.log")
            entry["compact_wall_seconds"] = run_process(export, record / "compact_process.log")
            with (compact / "NATIVE_SOLVER_OBJECTIVE_AUDIT.json").open("xb") as stream:
                stream.write((fit / "NATIVE_SOLVER_OBJECTIVE_AUDIT.json").read_bytes())
        phase = "verify"
        entry.update(verify_complete(fit, compact, parameters, inputs))
        check_immutable()
        entry["status"] = "COMPLETE_PENDING_EXTERNAL_VERIFICATION"
    except BaseException as exc:
        entry.update(status="RESOURCE_OR_EXECUTION_FAILURE_NOT_SCORED", phase=phase,
                     error=repr(exc), failure_preserved=True, automatic_retry=False)
        report_path = fit / "fit_report.json"
        if report_path.exists():
            entry["partial_fit_report_sha256"] = sha(report_path)
        entry["wall_seconds"] = time.monotonic() - started
        write_once(final_path, entry)
        append_event(dict(event="JOB_FAILED", task=task, rep=rep, phase=phase, error=repr(exc)))
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return entry
    entry["wall_seconds"] = time.monotonic() - started
    write_once(final_path, entry)
    append_event(dict(event="JOB_EXIT_READY_FOR_NEXT_CALL_VERIFICATION", task=task, rep=rep,
                      run_id=run_id, publishable=False,
                      compact_sha256=entry["compact_sha256"], audit_pass=True,
                      runtime_seconds=entry["runtime_seconds"], max_rss_kib=entry["max_rss_kib"]))
    return entry


def safe_run_id(run_id):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", run_id) is not None,
            "run-id must contain only 1-96 letters, digits, dots, underscores or hyphens")
    return run_id


def reserve_job(task, rep, requested_run_id, winner_hash):
    path = OUTPUT / "jobs" / task / f"rep{rep}.json"
    if path.exists():
        reservation = read_json(path)
        require(reservation["expanded_winner_sha256"] == winner_hash,
                "Existing job belongs to a different expanded winner")
        require(requested_run_id is None or reservation["run_id"] == requested_run_id,
                "This fixed job is already reserved; no second run-id or automatic retry")
        return reservation["run_id"]
    run_id = safe_run_id(requested_run_id or uuid.uuid4().hex)
    require(not (OUTPUT / "runs" / run_id).exists(), "Run directory already exists")
    for other in (OUTPUT / "jobs").glob("*/rep*.json"):
        require(read_json(other)["run_id"] != run_id, "Run-id belongs to another job")
    write_once(path, dict(task=task, rep=rep, run_id=run_id,
                         expanded_winner_sha256=winner_hash, maximum_native_attempts=1))
    return run_id


def stable_crc_sha(path):
    """Read every NPZ member through ZIP CRC and reject concurrent byte changes."""
    before = sha(path)
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, "NPZ member CRC failed: " + str(path))
    require(sha(path) == before, "NPZ changed during independent verification: " + str(path))
    return dict(sha256=before, zip_crc="PASS", size_bytes=Path(path).stat().st_size)


def transfer_status():
    jobs = []
    for task in TASKS:
        for rep in REPS:
            reservation_path = OUTPUT / "jobs" / task / f"rep{rep}.json"
            status = "NOT_STARTED"
            run_id = None
            if reservation_path.exists():
                reservation = read_json(reservation_path)
                run_id = reservation["run_id"]
                run_root = OUTPUT / "runs" / run_id
                owner = run_root / "OWNER_COMPLETION.json"
                finals = list((run_root / "records").glob("*/*/FINAL.json"))
                if owner.exists():
                    require(read_json(owner)["status"] == "OWNER_EXECUTION_ENDED_INTEGRITY_VERIFIED",
                            "Invalid owner completion status")
                    status = "COMPLETE"
                elif len(finals) == 1:
                    status = read_json(finals[0])["status"]
                else:
                    status = "RESERVED_OR_RUNNING"
            jobs.append(dict(task=task, rep=rep, run_id=run_id, status=status))
    complete = sum(job["status"] == "COMPLETE" for job in jobs)
    failures = [job for job in jobs if "FAILURE" in job["status"] or "INTERRUPTED" in job["status"]]
    return dict(status="COMPLETE" if complete == 10 else "INCOMPLETE" if failures else "IN_PROGRESS",
                planned_fits=10, complete_fits=complete, jobs=jobs,
                original_primary_unchanged=True, D4_scoring_performed=False,
                failures_are_not_zero_scores=True)


def verify_run(run_id, winner_hash, owner_session_id):
    """Invoke only in a later tool call after the single-fit session exits zero."""
    run_root = OUTPUT / "runs" / safe_run_id(run_id)
    finals = list((run_root / "records").glob("*/*/FINAL.json"))
    require(len(finals) == 1, "Unique run must contain exactly one finalized job")
    final_path = finals[0]
    final = read_json(final_path)
    require(final["run_id"] == run_id and final["expanded_winner_sha256"] == winner_hash,
            "Run/winner provenance mismatch")
    require(final["status"] == "COMPLETE_PENDING_EXTERNAL_VERIFICATION",
            "Cannot publish a failed or unfinished job")
    require(sources() == final["source_sha256"], "Code/runtime changed since the fit session")
    task, rep = final["task"], final["rep"]
    reservation = read_json(OUTPUT / "jobs" / task / f"rep{rep}.json")
    require(reservation["run_id"] == run_id, "Job reservation mismatch")
    source = Path(final["input_directory"])
    inputs = input_hashes(task, rep, source)
    require(inputs == final["input_hashes"], "Input changed after native completion")
    fit, compact = Path(final["fit_directory"]), Path(final["compact_directory"])
    require(fit.is_relative_to(run_root) and compact.is_relative_to(run_root), "Unsafe run artifact path")
    verified = verify_complete(fit, compact, final["parameters"], inputs)
    require(all(final[key] == value for key, value in verified.items()), "Finalized fit changed")
    archives = [fit / "effects.npz", fit / "native_unscaled.npz", compact / "compact.npz",
                source / "counts.npz", source / "axes.npz"]
    crc_sha = {str(path): stable_crc_sha(path) for path in archives}
    expected_digests = [verified["effects_sha256"], verified["native_unscaled_sha256"],
                        verified["compact_sha256"], inputs["counts.npz"], inputs["axes.npz"]]
    require(all(crc_sha[str(path)]["sha256"] == expected
                for path, expected in zip(archives, expected_digests)),
            "NPZ changed between report verification and independent CRC verification")
    require(input_hashes(task, rep, source) == inputs and sources() == final["source_sha256"],
            "Input/source changed during independent verification")
    owner_path = run_root / "OWNER_COMPLETION.json"
    created = (read_json(owner_path)["created_utc"] if owner_path.exists()
               else datetime.datetime.now(datetime.timezone.utc).isoformat())
    owner = dict(status="OWNER_EXECUTION_ENDED_INTEGRITY_VERIFIED", task=task, rep=rep, run_id=run_id,
                 verification="SEPARATE_PROCESS_AFTER_SINGLE_FIT_SESSION_EXIT_ZERO",
                 owner_session_id=owner_session_id, owner_exit_code=0, created_utc=created,
                 final_record_sha256=sha(final_path),
                 expanded_winner_sha256=winner_hash, fit_directory=str(fit),
                 compact_directory=str(compact), npz_crc_sha256=crc_sha,
                 npz_files={path.name: dict(sha256=crc_sha[str(path)]["sha256"], npz_crc_pass=True)
                            for path in archives[:3]},
                 input_hashes=inputs, source_sha256=final["source_sha256"],
                 audit_pass=True, compact_sha256=verified["compact_sha256"],
                 effects_sha256=verified["effects_sha256"],
                 native_unscaled_sha256=verified["native_unscaled_sha256"],
                 fit_report_sha256=verified["fit_report_sha256"],
                 native_solver_objective_audit_sha256=verified["certificate_sha256"],
                 input_stability_sha256=verified["serialization_stability_sha256"],
                 D4_scoring_performed=False, original_primary_unchanged=True)
    write_once(owner_path, owner)
    write_once(fit / "OWNER_COMPLETION.json", owner)
    write_once(compact / "OWNER_COMPLETION.json", owner)
    append_event(dict(event="INDEPENDENT_CRC_SHA_VERIFICATION_COMPLETE", run_id=run_id,
                      task=task, rep=rep, publishable=True,
                      owner_completion=str(run_root / "OWNER_COMPLETION.json"),
                      owner_completion_sha256=sha(run_root / "OWNER_COMPLETION.json")))
    summary = {**transfer_status(), "expanded_winner_sha256": winner_hash}
    if summary["status"] == "COMPLETE":
        write_once(OUTPUT / "EXPANDED_TRANSFER_EXECUTION.json", summary)
    print(json.dumps(summary, sort_keys=True))


def main():
    def terminate(signum, _frame):
        # Route normal termination through run_process cleanup so no native
        # child continues unrecorded after an interrupted driver.
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true", help="Verify D3 gate; no writes or fits")
    modes.add_argument("--select-only", action="store_true", help="Record immutable decision; no fits")
    modes.add_argument("--verify-run", metavar="RUN_ID", help="Next-call CRC/SHA verification; no fit")
    parser.add_argument("--task", choices=TASKS, help="Run exactly this one pooled D4 task")
    parser.add_argument("--rep", type=int, choices=REPS, help="Run exactly this one registered repetition")
    parser.add_argument("--run-id", help="Unique run directory label; defaults to a new UUID")
    parser.add_argument("--input-dir", type=Path, help="Explicit ready-marker-validated rebuilt pooled input")
    parser.add_argument("--owner-session-id", help="Completed prior exec session ID, required with --verify-run")
    args = parser.parse_args()
    if (args.task is None) != (args.rep is None):
        parser.error("--task and --rep must be provided together")
    if (args.check_only or args.select_only or args.verify_run) and (args.task or args.run_id or args.input_dir):
        parser.error("Read/selection/verification modes cannot request a fit")
    if args.run_id and not args.task:
        parser.error("--run-id requires --task and --rep")
    if args.input_dir and not args.task:
        parser.error("--input-dir requires --task and --rep")
    if bool(args.owner_session_id) != bool(args.verify_run):
        parser.error("--verify-run and --owner-session-id must be supplied together")
    try:
        decision = build_decision()
    except NotReady as exc:
        print(json.dumps(dict(status="BLOCKED_D3_INCOMPLETE", reason=str(exc), fits_started=0)))
        return 2
    if args.check_only:
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".driver.lock").open("a") as driver_lock:
        fcntl.flock(driver_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        winner_hash = save_decision(decision)
        if not decision["transfer_triggered"]:
            no_run = dict(status="NO_RUN_TRIGGER_FALSE", expanded_winner_sha256=winner_hash,
                          fits_started=0, D4_scoring_performed=False,
                          original_primary_unchanged=True)
            write_once(OUTPUT / "NO_RUN.json", no_run)
            print(json.dumps(no_run, sort_keys=True))
            return 0
        if args.verify_run:
            verify_run(args.verify_run, winner_hash, args.owner_session_id)
            return 0
        if args.task is None:
            print(json.dumps(dict(status="SELECTION_RECORDED_NO_FITS", expanded_winner_sha256=winner_hash)))
            return 0
        # One tool execution session owns exactly one fit and its compact. The
        # reservation prevents a later UUID from silently retrying that job.
        source_hashes = sources()
        source = (args.input_dir or ROOT / "inputs" / args.task / f"rep{args.rep}").resolve()
        inputs = input_hashes(args.task, args.rep, source)
        run_id = reserve_job(args.task, args.rep, args.run_id, winner_hash)
        result = run_job(args.task, args.rep, decision, winner_hash, source_hashes, inputs, run_id, source)
        check_immutable()
        print(json.dumps(dict(status=result["status"], run_id=run_id, publishable=False,
                              next_call=[sys.executable, str(Path(__file__)), "--verify-run", run_id,
                                         "--owner-session-id", "<completed-owner-session-id>"]),
                         sort_keys=True))
        return 0 if result["status"] == "COMPLETE_PENDING_EXTERNAL_VERIFICATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
