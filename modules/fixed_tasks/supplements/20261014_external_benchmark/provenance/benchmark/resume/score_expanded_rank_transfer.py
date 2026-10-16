#!/usr/bin/env python
"""One posthoc expanded-rank D4 score or combine owner; never fit or change gates.

Task and combine modes require fresh UUID output roots and run independently.
After the exec owner exits zero, invoke verify in a NEW tool call before using
its report. The original evaluator/audit mathematics and primary lock are read
unchanged; no LOCKED_BEFORE_D4_SCORING status or development gate is fabricated.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "2"
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "vendor_work"))
import atomic_npz
import run_expanded_rank_transfer as transfer
import expanded_posthoc_audit as independent_audit

TASKS = ["D4_Rest", "D4_Stim8hr"]
REPS = list(range(5))
SEED = 20261012
DRAWS = 2000
OWNER_STATUS = "OWNER_EXECUTION_ENDED_INTEGRITY_VERIFIED"
FAMILY_FLAG = "in_prespecified_five_contrast_family"
EVALUATOR_PATH = ROOT / "evaluation/evaluate_six_arm.py"
PINNED_MATH = {
    EVALUATOR_PATH: "4acd8cebce3162b25874518942dbed4fae8299cd8fead65da2ef1b789dd95237",
    ROOT / "independent/execution/evaluate_independent.py": "21cfcbe73404d7560922069de96b25fa706ac75900afb014a8aed1880fbc05cf",
    ROOT / "audit/audit_six_arm.py": "44b5840c8b60ae209fbbc4c7e686c8574ba16d6d3a42fd9bda4713867b6c367a",
}
SCOPE = (
    "Posthoc expanded-rank transfer on already-opened D3/D4 data. K150/200/300 "
    "selection is based only on the separately frozen D3 native-P25 rule. All "
    "intervals and contrasts are descriptive across that selection, conditional "
    "on the original observed input cells, fixed references, query panels, "
    "candidate pools, control geometry and D4 donor/states. This is not new "
    "blinded D4 validation or donor-population inference. The original rank100 "
    "primary results and continuation gate are unchanged; no new workflow gate "
    "is computed or applied."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return atomic_npz.sha(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, data):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".partial")
    with temporary.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    require(not path.exists(), "Append-only result already exists: " + str(path))
    os.rename(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    os.fsync(descriptor)
    os.close(descriptor)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_hashes():
    for path, expected in PINNED_MATH.items():
        require(sha(path) == expected, "Frozen scoring mathematics changed: " + str(path))
    paths = [Path(__file__), HERE / "expanded_posthoc_audit.py", HERE / "run_expanded_rank_transfer.py",
             ROOT / "vendor_work/atomic_npz.py", *PINNED_MATH]
    return {str(path): sha(path) for path in paths}


def verify_audit_source(audit, report_path):
    """A PASS status must refer to the exact report and compact being accepted."""
    report_path = Path(report_path).resolve()
    report = load_json(report_path)
    sources = audit.get("sources", [])
    require(audit.get("status") == "PASS" and len(sources) == 1,
            "Expected one independently audited task source")
    source = sources[0]
    require(Path(source["path"]).resolve() == report_path
            and source["sha256"] == sha(report_path)
            and source["compact_sha256"] == report["compact_sha256"]
            == sha(report_path.with_name("compact.npz")),
            "Independent audit is not bound to this task score report")


def validate_selection():
    """Check actual posthoc status and all D3 evidence before any D4 reference read."""
    winner_path = HERE / "expanded_rank_transfer/POSTHOC_EXPANDED_RANK_WINNER.json"
    decision = load_json(winner_path)
    winner_hash = sha(winner_path)
    require(winner_path.with_suffix(".sha256").read_text().split()[0] == winner_hash,
            "Expanded selection sidecar mismatch")
    require(decision == transfer.build_decision(), "Expanded decision no longer matches its D3 evidence")
    require(decision["status"] == "LOCKED_POSTHOC_TRANSFER" and decision["transfer_triggered"] is True,
            "Expanded transfer did not trigger")
    protocol_path = HERE / "POSTHOC_D3_RANK_BOUNDARY.json"
    protocol = load_json(protocol_path)
    config = next(c for c in protocol["fr_grid"] if c["id"] == decision["config_id"])
    ledger = load_json(HERE / "RANK_BOUNDARY_EXECUTION.json")
    score_owners = []
    for entry in ledger["fits"]:
        directory = Path(entry["evaluation_directory"])
        report_path = directory / "evaluation_report.json"
        owner_path = directory / "OWNER_COMPLETION.json"
        owner = load_json(owner_path)
        audit_path = directory / "audit/AUDIT_RESULT.json"
        audit = load_json(audit_path)
        require(owner["status"] == OWNER_STATUS and owner["owner_exit_code"] == 0
                and bool(owner["owner_session_id"]) and owner["npz_crc"] == "PASS"
                and owner["score_report_sha256"] == sha(report_path) == entry["score_report_sha256"]
                and owner["native_owner_completion_sha256"] == sha(entry["owner_completion"])
                and owner["owner_session_id"] == entry["score_owner_session_id"]
                and entry["score_owner_exit_code"] == 0 and audit["status"] == "PASS",
                "D3 score-owner or independent-score audit proof missing")
        verify_audit_source(audit, report_path)
        score_owners.append(dict(rank=entry["config"]["rank"], owner_completion=str(owner_path),
                                 owner_completion_sha256=sha(owner_path), audit_sha256=sha(audit_path)))
    return decision, config, protocol, dict(
        expanded_selection_path=str(winner_path), expanded_selection_sha256=winner_hash,
        expanded_selection_status=decision["status"], protocol_path=str(protocol_path),
        protocol_sha256=sha(protocol_path), expanded_rule_sha256=decision["rule_sha256"],
        original_primary_winner_sha256=decision["original_winner_lock_sha256"],
        D3_score_owner_evidence=score_owners)


def verify_native_owner(effect_path, owner_path, task, rep, decision, winner_hash):
    effect_path, owner_path = Path(effect_path).resolve(), Path(owner_path).resolve()
    require(effect_path.name == "effects.npz" and effect_path.is_relative_to(HERE),
            "Expected explicit append-only effects path")
    owner = load_json(owner_path)
    fit = effect_path.parent
    report_path, audit_path = fit / "fit_report.json", fit / "NATIVE_SOLVER_OBJECTIVE_AUDIT.json"
    report, audit = load_json(report_path), load_json(audit_path)
    stability_path = fit / "ATOMIC_SERIALIZATION_INPUT_STABILITY.json"
    stability = load_json(stability_path)
    require(owner["status"] == OWNER_STATUS and owner["owner_exit_code"] == 0
            and bool(owner["owner_session_id"]) and owner["task"] == task and owner["rep"] == rep
            and Path(owner["fit_directory"]).resolve() == fit
            and owner["expanded_winner_sha256"] == winner_hash
            and owner["fit_report_sha256"] == sha(report_path)
            and owner["native_solver_objective_audit_sha256"] == sha(audit_path)
            and owner["input_stability_sha256"] == sha(stability_path),
            "Native owner completion is not bound to this expanded task/repetition")
    parameters = dict(rank=decision["rank"], spca_alpha=0.1, lasso_alpha=0.0001,
                      spca_iter=50, seed=SEED + rep, threads=2)
    require(report["status"] == "COMPLETE" and report["parameters"] == parameters
            and audit["pass_"] is True and audit["original_fit_report_sha256"] == sha(report_path),
            "Native fit/audit/parameters mismatch")
    require(stability["status"] == "PASS" and stability["before"] == stability["after"],
            "Native input stability guard failed")
    for filename in ("effects.npz", "native_unscaled.npz", "compact.npz"):
        require(owner["npz_files"][filename]["npz_crc_pass"] is True, "Native owner lacks NPZ CRC proof")
    digest = atomic_npz.validate(effect_path)
    require(digest == report["effects_sha256"] == owner["npz_files"]["effects.npz"]["sha256"],
            "Owner-verified effects changed")
    return dict(input_rep=rep, path=str(effect_path), sha256=digest,
                owner_completion=str(owner_path), owner_completion_sha256=sha(owner_path),
                owner_session_id=owner["owner_session_id"], owner_exit_code=0,
                fit_report_sha256=sha(report_path), audit_sha256=sha(audit_path))


def descriptive_results(results, arrays, combined=False):
    """Metadata-only relabeling plus the requested PR-versus-weighted-F contrast."""
    prefix = "task_stratified_query" if combined else "query_only"
    bootstrap = "equal_state_bootstrap" if combined else "query_only_bootstrap"
    for k in (25, 10):
        result = results[str(k)]
        for item in result["contrasts"].values():
            item[FAMILY_FLAG] = False
            item["inference"] = "posthoc descriptive"
        left, right = result["arms"]["pertrank"], result["arms"]["fr_pertrank"]
        draws = arrays[f"{bootstrap}_p{k}"][:, 2] - arrays[f"{bootstrap}_p{k}"][:, 5]
        result["contrasts"]["pertrank_minus_fr_pertrank"] = dict(
            point=left["point"] - right["point"],
            input_rep_points=(np.asarray(left["input_rep_points"]) - np.asarray(right["input_rep_points"])).tolist(),
            **{prefix + "_95_interval": np.quantile(draws, [.025, .975]).tolist(),
               prefix + "_99_interval": np.quantile(draws, [.005, .995]).tolist(),
               FAMILY_FLAG: False}, inference="posthoc descriptive; excluded from original five-contrast family")
    return results


def common_fields():
    return dict(analysis_label="posthoc expanded-rank transfer", scope=SCOPE,
                original_primary_unchanged=True, primary_continuation_gate_computed=False,
                workflow_gate_applied=False, in_prespecified_five_contrast_family=False,
                simultaneous_intervals="All 95% and 99% intervals are posthoc descriptive. No simultaneous-family, equivalence or confirmatory claim.")


def verify_task_sources(report, decision):
    """Recheck all task bytes and owner-bound native evidence used by scoring."""
    provenance = report["provenance"]
    for label in ("task", "selection"):
        require(sha(provenance[label + "_path"]) == provenance[label + "_sha256"],
                "Frozen task input changed: " + label)
    owners, native = provenance["native_owner_evidence"], provenance["fr"]
    require(len(owners) == len(native) == 5, "Expected all five native evidence records")
    for rep, (owner, scored) in enumerate(zip(owners, native)):
        current = verify_native_owner(owner["path"], owner["owner_completion"],
                                      report["dataset"], rep, decision,
                                      provenance["expanded_selection_sha256"])
        require(current == owner, "Native scoring evidence changed after its initial verification")
        require(scored["input_rep"] == rep
                and Path(scored["effects_path"]).resolve() == Path(owner["path"])
                and scored["effects_sha256"] == owner["sha256"]
                and scored["fit_report_sha256"] == owner["fit_report_sha256"]
                and scored["solver_audit_sha256"] == owner["audit_sha256"],
                "Evaluator provenance differs from owner-verified native evidence")


def evaluate_task(args, out, evaluator, decision, config, protocol, provenance):
    require(len(set(map(str, args.effects))) == 5, "Provide five distinct effects in rep0..4 order")
    owners = [verify_native_owner(effect, owner, args.dataset, rep, decision,
                                  provenance["expanded_selection_sha256"])
              for rep, effect, owner in zip(REPS, args.effects, args.owner_completions)]
    # Every native owner and D3 selection check precedes reading D4 references.
    task_path = ROOT / "pooled_data" / args.dataset / "frozen_task.npz"
    selection_path = task_path.with_name("selection.npz")
    # Bind the bytes before the evaluator reads them, then check again after audit.
    task_hash, selection_hash = sha(task_path), sha(selection_path)
    z, reference = evaluator.assemble_pooled(task_path, REPS)
    require(reference is None, "Only frozen pooled references are allowed")
    selection = evaluator.load_npz(selection_path)
    fr, native = [], []
    for rep, effect in zip(REPS, args.effects):
        values, evidence = evaluator.native_fr(effect, z["candidates"], z["queries"], rep,
                                               config, protocol, winner=decision, selection=selection)
        fr.append(values)
        native.append(evidence)
    z["fr_scores"] = np.stack(fr)
    evaluator.validate_compact(z, args.dataset, "pooled")
    begin = time.monotonic()
    result, arrays, reference_seed = evaluator.score_arrays(z, draws=DRAWS, reference=None)
    require(reference_seed is None, "No new reference bootstrap is authorized")
    z["six_arm_p25"], z["six_arm_p10"] = arrays["six_arm_p25"], arrays["six_arm_p10"]
    atomic_npz.install()
    np.savez_compressed(out / "compact.npz", **z)
    np.savez_compressed(out / "evaluation_arrays.npz", **arrays)
    report = dict(status="COMPLETE", created_utc=now(), dataset=args.dataset, kind="pooled",
        config_id=config["id"], config=config, input_reps=REPS, arms=evaluator.ARMS,
        bootstrap_draws=DRAWS, query_seed=SEED, reference_seed=None, reference_batch_size=None,
        shape=dict(input_reps=5, queries=len(z["queries"]), candidates=len(z["candidates"]), blocks=30),
        primary_K=25, sensitivity_K=10, results=descriptive_results(result, arrays),
        elapsed_seconds=time.monotonic() - begin, **common_fields(),
        provenance={**provenance, "task_path": str(task_path), "task_sha256": task_hash,
                    "selection_path": str(selection_path), "selection_sha256": selection_hash,
                    "fr": native, "native_owner_evidence": owners,
                    "evaluator_sha256": sha(EVALUATOR_PATH),
                    "score_primitive_sha256": sha(ROOT / "independent/execution/evaluate_independent.py"),
                    "winner_lock_path": None, "winner_lock_sha256": None},
        compact_sha256=sha(out / "compact.npz"), evaluation_arrays_sha256=sha(out / "evaluation_arrays.npz"))
    report_path = out / "evaluation_report.json"
    save_json(report_path, report)
    audit = independent_audit.audit_task(report_path, out / "audit")
    require(audit["status"] == "PASS", "Independent posthoc task audit failed")
    verify_task_sources(report, decision)
    return report_path, dict(native_owners=owners, audit_sha256=sha(out / "audit/AUDIT_RESULT.json"))


def verify_report_owner(report_path, owner_path, winner_hash):
    report_path, owner_path = Path(report_path).resolve(), Path(owner_path).resolve()
    report, owner = load_json(report_path), load_json(owner_path)
    require(owner["status"] == OWNER_STATUS and owner["owner_exit_code"] == 0
            and bool(owner["owner_session_id"]) and owner["report_path"] == str(report_path)
            and owner["report_sha256"] == sha(report_path)
            and owner["expanded_selection_sha256"] == winner_hash
            and owner["independent_audit_pass"] is True,
            "Task score owner has not completed next-call verification")
    require(report["status"] == "COMPLETE" and report["analysis_label"] == "posthoc expanded-rank transfer"
            and report["provenance"]["expanded_selection_sha256"] == winner_hash
            and report["primary_continuation_gate_computed"] is False
            and "development_resource_gates" not in report, "Not a posthoc expanded task report")
    for filename, key in (("compact.npz", "compact_sha256"), ("evaluation_arrays.npz", "evaluation_arrays_sha256")):
        path = report_path.with_name(filename)
        digest = atomic_npz.validate(path)
        require(digest == report[key] == owner["npz_files"][filename]["sha256"]
                and owner["npz_files"][filename]["npz_crc_pass"] is True, "Verified task archive changed")
    audit_path = report_path.parent / "audit/AUDIT_RESULT.json"
    audit = load_json(audit_path)
    require(audit["status"] == "PASS" and owner["audit_result_sha256"] == sha(audit_path),
            "Independent task audit evidence changed")
    verify_audit_source(audit, report_path)
    for filename, digest in audit["audit_files_sha256"].items():
        require(sha(audit_path.parent / filename) == digest, "Independent task audit artifact changed")
    execution_path = report_path.parent / "POSTHOC_EXECUTION.json"
    require(owner["mode"] == "task" and owner["execution_sha256"] == sha(execution_path),
            "Task owner execution record changed")
    return report


def verify_combine_sources(report, winner_hash):
    """Revalidate the exact two task reports and owner certificates after use."""
    sources, owners = report["provenance"], report["task_owner_evidence"]
    require(len(sources) == len(owners) == 2, "Expected exactly two combination sources")
    for task, source, owner in zip(TASKS, sources, owners):
        require(sha(source["path"]) == source["sha256"]
                and sha(owner["path"]) == owner["sha256"],
                "Combination report or owner certificate changed")
        current = verify_report_owner(source["path"], owner["path"], winner_hash)
        require(current["dataset"] == task and current["config"] == report["config"],
                "Combination task source differs from its recorded configuration")


def combine_tasks(args, out, evaluator, decision, config, protocol, provenance):
    source_records = [dict(path=str(path.resolve()), sha256=sha(path)) for path in args.reports]
    owner_records = [dict(path=str(path.resolve()), sha256=sha(path)) for path in args.owner_completions]
    reports = [verify_report_owner(path, owner, provenance["expanded_selection_sha256"])
               for path, owner in zip(args.reports, args.owner_completions)]
    require([r["dataset"] for r in reports] == TASKS
            and all(r["kind"] == "pooled" and r["input_reps"] == REPS
                    and r["config"] == config and r["shape"]["queries"] == 300 for r in reports),
            "Combine exactly Rest then Stim, five repetitions and the same expanded configuration")
    arrays = [evaluator.load_npz(path.with_name("evaluation_arrays.npz")) for path in args.reports]
    compact = [evaluator.load_npz(path.with_name("compact.npz")) for path in args.reports]
    seeds = np.random.SeedSequence(SEED).spawn(2)
    saved, weights = {}, []
    for index, (z, seed) in enumerate(zip(compact, seeds)):
        sampled, weight = evaluator.old.query_draws(z["blocks"], DRAWS, seed)
        saved[f"state_{index}_sampled_blocks"] = sampled
        saved[f"state_{index}_query_weights"] = weight
        weights.append(weight)
    results = {}
    for k in (25, 10):
        scores = [array[f"six_arm_p{k}"] for array in arrays]
        boots = [weight @ score.mean(0).T for weight, score in zip(weights, scores)]
        equal_state = np.mean(boots, axis=0)
        results[str(k)] = evaluator.summarize(np.concatenate(scores, axis=2), equal_state,
                                              prefix="task_stratified_query")
        for index, boot in enumerate(boots):
            saved[f"state_{index}_query_bootstrap_p{k}"] = boot
        saved[f"equal_state_bootstrap_p{k}"] = equal_state
    atomic_npz.install()
    np.savez_compressed(out / "equal_state_arrays.npz", **saved)
    report = dict(status="COMPLETE", created_utc=now(), datasets=TASKS,
        winner_config_id=config["id"], config=config, results=descriptive_results(results, saved, combined=True),
        seed=SEED, state_seed_spawn_keys=[list(seed.spawn_key) for seed in seeds], draws=DRAWS,
        **common_fields(),
        combination="Independent task-specific query-block streams; exact equal-state mean. Different panels are not query-paired.",
        paired_input_reps="Repetition labels are equal-weighted for stability; they are not biological donors.",
        provenance=source_records,
        expanded_selection_provenance=provenance,
        task_owner_evidence=owner_records,
        arrays_sha256=sha(out / "equal_state_arrays.npz"))
    report_path = out / "equal_state_report.json"
    save_json(report_path, report)
    audit = independent_audit.audit_combine(args.reports, report_path, out / "audit")
    require(audit["status"] == "PASS", "Independent posthoc combined audit failed")
    verify_combine_sources(report, provenance["expanded_selection_sha256"])
    return report_path, dict(audit_sha256=sha(out / "audit/AUDIT_RESULT.json"))


def verify_output(args, provenance):
    out = args.run_root.resolve()
    execution_path = out / "POSTHOC_EXECUTION.json"
    execution = load_json(execution_path)
    require(execution["status"] == "AWAITING_OWNER_VERIFICATION"
            and execution["expanded_selection_sha256"] == provenance["expanded_selection_sha256"],
            "Scoring/combine owner did not finish successfully")
    require(source_hashes() == execution["source_sha256"], "Scoring source changed since execution")
    report_path = Path(execution["report_path"])
    require(report_path.parent == out and sha(report_path) == execution["report_sha256"], "Report changed")
    report = load_json(report_path)
    independent_audit.check_report(independent_audit.Checks({"checks": []}), report,
                                   combined=execution["mode"] == "combine")
    audit_path = out / "audit/AUDIT_RESULT.json"
    audit = load_json(audit_path)
    require(audit["status"] == "PASS" and sha(audit_path) == execution["audit_sha256"], "Independent audit changed")
    for filename, digest in audit["audit_files_sha256"].items():
        require(sha(audit_path.parent / filename) == digest, "Independent audit artifact changed")
    archives = {}
    for path in sorted(out.rglob("*.npz")):
        digest = atomic_npz.validate(path)
        require(execution["output_npz_sha256"][str(path.relative_to(out))] == digest,
                "Scoring output archive changed")
        archives[str(path.relative_to(out))] = dict(sha256=digest, npz_crc_pass=True)
    require(set(archives) == set(execution["output_npz_sha256"]), "Scoring archive set changed")
    if execution["mode"] == "task":
        decision = load_json(provenance["expanded_selection_path"])
        require(sha(provenance["expanded_selection_path"]) == provenance["expanded_selection_sha256"],
                "Expanded selection changed during owner verification")
        verify_task_sources(report, decision)
    elif execution["mode"] == "combine":
        verify_combine_sources(report, provenance["expanded_selection_sha256"])
    else:
        raise ValueError("Unknown scoring owner execution mode")
    owner_path = out / "OWNER_COMPLETION.json"
    created = load_json(owner_path)["created_utc"] if owner_path.exists() else now()
    owner = dict(status=OWNER_STATUS, owner_session_id=args.owner_session_id, owner_exit_code=0,
                 created_utc=created, report_path=str(report_path), report_sha256=sha(report_path),
                 mode=execution["mode"], analysis_label="posthoc expanded-rank transfer",
                 expanded_selection_sha256=provenance["expanded_selection_sha256"],
                 execution_sha256=sha(execution_path), npz_files=archives, npz_crc="PASS",
                 audit_result_sha256=sha(audit_path), independent_audit_pass=True,
                 primary_continuation_gate_computed=False, original_primary_unchanged=True,
                 separate_tool_call_after_owner_exit_zero=True)
    if owner_path.exists():
        require(load_json(owner_path) == owner, "Immutable owner completion differs")
    else:
        save_json(owner_path, owner)
    print(json.dumps(dict(status=OWNER_STATUS, owner_completion=str(owner_path),
                          owner_completion_sha256=sha(owner_path), report=str(report_path)), sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    task = commands.add_parser("task", help="One pooled D4 task; explicit rep0..4 inputs")
    task.add_argument("--dataset", choices=TASKS, required=True)
    task.add_argument("--effects", nargs=5, type=Path, required=True)
    task.add_argument("--owner-completions", nargs=5, type=Path, required=True)
    task.add_argument("--output-root", type=Path, required=True)
    combine = commands.add_parser("combine", help="One equal-state combination; Rest then Stim")
    combine.add_argument("--reports", nargs=2, type=Path, required=True)
    combine.add_argument("--owner-completions", nargs=2, type=Path, required=True)
    combine.add_argument("--output-root", type=Path, required=True)
    verify = commands.add_parser("verify", help="NEW-tool-call verification after owner exit zero")
    verify.add_argument("--run-root", type=Path, required=True)
    verify.add_argument("--owner-session-id", required=True)
    args = parser.parse_args()
    sources = source_hashes()
    decision, config, protocol, provenance = validate_selection()
    if args.mode == "verify":
        verify_output(args, provenance)
        return 0
    out = args.output_root.expanduser().resolve()
    require(out.is_relative_to(HERE) and not out.exists(), "Use a new output root below resume_20261014")
    suffix = out.name[-36:]
    require(str(uuid.UUID(suffix)) == suffix, "Output root basename must end with a canonical UUID")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    start = dict(status="STARTED", mode=args.mode, created_utc=now(), source_sha256=sources,
                 expanded_selection_sha256=provenance["expanded_selection_sha256"],
                 original_primary_unchanged=True, no_native_fit_executed=True)
    save_json(out / "START.json", start)
    try:
        with (ROOT / "heavy_memory.lock").open("a") as lock:
            waited = time.monotonic()
            fcntl.flock(lock, fcntl.LOCK_EX)
            lock_wait = time.monotonic() - waited
            evaluator = module_from_path("unchanged_expanded_score_core", EVALUATOR_PATH)
            runner = evaluate_task if args.mode == "task" else combine_tasks
            report_path, evidence = runner(args, out, evaluator, decision, config, protocol, provenance)
        require(source_hashes() == sources, "Scoring sources changed during execution")
        require(validate_selection()[3] == provenance, "Expanded D3 evidence changed during scoring")
        execution = dict(**{k: v for k, v in start.items() if k != "status"},
                         status="AWAITING_OWNER_VERIFICATION", report_path=str(report_path),
                         report_sha256=sha(report_path), **evidence,
                         output_npz_sha256={str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*.npz"))},
                         lock_wait_seconds=lock_wait, wall_seconds=time.monotonic() - started,
                         no_primary_gate_computed=True)
        save_json(out / "POSTHOC_EXECUTION.json", execution)
    except BaseException as exc:
        save_json(out / "FAILURE.json", dict(status="FAILED_NOT_OWNER_VERIFIED", error=repr(exc),
                  traceback=traceback.format_exc(), source_sha256=sources,
                  wall_seconds=time.monotonic() - started, partial_outputs_preserved=True))
        raise
    print(json.dumps(dict(status="AWAITING_OWNER_VERIFICATION", report=str(report_path),
                          run_root=str(out), publishable=False,
                          next_call=[sys.executable, str(Path(__file__)), "verify", "--run-root", str(out),
                                     "--owner-session-id", "<completed-owner-session-id>"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
