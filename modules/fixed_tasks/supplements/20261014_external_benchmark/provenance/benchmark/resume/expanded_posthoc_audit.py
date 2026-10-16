"""Independent descriptive replay for the expanded-rank D4 reports.

The unchanged audit_six_arm implementation supplies exact-tie scores and fresh
query-block draws. This adapter checks every report arm/contrast and every saved
draw, without evaluating or replacing the primary continuation gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "audit"))
sys.path.insert(0, str(ROOT / "vendor_work"))
from audit_six_arm import ARM_NAMES, run as independent_run
import atomic_npz

SEED = 20261012
DRAWS = 2000
REPS = list(range(5))
TASKS = ["D4_Rest", "D4_Stim8hr"]
FAMILY_FLAG = "in_prespecified_five_contrast_family"
SCOPE = (
    "Posthoc expanded-rank sensitivity; all intervals are descriptive across "
    "expanded-rank selection. Independent exact-tie scores and query-block "
    "bootstrap draws conditional on saved predictions, input selections, fixed "
    "references, query panels, candidate pools, control geometry and observed "
    "D4 donor/states. No donor-population inference or confirmatory claim. "
    "The original primary continuation gate is unchanged and is not computed here."
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def descriptive_audit(value):
    """Relabel legacy metadata only; preserve all independent numeric results."""
    if isinstance(value, list):
        return [descriptive_audit(item) for item in value]
    if not isinstance(value, dict):
        return value
    output = {}
    for key, item in value.items():
        if key == "conditional_query_block_ci99_bonferroni_five":
            key = "conditional_query_block_ci99_descriptive"
        output[key] = descriptive_audit(item)
    if "contrasts" in output:
        for item in output["contrasts"].values():
            item[FAMILY_FLAG] = False
    return output


class Checks:
    def __init__(self, result):
        self.result = result

    def ok(self, name, value, detail=None):
        self.result["checks"].append({"name": name, "pass": bool(value), "detail": detail})
        if not value:
            raise AssertionError(name)

    def close(self, name, actual, expected, tol=1e-12):
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        same_shape = actual.shape == expected.shape
        finite = np.isfinite(actual).all() and np.isfinite(expected).all()
        error = float(np.max(np.abs(actual - expected))) if same_shape and actual.size and finite else None
        self.ok(name, same_shape and finite
                and error is not None and error <= tol,
                {"max_abs_error": error, "tolerance": tol})


def check_report(checks, report, combined=False):
    checks.ok("report/complete", report["status"] == "COMPLETE")
    if combined:
        checks.ok("report/task_order", report["datasets"] == TASKS)
        checks.ok("report/draws", report["draws"] == DRAWS)
        checks.ok("report/seed", report["seed"] == SEED)
        checks.ok("report/child_streams", report["state_seed_spawn_keys"] == [[0], [1]])
    else:
        checks.ok("report/pooled_D4", report["dataset"] in TASKS and report["kind"] == "pooled")
        checks.ok("report/repetitions", report["input_reps"] == REPS)
        checks.ok("report/arms", report["arms"] == ARM_NAMES)
        checks.ok("report/draws", report["bootstrap_draws"] == DRAWS)
        checks.ok("report/seed", report["query_seed"] == SEED)
        checks.ok("report/fixed_reference", report.get("reference_seed") is None)
    checks.ok("report/no_primary_gate", "development_resource_gates" not in report)
    checks.ok("report/endpoints", set(report["results"]) == {"25", "10"})
    for k in (25, 10):
        summary = report["results"][str(k)]
        checks.ok(f"K{k}/arm_names", set(summary["arms"]) == set(ARM_NAMES))
        checks.ok(f"K{k}/contrasts_present", bool(summary["contrasts"]))
        for name, item in summary["contrasts"].items():
            checks.ok(f"K{k}/{name}/descriptive_family", item.get(FAMILY_FLAG) is False)


def compare_summary(checks, tag, rep_means, boots, report_summary, prefix):
    """Check all report statistics against independently computed six-arm draws."""
    rep_means, boots = np.asarray(rep_means), np.asarray(boots)
    checks.ok(tag + "/rep_shape", rep_means.shape == (5, 6))
    checks.ok(tag + "/draw_shape", boots.shape == (DRAWS, 6))
    for index, arm in enumerate(ARM_NAMES):
        item = report_summary["arms"][arm]
        checks.close(f"{tag}/{arm}/point", rep_means[:, index].mean(), item["point"])
        checks.close(f"{tag}/{arm}/input_reps", rep_means[:, index], item["input_rep_points"])
        checks.close(f"{tag}/{arm}/ci95", np.quantile(boots[:, index], [.025, .975]), item[prefix + "_95_interval"])
        if prefix + "_99_interval" in item:
            checks.close(f"{tag}/{arm}/ci99", np.quantile(boots[:, index], [.005, .995]), item[prefix + "_99_interval"])
    for name, item in report_summary["contrasts"].items():
        names = name.split("_minus_")
        checks.ok(f"{tag}/{name}/known_contrast", len(names) == 2 and all(n in ARM_NAMES for n in names))
        left, right = map(ARM_NAMES.index, names)
        reps = rep_means[:, left] - rep_means[:, right]
        draws = boots[:, left] - boots[:, right]
        checks.close(f"{tag}/{name}/point", reps.mean(), item["point"])
        checks.close(f"{tag}/{name}/input_reps", reps, item["input_rep_points"])
        for level, quantiles in ((95, [.025, .975]), (99, [.005, .995])):
            checks.close(f"{tag}/{name}/ci{level}", np.quantile(draws, quantiles), item[f"{prefix}_{level}_interval"])
        checks.ok(f"{tag}/{name}/descriptive_family", item.get(FAMILY_FLAG) is False)


def replay(files, out, k):
    atomic_npz.install()
    result = independent_run(files, out, k, DRAWS, SEED, True, 5)
    result = descriptive_audit(result)
    result["analysis_label"] = "posthoc expanded-rank transfer"
    result["ci_scope"] = SCOPE
    result[FAMILY_FLAG] = False
    result["primary_continuation_gate_computed"] = False
    save_json(out, result)
    atomic_npz.validate(out.with_suffix(".npz"))
    return result, load_arrays(out.with_suffix(".npz"))


def start(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    result = {"status": "RUNNING", "analysis_label": "posthoc expanded-rank transfer",
              "checks": [], "sources": [], "scope": SCOPE, FAMILY_FLAG: False,
              "primary_continuation_gate_computed": False}
    return outdir, result, Checks(result)


def finish(outdir, result):
    result["n_checks"] = len(result["checks"])
    result["script_sha256"] = sha(__file__)
    result["scorer_sha256"] = sha(ROOT / "audit/audit_six_arm.py")
    result["atomic_npz_sha256"] = sha(ROOT / "vendor_work/atomic_npz.py")
    result["audit_files_sha256"] = {p.name: sha(p) for p in sorted(outdir.iterdir()) if p.is_file()}
    save_json(outdir / "AUDIT_RESULT.json", result)


def source(checks, path, report):
    path = Path(path).resolve()
    compact = path.with_name("compact.npz")
    arrays = path.with_name("evaluation_arrays.npz")
    record = {"path": str(path), "sha256": sha(path),
              "compact_sha256": atomic_npz.validate(compact),
              "evaluation_arrays_sha256": atomic_npz.validate(arrays)}
    checks.ok("source/compact_hash", record["compact_sha256"] == report["compact_sha256"])
    checks.ok("source/evaluation_arrays_hash", record["evaluation_arrays_sha256"] == report["evaluation_arrays_sha256"])
    return record, compact, load_arrays(arrays)


def stable(checks, record):
    path = Path(record["path"])
    checks.ok("source/report_stable", sha(path) == record["sha256"])
    checks.ok("source/compact_stable", sha(path.with_name("compact.npz")) == record["compact_sha256"])
    checks.ok("source/evaluation_arrays_stable", sha(path.with_name("evaluation_arrays.npz")) == record["evaluation_arrays_sha256"])


def audit_task(report_path, outdir):
    """Replay one pooled D4 report; output files are new and never overwritten."""
    outdir, result, checks = start(outdir)
    try:
        report = load_json(report_path)
        check_report(checks, report)
        record, compact, saved = source(checks, report_path, report)
        result["sources"].append(record)
        for k in (25, 10):
            _, independent = replay([compact], outdir / f"task_K{k}.json", k)
            tag = f"{report['dataset']}/K{k}"
            scores = independent[f"condition_0_six_arm_p{k}"]
            boots = independent["condition_0_bootstrap_means"]
            compare_summary(checks, tag, scores.mean(2), boots, report["results"][str(k)], "query_only")
            checks.close(tag + "/per_query_scores", scores, saved[f"six_arm_p{k}"])
            checks.close(tag + "/bootstrap_scores", boots, saved[f"query_only_bootstrap_p{k}"])
            checks.ok(tag + "/sampled_blocks", np.array_equal(independent["condition_0_bootstrap_sampled_block_indices"], saved["sampled_query_blocks"]))
            counts = independent["condition_0_bootstrap_query_counts"]
            checks.close(tag + "/query_weights", counts / counts.sum(1)[:, None], saved["query_weights"])
        stable(checks, record)
        result["status"] = "PASS"
    except BaseException as exc:
        result.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        finish(outdir, result)
    return result


def audit_combine(report_paths, combined_path, outdir):
    """Replay task-stratified draws in Rest/Stim order, without primary gates."""
    outdir, result, checks = start(outdir)
    try:
        paths = [Path(p) for p in report_paths]
        checks.ok("sources/exactly_two_tasks", len(paths) == 2)
        reports = [load_json(p) for p in paths]
        checks.ok("sources/frozen_order", [r["dataset"] for r in reports] == TASKS)
        compacts, task_arrays = [], []
        for path, report in zip(paths, reports):
            check_report(checks, report)
            record, compact, arrays = source(checks, path, report)
            result["sources"].append(record)
            compacts.append(compact)
            task_arrays.append(arrays)
        combined_path = Path(combined_path).resolve()
        combined = load_json(combined_path)
        check_report(checks, combined, combined=True)
        combined_hash = sha(combined_path)
        arrays_path = combined_path.with_name("equal_state_arrays.npz")
        arrays_hash = atomic_npz.validate(arrays_path)
        checks.ok("combined/array_hash", arrays_hash == combined["arrays_sha256"])
        checks.ok("combined/source_binding", combined["provenance"] == [
            {"path": str(p.resolve()), "sha256": record["sha256"]}
            for p, record in zip(paths, result["sources"])])
        result["combined_source"] = {"path": str(combined_path), "sha256": combined_hash, "arrays_sha256": arrays_hash}
        saved = load_arrays(arrays_path)
        for k in (25, 10):
            _, independent = replay(compacts, outdir / f"equal_state_K{k}.json", k)
            boots, rep_means = [], []
            for i in (0, 1):
                tag = f"combined/K{k}/state{i}"
                scores = independent[f"condition_{i}_six_arm_p{k}"]
                boot = independent[f"condition_{i}_bootstrap_means"]
                checks.close(tag + "/per_query_scores", scores, task_arrays[i][f"six_arm_p{k}"])
                checks.ok(tag + "/sampled_blocks", np.array_equal(independent[f"condition_{i}_bootstrap_sampled_block_indices"], saved[f"state_{i}_sampled_blocks"]))
                counts = independent[f"condition_{i}_bootstrap_query_counts"]
                checks.close(tag + "/query_weights", counts / counts.sum(1)[:, None], saved[f"state_{i}_query_weights"])
                checks.close(tag + "/bootstrap_scores", boot, saved[f"state_{i}_query_bootstrap_p{k}"])
                # Per-task points and repetition means are invariant to RNG stream.
                for ai, arm in enumerate(ARM_NAMES):
                    item = reports[i]["results"][str(k)]["arms"][arm]
                    checks.close(tag + f"/{arm}/point", scores[:, ai].mean(), item["point"])
                    checks.close(tag + f"/{arm}/input_reps", scores[:, ai].mean(1), item["input_rep_points"])
                rep_means.append(scores.mean(2))
                boots.append(boot)
            mean_boot = np.mean(boots, axis=0)
            checks.close(f"combined/K{k}/bootstrap_scores", mean_boot, saved[f"equal_state_bootstrap_p{k}"])
            compare_summary(checks, f"combined/K{k}", np.mean(rep_means, axis=0), mean_boot,
                            combined["results"][str(k)], "task_stratified_query")
        for record in result["sources"]:
            stable(checks, record)
        checks.ok("combined/report_stable", sha(combined_path) == combined_hash)
        checks.ok("combined/arrays_stable", sha(arrays_path) == arrays_hash)
        result["status"] = "PASS"
    except BaseException as exc:
        result.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        finish(outdir, result)
    return result
