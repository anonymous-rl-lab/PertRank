#!/usr/bin/env python3
"""Offline, sequential dispatcher for the declared PertRank reproduction scopes.

This dispatcher never treats process success as proof of raw-data/model refitting.
Module-native reports retain the scientific level of each reproduced component.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
MANIFEST = "MANIFEST_SHA256.json"
REPORT = "REPRODUCTION_REPORT.json"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root=ROOT):
    """Verify all release payload, including nested manifests; omit Git metadata."""
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        raise ValueError("Outer MANIFEST_SHA256.json is missing; finalize the package before execution.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("algorithm") != "sha256" or manifest.get("schema_version") != 1:
        raise ValueError("Expected manifest schema_version=1 and algorithm=sha256.")
    entries = manifest.get("files")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Manifest files must be a nonempty relative-path mapping.")
    actual = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if Path(relative).parts[0] == ".git":
            continue
        if path.is_symlink():
            raise ValueError("Symbolic links are not permitted: " + str(path.relative_to(root)))
        if path.is_file():
            if relative != MANIFEST:
                actual.add(relative)
    expected = set(entries)
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise ValueError("Manifest inventory mismatch: " + json.dumps({"missing": missing, "extra": extra}))
    for relative in sorted(expected):
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts or rel.as_posix() != relative or relative == MANIFEST:
            raise ValueError("Invalid manifest path: " + relative)
        item = entries[relative]
        if not isinstance(item, (str, dict)):
            raise ValueError("Invalid manifest value: " + relative)
        expected_hash = item if isinstance(item, str) else item.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError("Invalid SHA256 entry: " + relative)
        path = root / relative
        if isinstance(item, dict) and "bytes" in item and path.stat().st_size != item["bytes"]:
            raise ValueError("Size mismatch: " + relative)
        if sha256(path) != expected_hash:
            raise ValueError("SHA256 mismatch: " + relative)
    return {"status": "VERIFIED", "files": len(entries),
            "manifest_sha256": sha256(manifest_path),
            "policy": "All release files, including nested manifests; root MANIFEST_SHA256.json and root .git metadata excluded."}


def resolve_python(value):
    resolved = shutil.which(value)
    if not resolved:
        raise ValueError("Python executable not found: " + value)
    # Preserve a virtual environment's executable path: resolving its symlink
    # to the base interpreter silently discards that environment's packages.
    return str(Path(resolved).absolute())


def statuses_in(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "status" and isinstance(item, str):
                yield item
            yield from statuses_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from statuses_in(item)


def incomplete_status(status):
    return status in {"STORED_RESULT_HASH_ONLY", "MISSING_INPUT_OR_EXPECTED",
                      "BUDGET_TIMEOUT", "BUDGET_NOT_RUN", "PARTIAL_REPLAY"} or status.startswith("MISSING_INPUT")


def offline_environment(output):
    guard = output / "runtime_guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import socket, sys\n"
        "sys.dont_write_bytecode = True\n"
        "def _offline(*args, **kwargs):\n"
        "    raise RuntimeError('Network access is disabled by the offline reproduction entry point')\n"
        "socket.socket.connect = _offline\n"
        "socket.socket.connect_ex = _offline\n"
        "socket.create_connection = _offline\n"
        "socket.getaddrinfo = _offline\n"
    )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(guard) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(output / "matplotlib_cache")
    return env


def run_module(module, output, interpreter, software_python, env, timeout_override):
    name = module["name"]
    target = output / name
    log_path = output / "logs" / (name + ".log")
    substitutions = {"output": str(target), "software_python": software_python}
    working_directory = ROOT / module["working_directory"]
    script = ROOT / module["script"]
    if module.get("copy_to_output", False):
        script = target / script.relative_to(working_directory)
        working_directory = target
    command = [interpreter, "-B", str(script)]
    command += [part.format(**substitutions) for part in module["arguments"]]
    budget = timeout_override or module["timeout_seconds"]
    result = {"module": name, "reproduction_level": module["reproduction_level"],
              "declared_scope": module["scope"], "command": command,
              "timeout_seconds": budget, "log": str(log_path.relative_to(output)),
              "output": str(target), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    started = time.monotonic()
    process = None
    try:
        if module.get("copy_to_output", False):
            shutil.copytree(ROOT / module["working_directory"], target)
            # The runtime receipt must come from this execution, not the archive.
            (target / module["report"]).unlink(missing_ok=True)
            result["execution_workspace"] = "Copied module payload; original module remains unchanged."
        if not script.is_file():
            raise FileNotFoundError("Registered module script is missing: " + str(script))
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=working_directory,
                                       stdout=log, stderr=subprocess.STDOUT, env=env,
                                       start_new_session=True)
            try:
                exit_code = process.wait(timeout=budget)
                result.update(exit_code=exit_code, outcome=("COMMAND_COMPLETED" if exit_code == 0 else
                              "PARTIAL_COVERAGE" if exit_code == 2 else "EXECUTION_OR_NUMERICAL_FAILURE"))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                result.update(exit_code=124, outcome="TIMEOUT_PARTIAL_COVERAGE")
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        result.update(exit_code=130, outcome="INTERRUPTED_PARTIAL_COVERAGE")
    except OSError as error:
        result.update(exit_code=1, outcome="EXECUTION_OR_NUMERICAL_FAILURE", error=str(error))
    result["wall_seconds"] = time.monotonic() - started
    native_path = target / module["report"]
    if native_path.is_file():
        try:
            native = json.loads(native_path.read_text())
            observed = sorted(set(statuses_in(native)))
            result.update(native_report=str(native_path.relative_to(output)),
                          native_report_sha256=sha256(native_path), native_statuses=observed)
            for field, expected in module.get("report_required_fields", {}).items():
                if native.get(field) != expected:
                    result.update(outcome="NATIVE_REPORT_CHECK_FAILED", error="Unexpected native report field: " + field)
            if any(status in {"FAIL", "FAILED", "EXECUTION_FAILED", "RESULT_MISMATCH"} for status in observed):
                result.update(outcome="NATIVE_REPORT_CHECK_FAILED", error="Native report contains a failed check.")
            if result["outcome"] == "COMMAND_COMPLETED" and any(incomplete_status(status) for status in observed):
                result["outcome"] = "COMMAND_COMPLETED_WITH_PARTIAL_COVERAGE"
        except (ValueError, OSError) as error:
            result.update(outcome="INVALID_NATIVE_REPORT", error=str(error))
    elif result["exit_code"] == 0:
        result.update(outcome="MISSING_NATIVE_REPORT", error="Module returned zero but did not write its declared report.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List module scopes; performs no replay.")
    parser.add_argument("--verify-only", action="store_true", help="Verify the complete outer SHA256 inventory without executing modules.")
    parser.add_argument("--only", nargs="+", action="append", metavar="MODULE", help="Run selected modules, optionally repeated; otherwise run the default modules.")
    parser.add_argument("--output-dir", type=Path, help="New output directory outside the source package.")
    parser.add_argument("--python", default=sys.executable, help="Python environment for numeric replay dependencies.")
    parser.add_argument("--software-python", help="Python environment containing the optional software/test dependencies.")
    parser.add_argument("--timeout-seconds", type=float, help="Override the per-module time budget (positive seconds).")
    parser.add_argument("--keep-going", action="store_true", help="Continue independent modules after a hard failure; failures remain nonzero.")
    args = parser.parse_args()
    try:
        registry = json.loads((ROOT / "MODULES.json").read_text())
        modules = registry["modules"]
        by_name = {module["name"]: module for module in modules}
        if args.list:
            print(json.dumps({"package_version": registry["package_version"], "modules": modules}, indent=2))
            return 0
        if args.timeout_seconds is not None and args.timeout_seconds <= 0:
            raise ValueError("--timeout-seconds must be positive.")
        selected = list(dict.fromkeys(name for group in args.only for name in group)) if args.only else [module["name"] for module in modules if module["default"]]
        unknown = sorted(set(selected) - set(by_name))
        if unknown:
            raise ValueError("Unknown modules: " + ", ".join(unknown) + "; use --list.")
        integrity = verify_manifest()
        if args.verify_only:
            print(json.dumps(integrity, indent=2))
            return 0
        if args.output_dir is None:
            raise ValueError("--output-dir is required for execution.")
        output = args.output_dir.expanduser().resolve()
        if output == ROOT or ROOT in output.parents:
            raise ValueError("Output directory must be outside the source package.")
        if output.exists():
            raise ValueError("Output directory must be new: " + str(output))
        interpreter = resolve_python(args.python)
        software_python = resolve_python(args.software_python or args.python)
        output.mkdir(parents=True)
        (output / "logs").mkdir()
        env = offline_environment(output)
        report = {"package_version": registry["package_version"], "status": "RUNNING",
                  "scientific_scope": registry["scientific_scope"], "manifest": integrity,
                  "selected_modules": selected, "modules": [],
                  "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "dispatcher_python": platform.python_version(), "numeric_python": interpreter,
                  "offline_guard": "Python socket/DNS blocking inherited by child Python processes; registered entry points perform no download or new model training.",
                  "numeric_library_threads": 1}
        started = time.monotonic()
        write_json(output / REPORT, report)
        hard_failure = False
        partial = False
        for name in selected:
            print("Running " + name + " ...", flush=True)
            result = run_module(by_name[name], output, interpreter, software_python, env, args.timeout_seconds)
            report["modules"].append(result)
            print(name + ": " + result["outcome"] + " (%.2f s)" % result["wall_seconds"], flush=True)
            is_partial = "PARTIAL_COVERAGE" in result["outcome"]
            is_hard = not is_partial and result["outcome"] != "COMMAND_COMPLETED"
            partial = partial or is_partial
            hard_failure = hard_failure or is_hard
            write_json(output / REPORT, report)
            if result["exit_code"] == 130 or (is_hard and not args.keep_going):
                break
        report["not_run"] = selected[len(report["modules"]):]
        partial = partial or bool(report["not_run"])
        report["wall_seconds"] = time.monotonic() - started
        report["all_requested_commands_exit_zero"] = not report["not_run"] and all(x["exit_code"] == 0 for x in report["modules"])
        report["status"] = "EXECUTION_OR_NUMERICAL_FAILURE" if hard_failure else "PARTIAL_COVERAGE" if partial else "REQUESTED_COMMANDS_COMPLETED"
        report["exit_code"] = 1 if hard_failure else 2 if partial else 0
        report["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(output / REPORT, report)
        print(str(output / REPORT), flush=True)
        return report["exit_code"]
    except (ValueError, OSError, KeyError) as error:
        print("Reproduction setup/integrity failure: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
