#!/usr/bin/env python3
"""Repeat the boundary-learning pipeline in a fresh, self-contained directory.

Example: python retrain_boundary.py --out ../boundary_retrained
A is imported unchanged from packaged historical assets. B/C are refitted.
If the predeclared internal feasibility gate fails, no outer fitting or held
scoring is performed. This wrapper is a retraining route, not a new validation.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCES = ("D3", "D1", "D2")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def relative_path(name: str) -> Path:
    p = Path(name)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe relative path: {name}")
    return p


def child_environment() -> dict:
    env = os.environ.copy()
    for p in (ROOT / ".runtime", ROOT.parent / "pertrank_neural_rank_20261115" / ".runtime"):
        if p.is_dir():
            old = env.get("PYTHONPATH")
            env["PYTHONPATH"] = str(p) + (os.pathsep + old if old else "")
            break
    return env


def main(output: Path) -> None:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output must be a fresh directory: {output}")
    for protected in (ROOT / "inputs", ROOT / "historical"):
        if protected == output or protected in output.parents:
            raise ValueError(f"Output cannot be inside input assets: {protected}")

    plan_path = ROOT / "PLAN_LOCK.json"
    plan = json.loads(plan_path.read_text())
    copied = {}
    for name, expected in plan["code_hashes"].items():
        p = ROOT / relative_path(name)
        if sha(p) != expected:
            raise RuntimeError(f"Locked source has changed: {name}")
        copied[name] = expected
    for name in ("BOUNDARY_CHECKS.json", "MODEL_CHECKS.json", "check_boundary.py",
                 "check_gate.py", "rebuild_boundary.py", "replay_boundary.py",
                 "retrain_boundary.py", "requirements.txt", "README.md"):
        p = ROOT / name
        if p.is_file():
            copied[name] = sha(p)
    check = ROOT / "BOUNDARY_CHECKS.json"
    if not check.is_file() or not json.loads(check.read_text())["all_passed"]:
        raise RuntimeError("Passing BOUNDARY_CHECKS.json is required")
    if sha(check) != plan["checks_sha256"]:
        raise RuntimeError("Frozen numerical-check record has changed")
    for name, expected in plan["input_hashes"].items():
        if sha(ROOT / relative_path(name)) != expected:
            raise RuntimeError(f"Frozen compact inputs have changed: {name}")
    for source in SOURCES:
        if not (ROOT / "inputs" / f"{source}_neural_inputs.npz").is_file():
            raise FileNotFoundError(f"Missing compact inputs for {source}")
    imported = ROOT / "historical" / "IMPORTED_A.json"
    if not imported.is_file():
        raise FileNotFoundError("Packaged historical/IMPORTED_A.json is required; no sibling-package fallback")
    if sha(imported) != plan["imported_a_sha256"]:
        raise RuntimeError("Imported A manifest has changed")
    historical = json.loads(imported.read_text())
    for name, expected in historical["files"].items():
        if sha(ROOT / relative_path(name)) != expected:
            raise RuntimeError(f"Imported A asset has changed: {name}")
    for directory in (ROOT / "inputs", ROOT / "historical"):
        for p in sorted(directory.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                copied[str(p.relative_to(ROOT))] = sha(p)

    output.mkdir(parents=True, exist_ok=False)
    for name, expected in copied.items():
        dst = output / relative_path(name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dst)
        if sha(dst) != expected:
            raise RuntimeError(f"Copied bytes changed: {name}")

    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    commands = []
    env = child_environment()
    with (output / "retraining.log").open("w") as log:
        def run(arguments, check_return=True):
            command = [sys.executable, "-u", str(output / arguments[0]), *arguments[1:]]
            commands.append(arguments)
            announcement = "Running " + " ".join(arguments)
            print(announcement, flush=True)
            log.write(announcement + "\n"); log.flush()
            return subprocess.run(command, cwd=output, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, check=check_return)
        run(["train_boundary.py", "--prepare"])
        for source in SOURCES:
            run(["train_boundary.py", "--inner", source])
        feasibility_process = run(["train_boundary.py", "--feasibility"], check_return=False)
        feasibility_path = output / "FEASIBILITY.json"
        if not feasibility_path.is_file():
            raise RuntimeError("Feasibility command did not produce FEASIBILITY.json; inspect retraining.log")
        feasibility = json.loads(feasibility_path.read_text())
        proceed = feasibility["proceed_to_outer"]
        if not isinstance(proceed, bool):
            raise TypeError("FEASIBILITY.proceed_to_outer must be a boolean")
        if proceed and feasibility_process.returncode != 0:
            raise RuntimeError("Feasibility claimed success but its process failed")
        if proceed:
            for source in SOURCES:
                run(["train_boundary.py", "--outer", source])
            run(["score_boundary.py"])

    new_plan = output / "PLAN_LOCK.json"
    report = dict(passed=proceed, scientific_success_assessed=False,
                  status="completed_fitting_and_scoring" if proceed else "stopped_at_internal_feasibility",
                  started_utc=started,
                  completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_plan_sha256=sha(plan_path), retrained_plan_sha256=sha(new_plan),
                  copied_file_sha256=copied, commands=commands,
                  feasibility_sha256=sha(feasibility_path), proceed_to_outer=proceed,
                  historical_A_refitted=False, original_artifacts_overwritten=False,
                  note="Same locked code and compact inputs, historical A imported unchanged, fresh B/C fitting. This is reproduction of exploratory development, not independent confirmation.")
    if proceed:
        locks = {}
        for source in SOURCES:
            p = output / "predictions" / f"{source}_PREDICTION_LOCK.json"
            lock = json.loads(p.read_text())
            if lock["plan_sha256"] != sha(new_plan):
                raise RuntimeError(f"Prediction plan mismatch for {source}")
            locks[source] = sha(p)
        report["prediction_lock_sha256"] = locks
        report["summary_sha256"] = sha(output / "results" / "SUMMARY.json")
        report["outer_fitting_performed"] = True
        report["held_scoring_performed"] = True
    else:
        if (output / "results" / "SUMMARY.json").exists():
            raise RuntimeError("Unexpected held summary despite failed feasibility")
        if any((output / "predictions").glob("*_PREDICTION_LOCK.json")):
            raise RuntimeError("Unexpected outer prediction locks despite failed feasibility")
        report["outer_fitting_performed"] = False
        report["held_scoring_performed"] = False
        report["feasibility_process_returncode"] = feasibility_process.returncode
    (output / "RETRAIN_CHECK.json").write_text(json.dumps(report, indent=2) + "\n")
    print(("RETRAINING_COMPLETE " if proceed else "RETRAINING_STOPPED_AT_FEASIBILITY ") + str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    main(parser.parse_args().out)
