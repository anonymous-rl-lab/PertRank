"""Fixed TRADE scalar adapter orchestration, without any data/network access.

The caller supplies complete fixed-axis log effects and gene-wise standard errors.
Only an immutable input digest plus validated completed output permits resumption.
No gene filtering, estimator substitution, or tuning is performed here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

RUN = Path(os.environ.get("PERT_RANK_STIM_ASSET_ROOT", "/workspace/scratch/6cdbaefbed9f/stim_run_20261011"))
SOURCES = RUN / "src" / "trade_frozen"
EXPECTED_SOURCE_SHA = {
    "trade_scalar_compact.R": "453a729283e530db8a716472172b7053982f2c8ff93da06185e2db7902556e7f",
    "R_TRADE_univariate.R": "b2fc72c0e5d7c2cde9631e73e33a95cdfed23a31b3a93c5d7141332452426320",
}
R_EXECUTABLE = RUN / "environment" / "bin" / "R"
RUNTIME_FILE = RUN / "environment" / "runtime_versions.json"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def input_digest(effects, standard_errors, targets, genes, replicates):
    """Digest canonical little-endian arrays and unambiguous length-prefixed axes."""
    h = hashlib.sha256()
    for name, array in (("effects", effects), ("standard_errors", standard_errors)):
        a = np.ascontiguousarray(array, dtype="<f8")
        h.update(json.dumps([name, list(a.shape)], separators=(",", ":")).encode())
        h.update(a.tobytes(order="C"))
    h.update(json.dumps([list(targets), list(genes), list(replicates)],
                        ensure_ascii=False, separators=(",", ":")).encode())
    return h.hexdigest()


def scientific_identity():
    for name, expected in EXPECTED_SOURCE_SHA.items():
        if sha256(SOURCES / name) != expected:
            raise ValueError(f"Frozen TRADE source mismatch: {name}")
    runtime = json.loads(RUNTIME_FILE.read_text())
    if runtime["R_version"] != "4.4.3" or runtime["packages"]["ashr"] != "2.2-63":
        raise ValueError("Fixed TRADE runtime unavailable; no replacement is permitted")
    return {"frozen_sources": EXPECTED_SOURCE_SHA,
            "runner_sha256": sha256(__file__),
            "exporter_sha256": sha256(SOURCES / "export_mixtures.R"),
            "runtime_sha256": sha256(RUNTIME_FILE),
            "R_launcher_sha256": sha256(R_EXECUTABLE),
            "sensitivity_source_sha256": sha256(Path(__file__).resolve().parent / "R_TRADE_univariate_squarem.R"), "optimizer_control": {"optmethod": "cxxMixSquarem", "tol": 1e-9, "maxiter": 20000}, "seed": 2026, "component_family": "halfuniform", "prior": "uniform",
            "n_sample": 0, "complete_gene_axis": True}


def analytic_moments(pi, a, b):
    """Independent exact moments for the saved uniform-mixture distribution."""
    pi, a, b = (np.asarray(x, dtype=np.float64).reshape(-1) for x in (pi, a, b))
    if not (len(pi) == len(a) == len(b) and len(pi) > 0):
        raise ValueError("Invalid mixture shape")
    if not all(np.isfinite(x).all() for x in (pi, a, b)):
        raise ValueError("Nonfinite mixture")
    if np.any(pi < 0) or np.any(b < a) or abs(pi.sum() - 1) >= 1e-6:
        raise ValueError("Invalid mixture probability or interval")
    mu = np.dot(pi, (a + b) / 2)
    second = np.dot(pi, (a*a + a*b + b*b) / 3)
    variance = np.dot(pi, (b-a)**2 / 12 + ((a+b)/2-mu)**2)
    if not np.isclose(second, variance + mu*mu, rtol=1e-10, atol=1e-13):
        raise ValueError("Independent analytic second moment mismatch")
    return {"latent_mean": float(mu), "S": float(second), "TI": float(variance)}


def _validated_outputs(out, effects, standard_errors, targets, genes, replicates):
    with (out / "result_weights.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    mixtures = json.loads((out / "result_mixtures.json").read_text())
    expected_n = len(targets) * len(replicates)
    if len(rows) != expected_n or len(mixtures) != expected_n:
        raise ValueError("Incomplete TRADE batch")
    weights = np.empty(effects.shape[:2], dtype=np.float64)
    diagnostics = []
    maximum_error = {"S": 0., "TI": 0., "latent_mean": 0., "weight": 0., "nu": 0.}
    for row, (i, j) in zip(rows, np.ndindex(effects.shape[:2])):
        target, replicate = targets[i], replicates[j]
        if row["target"] != target or int(row["replicate"]) != replicate or int(row["n_genes"]) != len(genes):
            raise ValueError("TRADE output axis mismatch")
        key = f"{target}::{replicate}"
        mixture = mixtures[key]
        g = mixture["fitted_g"]
        exact = analytic_moments(g["pi"], g["a"], g["b"])
        nu = float(np.mean(standard_errors[i, j]**2))
        exact.update(nu=nu, weight=exact["S"]/(exact["S"]+nu))
        for name, wanted in exact.items():
            actual = float(row[name])
            if not np.isfinite(actual) or not np.isclose(actual, wanted, rtol=1e-10, atol=1e-12):
                raise ValueError(f"Independent {name} mismatch at {key}")
            maximum_error[name] = max(maximum_error[name], abs(actual-wanted))
        if not 0 <= exact["weight"] <= 1 or exact["S"] < 0 or nu <= 0:
            raise ValueError("Invalid TRADE scalar")
        weights[i, j] = float(row["weight"])
        diagnostics.append({"target": target, "replicate": replicate,
                            "n_genes": len(genes), "n_components": len(g["pi"]),
                            "loglik": float(row["loglik"]), **exact})
    return weights, {"fits": expected_n, "independent_moment_maxabs": maximum_error,
                     "diagnostics": diagnostics}


def fit_trade_batch(effects, standard_errors, targets, genes, output_dir, replicates=None):
    """Fit [target, replicate, gene] arrays and return weights plus artifact metadata.

    The primary caller uses exactly five replicates, numbered 0--4. Arbitrary
    replicate counts are accepted only so synthetic audits can run tiny batches.
    Completed batches resume only if input, sources, runtime, and all outputs match.
    A failed batch raises RuntimeError and remains visibly incomplete.
    """
    effects = np.asarray(effects, dtype=np.float64)
    standard_errors = np.asarray(standard_errors, dtype=np.float64)
    if effects.ndim != 3 or effects.shape != standard_errors.shape or min(effects.shape) < 1:
        raise ValueError("TRADE inputs must have equal [target, replicate, gene] shapes")
    targets, genes = [str(x) for x in targets], [str(x) for x in genes]
    replicates = list(range(effects.shape[1])) if replicates is None else [int(x) for x in replicates]
    if effects.shape != (len(targets), len(replicates), len(genes)) or len(genes) < 2:
        raise ValueError("TRADE axes do not match arrays")
    if len(set(targets)) != len(targets) or len(set(genes)) != len(genes) or len(set(replicates)) != len(replicates):
        raise ValueError("Duplicate TRADE axis entries")
    if any(any(c in x for c in "\t\n\r") or "::" in x for x in targets + genes):
        raise ValueError("Unsupported separator in TRADE identifiers")
    if not np.isfinite(effects).all() or not np.isfinite(standard_errors).all() or np.any(standard_errors <= 0):
        raise ValueError("TRADE requires finite effects and positive standard errors; no gene filtering")
    if np.any(np.ptp(effects, axis=2) == 0):
        raise ValueError("Constant effect vector; no fallback estimator is permitted")
    identity = scientific_identity()
    digest = input_digest(effects, standard_errors, targets, genes, replicates)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    complete = out / "COMPLETE.json"
    if complete.exists():
        record = json.loads(complete.read_text())
        if record["identity"] != identity or record["input_digest"] != digest:
            raise ValueError("Completed TRADE checkpoint does not match current inputs or identity")
        for name, expected in record["outputs"].items():
            if sha256(out / name) != expected:
                raise ValueError(f"TRADE checkpoint output changed: {name}")
        weights, _ = _validated_outputs(out, effects, standard_errors, targets, genes, replicates)
        return {"weights": weights, "resumed": True, "manifest": record,
                "output_dir": str(out), "complete_path": str(complete)}
    if (out / "STARTED.json").exists():
        old = json.loads((out / "STARTED.json").read_text())
        if old["identity"] != identity or old["input_digest"] != digest:
            raise ValueError("Incomplete TRADE checkpoint belongs to different inputs or identity")
    started = time.monotonic()
    atomic_json(out / "STARTED.json", {"identity": identity, "input_digest": digest,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "shape": list(effects.shape)})
    # Compressed full-axis inputs make a batch independently replayable.
    inp_npz = out / "effects_input.npz"
    tmp = out / "effects_input.tmp.npz"
    np.savez_compressed(tmp, effects=effects, standard_errors=standard_errors,
                        targets=np.asarray(targets), genes=np.asarray(genes),
                        replicates=np.asarray(replicates))
    os.replace(tmp, inp_npz)
    inp = out / "input.tsv"
    with inp.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["target", "replicate", "gene", "effect", "se"])
        for i, j in np.ndindex(effects.shape[:2]):
            writer.writerows((targets[i], replicates[j], gene, format(e, ".17g"), format(se, ".17g"))
                             for gene, e, se in zip(genes, effects[i, j], standard_errors[i, j]))
    input_tsv_sha = sha256(inp)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    commands = [
        [str(R_EXECUTABLE), "--vanilla", "--slave", "--file=" + str(SOURCES / "trade_scalar_compact.R"),
         "--args", str(inp), str(out / "result"), str(Path(__file__).resolve().parent / "R_TRADE_univariate_squarem.R")],
        [str(R_EXECUTABLE), "--vanilla", "--slave", "--file=" + str(SOURCES / "export_mixtures.R"),
         "--args", str(out / "result_mixtures.rds"), str(out / "result_mixtures.json")],
    ]
    try:
        with (out / "run.log").open("w") as log:
            for command in commands:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                if result.returncode:
                    raise RuntimeError(f"R failed with status {result.returncode}; see {out / 'run.log'}")
        weights, validation = _validated_outputs(out, effects, standard_errors, targets, genes, replicates)
        log_text = (out / "run.log").read_text()
        if "TRADE ADAPTER COMPLETED" not in log_text:
            raise RuntimeError("R did not record completed adapter execution")
        atomic_json(out / "INDEPENDENT_VALIDATION.json", validation)
        files = ["effects_input.npz", "result_weights.tsv", "result_mixtures.rds",
                 "result_mixtures.json", "result_runtime.json", "run.log", "INDEPENDENT_VALIDATION.json"]
        record = {"identity": identity, "input_digest": digest, "input_tsv_sha256": input_tsv_sha,
                  "outputs": {name: sha256(out / name) for name in files},
                  "shape": list(effects.shape), "fits": validation["fits"],
                  "seconds": time.monotonic()-started, "warnings_present": "Warning" in log_text,
                  "independent_moment_maxabs": validation["independent_moment_maxabs"],
                  "truth_accessed": False}
        atomic_json(complete, record)
        inp.unlink()  # The exact arrays, TSV digest, and formatter are retained.
        (out / "FAILED.json").unlink(missing_ok=True)
        return {"weights": weights, "resumed": False, "manifest": record,
                "output_dir": str(out), "complete_path": str(complete)}
    except Exception as error:
        atomic_json(out / "FAILED.json", {"identity": identity, "input_digest": digest,
                    "input_tsv_sha256": input_tsv_sha, "error_type": type(error).__name__,
                    "message": str(error), "seconds": time.monotonic()-started})
        raise
