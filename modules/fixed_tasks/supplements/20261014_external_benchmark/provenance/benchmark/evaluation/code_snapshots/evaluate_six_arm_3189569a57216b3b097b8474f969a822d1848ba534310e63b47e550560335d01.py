#!/usr/bin/env python3
"""Frozen native FR benchmark. D4 evaluation requires a verified D3 winner lock.

This is the primary vectorized evaluator. The audit scorer is independent and
is deliberately neither imported nor executed here. All identity alignment is
by exact gene identifiers; neither queries nor targets are selected by scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD_PATH = ROOT / "independent/execution/evaluate_independent.py"
spec = importlib.util.spec_from_file_location("frozen_independent_scoring", OLD_PATH)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
ARMS = ["raw", "euclidean", "pertrank", "fr", "fr_euclidean", "fr_pertrank"]
CONTRASTS = {"fr_minus_raw": (3, 0), "fr_minus_euclidean": (3, 1),
             "pertrank_minus_fr": (2, 3), "fr_pertrank_minus_fr": (5, 3),
             "fr_pertrank_minus_fr_euclidean": (5, 4),
             "pertrank_minus_euclidean": (2, 1)}
PRIMARY_CONTRASTS = list(CONTRASTS)[:5]
DATASETS = ["D3_Rest", "D4_Rest", "D4_Stim8hr"]
SEED = 20261012


def sha(path):
    return old.sha256_file(path)


def save_json(path, value):
    p = Path(path)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def require_dataset(dataset):
    if dataset not in DATASETS:
        raise ValueError("Only opened D3 Rest and D4 Rest/Stim are permitted")


def input_fingerprint(provenance):
    """Normalize either native single-NPZ or prepared counts/axes provenance."""
    if "counts_sha256" in provenance and "axes_sha256" in provenance:
        identity = {k: provenance[k] for k in ("counts_sha256", "axes_sha256")}
        return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if "sha256" in provenance:
        return provenance["sha256"]
    raise ValueError("FR input lacks complete counts/axes or single-file SHA provenance")


def check_lock(path, protocol_path):
    """Validate the winning D3 grid and its exact result files before D4 reads."""
    path = Path(path)
    lock = json.loads(path.read_text())
    if lock.get("status") != "LOCKED_BEFORE_D4_SCORING":
        raise ValueError("D4 scoring requires a D3 winner lock")
    if lock["protocol_sha256"] != sha(protocol_path):
        raise ValueError("Protocol changed after winner selection")
    protocol = json.loads(Path(protocol_path).read_text())
    grid = protocol["fr_grid"]
    records = lock["full_grid_results"]
    if [r["config_id"] for r in records] != [g["id"] for g in grid]:
        raise ValueError("Winner lock must contain all configurations in frozen order")
    for record in records:
        if record["status"] != "COMPLETE":
            raise ValueError("A failed grid cannot yield a strongest-in-grid lock")
        if sha(record["score_file"]) != record["score_sha256"]:
            raise ValueError("D3 score file differs from winner lock")
    values = np.array([r["native_fr_P25"] for r in records])
    ties = np.flatnonzero(values.max() - values <= 1e-12)
    chosen = 0 if 0 in ties else int(ties[0])
    if lock["config_id"] != grid[chosen]["id"]:
        raise ValueError("Winner violates frozen max-P25/tie rule")
    for key in ("rank", "spca_alpha", "lasso_alpha"):
        if lock[key] != grid[chosen][key]:
            raise ValueError("Winner parameter mismatch")
    return lock


def native_fr(path, targets, queries, rep, config, protocol, *, winner=None, selection=None):
    path = Path(path)
    fit_path = path.with_name("fit_report.json")
    report = json.loads(fit_path.read_text())
    if report.get("status") != "COMPLETE":
        raise ValueError("Incomplete FR fit is not evaluable")
    if report["effects_sha256"] != sha(path):
        raise ValueError("FR effects hash mismatch")
    recovery = report["recover_audit"]
    if not recovery["analytic_audit_pass"] or recovery["convergence_warning"]:
        raise ValueError("Failed/nonconverged native FR is not evaluable")
    if report["official_commit"] != protocol["fr_commit"]:
        raise ValueError("FR commit differs from protocol")
    parameters = report["parameters"]
    for key in ("rank", "spca_alpha", "lasso_alpha"):
        if parameters[key] != config[key]:
            raise ValueError(f"FR {key} differs from locked configuration")
    if parameters["seed"] != SEED + rep:
        raise ValueError("FR input-repetition seed differs from frozen seed rule")
    z = load_npz(path)
    ft = np.asarray(z["targets"]).astype(str)
    fg = np.asarray(z["genes"]).astype(str)
    if len(set(ft)) != len(ft) or len(set(fg)) != len(fg):
        raise ValueError("Duplicate FR identity axes")
    if set(ft) != set(targets):
        raise ValueError("FR and frozen task candidate sets differ")
    if not set(queries).issubset(set(fg)):
        raise ValueError("FR omitted query genes; missing genes are not imputed")
    if selection is None:
        raise ValueError("FR scoring requires original input-cell selection")
    st = selection["target_ids"].astype(str)
    target_cells = selection["input_cells"][:, rep] if "input_cells" in selection else selection["A_input"]
    expected_cells = np.concatenate([selection["NTC_input"].ravel(), target_cells.ravel()]).astype(str)
    expected_labels = np.concatenate([np.full(len(selection["NTC_input"]), "__NTC__"), np.repeat(st, target_cells.shape[1])])
    observed_cells = z["cell_ids"].astype(str)
    observed_index = z["target_index"]
    if len(set(expected_cells)) != len(expected_cells) or len(set(observed_cells)) != len(observed_cells):
        raise ValueError("Duplicate target/control cell identities")
    if set(expected_cells) != set(observed_cells):
        raise ValueError("FR input cells differ from frozen matched-input repetition")
    if np.any((observed_index < -1) | (observed_index >= len(ft))):
        raise ValueError("FR cell target index outside valid range")
    observed_labels = np.where(observed_index == -1, "__NTC__", ft[np.maximum(observed_index, 0)])
    cell_to_label = dict(zip(observed_cells, observed_labels))
    if any(cell_to_label[c] != label for c, label in zip(expected_cells, expected_labels)):
        raise ValueError("FR input perturbation assignment differs from frozen selection")
    if not np.array_equal(fg, selection["gene_ids_full"].astype(str)):
        raise ValueError("FR raw gene axis differs from original full gene axis")
    it, ig = {x: i for i, x in enumerate(ft)}, {x: i for i, x in enumerate(fg)}
    effect = z["effect"][np.ix_([it[x] for x in targets], [ig[x] for x in queries])].T
    old.finite("native FR effects", effect)
    return effect.astype(np.float64), {
        "input_rep": rep, "effects_path": str(path.resolve()), "effects_sha256": sha(path),
        "fit_report_path": str(fit_path.resolve()), "fit_report_sha256": sha(fit_path),
        "input_provenance": report["input_provenance"], "input_fingerprint_sha256": input_fingerprint(report["input_provenance"]), "parameters": parameters,
        "runtime_seconds": report["runtime_seconds"], "native_effect_dtype": str(z["effect"].dtype),
        "matched_input_cell_and_target_assignment_check": "PASS", "matched_input_cell_count": len(expected_cells)}


def assemble_pooled(task_path, reps):
    task = load_npz(task_path)
    positions = np.asarray(task["responsive_positions"], dtype=int)
    queries = task["query_gene_ids"][positions].astype(str)
    targets = task["target_ids"].astype(str)
    raw = task["original_effects"][reps][:, :, positions].transpose(0, 2, 1)
    out = dict(raw_scores=raw.astype(np.float64),
               reference_scores=task["truth"][:, positions].T.astype(np.float64),
               w0=task["weight_Euclidean"][reps].astype(np.float64),
               wm=task["weight_PertRank"][reps].astype(np.float64),
               queries=queries, candidates=targets, blocks=task["blocks_responsive"],
               input_reps=np.asarray(reps))
    return out, None


def assemble_independent(prediction_path, reference_path):
    p, r = load_npz(prediction_path), load_npz(reference_path)
    old.validate_inputs(p, r, ROOT / "independent/execution/source_panel.npz", formal=True)
    out = dict(raw_scores=p["input_effects"].T[None].astype(np.float64),
               reference_scores=(r["reference_cells"].mean(1)-r["reference_control_cells"].mean(0)).T,
               w0=p["weights_euclidean"][None], wm=p["weights_pertrank"][None],
               queries=p["query_ids"].astype(str), candidates=p["target_ids"].astype(str),
               blocks=p["query_blocks"], input_reps=np.array([0]))
    return out, r


def validate_compact(z, dataset, kind):
    raw, fr, ref = z["raw_scores"], z["fr_scores"], z["reference_scores"]
    nr, nq, nt = raw.shape
    own = old.own_mask(z["queries"], z["candidates"])
    if fr.shape != raw.shape or ref.shape != (nq, nt) or z["w0"].shape != (nr, nt) or z["wm"].shape != (nr, nt):
        raise ValueError("Compact effect/weight axes mismatch")
    for key in ("raw_scores", "fr_scores", "reference_scores", "w0", "wm"):
        old.finite(key, z[key])
    if any(np.any((z[k] < 0) | (z[k] > 1)) for k in ("w0", "wm")):
        raise ValueError("Reliability weights outside [0,1]")
    if nt < 250 or nq != 300 or len(np.unique(z["blocks"])) != 30:
        raise ValueError("Formal task requires >=250 targets, 300 queries, 30 blocks")
    expected = 1 if dataset == "D3_Rest" or kind == "independent" else 5
    if nr != expected or not np.array_equal(z["input_reps"], np.arange(expected)):
        raise ValueError("Input repetitions differ from frozen protocol")
    z["own_target"] = np.where(own.any(1), own.argmax(1), -1)
    return own


def effects_for_arms(z):
    y, f, w0, wm = z["raw_scores"], z["fr_scores"], z["w0"][:, None], z["wm"][:, None]
    return np.stack([y, y*w0, y*wm, f, f*w0, f*wm], axis=1)


def summarize(scores, boot, prefix="query_only"):
    # scores [input_rep, arm, query]; boot [draw, arm].
    point = scores.mean(axis=(0, 2))
    means = scores.mean(2)
    out = {"arms": {name: {"point": float(point[i]), "input_rep_points": means[:, i].tolist(),
                prefix+"_95_interval": np.quantile(boot[:, i], [.025, .975]).tolist()}
              for i, name in enumerate(ARMS)}, "contrasts": {}}
    for name, (a, b) in CONTRASTS.items():
        draw = boot[:, a] - boot[:, b]
        out["contrasts"][name] = {"point": float(point[a]-point[b]),
            "input_rep_points": (means[:, a]-means[:, b]).tolist(),
            prefix+"_95_interval": np.quantile(draw, [.025, .975]).tolist(),
            prefix+"_99_interval": np.quantile(draw, [.005, .995]).tolist(),
            "in_prespecified_five_contrast_family": name in PRIMARY_CONTRASTS}
    return out


def score_arrays(z, *, draws=2000, reference=None, batch_size=16):
    own = old.own_mask(z["queries"], z["candidates"])
    effects = effects_for_arms(z)
    nr, na, nq, nt = effects.shape
    sampled, qweights = old.query_draws(z["blocks"], draws, SEED)
    arrays = {"sampled_query_blocks": sampled, "query_weights": qweights}
    result, memberships, per_query = {}, {}, {}
    for k in (25, 10):
        pred = old.probabilities(effects.reshape(nr*na, nq, nt), own, k)
        truth = old.probabilities(z["reference_scores"][None], own, k)
        scores = old.query_scores(pred, truth, k)[0].reshape(nr, na, nq)
        boot = qweights @ scores.mean(0).T
        per_query[k] = scores
        result[str(k)] = summarize(scores, boot)
        arrays[f"six_arm_p{k}"] = scores
        arrays[f"query_only_bootstrap_p{k}"] = boot
        memberships[k] = pred
    reference_seed = None
    if reference is not None:
        reference_seed = int.from_bytes(hashlib.sha256(
            ("PertRank-independent-guide-reference-v1|"+str(SEED)+"|D4_Stim8hr").encode()).digest()[:8], "little")
        seeds = np.random.SeedSequence(reference_seed).spawn(2)
        rrng, crng = [np.random.default_rng(s) for s in seeds]
        sampler = old.LaneSampler(reference["reference_lane"])
        csampler = old.LaneSampler(reference["reference_control_lane"][None])
        c = np.asarray(reference["reference_cells"], dtype=np.float64)
        nc = np.asarray(reference["reference_control_cells"], dtype=np.float64)
        joint = {k: np.empty((draws, na)) for k in (25, 10)}
        for start in range(0, draws, batch_size):
            stop = min(start+batch_size, draws)
            n = stop-start
            ref = sampler.means(c, rrng, n).transpose(1, 2, 0)
            ref -= csampler.means(nc[None], crng, n)[0][:, :, None]
            for k in (25, 10):
                qs = old.query_scores(memberships[k], old.probabilities(ref, own, k), k).reshape(n, nr, na, nq).mean(1)
                joint[k][start:stop] = np.einsum("baq,bq->ba", qs, qweights[start:stop])
        for k in (25, 10):
            sj = summarize(per_query[k], joint[k], prefix="joint")
            for section in ("arms", "contrasts"):
                for name, item in sj[section].items():
                    result[str(k)][section][name].update({a:b for a,b in item.items() if a.startswith("joint")})
            arrays[f"joint_bootstrap_p{k}"] = joint[k]
    return result, arrays, reference_seed


def evaluate(args):
    require_dataset(args.dataset)
    protocol = json.loads(args.protocol.read_text())
    # This check intentionally precedes reading FR outputs or task references.
    winner = check_lock(args.winner_lock, args.protocol) if args.dataset.startswith("D4") else None
    config_id = winner["config_id"] if winner else args.config_id
    config = next((g for g in protocol["fr_grid"] if g["id"] == config_id), None)
    if config is None:
        raise ValueError("Unknown frozen FR grid configuration")
    reps = list(range(1 if args.dataset == "D3_Rest" or args.kind == "independent" else 5))
    if len(args.fr_effects) != len(reps):
        raise ValueError("Provide one FR effects file per fixed input repetition, in order")
    if args.kind == "pooled":
        task_path = args.task or ROOT / "pooled_data" / args.dataset / "frozen_task.npz"
        z, reference = assemble_pooled(task_path, reps)
        sources = {"task_path": str(task_path.resolve()), "task_sha256": sha(task_path)}
        selection_path = args.selection or task_path.with_name("selection.npz")
    else:
        if args.dataset != "D4_Stim8hr":
            raise ValueError("Only D4 Stim independent guide task was frozen")
        z, reference = assemble_independent(args.prediction, args.reference)
        sources = {"prediction_path": str(args.prediction.resolve()), "prediction_sha256": sha(args.prediction),
                   "reference_path": str(args.reference.resolve()), "reference_sha256": sha(args.reference)}
        selection_path = args.selection or ROOT/"independent/execution/selection_D4_Stim8hr.npz"
    selection = load_npz(selection_path)
    sources.update(selection_path=str(selection_path.resolve()), selection_sha256=sha(selection_path))
    fr, provenance = [], []
    for rep, path in zip(reps, args.fr_effects):
        effect, source = native_fr(path, z["candidates"], z["queries"], rep, config, protocol, winner=winner, selection=selection)
        fr.append(effect)
        provenance.append(source)
    z["fr_scores"] = np.stack(fr)
    validate_compact(z, args.dataset, args.kind)
    args.out.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    result, arrays, rseed = score_arrays(z, draws=2000, reference=reference)
    z["six_arm_p25"] = arrays["six_arm_p25"]
    z["six_arm_p10"] = arrays["six_arm_p10"]
    np.savez_compressed(args.out / "compact.npz", **z)
    np.savez_compressed(args.out / "evaluation_arrays.npz", **arrays)
    report = dict(status="COMPLETE", created_utc=now(), dataset=args.dataset, kind=args.kind,
        config_id=config_id, config=config, input_reps=reps, arms=ARMS,
        bootstrap_draws=2000, query_seed=SEED, reference_seed=rseed, reference_batch_size=16 if rseed else None,
        shape=dict(input_reps=len(reps), queries=len(z["queries"]), candidates=len(z["candidates"]), blocks=30),
        primary_K=25, sensitivity_K=10, results=result, elapsed_seconds=time.monotonic()-begin,
        scope="Conditional on observed inputs/reference, query panels, control geometry, candidates and donor/state. Query-block resampling only." if not rseed else old.SCOPE,
        simultaneous_intervals="99% marginal percentile intervals form Bonferroni family for five prespecified contrasts at one task/K; PR-minus-E is extra descriptive and excluded from that family.",
        provenance={**sources, "protocol_path": str(args.protocol.resolve()), "protocol_sha256": sha(args.protocol),
            "evaluator_sha256": sha(__file__), "score_primitive_sha256": sha(OLD_PATH), "fr": provenance,
            "winner_lock_path": str(args.winner_lock.resolve()) if winner else None,
            "winner_lock_sha256": sha(args.winner_lock) if winner else None},
        compact_sha256=sha(args.out / "compact.npz"), evaluation_arrays_sha256=sha(args.out / "evaluation_arrays.npz"))
    save_json(args.out / "evaluation_report.json", report)
    print(json.dumps({"report": str(args.out / "evaluation_report.json"), "P25": result["25"]}))
    return report


def select_winner(args):
    protocol = json.loads(args.protocol.read_text())
    if args.out.exists():
        raise FileExistsError("Winner lock already exists; refusing to overwrite")
    reports = {json.loads(p.read_text())["config_id"]: p for p in args.reports}
    if len(reports) != len(protocol["fr_grid"]):
        raise ValueError("All frozen D3 configuration reports are required, with unique IDs")
    records, inputs = [], set()
    for config in protocol["fr_grid"]:
        p = reports[config["id"]]
        report = json.loads(p.read_text())
        if report["status"] != "COMPLETE" or report["dataset"] != "D3_Rest" or report["kind"] != "pooled" or report["input_reps"] != [0]:
            raise ValueError("Winner selection requires completed D3 Rest rep0")
        if report["provenance"]["protocol_sha256"] != sha(args.protocol):
            raise ValueError("D3 score protocol hash mismatch")
        if report["config"] != config:
            raise ValueError("D3 configuration differs from protocol")
        if report["shape"]["candidates"] != 1200 or report["shape"]["queries"] != 300:
            raise ValueError("D3 development scale differs from protocol")
        src = report["provenance"]["fr"][0]
        inputs.add(src["input_fingerprint_sha256"])
        records.append(dict(config_id=config["id"], status="COMPLETE", native_fr_P25=report["results"]["25"]["arms"]["fr"]["point"],
            score_file=str(p.resolve()), score_sha256=sha(p), input_sha256=src["input_fingerprint_sha256"],
            effects_path=src["effects_path"], effects_sha256=src["effects_sha256"], parameters=src["parameters"]))
    if len(inputs) != 1:
        raise ValueError("D3 grid must use identical raw input cells/counts")
    values = np.array([r["native_fr_P25"] for r in records])
    ties = np.flatnonzero(values.max()-values <= 1e-12)
    choice = 0 if 0 in ties else int(ties[0])
    config = protocol["fr_grid"][choice]
    lock = dict(status="LOCKED_BEFORE_D4_SCORING", created_utc=now(), config_id=config["id"],
                **{k:config[k] for k in ("rank", "spca_alpha", "lasso_alpha")}, seed=SEED,
                objective="native FR responsive query bidirectional P25", winner_P25=float(values[choice]),
                tie_rule="within 1e-12 choose official default then first listed", full_grid_results=records,
                D3_input_sha256=next(iter(inputs)), protocol_sha256=sha(args.protocol), selector_sha256=sha(__file__))
    save_json(args.out, lock)
    check_lock(args.out, args.protocol)
    print(json.dumps(lock))


def combine(args):
    if len(args.reports) != 2:
        raise ValueError("Equal-state summary requires exactly D4 Rest and D4 Stim")
    lock = check_lock(args.winner_lock, args.protocol)
    reports = [json.loads(p.read_text()) for p in args.reports]
    if [r["dataset"] for r in reports] != ["D4_Rest", "D4_Stim8hr"] or any(r["kind"] != "pooled" for r in reports):
        raise ValueError("Supply pooled Rest then pooled Stim, the frozen state order")
    if any(r["config_id"] != lock["config_id"] or r["provenance"]["winner_lock_sha256"] != sha(args.winner_lock) for r in reports):
        raise ValueError("Both states must use the same immutable D3 winner")
    arrays, compact = [], []
    for p, r in zip(args.reports, reports):
        ap, cp = p.with_name("evaluation_arrays.npz"), p.with_name("compact.npz")
        if sha(ap) != r["evaluation_arrays_sha256"] or sha(cp) != r["compact_sha256"]:
            raise ValueError("Evaluation arrays changed after task report")
        arrays.append(load_npz(ap))
        compact.append(load_npz(cp))
    # Task-specific panels and blocks differ; do not manufacture query pairing.
    state_seeds = np.random.SeedSequence(SEED).spawn(2)
    saved, qweights = {}, []
    for i, (z, ss) in enumerate(zip(compact, state_seeds)):
        sampled, w = old.query_draws(z["blocks"], 2000, ss)
        saved[f"state_{i}_sampled_blocks"] = sampled
        saved[f"state_{i}_query_weights"] = w
        qweights.append(w)
    result = {}
    for k in (25, 10):
        scores = [a[f"six_arm_p{k}"] for a in arrays]
        boots = [w @ s.mean(0).T for w, s in zip(qweights, scores)]
        boot = np.mean(boots, axis=0)
        # Stacking along query is only an exact equal-size averaging operation;
        # it does not align different query identities across states.
        stacked = np.concatenate(scores, axis=2)
        result[str(k)] = summarize(stacked, boot, prefix="task_stratified_query")
        for i, b in enumerate(boots): saved[f"state_{i}_query_bootstrap_p{k}"] = b
        saved[f"equal_state_bootstrap_p{k}"] = boot
    primary = result["25"]["contrasts"]
    task = [r["results"]["25"]["contrasts"] for r in reports]
    direct = primary["pertrank_minus_fr"]
    combo = primary["fr_pertrank_minus_fr"]
    geometry = primary["fr_pertrank_minus_fr_euclidean"]
    positive_both = lambda name: all(t[name]["point"] > 0 for t in task)
    lower = lambda item: item["task_stratified_query_95_interval"][0]
    gates = {"direct": bool(positive_both("pertrank_minus_fr") and direct["point"] >= .01 and lower(direct) > 0 and np.sum(np.array(direct["input_rep_points"]) > 0) >= 4),
             "complementarity": bool(positive_both("fr_pertrank_minus_fr") and combo["point"] >= .005 and lower(combo) > 0 and positive_both("fr_pertrank_minus_fr_euclidean") and lower(geometry) > 0)}
    gates["continue"] = gates["direct"] or gates["complementarity"]
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "equal_state_arrays.npz", **saved)
    out = dict(status="COMPLETE", created_utc=now(), datasets=[r["dataset"] for r in reports],
        winner_config_id=lock["config_id"], results=result, development_resource_gates=gates,
        seed=SEED, state_seed_spawn_keys=[list(s.spawn_key) for s in state_seeds], draws=2000,
        scope="Task-stratified query-block bootstrap, independent random streams within the two original different panels, then equal-state mean. Conditional on fixed cells/reference and observed D4 donor; not a donor-population interval.",
        paired_input_reps="The same repetition labels are equal-weighted across states for stability only; they are not biological replicate donors.",
        provenance=[{"path": str(p.resolve()), "sha256": sha(p)} for p in args.reports],
        arrays_sha256=sha(args.out / "equal_state_arrays.npz"))
    save_json(args.out / "equal_state_report.json", out)
    print(json.dumps({"report":str(args.out / "equal_state_report.json"), "gates":gates, "P25":result["25"]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--dataset", choices=DATASETS, required=True)
    e.add_argument("--kind", choices=["pooled", "independent"], default="pooled")
    e.add_argument("--task", type=Path)
    e.add_argument("--selection", type=Path, help="Original matched-input cell selection; inferred from task by default")
    e.add_argument("--prediction", type=Path, default=ROOT/"independent/execution/D4_Stim8hr/prediction.npz")
    e.add_argument("--reference", type=Path, default=ROOT/"independent/execution/D4_Stim8hr/reference.npz")
    e.add_argument("--fr-effects", nargs="+", type=Path, required=True)
    e.add_argument("--config-id", help="D3 only; D4 uses winner lock")
    e.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("select-winner")
    s.add_argument("--reports", nargs="+", type=Path, required=True)
    s.add_argument("--out", type=Path, default=ROOT/"FR_D3_WINNER_LOCK.json")
    c = sub.add_parser("combine")
    c.add_argument("--reports", nargs="+", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    for child in (e, s, c):
        child.add_argument("--protocol", type=Path, default=ROOT/"EXECUTION_SPEC.json")
        child.add_argument("--winner-lock", type=Path, default=ROOT/"FR_D3_WINNER_LOCK.json")
    args = p.parse_args()
    {"evaluate": evaluate, "select-winner": select_winner, "combine": combine}[args.command](args)


if __name__ == "__main__":
    main()
