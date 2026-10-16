"""AnnData adapters and labelled outputs for the unchanged PertRank engine.

Input cells and matching strata are supplied by the caller. No reference,
query selection, cell subsampling, model tuning or performance evaluation is
performed here. The counts path follows the archived full-axis adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._version import __version__

from .core import (
    ARMS, BETA, K_DIRECTIONS, _controls, _directions, _strings,
    nominate, reliability_weights,
)


def _queries(values: Sequence[str] | str) -> np.ndarray:
    values = [values] if isinstance(values, str) else values
    out = _strings(values, len(values), "queries")
    if not len(out) or len(set(out)) != len(out):
        raise ValueError("queries must be nonempty and unique")
    return out


def _options(top_k: int, seed: int) -> None:
    if not isinstance(top_k, (int, np.integer)) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")


def _read(matrix: Any, rows: Any, input_scale: str) -> np.ndarray:
    x = matrix[rows, :]
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expression values must form a finite cell-by-gene matrix")
    if (x < 0).any():
        raise ValueError("counts and log input require nonnegative values; scaled residuals are unsupported")
    if input_scale == "counts" and (x != np.floor(x)).any():
        raise ValueError("counts input must contain nonnegative integer raw counts")
    return x


def _strata(adata: Any, keys: Sequence[str]) -> np.ndarray:
    if isinstance(keys, str) or keys is None:
        raise ValueError("strata_keys must be an explicit sequence of obs keys; use [] to declare one stratum")
    keys = list(keys)
    if len(set(keys)) != len(keys):
        raise ValueError("strata_keys must be unique")
    if not keys:
        return np.repeat("__all__", adata.shape[0])
    arrays = []
    for key in keys:
        if key not in adata.obs:
            raise ValueError(f"Missing matching-stratum obs column {key!r}")
        arrays.append(_strings(adata.obs[key].to_numpy(), adata.shape[0], key))
    if len(arrays) == 1:
        return arrays[0]
    return np.array([json.dumps(list(parts), separators=(",", ":")) for parts in zip(*arrays)])


def _candidate_rows(stats: Mapping[str, Any], queries: np.ndarray,
                    probabilities: np.ndarray, top_k: int):
    gmap = {gene: i for i, gene in enumerate(stats["gene_ids"])}
    for qi, query in enumerate(queries):
        eligible = np.flatnonzero(stats["target_gene_ids"] != query)
        for di, sign in enumerate((1, -1)):
            for ai, arm in enumerate(ARMS):
                score = sign * stats["effects"][eligible, gmap[query]] * stats["weights"][arm][eligible]
                ordered = np.sort(score)
                for local, target in enumerate(eligible):
                    first = len(score) - int(np.searchsorted(ordered, score[local], side="right")) + 1
                    last = len(score) - int(np.searchsorted(ordered, score[local], side="left"))
                    yield {
                        "method": arm, "query_id": str(query),
                        "direction": "up" if sign > 0 else "down",
                        "target_id": str(stats["target_ids"][target]),
                        "target_gene_id": str(stats["target_gene_ids"][target]),
                        "original_effect": float(stats["effects"][target, gmap[query]]),
                        "weight": float(stats["weights"][arm][target]),
                        "signed_score": float(score[local]), "rank_first": first,
                        "rank_last": last,
                        "tie_inclusion_probability": float(probabilities[ai, qi, di, target]),
                        "top_k": top_k,
                    }


@dataclass
class PertRankResult:
    """Labelled nominations, effects, reliability diagnostics and provenance.

    ``effects`` has target IDs as its index and gene IDs as columns. ``weights``
    contains one row per target and the Unweighted, Euclidean and PertRank
    columns. ``controls`` holds the fitted matched-control moments and leading
    covariance directions. ``statistics`` retains the original reliability-
    input statistics, including source effects, even after ``rank_effects``
    supplies an external readout. Replaying those statistics therefore returns
    the source readout; external estimates remain in ``effects``. Nothing is
    written into AnnData.

    The tie inclusion probabilities describe random boundary-tie selection;
    they are not probabilities of biological validity.
    """
    nominations: pd.DataFrame
    all_candidate_scores: pd.DataFrame
    weights: pd.DataFrame
    effects: pd.DataFrame
    controls: dict[str, Any]
    statistics: dict[str, Any]
    metadata: dict[str, Any]

    def rank_effects(self, effects: pd.DataFrame, queries: Sequence[str] | str | None = None,
                    *, top_k: int | None = None, seed: int | None = None) -> PertRankResult:
        """Reuse these fixed weights on a labelled external effect estimate."""
        return rank_effects(self, effects, queries, top_k=top_k, seed=seed)


def _result(stats: dict[str, Any], queries: np.ndarray, top_k: int,
            seed: int, metadata: dict[str, Any]) -> PertRankResult:
    _options(top_k, seed)
    records, probabilities = nominate(stats, queries, top_k=top_k, seed=seed)
    weights = pd.DataFrame({
        "target_id": stats["target_ids"], "target_gene_id": stats["target_gene_ids"],
        "n_input_cells": stats["n_cells"], "stratum": stats["batch_ids"],
        "energy": stats["energy"], "noise_trace": stats["noise_trace"],
        **stats["weights"],
    }).set_index("target_id", drop=False)
    effects = pd.DataFrame(stats["effects"], index=stats["target_ids"], columns=stats["gene_ids"])
    effects.index.name = "target_id"
    effects.columns.name = "gene_id"
    metadata = {
        "weight_gene_count": int(len(stats["gene_ids"])),
        "input_selection": "All supplied input cells; no sampling or reference access",
        "effect_origin": "source", **metadata,
        "pertrank_version": __version__,
        "k": K_DIRECTIONS, "beta": BETA, "top_k": int(top_k),
        "tie_seed": int(seed), "query_ids": queries.tolist(),
        "targets": int(len(stats["target_ids"])),
        "tie_probability_semantics": "Boundary-tie inclusion, not biological confidence",
    }
    return PertRankResult(
        pd.DataFrame.from_records(records),
        pd.DataFrame.from_records(_candidate_rows(stats, queries, probabilities, top_k)),
        weights, effects, stats.get("control_models", {}), stats, metadata,
    )


def rank(adata: Any, queries: Sequence[str] | str, *, input_scale: str,
         strata_keys: Sequence[str], target_key: str = "target_ids",
         target_gene_key: str = "target_gene_ids", control_key: str = "is_control",
         layer: str | None = None, top_k: int = 25, seed: int = 2026,
         batch_rows: int = 512, full_gene_axis: bool = False,
         library_totals_key: str | None = None) -> PertRankResult:
    """Prioritize perturbations for supplied query genes using fixed k=5,beta=.9.

    Parameters
    ----------
    adata
        AnnData with unique cell and gene IDs. Dense, sparse CSR/CSC and backed
        matrices are read in row blocks; matched controls and one target at a
        time are materialized for exact covariance and sample variance.
    queries
        Gene IDs present on the analysis axis. Own-target candidates are
        excluded independently for every query.
    input_scale
        Required: ``"counts"`` or ``"log"``. Counts use the full-library
        median-control normalization from the paper adapter. Log input must
        already be nonnegative log-normalized expression on one supplied gene
        axis; it is used unchanged and must not be scaled residuals.
    strata_keys
        Required obs keys defining matched controls. ``[]`` explicitly
        declares a single stratum. Counts support one stratum in this release;
        log input supports several, with each target confined to one stratum.
    target_key, target_gene_key, control_key
        Obs columns containing perturbation IDs, corresponding gene IDs, and
        Boolean/0-or-1 control flags. Control rows require nonempty ID values.
    layer
        Expression layer, or ``None`` for X. The selected matrix alone is read.
    full_gene_axis
        Counts require ``True``: the caller certifies that row totals can be
        calculated from the complete raw library axis, before gene filtering.
        This declaration cannot detect a previously subsetted matrix. Optional
        library_totals_key provides a check against retained full-library totals.
    library_totals_key
        Optional obs column of original full-library totals; when supplied for
        counts it must exactly equal the selected matrix's row sums.

    Returns
    -------
    PertRankResult
        Three method arms, labelled effects, target weights and control models.
        The input AnnData is not modified. ``top_k`` must fit every query's
        eligible candidate pool; it is never silently reduced.
    """
    if input_scale not in {"counts", "log"}:
        raise ValueError("input_scale must explicitly be 'counts' or 'log'")
    _options(top_k, seed)
    if not isinstance(batch_rows, (int, np.integer)) or isinstance(batch_rows, bool) or batch_rows < 1:
        raise ValueError("batch_rows must be a positive integer")
    if not all(hasattr(adata, key) for key in ("shape", "obs", "var_names", "obs_names", "X")):
        raise TypeError("adata must provide the AnnData matrix and axis interface")
    n, g = adata.shape
    if n == 0 or g == 0:
        raise ValueError("AnnData must contain cells and genes")
    genes = _strings(adata.var_names, g, "var_names")
    cells = _strings(adata.obs_names, n, "obs_names")
    if len(set(genes)) != g or len(set(cells)) != n:
        raise ValueError("AnnData gene and cell IDs must be unique")
    for key in (target_key, target_gene_key, control_key):
        if key not in adata.obs:
            raise ValueError(f"Missing obs column {key!r}")
    targets = _strings(adata.obs[target_key].to_numpy(), n, target_key)
    own = _strings(adata.obs[target_gene_key].to_numpy(), n, target_gene_key)
    ctrl = _controls(adata.obs[control_key].to_numpy(), n)
    batches = _strata(adata, strata_keys)
    names = np.unique(targets[~ctrl])
    if not len(names) or set(names) & set(targets[ctrl]):
        raise ValueError("Distinct perturbation and control target IDs are required")
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Missing AnnData layer {layer!r}")
    matrix = adata.X if layer is None else adata.layers[layer]
    if matrix is None:
        raise ValueError("The selected expression matrix is empty")
    queries = _queries(queries)
    if input_scale == "counts":
        if full_gene_axis is not True:
            raise ValueError("counts require full_gene_axis=True and the complete raw library gene axis")
        if len(set(batches)) != 1:
            raise ValueError("counts input supports one prespecified matching stratum; use separately normalized log input for an explicit multi-stratum design")
        if ctrl.sum() < 6:
            raise ValueError("At least six matched input controls are required")
        totals = np.zeros(n)
        csums = np.zeros(g)
        for start in range(0, n, batch_rows):
            stop = min(start + batch_rows, n)
            x = _read(matrix, slice(start, stop), input_scale)
            totals[start:stop] = x.sum(1)
            csums += x[ctrl[start:stop]].sum(0)
        if (totals <= 0).any():
            raise ValueError("Zero-library cells must be QC-filtered before input")
        if library_totals_key is not None:
            if library_totals_key not in adata.obs:
                raise ValueError(f"Missing full-library total column {library_totals_key!r}")
            if not np.array_equal(np.asarray(adata.obs[library_totals_key], dtype=float), totals):
                raise ValueError("Provided library totals differ from row sums; supply the full raw gene axis")
        keep = np.flatnonzero(csums > 0)
        scales = {str(batches[0]): float(np.median(totals[ctrl]))}
    else:
        if library_totals_key is not None:
            raise ValueError("library_totals_key applies only to counts input")
        keep = np.arange(g)
        totals, scales = None, {}
    if set(queries) - set(genes[keep]):
        raise ValueError("Queries must be present on the input-control gene axis")
    models = {}

    def expression(rows: np.ndarray, control: bool = False) -> np.ndarray:
        out = np.empty((len(rows), len(keep)), dtype=float)
        for start in range(0, len(rows), batch_rows):
            ix = rows[start:start + batch_rows]
            x = _read(matrix, ix, input_scale)[:, keep]
            if input_scale == "counts":
                x = np.log1p(x * (scales[str(batches[ix[0]])] / totals[ix])[:, None])
                if control:
                    x = x.astype(np.float32).astype(np.float64)
            out[start:start + len(ix)] = x
        return out

    for batch in np.unique(batches[~ctrl]):
        ci = np.flatnonzero(ctrl & (batches == batch))
        if len(ci) < 6:
            raise ValueError(f"Stratum {batch} needs at least six matched input controls")
        xc = expression(ci, control=True)
        mean = xc.mean(0)
        directions, eigenvalues = _directions(xc - mean)
        models[str(batch)] = dict(directions=directions, eigenvalues=eigenvalues,
                                 mean=mean, variance=xc.var(0, ddof=1), n_control=len(ci))
    stats = []
    for name in names:
        rows = np.flatnonzero((~ctrl) & (targets == name))
        bs, gs = np.unique(batches[rows]), np.unique(own[rows])
        if len(bs) != 1:
            raise ValueError(f"Target {name} spans matching strata; supply a prespecified pooling design")
        if len(gs) != 1 or len(rows) < 2:
            raise ValueError(f"Target {name} needs >=2 cells and one target_gene_id")
        model = models[str(bs[0])]
        xt = expression(rows)
        effect = xt.mean(0) - model["mean"]
        variance = xt.var(0, ddof=1)
        py = effect @ model["directions"]
        pv = (xt @ model["directions"]).var(0, ddof=1)
        depth = len(rows)
        stats.append((effect, float(effect @ effect),
                      float(variance.sum() / depth + model["variance"].sum() / model["n_control"]),
                      py * py, pv / depth + model["eigenvalues"] / model["n_control"],
                      depth, str(bs[0]), str(gs[0])))
    statistics = dict(
        gene_ids=genes[keep], target_ids=names, effects=np.array([s[0] for s in stats]),
        energy=np.array([s[1] for s in stats]), noise_trace=np.array([s[2] for s in stats]),
        direction_energy=np.array([s[3] for s in stats]), direction_noise=np.array([s[4] for s in stats]),
        n_cells=np.array([s[5] for s in stats]), batch_ids=np.array([s[6] for s in stats]),
        target_gene_ids=np.array([s[7] for s in stats]), control_models=models,
    )
    statistics["weights"] = reliability_weights(statistics["energy"], statistics["noise_trace"],
                                                statistics["direction_energy"], statistics["direction_noise"])
    metadata = dict(
        input_format="AnnData", input_scale=input_scale, layer=layer, strata_keys=list(strata_keys),
        normalization_scales=scales, full_gene_count=g, weight_gene_positions=keep.tolist(),
        n_input_controls=int(ctrl.sum()), n_input_target_cells=int((~ctrl).sum()),
        target_depths=sorted(set(statistics["n_cells"].tolist())), batch_rows=int(batch_rows),
        full_gene_axis_declared=bool(full_gene_axis), library_totals_key=library_totals_key,
        normalization=("log1p(count * median full-library input-control total / full-library total)"
                       if input_scale == "counts" else "Caller-supplied log expression, unchanged"),
        control_precision=("float32 per log value, then float64 moments" if input_scale == "counts" else "float64"),
        target_precision="float64",
    )
    return _result(statistics, queries, top_k, seed, metadata)


def from_statistics(statistics: Mapping[str, Any], queries: Sequence[str] | str,
                    *, top_k: int = 25, seed: int = 2026,
                    effect_gene_subset: bool = False,
                    weight_gene_count: int | None = None) -> PertRankResult:
    """Rank archived sufficient statistics without reading or fitting cells.

    Required keys are ``gene_ids``, ``target_ids``, ``target_gene_ids``,
    ``effects``, ``energy``, ``noise_trace``, ``direction_energy`` and
    ``direction_noise``. Direction arrays must contain the five original
    matched-control directions. Optional ``n_cells``, ``batch_ids`` and
    ``control_models`` retain provenance. Weights are always recomputed using
    the fixed paper equation; supplied weights are not trusted or reused.
    By default, effects must contain the complete original weight gene axis
    and their squared norm must equal ``energy``. For an explicitly declared
    stored subset of effect columns, set ``effect_gene_subset=True`` and
    provide the original ``weight_gene_count``. Whole-axis energy, noise and
    projected statistics remain unchanged; they cannot be derived from the
    subset. The subset squared norm must not exceed whole-axis energy.
    A ``rank_effects`` result retains these original source statistics, so
    passing its ``statistics`` here replays the source effects, not its external
    readout. Use ``rank_effects`` again to combine saved external effects.
    """
    required = {"gene_ids", "target_ids", "target_gene_ids", "effects", "energy",
                "noise_trace", "direction_energy", "direction_noise"}
    missing = required - statistics.keys()
    if missing:
        raise ValueError("Missing sufficient statistics: " + ", ".join(sorted(missing)))
    _options(top_k, seed)
    effect = np.asarray(statistics["effects"], dtype=float)
    if effect.ndim != 2 or min(effect.shape) == 0 or not np.isfinite(effect).all():
        raise ValueError("effects must be a nonempty finite target-by-gene matrix")
    nt, ng = effect.shape
    stats = {key: np.asarray(statistics[key]).copy() for key in required}
    for key, length in (("gene_ids", ng), ("target_ids", nt), ("target_gene_ids", nt)):
        stats[key] = _strings(statistics[key], length, key)
    if len(set(stats["gene_ids"])) != ng or len(set(stats["target_ids"])) != nt:
        raise ValueError("gene_ids and target_ids must be unique")
    stats["effects"] = effect.copy()
    for key in ("energy", "noise_trace", "direction_energy", "direction_noise"):
        stats[key] = np.asarray(statistics[key], dtype=float).copy()
    if stats["energy"].shape != (nt,) or stats["noise_trace"].shape != (nt,):
        raise ValueError("energy and noise_trace must align with target IDs")
    if stats["direction_energy"].shape != (nt, K_DIRECTIONS):
        raise ValueError("direction_energy must have exactly five frozen directions per target")
    if not isinstance(effect_gene_subset, (bool, np.bool_)):
        raise ValueError("effect_gene_subset must be a Boolean declaration")
    if weight_gene_count is not None and (
            not isinstance(weight_gene_count, (int, np.integer))
            or isinstance(weight_gene_count, bool) or weight_gene_count < 1):
        raise ValueError("weight_gene_count must be a positive integer")
    observed_energy = np.sum(effect * effect, axis=1)
    if effect_gene_subset:
        if (not isinstance(weight_gene_count, (int, np.integer)) or isinstance(weight_gene_count, bool)
                or weight_gene_count < ng):
            raise ValueError("A declared effect gene subset requires integer weight_gene_count >= supplied gene count")
        if np.any(observed_energy > stats["energy"] + 1e-6 * np.maximum(1, stats["energy"])):
            raise ValueError("Effect subset energy exceeds the saved full-axis energy")
    else:
        if weight_gene_count is not None and weight_gene_count != ng:
            raise ValueError("Full-axis effects require weight_gene_count to equal the supplied gene count")
        weight_gene_count = ng
        if not np.allclose(stats["energy"], observed_energy, rtol=1e-6, atol=1e-10):
            raise ValueError("energy does not match the full supplied effect gene axis")
    stats["weights"] = reliability_weights(stats["energy"], stats["noise_trace"],
                                           stats["direction_energy"], stats["direction_noise"])
    depth = np.asarray(statistics.get("n_cells", np.full(nt, np.nan)), dtype=float)
    if depth.shape != (nt,) or ("n_cells" in statistics and
                               (not np.isfinite(depth).all() or (depth < 2).any() or (depth != np.floor(depth)).any())):
        raise ValueError("n_cells must align to targets and contain integer depths >=2")
    stats["n_cells"] = depth
    stats["batch_ids"] = _strings(statistics.get("batch_ids", np.repeat("not_recorded", nt)), nt, "batch_ids")
    stats["control_models"] = statistics.get("control_models", {})
    return _result(stats, _queries(queries), top_k, seed,
                   {"input_format": "Archived sufficient statistics", "input_scale": "archived",
                    "source": "Caller-provided frozen sufficient statistics; no model fit",
                    "effect_gene_subset": bool(effect_gene_subset),
                    "weight_gene_count": int(weight_gene_count), "effect_gene_count": ng,
                    "input_selection": "Archived selected inputs; no cells read or resampled"})


def rank_effects(result: PertRankResult, effects: pd.DataFrame,
                 queries: Sequence[str] | str | None = None, *, top_k: int | None = None,
                 seed: int | None = None) -> PertRankResult:
    """Apply existing weights to labelled external target-by-gene effects.

    Every fitted target must occur exactly once in the index. Columns may be
    a subset of the fitted gene axis, but must contain every requested query.
    Rows and columns are aligned by identifiers, never by their position.
    Source weights and their diagnostics stay fixed; uncertainty is not
    re-estimated for the external estimator. The returned ``statistics`` retain
    the original source effects and axis; ``effects`` contain the external
    readout. Source and returned objects do not share mutable statistics.
    Repeated calls may use any gene retained in the original source statistics.
    """
    if not isinstance(result, PertRankResult) or not isinstance(effects, pd.DataFrame):
        raise TypeError("Supply a PertRankResult and a target-by-gene pandas DataFrame")
    targets = _strings(effects.index, len(effects), "external target IDs")
    genes = _strings(effects.columns, effects.shape[1], "external gene IDs")
    if len(set(targets)) != len(targets) or len(set(genes)) != len(genes):
        raise ValueError("External effect target and gene IDs must be unique")
    if set(targets) != set(result.statistics["target_ids"]):
        raise ValueError("External effects must contain exactly the fitted target IDs")
    if not set(genes).issubset(set(result.statistics["gene_ids"])):
        raise ValueError("External effect genes must belong to the fitted gene namespace and axis")
    aligned = effects.copy()
    aligned.index, aligned.columns = targets, genes
    aligned = aligned.loc[result.statistics["target_ids"]]
    values = aligned.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("External effects must contain finite values")
    queries = _queries(result.metadata["query_ids"] if queries is None else queries)
    if set(queries) - set(genes):
        raise ValueError("External effects must include every requested query")
    source_statistics = deepcopy(result.statistics)
    stats = {**source_statistics, "effects": values, "gene_ids": genes}
    metadata = {**deepcopy(result.metadata), "effect_estimator": "external labelled effects",
                "effect_origin": "external",
                "weight_source": "Unchanged source-cell reliability statistics",
                "weight_gene_count": result.metadata["weight_gene_count"],
                "source_effect_gene_count": len(source_statistics["gene_ids"]),
                "external_effect_gene_count": len(genes), "effect_gene_count": len(genes)}
    answer = _result(stats, queries, result.metadata["top_k"] if top_k is None else top_k,
                     result.metadata["tie_seed"] if seed is None else seed, metadata)
    # Reliability statistics and readout are distinct: preserve the source
    # energy/effect relationship and its labelled control-model gene axis.
    answer.statistics = source_statistics
    return answer
