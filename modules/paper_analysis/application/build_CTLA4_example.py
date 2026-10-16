#!/usr/bin/env python3
"""Export a fixed descriptive case from completed, frozen Stim predictions.

Query CTLA4 was named in the application discussion, before this export.
The first input group and both directions are used without selecting on gain.
This script is separate from the new-input application and uses the existing
internal deep RNA reference only for an explicit descriptive annotation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from pertrank.core import _boundary_probabilities
from run_counts import write_csv

QUERY, GROUP, K = "ENSG00000163599", 0, 25


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--query-context", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pred_path = args.run_root/"input/main_nontrade_predictions.npz"
    ref_path = args.run_root/"evaluation/main_reference.npz"
    pred, ref, q = load(pred_path), load(ref_path), load(args.query_context)
    assert np.array_equal(pred["target_ids"], ref["target_ids"])
    assert np.array_equal(pred["target_ids"], q["target_ids"])
    assert np.array_equal(pred["query_gene_ids"], ref["query_gene_ids"])
    assert np.array_equal(pred["query_gene_ids"], q["query_gene_ids"])
    qi = np.flatnonzero(pred["query_gene_ids"] == QUERY)
    if len(qi) != 1:
        raise ValueError("CTLA4 is unavailable; do not replace it by an outcome-selected query")
    qi = int(qi[0]); keep = np.flatnonzero(q["target_ids"] != QUERY)
    targets, names = q["target_ids"][keep], q["target_names"][keep]
    allrows, lists, changed, summary, groups = [], [], [], [], []
    display = {}
    for di, sign in enumerate((1, -1)):
        direction = "up" if sign > 0 else "down"
        effect = pred["original_effects"][GROUP, keep, qi]
        truth = ref["effects"][keep, qi]
        rp = _boundary_probabilities(sign*truth, K)
        for group in range(5):
            gp = {arm: _boundary_probabilities(sign*pred["prediction_"+arm][group, keep, qi], K)
                  for arm in ("raw", "Euclidean", "PertRank")}
            groups.append(dict(input_group_zero_based=group, direction=direction,
                raw_P25=float(gp["raw"]@rp/K), Euclidean_P25=float(gp["Euclidean"]@rp/K),
                PertRank_P25=float(gp["PertRank"]@rp/K),
                delta_P25=float((gp["PertRank"]-gp["Euclidean"])@rp/K)))
        keys = np.array([hashlib.sha256(f"2026|{QUERY}|{direction}|{t}".encode()).hexdigest() for t in targets])
        probs, chosen, scores = {}, {}, {}
        for arm in ("raw", "Euclidean", "PertRank"):
            scores[arm] = sign*pred["prediction_"+arm][GROUP, keep, qi]
            probs[arm] = _boundary_probabilities(scores[arm], K)
            chosen[arm] = np.lexsort((keys, -scores[arm]))[:K]
            display[direction, arm] = names[chosen[arm]].tolist()
            for rank, i in enumerate(chosen[arm], 1):
                lists.append(dict(direction=direction, method=arm, rank=rank, target_id=str(targets[i]),
                     target_name=str(names[i]), original_effect=float(effect[i]),
                     weight=float(pred["weight_"+arm][GROUP, keep[i]]), signed_score=float(scores[arm][i]),
                     nomination_tie_probability=float(probs[arm][i]), internal_reference_tie_probability=float(rp[i])))
        e, p = probs["Euclidean"], probs["PertRank"]
        e_set, p_set = set(chosen["Euclidean"].tolist()), set(chosen["PertRank"].tolist())
        summary.append(dict(direction=direction, candidates=len(keep), common=len(e_set&p_set),
                    swapped_in=len(p_set-e_set), swapped_out=len(e_set-p_set),
                    raw_P25=float(probs["raw"]@rp/K), Euclidean_P25=float(e@rp/K),
                    PertRank_P25=float(p@rp/K), delta_P25=float((p-e)@rp/K),
                    boundary_fractional_candidates_E=int(((e>0)&(e<1)).sum()),
                    boundary_fractional_candidates_P=int(((p>0)&(p<1)).sum())))
        for i in range(len(keep)):
            change = ("shared" if i in p_set&e_set else "in" if i in p_set else "out" if i in e_set else "unselected")
            row = dict(direction=direction, target_id=str(targets[i]), target_name=str(names[i]),
                    list_membership=change, original_effect=float(effect[i]), internal_reference_effect=float(truth[i]),
                    Euclidean_weight=float(pred["weight_Euclidean"][GROUP, keep[i]]),
                    PertRank_weight=float(pred["weight_PertRank"][GROUP, keep[i]]),
                    Euclidean_signed_score=float(scores["Euclidean"][i]),
                    PertRank_signed_score=float(scores["PertRank"][i]),
                    Euclidean_inclusion_probability=float(e[i]), PertRank_inclusion_probability=float(p[i]),
                    internal_reference_inclusion_probability=float(rp[i]),
                    overlap_change_contribution=float((p[i]-e[i])*rp[i]/K))
            allrows.append(row)
            if change in ("in", "out"):
                changed.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/"CTLA4_all_candidates.csv", allrows)
    write_csv(args.out/"CTLA4_top25_lists.csv", lists)
    write_csv(args.out/"CTLA4_changed_candidates.csv", changed)
    write_csv(args.out/"CTLA4_summary.csv", summary)
    write_csv(args.out/"CTLA4_all_groups.csv", groups)
    registered = load(args.run_root/"evaluation/registered_scores.npz")
    rq = int(np.flatnonzero(registered["main_query_gene_ids"] == QUERY)[0])
    for row in groups:
        for arm in ("raw", "Euclidean", "PertRank"):
            ai = int(np.flatnonzero(registered["main_arm_names"] == arm)[0])
            value = registered["main_scores"][ai, row["input_group_zero_based"], rq,
                                              0 if row["direction"] == "up" else 1]
            assert abs(value-row[arm+"_P25"]) < 1e-12, (arm, row, value)
    provenance = dict(query_id=QUERY, query_symbol="CTLA4", input_group_zero_based=GROUP,
        selection="Previously named biological query; first input group; both directions; no selection by gain",
        top_k=K, query_in_responsive=bool(qi in q["responsive_positions"]),
        query_in_tercile=bool(qi in q["tercile_positions"]),
        all_ten_query_scores_match_registered=True,
        all_groups_mean_delta_P25=float(np.mean([s["delta_P25"] for s in groups])),
        sources={"predictions":sha(pred_path),"reference":sha(ref_path),"query_context":sha(args.query_context)},
        outcome="Descriptive reuse of completed internal RNA task, not independent support or a new hypothesis test")
    (args.out/"CTLA4_PROVENANCE.json").write_text(json.dumps(provenance, indent=2)+"\n")
    text = ["# CTLA4: a complete descriptive nomination example", "",
        "This example follows the previously named CTLA4 query (ENSG00000163599) in D4 Stim8hr. It uses the first frozen 20-cell input group and reports both directions. The query, group and direction were not chosen by their gain. CTLA4 is in the responsive panel. All lists use the same candidate pool and deep RNA reference.", "",
        "The tables show what the already-completed method changes in an experimental shortlist. Internal deep-reference support is an annotation of the existing endpoint, not independent biological validation or a new scientific increment.", "",
        "| Direction | Common positions | Changed in / out | Euclidean P@25 | PertRank P@25 | Difference |",
        "|---|---:|---:|---:|---:|---:|"]
    for s in summary:
        text.append(f"| {s['direction']} | {s['common']} | {s['swapped_in']} / {s['swapped_out']} | {s['Euclidean_P25']:.3f} | {s['PertRank_P25']:.3f} | {s['delta_P25']:+.3f} |")
    text += ["", f"Across all five existing input groups and both directions, CTLA4's mean PertRank-minus-Euclidean difference is {provenance['all_groups_mean_delta_P25']:+.3f}. `CTLA4_all_groups.csv` retains all ten values, each checked against the registered scoring output. The first group's larger or smaller difference must not be presented as a typical query gain."]
    text += ["", "Both complete Euclidean and PertRank lists are shown below. `CTLA4_top25_lists.csv` also includes the unweighted lists, effects, weights, scores and tie probabilities. `CTLA4_changed_candidates.csv` includes every changed candidate; `CTLA4_all_candidates.csv` retains the full eligible pool, including candidates omitted by both methods.", "",
             "| Rank | Up: Euclidean | Up: PertRank | Down: Euclidean | Down: PertRank |",
             "|---:|---|---|---|---|"]
    for rank in range(K):
        text.append(f"| {rank+1} | {display['up','Euclidean'][rank]} | {display['up','PertRank'][rank]} | {display['down','Euclidean'][rank]} | {display['down','PertRank'][rank]} |")
    text += ["", "An up list contains perturbations with the highest signed CTLA4 RNA scores; a down list reverses the sign. A target's weight multiplies its complete effect profile, while each query's original effect determines its ordering. Weights are not biological confidence probabilities. Reference overlap uses independent expected boundary inclusion, not shared deterministic tie keys. The own-gene candidate is excluded by canonical identity.", "",
             "`build_CTLA4_example.py` rebuilds these display tables from the completed run's predictions, reference and query context. The source hashes are recorded in `CTLA4_PROVENANCE.json`. This display is separate from `run_counts.py`, which generates new-data nominations without reading reference outcomes.", ""]
    (args.out/"CTLA4_worked_example.md").write_text("\n".join(text))
    print(json.dumps(dict(provenance=provenance, summary=summary), indent=2))


if __name__ == "__main__":
    main()
