# CTLA4: a complete descriptive nomination example

This example follows the previously named CTLA4 query (ENSG00000163599) in D4 Stim8hr. It uses the first frozen 20-cell input group and reports both directions. The query, group and direction were not chosen by their gain. CTLA4 is in the responsive panel. All lists use the same candidate pool and deep RNA reference.

The tables show what the already-completed method changes in an experimental shortlist. Internal deep-reference support is an annotation of the existing endpoint, not independent biological validation or a new scientific increment.

| Direction | Common positions | Changed in / out | Euclidean P@25 | PertRank P@25 | Difference |
|---|---:|---:|---:|---:|---:|
| up | 19 | 6 / 6 | 0.520 | 0.640 | +0.120 |
| down | 17 | 8 / 8 | 0.200 | 0.280 | +0.080 |

Across all five existing input groups and both directions, CTLA4's mean PertRank-minus-Euclidean difference is +0.028. `CTLA4_all_groups.csv` retains all ten values, each checked against the registered scoring output. The first group's larger or smaller difference must not be presented as a typical query gain.

Both complete Euclidean and PertRank lists are shown below. `CTLA4_top25_lists.csv` also includes the unweighted lists, effects, weights, scores and tie probabilities. `CTLA4_changed_candidates.csv` includes every changed candidate; `CTLA4_all_candidates.csv` retains the full eligible pool, including candidates omitted by both methods.

| Rank | Up: Euclidean | Up: PertRank | Down: Euclidean | Down: PertRank |
|---:|---|---|---|---|
| 1 | NFKBIL1 | NFKBIL1 | UBXN1 | UBXN1 |
| 2 | TENT2 | CCDC138 | TESK2 | TESK2 |
| 3 | CCDC138 | TENT2 | PPDPF | PPDPF |
| 4 | DENND1B | EDC3 | ZNF426 | ZNF426 |
| 5 | EDC3 | PARP14 | CSTF2T | ATP7A |
| 6 | PARP14 | FHL1 | CASZ1 | DIS3L2 |
| 7 | NCKAP5L | DENND1B | MMP9 | TADA2B |
| 8 | PAK1 | PAK1 | DIS3L2 | HSDL2 |
| 9 | SFTPA1 | SFTPA1 | IFT27 | UBL7 |
| 10 | FHL1 | NCKAP5L | CDKN2C | WDR54 |
| 11 | CLCC1 | EIF1AX | NEK6 | RAB27B |
| 12 | LRP2BP | LRP2BP | ATP7A | IFT27 |
| 13 | EIF1AX | ZMAT1 | SH3YL1 | ZNF148 |
| 14 | LYPD3 | CLCC1 | APBB3 | MMP9 |
| 15 | C16orf91 | ZNF232 | TADA2B | ZNFX1 |
| 16 | GPAA1 | HIST1H2BI | CSTA | DHFR2 |
| 17 | RNF181 | C16orf91 | PAK3 | DLG4 |
| 18 | ZMAT1 | DEFA6 | LGALS9 | CLDN5 |
| 19 | KIAA0930 | PRR13 | MTM1 | ELOF1 |
| 20 | HIST1H2BI | ZNF165 | RAB27B | PAK3 |
| 21 | ZNF232 | MYB | HSDL2 | VHL |
| 22 | CUEDC2 | PIGB | ZNF148 | IRF9 |
| 23 | DEFA6 | TCHP | WDR54 | LGALS9 |
| 24 | ATXN7L3 | DDX58 | PDE1B | MTM1 |
| 25 | ECHS1 | LYPD3 | DHFR2 | VASP |

An up list contains perturbations with the highest signed CTLA4 RNA scores; a down list reverses the sign. A target's weight multiplies its complete effect profile, while each query's original effect determines its ordering. Weights are not biological confidence probabilities. Reference overlap uses independent expected boundary inclusion, not shared deterministic tie keys. The own-gene candidate is excluded by canonical identity.

`build_CTLA4_example.py` rebuilds these display tables from the completed run's predictions, reference and query context. The source hashes are recorded in `CTLA4_PROVENANCE.json`. This display is separate from `run_counts.py`, which generates new-data nominations without reading reference outcomes.
