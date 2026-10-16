# Manuscript display tables

These 97 CSV files preserve the displayed values of the current v25.2 manuscript: three main-text tables and 94 Supplementary Information tables. `TABLE_INDEX.json` maps each file to its manuscript section and source line. An index label starting “Displayed table in” describes an unnumbered table; it does not assign a new Supplementary Table number.

The CSVs are frozen display sources, including the manuscript’s original rounding. Their presence does not establish that every displayed cell is automatically regenerated or checked by the default replay. The package’s statistical computations and their starting points are specified by `MODULES.json`, `docs/REPRODUCTION_SCOPE.md` and the fresh module receipts.

## Historical verification table

`PertRank_Supplement_v25.2_table_060.csv` is manuscript **Table SN9.1, Current reproduction scope**, retained verbatim as part of the manuscript record. It describes work and verification completed during the research and earlier releases, including full-state or raw-CSR checks. The compact v1.5.2 package starts some of those analyses from saved query scores or other intermediates. This CSV therefore records historical completion; it is not a claim that v1.5.2 reruns those upstream stages. Use the current module registry and reproduction-scope document to identify executable steps and retained inputs.

A14 remains a current-manuscript historical result with unavailable per-cell input, explicitly archived under `modules/early_intermediates/saved_only/A14/`. Original A19 and reconstructed A19-R retain separate identities. Neither record is silently upgraded by successful execution of another module.

The v1.5.2 correction changes index descriptions and scope documentation only. All 97 scientific CSV files are byte-identical to v1.5.1.
