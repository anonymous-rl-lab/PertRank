# Reproducing the Mixscale comparison

From the package root, use the portable entry:

```bash
python reproduce.py --only sensitivity_replay --output-dir ../sensitivity-reproduced
```

This reconstructs the retained coefficient and scoring arrays, reranks all
1,200 candidates for the original 300 D3 queries in both directions, checks
2,400 query-direction scores across four methods, and repeats the original
2,000-draw query-block summaries. It also checks the four saved 200-draw
cell-sensitivity distributions. The adverse weight-refit findings and the
opened-D3 development identity remain part of the reported evidence.

Native fitting and fixed-coefficient scoring are distinct operations. The
bundled native scripts, `MIXSCALE_BOUNDED_PROTOCOL.md`,
`MIXSCALE_EVAL_FREEZE.json` and `MIXSCALE_NATIVE_CONFIGURATION.md` specify the
original input roles, method configuration and completion rules. The source
scripts retain their frozen upstream paths for provenance; they are not a
second self-contained raw-cell runner. Repeating native fitting requires the
original raw count/cell metadata inputs, upstream Mixscale commit and compatible
R/Seurat dependencies. These upstream stages are outside the compact replay.

The original native `beta_weight` coefficients remain the ranking readout.
Specified expression-filter omissions receive zero; unsuccessful fits and
nonfinite outputs are rejected. Native completion requires all 1,200 targets.
The source map `EXECUTION_SOURCE_HASH_ALIASES.json` lists only files actually
available in this directory; scientific freeze records retain their original
source hashes separately.
