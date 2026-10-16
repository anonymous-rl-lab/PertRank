# Packaging provenance

`build_ranking_data.py` constructs the exact sparse ranking support from the complete frozen source archive. It is a post-result storage conversion, not a new prediction procedure. The normal user command is `python reproduce.py`; it does not run this builder or need the large source files.

To rebuild the ranking compaction when the complete original workspace is available:

```bash
python tools/build_ranking_data.py --source-root /path/to/complete_archive --destination /path/to/new_package
```

The builder requires the source paths and official score files documented in its source, and refuses numerical discrepancies. The weight and depth source/key mappings are separately documented in `provenance/development_sources.json`, `provenance/d4_weight_sources.json` and `provenance/depth_sources.json`. The original acquisition/fitting code retained under `provenance/code/` records provenance and may depend on the complete archive; it is not the self-contained public replay entry point.
