# Delivered file integrity

The root `MANIFEST_SHA256.json` contains every delivered file except itself and optional local `.git` metadata. Unknown extra files, missing files and changed bytes are rejected. Run all builds, tests and generated outputs outside the protected package.

Active component manifests describe the maintained inputs. Some scientific locks and source hashes describe the original logical computation; they are distinct from a compressed container's physical hash. Lossless NPZ recompression changes container bytes while preserving every internal NPY member, dtype, shape and data byte. `validation/LOSSLESS_ARRAY_COMPRESSION.json` records that verification.

Compact payloads are restored in the output directory and checked against their declared logical arrays/files before computation. Anonymous substitutions affect source-path metadata only. The current replay checks all declared numerical outputs after those substitutions.

Repeated archival fit/runtime reports are stored once; `SHARED_PROVENANCE.csv` identifies the canonical records. Numerical inputs, negative findings and substantive comparison protocols are retained.
