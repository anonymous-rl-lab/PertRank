# Lossless input storage

Run `python reproduce.py --output-dir ../pertrank-reproduced` from the package
root. The dispatcher copies the executable modules to the output directory and
restores the shared inputs before running the selected scientific workflows.
No download is needed after installing dependencies. Every data block is included
in this repository; numbered blocks are parts of one internal store.

The two indexes cover 113 NPZ files. The store keeps original ZIP metadata and
NPY byte strings, deduplicates identical byte strings, and applies reversible
byte shuffling followed by XZ compression. Restoration verifies the stored
blocks, every original byte string, every compressed ZIP stream and the complete
original NPZ file. It preserves dtype, array order, values, file size and SHA256.
The native rvalues module additionally reuses identical effect values from the
restored fixed-task and paper-analysis inputs, then verifies all 50 native files.

The tested environment is Python 3.12.14, NumPy 2.3.5 and zlib 1.3.2; scientific
dependencies are pinned in the root requirements file. Exact ZIP-stream
reconstruction also depends on the compression runtime. If that runtime produces
a different stream, restoration fails explicitly before scientific computation.
It never rounds data or relaxes a checksum to continue. The runtime version can
be inspected with:

```bash
python -c "import sys, zlib; print(sys.version); print(zlib.ZLIB_RUNTIME_VERSION)"
```

Restoration produces a separate `STORAGE_RESTORATION.json` receipt. Scientific
module receipts then record the actual ranking, resampling, checkpoint inference
or plotting checks. A storage-integrity pass alone is not a model-refitting pass.

The original uncompressed module paths remain available under `OUTPUT/_source/`.
All root commands use that execution tree automatically. Lower-level commands
in component READMEs should be run against the restored copy.
