# Optional small update ZIP

The complete `PertRank_D1_D4_GitHub_Core.zip` already contains this supplement.
If you download the complete updated archive, no separate update is needed.

`PertRank_Supplement_Update_under10MB.zip` is an optional overlay for the
original version 1.0.0 archive with SHA-256:

```text
0998d363a60a020213ceeb4908722ad578c9711f05ac09e6f2629aec892d45ed
```

The overlay contains the new modules and updated root README, version,
reproduction entry point and complete manifest. It is not self-contained:
it reuses the original D1–D4 numerical data and reference records.

1. Extract a fresh copy of the original version 1.0.0 archive.
2. Extract the update ZIP into the same parent directory, allowing replacement
   of the included root files. Both archives use the same top-level folder.
3. From `PertRank_D1_D4_GitHub_Core`, run:

```bash
python -m pip install -r requirements.txt
python reproduce.py --output-dir ../PertRank_reproduced
```

Use a new output directory. The verifier rejects any missing, changed or extra
package file. Keep any personal notes outside the immutable reproduction
folder. A failed manifest check must not be bypassed.
