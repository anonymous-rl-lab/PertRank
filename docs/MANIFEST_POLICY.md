# Current payload and historical source hashes

The root `MANIFEST_SHA256.json` defines every file in the current v1.5.4 review payload. Executed component manifests describe their maintained payloads and dependencies. Registration aliases alter the bytes of a small number of documentation and provenance fields; the affected current dependency hashes are updated accordingly. Numeric arrays and scientific definitions are preserved.

Historical source manifests, prediction locks, original output provenance and `ORIGINAL_V150_*` manifests retain the hashes of their original artifacts. They are evidence records, not substitutes for the current release manifest. In particular, a saved original D4 result hash identifies the unmodified record retained in the private provenance archive, whereas current replay checks use the review derivative's payload hash.

Some analysis-local manifests describe the original expanded input layout. Compact reconstruction restores those inputs outside the release tree and checks numeric array hashes; the current root manifest checks the stored compressed representation. A historical expanded-input filename does not mean a physical duplicate is required in the compact source directory.

`RELEASE_MIGRATION.csv` records changed, removed and added paths relative to v1.5.1. Scientific and numerical validation receipts explain what was actually recalculated. Archived provenance is not relabelled as a new successful run.
