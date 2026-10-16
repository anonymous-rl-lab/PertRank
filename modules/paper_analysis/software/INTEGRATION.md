# Software integration

This tree supplies PertRank `0.2.0rc2`, aligned with manuscript v25.2. It replaces the previous `0.2.0rc1` software source in the scientific reproduction bundle. The software version and the reproduction-bundle version identify different deliverables.

The scientific equations, fixed five-direction geometry, attenuation 0.9 and archived tutorial data are retained. Interface validation, external-effect provenance and documentation have been updated. The result's source reliability statistics stay distinct from the effects currently used to rank candidates. See [CHANGELOG](CHANGELOG.md) and [current validation](docs/validation.md).

`tests/reference_core_v21_1.py` remains a frozen regression oracle: its historical name records where the unchanged numerical engine originated, rather than the version of the current manuscript. Similarly, `provenance/ORIGINAL_RELEASE.json` and `ORIGINAL_SHA256SUMS.txt` identify the initial separate software release; they are historical provenance, not manifests for the current tree. Current release checksums identify the delivered version.

The integrated scientific archive carries this source once. Standalone wheel and source distributions may be supplied separately for installation. Tutorial fixtures and numerical scientific results are shared with the existing reproduction chain; no scientific model is fitted as part of this software revision.
