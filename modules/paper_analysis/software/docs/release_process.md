# Private release and public handoff

## Current delivery

Version `0.2.0rc2` is a private review artifact aligned with manuscript v25.2. Install from this source tree or from the separately supplied wheel. The archive includes the implementation, API documentation, real-data tutorial fixtures, provenance and software checks. The scientific implementation retains the frozen manuscript equations and fixed parameters; interface and provenance corrections are listed in [CHANGELOG](../CHANGELOG.md).

The distribution and import name `pertrank` are the intended names for this artifact. The existence of a local wheel does not reserve a public registry name. No author identity, public repository, package-registry upload or community endorsement is implied by the private metadata.

## Rebuild and verify

From the package root:

```bash
python -m pip install '.[test]' build
python -m pytest
python -m build
```

Use a fresh environment to install the resulting wheel and execute the tutorials. [Current validation](validation.md) records the performed source and distribution checks. Verify the current delivery checksums before installation. The original separate-release metadata and manifest in `provenance/` describe the initial release, not this revised tree. Keep distribution artifacts and their current release checksums together so the installed version can be identified.

Review artifacts should preserve method and source-data attribution while keeping manuscript author identities, workstation paths, access tokens and private links out of public-facing metadata. Source dataset citation alone is not a claim that the tutorial redistributes the complete atlas.

## What the tutorials reproduce

The compact real-data fixtures store previously computed effects and sufficient statistics from the named public screens. Their tutorial outputs reproduce the corresponding ranking operations from those statistics. They are not substitutes for the full raw-cell reproduction archive. AnnData cell input is supported by the package and checked separately against the frozen cell engine.

No new biological comparison, hyperparameter search or reference-data analysis is required for this software release. Regression checks address implementation identity and interface behavior.

## Public release fields to complete

Before public distribution, the rights holder supplies the software license, public attribution and maintaining person or team. The maintainers verify the registry name, repository URL, documentation URL, data redistribution terms and citation record. They then enable the supplied CI workflow in the public repository and link its successful run to the versioned release.

For double-blind review, retain the private artifact and defer author-linked registry uploads or community contributions until the authors choose the public-release date. Public listing and upstream acceptance remain separate from the private artifact's technical completion.

After the public fields and rights are resolved, the maintainer can run the standard package-index checks on the built distributions, publish through the chosen registry's authenticated release process and verify installation in a clean environment. Credentials belong in the registry or CI secret mechanism, never in the source archive.
