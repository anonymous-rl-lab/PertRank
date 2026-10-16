# Fixture curation provenance

The two curation scripts record how the packaged numerical fixtures were
selected from the existing manuscript archives. They are archival provenance
scripts: their relative input paths refer to the original preparation
workspace, not to files shipped inside the lightweight review release.

All data needed to run the tutorials is already installed in
`pertrank/data/`. No curation script, original atlas download or cell-level
model fit is required to run the package, tests or notebooks. To audit curation
against the original archives, obtain the exact source files named and hashed
in each `*.provenance.json`, then map the source paths before running these
scripts. Inputs are summarized cell statistics; evaluation references are not
used by these scripts.

`check_tutorials.py` is the portable offline execution check. Unlike the
archival curation scripts, it runs directly in this release using only the
installed package and bundled notebooks.
