# Software and fixture checks

`check_tutorials.py --output-dir PATH` runs every code cell in both notebooks with the installed package and keeps generated tables outside the protected source tree. `check_installation.py` checks installed runtime and resource bytes against this source.

`build_pypi_description.py` generates a long description without repository-relative web links. `check_distribution.py DIST` verifies the three requested documentation files in the source archive, the description in the wheel and both fixture arrays.

## Fixture curation

`curate_replogle.py` and `curate_tcell.py` preserve the numerical conversion of upstream statistics into tutorial fixtures. They take explicit input paths; use `--help` for all arguments. Exact upstream input SHA256 values and logical filenames are recorded in the packaged `*.provenance.json`. These upstream files are not shipped in the lightweight software source; no curation step is needed to run the installed examples. The scripts write to a new output directory and do not overwrite the packaged fixture arrays.

Curation CLI and syntax are checked here; upstream curation is not rerun by the current validation. The precomputed fixtures retain their verified content hashes. No reference fitting or biological performance evaluation is part of fixture construction.
