# Distribution and release process

## Current status

This anonymous review source is version `0.2.0rc3` and accompanies the current
manuscript. It preserves the numerical API, fixed parameters and tutorial
arrays. No public upload or ecosystem acceptance is performed by this package.

The inherited license still states that the public license is pending author
selection. The rights holder must confirm the intended license before a public
release. This audit does not assign a license.

## Rebuild and verify

Copy the software source outside the checksum-protected reproduction bundle.
From that copy, build into a new directory:

```bash
python -m pip install '.[test]' build twine
python scripts/build_pypi_description.py
python -m pytest
python -m build --outdir ../pertrank-dist-rc3
python scripts/check_distribution.py ../pertrank-dist-rc3
python -m twine check ../pertrank-dist-rc3/*
```

Install the resulting wheel in a fresh environment, then run
`scripts/check_installation.py` and
`python scripts/check_tutorials.py --output-dir ../pertrank-tutorial-check`.
[Current validation](validation.md) identifies the checks actually completed.
Do not overwrite published
version files or mix old distributions into an upload directory.

## Documentation links

The source distribution includes `docs/scientific_design.md`, `docs/api.md`
and `tutorials/01_replogle.ipynb`. The wheel includes runtime code and tutorial
fixtures; installing it does not install the standalone documentation tree.
PyPI does not turn source-archive paths into separately hosted web pages.

`README.md` uses HTTPS links to the scientific design, API reference and
Replogle tutorial in the anonymous GitHub repository. The API scoring-rule
link retains its `#evaluate_reference` anchor. Other documentation and
tutorial links remain local for browsing the extracted source tree.

`PYPI_DESCRIPTION.md`, generated from the README, preserves these HTTPS links
and converts the remaining local links into explicit source-archive paths.
It is the configured long description in `pyproject.toml`. This keeps the
three online documentation targets clickable without creating false PyPI
URLs for files that remain available through the source archive.

## Scientific and execution scope

Both authentic tutorial fixtures contain retained effects and sufficient
statistics from the named screens. They recompute reliability and rankings;
they do not repeat raw-cell preprocessing. The AnnData quickstart is synthetic
and checks software use. Scientific uncertainty, source aggregation and
PertRank-Q are reproduced separately in the complete code bundle.

No new biological comparison, parameter search or reference analysis is
performed by this software update. Package publication, remote CI execution,
community registration and original-data reanalysis are not implied by a
successful local build. Keep credentials outside source files. A subsequent
registry upload remains an explicit maintainer action.
