"""Check built source/wheel archives and the PyPI documentation-link contract.

Run: python scripts/check_distribution.py /path/to/dist
No archive member is extracted or executed.
"""
import argparse
import json
from pathlib import Path
import re
import tarfile
import zipfile

from build_pypi_description import render

REQUIRED_DOCS = ('docs/scientific_design.md', 'docs/api.md',
                 'tutorials/01_replogle.ipynb')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dist', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    description = (root / 'PYPI_DESCRIPTION.md').read_text(encoding='utf-8')
    if description != render():
        raise RuntimeError('Regenerate PYPI_DESCRIPTION.md from README.md')
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', description):
        if not target.startswith(('https://', 'http://', 'mailto:', '#')):
            raise RuntimeError(f'PyPI description has a local URL: {target}')
    version = re.search(r'^version = "([^"]+)"', (root / 'pyproject.toml').read_text(), re.M).group(1)
    source = args.dist / f'pertrank-{version}.tar.gz'
    wheel = args.dist / f'pertrank-{version}-py3-none-any.whl'
    with tarfile.open(source, 'r:gz') as archive:
        for relative in (*REQUIRED_DOCS, 'PYPI_DESCRIPTION.md'):
            member = f'pertrank-{version}/{relative}'
            data = archive.extractfile(member).read()
            if data != (root / relative).read_bytes():
                raise RuntimeError(f'Source archive mismatch: {relative}')
        if json.loads(archive.extractfile(f'pertrank-{version}/tutorials/01_replogle.ipynb').read())['nbformat'] != 4:
            raise RuntimeError('Invalid Replogle notebook format')
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(f'pertrank-{version}.dist-info/METADATA').decode()
        if description.strip() not in metadata:
            raise RuntimeError('Wheel long description differs from the verified PyPI description')
        for name in ('replogle_k562.npz', 'tcell_d3_rest.npz'):
            if archive.read(f'pertrank/data/{name}') != (root / 'src/pertrank/data' / name).read_bytes():
                raise RuntimeError(f'Wheel fixture mismatch: {name}')
    print(json.dumps({'status':'PASS', 'version':version,
                      'source_document_paths':list(REQUIRED_DOCS),
                      'pypi_relative_links':0,
                      'wheel_fixture_identity':'PASS'}, indent=2))


if __name__ == '__main__':
    main()
