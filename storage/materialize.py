#!/usr/bin/env python3
"""Restore exact scientific input files into a separate execution tree."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time

sys.dont_write_bytecode = True
from exact_npz import restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    source, target = args.source_root.resolve(), args.output_root.resolve()
    if target.exists() or target == source or target.is_relative_to(source):
        parser.error('Choose a new execution directory outside the package.')
    start = time.monotonic()
    target.mkdir(parents=True)
    shutil.copytree(source/'modules', target/'modules',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    reports = {}
    for index in sorted((source/'storage').glob('*_index.json')):
        reports[index.name] = restore(source, target, index.name)
        print(index.name + ': exact file restoration PASS', flush=True)
    report = {'status':'PASS', 'storage_indexes':reports,
              'scope':'Lossless file restoration; scientific computation runs separately.',
              'precision_changed':False, 'seconds':time.monotonic()-start}
    (target/'STORAGE_RESTORATION.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
