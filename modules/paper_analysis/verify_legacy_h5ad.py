#!/usr/bin/env python3
"""Run the unchanged v20 fixture with its original object-string convention.

Pandas 3 infers nullable StringArray columns by default; AnnData 0.12 does not
write them without an experimental opt-in. Disabling that inference only
while constructing this synthetic legacy fixture preserves its original
metadata dtype. Counts, numerical functions and assertions are unchanged.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--application-dir', type=Path, required=True)
args = p.parse_args()
root = args.application_dir.resolve()
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location('legacy_h5ad_verification', root / 'verify_h5ad.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    pd.get_option('future.infer_string')
except pd.errors.OptionError:
    module.main()
else:
    with pd.option_context('future.infer_string', False):
        module.main()
