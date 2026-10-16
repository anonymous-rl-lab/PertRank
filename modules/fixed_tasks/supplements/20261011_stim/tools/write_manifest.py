#!/usr/bin/env python3
"""Seal the exact module files after building and independent review."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
       for p in sorted(root.rglob('*')) if p.is_file() and
       p.name!='MANIFEST.json' and '__pycache__' not in p.parts and not p.name.endswith('.pyc')}
(root/'MANIFEST.json').write_text(json.dumps({
    'schema_version':1,'module':'D4_Stim8hr_exact_sufficient_replay',
    'file_sha256':files,'self_contained':True,
    'scope':'Frozen fixed-method result reproduction; not arbitrary refitting or complete raw-data reconstruction'},indent=2)+'\n')
print(len(files),'files sealed')
