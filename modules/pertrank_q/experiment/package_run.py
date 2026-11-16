"""Compact self-contained package of the bounded B 2x2 update."""
from pathlib import Path
import hashlib, json, shutil, zipfile
ROOT = Path(__file__).resolve().parent
NAME = 'PertRank_B_Optimization_20261115'
BASE = ROOT.parent / 'pertrank_boundary_learning_20261115' / 'deliverable' / 'PertRank_Boundary_Learning_20261115'
DEST = ROOT / 'deliverable' / NAME
if DEST.exists():
    raise RuntimeError('Existing deliverable preserved')
assert (ROOT / 'results/SUMMARY.json').is_file()
assert (ROOT / 'replay_optimization.py').is_file()
DEST.mkdir(parents=True)
shutil.copytree(BASE, DEST / 'baseline_B')
files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py', '.md', '.json', '.txt', '.log') and p.name != 'DELIVERY.json']
for folder in ('fits', 'selections', 'audit', 'epoch_pair_update'):
    files += [p for p in (ROOT / folder).rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'
              and not (p.suffix == '.npz' and any(part in ('predictions', 'per_task', 'per_source') for part in p.relative_to(ROOT).parts))
              and not any(part.startswith('verification') or part.startswith('replay_') for part in p.relative_to(ROOT).parts[:-1])]
files += list((ROOT / 'predictions').glob('*LOCK.json'))
files += [p for p in (ROOT / 'results').iterdir() if p.is_file() and p.suffix in ('.json', '.md', '.txt', '.log')]
for p in files:
    q = DEST / p.relative_to(ROOT)
    q.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, q)
manifest = {str(p.relative_to(DEST)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.rglob('*')) if p.is_file()}
(DEST / 'MANIFEST_SHA256.json').write_text(json.dumps(manifest, indent=2) + '\n')
archive = ROOT / 'deliverable' / (NAME + '.zip')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in sorted(DEST.rglob('*')):
        if p.is_file(): z.write(p, p.relative_to(DEST.parent))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name, expected in manifest.items():
        assert hashlib.sha256(z.read(NAME + '/' + name)).hexdigest() == expected
size = archive.stat().st_size
report = dict(path=str(archive), bytes=size, MB_decimal=size/1e6, files=len(manifest)+1,
              sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
              zip_crc_checked=True, all_manifest_hashes_checked=True, under_10MB=size < 10_000_000)
(ROOT / 'DELIVERY.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
