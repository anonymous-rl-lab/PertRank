"""Build a compact, versioned reproduction archive; retain all failed results."""
from pathlib import Path
import json,shutil,hashlib,zipfile
ROOT=Path(__file__).resolve().parent
NAME='PertRank_Boundary_Learning_20261115'
DEST=ROOT/'deliverable'/NAME
if DEST.exists(): raise RuntimeError('Existing deliverable preserved')
DEST.mkdir(parents=True)
files=[]
for p in ROOT.iterdir():
 if p.is_file() and p.suffix in ('.py','.md','.json','.txt','.log'): files.append(p)
for folder in ('inputs','fits','historical','selections','audit'):
 for p in (ROOT/folder).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':files.append(p)
files+=list((ROOT/'predictions').glob('*LOCK.json'))
files+=[p for p in (ROOT/'results').iterdir() if p.is_file()]
for p in files:
 q=DEST/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
manifest={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.rglob('*')) if p.is_file()}
(DEST/'MANIFEST_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=ROOT/'deliverable'/f'{NAME}.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in sorted(DEST.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(DEST.parent))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for name,expected in manifest.items(): assert hashlib.sha256(z.read(NAME+'/'+name)).hexdigest()==expected
size=archive.stat().st_size
assert size<10_000_000,size
result={'path':str(archive),'bytes':size,'MB_decimal':size/1e6,'files':len(manifest)+1,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'all_manifest_hashes_checked':True,'zip_crc_checked':True}
(ROOT/'DELIVERY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
