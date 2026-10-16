#!/usr/bin/env python3
"""Restore only the two exact members from the identified, already saved archive."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True,type=Path)
    ap.add_argument('--dependency',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
    dep=json.loads(a.dependency.read_text());assert sha(a.archive)==dep['archive_sha256']
    a.out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.archive) as z:
        for name,rec in dep['members'].items():
            dest=a.out/name
            if dest.exists():assert sha(dest)==rec['sha256'];continue
            stage=dest.with_suffix(dest.suffix+'.tmp')
            with z.open(rec['archive_member']) as src,stage.open('xb') as out:shutil.copyfileobj(src,out)
            assert sha(stage)==rec['sha256'];stage.replace(dest)
    print(json.dumps({'status':'EXACT_MEMBERS_RESTORED','library_file_id':dep['library_file_id'],'native_refits':0}))

if __name__=='__main__':main()
