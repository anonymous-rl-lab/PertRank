"""Serialization-only NPZ publication guard; array values and compression unchanged."""
from pathlib import Path
import hashlib, os, tempfile, zipfile
import numpy as np

_SAVEZ = np.savez
_SAVEZ_COMPRESSED = np.savez_compressed
EVENTS = []

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def validate(path):
    with zipfile.ZipFile(path) as archive:
        bad=archive.testzip()
        if bad is not None:raise ValueError('NPZ CRC failure: '+str(path)+' member '+bad)
    return sha(path)

def publish(original, file, *args, **kwargs):
    if not isinstance(file,(str,bytes,os.PathLike)):
        raise TypeError('Atomic NPZ wrapper requires an explicit path')
    path=Path(os.fsdecode(file))
    if path.suffix!='.npz':path=Path(str(path)+'.npz')
    if path.exists():raise FileExistsError('Refusing to overwrite published NPZ: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.writing.npz',dir=path.parent)
    os.close(fd); temp=Path(name)
    original(temp,*args,**kwargs)
    with temp.open('rb') as f:os.fsync(f.fileno())
    digest=validate(temp)
    os.replace(temp,path)
    directory=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
    if validate(path)!=digest:raise AssertionError('NPZ bytes changed during atomic publication: '+str(path))
    EVENTS.append(dict(path=str(path.resolve()),sha256=digest,size_bytes=path.stat().st_size,crc_pass=True,atomic_rename=True))

def install():
    np.savez=lambda file,*a,**k:publish(_SAVEZ,file,*a,**k)
    np.savez_compressed=lambda file,*a,**k:publish(_SAVEZ_COMPRESSED,file,*a,**k)
