"""Restore losslessly stored NPZ containers, including their original file hashes.

Only storage representation changes. Each reconstructed compressed ZIP stream,
NPY blob and complete NPZ container is SHA256 checked before use.
"""
from pathlib import Path
import hashlib
import io
import json
import lzma
import tarfile
import zipfile
import zlib
import numpy as np


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _deflate(data, level):
    obj = zlib.compressobj(level, zlib.DEFLATED, -15)
    return obj.compress(data) + obj.flush()


def _compress(data, record):
    method = record['method']
    if method == 'stored':
        output = data
    elif method == 'deflate':
        output = _deflate(data, record['level'])
    elif method == 'zip_lzma':
        obj = zipfile.LZMACompressor()
        output = obj.compress(data) + obj.flush()
    else:
        raise ValueError('Unknown lossless stream method: ' + str(method))
    if _sha(output) != record['stream_sha256']:
        raise ValueError('Compressed stream differs from retained original; use the documented Python/zlib environment.')
    return output


def restore(source_root, destination_root, index_name='fixed_tasks_index.json'):
    """Restore packed NPZ files into an existing external execution tree."""
    source_root, destination_root = Path(source_root).resolve(), Path(destination_root).resolve()
    if destination_root == source_root or destination_root.is_relative_to(source_root):
        raise ValueError('Restoration destination must be outside the immutable package.')
    storage = source_root / 'storage'
    index = json.loads((storage / index_name).read_text())
    chunks = index.get('payload_chunks')
    if chunks:
        parts = []
        for record in chunks:
            data = (storage / record['name']).read_bytes()
            if len(data) != record['bytes'] or _sha(data) != record['sha256']:
                raise ValueError('Lossless payload chunk mismatch: ' + record['name'])
            parts.append(data)
        payload_bytes = b''.join(parts)
    else:
        payload_bytes = (storage / index['payload']).read_bytes()
    if _sha(payload_bytes) != index['payload_sha256']:
        raise ValueError('Lossless payload SHA256 mismatch.')
    blobs = {}
    with tarfile.open(fileobj=io.BytesIO(payload_bytes), mode='r:xz') as archive:
        for member in archive:
            if not member.isfile() or member.name not in index['blobs']:
                raise ValueError('Unexpected payload member: ' + member.name)
            record = index['blobs'][member.name]
            data = archive.extractfile(member).read()
            if len(data) != record['bytes']:
                raise ValueError('Payload member size mismatch: ' + member.name)
            width = record['shuffle_width']
            if width > 1:
                count = (len(data) // width) * width
                data = np.frombuffer(data[:count], dtype=np.uint8).reshape(width, -1).T.tobytes() + data[count:]
            if _sha(data) != member.name:
                raise ValueError('Original blob SHA256 mismatch: ' + member.name)
            blobs[member.name] = data
    if set(blobs) != set(index['blobs']):
        raise ValueError('Incomplete lossless payload.')
    restored = []
    for relative, record in index['files'].items():
        target = (destination_root / relative).resolve()
        if not target.is_relative_to(destination_root) or Path(relative).is_absolute():
            raise ValueError('Invalid restore path: ' + relative)
        parts = []
        for segment in record['segments']:
            data = blobs[segment['blob']]
            parts.append(_compress(data, segment) if 'method' in segment else data)
        result = b''.join(parts)
        if len(result) != record['bytes'] or _sha(result) != record['sha256']:
            raise ValueError('Restored container mismatch: ' + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result)
        restored.append(relative)
    return {'status':'PASS', 'restored_files':len(restored),
            'restored_bytes':sum(r['bytes'] for r in index['files'].values()),
            'all_original_file_hashes_match':True, 'all_original_blob_hashes_match':True,
            'precision_changed':False, 'source_files_modified':False}
