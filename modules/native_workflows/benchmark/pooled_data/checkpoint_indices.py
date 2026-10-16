#!/usr/bin/env python3
"""Package costly structural source recovery without temporary HTTP caches."""
from pathlib import Path
import zipfile,json,hashlib,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
FILES=['selection.npz','selection.lock.json','control_model.npz','query_context.npz','r20_context.npz']
RAW=['hdf_objects.json','gene_ids_full.npy','indptr.npy','indptr_report.json','selected_chunk_manifest.json','PREPARE_COMPLETE.json','COST_PLAN.json','INPUT_IDENTITY.json']
paths=list(HERE.glob('*.py'))
for task in ['D3_Rest','D4_Rest','D4_Stim8hr']:
    d=HERE/task
    assert (d/'raw/COST_PLAN.json').exists(),task+' structural plan incomplete'
    paths.extend(d/name for name in FILES if (d/name).exists())
    paths.extend(d/'raw'/name for name in RAW)
for name in ['extract_selected_rows.py','extract_selected_rows_fast.py','vendor/hdf5_chunk_index.py','vendor/row_transport.py']:
    paths.append(ROOT/'independent/execution'/name)
manifest={'created_utc_epoch':time.time(),'purpose':'costly source-index checkpoint and exact pooled input-role identities; no original raw source shards','files':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in paths}}
out=ROOT/'Pooled_Input_Index_Checkpoint_20261012.zip'
with zipfile.ZipFile(out.with_suffix('.tmp.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
    for p in paths:z.write(p,str(p.relative_to(ROOT)))
    z.writestr('INDEX_CHECKPOINT_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
out.with_suffix('.tmp.zip').replace(out)
print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(paths)},indent=2))
