#!/usr/bin/env python3
"""Freeze the validated new diagnostic before full source acquisition."""
import hashlib,json,time
from pathlib import Path
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=P/'RUN_FREEZE.json'
    if out.exists():
        r=json.loads(out.read_text())
        for name,want in r['file_sha256'].items():assert sha(P/name)==want,name
        print(json.dumps({'status':'EXISTING_FREEZE_VERIFIED','sha256':sha(out)}));return
    smoke=json.loads((P/'SMOKE_PASS.json').read_text())
    assert smoke['status']=='SMOKE_PASS' and smoke['reader_sha256']==sha(P/'read_selected.py')
    assert smoke['selection_sha256']==sha(P/'selection.npz')
    files=['PROTOCOL.json','selection.npz','selection_report.json','ntc_selected_cells.json','target_selected_cells.json',
      'prepare_metadata.py','select_guides.py','select_guides_portable.py','read_selected.py','stats_impl.py','evaluate.py','freeze_run.py',
      'RUNTIME.json','frozen_assets/SOURCE_PROVENANCE.json','frozen_assets/A15_D3_cohort.npz',
      'frozen_assets/ntc_original/original_input_ntc_metadata.npz','frozen_assets/ntc_original/original_input_ntc_raw.npz',
      'frozen_assets/ntc_original/selection_check.json','vendor/metadata_source.py','vendor/hdf5_chunk_index.py','vendor/row_transport.py',
      'metadata/metadata_report.json','metadata/hdf_objects.json','metadata/obs_metadata.npz']
    obj={'status':'FROZEN_BEFORE_FULL_ACQUISITION','new_replanning_not_interrupted_script_replay':True,
      'utc_epoch':time.time(),'file_sha256':{name:sha(P/name) for name in files},'smoke_sha256':sha(P/'SMOKE_PASS.json'),
      'source_identity':smoke['source_identity'],'full_rows':23840,'HTTP_workers':8,'BLAS_threads':1}
    out.write_text(json.dumps(obj,indent=2)+'\n');print(json.dumps({'status':obj['status'],'sha256':sha(out)}))
if __name__=='__main__':main()
