#!/usr/bin/env python3
"""Summarize completed existing-input recovery, without reading source data."""
from pathlib import Path
import datetime,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    tasks={};audits=[]
    for task,nrep in [('D3_Rest',1),('D4_Rest',5),('D4_Stim8hr',5)]:
        task_audits=[]
        for rep in range(nrep):
            path=ROOT/'inputs'/task/f'rep{rep}'/'INPUT_AUDIT.json'
            record=json.loads(path.read_text())
            assert record['status'] in ('EXACT_RAW_R20_REPLAY_PASS','EXACT_RAW_REPLAY_PASS')
            task_audits.append(record);audits.append(record)
        with np.load(ROOT/'pooled_data'/task/'selection.npz',allow_pickle=False) as selection:
            unique_cells=len(selection['all_selected_rows'])
        tasks[task]={'replicates':nrep,'unique_raw_input_cells':unique_cells,'audits':task_audits}
    result={'status':'ALL_POOLED_INPUTS_RECOVERED_AND_AUDITED','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tasks':tasks,'replicate_inputs':len(audits),'unique_raw_input_cells_across_sources':sum(x['unique_raw_input_cells'] for x in tasks.values()),'access_scope':['D3_Rest','D4_Rest','D4_Stim8hr'],'source_reads_excluded_by_recovery_code':['D1_Stim8hr','D2_Stim8hr'],'interpretation':'Exact replay of previously opened pooled-guide tasks. These are not new held-out donor confirmations. D3 preserves the archived control center; its raw re-estimation difference is documented in its audit.','io_adjustment_record':'pooled_data/IO_SCHEDULING_ADJUSTMENT.json'}
    out=ROOT/'pooled_data/RECOVERY_COMPLETE.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'replicate_inputs':len(audits),'unique_raw_input_cells':result['unique_raw_input_cells_across_sources'],'output':str(out)}))

if __name__=='__main__':main()
