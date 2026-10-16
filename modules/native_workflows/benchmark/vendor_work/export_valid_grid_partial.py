"""Export only completed existing fits; never scores or alters a grid status."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time
base=Path(__file__).resolve().parents[1]
grid=base/'fr_outputs/D3_Rest'
report=json.loads((grid/'GRID_EXECUTION.json').read_text())
output=base/'fr_compact/D3_Rest'; output.mkdir(parents=True,exist_ok=True)
ledger={'source_grid_status':report['status'],'score_read':False,'runs':[],'status':'RUNNING'}
for row in report['runs']:
    if row['status']!='COMPLETE': continue
    name=row['id']+'_rep0'
    with open(base/'heavy_memory.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        cmd=[sys.executable,str(Path(__file__).resolve().parent/'export_compact_fit.py'),'--fit-dir',str(grid/name),'--axes',str(base/'inputs/D3_Rest/rep0/axes.npz'),'--out',str(output/name)]
        proc=subprocess.run(cmd,capture_output=True,text=True)
    entry={'id':row['id'],'returncode':proc.returncode,'output':proc.stdout,'error':proc.stderr}
    ledger['runs'].append(entry)
    (output/'PARTIAL_COMPACT_EXECUTION.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(entry),flush=True)
    if proc.returncode: raise RuntimeError(name)
ledger['status']='VALID_SUBSET_EXPORTED'
(base/'fr_compact/D3_Rest/PARTIAL_COMPACT_EXECUTION.json').write_text(json.dumps(ledger,indent=2)+'\n')
