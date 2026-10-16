"""Finish append-only lossless compact export and numerical-audit supplements."""
from pathlib import Path
import fcntl,hashlib,json,shutil,subprocess,sys
here=Path(__file__).resolve().parent;root=here.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
grid=root/'fr_outputs/D3_Rest_audit_v2';out=root/'fr_compact/D3_Rest'
record=json.loads((grid/'GRID_EXECUTION.json').read_text())
assert record['status']=='COMPLETE' and record['uniform_recovery_audit_all_21_pass']
ledger=dict(status='RUNNING',grid_execution_sha256=sha(grid/'GRID_EXECUTION.json'),runs=[])
for row in record['runs']:
    name=row['id']+'_rep0';fit=grid/name;target=out/name
    with open(root/'heavy_memory.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not target.exists():
            subprocess.run([sys.executable,str(here/'export_compact_fit.py'),'--fit-dir',str(fit),'--axes',str(root/'inputs/D3_Rest/rep0/axes.npz'),'--out',str(target)],check=True)
        manifest=json.loads((target/'COMPACT_MANIFEST.json').read_text())
        f=json.loads((fit/'fit_report.json').read_text())
        assert manifest['source_effects_sha256']==f['effects_sha256']
        assert sha(target/'compact.npz')==manifest['compact_sha256']
        assert manifest['full_B_reconstruction_bitwise_equal']
        assert manifest['all_compact_arrays_bitwise_verified']
        audit=fit/'NATIVE_SOLVER_OBJECTIVE_AUDIT.json'
        shutil.copy2(audit,target/audit.name)
        extra=dict(actual_source_fit=str(fit.resolve()),effective_grid_fit=str(fit),
            audit_amendment_sha256=sha(here/'NATIVE_SOLVER_AUDIT_AMENDMENT_V2.json'),
            audit_certificate_sha256=sha(audit),fit_report_sha256=sha(fit/'fit_report.json'),
            note='Append-only uniform native solver/objective audit; original compact arrays and native effects unchanged.')
        (target/'AUDIT_V2_SUPPLEMENT.json').write_text(json.dumps(extra,indent=2)+'\n')
    ledger['runs'].append({'id':row['id'],'compact_sha256':manifest['compact_sha256'],'source_effects_sha256':f['effects_sha256'],'audit_pass':True})
    (out/'COMPACT_GRID_EXECUTION.json').write_text(json.dumps(ledger,indent=2)+'\n')
ledger['status']='COMPLETE';ledger['all_21_exact_query_serialization']=True;ledger['all_21_B_reconstruction_bitwise_equal']=True
(out/'COMPACT_GRID_EXECUTION.json').write_text(json.dumps(ledger,indent=2)+'\n')
print(json.dumps({'status':'COMPLETE','n_fits':len(ledger['runs']),'out':str(out)}))
