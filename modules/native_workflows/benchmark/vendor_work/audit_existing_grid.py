"""Read-only native fit audit; adds diagnostic files, never references or scores."""
from pathlib import Path
import hashlib,json
import numpy as np
from native_recovery_audit import audit_recovery
base=Path(__file__).resolve().parents[1]
labels=np.load(base/'inputs/D3_Rest/rep0/axes.npz',allow_pickle=False)['labels']
grid=base/'fr_outputs/D3_Rest'
rows=json.loads((grid/'GRID_EXECUTION.json').read_text())['runs']
out=[]
for row in rows:
    fit=grid/(row['id']+'_rep0')
    report=json.loads((fit/'fit_report.json').read_text())
    with np.load(fit/'native_unscaled.npz',allow_pickle=False) as z:
        audit=audit_recovery(z['U'],z['U_tilde'],labels,
             report['parameters']['lasso_alpha'],report['recover_audit']['dual_gap'],
             report['recover_audit']['tolerance'],report['recover_audit']['convergence_warning'])
    audit['original_fit_status']=report['status']
    audit['original_fit_report_sha256']=hashlib.sha256((fit/'fit_report.json').read_bytes()).hexdigest()
    audit['audit_source_sha256']=hashlib.sha256((Path(__file__).resolve().parent/'native_recovery_audit.py').read_bytes()).hexdigest()
    (fit/'NATIVE_SOLVER_OBJECTIVE_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    summary={k:audit[k] for k in ['pass_','coefficient_max_absolute_error','ntc_coefficient_max_absolute_error','coefficient_count_above_legacy_tolerance','max_objective_gap_to_native_tolerance']}
    summary['id']=row['id'];out.append(summary);print(json.dumps(summary),flush=True)
(base/'vendor_work/UNSCORED_GRID_SOLVER_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
