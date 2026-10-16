"""Add a uniform fit-bound numerical audit certificate without changing a fit."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from native_recovery_audit import audit_recovery

def main():
    p=argparse.ArgumentParser();p.add_argument('--fit-dir',required=True);p.add_argument('--axes',required=True);a=p.parse_args()
    fit=Path(a.fit_dir);here=Path(__file__).resolve().parent
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report=json.loads((fit/'fit_report.json').read_text())
    labels=np.load(a.axes,allow_pickle=False)['labels']
    if sha(a.axes)!=report['input_provenance']['axes_sha256']: raise AssertionError('Input axes changed')
    z=np.load(fit/'native_unscaled.npz',allow_pickle=False)
    audit=audit_recovery(z['U'],z['U_tilde'],labels,report['parameters']['lasso_alpha'],
        report['recover_audit']['dual_gap'],report['recover_audit']['tolerance'],report['recover_audit']['convergence_warning'])
    audit.update(original_fit_status=report['status'],original_fit_report_sha256=sha(fit/'fit_report.json'),audit_source_sha256=sha(here/'native_recovery_audit.py'))
    path=fit/'NATIVE_SOLVER_OBJECTIVE_AUDIT.json';payload=json.dumps(audit,indent=2)+'\n'
    if path.exists():assert path.read_text()==payload
    else:path.write_text(payload)
    if not audit['pass_']:raise AssertionError('Native solver/objective audit rejected fit')
    print(json.dumps({'fit':str(fit),'audit_pass':True,'certificate_sha256':sha(path)}))
if __name__=='__main__':main()
