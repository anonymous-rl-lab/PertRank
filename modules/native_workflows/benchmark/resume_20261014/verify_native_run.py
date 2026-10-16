"""Separate-tool completion attestation for one rank/depth owner that exited zero."""
from pathlib import Path
import argparse,datetime,json,shutil,sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'vendor_work'));import atomic_npz
p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--owner-session-id',required=True);a=p.parse_args()
fits=[p for p in a.run_root.rglob('fit_report.json') if 'rank_boundary_outputs' in p.parts or 'native_fr_outputs' in p.parts];assert len(fits)==1
report_path=fits[0];fit=report_path.parent;report=json.loads(report_path.read_text());assert report['status']=='COMPLETE'
is_depth='native_fr_outputs' in fit.parts
compact=Path(str(fit).replace('/native_fr_outputs/','/native_fr_compact/').replace('/rank_boundary_outputs/','/rank_boundary_compact/'))
checks={}
for path,key in [(fit/'effects.npz','effects_sha256'),(fit/'native_unscaled.npz','native_unscaled_sha256')]:
 digest=atomic_npz.validate(path);assert digest==report[key];checks[path.name]={'sha256':digest,'npz_crc_pass':True}
manifest=json.loads((compact/'COMPACT_MANIFEST.json').read_text());digest=atomic_npz.validate(compact/'compact.npz');assert digest==manifest['compact_sha256'];checks['compact.npz']={'sha256':digest,'npz_crc_pass':True}
stability=json.loads((fit/'ATOMIC_SERIALIZATION_INPUT_STABILITY.json').read_text());assert stability['status']=='PASS' and stability['before']==stability['after']
for name,digest in stability['after'].items():assert atomic_npz.sha(Path(report['input_provenance']['input_directory'])/name)==digest
audit=json.loads((fit/'NATIVE_SOLVER_OBJECTIVE_AUDIT.json').read_text());assert audit['pass_'] and audit['original_fit_report_sha256']==atomic_npz.sha(report_path)
files={name:atomic_npz.sha(fit/name) for name in ['effects.npz','fit_report.json','NATIVE_SOLVER_OBJECTIVE_AUDIT.json']}
evidence={'status':'OWNER_EXECUTION_ENDED_INTEGRITY_VERIFIED','owner_session_id':a.owner_session_id,'owner_exit_code':0,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'fit_directory':str(fit.resolve()),'compact_directory':str(compact.resolve()),'npz_files':checks,'npz_crc':'PASS','files_sha256':files,
 'fit_report_sha256':files['fit_report.json'],'native_solver_objective_audit_sha256':files['NATIVE_SOLVER_OBJECTIVE_AUDIT.json'],
 'input_stability_sha256':atomic_npz.sha(fit/'ATOMIC_SERIALIZATION_INPUT_STABILITY.json'),'kind':'depth' if is_depth else 'rank','rank':report['parameters']['rank']}
if is_depth:evidence.update(curve=fit.parent.name,depth=int(fit.name[1:]))
for dest in [fit/'OWNER_COMPLETION.json',compact/'OWNER_COMPLETION.json',a.run_root/'OWNER_COMPLETION.json']:dest.write_text(json.dumps(evidence,indent=2)+'\n')
with (HERE/'NATIVE_OWNER_VERIFIED.jsonl').open('a') as f:f.write(json.dumps(evidence)+'\n')
print(json.dumps({'status':evidence['status'],'fit_directory':str(fit),'compact_directory':str(compact),'owner_completion':str(fit/'OWNER_COMPLETION.json'),'runtime_seconds':report['runtime_seconds'],'max_rss_kib':report['max_rss_kib']}))
