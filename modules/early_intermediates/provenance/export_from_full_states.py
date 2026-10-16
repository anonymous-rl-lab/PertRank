from pathlib import Path
import os,sys,json,hashlib,importlib,runpy,numpy as np,subprocess,time,shutil
BASE=Path(os.environ['PERTRANK_FULL_SOURCE']).resolve();REPRO=BASE/'repro_v10';B=REPRO/'same_budget_weight';S3=B/'stage3';WORK=Path(os.environ.get('PERTRANK_EXPORT_WORK','early_export_work')).resolve()

def export_one(script,dest,extra):
 os.environ.update(PR10_ROOT=str(REPRO),PR10_DATA=str(REPRO/'data'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 for p in [REPRO,B,B/'scripts',B/'stage2',S3,script.parent]:sys.path.insert(0,str(p))
 import pr10,pr10.bootstrap,pr10.scoring
 arrays={};records=[]
 def save(a):
  a=np.asarray(a);key=hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()[:24];arrays[key]=a;return key
 origbb=pr10.block_bootstrap;origsc=pr10.score_rows
 def bb(pa,pb,blocks,B=2000,seed=2026,margin=None):
  z=origbb(pa,pb,blocks,B,seed,margin);records.append({'kind':'paired_query_bootstrap','pa':save(pa),'pb':save(pb),'blocks':save(blocks),'B':B,'seed':seed,'margin':margin,'expected':z});return z
 def sc(*a,**kw):
  z=origsc(*a,**kw)
  records.append({'kind':'query_score_mean','scores':{k:save(v) for k,v in z.items()},'expected_means':{k:float(np.asarray(v).mean()) for k,v in z.items()}});return z
 pr10.block_bootstrap=pr10.bootstrap.block_bootstrap=bb;pr10.score_rows=pr10.scoring.score_rows=sc
 dest.mkdir(parents=True,exist_ok=True);sys.argv=[str(script),*extra,'--out',str(dest/'analysis_output')]
 if script.name=='design_ablation_b1.py':
  m=importlib.import_module('design_ablation_b1');expected=json.loads((S3/'results/b1_design_ablation.json').read_text());settings=[]
  def cell(label,c,corr,w0,wM,stored,fails):
   E=m.arms(c['Y'],corr,w0,wM);scores={arm:m.exp_scores(E[arm],c) for arm in m.ARMS}
   row=expected['cells'][label]
   for arm,x in scores.items():assert abs(float(x.mean())-row['means'][arm])<1e-12,(label,arm)
   if '(as executed)' not in label:
    settings.append({'label':label,'scores':{k:save(v) for k,v in scores.items()},'blocks':save(c['blocks']),'w0':save(w0),'wM':save(wM),'energy_removed':save(1-((c['Y']-corr)**2).sum(0)/np.maximum((c['Y']**2).sum(0),1e-300)),'expected':row})
   return row
  m.run_cell=cell;m.main();(dest/'factorial_settings.json').write_text(json.dumps(settings,indent=2));assert len(settings)==14
 else:runpy.run_path(str(script),run_name='__main__')
 np.savez_compressed(dest/'arrays.npz',**arrays);(dest/'records.json').write_text(json.dumps(records,separators=(',',':')));(dest/'source.json').write_text(json.dumps({'script':str(script.relative_to(REPRO)),'sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'start_level':'saved query-direction scores and bootstrap inputs; not raw counts'},indent=2))

if len(sys.argv)>1 and sys.argv[1]=='one':export_one(Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4:]);raise SystemExit
WORK.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(BASE));from reproduce_early import analyses
names={'stage1_energy_source','stage1_reliability','evaluate_stage2','diagnostics_stage2','evaluate_w1_e4','evaluate_w1_e4_50','diagnostics_w1_e4','evaluate_w1_e5','evaluate_d1_confirm','evaluate_w1_e1','evaluate_w1_e3','posthoc_nonquery_decomposition','correct_d1_self_gene','design_ablation_b1','cross_donor_b2','evaluate_b3_baselines','evaluate_b6_entry_shrinkage','evaluate_d2_prospective','direction1_q1_similarity','direction1_q2_cases','evaluate_b8_protein_readout','b8_floor_headroom','evaluate_depth_ladder'}
# Factorial first, then all other already-validated saved-input analyses.
tasks=sorted([x for x in analyses() if x['script'].stem in names],key=lambda t:t['script'].stem!='design_ablation_b1')
rows=[];started=time.monotonic()
for t in tasks:
 name=t['script'].stem;dest=WORK/name;dest.mkdir(exist_ok=True);env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 st=time.monotonic();p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'one',str(t['script']),str(dest),*t['extra']],env=env,text=True,capture_output=True,timeout=180);(dest/'export.log').write_text(p.stdout+p.stderr)
 row={'analysis':name,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-st};rows.append(row);print(json.dumps(row),flush=True)
 (WORK/'EXPORT_PROGRESS.json').write_text(json.dumps(rows,indent=2))
 if p.returncode:print(p.stderr[-2000:],flush=True)
print('TOTAL_SECONDS',time.monotonic()-started)
