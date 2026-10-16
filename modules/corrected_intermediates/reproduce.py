"""Paper inference from compact saved intermediates; never refit models or controls."""
from pathlib import Path
import argparse,json,sys,subprocess,os,time
H=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--fixed-root',type=Path,default=H.parent/'fixed_tasks');a=p.parse_args();out=a.output_dir.resolve();assert not out.exists();out.mkdir(parents=True);t=time.time()
 report={'status':'RUNNING','tier':'compact_intermediate_recalculation','components':[],'not_recomputed':['raw counts or preprocessing','control geometry','target reliability weights','native estimator fitting','original A19 cell selection'],'outside_scope':['SN1.1 two-input external RNA replacement','protein tasks covered elsewhere']}
 for name in ['guide','a18r','celleq','a19r']:
  cmd=[sys.executable,str(H/name/'reproduce.py'),'--output-dir',str(out/name)]
  if name=='a19r':cmd+=['--fixed-root',str(a.fixed_root.resolve())]
  with (out/(name+'.log')).open('w')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
  if r.returncode:raise RuntimeError((name,r.returncode,str(out/(name+'.log'))))
  rec=json.loads((out/name/'REPRODUCTION_REPORT.json').read_text());assert rec['status']=='PASS';report['components'].append({'module':name,'status':'PASS','receipt':str(out/name/'REPRODUCTION_REPORT.json'),'tier':rec.get('tier',rec.get('reproduction_level',rec.get('scope')))})
 report.update(status='PASS',seconds=time.time()-t);(out/'REPRODUCTION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
