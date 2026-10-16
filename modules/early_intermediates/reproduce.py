#!/usr/bin/env python3
"""Recompute early-paper statistics from saved per-query scores, not raw expression."""
from pathlib import Path
import argparse,csv,hashlib,json,time,numpy as np
from bootstrap import block_bootstrap
ROOT=Path(__file__).resolve().parent

def compare(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  if set(a)!=set(b):return [path+': keys differ']
  return [d for k in a if k!='created_utc' for d in compare(a[k],b[k],path+'/'+k)]
 if isinstance(a,list) and isinstance(b,list):
  if len(a)!=len(b):return [path+': length differs']
  return [d for i,(x,y) in enumerate(zip(a,b)) for d in compare(x,y,path+'/'+str(i))]
 if isinstance(a,(int,float)) and isinstance(b,(int,float)):
  if np.isnan(a) and np.isnan(b):return []
  return [] if a==b or abs(a-b)<=1e-12*max(1,abs(b)) else [f'{path}: {a} != {b}']
 return [] if a==b else [f'{path}: {a!r} != {b!r}']

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args();out=args.output_dir.resolve()
 if out.exists() and any(out.iterdir()):ap.error('output-dir must be new or empty')
 out.mkdir(parents=True,exist_ok=True);start=time.monotonic();manifest=json.loads((ROOT/'MANIFEST.json').read_text());bad=[]
 for rel,want in manifest['files'].items():
  p=ROOT/rel
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=want:bad.append(rel)
 if bad:
  (out/'report.json').write_text(json.dumps({'status':'INTEGRITY_FAILED','files':bad},indent=2));return 1
 arrays=np.load(ROOT/'data/intermediates.npz',allow_pickle=False);catalog=json.loads((ROOT/'data/records.json').read_text());rows=[];fail=[]
 for name,records in catalog.items():
  mismatches=[];computed=[]
  for i,r in enumerate(records):
   if r['kind']=='paired_query_bootstrap':
    got=block_bootstrap(arrays[r['pa']],arrays[r['pb']],arrays[r['blocks']],r['B'],r['seed'],r['margin']);diff=compare(got,r['expected']);computed.append({'index':i,'kind':r['kind'],'value':got,'reference_paths':r.get('reference_paths',[])})
   else:
    got={k:float(arrays[v].mean()) for k,v in r['scores'].items()};diff=compare(got,r['expected_means']);computed.append({'index':i,'kind':r['kind'],'value':got})
   mismatches.extend([str(i)+': '+d for d in diff])
  (out/(name+'.json')).write_text(json.dumps(computed,indent=2)+'\n');rows.append({'analysis':name,'statistic_calls':len(records),'status':('RECOMPUTED_STATISTICS_MATCH' if records else 'ARCHIVED_OUTPUT_ONLY') if not mismatches else 'MISMATCH','mismatches':mismatches[:20]});fail.extend(mismatches)
 from replay_descriptive import run
 for filename,got in run(out).items():
  want=json.loads((ROOT/'archived_outputs'/filename).read_text());differences=compare(got,want);fail.extend([filename+': '+d for d in differences]);(out/filename).write_text(json.dumps(got,indent=2)+'\n')
  name={'stage1_reliability.json':'stage1_reliability','direction1_similarity.json':'direction1_q1_similarity','direction1_cases_D1.json':'direction1_q2_cases'}[filename]
  row=next(x for x in rows if x['analysis']==name);row.update(status='RECOMPUTED_DESCRIPTIVE_MATCH' if not differences else 'MISMATCH',descriptive_replay=True,mismatches=differences[:20])
 settings=json.loads((ROOT/'data/factorial_settings.json').read_text());factorial={}
 for r in settings:
  scores={k:arrays[v] for k,v in r['scores'].items()};blocks=arrays[r['blocks']]
  means={k:float(v.mean()) for k,v in scores.items()};contrasts={}
  for label,want in r['expected']['comparisons'].items():
   a,b=label.split(' - ');bb=block_bootstrap(scores[a],scores[b],blocks);got={k:bb[k] for k in want};contrasts[label]=got;fail.extend([r['label']+': '+d for d in compare(got,want)])
  fail.extend([r['label']+': '+d for d in compare(means,r['expected']['means'])])
  energy=float(np.median(arrays[r['energy_removed']]));want=r['expected']['query_row_energy_removed_by_A']['median_over_targets'];fail.extend(compare(energy,want,r['label']+'/energy'))
  factorial[r['label']]={'means':means,'comparisons':contrasts,'median_energy_removed':energy,'mean_w0':float(arrays[r['w0']].mean()),'mean_wM':float(arrays[r['wM']].mean())}
 (out/'factorial_14_settings.json').write_text(json.dumps(factorial,indent=2)+'\n')
 for r in csv.DictReader((ROOT/'data/v25_design_geometry.csv').open()):
  got=factorial[r['source_setting']]['comparisons'][r['contrast']];want={'diff':float(r['point']),'ci95':[float(r['lower_95']),float(r['upper_95'])]};fail.extend(compare({k:got[k] for k in want},want,'current_paper/'+r['setting']))
 report={'status':'STATISTICAL_REPLAY_MATCH' if not fail else 'MISMATCH','reproduction_level':'per-query score and target-statistic intermediates to means, paired bootstrap intervals and factorial contrasts','analysis_count':sum(x['statistic_calls']>0 or x.get('descriptive_replay') for x in rows),'archived_only_analysis_count':sum(x['status']=='ARCHIVED_OUTPUT_ONLY' for x in rows),'current_paper_geometry_rows':14,'statistic_calls':sum(x['statistic_calls'] for x in rows),'factorial_settings':len(factorial),'factorial_arms':6,'elapsed_seconds':time.monotonic()-start,'analyses':rows,'mismatches':fail[:40],'not_recomputed':['raw-cell preprocessing','gene effects or weights from full expression states','rank construction from candidate-by-gene scores','A14 missing per-cell protein input','scientific fields outside captured mean/bootstrap/target-statistic operations']}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['status','analysis_count','statistic_calls','factorial_settings','elapsed_seconds']},indent=2));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
