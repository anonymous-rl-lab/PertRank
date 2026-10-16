#!/usr/bin/env python3
"""Frozen post-hoc score-array reanalysis; no expression access or model fitting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import argparse,hashlib,json,datetime,zipfile,time
import numpy as np
P=Path(__file__).resolve().parent
ROOT=P.parents[2]
FR=ROOT/'fr_benchmark/resume_20261014/expanded_rank_transfer/evaluation'
ASH=ROOT/'fr_benchmark/ash_work/resume_20261014/depth_evaluation'
# Explicit paths avoid accidentally selecting a later replay.
SOURCES={
 'rest':FR/'D4_Rest_0d19950f-5b9c-44d4-a49b-aa3c9be8da6a',
 'stim':FR/'D4_Stim8hr_a5b0195b-a4da-417f-8209-c83d8246fcbc',
 'equal':FR/'equal_state_a138dbae-36d3-4bb3-bf39-0d4f9dd5e17c',
 'n20':ASH/'n20','n50':ASH/'n50','n100':ASH/'n100'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(name,value):
 p=P/name;t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def js(p):return json.loads(p.read_text())
def freeze():
 assert not (P/'FROZEN_ANALYSIS.json').exists()
 files={}
 for tag,d in SOURCES.items():
  names=['equal_state_arrays.npz','equal_state_report.json','OWNER_COMPLETION.json'] if tag=='equal' else ['evaluation_arrays.npz','evaluation_report.json','compact.npz']
  if tag in ['rest','stim']:names+=['OWNER_COMPLETION.json']
  else:
   if tag!='equal':names+=['POST_EXEC_VERIFIED.json','COMMON_THREE_ARM_MATCH.json']
  for n in names:
   p=d/n;assert p.exists();files[tag+'/'+n]={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
 save('FROZEN_ANALYSIS.json',dict(created_utc=now(),status='POSTHOC_INPUTS_AND_RULES_FROZEN_BEFORE_INTERACTION_COMPUTATION',script_sha256=sha(__file__),files=files,
  planned={'A':'I=(FRxPertRank-FRxEuclidean)-(PertRank-Euclidean), D4 Rest and Stim separately and equal-state combined; FR rank300. Component gains also reported.',
   'B':'J=(PertRank-ASH)_20-(PertRank-ASH)_100; n50 shown descriptively without additional selected contrast.',
   'metrics':{'primary':25,'secondary':10},'draws':2000,'intervals':'Unadjusted descriptive percentile 95% and 99% using numpy quantile linear.',
   'pairing':'Within each state identical paired query-block draws; combined uses saved independent-state bootstrap streams. Depths use identical saved blocks/weights/query order after exact verification.',
   'new_random_seed':None,'new_model_fits':0,'expression_matrices_read':False},
  scope='Known point estimates motivated this analysis. Local freeze is not prospective registration or independent confirmation. All intervals conditional on observed input cells, fixed references, candidate pools, panels and geometry. Same D4 donor in two states, not two biological replicates. Depth uses one nested chain, not donor/chain-population inference. No equivalence, non-inferiority, universal method independence or calibrated method-switching claim.'))
 print(json.dumps({'frozen':str(P/'FROZEN_ANALYSIS.json'),'sha256':sha(P/'FROZEN_ANALYSIS.json')}))
def verify_sources(m):
 assert sha(__file__)==m['script_sha256']
 for x in m['files'].values():
  assert Path(x['path']).stat().st_size==x['bytes'] and sha(x['path'])==x['sha256'],x['path']
def array_npz(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in (keys or z.files)}
def summ(point,draws):
 return {'point':float(point),'descriptive_conditional_95_interval':np.quantile(draws,[.025,.975]).tolist(),'descriptive_conditional_99_interval':np.quantile(draws,[.005,.995]).tolist(),'bootstrap_sd':float(np.std(draws,ddof=1)),'draws':len(draws)}
def run():
 start=time.monotonic();m=js(P/'FROZEN_ANALYSIS.json');verify_sources(m)
 assert not (P/'RESULTS.json').exists()
 a={t:array_npz(d/('equal_state_arrays.npz' if t=='equal' else 'evaluation_arrays.npz')) for t,d in SOURCES.items()}
 axes={t:array_npz(d/'compact.npz',['queries','candidates','blocks']) for t,d in SOURCES.items() if t!='equal'}
 reports={t:js(d/('equal_state_report.json' if t=='equal' else 'evaluation_report.json')) for t,d in SOURCES.items()}
 checks=[]
 def eq(name,x,y):assert np.array_equal(x,y),name;checks.append(name)
 def close(name,x,y):np.testing.assert_allclose(x,y,rtol=0,atol=1e-12);checks.append(name)
 for t in ['rest','stim','n20','n50','n100']:
  assert sha(SOURCES[t]/'evaluation_arrays.npz')==reports[t]['evaluation_arrays_sha256']
  assert sha(SOURCES[t]/'compact.npz')==reports[t]['compact_sha256']
  assert len(np.unique(axes[t]['queries']))==300 and len(np.unique(axes[t]['blocks']))==30
  w=a[t]['query_weights'];close(t+'_weight_sum',w.sum(1),np.ones(2000))
  ids=np.unique(axes[t]['blocks']);bc=np.zeros((2000,len(ids)),int)
  for i,b in enumerate(ids):bc[:,i]=np.sum(a[t]['sampled_query_blocks']==i,axis=1)
  counts=bc[:,np.searchsorted(ids,axes[t]['blocks'])]
  close(t+'_blocks_reproduce_weights',counts/counts.sum(1)[:,None],w)
  for k in [25,10]:
   score=a[t][('six' if t in ['rest','stim'] else 'four')+f'_arm_p{k}']
   close(t+f'_saved_bootstrap_p{k}',w@score.mean(0).T,a[t][f'query_only_bootstrap_p{k}'])
   close(t+f'_reported_means_p{k}',score.mean((0,2)),[reports[t]['results'][str(k)]['arms'][x]['point'] for x in reports[t]['arms']])
 for t in ['n50','n100']:
  for key in ['queries','candidates','blocks']:eq(t+'_'+key+'_same_n20',axes['n20'][key],axes[t][key])
  for key in ['query_weights','sampled_query_blocks']:eq(t+'_'+key+'_same_n20',a['n20'][key],a[t][key])
 assert reports['rest']['arms']==reports['stim']['arms']==['raw','euclidean','pertrank','fr','fr_euclidean','fr_pertrank']
 for t in ['n20','n50','n100']:assert reports[t]['arms']==['raw','euclidean','pertrank','ash']
 assert reports['equal']['arrays_sha256']==sha(SOURCES['equal']/'equal_state_arrays.npz')
 results={'A_geometry_interaction':{},'B_depth_interaction':{}};drawout={};queryout={}
 for k in [25,10]:
  states=[]
  for i,t in enumerate(['rest','stim']):
   scores=a[t][f'six_arm_p{k}'].mean(0);boot=a[t][f'query_only_bootstrap_p{k}'];pt=scores.mean(1)
   combboot=a['equal'][f'state_{i}_query_bootstrap_p{k}']
   close(t+f'_combined_saved_bootstrap_p{k}',a['equal'][f'state_{i}_query_weights']@scores.T,combboot)
   states.append(pt)
   vals={'raw_geometry':(pt[2]-pt[1],boot[:,2]-boot[:,1]),'FR_geometry':(pt[5]-pt[4],boot[:,5]-boot[:,4]),'I':((pt[5]-pt[4])-(pt[2]-pt[1]),(boot[:,5]-boot[:,4])-(boot[:,2]-boot[:,1]))}
   results['A_geometry_interaction'].setdefault(t,{})[str(k)]={n:summ(*v) for n,v in vals.items()}
   for n,v in vals.items():drawout[f'A_{t}_P{k}_{n}']=v[1]
   queryout[f'A_{t}_P{k}_I']=(scores[5]-scores[4])-(scores[2]-scores[1])
  boot=a['equal'][f'equal_state_bootstrap_p{k}'];pt=np.mean(states,0)
  close(f'equal_state_bootstrap_p{k}',boot,(a['equal'][f'state_0_query_bootstrap_p{k}']+a['equal'][f'state_1_query_bootstrap_p{k}'])/2)
  vals={'raw_geometry':(pt[2]-pt[1],boot[:,2]-boot[:,1]),'FR_geometry':(pt[5]-pt[4],boot[:,5]-boot[:,4]),'I':((pt[5]-pt[4])-(pt[2]-pt[1]),(boot[:,5]-boot[:,4])-(boot[:,2]-boot[:,1]))}
  results['A_geometry_interaction'].setdefault('equal_state',{})[str(k)]={n:summ(*v) for n,v in vals.items()}
  for n,v in vals.items():drawout[f'A_equal_state_P{k}_{n}']=v[1]
  depth={};qdepth={};bdepth={}
  for t in ['n20','n50','n100']:
   score=a[t][f'four_arm_p{k}'].mean(0);q=score[2]-score[3];b=a[t][f'query_only_bootstrap_p{k}'][:,2]-a[t][f'query_only_bootstrap_p{k}'][:,3]
   qdepth[t]=q;bdepth[t]=b;depth[t]=summ(q.mean(),b);drawout[f'B_{t}_P{k}_PR_minus_ASH']=b;queryout[f'B_{t}_P{k}_PR_minus_ASH']=q
  q=qdepth['n20']-qdepth['n100'];b=bdepth['n20']-bdepth['n100'];close(f'depth_P{k}_paired_direct',a['n20']['query_weights']@q,b)
  depth['J_n20_minus_n100']=summ(q.mean(),b);drawout[f'B_P{k}_J']=b;queryout[f'B_P{k}_J']=q
  results['B_depth_interaction'][str(k)]=depth
 for t,z in axes.items():
  for key in ['queries','candidates','blocks']:queryout[f'{t}_{key}']=z[key]
 for t,z in a.items():
  for key in z:
   if 'query_weights' in key or 'sampled' in key:drawout[f'{t}_{key}']=z[key]
 np.savez_compressed(P/'interaction_bootstrap_draws.npz',**drawout)
 np.savez_compressed(P/'per_query_contrasts_and_axes.npz',**queryout)
 results.update(status='COMPUTATION_COMPLETE_PENDING_SEPARATE_EXEC_VERIFICATION',created_utc=now(),frozen_analysis_sha256=sha(P/'FROZEN_ANALYSIS.json'),input_checks=checks,scope=m['scope'],runtime_seconds=time.monotonic()-start,outputs={n:sha(P/n) for n in ['interaction_bootstrap_draws.npz','per_query_contrasts_and_axes.npz']})
 save('RESULTS.json',results)
 print(json.dumps({'results':str(P/'RESULTS.json'),'A_P25':results['A_geometry_interaction']['equal_state']['25'],'B_P25':results['B_depth_interaction']['25'],'checks':len(checks),'elapsed_seconds':results['runtime_seconds']},ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run']);a=p.parse_args();globals()[a.mode]()
