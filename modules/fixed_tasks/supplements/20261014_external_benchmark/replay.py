#!/usr/bin/env python3
"""Offline reproduction of archived external-benchmark statistical summaries.
No fitting, expression access, network access, new random draws or new analysis.
"""
import os
for _v in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[_v]='1'
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
ATOL=1e-12

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(a):
 h=hashlib.sha256();h.update(a.dtype.str.encode());h.update(json.dumps(list(a.shape),separators=(',',':')).encode());h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()

def load_data():
 m=json.loads((HERE/'SUFFICIENT_DATA.json').read_text())
 if sha(HERE/'scores.npz')!=m['archive_sha256']:raise ValueError('Score archive SHA mismatch')
 decoded={}
 with np.load(HERE/'scores.npz',allow_pickle=False)as z:
  for k,r in m['arrays'].items():
   if r['codec']=='column_dictionary':a=z[k+'_values'][z[k+'_codes']][:,z[k+'_column_index']]
   elif r['codec']=='dictionary':a=z[k+'_values'][z[k+'_codes']]
   elif r['codec']=='narrow_integer':a=z[k+'_data'].astype(np.dtype(r['dtype']))
   else:a=z[k+'_data']
   if a.dtype.str!=r['dtype']or list(a.shape)!=r['shape']or array_sha(a)!=r['array_sha256']:raise ValueError('Array identity mismatch: '+k)
   decoded[k]=a
 data={}
 for name,k in m['bindings'].items():
  g,f=name.split('/',1);data.setdefault(g,{})[f]=decoded[k]
 return m,data

def compare(actual,expected,path='results'):
 if isinstance(expected,dict):
  if set(actual)!=set(expected):raise AssertionError('Keys differ: '+path)
  return sum(compare(actual[k],v,path+'.'+k)for k,v in expected.items())
 if isinstance(expected,list):
  if len(actual)!=len(expected):raise AssertionError('Length differs: '+path)
  return sum(compare(a,b,path+f'[{i}]')for i,(a,b)in enumerate(zip(actual,expected)))
 if isinstance(expected,(int,float))and not isinstance(expected,bool):
  if not np.isfinite(actual)or abs(float(actual)-expected)>ATOL:raise AssertionError(f'{path}: {actual} != {expected}')
  return 1
 if actual!=expected:raise AssertionError('Metadata differs: '+path)
 return 0

def infer_pair(name,arms):
 for a in arms:
  for b in arms:
   if a+'_minus_'+b==name:return arms.index(a),arms.index(b)
 raise ValueError('Unknown contrast '+name)

def summarize(scores,boots,arms,template):
 point=scores.mean(axis=(0,2));rep=scores.mean(2);out={}
 for section,items in template.items():
  if section not in ('arms','contrasts'):raise ValueError('Unknown result section: '+section)
  out[section]={}
  for name,fields in items.items():
   if section=='arms':
    a=arms.index(name);p=point[a];r=rep[:,a];bs={k:v[:,a]for k,v in boots.items()}
   else:
    a,b=infer_pair(name,arms);p=point[a]-point[b];r=rep[:,a]-rep[:,b];bs={k:v[:,a]-v[:,b]for k,v in boots.items()}
   item={}
   for field,value in fields.items():
    if field=='point':item[field]=float(p)
    elif field=='input_rep_points':item[field]=r.tolist()
    elif field.endswith('_95_interval')or field.endswith('_99_interval'):
     prefix,level,_=field.rsplit('_',2);q=[.025,.975]if level=='95'else[.005,.995]
     item[field]=np.quantile(bs[prefix],q).tolist()
    elif isinstance(value,bool)or value is None or isinstance(value,str):item[field]=value
    else:raise ValueError('Unimplemented numeric result field: '+field)
   out[section][name]=item
 return out

def write_csv(path,rows):
 if not rows:return
 with path.open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args()
 if a.output_dir.exists():p.error('--output-dir must be new')
 m,data=load_data();expected={};groups={g['id']:g for g in m['groups']}
 for g in m['groups']:
  rp=HERE/g['original_report']
  if sha(rp)!=g['report_sha256']:raise ValueError('Original report SHA mismatch: '+g['id'])
  expected[g['id']]=json.loads(rp.read_text())
 results={};scores_cache={};boot_cache={};checks=0;armrows=[];contrastrows=[]
 for g in sorted(m['groups'],key=lambda x:x['kind']=='equal_state'):
  gid=g['id'];d=data[gid];r=expected[gid];arms=g['arms'];results[gid]={'results':{},'source_report_sha256':g['report_sha256'],'source_path':g['source_path'],'reported_extra_contrasts':{}}
  for k,template in r['results'].items():
   if g['kind']=='task':
    sk=next(f for f in [f'six_arm_p{k}',f'four_arm_p{k}']if f in d)
    scores=d[sk];boot=d['query_weights']@scores.mean(0).T;boots={'query_only':boot}
    if f'joint_bootstrap_p{k}'in d:boots['joint']=d[f'joint_bootstrap_p{k}']
   else:
    states=g['states'];state_scores=[scores_cache[(s,k)]for s in states]
    if len(set(v.shape[2]for v in state_scores))!=1:raise ValueError('Original equal-state implementation requires equal query-panel sizes')
    state_boots=[d[f'state_{i}_query_weights']@v.mean(0).T for i,v in enumerate(state_scores)]
    boot=np.mean(state_boots,axis=0);scores=np.concatenate(state_scores,axis=2);boots={'task_stratified_query':boot}
   scores_cache[(gid,k)]=scores;boot_cache[(gid,k)]=boot
   got=summarize(scores,boots,arms,template);checks+=compare(got,template,gid+'.'+k);results[gid]['results'][k]=got
   for er in g.get('extra_reports',[]):
    ep=HERE/er['path']
    if sha(ep)!=er['sha256']:raise ValueError('Extra report SHA mismatch')
    et=json.loads(ep.read_text())['results'][k]
    eg=summarize(scores,boots,arms,{'contrasts':et})['contrasts']
    checks+=compare(eg,et,gid+'.extra.'+k)
    results[gid]['reported_extra_contrasts'].setdefault(k,{}).update(eg)
   label=r.get('dataset',','.join(r.get('datasets',[])))
   depth=r.get('depth','')
   config=r.get('config_id',r.get('winner_config_id',r.get('curve','')))
   if not label and depth:label='D3_Rest'
   task_kind=r.get('kind','equal_state'if g['kind']=='equal_state'else'pooled_posthoc_depth')
   for arm,vals in got['arms'].items():
    interval=next(v for field,v in vals.items()if field.endswith('_95_interval'))
    armrows.append(dict(group=gid,dataset=label,task_kind=task_kind,depth=depth,config=config,K=k,arm=arm,point=vals['point'],ci95_lower=interval[0],ci95_upper=interval[1]))
   all_contrasts={**got['contrasts'],**results[gid]['reported_extra_contrasts'].get(k,{})}
   for contrast,vals in all_contrasts.items():
    for field,v in vals.items():
     if field.endswith('_95_interval'):
      contrastrows.append(dict(group=gid,dataset=label,task_kind=task_kind,depth=depth,config=config,K=k,contrast=contrast,point=vals['point'],uncertainty=field,ci95_lower=v[0],ci95_upper=v[1]))
 a.output_dir.mkdir(parents=True)
 (a.output_dir/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
 write_csv(a.output_dir/'method_scores.csv',armrows);write_csv(a.output_dir/'paired_contrasts.csv',contrastrows)
 receipt={'status':'PASS_ARCHIVED_SCORE_STATISTICS_REPLAY','groups':len(groups),'numeric_values_compared':checks,'absolute_tolerance':ATOL,'original_data_sha256':m['archive_sha256'],'new_model_fits':0,'expression_matrices_read':False,'new_bootstrap_draws':0,'numpy_version':np.__version__,
 'scope':'Recomputes fixed-task score summaries and intervals from archived per-query scores and saved bootstrap designs. Joint reference-bootstrap score draws retained. Does not refit methods from raw expression.'}
 (a.output_dir/'REPRODUCTION_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
