#!/usr/bin/env python3
"""Packaging-only conversion of completed D3 predictions to exact top-K memberships."""
from pathlib import Path
import io,json,hashlib,zipfile
import numpy as np
BASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'d3';OUT.mkdir(exist_ok=True)
OLD=BASE/'package_v130/extracted/PertRank_External_Benchmark_Code_20261012'
refpath=next((BASE/'package_v140/extracted').rglob('rank150_default_lasso/compact.npz'))
with np.load(refpath,allow_pickle=False) as z:z={k:z[k] for k in z.files}
Q=z['queries'];T=z['candidates'];OWN=Q[:,None]==T[None,:]
arrays={'queries':Q,'targets':T,'blocks':z['blocks']};records={};sources=[]
def sha(b):return hashlib.sha256(b).hexdigest()
def bind(path,member=None,data=None):
 sources.append({'archive_or_path':str(path.relative_to(BASE)),'member':member,'sha256':sha(path.read_bytes() if data is None else data)})
def encode(name,effects,k):
 y=np.asarray(effects,dtype=np.float64)
 assert y.shape==(300,1200) and np.isfinite(y).all()
 s=np.stack((y,-y));s=np.where(OWN,-np.inf,s)
 cutoff=np.partition(s,1200-k,axis=-1)[...,-k,None]
 above=s>cutoff;tied=s==cutoff
 slots=k-above.sum(-1)
 # Membership values are preserved exactly with the original arithmetic.
 a=np.packbits(above,axis=-1,bitorder='little');b=np.packbits(tied,axis=-1,bitorder='little')
 aa=np.unpackbits(a,axis=-1,count=1200,bitorder='little').astype(bool)
 bb=np.unpackbits(b,axis=-1,count=1200,bitorder='little').astype(bool)
 assert np.array_equal(aa,above) and np.array_equal(bb,tied)
 assert np.array_equal(above.astype(float)+tied*(slots/tied.sum(-1))[...,None],aa.astype(float)+bb*((k-aa.sum(-1))/bb.sum(-1))[...,None])
 arrays[f'{name}__k{k}__above']=a;arrays[f'{name}__k{k}__tied']=b
for k in [25,10]:
 encode('reference',z['reference_scores'],k)
 for name,effect in [('raw',z['raw_scores'][0]),('euclidean',z['raw_scores'][0]*z['w0'][0]),('pertrank',z['raw_scores'][0]*z['wm'][0])]:encode(name,effect,k)
bind(refpath)
frzip=BASE/'package_v140/downloads/PertRank_FR_D3_All21_Compact_20261012.zip';bind(frzip)
with zipfile.ZipFile(frzip) as archive:
 for member in sorted(n for n in archive.namelist() if n.endswith('/compact.npz')):
  config=Path(member).parent.name.removesuffix('_rep0');data=archive.read(member);bind(frzip,member,data)
  with np.load(io.BytesIO(data),allow_pickle=False) as native:
   assert np.array_equal(native['targets'],T) and np.array_equal(native['query_ids'],Q)
   assert np.array_equal(native['input_effects'].T,z['raw_scores'][0])
   assert np.array_equal(native['weights_euclidean'],z['w0'][0]) and np.array_equal(native['weights_pertrank'],z['wm'][0])
   f=native['query_effect'].T
  for k in [25,10]:
   for arm,effect in [('fr',f),('fr_euclidean',f*z['w0'][0]),('fr_pertrank',f*z['wm'][0])]:encode(config+'__'+arm,effect,k)
  report=OLD/'evaluation/D3_Rest'/config/'evaluation_report.json'
  records[config]=json.loads(report.read_text());bind(report)
  print(config,flush=True)
ashzip=BASE/'package_v140/downloads/ASH_D3_default_replay_20261014.zip';bind(ashzip)
member='ash_work/resume_20261014/D3_replay/default_uniform_fdr/posterior.npz'
with zipfile.ZipFile(ashzip) as archive:data=archive.read(member);bind(ashzip,member,data)
with np.load(io.BytesIO(data),allow_pickle=False) as az:
 ti={x:i for i,x in enumerate(az['row_ids'].astype(str))};gi={x:i for i,x in enumerate(az['genes'].astype(str))}
 assert set(ti)==set(T) and set(Q).issubset(gi)
 f=az['posterior'][np.ix_([ti[x] for x in T],[gi[x] for x in Q])].T
encode('ash',f,25)
ashreport=OLD/'ash_work/resume_20261014/D3_DEFAULT_REPLAY_CONFIRMATION.json';records['ash_default']=json.loads(ashreport.read_text());bind(ashreport)
np.savez_compressed(OUT/'memberships.npz',**arrays)
(OUT/'expected_reports.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
(OUT/'SOURCE_MANIFEST.json').write_text(json.dumps({'format':1,'sources':sources,'arms_fr':['raw','euclidean','pertrank','fr','fr_euclidean','fr_pertrank'],'candidate_count':1200,'query_count':300,'input_repeats':[0],'query_bootstrap_seed':20261012,'query_bootstrap_draws':2000,'shape_membership':[2,300,1200],'packing':'numpy packbits along target dimension, little bit order; exact strictly-above and exact tied-at-cutoff masks','top_k':[25,10],'ash_top_k':[25],'source_fits_refit':False,'original_expression_read':False,'representation_checks':'For every stored arm and K, above/tied bitmasks decoded exactly and resulting float64 membership probabilities identical.'},indent=2,ensure_ascii=False)+'\n')
print('membership bytes',(OUT/'memberships.npz').stat().st_size,'total raw',sum(p.stat().st_size for p in OUT.iterdir()),flush=True)
