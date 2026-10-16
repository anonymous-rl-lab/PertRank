#!/usr/bin/env python3
"""Correct full-axis D1 panel, then score frozen seven arms in a separate phase."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,time
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
STATE=ROOT/'v16_execution/assets/canonical/Pertrescue_v6/code/repro_v10/same_budget_weight/stage3/data/d1_confirm_state.npz'
MAP=STATE.parent/'d1_own_gene_map.npz'
COMPACT=ROOT/'v17_experiments/delivery/compact/D1_Rest_state.npz'
MOMENTS=ROOT/'v16_execution/inventory/stage1200/D1_Rest/full_gene_moments_smoke.npz'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def module(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def dump(p,z):Path(p).write_text(json.dumps(z,indent=2,allow_nan=False)+'\n')
def check():
 f=json.loads((HERE/'PANEL_SCORE_FREEZE.json').read_text())
 for p,want in f['files'].items():assert sha(ROOT/p)==want,('changed frozen input/code',p)
 return sha(HERE/'PANEL_SCORE_FREEZE.json')
def build():
 freeze=check();r=json.loads((HERE/'FULL_REBUILD_PASS.json').read_text());assert r['status']=='PASS'
 assert sha(HERE/'full_planning_truth.npz')==r['file_sha256']
 z=np.load(STATE);o=np.load(MAP);f=np.load(HERE/'full_planning_truth.npz');p=np.load(COMPACT)
 assert np.array_equal(f['target_ids'],o['ids'])and np.array_equal(f['genes_keep'],z['genes_keep'])
 assert np.array_equal(p['target_ids'],o['ids'])
 pool=(f['planning_sum']/100-z['cm']).T
 oldq=np.argsort(-pool.var(1))[:300]
 assert np.array_equal(oldq,z['q_resp']),'unfixed full-axis response selection must reproduce historical panel'
 Q=module('archived_queries',ROOT/'v9_review/code/Pertrescue_Code_v9/code/repro_v10/pr10/queries.py')
 qt=Q.tercile_queries(z['cm'],z['genes_keep'],'tcell-probe|query|')
 assert np.array_equal(qt,z['q_terc'])
 oldblocks=Q.make_blocks(pool[oldq]);oldterc=Q.make_blocks(pool[qt])
 assert np.array_equal(oldblocks,z['blocks_responsive']) and np.array_equal(oldterc,z['blocks_terciles'])
 gm={str(g):i for i,g in enumerate(z['genes_keep'])};own=np.array([gm.get(str(g),-1)for g in o['ids']])
 assert np.array_equal(own,o['own_row'])
 corrected=pool.copy()
 for j,g in enumerate(own):
  if g>=0:corrected[g,j]=0
 q=np.argsort(-corrected.var(1))[:300];blocks=Q.make_blocks(pool[q]);union=np.union1d(q,qt)
 # Full-gene input mean is already frozen from the original five actual input groups.
 m=np.load(MOMENTS);assert np.array_equal(m['target_ids'],o['ids'])and np.array_equal(m['gene_ids'],z['genes_keep'])
 means=m['mean'];y=(means[:,:,union]-z['cm'][union]).transpose(1,2,0)
 oldy=(means[:,:,z['qu']]-z['cm'][z['qu']]).transpose(1,2,0);del means
 truth=f['truth_sum'][:,union].T/100-z['ce'][union,None]
 oldtruth=f['truth_sum'][:,z['qu']].T/100-z['ce'][z['qu'],None]
 assert np.max(np.abs(oldy-p['original_log_effects']))<1e-12
 assert np.max(np.abs(oldtruth-p['truth']))<1e-12
 u={int(g):i for i,g in enumerate(union)}
 np.savez_compressed(HERE/'corrected_panel_predictions_locked.npz',original_log_effects=y,truth=truth,
  query_ids=z['genes_keep'][union],target_ids=o['ids'],weights=p['weights'],arm_names=p['arm_names'],
  responsive_positions=np.array([u[int(i)]for i in q]),blocks_responsive=blocks,
  tercile_positions=np.array([u[int(i)]for i in qt]),blocks_terciles=oldterc,
  corrected_response_keep_positions=q,union_keep_positions=union,
  old_query_log_effects=oldy,old_query_truth=oldtruth)
 names=lambda x:z['genes_keep'][x].tolist()
 report=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),identity='D1 deterministic full-axis panel correction',
  panel_score_freeze_sha256=freeze,original_panel_reproduced_exactly=True,original_blocks_reproduced_exactly=True,
  old_query_input_max_abs=float(np.max(np.abs(oldy-p['original_log_effects']))),old_query_truth_max_abs=float(np.max(np.abs(oldtruth-p['truth']))),
  original_response_genes=names(oldq),corrected_response_genes=names(q),removed=names(np.setdiff1d(oldq,q)),added=names(np.setdiff1d(q,oldq)),
  number_replaced=int(len(np.setdiff1d(oldq,q))),own_genes_corrected=int((own>=0).sum()),
  inputs_unchanged=True,all_seven_weight_arms_unchanged=True,no_D4_access=True,
  prediction_file='corrected_panel_predictions_locked.npz',prediction_sha256=sha(HERE/'corrected_panel_predictions_locked.npz'),
  full_rebuild_sha256=sha(HERE/'full_planning_truth.npz'),source_compact_sha256=sha(COMPACT))
 dump(HERE/'PANEL_PREDICTION_LOCK.json',report);print(json.dumps(report,indent=2),flush=True)
def score():
 check();lock=json.loads((HERE/'PANEL_PREDICTION_LOCK.json').read_text());assert sha(HERE/lock['prediction_file'])==lock['prediction_sha256']
 common=module('v17_scoring_common',ROOT/'v17_experiments/evaluation/common.py');ev=common.load_original()
 z=np.load(HERE/lock['prediction_file']);p=np.load(COMPACT);arms=z['arm_names'].astype(str).tolist();weights=z['weights']
 oldscores=np.load(ROOT/'v17_experiments/evaluation/results/combined/D1_Rest_query_scores.npz')
 arrays={};results={};parity={}
 for panel in ('responsive','terciles'):
  pk=panel+'_positions' if panel=='responsive'else'tercile_positions';bk='blocks_'+panel
  pos=p[pk];own=ev.own_mask(p['query_ids'][pos],p['target_ids'])
  a=np.array([[ev.score(z['old_query_log_effects'][g,pos]*weights[i,g,None,:],z['old_query_truth'][pos],own,25)for g in range(5)]for i in range(len(arms))])
  want=oldscores[panel+'_scores'];assert a.shape==want.shape and np.max(abs(a-want))<1e-12
  parity[panel]=float(np.max(abs(a-want)))
  pos=z[pk];own=ev.own_mask(z['query_ids'][pos],z['target_ids'])
  a=np.array([[ev.score(z['original_log_effects'][g,pos]*weights[i,g,None,:],z['truth'][pos],own,25)for g in range(5)]for i in range(len(arms))])
  arrays[panel+'_scores']=a;results[panel]=common.summarize(a,arms,z[bk],ev)
 np.savez_compressed(HERE/'D1_corrected_query_scores.npz',**arrays,arm_names=np.array(arms))
 report=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),donor='D1_Rest',identity='posthoc deterministic panel correction',
  panel_lock_sha256=sha(HERE/'PANEL_PREDICTION_LOCK.json'),score_file_sha256=sha(HERE/'D1_corrected_query_scores.npz'),
  legacy_score_parity_max_abs=parity,number_replaced=lock['number_replaced'],arms=arms,panels=results)
 dump(HERE/'D1_corrected_results.json',report);print(json.dumps(results['responsive']['repetition_average'],indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['build','score'],required=True);a=p.parse_args()
 build()if a.mode=='build'else score()
