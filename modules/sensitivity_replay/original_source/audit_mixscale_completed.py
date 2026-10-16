"""Read-only completion/score audit; requires all 1200 official native fits."""
from pathlib import Path
import argparse,csv,json,hashlib,sys
import numpy as np
A=argparse.ArgumentParser();A.add_argument('--fit',required=True);A.add_argument('--results',required=True);args=A.parse_args()
OUT=Path(__file__).resolve().parent;BASE=OUT.parents[1];SN=BASE/'strengthening_neighbor';F=Path(args.fit);Q=Path(args.results)
sys.path.insert(0,str(SN));from preflight_cell_states import load
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((Q/'mixscale_D3_results.json').read_text());run=json.loads((F/'RUN_COMPLETE.json').read_text());status=json.loads((F/'target_status.json').read_text());stored=np.load(Q/'mixscale_prediction_and_scores.npz')
assert run['mode']=='full' and run['n_targets']==1200 and len(status)==1200
assert run['source_commit']=='3d3a2cd12e62382df01c056536d967534b2de3a9'
assert len(list(F.glob('target_*.csv')))==1200
assert len(list(F.glob('status_*.json')))==1200
assert not list(F.glob('error_*.json')), 'native errors must not be recoded as filtered zeros'
for k,v in r['freeze']['files'].items():assert sha(SN/k)==v,(k,'freeze mismatch')
s,X,cs,yt,yc=load('D3');del X;c=cs['responsive'];T=c['T'];nq,nt=T.shape;assert nt==1200 and nq==300
ids=list(csv.DictReader((SN/'d3_prepared/targets.csv').open()));qs=list(csv.DictReader((SN/'d3_prepared/queries.csv').open()));fm=list(csv.DictReader((F/'feature_name_map.csv').open()));mapf={x['source_feature_id']:x['actual_feature_id'] for x in fm};assert len(mapf)==len(fm)==18130 and len(set(mapf.values()))==18130
qids=[mapf[x['feature_id']] for x in qs];qi={v:i for i,v in enumerate(qids)};assert len(qi)==300 and qids==list(stored['query_ids']) and list(s['names'])==list(stored['target_names'])
source_q_index={x['feature_id']:i for i,x in enumerate(qs)}
identity_own=np.array([source_q_index.get(x['target_group'],-1) for x in ids])
assert np.array_equal(identity_own,c['sq']), 'own exclusion must use true feature IDs'
E=np.zeros((nq,nt));tested=np.zeros((nq,nt),bool);native={};nweighted=0
for j,st in enumerate(status):
 assert st['index']==j+1 and st['target_group']==ids[j]['target_group'] and ids[j]['target_name']==s['names'][j]
 assert json.loads((F/f'status_{j+1:04d}.json').read_text())==st
 assert st['execution_adapter']=='verified target-local exact representation'
 assert st['n_all_returned']>=st['n_query_returned']
 declared=st['fit_methods'];declared={declared} if isinstance(declared,str) else set(declared)
 f=F/f'target_{j+1:04d}.csv';rows=list(csv.DictReader(f.open()));assert len(rows)==st['n_query_returned'];seen=set()
 assert sha(f)==r['input_fit_sha256'][f.name]
 for x in rows:
  gene=x['gene_ID'];assert gene in qi and gene not in seen;seen.add(gene);v=float(x['beta_weight']);assert np.isfinite(v)
  E[qi[gene],j]=v;tested[qi[gene],j]=True
  method=x['DE_method'];assert method in declared;native[method]=native.get(method,0)+1
 if st.get('native_score_present',False):nweighted+=1
assert np.array_equal(E,stored['coefficient']) and np.array_equal(tested,stored['tested'])
assert tested.sum()==r['native_filter_coverage']['query_target_pairs_fitted'] and tested.size==r['native_filter_coverage']['query_target_pairs_total']
assert nweighted==run['n_native_weighted']
# Full-sort exact tie probabilities, independent of original scorer/partition implementation.
def pr(E):
 v=np.stack([E,-E],axis=1).reshape(2*nq,nt).copy();mask=np.repeat(np.arange(nq),2)[:,None]==c['sq'][None,:];v[mask]=-np.inf
 threshold=np.sort(v,axis=1)[:,-25,None];hi=v>threshold;eq=v==threshold
 p=hi.astype(float)+eq*((25-hi.sum(1))/eq.sum(1))[:,None]
 assert np.allclose(p.sum(1),25) and not np.any(p[mask]);return p
truth=pr(T);scores={}
for name,effect in [('raw',c['Y']),('w0y',c['Y']*c['w0y']),('W1',c['Y']*c['W1']),('Mixscale_beta_weight',E)]:
 v=(pr(effect)*truth).sum(1).reshape(nq,2)/25;scores[name]=v
 assert np.max(abs(v-stored['score_'+name]))<1e-12
 assert abs(float(v.mean())-r['methods_means'][name])<1e-12
original_means={'raw':.2773123584257502,'w0y':.2757554779794853,'W1':.3113689768996245}
assert max(abs(r['methods_means'][k]-v) for k,v in original_means.items())<1e-12
# Independently recompute original B2000 query-block intervals by sampled block
# sums/counts, not the archived concatenate-row implementation. No new inferential rule.
blocks=np.unique(c['blocks']);members=[np.flatnonzero(c['blocks']==b) for b in blocks]
counts=np.array([len(x) for x in members]);rng=np.random.default_rng(2026)
draw_index=rng.choice(len(blocks),size=(2000,len(blocks)),replace=True)
ci_check={}
for left,right in [('W1','Mixscale_beta_weight'),('w0y','Mixscale_beta_weight'),('Mixscale_beta_weight','raw')]:
 d=(scores[left]-scores[right]).mean(1);sums=np.array([d[x].sum() for x in members])
 draws=sums[draw_index].sum(1)/counts[draw_index].sum(1)
 computed={'diff':float(d.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),'p_one_sided':float(np.mean(draws<=0))}
 label=left+' - '+right;wanted=r['comparisons'][label]
 diff=max(abs(computed['diff']-wanted['diff']),float(np.max(abs(np.array(computed['ci95'])-wanted['ci95']))),abs(computed['p_one_sided']-wanted['p_one_sided']))
 assert diff<=1e-12,(label,computed,wanted)
 ci_check[label]={'independent':computed,'maximum_numeric_difference':diff}
report={'status':'PASS','scope':'completed official native D3 development benchmark with original candidate/query set; not external or prospective confirmation','targets':1200,'queries':300,'all_csv_hashes_verified':True,'native_query_pairs':int(tested.sum()),'all_query_pairs':int(tested.size),'native_coefficient_and_coverage_exact':True,'independent_fullsort_score_max_abs':float(max(np.max(abs(v-stored['score_'+name])) for name,v in scores.items())),'native_methods_returned_query_rows':native,'targets_with_native_score':nweighted,'means':r['methods_means'],'comparisons':r['comparisons'],'freeze_hashes_verified':True,'source_hashes':{str(f):sha(f) for f in [F/'RUN_COMPLETE.json',F/'target_status.json',Q/'mixscale_D3_results.json',Q/'mixscale_prediction_and_scores.npz']}}
report.update(independent_query_block_CI_check=ci_check,original_raw_w0y_W1_means_reproduced=True,own_exclusion_feature_identity_exact=True,native_error_files=0,individual_statuses_equal_aggregate=True,
 estimation_scope='Native signed beta_weight ranking under specified official expression filters; missing filtered pairs zero, no failed fit zero-fill. Conditional query-block intervals with source cells/control/query/candidate/deep reference fixed; not DE calibration, all Mixscale outputs, or cross-study generality.')
(OUT/'independent_mixscale_completed_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
