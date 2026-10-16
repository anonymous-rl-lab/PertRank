"""Evaluate completed official coefficients, never fit with deep references.
Missing native-expression-filter entries get pre-specified zero. Errors are not zeros.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from preflight_cell_states import load,A
from conditional_cell_bootstrap import sha
HERE=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--fit',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
 ip=Path(args.input);fp=Path(args.fit);op=Path(args.out);op.mkdir(exist_ok=True,parents=True)
 fz=json.loads((HERE/'MIXSCALE_EVAL_FREEZE.json').read_text())
 for f,h in fz['files'].items():assert sha(HERE/f)==h
 run=json.loads((fp/'RUN_COMPLETE.json').read_text());assert run['mode']=='full' and run['n_targets']==1200
 assert run['source_commit']=='3d3a2cd12e62382df01c056536d967534b2de3a9'
 status=json.loads((fp/'target_status.json').read_text());assert len(status)==1200
 targets=list(csv.DictReader(open(ip/'targets.csv')));qs=list(csv.DictReader(open(ip/'queries.csv')))
 fm=list(csv.DictReader(open(fp/'feature_name_map.csv')));fmap={r['source_feature_id']:r['actual_feature_id'] for r in fm};assert len(fmap)==len(fm) and len(set(fmap.values()))==len(fm)
 featureids=[fmap[q['feature_id']] for q in qs];qindex={v:i for i,v in enumerate(featureids)};assert len(qindex)==len(qs)
 s,X,cs,yt,yc=load('D3');c=cs['responsive'];nt=len(s['names']);assert nt==1200;del X
 assert [t['target_name'] for t in targets]==list(s['names'])
 E=np.zeros((len(qs),nt));tested=np.zeros_like(E,dtype=bool);methods={};hashes={}
 for j,st in enumerate(status):
  assert st['target_group']==targets[j]['target_group'] and st['index']==j+1
  f=fp/f'target_{j+1:04d}.csv';hashes[f.name]=sha(f)
  rows=list(csv.DictReader(open(f)));seen=set()
  for r in rows:
   g=r['gene_ID'];assert g in qindex and g not in seen;seen.add(g)
   v=float(r['beta_weight']);assert np.isfinite(v)
   E[qindex[g],j]=v;tested[qindex[g],j]=True
  assert len(rows)==st['n_query_returned'];methods[targets[j]['target_name']]=st['fit_methods']
 scores={a:A.exp_scores(e,c) for a,e in [('raw',c['Y']),('w0y',c['Y']*c['w0y']),('W1',c['Y']*c['W1']),('Mixscale_beta_weight',E)]}
 result={'status':'single opened-D3 development comparison; not external or prospective confirmation','implementation_commit':run['source_commit'],
 'ranking_output':'native beta_weight, signed query direction; native filtered pairs zero; no errors treated as zero',
 'targets':nt,'queries':len(qs),'methods_means':{a:float(v.mean()) for a,v in scores.items()},
 'native_filter_coverage':{'query_target_pairs_fitted':int(tested.sum()),'query_target_pairs_total':int(tested.size),'fraction_fitted':float(tested.mean()),'targets_with_no_query_returned':int((tested.sum(0)==0).sum())},
 'comparisons':{f'{a} - {b}':A.summ(A.block_bootstrap(scores[a],scores[b],c['blocks'])) for a,b in [('W1','Mixscale_beta_weight'),('w0y','Mixscale_beta_weight'),('Mixscale_beta_weight','raw')]},
 'interval_scope':'original query-block conditional interval; fixed cells, controls, features, candidates and deep truth; not full sampling uncertainty',
 'fit_methods':methods,'run':run,'input_fit_sha256':hashes,'freeze':fz}
 np.savez_compressed(op/'mixscale_prediction_and_scores.npz',coefficient=E,tested=tested,query_ids=np.array(featureids),target_names=s['names'],**{f'score_{a}':v for a,v in scores.items()})
 (op/'mixscale_D3_results.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['fit_methods','input_fit_sha256']},indent=2))
if __name__=='__main__':main()
