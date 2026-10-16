"""Exact query-score and fixed-group inference replay; no cell or weight refitting."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
from bootstrap import block_bootstrap
from donor_bootstrap import donor_mean_bootstrap
H=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();out=a.output_dir.resolve();assert not out.exists();out.mkdir(parents=True)
 t=time.time();cache={};results={};means={};checked=0;err=0.
 for p in sorted((H/'capture').rglob('*.json')):
  rec=json.loads(p.read_text());z=np.load(p.with_suffix('.npz'),allow_pickle=False);label=str(p.relative_to(H/'capture'))
  results[label]=[];means[label]=[]
  for op in rec['operations']:
   key=json.dumps({k:v for k,v in op.items()if k not in ('expected','original_result_paths')},sort_keys=True)
   if key not in cache:
    fn=block_bootstrap if op['kind']=='block_bootstrap'else donor_mean_bootstrap
    cache[key]=fn(*[z[k]for k in op['arrays']],*op['args'],**op['kwargs'])
   got=cache[key];want=op['expected'];assert set(got)==set(want)
   for k,v in want.items():
    e=float(np.max(np.abs(np.asarray(got[k])-np.asarray(v))));err=max(err,e);assert e<=1e-12,(label,k,e)
   results[label].append(got);checked+=1
  for m in rec['score_means']:
   v=float(z[m['array']].mean());assert abs(v-m['expected_mean'])<=1e-15;means[label].append(v)
 # Fixed per-query correlation and weight vectors are retained diagnostic intermediates.
 diagnostics={};dz=np.load(H/'fixed_diagnostics.npz',allow_pickle=False)
 for mode in ['legacy','corrected']:
  old=json.loads((H/'archive'/mode/'a15_guide_split.json').read_text());diagnostics[mode]={}
  for donor in ['D3_Rest','D1_Rest','D2_Rest']:
   med=float(np.median(dz[f'{mode}_{donor}_correlation']));assert med==old['gate'][donor]['median_across_candidate_correlation']
   diagnostics[mode][donor]={'median_across_candidate_correlation':med,'weights':{}}
   for label,tag in [('A shallow vs B deep (primary)',''),('B shallow vs A deep (reverse)','B')]:
    w0=dz[f'{mode}_{donor}_w0{tag}'];w1=dz[f'{mode}_{donor}_w1{tag}'];v={'W1_zero_fraction':float((w1<=0).mean()),'w0y_zero_fraction':float((w0<=0).mean()),'W1_mean':float(w1.mean()),'w0y_mean':float(w0.mean())}
    for k,x in v.items():assert x==old['donors'][donor]['assignments'][label]['weights'][k]
    diagnostics[mode][donor]['weights'][label]=v
 (out/'fixed_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
 # Fresh observed lane summaries and archived null calibration from exact counts.
 lane={};null_expected=json.loads((H/'archive/legacy/a15b_lane_null_calibration.json').read_text());sims=null_expected['integrity']['simulations_per_target'];seed=null_expected['integrity']['seed']
 for donor in ['D3_Rest','D1_Rest','D2_Rest']:
  z=np.load(H/f'{donor}_lane_counts.npz');c=z['count_100'];rng=np.random.default_rng(seed);obs=[];null=[]
  for row in c:
   d=[x.astype(float)/x.sum() for x in row];obs.append(.5*float(np.abs(d[0]-d[1]).sum()));p=row.sum(0).astype(float);p/=p.sum()
   null.append(float(np.mean([.5*float(np.abs(rng.multinomial(100,p)/100-rng.multinomial(100,p)/100).sum())for _ in range(sims)])))
  obs=np.array(obs);null=np.array(null);r={'targets':len(obs),'lanes':len(z['lane_categories']),'observed_tvd':{'median':float(np.median(obs)),'quartiles':[float(np.percentile(obs,q))for q in [25,75]]},'null_tvd':{'median':float(np.median(null)),'quartiles':[float(np.percentile(null,q))for q in [25,75]]},'fraction_observed_above_null':float(np.mean(obs>null)),'median_difference':float(np.median(obs-null)),'fraction_disjoint_lanes':float(np.mean(~np.any((c[:,0]>0)&(c[:,1]>0),axis=1)))}
  assert r==null_expected['donors'][donor];lane[donor]=r
 (out/'inference_results.json').write_text(json.dumps(results,indent=2)+'\n');(out/'score_means.json').write_text(json.dumps(means,indent=2)+'\n');(out/'lane_calibration.json').write_text(json.dumps(lane,indent=2)+'\n')
 report={'status':'PASS','tier':'frozen_query_scores_and_group_labels_to_means_and_intervals','seconds':time.time()-t,'bootstrap_calls_checked':checked,'unique_bootstrap_computations':len(cache),'score_means_checked':sum(map(len,means.values())),'max_abs_error':err,'lane_calibration':'recalculated_from_integer_counts_exact','not_recomputed':['guide expression effects','target weights','candidate reranking','original guide cell assignment'],'archive':'Original full result reports retained separately; unrecalculated constants and original decision text remain historical records; weight summaries are recalculated from retained fixed weight vectors.'}
 (out/'REPRODUCTION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
