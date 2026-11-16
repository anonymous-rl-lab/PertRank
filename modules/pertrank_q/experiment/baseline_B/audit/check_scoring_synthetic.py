"""Independent exact-tie synthetic scoring audit. Never opens real data."""
from pathlib import Path
import sys, json, itertools, hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import score_boundary as s

def independent_membership(scores, own, k):
 out=np.zeros((2,)+scores.shape)
 for d,sign in enumerate((1,-1)):
  for q in range(len(scores)):
   eligible=np.flatnonzero(~own[q]);row=sign*scores[q]
   rem=k
   for value in sorted(set(row[eligible]),reverse=True):
    ids=eligible[row[eligible]==value]
    mass=min(rem,len(ids));out[d,q,ids]=mass/len(ids);rem-=mass
    if rem==0:break
   assert rem==0
 return out

def independent_p(scores,ref,own,k):
 return (independent_membership(scores,own,k)*independent_membership(ref,own,k)).sum(-1).mean(0)/k

def independent_error(scores,ref,own):
 out=[]
 for q in range(len(ref)):
  err=[]
  for i,j in itertools.combinations(np.flatnonzero(~own[q]),2):
   if ref[q,i]==ref[q,j]:continue
   ds=scores[q,i]-scores[q,j];dr=ref[q,i]-ref[q,j]
   err.append(.5 if ds==0 else float(ds*dr<0))
  out.append(np.mean(err) if err else .5)
 return np.array(out)

def independent_ci(diff,blocks):
 ids=np.unique(blocks); sums=np.array([sum(diff[blocks==b]) for b in ids]); n=np.array([sum(blocks==b) for b in ids])
 ix=np.random.default_rng(2026111518).integers(len(ids),size=(2000,len(ids)))
 return np.quantile(sums[ix].sum(1)/n[ix].sum(1),[.025,.975])

def fake(rng,t):
 q=9;ids=np.array([f'T{i}' for i in range(t)]);qids=ids[:q];own=qids[:,None]==ids[None,:]
 base=rng.integers(-4,5,size=(q,t)).astype(float)
 ref=rng.integers(-3,4,size=(q,t)).astype(float)
 p=dict(base=base,raw=rng.integers(-4,5,size=(q,t)).astype(float),euclidean=rng.integers(-4,5,size=(q,t)).astype(float),
  scores=rng.integers(-5,6,size=(3,2,q,t)).astype(float),seeds=np.array(s.SEEDS),arm_names=np.array(s.ARMS),
  ownmask=own,query_ids=qids,target_ids=ids,query_block=np.array([0,0,1,1,2,2,3,3,4]))
 e={k:p[k] for k in ('ownmask','query_ids','target_ids','query_block')}
 e.update(reference=ref,reference_replicates=np.repeat(ref[:,:,None],3,axis=2))
 return p,e

def main():
 assert s.ARMS==('base_context','boundary_context','boundary_pair')
 assert s.PRIMARY=='boundary_pair'
 assert s.COMPARATORS==('original_PertRank','base_context','boundary_context')
 rng=np.random.default_rng(90821); checks=0
 for trial in range(25):
  halves=[];direct=[];inputs=[]
  for t in (24,27):
   p,e=fake(rng,t);a,z=s.score_task(p,e);halves.append(a);inputs.append((p,e))
   scores=[p['base'],p['raw'],p['euclidean']]+list(p['scores'].mean(1))
   pq=np.array([independent_p(v,e['reference'],e['ownmask'],10) for v in scores]);direct.append(pq)
   np.testing.assert_allclose(a['precision_per_query'],pq,atol=1e-14,rtol=0)
   mem=np.array([independent_membership(v,e['ownmask'],10) for v in scores]);np.testing.assert_allclose(a['membership'],mem,atol=1e-14,rtol=0)
   err=np.array([independent_error(v,e['reference'],e['ownmask']) for v in scores]);np.testing.assert_allclose(a['pairwise_error_per_query'],err,atol=1e-14,rtol=0)
   spq=np.array([[independent_p(v,e['reference'],e['ownmask'],10) for v in arm] for arm in p['scores']]);np.testing.assert_allclose(a['seed_precision_per_query'],spq,atol=1e-14,rtol=0)
   for c,name in enumerate(s.COMPARATORS):
    ci=s.METHODS.index(name);np.testing.assert_allclose(a['primary_delta_per_query'][c],pq[s.PI]-pq[ci],atol=1e-14,rtol=0)
   for mi,name in enumerate(s.METHODS):
    expected=float((pq[mi]-pq[0]).mean());assert abs(z['backtrace'][name]['net_support_over_2QK']-expected)<1e-12
   bi,ai=(s.METHODS.index(x) for x in ('boundary_context','base_context'))
   np.testing.assert_allclose(a['secondary_context_delta_per_query'],pq[bi]-pq[ai],atol=1e-14,rtol=0)
   np.testing.assert_allclose(a['secondary_context_seed_delta_per_query'],spq[1]-spq[0],atol=1e-14,rtol=0)
   assert abs(z['secondary_context_backtrace']['net_support_over_2QK']-float((pq[bi]-pq[ai]).mean()))<1e-12
   checks+=10
  source,summary=s.score_source(halves);mean=np.mean(direct,axis=0)
  np.testing.assert_allclose(source['precision_per_query'],mean,atol=1e-14,rtol=0)
  for c,name in enumerate(s.COMPARATORS):
   ci=s.METHODS.index(name);delta=mean[s.PI]-mean[ci]
   np.testing.assert_allclose(summary['primary_contrasts'][name]['conditional95'],independent_ci(delta,source['query_block']),atol=1e-14,rtol=0)
  bi,ai=(s.METHODS.index(x) for x in ('boundary_context','base_context'));delta=mean[bi]-mean[ai]
  np.testing.assert_allclose(summary['secondary_context_contrast']['conditional95'],independent_ci(delta,source['query_block']),atol=1e-14,rtol=0)
  assert abs(summary['secondary_context_backtrace']['net_support_over_2QK']-float(delta.mean()))<1e-12
  for j,seed in enumerate(s.SEEDS):
   dd=np.mean([a['seed_precision_per_query'][1,j]-a['seed_precision_per_query'][0,j] for a in halves],axis=0)
   np.testing.assert_allclose(summary['secondary_context_seed_contrasts'][str(seed)]['conditional95'],independent_ci(dd,source['query_block']),atol=1e-14,rtol=0)
  checks+=8
 report={'passed':True,'synthetic_only':True,'real_reference_access':False,'two_half_trials':25,'checks':checks,'scorer_sha256':hashlib.sha256((ROOT/'score_boundary.py').read_bytes()).hexdigest(),'tested':['exact ties versus independently grouped ranks','score ensemble versus rank ensemble','seed axes','pairwise error versus brute force','primary comparison axes','turnover identities','two-half mean','paired block confidence intervals','secondary B-A per-task/source/seed comparisons and turnover']}
 (ROOT/'audit/SCORING_SYNTHETIC_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
