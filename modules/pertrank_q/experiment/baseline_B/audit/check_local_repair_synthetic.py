"""Independent rational audit of the post-hoc local capacity rule; synthetic only."""
from pathlib import Path
from fractions import Fraction
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'audit'))
import local_repair_probe as p

def ranks_one(row,own,k):
 result=[Fraction(0) for _ in row];remain=k
 for val in sorted(set(row[~own]),reverse=True):
  ids=[i for i,x in enumerate(row) if not own[i] and x==val]
  take=min(remain,len(ids))
  for i in ids:result[i]=Fraction(take,len(ids))
  remain-=take
  if remain==0:break
 assert remain==0
 return result

def reference(score,base,own,k,m):
 result=np.zeros((2,)+score.shape)
 for d,sign in enumerate([1,-1]):
  for q in range(len(base)):
   core=ranks_one(sign*base[q],own[q],k-m);top=ranks_one(sign*base[q],own[q],k+m)
   cap=[t-c for t,c in zip(top,core)];alloc=[Fraction(0) for c in cap];remain=Fraction(m)
   for val in sorted(set((sign*score[q])[~own[q]]),reverse=True):
    ids=[i for i,x in enumerate(sign*score[q]) if x==val and cap[i]>0 and not own[q,i]]
    total=sum([cap[i] for i in ids],Fraction(0))
    if total==0:continue
    used=min(remain,total)
    for i in ids:alloc[i]=cap[i]*used/total
    remain-=used
    if remain==0:break
   assert remain==0
   result[d,q]=[float(c+a) for c,a in zip(core,alloc)]
 return result

def main():
 rng=np.random.default_rng(9172);n=0;maxerr=0.
 for k,m,t in [(3,1,7),(5,2,10),(10,2,13),(10,2,27)]:
  for trial in range(25):
   base=rng.integers(-2,3,(4,t)).astype(float);scores=rng.integers(-2,3,(4,t)).astype(float)
   if trial%3==0:base[0]=0
   if trial%4==0:scores[1]=0
   own=np.zeros((4,t),bool);own[np.arange(4),rng.integers(t,size=4)]=True
   got,core,cap=p.repair(scores,base,own,k,m);expect=reference(scores,base,own,k,m)
   err=float(np.max(abs(got-expect)));maxerr=max(maxerr,err);assert err<1e-12
   baseline,_,_=p.repair(base,base,own,k,m);np.testing.assert_allclose(baseline,reference(base,base,own,k,m),atol=1e-12,rtol=0)
   perm=rng.permutation(t);a,_,_=p.repair(scores[:,perm],base[:,perm],own[:,perm],k,m);np.testing.assert_allclose(a,got[:,:,perm],atol=1e-12,rtol=0)
   a,_,_=p.repair(-scores,-base,own,k,m);np.testing.assert_allclose(a,got[::-1],atol=1e-12,rtol=0)
   assert np.all(got>=core-1e-12) and np.all(got<=core+cap+1e-12);n+=1
 bug=None
 try:
  base=np.zeros((2,15));own=np.zeros_like(base,bool)
  got,_,_=p.repair(base,base,own,5,2)
  np.testing.assert_allclose(got,np.full((2,2,15),5/15),atol=1e-12,rtol=0)
 except Exception as e:bug=type(e).__name__+': '+str(e)
 report={'synthetic_only':True,'reference_access':False,'rational_comparison_cases':n,'maximum_absolute_error':maxerr,'ties_baseline_permutation_signflip_passed':True,'no_self_target_case_passed':bug is None,'no_self_target_error':bug,'scorer_sha256':hashlib.sha256((ROOT/'audit/local_repair_probe.py').read_bytes()).hexdigest(),'passed':bug is None}
 (ROOT/'audit/LOCAL_REPAIR_INDEPENDENT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
