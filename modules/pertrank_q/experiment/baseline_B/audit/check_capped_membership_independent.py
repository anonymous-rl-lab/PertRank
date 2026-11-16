"""Independent synthetic audit of fixed Top8-core / Top12-cap list allocation."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def exact_member(x,own,k):
 out=np.zeros_like(x,dtype=float)
 for q,row in enumerate(x):
  elig=np.flatnonzero(~own[q]);v=row[elig]
  threshold=sorted(v,reverse=True)[k-1]
  higher=v>threshold;tied=v==threshold
  out[q,elig]=higher+tied*(k-higher.sum())/tied.sum()
 return out

def reference_allocate(scores,cap,budget=2.):
 result=np.zeros_like(cap,dtype=float)
 for q in range(len(scores)):
  remaining=budget
  for value in sorted(set(scores[q,cap[q]>0]),reverse=True):
   group=(scores[q]==value)&(cap[q]>0);mass=float(cap[q,group].sum())
   take=min(remaining,mass)
   result[q,group]=cap[q,group]*(take/mass)
   remaining-=take
   if remaining<=1e-12:break
  assert abs(remaining)<1e-10
 return result

def test_impl(function):
 rng=np.random.default_rng(2026111522);trials=0;maxdiff=0.
 for n in (13,17,26,35):
  for repetition in range(25):
   base=rng.integers(-3,4,size=(7,n)).astype(float) if repetition%2 else rng.normal(size=(7,n))
   score=rng.integers(-2,3,size=(7,n)).astype(float) if repetition%3==0 else rng.normal(size=(7,n))
   own=np.zeros_like(base,dtype=bool);own[np.arange(7),rng.integers(n,size=7)]=True
   for direction in (1.,-1.):
    b=direction*base;s=direction*score
    core=exact_member(b,own,8);cap=exact_member(b,own,12)-core
    expected=core+reference_allocate(s,cap)
    got=function(b,s,own)
    np.testing.assert_allclose(got,expected,rtol=0,atol=1e-12)
    np.testing.assert_allclose(got.sum(1),10,rtol=0,atol=1e-12)
    assert np.all(got>=core-1e-12) and np.all(got<=core+cap+1e-12)
    assert np.all(got[own]==0)
    baseline=function(b,b,own)
    np.testing.assert_allclose(baseline,exact_member(b,own,10),rtol=0,atol=1e-12)
    perm=rng.permutation(n);back=np.argsort(perm)
    changed=function(b[:,perm],s[:,perm],own[:,perm])[:,back]
    np.testing.assert_allclose(changed,got,rtol=0,atol=1e-12)
    maxdiff=max(maxdiff,float(abs(got-expected).max()));trials+=1
 return {'synthetic_only':True,'trials':trials,'baseline_identity':True,'candidate_permutation_equivariance':True,'self_exclusion':True,'fixed_mass_10':True,'core_and_cap_constraints':True,'independent_reference_max_difference':maxdiff}

if __name__=='__main__':
 from local_repair_probe import repair
 report=test_impl(lambda base,score,own:repair(score,base,own,10,2)[0][0])
 report['production_function']='local_repair_probe.repair'
 report['passed']=True
 (ROOT/'audit/LOCAL_REPAIR_INDEPENDENT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
