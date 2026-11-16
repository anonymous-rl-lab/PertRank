#!/usr/bin/env python3
"""Independent rational fractional-tie and complete-list arithmetic checks."""
from pathlib import Path
from fractions import Fraction
import hashlib,json
import numpy as np
from score_optimization import membership,pairwise_error,turnover,mass_summary,task_arrays,source_arrays,METHODS,ARMS,SEEDS,COMPARATORS,K
ROOT=Path(__file__).resolve().parent

def oracle(row,own,k,sign):
    allowed=[i for i in range(len(row)) if not own[i]]
    values=sorted([sign*row[i] for i in allowed],reverse=True)
    cut=values[k-1];n_above=sum(v>cut for v in values);n_equal=values.count(cut)
    p=Fraction(k-n_above,n_equal)
    return [Fraction(0) if own[i] or sign*row[i]<cut else Fraction(1) if sign*row[i]>cut else p for i in range(len(row))]

def main():
    rng=np.random.default_rng(2026111519);checks=0;maximum_error=0.
    for trial in range(100):
        q=3;t=int(rng.integers(14,28));s=rng.integers(-3,4,(q,t)).astype(float);r=rng.integers(-2,3,(q,t)).astype(float)
        own=np.zeros((q,t),bool);own[np.arange(q),rng.integers(t,size=q)]=True
        a=membership(s,own,K);b=membership(r,own,K)
        for d,sign in enumerate((1.,-1.)):
            for j in range(q):
                oa=oracle(s[j].tolist(),own[j].tolist(),K,sign);ob=oracle(r[j].tolist(),own[j].tolist(),K,sign)
                error=float(np.max(abs(a[d,j]-np.array(list(map(float,oa))))));maximum_error=max(maximum_error,error);assert error<1e-15
                value=float(sum(x*y for x,y in zip(oa,ob))/K)
                assert abs(value-float((a[d,j]*b[d,j]).sum()/K))<1e-14
                checks+=2
        # Counterpart score changes all parts of the full list; identity must still hold.
        c=membership(rng.normal(size=(q,t)),own,K)
        diff=((c-a)*b).sum(-1).mean(0)/K
        v=mass_summary(turnover(c,a,b),diff);assert abs(v['net_over_2QK']-diff.mean())<1e-12;checks+=1
        err,count=pairwise_error(s,r,own)
        for j in range(q):
            losses=[]
            for i in range(t):
                for z in range(i+1,t):
                    if own[j,i] or own[j,z] or r[j,i]==r[j,z]:continue
                    losses.append(.5 if s[j,i]==s[j,z] else float((s[j,i]>s[j,z])!=(r[j,i]>r[j,z])))
            assert count[j]==len(losses)
            assert abs(err[j]-(np.mean(losses) if losses else .5))<1e-14;checks+=1
        assert np.all(a[:,own]==0);assert np.max(abs(a.sum(-1)-K))<1e-10;checks+=2
    # Exercise task/source dimensions, seed score averaging, exact all-method identities.
    q,t=7,20;own=np.zeros((q,t),bool);own[np.arange(q),np.arange(q)]=True
    base=rng.normal(size=(q,t));scores=rng.normal(size=(len(ARMS),len(SEEDS),q,t));ref=rng.normal(size=(q,t))
    p=dict(base=base,raw=base*1.5,euclidean=base*.9,scores=scores,ownmask=own,
           query_ids=np.array([f'q{i}' for i in range(q)]),query_block=np.arange(q)//2,target_ids=np.array([f't{i}' for i in range(t)]))
    e={k:p[k] for k in ('ownmask','query_ids','query_block','target_ids')};e.update(reference_replicates=np.repeat(ref[:,:,None],3,axis=2))
    e['reference']=e['reference_replicates'].mean(-1)
    a,res=task_arrays(p,e);merged,sr=source_arrays([a,a])
    assert len(sr['means'])==len(METHODS)
    for i,name in enumerate(METHODS):assert sr['means'][name]==res['means'][name]
    for comp in COMPARATORS:
        for name in METHODS:assert abs(sr['backtrace'][comp][name]['net_over_2QK']-sr['contrasts'][comp][name]['delta'])<1e-12;checks+=1
    assert np.array_equal(merged['precision_per_query'],a['precision_per_query']);checks+=1
    result=dict(passed=True,checks=checks,random_cases=100,rational_membership_max_error=maximum_error,
        scoring_code_sha256=hashlib.sha256((ROOT/'score_optimization.py').read_bytes()).hexdigest(),
        references_accessed=False,training_performed=False)
    (ROOT/'SCORING_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
