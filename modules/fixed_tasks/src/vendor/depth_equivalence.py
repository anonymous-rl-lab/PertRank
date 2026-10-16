"""v16b inverse definitions; actual maxclust labels accepted without forced reclustering.
Only the exact30 assertion/report field is adapted to the frozen D4 block rule.
No extrapolation or discarded boundary draws.
"""
import numpy as np
DEPTHS=np.arange(20,51,5)
SEED=2026
BOOT=2000

def crossing(curve,target):
 curve=np.asarray(curve,float); target=float(target)
 allcross=[];plateaus=[]
 for i in range(len(curve)-1):
  a,b=curve[i:i+2];lo,hi=map(float,DEPTHS[i:i+2])
  if a==b==target:plateaus.append([lo,hi]);allcross.append(lo)
  elif a!=b and min(a,b)<=target<=max(a,b):allcross.append(lo+(target-a)*(hi-lo)/(b-a))
 allcross=sorted(set(allcross))
 base={'target_score':target,'raw_scores':curve.tolist(),'nonmonotone_steps':np.flatnonzero(np.diff(curve)<0).tolist(),
  'all_crossings':allcross,'plateau_intervals':plateaus,'multiple_crossings':len(allcross)>1,
  'crossing_range':[allcross[0],allcross[-1]] if allcross else None}
 if curve[0]>=target:return {**base,'state':'left_boundary_raw20_already_attains','n_equivalent':None,'measured_bracket':None}
 for i in range(len(curve)-1):
  a,b=curve[i:i+2]
  if a<target<=b:
   n=float(DEPTHS[i]+(target-a)*5/(b-a))
   return {**base,'state':'identified','n_equivalent':n,'measured_bracket':DEPTHS[i:i+2].tolist()}
 return {**base,'state':'not_attained_in_tested_range','n_equivalent':None,'measured_bracket':None}

def code_cross(curve,target):
 if curve[0]>=target:return -np.inf,-1
 for i in range(6):
  if curve[i]<target<=curve[i+1]:return float(DEPTHS[i]+5*(target-curve[i])/(curve[i+1]-curve[i])),0
 return np.inf,1

def quant_extended(x,p):
 x=np.sort(np.asarray(x,float));a=(len(x)-1)*p;i=int(np.floor(a));j=int(np.ceil(a));t=a-i
 if i==j or x[i]==x[j]:v=x[i]
 elif not np.isfinite(x[i]):v=x[i]
 elif not np.isfinite(x[j]):v=x[j]
 else:v=float((1-t)*x[i]+t*x[j])
 if np.isneginf(v):return {'state':'left_or_unidentified_boundary','value':None}
 if np.isposinf(v):return {'state':'right_or_unidentified_boundary','value':None}
 return {'state':'finite','value':float(v)}

def weights_for_bootstrap(blocks):
 labels=np.unique(blocks);assert 1<=len(labels)<=30
 members={v:np.flatnonzero(blocks==v) for v in labels}
 rng=np.random.default_rng(SEED);w=np.empty((BOOT,len(blocks)))
 for b in range(BOOT):
  ix=np.concatenate([members[v] for v in rng.choice(labels,len(labels),replace=True)])
  w[b]=np.bincount(ix,minlength=len(blocks))/len(ix)
 return w

def summarize(raw,arms,blocks):
 # raw depth,anchor,query,sign; arms raw/euclidean/pertrank,anchor,query,sign
 rc=raw.mean(1).mean(-1);ac=arms.mean(1).mean(-1);meanraw=rc.mean(1)
 w=weights_for_bootstrap(blocks);bc=w@rc.T;ba=w@ac.T
 out={'raw_curve':meanraw.tolist(),'raw_curves_by_anchor':raw.mean((2,3)).T.tolist(),
  'method20_scores':dict(zip(('raw','Euclidean','PertRank'),ac.mean(1).tolist())),
  'nonmonotone_raw_steps':np.flatnonzero(np.diff(meanraw)<0).tolist(),'methods':{},
  'depths':DEPTHS.tolist(),'bootstrap_draws':BOOT,'bootstrap_seed':SEED,'query_blocks':int(len(np.unique(blocks))),
  'scope':'conditional on realized100-cell reservoir, fixed controls, queries, candidate pool and reference; nested/overlapping anchors are not independent experiments; no sequencing-cost claim'}
 nums=np.empty((BOOT,2));states=np.empty((BOOT,2),np.int8)
 for j,(name,mi) in enumerate((('Euclidean',1),('PertRank',2))):
  point=crossing(meanraw,ac[mi].mean())
  for b in range(BOOT):nums[b,j],states[b,j]=code_cross(bc[b],ba[b,mi])
  counts={name_:int(np.sum(states[:,j]==code)) for name_,code in [('left_boundary',-1),('identified',0),('not_attained',1)]}
  point.update({'bootstrap_state_counts':counts,'bootstrap_state_fractions':{k:v/BOOT for k,v in counts.items()},
    'conditional_inverse_95_endpoints':[quant_extended(nums[:,j],.025),quant_extended(nums[:,j],.975)]})
  if point['n_equivalent'] is not None:
   point['relative_extra_ranking_input_cells']=point['n_equivalent']/20-1
   point['arithmetic_reduction_relative_to_equivalent_depth']=1-20/point['n_equivalent']
  out['methods'][name]=point
 pairs=(states==0).all(1);ds=np.full(BOOT,np.nan);ds[pairs]=nums[pairs,1]-nums[pairs,0]
 lo=np.where(pairs,ds,-np.inf);hi=np.where(pairs,ds,np.inf)
 n0=out['methods']['Euclidean']['n_equivalent'];nm=out['methods']['PertRank']['n_equivalent']
 out['paired_metric_equivalent_depth_difference']={'point':None if n0 is None or nm is None else nm-n0,
  'identified_draws':int(pairs.sum()),'unidentified_draws':int((~pairs).sum()),
  'conservative_conditional_inverse_95_envelope':[quant_extended(lo,.025),quant_extended(hi,.975)],
  'definition':'paired finite difference when both crossings are identifiable; each other draw retained as interval[-infinity,+infinity], not discarded'}
 d=ac[2]-ac[1];db=w@d
 out['PertRank_minus_Euclidean_P25']={'point':float(d.mean()),'paired_query_block_95_interval':np.quantile(db,[.025,.975]).tolist()}
 d=ac[2]-ac[0];db=w@d
 out['PertRank_minus_raw20_P25']={'point':float(d.mean()),'paired_query_block_95_interval':np.quantile(db,[.025,.975]).tolist()}
 out['PertRank20_minus_raw_by_depth']=[{'depth':int(DEPTHS[i]),'point':float((ac[2]-rc[i]).mean()),
  'paired_query_block_95_interval':np.quantile(ba[:,2]-bc[:,i],[.025,.975]).tolist()} for i in range(7)]
 return out,{'raw_curve_bootstrap':bc,'method20_bootstrap':ba,'inverse_values':nums,'inverse_states':states,'paired_depth_difference':ds}
