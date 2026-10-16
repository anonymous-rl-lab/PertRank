from pathlib import Path
import sys,os,runpy,json,hashlib
import numpy as np
root=Path(sys.argv[1]);data=Path(sys.argv[2]);script=sys.argv[3];out=Path(sys.argv[4]);capture=Path(sys.argv[5]);original=sys.argv[6];extra=sys.argv[7:]
os.environ['PR10_ROOT']=str(root);os.environ['PR10_DATA']=str(root/'data')
stage=root/'same_budget_weight/stage3';sys.path[:0]=[str(stage),str(stage.parent/'stage2'),str(stage.parent/'scripts'),str(root)]
import pr10, a6_common as A, sizing_depth_ladder as SZ
arrays={};ops=[];scores=[]
def put(x):
 x=np.asarray(x);k='a'+hashlib.sha256(x.dtype.str.encode()+str(x.shape).encode()+x.tobytes()).hexdigest()[:24]
 arrays[k]=x;return k
old=pr10.block_bootstrap
def block(pa,pb,blocks,*a,**kw):
 result=old(pa,pb,blocks,*a,**kw)
 ops.append({'kind':'block_bootstrap','arrays':[put(pa),put(pb),put(blocks)],'args':list(a),'kwargs':kw,'expected':result})
 return result
oldscore=pr10.score_rows
def score(*a,**kw):
 r=oldscore(*a,**kw);scores.append({'array':put(r['exp']),'expected_mean':float(r['exp'].mean())});return r
old_donor=SZ.donor_mean_bootstrap
def donor(diff,groups,labels,*a,**kw):
 r=old_donor(diff,groups,labels,*a,**kw)
 ops.append({'kind':'donor_mean_bootstrap','arrays':[put(diff),put(groups),put(labels)],'args':list(a),'kwargs':kw,'expected':r});return r
for m in list(sys.modules.values()):
 if getattr(m,'block_bootstrap',None)is old:setattr(m,'block_bootstrap',block)
 if getattr(m,'score_rows',None)is oldscore:setattr(m,'score_rows',score)
SZ.donor_mean_bootstrap=donor
sys.argv=[original,str(root),str(data),script,str(out)]+extra
runpy.run_path(original,run_name='__main__')
capture.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(capture.with_suffix('.npz'),**arrays)
capture.with_suffix('.json').write_text(json.dumps({'script':script,'operations':ops,'score_means':scores},indent=2)+'\n')
