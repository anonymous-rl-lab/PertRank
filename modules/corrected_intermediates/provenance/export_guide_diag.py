from pathlib import Path
import os,sys,numpy as np,json
R=Path('/workspace/scratch/9bb77e8faf2b');H=R/'code_v151_release/staging/corrected_intermediates/guide';work=R/'code_v151_release/qa/guide_export';D={}
root=work/'corrected';stage=root/'same_budget_weight/stage3';os.environ['PR10_ROOT']=str(root);os.environ['PR10_DATA']=str(root/'data');sys.path[:0]=[str(stage),str(stage.parent/'stage2'),str(stage.parent/'scripts'),str(root)]
import evaluate_a15_guide_split as E
for mode in ['legacy','corrected']:
 for donor in ['D3_Rest','D1_Rest','D2_Rest']:
  z=np.load(work/mode/'same_budget_weight/stage3/data'/f'a15_{donor}_guide_split.npz');pos=np.searchsorted(z['qu'],z['q_resp']);cm=z['cm_q'][None,:];ta=(z['Aq']/100-cm).T[pos];tb=(z['Bq']/100-cm).T[pos]
  D[f'{mode}_{donor}_correlation']=np.array([np.corrcoef(ta[i],tb[i])[0,1]for i in range(len(pos))])
  for tag in ['', 'B']:
   w0,w1,_=E.weights(z); # Correct assignment below preserves the tag-specific original formula.
   w0,w1,_=E.weights(z,tag)
   D[f'{mode}_{donor}_w0{tag}']=w0;D[f'{mode}_{donor}_w1{tag}']=w1
np.savez_compressed(H/'fixed_diagnostics.npz',**D)
