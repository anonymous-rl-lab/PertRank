from pathlib import Path
import numpy as np
p=Path('code_v151_release/staging/early_intermediates/data/direction_control_states.npz');z=np.load(p,allow_pickle=False);out={k:z[k] for k in z.files if not k.endswith('_U')};names=['D3','D1','D2']
for d in names:out[d+'_U']=z[d+'_U'][:,:1]
for i,a in enumerate(names):
 for b in names[i+1:]:
  ga=z[a+'_genes'];gb=z[b+'_genes'];shared=sorted(set(ga)&set(gb));ia={g:i for i,g in enumerate(ga)};ib={g:i for i,g in enumerate(gb)};Xa=z[a+'_U'][[ia[g] for g in shared]];Xb=z[b+'_U'][[ib[g] for g in shared]];Na=Xa/np.linalg.norm(Xa,axis=0,keepdims=True);Nb=Xb/np.linalg.norm(Xb,axis=0,keepdims=True);Qa=np.linalg.qr(Xa)[0];Qb=np.linalg.qr(Xb)[0];key=a+'_'+b
  out[key+'_cos']=Na.T@Nb;out[key+'_QaTb']=Qa.T@Nb[:,:5];out[key+'_QbTa']=Qb.T@Na[:,:5];out[key+'_gram_a']=Na.T@Na;out[key+'_gram_b']=Nb.T@Nb
p.unlink();np.savez_compressed(p,**out);print(p.stat().st_size)
