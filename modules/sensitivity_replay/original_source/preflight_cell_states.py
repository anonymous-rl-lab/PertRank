import os, sys, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'restored/working_v9_with_v6_inputs/code/repro_v10'
os.environ['PR10_ROOT']=str(ROOT);os.environ['PR10_DATA']=str(ROOT/'data')
S3=ROOT/'same_budget_weight/stage3';sys.path.insert(0,str(S3))
import a6_common as A
import evaluate_stage2 as es
import evaluate_w1_e4 as ev
import evaluate_d2_prospective as d2

def load(donor):
 s=np.load(S3/'data'/f'b3_{donor}_query_cells.npz',allow_pickle=False)
 X=np.log1p(s['counts'].astype(float)*(float(s['scale'])/np.maximum(s['total_counts'],1))[:,None]);nc=int(s['n_controls']);nt=len(s['names'])
 if donor=='D3':
  z=es.load('e4_tcell.npz');w=ev.w1(np.load(S3/'data/e4_w1_stage3.npz'),5,.9)[0]
  cs={c['rule']:dict(c,W1=w,q=z['q_resp'] if c['rule']=='responsive' else z['q_terc']) for c in es.sources()['E4']['cells'] if c['depth']=='20'}
 else:
  z=np.load(S3/'data'/('d1_confirm_state.npz' if donor=='D1' else 'd2_prospective_state.npz'))
  allcs=A.d1_cells(z,'corrected') if donor=='D1' else d2.d2_cells(z)
  cs={rule:c for (rule,depth),c in allcs.items() if depth=='20'}
 assert np.array_equal(s['names'],z['names'])
 assert np.array_equal(s['query_genes'],z['keep'][z['qu']])
 yt=X[nc:].reshape(nt,20,-1).mean(1).T;yc=X[:nc].mean(0)
 for c in cs.values():
  c['rows']=np.searchsorted(z['qu'],c['q'])
  c['fixed_control_mean']=(z['cm_q'] if donor=='D3' else z['cm'][z['qu']])[c['rows']]
  c['count_data_state'] = str(S3/'data'/f'b3_{donor}_query_cells.npz')
 return s,X,cs,yt,yc

def check():
 out={}
 for d in ['D3','D1','D2']:
  s,X,cs,yt,yc=load(d);v={}
  for rule,c in cs.items():
   E=(yt-yc[:,None])[c['rows']]
   v[rule]={'max_abs_Y_diff':float(np.max(abs(E-c['Y']))),'scores':{}}
   for name,w in [('raw',np.ones(yt.shape[1])),('w0y',c['w0y']),('W1',c['W1'])]:
    orig=A.exp_scores(c['Y']*w,c).mean();new=A.exp_scores(E*w,c).mean()
    v[rule]['scores'][name]={'archived_state':float(orig),'count_reconstruction':float(new),'difference':float(new-orig)}
  out[d]=v
 print(json.dumps(out,indent=2))
 return out
if __name__=='__main__':
 out=check();Path(__file__).with_name('cell_input_preflight.json').write_text(json.dumps(out,indent=2))
