from pathlib import Path
import sys,argparse,json,time
import numpy as np
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'guide'))
from bootstrap import block_bootstrap

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();out=a.output_dir.resolve();assert not out.exists();out.mkdir(parents=True);start=time.time()
 expected=json.loads((H/'A18_R_strict_replication.json').read_text());result={};maxerr=0.
 for donor in ['D3_Rest','D1_Rest']:
  z=np.load(H/f'{donor}_A18_R_query_values.npz');pm=z['pm'];h=z['p_B']*z['p_D2'];counts=(pm*h[None,...]).sum(-1)
  np.testing.assert_array_equal(counts,z['counts']);r={'strict_counts':{arm:float(counts[i].mean())for i,arm in enumerate(['raw','w0y','W1'])},'primary':{},'secondary_z':{'strict_counts':{arm:float(z['secondary_counts'][i].mean())for i,arm in enumerate(['raw','w0y','W1'])}},'turnover':{}}
  for bi,label in [(0,'W1_vs_raw'),(1,'W1_vs_w0y')]:
   r['primary'][label]=block_bootstrap(counts[2],counts[bi],z['blocks'],B=2000,seed=2026)
   r['secondary_z'][label]=block_bootstrap(z['secondary_counts'][2],z['secondary_counts'][bi],z['blocks'],B=2000,seed=2026)
   add=np.maximum(pm[2]-pm[bi],0);rem=np.maximum(pm[bi]-pm[2],0);sa=(add*h).sum(-1);sr=(rem*h).sum(-1)
   r['turnover'][label]={'mean_changed_slots':float(add.sum(-1).mean()),'added_support_rate':float(sa.sum()/add.sum()),'removed_support_rate':float(sr.sum()/rem.sum()),'gain_identity_max_abs_error':float(np.max(np.abs(sa-sr-(counts[2]-counts[bi]))))}
  def check(x,y):
   nonlocal maxerr
   if isinstance(x,dict):
    for k,v in x.items():check(v,y[k])
   else:
    e=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));maxerr=max(maxerr,e);assert e<=1e-12
  check(r,expected['donors'][donor]);result[donor]=r
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');report={'status':'PASS','tier':'stored_membership_to_strict_support_and_paired_intervals','max_abs_error':maxerr,'seconds':time.time()-start,'not_recomputed':['effects','weights','rankings','original missing A18 script'],'identity':'A18-R reconstruction'}
 (out/'REPRODUCTION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
