from pathlib import Path
import argparse,json,time
import numpy as np
H=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();out=a.output_dir.resolve();assert not out.exists();out.mkdir(parents=True);start=time.time()
 z=np.load(H/'CELLEQ_R_query_scores.npz');want=json.loads((H/'CELLEQ_R_results.json').read_text());result={};errors=[]
 for d in ['D3','D1','D2']:
  depths=[20,30,50]+([100]if d=='D3'else[]);m={str(n):float(z[f'{d}_raw{n}'].mean())for n in depths};w=float(z[f'{d}_PertRank20'].mean());brackets=[(lo,hi,lo+(w-m[str(lo)])/(m[str(hi)]-m[str(lo)])*(hi-lo))for lo,hi in zip(depths[:-1],depths[1:])if m[str(lo)]<=w<=m[str(hi)]and m[str(hi)]>m[str(lo)]]
  r={'raw_scores':m,'PertRank20':w,'equivalent_depth':brackets[0][2]};r.update(extra_cells=r['equivalent_depth']-20,extra_fraction=(r['equivalent_depth']-20)/20,saving_fraction=(r['equivalent_depth']-20)/r['equivalent_depth']);result[d]=r
  for k,v in r.items():
   if isinstance(v,dict):errors.extend(abs(x-want['donors'][d][k][n])for n,x in v.items())
   else:errors.append(abs(v-want['donors'][d][k]))
 assert max(errors)<=1e-12
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');report={'status':'PASS','tier':'stored_query_depth_scores_to_means_and_inverse_interpolation','identity':'Historical CELLEQ-R; distinct from current seven-depth calibration','max_abs_error':max(errors),'seconds':time.time()-start}
 (out/'REPRODUCTION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
