from pathlib import Path
import os,sys,json,runpy,numpy as np,csv,shutil
base=Path('code_v150_release/staging/early_experiments/repro_v10').resolve();b=base/'same_budget_weight';s=b/'stage3';out=Path('code_v151_release/staging/early_intermediates');dest=out/'data';os.environ.update(PR10_ROOT=str(base),PR10_DATA=str(base/'data'),OPENBLAS_NUM_THREADS='1')
for p in [base,b,b/'scripts',b/'stage2',s]:sys.path.insert(0,str(p))
work=Path('code_v151_release/qa/early_extra_export').resolve();work.mkdir(exist_ok=True);sys.argv=['stage1_reliability.py','--out',str(work)]
g=runpy.run_path(str(b/'scripts/stage1_reliability.py'));arrays={k:g[k] for k in ['s20','s30','sA','sB']};D=np.load(base/'data/e4_tcell.npz');arrays['vb20']=D['vb_20']
D=np.load(base/'data/e5_kolf.npz');arrays.update({f'kolf_s{n}':D[f'm2_mfull_{n}']-(D[f'vb_full_{n}']/n+D['cvm_mfull'][D['target_batch']]/2000) for n in [20,30]})
D=np.load(base/'data/e1_stage6.npz');from pr10 import nu
arrays.update({f'e1_s{n}':D[f'm2_{n}']-nu(D[f'vb{n}'],n,float(D['cvm']),int(D['ncin'])) for n in [20,30]});np.savez_compressed(dest/'reliability_target_energy.npz',**arrays)
import direction1_q1_similarity as Q
arr={}
for d,(genes,U,cm,lam,tr) in Q.donors().items():
 arr.update({d+'_genes':genes,d+'_U':U,d+'_cm':cm,d+'_lam':lam,d+'_trace':np.array(tr)})
np.savez_compressed(dest/'direction_control_states.npz',**arr)
shutil.copy2(s/'data/tcell_gene_symbols.json',dest/'tcell_gene_symbols.json')
shutil.copy2('code_v151_release/qa/early_exports/direction1_q2_cases/analysis_output/direction1_cases_D1_genes.csv',dest/'direction1_cases_D1_genes.csv')
rel=json.loads((s/'results/b7_relation_table.json').read_text())['relations'];small=[r for r in rel if r['source']=='D1' and r['target'] in ('TP53','RFX5') and r['kind']=='w0y-specific'];(dest/'direction1_case_relations.json').write_text(json.dumps(small,separators=(',',':')))
print([(p.name,p.stat().st_size) for p in dest.iterdir() if p.name in ['reliability_target_energy.npz','direction_control_states.npz','direction1_case_relations.json']])
