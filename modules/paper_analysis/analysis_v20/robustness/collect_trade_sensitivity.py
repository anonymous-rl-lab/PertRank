"""Collect complete-cohort fit outputs, validate moments, and score once."""
from pathlib import Path
import os,sys,csv,json,gzip,hashlib,shutil
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'vendor'))
import stim_stats as st
RUN=Path(os.environ.get('PERT_RANK_STIM_ASSET_ROOT','/workspace/scratch/6cdbaefbed9f/stim_run_20261011'))
WORK=Path(os.environ.get('PERT_RANK_TRADE_WORK','/workspace/scratch/6cdbaefbed9f/paper_v20_20261011/work_robustness/trade_full'))
status=json.loads((WORK/'RUN_STATUS.json').read_text());assert status['status']=='complete' and status['completed_fits']==2805,status
p=np.load(RUN/'input/main_nontrade_predictions.npz');q=np.load(RUN/'input/main_query_context.npz');ref=np.load(RUN/'evaluation/main_reference.npz');reg=np.load(RUN/'evaluation/registered_scores.npz')
assert np.array_equal(p['target_ids'],ref['target_ids']) and np.array_equal(p['query_gene_ids'],ref['query_gene_ids'])
assert np.array_equal(p['target_ids'],q['target_ids']) and np.array_equal(p['query_gene_ids'],q['query_gene_ids'])
reference_identity=None
def digest(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as fh:
  for piece in iter(lambda:fh.read(8<<20),b''):h.update(piece)
 return h.hexdigest()
index={t:i for i,t in enumerate(p['target_ids'])};weights=np.full((5,561),np.nan);moments=[];mixtures={};check={'S':0.,'weight':0.,'nu':0.};batch_provenance=[]
for b in sorted(WORK.glob('targets_*')):
 complete=json.loads((b/'COMPLETE.json').read_text())
 for filename,sha in complete['outputs'].items():assert digest(b/filename)==sha,(b.name,filename)
 if reference_identity is None:reference_identity=complete['identity']
 assert complete['identity']==reference_identity,(b.name,'identity')
 assert complete['identity']['optimizer_control']=={'eps':1e-4}
 assert complete['identity']['sensitivity_source_sha256']==digest(HERE/'R_TRADE_univariate_eps.R')
 rows=list(csv.DictReader((b/'result_weights.tsv').open(),delimiter='\t'));mix=json.loads((b/'result_mixtures.json').read_text());inputs=np.load(b/'effects_input.npz')
 input_se=inputs['standard_errors']
 input_genes=inputs['genes']
 assert np.array_equal(inputs['replicates'],np.arange(5))
 for target in inputs['targets']:
  original=np.load(RUN/f'input/main/target_{index[str(target)]:04d}.npz')
  assert np.array_equal(original['gene_ids_keep'],input_genes)
 for i,row in enumerate(rows):
  assert np.isfinite(float(row['loglik'])),(b.name,row['target'],row['replicate'],'nonfinite loglik')
  j=index[row['target']];r=int(row['replicate']);key=row['target']+'::'+str(r);g=mix[key]['fitted_g'];pi,a,bb=[np.array(g[k],float) for k in ['pi','a','b']]
  assert np.all(pi>=0) and abs(pi.sum()-1)<1e-6
  s=float(np.sum(pi*(a*a+a*bb+bb*bb)/3));nu=float(np.mean(input_se[i//5,r]**2));w=s/(s+nu)
  for k,x in [('S',s),('nu',nu),('weight',w)]:
   err=abs(x-float(row[k]));check[k]=max(check[k],err);assert err<1e-10,(key,k,err)
  assert np.isnan(weights[r,j]);weights[r,j]=float(row['weight']);moments.append(row);mixtures[key]=mix[key]
 batch_provenance.append({'batch':b.name,'input_digest':complete['input_digest'],'input_tsv_sha256':complete['input_tsv_sha256'],'fits':complete['fits'],'seconds':complete['seconds'],'warnings_present':complete['warnings_present'],'outputs_sha256':complete['outputs']})
assert np.isfinite(weights).all() and len(moments)==2805
arms={name:p['weight_'+name] for name in ['raw','Euclidean','PR_M','PertRank']};arms['TRADE_scalar']=weights
scores,names=st.score_predictions(p['original_effects'],ref['effects'],arms,q['query_gene_ids'],q['target_ids'])
replayerr=max(float(np.max(abs(scores[list(names).index(a)]-reg['main_scores'][list(reg['main_arm_names']).index(a)]))) for a in ['raw','Euclidean','PR_M','PertRank']);assert replayerr<1e-14
result={'classification':'posthoc uniformly regularized numerical sensitivity','original_H2b_status':'not_evaluable; unchanged','uniform_optimizer_control':{'eps':1e-4},'fit_count':2805,'runtime':status,'moment_validation_maxabs':check,'unchanged_arm_score_replay_maxabs':replayerr,'panels':{}}
for panel,key in [('responsive','responsive_positions'),('terciles','tercile_positions')]:
 x=scores[:,:,q[key],:];means={a:float(x[i].mean()) for i,a in enumerate(names)};out={'means':means,'contrasts':{}}
 for a,b in [('PertRank','TRADE_scalar'),('TRADE_scalar','Euclidean'),('TRADE_scalar','raw'),('TRADE_scalar','PR_M'),('PertRank','Euclidean'),('PertRank','PR_M')]:
  d=(x[list(names).index(a)]-x[list(names).index(b)]).mean((0,2));out['contrasts'][a+'_minus_'+b]=st.paired_block_interval(d,q['blocks_'+panel])
 result['panels'][panel]=out
np.savez_compressed(HERE/'TRADE_SENSITIVITY_REPLAY_INPUTS.npz',effects=p['original_effects'],reference=ref['effects'],target_ids=q['target_ids'],query_gene_ids=q['query_gene_ids'],responsive_positions=q['responsive_positions'],tercile_positions=q['tercile_positions'],blocks_responsive=q['blocks_responsive'],blocks_terciles=q['blocks_terciles'],**{'weight_'+k:v for k,v in arms.items()})
np.savez_compressed(HERE/'TRADE_SENSITIVITY_SCORES.npz',scores=scores,arm_names=np.array(names))
with (HERE/'TRADE_SENSITIVITY_FIT_MOMENTS.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(moments[0]));writer.writeheader();writer.writerows(moments)
with gzip.open(HERE/'TRADE_SENSITIVITY_MIXTURES.json.gz','wt') as f:json.dump(mixtures,f,separators=(',',':'))
(HERE/'TRADE_BATCH_PROVENANCE.json').write_text(json.dumps(batch_provenance,indent=2)+'\n')
(HERE/'TRADE_SENSITIVITY_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
