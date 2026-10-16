#!/usr/bin/env python3
"""Fixed native ASH for the separate frozen 250-target nested-depth diagnostic."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,subprocess,sys,uuid,zipfile
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1];D=ROOT/'posthoc_depth_20261014'
PROTOCOL_SHA='bd529ba31eaf41cfff522716835d2f95aedc61bfd7a3fdae68e4140e7e6bb546'
def sha(p):
 with open(p,'rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def crc(p):
 with zipfile.ZipFile(p)as z:assert z.testzip()is None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--depth',type=int,choices=[20,50,100],required=True);a=ap.parse_args();n=a.depth
 assert sha(D/'POSTHOC_DEPTH_PROTOCOL.json')==PROTOCOL_SHA
 protocol=json.loads((D/'POSTHOC_DEPTH_PROTOCOL.json').read_text());source=D/'inputs'/f'n{n}';audit=json.loads((source/'INPUT_AUDIT.json').read_text())
 assert audit['status']=='FROZEN_POSTHOC_NESTED_DEPTH_INPUT_PASS' and audit['depth']==n and audit['targets']==250 and audit['protocol_sha256']==PROTOCOL_SHA
 assert audit['selection_sha256']==sha(D/'selection.npz')==protocol['files_sha256']['selection.npz']
 assert all(audit[k]for k in ['same_original_2000_NTC','raw_totals_exact','full_gene_axis_exact','same_query_panel','same_archived_reference_subset','weights_recomputed_with_this_n'])
 for name,want in audit['files_sha256'].items():assert sha(source/name)==want;crc(source/name)
 uid=uuid.uuid4().hex[:12];inpdir=P/'depth_inputs'/f'n{n}';inpstage=inpdir.with_name(inpdir.name+'.__staging_'+uid);assert not inpdir.exists();inpstage.mkdir(parents=True)
 with(ROOT/'heavy_memory.lock').open('a')as memory:
  fcntl.flock(memory,fcntl.LOCK_EX);s=load(source/'full_effect_stats.npz');axes=load(source/'axes.npz');selection=load(D/'selection.npz')
  assert int(s['n_control'])==2000 and int(s['n_target'])==n and int(s['variance_ddof'])==1
  assert np.array_equal(s['target_ids'],selection['target_ids']) and np.array_equal(s['target_ids'],axes['target_ids'])
  assert s['effects'].shape==s['standard_errors'].shape==(250,13390) and np.isfinite(s['effects']).all() and np.isfinite(s['standard_errors']).all() and(s['standard_errors']>0).all()
  qi=s['query_keep_columns'];assert np.array_equal(s['gene_ids'][qi],axes['query_ids']) and len(axes['query_ids'])==300 and len(np.unique(axes['query_blocks']))==30
  assert np.array_equal(s['effects'][:,qi],axes['input_effects']) and int(axes['target_cells_per_group'])==n
  assert np.allclose(s['standard_errors']**2,s['target_var']/n+s['control_var']/2000,rtol=1e-14,atol=1e-16)
  with(inpstage/'full.npz').open('wb')as f:np.savez_compressed(f,effects=s['effects'],ses=s['standard_errors'],row_ids=s['target_ids'],genes=s['gene_ids']);f.flush();os.fsync(f.fileno())
  crc(inpstage/'full.npz');validation={'status':'FROZEN_POSTHOC_DEPTH_ASH_INPUT_PASS','depth':n,'source_input_directory':str(source),'source_input_audit_sha256':sha(source/'INPUT_AUDIT.json'),'source_fullstats_sha256':sha(source/'full_effect_stats.npz'),'full_npz_sha256':sha(inpstage/'full.npz'),'protocol_sha256':PROTOCOL_SHA,'target_count':250,'gene_count':13390,'query_count':300,'n_control':2000,'n_target':n,'fixed_configuration':'default_uniform_fdr','no_ASH_weighted_variants':True,'main_benchmark':False}
  (inpstage/'INPUT_VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n');inpstage.rename(inpdir)
 del s,axes,selection
 dest=P/'depth_transfer'/f'n{n}';stage=dest.with_name(dest.name+'.__staging_'+uid);assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True)
 lock=dict(status='FROZEN_DEPTH_ASH_SINGLE_TASK_EXECUTION',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),depth=n,configuration='default_uniform_fdr',protocol_sha256=PROTOCOL_SHA,input_sha256=validation['full_npz_sha256'],source_code_sha256=sha(__file__),native_runner_sha256=sha(ROOT/'ash_work/posterior_runner.R'),sharded_runner_sha256=sha(ROOT/'ash_work/run_sharded.py'),R_streams=2,R_threads=1,main_benchmark=False,staging=str(stage),destination=str(dest))
 lp=dest.with_name(dest.name+'_EXECUTION_LOCK.json');assert not lp.exists();lp.write_text(json.dumps(lock,indent=2)+'\n')
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
 with dest.with_name(dest.name+'_stdout.log').open('w')as f:r=subprocess.run([sys.executable,str(ROOT/'ash_work/run_sharded.py'),'--input',str(inpdir/'full.npz'),'--output',str(stage),'--shards','2'],env=env,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0
 ex=json.loads((stage/'execution.json').read_text());rt=json.loads((stage/'runtime.json').read_text());assert ex['returncode']==0 and rt['status']=='COMPLETE' and rt['failed_rows']==0 and ex['posterior_npz_sha256']==sha(stage/'posterior.npz');crc(stage/'posterior.npz')
 for file in stage.rglob('*'):
  if file.is_file():
   with file.open('rb')as f:os.fsync(f.fileno())
 (stage/'PUBLICATION.json').write_text(json.dumps({'status':'STAGED_COMPLETE_ATOMIC_RENAME','stage':str(stage),'destination':str(dest),'lock_sha256':sha(lp)},indent=2)+'\n');stage.rename(dest)
 print(json.dumps({'status':'COMPLETE_ONE_POSTHOC_DEPTH_TASK','depth':n,'directory':str(dest),'posterior_sha256':ex['posterior_npz_sha256'],'main_benchmark':False}),flush=True)
if __name__=='__main__':main()
