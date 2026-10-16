#!/usr/bin/env python3
"""Prepare real covariates for the locked D3 input; no fitting or references."""
from pathlib import Path
import argparse,json,hashlib,fcntl,zipfile
import numpy as np
from scipy.sparse import load_npz
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--metadata',type=Path,required=True);ap.add_argument('--metadata-audit',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 ma=json.loads(a.metadata_audit.read_text());assert ma['status']=='D3_GP_METADATA_READY_NO_NEW_EXPRESSION' and ma['X_expression_read'] is False and ma['metadata_npz_sha256']==sha(a.metadata)
 inp=ROOT/'inputs/D3_Rest/rep0';fixed={'counts.npz':'d349da549202ba1e4618b684b6d940aaf3eedfa3be18b97ba5472bece830c0d7','axes.npz':'752d9f232b7c4a2ad2dd53f4fcbe12d03f0efb5c3ee40e1cc0a0102096aabb59'}
 for name,h in fixed.items():
  assert sha(inp/name)==h
  with zipfile.ZipFile(inp/name) as z:assert z.testzip() is None
 with (ROOT/'heavy_memory.lock').open('a') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
  with np.load(a.metadata,allow_pickle=False) as z:m={k:z[k] for k in z.files}
  with np.load(inp/'axes.npz',allow_pickle=False) as z:cell_ids=z['cell_ids'];genes=z['gene_ids']
  assert np.array_equal(cell_ids,m['cell_ids']) and np.array_equal(genes,m['gene_ids'])
  cats=m['lane_categories'].astype(str);codes=m['lane_codes'];symbols=m['gene_symbols'].astype(str)
  assert len(cats)==24 and len(set(cats))==24 and np.all((codes>=0)&(codes<24))
  mt=np.char.startswith(np.char.upper(symbols),'MT-');assert np.array_equal(mt,m['mito_flag']) and mt.any()
  x=load_npz(inp/'counts.npz').tocsr();total=np.asarray(x.sum(1)).ravel();nfeat=np.asarray((x>0).sum(1)).ravel();mito=np.asarray(x[:,mt].sum(1)).ravel()/total
  values=np.column_stack([np.log(total),nfeat/total,mito,np.eye(24)[codes,1:]]).astype(np.float32)
  assert values.shape==(26000,26) and np.isfinite(values).all()
  assert not a.out.exists();a.out.mkdir(parents=True)
  cv=a.out/'covariates.npz';np.savez_compressed(cv,values=values,cell_ids=cell_ids,covariate_names=np.array(['log_total_umi','detected_features_per_umi','mitochondrial_fraction']+['lane_'+v for v in cats[1:]]),lane_categories=cats,dropped_lane=np.array(cats[0]),mitochondrial_gene_ids=genes[mt])
  files={name:{'path':str((inp/name).resolve()),'sha256':h} for name,h in fixed.items()};files['covariates.npz']={'path':str(cv.resolve()),'sha256':sha(cv)}
  report={'status':'FROZEN_REAL_D3_RESOURCE_PILOT_INPUTS','dataset':'D3_Rest','rep':0,'files':files,'metadata_file':{'path':str(a.metadata.resolve()),'sha256':sha(a.metadata)},'metadata_audit':{'path':str(a.metadata_audit.resolve()),'sha256':sha(a.metadata_audit)},'covariates_metadata_verified':True,'reference_cells_used':False,'new_expression_fetched':False,'mitochondrial_definition':'upper-case gene symbol starts MT-; UMI fraction, not percent','control_and_target_ids':'Exact original26000-cell order','source_categorical_lanes':24,'technical_covariates':26,'preparation_code_sha256':sha(__file__)}
  (a.out/'REAL_D3_INPUT_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'manifest':str(a.out/'REAL_D3_INPUT_MANIFEST.json'),'mitochondrial_genes':int(mt.sum())}))
if __name__=='__main__':main()
