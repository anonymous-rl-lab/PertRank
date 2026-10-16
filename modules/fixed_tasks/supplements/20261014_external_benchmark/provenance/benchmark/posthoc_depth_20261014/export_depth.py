#!/usr/bin/env python3
"""Export frozen250-target nested depths with original D3 control geometry."""
from pathlib import Path
import argparse,fcntl,json,time,sys,hashlib,os,zipfile,uuid
import numpy as np
from scipy.sparse import vstack,save_npz,load_npz
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'pooled_data'));from export_inputs import role,load,sha

def atomic_npz(path, writer, *args, **kwargs):
 path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
 with tmp.open('wb')as f:
  writer(f,*args,**kwargs);f.flush();os.fsync(f.fileno())
 with zipfile.ZipFile(tmp)as z:assert z.testzip()is None
 os.replace(tmp,path)
def sparse_npz(path,x):return atomic_npz(path,save_npz,x)
def dense_npz(path,**kwargs):return atomic_npz(path,np.savez_compressed,**kwargs)

def validate_bindings(depth):
 protocol=json.loads((HERE/'POSTHOC_DEPTH_PROTOCOL.json').read_text());lock=json.loads((HERE/'selection.lock.json').read_text())
 assert sha(HERE/'selection.npz')==lock['selection_sha256']==protocol['files_sha256']['selection.npz']
 assert sha(HERE/'selection.lock.json')==protocol['files_sha256']['selection.lock.json']
 audit=json.loads((ROOT/'recovery_20261014/data/D3_DEPTH_METADATA_AUDIT.json').read_text())
 for name,h in audit['provenance'].items():assert sha(ROOT/name)==h,name
 s=load(HERE/'selection.npz');original=load(ROOT/'pooled_data/D3_Rest/selection.npz');c=load(ROOT/'pooled_data/D3_Rest/r20_context.npz');ix=s['target_indices']
 olddir=ROOT/'archives/R20_Frozen_Design_20261006/research_round_20261006/new_experiment';old=load(olddir/'selection/D3_Rest/selection.npz');rep=json.loads((olddir/'selection/D3_Rest/selection_report.json').read_text())
 assert sha(olddir/'selection/D3_Rest/selection.npz')==rep['selection_sha256']==lock['parent_original_selection_sha256']
 assert sha(olddir/'reader/make_selection.py')==rep['script_sha256']
 assert rep['planning_overlap']==0 and rep['evaluation_overlap']==0
 assert np.array_equal(s['NTC_input'],original['NTC_input']) and len(s['NTC_input'])==2000
 assert np.array_equal(s['target_ids'],c['target_ids'][ix])
 assert np.array_equal(s['gene_ids_full'],c['gene_ids_full']) and np.array_equal(s['gene_ids_full'],original['gene_ids_full'])
 assert np.array_equal(s['input_cells'],original['input_cells'][ix].reshape(250,100))
 assert np.array_equal(s['input_cells'],old['rows'][old['internal_positions'][ix]].reshape(250,100))
 assert np.array_equal(s['input_cells'][:,:20],original['input_cells'][ix,0])
 assert len(np.unique(s['input_cells']))==25000 and not np.intersect1d(s['input_cells'],s['NTC_input']).size
 if depth!=20:
  first=HERE/'inputs/n20';a=json.loads((first/'INPUT_AUDIT.json').read_text())
  assert a['status']=='FROZEN_POSTHOC_NESTED_DEPTH_INPUT_PASS' and a['depth']==20
  assert a['selection_sha256']==lock['selection_sha256'] and a['protocol_sha256']==sha(HERE/'POSTHOC_DEPTH_PROTOCOL.json')
  assert max(a['n20_frozen_replay_errors'].values())<1e-10
  for name,h in a['files_sha256'].items():assert sha(first/name)==h,name
 return {'frozen_protocol_and_selection_hashes_verified':True,'original_NTC_target_gene_axes_exact':True,'depth100_cells_equal_exact_original_R20_five_groups':True,'n20_equal_original_rep0':True,'planning_reference_disjointness_evidence':'Strict subset inheritance from hash-verified original selection, archived zero-overlap report and archived selector source. Missing planning/reference row IDs were not independently re-compared this run.','original_report_planning_overlap':rep['planning_overlap'],'original_report_reference_overlap':rep['evaluation_overlap'],'n20_audit_precedes_deeper_export':depth!=20}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--depth',type=int,choices=[20,50,100],required=True);a=ap.parse_args();n=a.depth;t=time.monotonic()
 bindings=validate_bindings(n)
 with (ROOT/'heavy_memory.lock').open('a')as lock:
  print(json.dumps({'event':'WAIT_DEPTH_EXPORT_HEAVY_LOCK','depth':n}),flush=True);fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
  s=load(HERE/'selection.npz');c=load(ROOT/'pooled_data/D3_Rest/r20_context.npz');f=load(ROOT/'pooled_data/D3_Rest/frozen_task.npz');ix=s['target_indices'];ntc=s['NTC_input'];tr=s['input_cells'][:,:n];nt=len(tr);g=len(c['keep'])
  if n==20:
   # The nested first20 are exactly the already archived D3rep0 cells.
   # Reuse that stable, hash-verified container while deeper raw recovery runs.
   src=ROOT/'inputs/D3_Rest/rep0';sa=json.loads((src/'INPUT_AUDIT.json').read_text())
   assert sa['status']=='EXACT_RAW_R20_REPLAY_PASS'
   for name in ['counts.npz','axes.npz']:assert sha(src/name)==sa['files_sha256'][name]
   ax=load(src/'axes.npz');assert np.array_equal(ax['gene_ids'],s['gene_ids_full'])
   want=np.r_[ntc,tr.ravel()];order=np.argsort(ax['cell_ids']);source_pos=order[np.searchsorted(ax['cell_ids'][order],want)];assert np.array_equal(ax['cell_ids'][source_pos],want)
   xx=load_npz(src/'counts.npz');subset=xx[source_pos];del xx;cn=subset[:len(ntc)];tx=subset[len(ntc):];del subset
   bindings['n20_raw_source']='Stable archived original D3rep0 counts, exact frozen SHA and selected cell IDs verified'
   bindings['n20_raw_source_sha256']=sa['files_sha256']['counts.npz']
  else:
   assert (HERE/'RAW_OWNER_COMPLETE.json').exists(),'Wait for raw extraction owner exec to finish before consuming new batches'
   cn=role(HERE,'NTC_input',ntc);tx100=role(HERE,'A_input',s['input_cells'].ravel());take=(np.arange(nt)[:,None]*100+np.arange(n)).ravel();tx=tx100[take];del tx100
  raw=vstack([cn,tx],format='csr');ids=np.r_[ntc,tr.ravel()];tot=np.asarray(raw.sum(1)).ravel();sp=np.searchsorted(s['all_selected_rows'],ids);assert np.array_equal(s['all_selected_rows'][sp],ids);assert np.array_equal(s['selected_total_counts'][sp],tot)
  assert np.array_equal(np.flatnonzero(np.asarray(cn.sum(0)).ravel()>0),c['keep'])
  cc=np.log1p(cn[:,c['keep']].toarray().astype(float)*(float(c['scale'])/tot[:len(ntc)])[:,None]);cv=cc.var(0,ddof=1);cm=cc.mean(0);del cc
  rel=float(np.linalg.norm(cm-c['cm_keep'])/np.linalg.norm(c['cm_keep']));assert rel<1e-3
  effects=np.empty((nt,g));targetvar=np.empty((nt,g));se=np.empty((nt,g));m2=np.empty(nt);vb=np.empty(nt);pm=np.empty((nt,5));pv=np.empty((nt,5))
  for j in range(nt):
   x=np.log1p(tx[j*n:(j+1)*n,c['keep']].toarray().astype(float)*(float(c['scale'])/tot[len(ntc)+j*n:len(ntc)+(j+1)*n])[:,None]);y=x.mean(0)-c['cm_keep'];v=x.var(0,ddof=1);ps=x@c['U'][:,:5]
   effects[j]=y;targetvar[j]=v;se[j]=np.sqrt(v/n+cv/2000);m2[j]=np.mean(y*y);vb[j]=v.mean();pm[j]=ps.mean(0)-c['proj_control'][:5];pv[j]=ps.var(0,ddof=1)
  def weight(beta):
   energy=m2-beta*np.sum(pm*pm,axis=-1)/g;noise=vb/n+float(c['trC'])/(g*2000)-beta*np.sum(pv/n+c['lam'][:5]/2000,axis=-1)/g
   return np.clip(1-noise/np.maximum(energy,1e-300),0,1)
  w0=weight(0);wm=weight(.9);y=effects[:,c['qu']];errors={}
  if n==20:
   errors={'input_effects':float(np.max(np.abs(y-f['original_effects'][0,ix]))),'weights_euclidean':float(np.max(np.abs(w0-f['weight_Euclidean'][0,ix]))),'weights_pertrank':float(np.max(np.abs(wm-f['weight_PertRank'][0,ix])))};assert max(errors.values())<1e-10,errors
  out=HERE/'inputs'/f'n{n}';out.mkdir(parents=True,exist_ok=True);assert not (out/'INPUT_AUDIT.json').exists();sparse_npz(out/'counts.npz',raw);pos=c['responsive_positions'];columns=c['keep'][c['qu'][pos]]
  dense_npz(out/'axes.npz',labels=np.r_[np.full(len(ntc),-1,np.int32),np.repeat(np.arange(nt,dtype=np.int32),n)],gene_ids=s['gene_ids_full'],target_ids=s['target_ids'],target_names=s['target_names'],query_ids=c['query_gene_ids'][pos],query_blocks=c['blocks_responsive'],query_columns=columns,input_effects=y[:,pos],weights_euclidean=w0,weights_pertrank=wm,cell_ids=ids,reference_effects=c['truth'].T[ix][:,pos],rep=np.array(0),task=np.array(f'D3_DEPTH250_n{n}'),scale=c['scale'],input_control_count=np.array(2000),target_cells_per_group=np.array(n),all_query_ids=c['query_gene_ids'],all_query_columns=c['keep'][c['qu']],all_input_effects=y,all_reference_effects=c['truth'].T[ix],responsive_positions=pos,tercile_positions=c['tercile_positions'],blocks_terciles=c['blocks_terciles'],original_target_indices=ix)
  dense_npz(out/'full_effect_stats.npz',effects=effects,standard_errors=se,target_var=targetvar,gene_ids=c['gene_ids_keep'],target_ids=s['target_ids'],query_keep_columns=c['qu'][pos],control_mean=c['cm_keep'],control_mean_raw=cm,control_var=cv,scale=c['scale'],n_control=np.array(2000),n_target=np.array(n),variance_ddof=np.array(1))
  report={'status':'FROZEN_POSTHOC_NESTED_DEPTH_INPUT_PASS','depth':n,'targets':nt,'cells':len(ids),'nnz':int(raw.nnz),'same_original_2000_NTC':True,'raw_totals_exact':True,'full_gene_axis_exact':True,'same_query_panel':True,'same_archived_reference_subset':True,'control_mean_relative_historical_error':rel,'n20_frozen_replay_errors':errors,'weights_recomputed_with_this_n':True,'weight_geometry':'original historical D3 fixed controls, scale, trace, first5 axes, eigenvalues','selection_sha256':sha(HERE/'selection.npz'),'protocol_sha256':sha(HERE/'POSTHOC_DEPTH_PROTOCOL.json'),'files_sha256':{p.name:sha(p)for p in out.glob('*.npz')},'seconds':time.monotonic()-t,'main_benchmark':False}
  report['metadata_binding_checks']=bindings
  (out/'INPUT_AUDIT.PREPARED.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
