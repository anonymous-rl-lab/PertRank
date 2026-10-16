"""Metadata-only selection for real disjoint-cell repetitions, 2026-10-06.
No raw X/data values, new weights, or scores are read/computed.
Paths are command-line inputs; all outputs are confined to --out.
"""
from pathlib import Path
from collections import OrderedDict
import argparse,hashlib,json,os,re,time,ctypes,struct
import numpy as np
import h5py
import requests

BASE='https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/{}.assigned_guide.h5ad'
DONORS=('D3_Rest','D1_Rest','D2_Rest')
EXPECTED_N={'D3_Rest':1200,'D1_Rest':1127,'D2_Rest':1200}
EXPECTED_EXT={'D3_Rest':52,'D1_Rest':59,'D2_Rest':63}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def strings(a):return np.array([x.decode() if isinstance(x,bytes)else str(x)for x in a])
def hx(s):return hashlib.sha256(s.encode()).hexdigest()
def atomic_json(p,x):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()));os.replace(t,p)
def atomic_npz(p,**x):
 p=Path(p);t=p.with_suffix('.tmp.npz');np.savez_compressed(t,**x);os.replace(t,p)

class MetaRange:
 """Strict, cached HTTP ranges, never interprets target expression."""
 def __init__(self,url,cache,block=1<<20):
  self.url=url;self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True);self.bs=block;self.pos=0;self.s=requests.Session();self.calls=0;self.bytes=0;self.mem=OrderedDict()
  r=self.s.get(url,headers={'Range':'bytes=0-0'},timeout=45);r.raise_for_status();m=re.fullmatch(r'bytes 0-0/(\d+)',r.headers.get('Content-Range',''));assert r.status_code==206 and m and len(r.content)==1
  self.size=int(m[1]);self.etag=r.headers.get('ETag');ident={'url':url,'size':self.size,'etag':self.etag};r.close()
  ip=self.cache/'SOURCE.json'
  if ip.exists():assert json.loads(ip.read_text())==ident,'metadata source changed'
  else:atomic_json(ip,ident)
 def _block(self,k):
  if k in self.mem:self.mem.move_to_end(k);return self.mem[k]
  a=k*self.bs;n=min(self.bs,self.size-a);p=self.cache/f'{a}_{n}.bin'
  if p.exists():b=p.read_bytes();assert len(b)==n
  else:
   for attempt in range(5):
    try:
     r=self.s.get(self.url,headers={'Range':f'bytes={a}-{a+n-1}'},timeout=60)
     if r.status_code in (429,503):
      delay=max(float(r.headers.get('Retry-After','0') or 0),2**(attempt+1));r.close();time.sleep(delay);continue
     assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {a}-{a+n-1}/{self.size}',(r.status_code,r.headers.get('Content-Range'))
     assert r.headers.get('ETag')==self.etag,'source changed';b=r.content;r.close();assert len(b)==n
     t=p.with_suffix('.tmp');t.write_bytes(b);os.replace(t,p);self.calls+=1;self.bytes+=n;break
    except Exception:
     if attempt==4:raise
     time.sleep(2**attempt)
   else:raise IOError('range fetch exhausted')
  self.mem[k]=b
  if len(self.mem)>32:self.mem.popitem(last=False)
  return b
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  n=min(n,self.size-self.pos);out=bytearray()
  while n>0:
   k,o=divmod(self.pos,self.bs);b=self._block(k);take=min(n,len(b)-o);assert take>0;out+=b[o:o+take];self.pos+=take;n-=take
  return bytes(out)
 def readinto(self,b):x=self.read(len(b));b[:len(x)]=x;return len(x)
 def seek(self,o,w=0):self.pos={0:o,1:self.pos+o,2:self.size+o}[w];return self.pos
 def tell(self):return self.pos
 def readable(self):return True
 def seekable(self):return True
 def writable(self):return False
 def flush(self):pass
 def close(self):self.s.close()
 @property
 def closed(self):return False

def cat(o,k):
 x=o[k]
 if isinstance(x,h5py.Group):return strings(x['categories'][:]),x['codes'][:].astype(np.int64)
 vals=strings(x[:]);c,i=np.unique(vals,return_inverse=True);return c,i.astype(np.int64)
def object_address(x):
 # H5O_INFO_BASIC only. h5py.h5o.get_info() defaults can traverse the entire
 # chunk metadata index, which is prohibitively expensive on remote CSR files.
 lib=ctypes.CDLL(h5py.h5.__file__);fn=lib.H5Oget_info2;fn.argtypes=[ctypes.c_longlong,ctypes.c_void_p,ctypes.c_uint];fn.restype=ctypes.c_int;b=ctypes.create_string_buffer(1024);assert fn(x.id.id,b,1)==0;return struct.unpack_from('<Q',b.raw,8)[0]

def metadata(url,folder):
 mp=folder/'obs_metadata.npz';jp=folder/'metadata_report.json'
 if mp.exists() and jp.exists() and (folder/'hdf_objects.json').exists():return np.load(mp,allow_pickle=False),json.loads(jp.read_text())
 t=time.time();rf=MetaRange(url,folder/'metadata_range_cache');o={}
 with h5py.File(rf,'r')as f:
  obs=f['obs'];fields=list(obs.keys());print('OBS',fields,flush=True)
  for name,key in [('gene','perturbed_gene_name'),('id','perturbed_gene_id'),('type','guide_type'),('guide','guide_id')]:
   o[name+'_categories'],o[name+'_codes']=cat(obs,key);print('METADATA',key,len(o[name+'_codes']),round(time.time()-t,1),flush=True)
  lane_keys=[k for k in fields if k.lower()in['lane','lane_id','sequencing_lane','lane_name']]
  if not lane_keys:raise RuntimeError(f'No unambiguous lane field: {fields}')
  assert len(lane_keys)==1,lane_keys;lane=lane_keys[0];o['lane_categories'],o['lane_codes']=cat(obs,lane)
  o['low_quality']=obs['low_quality'][:];print('METADATA low_quality',round(time.time()-t,1),flush=True);o['total_counts']=obs['total_counts'][:].astype(float);print('METADATA total_counts',round(time.time()-t,1),flush=True);o['gene_ids_full']=strings(f['var/_index'][:]);specs={}
  for key in ['data','indices','indptr']:
   z=f['X/'+key];print('METADATA X-layout',key,round(time.time()-t,1),flush=True);specs[key]={'object_address':object_address(z),'dtype':z.dtype.str,'length':len(z),'itemsize':z.dtype.itemsize,'chunk_size':z.chunks[0],'compression':z.compression,'shuffle':z.shuffle,'fletcher32':z.fletcher32,'scaleoffset':z.scaleoffset};assert z.compression is None and not z.shuffle and not z.fletcher32 and z.scaleoffset is None
  shape=list(map(int,f['X'].attrs['shape']))
 report={'source_url':url,'source_size':rf.size,'etag':rf.etag,'shape':shape,'lane_field':lane,'lane_categories':o['lane_categories'].tolist(),'metadata_network_calls':rf.calls,'metadata_network_bytes':rf.bytes,'seconds':time.time()-t,'raw_X_values_read':False}
 atomic_npz(mp,**o);atomic_json(jp,report);atomic_json(folder/'hdf_objects.json',specs);rf.close();return np.load(mp,allow_pickle=False),report

def context(canonical,donor):
 stage=canonical/'same_budget_weight/stage3';d=donor[:2]
 if d=='D3':
  p=canonical/'data/e4_tcell.npz';z=np.load(p,allow_pickle=True);gp=stage/'data/e4_w1_stage3.npz';g=np.load(gp,allow_pickle=True);cm=z['cm_keep'];ce=z['ce_q'];scale=g['scale'];cvm=z['cvm'];audit_ps=g['proj_sum'];audit_pq=g['proj_sumsq'];mapfile=canonical/'same_budget_weight/inputs/own_gene_map.npz';own=np.load(mapfile,allow_pickle=True)['ids'];source=[p,gp,mapfile]
 else:
  p=stage/f'data/{"d1_confirm" if d=="D1" else "d2_prospective"}_state.npz'
  z=np.load(p,allow_pickle=True);g=z;cm=z['cm'];ce=z['ce'][z['qu']];scale=z['scale'];cvm=z['cvm'];audit_ps=z['proj_sum'][0];audit_pq=z['proj_sumsq'][0];source=[p]
  if d=='D1':mapfile=stage/'data/d1_own_gene_map.npz';own=np.load(mapfile,allow_pickle=True)['ids'];source.append(mapfile)
  else:own=z['ids']
 qu=np.asarray(z['qu'],int);genes=strings(z['genes_keep']);names=strings(z['names']);own=strings(own);qpos={int(v):i for i,v in enumerate(qu)}
 out={'internal_names':names,'internal_ids':own,'keep':np.asarray(z['keep'],int),'gene_ids_keep':genes,'qu':qu,'query_keep_positions':qu,'query_gene_ids':genes[qu], 'responsive_positions':np.array([qpos[int(v)]for v in z['q_resp']],int),'tercile_positions':np.array([qpos[int(v)]for v in z['q_terc']],int),'blocks_responsive':z['blocks_responsive'],'blocks_terciles':z['blocks_terciles'],'U':g['directions'],'cm_keep':cm,'cm_q':cm[qu],'ce_q':ce,'lam':g['eigenvalues'],'trC':g['control_trace'],'proj_control':g['proj_control'],'n_control':g['n_control'],'scale':scale,'cvm':cvm,'truth':z['S4_q'].astype(float)/100-ce[:,None], 'audit_S0_q':z['S0_q'],'audit_m2_20':z['m2_20'],'audit_vb_20':z['vb_20'],'audit_proj_sum':audit_ps,'audit_proj_sumsq':audit_pq}
 if 'S3_q'in z:out['expected_reserve_Sq']=z['S3_q']
 out['self_row']=np.array([{g:i for i,g in enumerate(genes[qu])}.get(x,-1)for x in own],int)
 return out,{str(p):sha(p)for p in source}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--donor',choices=DONORS,required=True);ap.add_argument('--canonical',type=Path);ap.add_argument('--a19r',type=Path,required=True);args=ap.parse_args();root=args.root.resolve();out=args.out.resolve()/args.donor;out.mkdir(parents=True,exist_ok=True);canonical=(args.canonical or root/'restored/PertRank_Guide_Correction_20261005/canonical/repro_v10').resolve();donor=args.donor
 if (out/'selection_report.json').exists():raise RuntimeError('Completed selection exists: use a new output directory, do not overwrite')
 t=time.time();ctx,sourcefiles=context(canonical,donor);meta0,mr=metadata(BASE.format(donor),out);meta={k:meta0[k] for k in meta0.files};meta0.close();gc=meta['gene_categories'];gcode=meta['gene_codes'];types=meta['type_categories'];tc=meta['type_codes'];ids=meta['id_categories'];ic=meta['id_codes'];tot=meta['total_counts'];allgenes=meta['gene_ids_full'];assert np.array_equal(allgenes[ctx['keep']],ctx['gene_ids_keep'])
 bad=np.array([('multi'in x.lower())or x in ['NTC','non-targeting','nan']for x in gc]);ok=(~meta['low_quality'])&(gcode>=0)&(tc>=0)&(tc==np.flatnonzero(types=='targeting')[0])&(~bad[np.clip(gcode,0,len(gc)-1)])
 sel=np.flatnonzero(ok);order=np.argsort(gcode[sel],kind='stable');gs=gcode[sel][order];edges=np.flatnonzero(np.r_[True,gs[1:]!=gs[:-1]]);by={int(gs[a]):sel[order[a:b]]for a,b in zip(edges,np.r_[edges[1:],len(gs)])};pos={x:i for i,x in enumerate(gc)};salt=f'pertrescue-tcell-probe|{donor}|';eligible=[i for i,c in by.items()if len(c)>=300];eligible.sort(key=lambda i:hx(salt+str(gc[i])));cohort=[str(gc[i])for i in eligible[-min(1200,len(eligible)):]];assert np.array_equal(cohort,ctx['internal_names']);assert len(cohort)==EXPECTED_N[donor]
 def perm(nm,ss):
  cells=by[pos[nm]];seed=int(hx(ss)[:16],16);return cells[np.random.default_rng(seed).permutation(len(cells))]
 def target_id(nm):
  codes=np.unique(ic[by[pos[nm]]]);assert len(codes)==1 and codes[0]>=0,(nm,codes);return str(ids[codes[0]])
 rebuilt_ids=np.array([target_id(nm)for nm in cohort]);assert np.array_equal(rebuilt_ids,ctx['internal_ids'])
 internal_cells=np.array([perm(n,salt+n)[100:200].reshape(5,20)for n in cohort]);assert len(np.unique(internal_cells))==internal_cells.size
 audit_ix=np.array(sorted(range(len(cohort)),key=lambda j:hx('pertrank-real-cells-20261006|audit|'+donor+'|'+cohort[j]))[:3],int);audit_cells=np.array([perm(cohort[i],salt+cohort[i])[:20]for i in audit_ix])
 planning=np.concatenate([perm(n,salt+n)[:100]for n in cohort]);evaluation=np.concatenate([perm(n,salt+n)[200:300]for n in cohort]);assert not np.intersect1d(internal_cells,planning).size and not np.intersect1d(internal_cells,evaluation).size
 extfile=args.a19r.resolve()/donor/'selection.json';recs=json.loads(extfile.read_text());extnames=[x['target']for x in recs];assert len(extnames)==EXPECTED_EXT[donor];extcells=[];overlap=[]
 for rr in recs:
  nm=rr['target'];assert len(by[pos[nm]])==rr['qualifying_cells'];assert target_id(nm)==rr['ensembl_id']
  if nm in cohort:j=cohort.index(nm);extcells.append(internal_cells[j,:2]);overlap.append(j)
  else:extcells.append(perm(nm,f'pertrank-real-cells-20261006|external|{donor}|{nm}')[:40].reshape(2,20));overlap.append(-1)
 extcells=np.array(extcells);rows=np.unique(np.r_[internal_cells.ravel(),extcells.ravel(),audit_cells.ravel()]);ip=np.searchsorted(rows,internal_cells);ep=np.searchsorted(rows,extcells);ap0=np.searchsorted(rows,audit_cells)
 assert np.array_equal(rows[ip],internal_cells) and np.array_equal(rows[ep],extcells) and np.array_equal(rows[ap0],audit_cells)
 idx=np.full(len(rows),-1,int);grp=np.full(len(rows),-1,int);exidx=idx.copy();exgrp=grp.copy();af=np.zeros(len(rows),bool)
 for j in range(len(cohort)):
  for b in range(5):idx[ip[j,b]]=j;grp[ip[j,b]]=b
 for j in range(len(extnames)):
  for b in range(2):exidx[ep[j,b]]=j;exgrp[ep[j,b]]=b
 af[ap0]=True;ctx['gene_ids_full']=allgenes;ctx['external_names']=np.array(extnames);ctx['external_ids']=np.array([target_id(n)for n in extnames]);qmap={v:i for i,v in enumerate(ctx['query_gene_ids'])};ctx['external_self_row']=np.array([qmap.get(v,-1)for v in ctx['external_ids']],int)
 ctx['audit_target_indices']=audit_ix
 ctx.update({'donor':np.array(donor),'G':np.array(len(ctx['keep'])),'n_weight_genes':np.array(len(ctx['keep'])),'names':ctx['internal_names'],'target_ids':ctx['internal_ids'],'genes_keep':ctx['gene_ids_keep'],'resp_pos':ctx['responsive_positions'],'terc_pos':ctx['tercile_positions'],'T':ctx['truth'],'external_target_ids':ctx['external_ids']})
 for key in ['audit_S0_q']:ctx[key]=ctx[key][:,audit_ix]
 for key in ['audit_m2_20','audit_vb_20','audit_proj_sum','audit_proj_sumsq']:ctx[key]=ctx[key][audit_ix]
 ctx['audit_m2']=ctx['audit_m2_20'];ctx['audit_vb']=ctx['audit_vb_20'];ctx['audit_Yq']=ctx['audit_S0_q'].T/20-ctx['cm_q'][None,:]
 ctx['audit_pmean']=ctx['audit_proj_sum'][:,:5]/20-ctx['proj_control'][None,:5]
 ctx['audit_pvar']=(ctx['audit_proj_sumsq'][:,:5]-ctx['audit_proj_sum'][:,:5]**2/20)/19
 atomic_npz(out/'context.npz',**ctx)
 atomic_npz(out/'selection.npz',cell_ids=rows,rows=rows,total_counts=tot[rows],internal_positions=ip,external_positions=ep,audit_positions=ap0,internal_names=np.array(cohort),external_names=np.array(extnames),internal_ids=rebuilt_ids,external_ids=ctx['external_ids'],internal_target_index=idx,internal_group_index=grp,target_index=idx,group_index=grp,external_target_index=exidx,external_group_index=exgrp,audit_flag=af,audit_target_indices=audit_ix,external_internal_indices=np.array(overlap),guide_codes=meta['guide_codes'][rows],guide_categories=meta['guide_categories'],lane_codes=meta['lane_codes'][rows],lane_categories=meta['lane_categories'])
 comp=[]
 for j,nm in enumerate(cohort):
  for b in range(5):
   rr=internal_cells[j,b];gu,gn=np.unique(meta['guide_codes'][rr],return_counts=True);la,ln=np.unique(meta['lane_codes'][rr],return_counts=True);comp.append({'target':nm,'replicate':b,'guide_counts':{str(meta['guide_categories'][v])if v>=0 else'unassigned':int(n)for v,n in zip(gu,gn)},'lane_counts':{str(meta['lane_categories'][v])if v>=0 else'unassigned':int(n)for v,n in zip(la,ln)}})
 atomic_json(out/'composition.json',comp)
 report={'donor':donor,'source_url':BASE.format(donor),'source_size':mr['source_size'],'etag':mr['etag'],'internal_candidates':len(cohort),'external_candidates':len(extnames),'external_overlap':sum(np.array(overlap)>=0),'unique_rows':len(rows),'internal_cells':internal_cells.size,'external_only_cells':40*sum(np.array(overlap)<0),'audit_cells':audit_cells.size,'audit_targets':[cohort[i]for i in audit_ix],'queries':len(ctx['qu']),'weight_genes':len(ctx['keep']),'internal_shape':list(ip.shape),'external_shape':list(ep.shape),'planning_overlap':0,'evaluation_overlap':0,'raw_target_expression_read':False,'historical_salt':salt,'external_seed_salt':'pertrank-real-cells-20261006|external|{DONOR}|{TARGET}','forced_balance':False,'input_sha256':sourcefiles,'a19r_selection_sha256':sha(extfile),'script_sha256':sha(__file__),'selection_sha256':sha(out/'selection.npz'),'context_sha256':sha(out/'context.npz'),'seconds':time.time()-t}
 atomic_json(out/'selection_report.json',report);print(json.dumps(report,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()),flush=True)
if __name__=='__main__':main()
