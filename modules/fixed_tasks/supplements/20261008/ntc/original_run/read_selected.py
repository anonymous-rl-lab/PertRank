#!/usr/bin/env python3
"""Exact-byte D3 Rest selected-row reader with checkpoints and source identity checks."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import h5py
from scipy.sparse import csr_matrix,save_npz,load_npz
ROOT=Path(__file__).resolve().parent;REC=ROOT.parents[1]
VENDOR=ROOT/'vendor'
sys.path.insert(0,str(VENDOR))
from hdf5_chunk_index import Transport,chunk_manifest
from row_transport import plans,packs_for
from metadata_source import MetaRange,strings

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def js(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p)
def arrays(p,**x):
    p=Path(p);t=p.with_name(p.stem+'.tmp.npz');np.savez_compressed(t,**x);os.replace(t,p)
def fetch_iter(tr,items,address,size,workers=8,delete=False):
    it=iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={}
        def submit(v):pending[pool.submit(tr.fetch,int(address(v)),int(size(v)))]=v
        for _ in range(2*workers):
            try:submit(next(it))
            except StopIteration:break
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                item=pending.pop(future);body=future.result();yield item,body
                if delete:(tr.cache/f'{int(address(item))}_{int(size(item))}.bin').unlink(missing_ok=True)
                try:submit(next(it))
                except StopIteration:pass

class Reader:
    def __init__(self,workers=8,manifest_scope='all'):
        self.workers=workers;self.sel=np.load(ROOT/'selection.npz',allow_pickle=False)
        report=json.loads((ROOT/'selection_report.json').read_text());assert sha(ROOT/'selection.npz')==report['selection_sha256']
        self.identity=report['source_identity'];self.specs=json.loads((ROOT/'metadata/hdf_objects.json').read_text())
        self.allowed=self.sel['cell_ids'];self.total=dict(zip(self.allowed.tolist(),self.sel['total_counts'].tolist()))
        self.cache=ROOT/'transport';self.cache.mkdir(exist_ok=True)
        self.ptr=self.indptr();self.manifest=self.chunk_map();self.tr=self.transport('payload')
    def transport(self,tag):
        tr=Transport(self.identity['url'],self.cache/tag);assert tr.expected==self.identity;return tr
    def indptr(self):
        p=ROOT/'metadata/indptr.npy';rp=ROOT/'metadata/indptr_report.json'
        if p.exists():
            assert sha(p)==json.loads(rp.read_text())['sha256'];return np.load(p,mmap_mode='r')
        s=self.specs['indptr'];coords=list(range(0,s['length'],s['chunk_size']));tr=self.transport('index')
        manifest,audit=chunk_manifest(tr,{'indptr':s},{'indptr':coords},self.workers)
        tr=self.transport('indptr');buf=np.empty(len(coords)*s['chunk_size'],dtype=s['dtype'])
        for (coord,rec),body in fetch_iter(tr,list(manifest['indptr'].items()),lambda v:v[1]['offset'],lambda v:v[1]['size'],self.workers,True):
            buf[int(coord):int(coord)+s['chunk_size']]=np.frombuffer(body,dtype=s['dtype'])
        out=buf[:s['length']];assert out[0]==0 and out[-1]==self.specs['data']['length'] and np.all(np.diff(out)>=0)
        np.save(p,out);js(rp,{'sha256':sha(p),'audit':audit,'source_identity':self.identity});return out
    def chunk_map(self):
        p=ROOT/'metadata/selected_chunk_manifest.json';rh=hashlib.sha256(self.allowed.tobytes()).hexdigest()
        if p.exists():
            z=json.loads(p.read_text());assert z['source_identity']==self.identity and z['rows_sha256']==rh;return z['chunks']
        specs={k:self.specs[k] for k in ['data','indices']}
        coords={k:sorted({c for row in self.allowed for c in range(int(self.ptr[row])//sp['chunk_size']*sp['chunk_size'],int(self.ptr[row+1]),sp['chunk_size'])}) for k,sp in specs.items()}
        tr=self.transport('index');mm,audit=chunk_manifest(tr,specs,coords,self.workers)
        js(p,{'chunks':mm,'audit':audit,'source_identity':self.identity,'rows_sha256':rh});return mm
    def rows(self,rows):
        rows=np.asarray(rows,np.int64);pos=np.searchsorted(self.allowed,rows)
        assert np.all(pos<len(self.allowed)) and np.array_equal(self.allowed[pos],rows) and len(np.unique(rows))==len(rows)
        pieces,ip=plans(self.manifest,rows,self.ptr,self.specs);packs=packs_for(pieces,4096,8<<20)
        data=np.empty(int(ip[-1]),np.uint32);indices=np.empty(int(ip[-1]),np.int32);ng=len(self.sel['gene_ids_full'])
        for pack,body in fetch_iter(self.tr,packs,lambda v:v['start'],lambda v:v['end']-v['start'],self.workers,True):
            for address,end,kind,dest,count in pack['pieces']:
                arr=np.frombuffer(body,dtype=self.specs[kind]['dtype'],count=count,offset=address-pack['start'])
                if kind=='data':
                    assert np.all(np.isfinite(arr)) and np.all(arr>=0) and np.all(arr==np.floor(arr));data[dest:dest+count]=arr
                else:assert np.all((arr>=0)&(arr<ng));indices[dest:dest+count]=arr
        x=csr_matrix((data,indices,ip),shape=(len(rows),ng));assert np.array_equal(np.asarray(x.sum(1)).ravel(),np.array([self.total[int(r)] for r in rows]))
        return x

def smoke():
    t=time.time();r=Reader();oldroot=ROOT/'frozen_assets/ntc_original'
    old=np.load(oldroot/'original_input_ntc_metadata.npz',allow_pickle=False);shared=np.intersect1d(r.allowed,old['cell_ids']);fresh=np.setdiff1d(r.allowed,shared)
    rows=np.sort(np.r_[shared[np.linspace(0,len(shared)-1,8,dtype=int)],fresh[np.linspace(0,len(fresh)-1,8,dtype=int)]])
    start=time.time();x=r.rows(rows);elapsed=time.time()-start
    ox=load_npz(oldroot/'original_input_ntc_raw.npz');reused=[]
    for j,row in enumerate(rows):
        ix=np.flatnonzero(old['cell_ids']==row)
        if len(ix):assert (x[j]!=ox[int(ix[0])]).nnz==0;reused.append(int(row))
    rf=MetaRange(r.identity['url'],ROOT/'smoke_h5py_cache',block=1<<16)
    assert rf.etag==r.identity['etag'] and rf.size==r.identity['size']
    with h5py.File(rf,'r') as f:
        assert np.array_equal(strings(f['var/_index'][:]),r.sel['gene_ids_full'].astype(str))
        for j in [0,len(rows)//2,len(rows)-1]:
            row=int(rows[j]);a,b=map(int,f['X/indptr'][row:row+2]);lo,hi=x.indptr[j:j+2]
            assert np.array_equal(x.data[lo:hi],f['X/data'][a:b]) and np.array_equal(x.indices[lo:hi],f['X/indices'][a:b])
    rf.close();save_npz(ROOT/'smoke_raw.npz',x);np.save(ROOT/'smoke_cell_ids.npy',rows)
    report={'status':'SMOKE_PASS','rows':rows.tolist(),'n_rows':len(rows),'source_identity':r.identity,'selected_source_rows_exact_vs_h5py':3,
      'rows_exact_vs_archived_raw':len(reused),'raw_total_UMI_exact':True,'source_gene_axis_exact':True,'payload_seconds':elapsed,
      'payload_rows_per_second':len(rows)/elapsed,'projected_23840_row_payload_seconds':len(r.allowed)*elapsed/len(rows),
      'payload_network_calls':r.tr.calls,'payload_network_bytes':r.tr.nbytes,'elapsed_total_seconds':time.time()-t,
      'raw_sha256':sha(ROOT/'smoke_raw.npz'),'selection_sha256':sha(ROOT/'selection.npz'),'reader_sha256':sha(__file__)}
    js(ROOT/'SMOKE_PASS.json',report);print(json.dumps(report,indent=2),flush=True)

def full():
    assert json.loads((ROOT/'SMOKE_PASS.json').read_text())['status']=='SMOKE_PASS'
    freeze=json.loads((ROOT/'RUN_FREEZE.json').read_text())
    for p,want in freeze['file_sha256'].items():assert sha(ROOT/p)==want,('freeze mismatch',p)
    r=Reader();out=ROOT/'raw_batches';out.mkdir(exist_ok=True)
    total=len(r.allowed);t=time.time();batch_size=256
    for k,a in enumerate(range(0,total,batch_size)):
        rows=r.allowed[a:a+batch_size];p=out/f'batch_{k:04d}.npz';rp=p.with_suffix('.json')
        if p.exists() and rp.exists():
            z=json.loads(rp.read_text());assert z['sha256']==sha(p) and z['rows']==rows.tolist();continue
        s=time.time();x=r.rows(rows);tmp=p.with_name(p.stem+'.tmp.npz');save_npz(tmp,x);os.replace(tmp,p)
        z={'rows':rows.tolist(),'sha256':sha(p),'n_rows':len(rows),'nnz':int(x.nnz),'seconds':time.time()-s};js(rp,z)
        print(json.dumps({'batch':k,'completed_rows':min(a+batch_size,total),'total_rows':total,'elapsed_seconds':time.time()-t,'batch_seconds':z['seconds'],'network_calls':r.tr.calls,'network_bytes':r.tr.nbytes}),flush=True)
    js(ROOT/'RAW_READ_COMPLETE.json',{'status':'RAW_READ_COMPLETE','rows':total,'batches':(total+batch_size-1)//batch_size,'seconds':time.time()-t,'selection_sha256':sha(ROOT/'selection.npz'),'reader_sha256':sha(__file__),'network_calls':r.tr.calls,'network_bytes':r.tr.nbytes,'retry_events':r.tr.retry_events})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','full']);a=p.parse_args()
    smoke() if a.mode=='smoke' else full()
