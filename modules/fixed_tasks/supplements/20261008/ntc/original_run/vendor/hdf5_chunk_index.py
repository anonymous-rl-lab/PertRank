"""Strict, cached coordinate-pruned reader for v1 HDF5 B-tree chunk indexes.

Adapted from the existing build_chunk_manifest_fast.py transport only. No cell
selection or numeric analysis occurs here. The caller supplies the observed
dataset object-header addresses, dtype size, chunk size, and needed coordinates.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests,threading,struct,bisect,time,hashlib,json,os

class Transport:
 def __init__(self,url,cache):
  self.url=url;self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
  identity=self.cache/'SOURCE.json'
  h=requests.head(url,timeout=30);h.raise_for_status()
  expected={'url':url,'etag':h.headers.get('ETag'),'size':int(h.headers['Content-Length'])}
  if identity.exists():
   old=json.loads(identity.read_text());assert old['url']==url,'cache belongs to a different source'
   if 'etag' in old:assert old==expected,'source version changed'
  identity.write_text(json.dumps(expected,indent=2));self.expected=expected
  self.local=threading.local();self.calls=0;self.nbytes=0;self.lock=threading.Lock();self.status_counts={};self.retry_events=0
 def fetch(self,address,size):
  path=self.cache/f'{address}_{size}.bin'
  if path.exists():
   b=path.read_bytes();assert len(b)==size;return b
  if not hasattr(self.local,'session'):self.local.session=requests.Session()
  for attempt in range(5):
   try:
    end=address+size-1
    r=self.local.session.get(self.url,headers={'Range':f'bytes={address}-{end}'},timeout=45)
    with self.lock:self.status_counts[r.status_code]=self.status_counts.get(r.status_code,0)+1
    retry_after=r.headers.get('Retry-After')
    if r.status_code in (429,503):
     event={'time':time.time(),'status':r.status_code,'retry_after':retry_after,'attempt':attempt,'address':address,'size':size}
     with self.lock:
      with (self.cache/'TRANSPORT_RETRY_EVENTS.jsonl').open('a')as f:f.write(json.dumps(event)+'\n')
     r.close()
     if attempt==4:raise RuntimeError(f'Throttled after five attempts: {r.status_code}')
     try:delay=float(retry_after)
     except (TypeError,ValueError):delay=2**(attempt+1)
     time.sleep(max(delay,2**(attempt+1)))
     continue
    if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {address}-{end}/'):
     raise RuntimeError(f'Invalid range response: {r.status_code}, {r.headers.get("Content-Range")}')
    b=r.content;r.close()
    assert len(b)==size,(len(b),size)
    assert r.headers.get('ETag')==self.expected['etag'],'source ETag changed'
    tmp=path.with_suffix(f'.{threading.get_ident()}.tmp');tmp.write_bytes(b);os.replace(tmp,path)
    with self.lock:self.calls+=1;self.nbytes+=size
    return b
   except Exception as exc:
    with self.lock:
     self.retry_events+=1
     with (self.cache/'TRANSPORT_RETRY_EVENTS.jsonl').open('a')as f:f.write(json.dumps({'time':time.time(),'exception':type(exc).__name__,'message':str(exc),'attempt':attempt,'address':address,'size':size})+'\n')
    if attempt==4:raise
    time.sleep(2**attempt)

def layout_root(transport,object_address):
 raw=transport.fetch(int(object_address),272)
 if raw[0]!=1:raise ValueError('Only observed HDF5 v1 object headers are supported')
 nmsg=struct.unpack_from('<H',raw,2)[0];p=16
 for _ in range(nmsg):
  typ,size=struct.unpack_from('<HH',raw,p);b=raw[p+8:p+8+size]
  if typ==8:
   assert b[:3]==bytes([3,2,2]),b[:3]
   return struct.unpack_from('<Q',b,3)[0]
  p+=8+size
 raise ValueError('No v3 chunk-layout message in the observed header')

def chunk_manifest(transport,specs,coordinates,workers=16):
 """specs[name]={object_address,chunk_size,itemsize}; coords are element offsets."""
 coords={name:sorted(set(int(x) for x in coordinates[name])) for name in specs}
 pending=[(name,layout_root(transport,spec['object_address'])) for name,spec in specs.items()]
 visited=set();out={name:{} for name in specs};t=time.monotonic()
 def parse(task):
  name,addr=task;raw=transport.fetch(addr,2096)
  assert raw[:4]==b'TREE' and raw[4]==1,(name,addr,raw[:6])
  level=raw[5];n=struct.unpack_from('<H',raw,6)[0];assert 0<n<=64
  keys=[struct.unpack_from('<IIQQ',raw,24+i*32) for i in range(n+1)]
  # HDF5 chunk keys include a trailing byte-coordinate. At the last leaf,
  # the sentinel can repeat the final element coordinate with tail=itemsize.
  # Comparing only the first coordinate incorrectly rejects that valid bound.
  assert all(keys[i][2:]<keys[i+1][2:] for i in range(n)),(name,addr,keys[-2:])
  children=[struct.unpack_from('<Q',raw,48+i*32)[0] for i in range(n)]
  selected=[];cc=coords[name]
  for i in range(n):
   lower,upper=keys[i][2],keys[i+1][2];j=bisect.bisect_left(cc,lower)
   if j<len(cc) and keys[i][2:] <= (cc[j],0) < keys[i+1][2:]:selected.append((keys[i],children[i]))
  return name,addr,level,selected
 with ThreadPoolExecutor(max_workers=workers) as pool:
  while pending:
   nxt=[]
   for name,addr,level,selected in pool.map(parse,pending):
    assert (name,addr) not in visited;visited.add((name,addr))
    for key,child in selected:
     if level:nxt.append((name,child))
     else:
      size,mask,coord,tail=key;sp=specs[name]
      assert mask==0 and tail==0 and size==sp['chunk_size']*sp['itemsize'] and coord%sp['chunk_size']==0,(name,key)
      out[name][coord]={'offset':child,'size':size,'filter_mask':mask}
   pending=nxt
   print('index_progress','nodes',len(visited),'next',len(pending),'chunks',sum(map(len,out.values())),'seconds',round(time.monotonic()-t,1),flush=True)
 for name in specs:assert set(out[name])==set(coords[name]),(name,len(out[name]),len(coords[name]))
 return out,{'metadata_nodes':len(visited),'network_calls':transport.calls,'network_bytes':transport.nbytes,'seconds':time.monotonic()-t}

def coalesce(manifest,max_gap=2<<20,max_size=16<<20):
 pieces=sorted((v['offset'],v['offset']+v['size'],name,int(coord)) for name,part in manifest.items() for coord,v in part.items())
 packs=[]
 for start,end,name,coord in pieces:
  if packs and start-packs[-1]['end']<=max_gap and end-packs[-1]['start']<=max_size:
   packs[-1]['end']=max(packs[-1]['end'],end);packs[-1]['pieces'].append((start,end,name,coord))
  else:packs.append({'start':start,'end':end,'pieces':[(start,end,name,coord)]})
 return packs
