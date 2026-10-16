"""Pure exact-byte plan builder. No network, selection, normalization or scoring."""
import numpy as np

def plans(manifest,rows,ptr,specs):
 # Every transported source segment is solely from an authorized selected row.
 # Coalescing may carry intervening bytes, which are never decoded or analyzed.
 pieces=[];ip=np.r_[0,np.cumsum(ptr[rows+1]-ptr[rows])]
 for r,cell in enumerate(rows):
  lo,hi=int(ptr[cell]),int(ptr[cell+1])
  for kind in ['data','indices']:
   sp=specs[kind];ck=int(sp.get('chunk_size',sp.get('chunk',[0])[0]));item=int(sp['itemsize'])
   for coord in range(lo//ck*ck,hi,ck):
    a=max(lo,coord);b=min(hi,coord+ck);mc=manifest[kind][str(coord)] if str(coord)in manifest[kind] else manifest[kind][coord]
    addr=int(mc['offset'])+(a-coord)*item;end=addr+(b-a)*item
    pieces.append((addr,end,kind,int(ip[r]+a-lo),b-a))
 return sorted(pieces),ip

def packs_for(pieces,gap,maxsize=8<<20):
 packs=[]
 for v in pieces:
  a,b,kind,dest,n=v
  if packs and a-packs[-1]['end']<=gap and b-packs[-1]['start']<=maxsize:packs[-1]['end']=max(packs[-1]['end'],b);packs[-1]['pieces'].append(v)
  else:packs.append({'start':a,'end':b,'pieces':[v]})
 return packs
