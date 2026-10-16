from pathlib import Path
import sys,json,collections,bisect,datetime
import numpy as np
HERE=Path(__file__).resolve().parent;RUN=Path('/workspace/scratch/6cdbaefbed9f/stim_run_20261011')
sys.path.insert(0,str(RUN/'src'))
from reader_vendor.row_transport import plans
role=np.load(RUN/'role_rows.npz');available=set();counts=collections.Counter();missing=collections.Counter()
for f in (RUN/'raw_rows').glob('*.json'):
 d=json.loads(f.read_text())
 if f.with_suffix('.npz').exists():available.update(d['rows']);counts[d['role']]+=1
 else:missing[d['role']]+=1
covered=np.isin(role['main_input'],list(available));bins=[]
for f in (RUN/'transport/ranges').glob('*.bin'):
 a,s=map(int,f.stem.split('_'));bins.append((a,a+s))
bins.sort();starts=[];ends=[]
for a,b in bins:
 if ends and a<=ends[-1]:ends[-1]=max(b,ends[-1])
 else:starts.append(a);ends.append(b)
sp=json.loads((RUN/'metadata/hdf_objects.json').read_text());mf=json.loads((RUN/'metadata/input_chunk_manifest.json').read_text())['chunks'];ptr=np.load(RUN/'metadata/indptr.npy',mmap_mode='r')
pieces,_=plans(mf,role['main_input'].ravel(),ptr,sp);covered_bytes=0;missing_bytes=0;missing_pieces=0;exact_cached_bytes=0
for a,b,*_ in pieces:
 i=bisect.bisect_right(starts,a)-1
 j=max(i,0)
 while j<len(starts) and starts[j]<b:
  exact_cached_bytes+=max(0,min(b,ends[j])-max(a,starts[j]));j+=1
 if i>=0 and ends[i]>=b:covered_bytes+=b-a
 else:missing_bytes+=b-a;missing_pieces+=1
r={'classification':'asset feasibility, no new ranking result','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'raw_files_present_by_role':dict(counts),'raw_files_missing_by_role':dict(missing),'main_input_cells_total':int(covered.size),'main_input_cells_in_existing_raw_npz':int(covered.sum()),'targets_with_all100_input_cells':int(covered.all(1).sum()),'transport_ranges':len(bins),'required_input_payload_bytes':covered_bytes+missing_bytes,'fully_covered_payload_bytes':covered_bytes,'missing_or_partially_covered_payload_bytes':missing_bytes,'missing_pieces':missing_pieces,'exact_cached_payload_bytes':exact_cached_bytes,'exact_missing_payload_bytes':covered_bytes+missing_bytes-exact_cached_bytes,'available_statistics':'full-gene per-group means and variances plus projections onto original U','required_missing_statistics':'per-target covariance under replacement directions: diag(U_new^T Cov_target U_new)','reason':'Per-gene diagonal variances and original five-direction projected variances do not identify variance in arbitrary replacement directions. Target raw rows were pruned after checkpointing; payload packs were also pruned.','status':'requires_reacquiring_original_locked_input_rows'}
(HERE/'CONTROL_REPLACEMENT_ASSET_REPORT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
