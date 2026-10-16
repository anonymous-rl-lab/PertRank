#!/usr/bin/env python3
"""Post-result lossless top-25 support delta; never refits or selects a method.

Full original float64 scoring inputs are required only for this builder. The
output adds missing TRADE boundary entries to the unmodified D1-D4 core support.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def membership(ids, values, nt):
    # Full candidate axis preserves the original float64 summation order.
    out=np.zeros((2,nt), dtype=np.float64)
    for j, sign in enumerate((1.,-1.)):
        v=values*sign; threshold=np.sort(v)[-25]
        above=v>threshold; tied=v==threshold
        out[j,ids[above]]=1.
        out[j,ids[tied]]=(25-int(above.sum()))/int(tied.sum())
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root-base-data',type=Path,required=True,
                   help='D1-D4 repository root, containing data/D4/ranking.npz')
    p.add_argument('--full-scoring-inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args(); rp=a.root_base_data/'data/D4/ranking.npz'
    base=load(rp);full=load(a.full_scoring_inputs)
    y=full['raw_Y'];truth=full['truth'];weights=full['weights']
    assert y.shape==(5,593,589) and weights.shape==(5,5,589)
    assert list(full['arm_names'])==['raw','Euclidean','PertRank','PR_M','TRADE_scalar']
    assert np.array_equal(weights[:4],base['weights'])
    for s,b in [('query_gene_ids','query_ids'),('target_ids','target_ids')]:
        assert np.array_equal(full[s],base[b]),s
    offsets=[0];allids=[];values=[];fullh=hashlib.sha256();sparseh=hashlib.sha256()
    comparisons=0
    for g in range(5):
        for q in range(593):
            row=g*593+q; sl=slice(int(base['Y_offsets'][row]),int(base['Y_offsets'][row+1]))
            oldids=base['Y_ids'][sl].astype(int)
            assert np.array_equal(y[g,q,oldids],base['Y_values'][sl])
            eligible=np.flatnonzero(np.arange(589)!=base['own_index'][q])
            fulltrade=membership(eligible,y[g,q,eligible]*weights[4,g,eligible],589)
            need=np.flatnonzero(np.any(fulltrade>0,axis=0))
            missing=np.setdiff1d(need,oldids)
            ids=np.union1d(oldids,missing)
            allids.extend(missing);values.extend(y[g,q,missing]);offsets.append(len(allids))
            for arm in range(5):
                f=membership(eligible,y[g,q,eligible]*weights[arm,g,eligible],589)
                s=membership(ids,y[g,q,ids]*weights[arm,g,ids],589)
                assert np.array_equal(f,s),(g,q,arm)
                fullh.update(f.tobytes());sparseh.update(s.tobytes());comparisons+=f.size
            ts=slice(int(base['truth_offsets'][q]),int(base['truth_offsets'][q+1]))
            tids=base['truth_ids'][ts].astype(int)
            assert np.array_equal(truth[q,tids],base['truth_values'][ts])
            assert np.array_equal(membership(tids,truth[q,tids],589),membership(eligible,truth[q,eligible],589))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,Y_offsets=np.asarray(offsets,np.uint32),
                        Y_ids=np.asarray(allids,np.uint16),Y_values=np.asarray(values,np.float64))
    report={'status':'PASS','identity':'post-result exact top-25 support delta, not a prospective lock',
            'base_ranking_sha256':sha(rp),'full_scoring_inputs_sha256':sha(a.full_scoring_inputs),
            'delta_sha256':sha(a.output),'delta_entries':len(values),'delta_bytes':a.output.stat().st_size,
            'prediction_membership_values_compared':comparisons,'all_boundary_ties_preserved':True,
            'all_five_arm_memberships_exact':True,'truth_memberships_exact':True,
            'float_quantization':False,'original_base_changed':False,
            'full_membership_stream_sha256':fullh.hexdigest(),
            'sparse_membership_stream_sha256':sparseh.hexdigest(),
            'scope':'Fixed five arms, five groups, original panels and candidate pool, K=25 only. Arbitrary new weights are unsupported.'}
    a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
