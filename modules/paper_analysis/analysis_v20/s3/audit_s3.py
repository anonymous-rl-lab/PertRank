#!/usr/bin/env python3
"""Independent algebra/FWL audit; does not call the analysis regression code."""
from pathlib import Path
import json, csv
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    with np.load(ROOT/'inputs/alignment_pretruth.npz',allow_pickle=False) as z:
        design=z['design_design'];masks=z['design_query_target_masks']
    with np.load(ROOT/'results/replay_statistics.npz',allow_pickle=False) as z:
        obs=z['observed_contributions'];null=z['neutral_contributions']
        orig=z['observed_coefficients'];neutral=z['neutral_coefficients']
        bins=z['alignment_bin'];acc=z['accounting'];delta=z['primary_query_delta']
        blocks=z['blocks_30']
    fwl=[]
    for q,mask in enumerate(masks):
        x=design[mask,1];r=design[mask,2]
        xc=x-x.mean();rc=r-r.mean()
        residual=xc-rc*np.dot(rc,xc)/np.dot(rc,rc)
        fwl.append([np.dot(residual,values[mask,q])/np.dot(residual,residual) for values in (obs,null)])
    fwl=np.asarray(fwl)
    reg_error=max(np.max(np.abs(fwl[:,0]-orig[:,1])),np.max(np.abs(fwl[:,1]-neutral[:,1])))
    assert reg_error<1e-15
    alignment_nets=np.array([obs[bins==a].sum(0)*25 for a in range(4)]).T
    acc_error=np.max(np.abs(alignment_nets-acc[:,:,:,2].sum(2)))
    assert acc_error<1e-13
    sum_error=np.max(np.abs(obs.sum(0)-delta))
    assert sum_error<1e-14
    with (ROOT/'results/leave_one_block_out.csv').open() as f: rows=list(csv.DictReader(f))
    loo_error=max(abs(float(row['primary_P_at_25'])-delta[blocks!=int(row['omitted_block'])].mean()) for row in rows)
    assert loo_error<1e-15
    with (ROOT/'results/s3_weight_transitions.csv').open() as f: rows=list(csv.DictReader(f))
    assert sum(int(row['target_input_count']) for row in rows)==2805
    transition_error=abs(sum(float(row['net_supported']) for row in rows)-25*delta.mean())
    assert transition_error<1e-14
    audit={'status':'PASS','independent_FWL_max_coefficient_error':float(reg_error),
      'target_contributions_vs_gain_loss_max_error':float(acc_error),
      'target_sum_vs_primary_max_error':float(sum_error),'leave_one_block_out_max_error':float(loo_error),
      'transition_sum_vs_primary_error':float(transition_error),
      'scope':'Independent residualization and accounting; no new inference or changed registered endpoint.'}
    (ROOT/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
