#!/usr/bin/env python3
"""Rebuild historical D3 NTC IDs; preserve R20 shallow target selection."""
from pathlib import Path
import sys,json,hashlib,shutil,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ARC=ROOT/'archives'
SRC=ARC/'R20_Frozen_Design_20261006/research_round_20261006/new_experiment'
spec=importlib.util.spec_from_file_location('old_selection',SRC/'reader/make_selection.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def main():
    d=Path(__file__).parent/'D3_Rest';d.mkdir(parents=True,exist_ok=True);md=d/'metadata';md.mkdir(exist_ok=True)
    z,mr=old.metadata(old.BASE.format('D3_Rest'),md);meta={k:z[k]for k in z.files};z.close()
    c=load(SRC/'selection/D3_Rest/context.npz');s=load(SRC/'selection/D3_Rest/selection.npz')
    # Exactly the original historical NTC eligibility and seeded selection.
    ok=(~meta['low_quality'])&(meta['gene_codes']>=0)&(meta['type_codes']>=0)
    ii=int(np.flatnonzero(meta['type_categories']=='non-targeting')[0]);ntc=np.flatnonzero(ok&(meta['type_codes']==ii))
    salt='pertrescue-tcell-probe|D3_Rest|';ntc=np.array(sorted(ntc.tolist(),key=lambda i:hashlib.sha256((salt+'ctrl|'+str(i)).encode()).hexdigest())[:2000],np.int64)
    assert float(np.median(meta['total_counts'][ntc]))==float(c['scale'])
    rows=s['rows'][s['internal_positions']]
    ws=load(ARC/'PertRank_D1_D4_GitHub_Core_v1.2.0/PertRank_D1_D4_GitHub_Core/data/D3/weight_stats.npz')
    assert np.array_equal(np.sort(ntc),np.sort(ws['input_control_ids']))
    assert np.array_equal(rows,ws['input_cell_ids'])
    assert rows.shape==(1200,5,20) and not np.intersect1d(rows,ntc).size
    # Only one development repetition is planned for tuning. The existing
    # R20 other groups remain in archived source and are not decoded here.
    selected=np.unique(np.r_[ntc,rows[:,0].ravel()]);assert len(selected)==26000
    sp=d/'selection.npz';np.savez_compressed(sp,all_selected_rows=selected,selected_total_counts=meta['total_counts'][selected],gene_ids_full=meta['gene_ids_full'],target_ids=c['target_ids'],target_names=c['internal_names'],input_cells=rows,NTC_input=ntc,A_input=rows[:,0])
    source={'url':mr['source_url'],'etag':mr['etag'],'size':mr['source_size']}
    lock={'task':'D3_Rest','selection_sha256':old.sha(sp),'source_identity':source,'historical_NTC_rule':salt+'ctrl|row SHA256 first2000 eligible','target_selection_sha256':old.sha(SRC/'selection/D3_Rest/selection.npz'),'context_sha256':old.sha(SRC/'selection/D3_Rest/context.npz'),'purpose':'R20 rep0 exact cells for external-method development; previously opened data','D1_D2_Stim_access':False}
    (d/'selection.lock.json').write_text(json.dumps(lock,indent=2)+'\n');shutil.copy2(SRC/'selection/D3_Rest/context.npz',d/'r20_context.npz')
    raw=d/'raw';raw.mkdir(exist_ok=True);shutil.copy2(md/'hdf_objects.json',raw/'hdf_objects.json');np.save(raw/'gene_ids_full.npy',meta['gene_ids_full'])
    print(json.dumps({'task':'D3_Rest','rep0_cells':len(selected),'targets':1200,'selection_sha256':old.sha(sp),'metadata_expression_read':False},indent=2),flush=True)
if __name__=='__main__':main()
