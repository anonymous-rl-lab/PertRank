#!/usr/bin/env python3
"""Bind opened pooled tasks to their existing exact row selections.

Only D4 Rest/Stim sources are allowed. All five original shallow groups are
locked up front; payload reads remain separately and explicitly selected.
"""
from pathlib import Path
import argparse, hashlib, json, shutil
import numpy as np
import requests

ROOT=Path(__file__).resolve().parents[1]
ARC=ROOT/'archives'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('task',choices=['D4_Rest','D4_Stim8hr']);a=ap.parse_args()
    out=Path(__file__).parent/a.task;out.mkdir(parents=True,exist_ok=True)
    if a.task=='D4_Rest':
        d=ARC/'PertRank_D4_OSF_core_20261008/result/data';cp=d/'cohort_core.npz';c=load(cp)
        rows=c['input_cells'];ntc=c['cin']; all0=c['selected_source_rows'];tot0=c['selected_total_counts']
        model=d/'control_model.npz';qp=d/'query_context.npz';pp=d/'prediction_core.npz';tp=d/'truth.npz'
        pred=load(pp);y=pred['Y'].transpose(0,2,1);w0=pred['weights'][list(pred['arm_names']).index('Euclidean')];wm=pred['weights'][list(pred['arm_names']).index('PertRank')]
        truth=load(tp)['truth'].T
        source={'url':f'https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/{a.task}.assigned_guide.h5ad'}
        h=requests.head(source['url'],timeout=30);h.raise_for_status();source.update(etag=h.headers['ETag'],size=int(h.headers['Content-Length']))
        files=[cp,model,qp,pp,tp]
    else:
        d=ARC/'PertRank_Stim_prediction_lock/PertRank_Stim_prediction_lock';cp=d/'cohort_roles.npz';c=load(cp)
        rows=c['role__main_input'].reshape(-1,5,20);ntc=c['role__input_control']
        n=load(ROOT/'independent/metadata_counts/D4_Stim8hr/normalization_metadata.npz');all0=np.arange(len(n['total_counts']));tot0=n['total_counts']
        model=d/'input/control_model.npz';qp=d/'input/main_query_context.npz';pp=d/'input/main_nontrade_predictions.npz';pred=load(pp)
        y=pred['original_effects'];w0=pred['weight_Euclidean'];wm=pred['weight_PertRank']
        source=json.loads((d/'SOURCE_IDENTITY.json').read_text());source={k:source[k]for k in ('url','size','etag')}
        # Mean reference in the archival recovery consists of full-G mean
        # vectors; independent evaluation control is encoded in support.
        support=load(ARC/'PertRank_Stim_Exact_Reproduction_v1.2.0/PertRank_Stim_Exact_Reproduction/data/main_support.npz')
        # Sparse support cannot stand in for the full reference (values below
        # the saved top-K boundaries were omitted). Recover full means and
        # evaluation baseline from the matching saved boundary entries.
        q=load(qp);tr=[];rd=ARC/'PertRank_Stim_reference_recovery/PertRank_Stim_reference_recovery/evaluation/main'
        for i in range(len(c['target_ids'])):tr.append(load(rd/f'target_{i:04d}.npz')['mean'][q['qu']])
        tr=np.stack(tr);ce=[]
        for j in range(tr.shape[1]):
            lo,hi=support['truth_offsets'][j:j+2];ix=support['truth_ids'][lo:hi];v=support['truth_values'][lo:hi]
            dif=tr[ix,j]-v
            assert np.ptp(dif)<1e-12
            ce.append(float(np.median(dif)))
        truth=tr-np.array(ce)[None,:]
        files=[cp,model,qp,pp,d/'SOURCE_IDENTITY.json']
    assert rows.shape[1:]==(5,20) and ntc.shape==(2000,)
    selected=np.unique(np.r_[ntc,rows.ravel()]);pos=np.searchsorted(all0,selected);assert np.array_equal(all0[pos],selected)
    assert len(selected)==rows.size+len(ntc)
    q=load(qp);m=load(model)
    assert np.array_equal(c['target_ids'],q['target_ids']) and np.array_equal(q['target_ids'],pred['target_ids'])
    roles={'NTC_input':ntc,'A_input':rows[:,0]}
    roles.update({f'input_group{r}':rows[:,r] for r in range(1,5)})
    p=out/'selection.npz'
    np.savez_compressed(p,all_selected_rows=selected,selected_total_counts=tot0[pos],gene_ids_full=c['gene_ids_full'],target_ids=c['target_ids'],target_names=c['target_names'],input_cells=rows,**roles)
    lock={'task':a.task,'purpose':'replay original pooled-guide input groups for external-method benchmark','all_sources_previously_opened':True,'selection_sha256':sha(p),'source_identity':source,'source_file_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},'payload_permission':'Only explicitly requested input groups and original input NTC; no D1/D2 Stim source permitted'}
    (out/'selection.lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    shutil.copy2(model,out/'control_model.npz');shutil.copy2(qp,out/'query_context.npz')
    np.savez_compressed(out/'frozen_task.npz',original_effects=y,weight_Euclidean=w0,weight_PertRank=wm,truth=truth,target_ids=c['target_ids'],target_names=c['target_names'],query_gene_ids=q['query_gene_ids'],gene_ids_full=c['gene_ids_full'],responsive_positions=q['responsive_positions'],blocks_responsive=q['blocks_responsive'],tercile_positions=q['tercile_positions'],blocks_terciles=q['blocks_terciles'])
    if a.task=='D4_Stim8hr':
        # Reuse immutable CSR structural metadata; missing selected chunk
        # coordinates are looked up by the unchanged source index parser.
        rd=out/'raw';rd.mkdir(exist_ok=True);src=ROOT/'independent/execution/D4_Stim8hr/raw'
        for name in ['indptr.npy','indptr_report.json','hdf_objects.json','gene_ids_full.npy']:shutil.copy2(src/name,rd/name)
    print(json.dumps({'task':a.task,'targets':len(c['target_ids']),'rep0_rows':rows[:,0].size+len(ntc),'all5_rows':len(selected),'selection_sha256':sha(p),'frozen_task_sha256':sha(out/'frozen_task.npz')},indent=2))
if __name__=='__main__':main()
