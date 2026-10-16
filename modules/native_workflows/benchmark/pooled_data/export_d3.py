#!/usr/bin/env python3
"""Export D3 R20 group0, audit all frozen group moments and support values."""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.sparse import save_npz,vstack
from export_inputs import role,load,sha,ROOT

def main():
    t=time.monotonic();d=Path(__file__).parent/'D3_Rest';s=load(d/'selection.npz');c=load(d/'r20_context.npz')
    core=ROOT/'archives/PertRank_D1_D4_GitHub_Core_v1.2.0/PertRank_D1_D4_GitHub_Core'
    ws=load(core/'data/D3/weight_stats.npz');rk=load(core/'data/D3/ranking.npz')
    assert np.array_equal(c['target_ids'],ws['target_ids']) and np.array_equal(c['target_ids'],rk['target_ids'])
    assert np.array_equal(c['query_gene_ids'],rk['query_ids'])
    assert np.array_equal(np.sort(s['NTC_input']),np.sort(ws['input_control_ids']));assert np.array_equal(s['input_cells'],ws['input_cell_ids'])
    ntc=s['NTC_input'];tr=s['input_cells'][:,0];cn=role(d,'NTC_input',ntc);tx=role(d,'A_input',tr.ravel());raw=vstack([cn,tx],format='csr');ids=np.r_[ntc,tr.ravel()]
    tot=np.asarray(raw.sum(1)).ravel();sp=np.searchsorted(s['all_selected_rows'],ids);assert np.array_equal(s['all_selected_rows'][sp],ids);assert np.array_equal(s['selected_total_counts'][sp],tot)
    assert np.array_equal(np.flatnonzero(np.asarray(cn.sum(0)).ravel()>0),c['keep'])
    cc=np.log1p(cn[:,c['keep']].toarray().astype(float)*(float(c['scale'])/tot[:len(ntc)])[:,None]);cv=cc.var(0,ddof=1);cm=cc.mean(0)
    # Historical D3 stream accumulated in float32; the archived fixed control
    # center remains the method center for exact R20 reproduction.
    cmerr=float(np.max(np.abs(cm-c['cm_keep'])));rel=float(np.linalg.norm(cm-c['cm_keep'])/np.linalg.norm(c['cm_keep']));assert rel<1e-3
    del cc
    n=len(tr);g=len(c['keep']);effects=np.empty((n,g));se=np.empty((n,g));target_var=np.empty((n,g));m2=np.empty(n);vb=np.empty(n);pm=np.empty((n,5));pv=np.empty((n,5))
    for j in range(n):
        dense=tx[j*20:(j+1)*20,c['keep']].toarray().astype(float);totj=tot[len(ntc)+j*20:len(ntc)+(j+1)*20]
        x=np.log1p(dense*(float(c['scale'])/totj)[:,None]);y=x.mean(0)-c['cm_keep'];v=x.var(0,ddof=1);ps=x@c['U'][:,:5]
        m2[j]=np.mean(y*y);vb[j]=v.mean();pm[j]=ps.mean(0)-c['proj_control'][:5];pv[j]=ps.var(0,ddof=1)
        effects[j]=y;se[j]=np.sqrt(v/20+cv/2000);target_var[j]=v
    errors={}
    for name,val in [('log_m2',m2),('log_vb',vb),('log_projection_mean',pm),('log_projection_variance',pv)]:
        err=float(np.max(np.abs(val-ws[name][:,0])));errors[name]=err;assert err<1e-10,(name,err)
    def weight(beta):
        energy=m2-beta*np.sum(pm*pm,axis=-1)/g;noise=vb/20+float(c['trC'])/(g*2000)-beta*np.sum(pv/20+c['lam'][:5]/2000,axis=-1)/g
        return np.clip(1-noise/np.maximum(energy,1e-300),0,1)
    we=weight(0);wm=weight(.9);fwe=ws['recorded_weights'][1,0];fwm=ws['recorded_weights'][2,0]
    errors['weights_euclidean']=float(np.max(np.abs(we-fwe)));errors['weights_pertrank']=float(np.max(np.abs(wm-fwm)));assert max(errors.values())<1e-10
    y=effects[:,c['qu']];support_error=0
    for j in range(len(c['qu'])):
        lo,hi=rk['Y_offsets'][j:j+2];ix=rk['Y_ids'][lo:hi];val=rk['Y_values'][lo:hi]
        support_error=max(support_error,float(np.max(np.abs(y[ix,j]-val))))
    assert support_error<1e-12;errors['saved_Y_support']=support_error
    f={'original_effects':y[None,:,:],'weight_Euclidean':fwe[None,:],'weight_PertRank':fwm[None,:],'truth':c['truth'].T,'target_ids':c['target_ids'],'target_names':c['internal_names'],'query_gene_ids':c['query_gene_ids'],'gene_ids_full':s['gene_ids_full'],'responsive_positions':c['responsive_positions'],'blocks_responsive':c['blocks_responsive'],'tercile_positions':c['tercile_positions'],'blocks_terciles':c['blocks_terciles']}
    np.savez_compressed(d/'frozen_task.npz',**f)
    out=ROOT/'inputs/D3_Rest/rep0';out.mkdir(parents=True,exist_ok=True);save_npz(out/'counts.npz',raw)
    pos=c['responsive_positions'];columns=c['keep'][c['qu'][pos]]
    np.savez_compressed(out/'axes.npz',labels=np.r_[np.full(len(ntc),-1,dtype=np.int32),np.repeat(np.arange(n,dtype=np.int32),20)],gene_ids=s['gene_ids_full'],target_ids=c['target_ids'],target_names=c['internal_names'],query_ids=c['query_gene_ids'][pos],query_blocks=c['blocks_responsive'],query_columns=columns,input_effects=y[:,pos],weights_euclidean=fwe,weights_pertrank=fwm,cell_ids=ids,reference_effects=c['truth'].T[:,pos],rep=np.array(0),task=np.array('D3_Rest'),scale=c['scale'],input_control_count=np.array(len(ntc)),target_cells_per_group=np.array(20),all_query_ids=c['query_gene_ids'],all_query_columns=c['keep'][c['qu']],all_input_effects=y,all_reference_effects=c['truth'].T,responsive_positions=pos,tercile_positions=c['tercile_positions'],blocks_terciles=c['blocks_terciles'])
    np.savez_compressed(out/'full_effect_stats.npz',effects=effects,standard_errors=se,target_var=target_var,gene_ids=c['gene_ids_keep'],target_ids=c['target_ids'],query_keep_columns=c['qu'][pos],control_mean=c['cm_keep'],control_mean_raw=cm,control_var=cv,scale=c['scale'],n_control=np.array(2000),n_target=np.array(20),variance_ddof=np.array(1))
    report={'status':'EXACT_RAW_R20_REPLAY_PASS','task':'D3_Rest','rep':0,'targets':n,'cells':len(ids),'control_count':len(ntc),'genes':raw.shape[1],'weight_genes':g,'queries':len(pos),'raw_totals_exact':True,'frozen_moment_and_support_errors':errors,'historical_control_float32_accumulation':{'max_abs_error':cmerr,'relative_l2_error':rel,'tolerance_relative':1e-3,'frozen_control_center_used':True},'source_selection_sha256':sha(d/'selection.npz'),'files_sha256':{p.name:sha(p)for p in out.glob('*.npz')},'seconds':time.monotonic()-t,'note':'R20 real-cell group0 on opened development data; exact archived weights used after independent numerical replay.'}
    report['standard_errors']={'min':float(se.min()),'max':float(se.max()),'zero_count':int(np.count_nonzero(se==0)),'ddof':1,'target_n':20,'control_n':2000}
    (out/'INPUT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
