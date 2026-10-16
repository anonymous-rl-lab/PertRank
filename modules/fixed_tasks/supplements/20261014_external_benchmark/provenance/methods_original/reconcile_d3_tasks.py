#!/usr/bin/env python3
"""Read-only D3 task reconciliation from restored frozen inputs; no model fitting."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
CORE = ROOT / 'archives/PertRank_D1_D4_GitHub_Core_v1.2.0/PertRank_D1_D4_GitHub_Core'
R20 = ROOT / 'archives/R20_Frozen_Design_20261006/research_round_20261006/new_experiment'
DATA = ROOT / 'pooled_data/D3_Rest'

def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def main():
    task=load(DATA/'frozen_task.npz');ctx=load(DATA/'r20_context.npz')
    selection=load(DATA/'selection.npz');rsel=load(R20/'selection/D3_Rest/selection.npz')
    rank=load(CORE/'data/D3/ranking.npz');ws=load(CORE/'data/D3/weight_stats.npz')
    ref=load(CORE/'reference_results/D3_query_scores.npz')
    compact_audit=json.loads((CORE/'data/D3/RANKING_COMPACTION.json').read_text())
    checks={}
    def exact(label,a,b):
        checks[label]={'pass':bool(np.array_equal(a,b)), 'shape':list(np.shape(a))}
        assert checks[label]['pass'],label
    exact('candidate_ids_task_context',task['target_ids'],ctx['target_ids'])
    exact('candidate_ids_task_core',task['target_ids'],rank['target_ids'])
    exact('candidate_ids_task_weight_statistics',task['target_ids'],ws['target_ids'])
    exact('union_query_ids_task_core',task['query_gene_ids'],rank['query_ids'])
    exact('union_query_ids_task_context',task['query_gene_ids'],ctx['query_gene_ids'])
    exact('full_gene_axis_task_context',task['gene_ids_full'],ctx['gene_ids_full'])
    exact('full_gene_axis_task_selection',task['gene_ids_full'],selection['gene_ids_full'])
    exact('weight_gene_axis_context_statistics',ctx['gene_ids_keep'],ws['gene_ids'])
    for key in ('responsive_positions','tercile_positions','blocks_responsive','blocks_terciles'):
        exact(key+'_task_core',task[key],rank[key])
        exact(key+'_task_context',task[key],ctx[key])
    exact('input_cells_selection_original_r20',selection['input_cells'],rsel['rows'][rsel['internal_positions']])
    exact('input_cells_selection_core',selection['input_cells'],ws['input_cell_ids'])
    exact('input_controls_same_set',np.sort(selection['NTC_input']),np.sort(ws['input_control_ids']))
    exact('rep0_target_cells',selection['A_input'],selection['input_cells'][:,0])
    exact('reference_full_effects',task['truth'].T,ctx['truth'])
    checks['r20_context_file_identical']={'pass':sha(DATA/'r20_context.npz')==sha(R20/'selection/D3_Rest/context.npz')}
    assert checks['r20_context_file_identical']['pass']
    checks['reference_full_content_matches_core_compaction_provenance']={
        'pass':array_sha(ctx['truth'])==compact_audit['full_source_truth_content_sha256'],
        'array_sha256':array_sha(ctx['truth'])}
    assert checks['reference_full_content_matches_core_compaction_provenance']['pass']
    assert task['original_effects'].shape[0]==1 and ws['recorded_weights'].shape[1]==5
    weights_mod=module('reconciliation_weight_functions',CORE/'src/reproduce_development_weights.py')
    for arm,beta in [('Euclidean',0),('PertRank',.9)]:
        ai=list(rank['arm_names']).index(arm)
        recomputed=weights_mod.metric_weights(ws,beta)
        exact(arm+'_all_five_weights_from_original_statistics',recomputed,rank['weights'][ai])
        exact(arm+'_rep0_weight_task_core',task['weight_'+arm][0],rank['weights'][ai,0])
    sys.path.insert(0,str(CORE/'src/vendor'))
    replay=module('reconciliation_original_compact_scorer',CORE/'src/replay_scores.py')
    historical=module('reconciliation_r20_scorer',R20/'evaluation/evaluate_real_cells.py')
    new=module('reconciliation_new_scorer',ROOT/'evaluation/evaluate_six_arm.py')
    arms=['raw','Euclidean','PertRank'];armix=[list(rank['arm_names']).index(a) for a in arms]
    pos=task['responsive_positions'];nt=len(task['target_ids']);nq=len(rank['query_ids'])
    # Reconstruct the original P@25 memberships only. The compact file does not
    # promise valid arbitrary interior effects, K values, or new method weights.
    truth_membership=replay.truth_membership(rank)
    scores=np.empty((3,5,300,2),dtype=np.float64)
    for rep in range(5):
        for qi,q in enumerate(pos):
            row=rep*nq+q;lo,hi=rank['Y_offsets'][row:row+2]
            ids=rank['Y_ids'][lo:hi];values=rank['Y_values'][lo:hi]
            for ai,oldai in enumerate(armix):
                p=replay.signed_membership(ids,values*rank['weights'][oldai,rep,ids],nt)
                scores[ai,rep,qi]=(p*truth_membership[:,q]).sum(1)/25
    exact('all_five_three_arm_query_direction_scores_vs_archived',scores,ref['responsive_scores'][armix])
    # Original full rep0 effects differ from archived support by floating-point
    # reconstruction noise. Verify this explicitly and verify its score impact.
    support_error=0.
    for q in range(nq):
        lo,hi=rank['Y_offsets'][q:q+2];ids=rank['Y_ids'][lo:hi]
        support_error=max(support_error,float(np.max(np.abs(task['original_effects'][0,ids,q]-rank['Y_values'][lo:hi]))))
    checks['rep0_effect_support']={'pass':support_error<1e-12,'maximum_absolute_error':support_error,'bitwise_identical':support_error==0}
    assert checks['rep0_effect_support']['pass']
    own=historical.own_mask(task['query_gene_ids'][pos],task['target_ids'])
    own_index=np.where(own.any(1),own.argmax(1),-1)
    exact('self_target_exclusion_indices',own_index,rank['own_index'][pos])
    pred=task['original_effects'][0][:,pos].T
    three=np.stack([pred,pred*task['weight_Euclidean'][0],pred*task['weight_PertRank'][0]])
    full_rep0=np.stack([historical.score(y,task['truth'][:,pos].T,own,25) for y in three])
    exact('rep0_full_effect_original_scorer_vs_compact_scores',full_rep0,scores[:,0])
    native_membership=new.old.probabilities(three,own,25)
    native_truth=new.old.probabilities(task['truth'][:,pos].T[None],own,25)
    native_scores=new.old.query_scores(native_membership,native_truth,25)[0]
    diff=float(np.max(np.abs(native_scores-scores[:,0].mean(-1))))
    checks['native_benchmark_scorer_rep0_vs_original']={'pass':diff<1e-14,'max_abs_query_score_error':diff,'reason_for_nonbitwise_arithmetic_if_any':'einsum sums signed overlaps then divides by 2K; original sums per sign then averages'}
    assert checks['native_benchmark_scorer_rep0_vs_original']['pass']
    means=scores.mean((2,3));avg=scores.mean(1).mean((1,2))
    paths=[Path(__file__),DATA/'frozen_task.npz',DATA/'selection.npz',DATA/'selection.lock.json',DATA/'r20_context.npz',ROOT/'inputs/D3_Rest/rep0/INPUT_AUDIT.json',ROOT/'inputs/D3_Rest/rep0/axes.npz',ROOT/'pooled_data/export_d3.py',ROOT/'evaluation/evaluate_six_arm.py',ROOT/'independent/execution/evaluate_independent.py',R20/'selection/D3_Rest/context.npz',R20/'selection/D3_Rest/selection.npz',R20/'selection/D3_Rest/selection_report.json',R20/'reader/read_real_cells.py',R20/'reader/make_selection.py',R20/'evaluation/evaluate_real_cells.py',CORE/'data/D3/ranking.npz',CORE/'data/D3/weight_stats.npz',CORE/'data/D3/RANKING_COMPACTION.json',CORE/'reference_results/D3_query_scores.npz',CORE/'src/replay_scores.py',CORE/'src/reproduce_development_weights.py']
    report={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),'scope':'D3 Rest only; archived input/statistic/ranking reconciliation, no model refit, no raw expression reread, no D1/D2 Stim access','conclusion':'For raw, Euclidean and PertRank P@25, the numerical discrepancy is explained by using original input repetition 0 rather than the equal average of all five original disjoint 20-cell input repetitions. Candidate/query/weight-gene axes, controls, reference, weights and scoring rule are unchanged. Tiny effect reconstruction and floating-point scoring-order differences were measured and do not materially alter scores.','task':{'targets':nt,'responsive_queries':len(pos),'query_blocks':int(len(np.unique(task['blocks_responsive']))),'union_queries':nq,'full_gene_axis':len(task['gene_ids_full']),'weight_genes':len(ctx['gene_ids_keep']),'controls':len(selection['NTC_input']),'target_cells_per_input':20,'original_input_repetitions':5,'native_development_input_repetitions':[0],'normalization':'log1p(count * 8261.5 / cell_total), fixed historical control centering for the three scalar arms','scale':float(ctx['scale']),'reference':'Same archived 100-cell target mean minus archived reference-control mean; full reference content hash matches original Core compaction provenance','reference_cell_role':'original deterministic target positions [200:300], from frozen R20 selection code; input cells are [100:200] split into five groups','input_rep0_cell_role':'original deterministic target positions [100:120]','weight_parameters':{'directions':5,'beta':.9},'self_target_exclusion':True,'ties':'exact independent boundary membership in each signed list','P25':'mean of positive/negative top-25 reference overlap, then 300 queries; main-text result also averages five input groups'},'scores':{a:{'per_rep_P25':means[i].tolist(),'five_rep_average_P25':float(avg[i]),'native_benchmark_rep0_P25':float(native_scores[i].mean()),'five_rep_minus_rep0':float(avg[i]-native_scores[i].mean())}for i,a in enumerate(arms)},'checks':checks,'limitations':['This reconciliation covers the three common scalar arms and their P@25 point estimates. Native FR uses its own official preprocessing (including TP10K), so identical method preprocessing is not claimed.','The native FR fitting axis can differ from the 13,390-gene PertRank weighting axis; this does not explain the scalar-arm score discrepancy.','Five-group scores were recomputed from the original lossless fixed-arm top-25 compact representation, not by fetching all raw cells again.','Historical and new bootstrap implementations/seeds differ; equality of confidence intervals is not claimed.','No manuscript content was edited.'], 'sources':{str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size}for p in paths}}
    out=OUT/'D3_TASK_RECONCILIATION.json';out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    np.savez_compressed(OUT/'D3_TASK_RECONCILIATION_SCORES.npz',arm_names=np.array(arms),responsive_query_ids=task['query_gene_ids'][pos],responsive_query_blocks=task['blocks_responsive'],five_rep_query_direction_scores=scores,native_rep0_query_scores=native_scores)
    methods=('For native external-method development, we used input repetition 0 of the previously defined D3 Rest pooled-guide task (the first 20 cells within each target\'s reserved 100-cell input block; deterministic target-cell positions 100–119). The main five-input analysis used all five mutually disjoint 20-cell groups from that same block. Both analyses retained the same 1,200 candidates, 300 responsive queries and 30 query blocks, 2,000 input non-targeting controls, 13,390-gene weighting axis, control-derived directions, and 100-cell-per-target deep reference. For unweighted, Euclidean and PertRank scoring, the original log normalization scale (8,261.5 counts), historical control center and fixed weighting parameters were retained. We excluded each query\'s own target and used the same exact boundary-tie expectation for positive and negative top-25 reference overlap. Replaying all five archived input groups gave P@25 values of '+', '.join(f'{float(avg[i]):.10f}' for i in range(3))+' for unweighted, Euclidean and PertRank ranking, respectively; restricting to input repetition 0 gave '+', '.join(f'{float(native_scores[i].mean()):.10f}' for i in range(3))+'. Thus, the 0.31310 versus 0.30515 PertRank values reflect five-input averaging versus a single development input, rather than a change in candidate pool, query panel or reference. Native FR retained its official method-specific preprocessing; the equivalence stated here concerns the shared task and the three scalar arms, not identical preprocessing across methods.\n')
    (OUT/'D3_TASK_RECONCILIATION_Methods.txt').write_text(methods)
    print(json.dumps({'status':report['status'],'checks':len(checks),'scores':report['scores'],'effect_support_max_error':support_error,'native_score_max_error':diff,'report':str(out)},indent=2))

if __name__=='__main__':main()
