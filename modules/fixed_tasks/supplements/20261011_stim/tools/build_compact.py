#!/usr/bin/env python3
"""Retrospective exact fixed-method support compaction of a completed Stim run.

This builder needs the original full local experiment. The public replay does
not. No original input, statistic, definition, value or scientific source is
modified. Build audit verifies every original membership and scalar weight.
"""
from pathlib import Path
import argparse, ast, hashlib, json, shutil, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCES = ('frozen_engine.py', 'stats_frozen_exact_score.py', 'stats_frozen_summary.py',
           'stats_frozen_wild_cluster_t.py', 'stim_stats.py', 'stim_secondary.py')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def module_adapter(source):
    """Copy result orchestration; replace only effect-to-membership inputs.

    All numerical estimators, score summaries, bootstrap and regression remain
    byte-identical modules. This adapter accepts exactly reconstructed scores
    and contributions instead of allocating absent interior effect values.
    """
    text = source.read_text(); tree = ast.parse(text)
    functions = {x.name: ast.get_source_segment(text,x) for x in tree.body if isinstance(x,ast.FunctionDef)}
    f = functions['score_and_analyze']
    replacements = [
        ("def score_and_analyze(main, query, truth, unavailable, *, cross=None, alignment=None):",
         "def analyze(main, query, unavailable, *, cross=None, alignment=None):"),
        ("weights = weights_from_lock(main, unavailable)", "weights = main['weights']"),
        ("scores, names = stats.score_predictions(main['original_effects'], truth, weights,\n                                            main['query_gene_ids'], main['target_ids'])",
         "scores, names = main['scores'], main['arm_names']"),
        ("xp, xq, xt = cross['predictions'], cross['query'], cross['truth']", "xp, xq = cross['predictions'], cross['query']"),
        ("xw = weights_from_lock(xp, {}, crossguide=True)", "xw = xp['weights']"),
        ("xs, xn = stats.score_predictions(xp['original_effects'], xt, xw,\n                                         xp['query_gene_ids'], xp['target_ids'])",
         "xs, xn = xp['scores'], xp['arm_names']"),
        ("contributions = secondary.nomination_contributions(\n            main['original_effects'][:, :, rp], truth[:, rp], weights['PertRank'],\n            weights['Euclidean'], main['query_gene_ids'][rp], main['target_ids'])",
         "contributions = main['alignment_contributions']"),
    ]
    for before, after in replacements:
        assert f.count(before) == 1, before
        f = f.replace(before,after)
    out = ('"""Result orchestration adapted solely to consume exact reconstructed memberships.\n'
           'Numerical decision/bootstrap/regression implementations are unchanged.\n"""\n'
           'import numpy as np\nimport stim_stats as stats\nimport stim_secondary as secondary\n'
           'from stats_frozen_summary import all_contrasts\n\n' + functions['full_arm_table'] + '\n\n' + f + '\n')
    (ROOT/'compact_analysis.py').write_text(out)
    return [{'before':a,'after':b} for a,b in replacements]

def independent_sort(values, own):
    nt=len(values); ids=np.flatnonzero(~own); p=np.zeros((2,nt),np.float64)
    for si,sign in enumerate((1.,-1.)):
        s=sign*values[ids]; boundary=np.sort(s)[-25]
        above,tied=s>boundary,s==boundary
        p[si,ids[above]]=1.
        p[si,ids[tied]]=(25-int(above.sum()))/int(tied.sum())
    return p

def build_task(source, task, exact, support, engine):
    start=time.monotonic()
    pred_path=source/f'input/{task}_nontrade_predictions.npz'
    ref_path=source/f'evaluation/{task}_reference.npz'
    query_path=source/f'input/{task}_query_context.npz'
    pred, ref, query = [support.load_npz(p) for p in (pred_path,ref_path,query_path)]
    y,truth=pred['original_effects'],ref['effects']; ng,nt,nq=y.shape
    assert ng==5 and truth.shape==(nt,nq) and y.dtype==truth.dtype==np.float64
    names=tuple(a for a in ('raw','Euclidean','TRADE_scalar','PR_M','PertRank') if 'weight_'+a in pred)
    assert 'TRADE_scalar' not in names
    own=exact.own_mask(pred['query_gene_ids'],pred['target_ids'])
    tp=exact.probabilities(truth.T,own,25)
    ty_ids=[];ty_values=[];ty_offsets=[0]
    for q in range(nq):
        full=tp[[q,nq+q]]
        assert np.array_equal(full,independent_sort(truth[:,q],own[q]))
        ids=np.flatnonzero(np.any(full>0,axis=0)); vals=truth[ids,q]
        rebuilt=np.zeros((2,nt));rebuilt[:,ids]=support.probability_row(vals)
        assert np.array_equal(full,rebuilt)
        ty_ids.extend(ids);ty_values.extend(vals);ty_offsets.append(len(ty_ids))
    y_ids=[];y_values=[];y_offsets=[0]; row_counts=[]
    fullhash=hashlib.sha256();compacthash=hashlib.sha256()
    for group in range(5):
        fulls=[]
        for name in names:
            p=exact.probabilities((pred['weight_'+name][group,:,None]*y[group]).T,own,25)
            fulls.append(p);fullhash.update(p.tobytes())
        fulls=np.asarray(fulls)
        compact_ps=np.zeros_like(fulls)
        for q in range(nq):
            keep=np.any(fulls[:,[q,nq+q],:]>0,axis=(0,1)) | np.any(tp[[q,nq+q]]>0,axis=0)
            ids=np.flatnonzero(keep);values=y[group,ids,q]
            assert not np.any(own[q,ids])
            for ai,name in enumerate(names):
                v=values*pred['weight_'+name][group,ids]
                p=support.probability_row(v)
                compact_ps[ai,q,ids],compact_ps[ai,nq+q,ids]=p[0],p[1]
                assert np.array_equal(compact_ps[ai,[q,nq+q]],fulls[ai,[q,nq+q]])
                assert np.array_equal(fulls[ai,[q,nq+q]],independent_sort(pred['weight_'+name][group]*y[group,:,q],own[q]))
            y_ids.extend(ids);y_values.extend(values);y_offsets.append(len(y_ids));row_counts.append(len(ids))
        for p in compact_ps:compacthash.update(p.tobytes())
    assert fullhash.hexdigest()==compacthash.hexdigest()
    keys=('log_aE','log_nuE','log_aM','log_nuM')+(('pr_aM','pr_nuM') if 'PR_M' in names else ())
    moments={k:np.empty((5,nt),np.float64) for k in keys}
    moment_sources={}
    for j in range(nt):
        path=source/f'input/{task}/target_{j:04d}.npz'
        with np.load(path,allow_pickle=False) as z:
            assert str(z['target_id'])==str(pred['target_ids'][j])
            for k in keys:moments[k][:,j]=z[k]
        moment_sources[str(path.relative_to(source))]=sha(path)
    arrays={
        'Y_offsets':np.asarray(y_offsets,np.uint32),'Y_ids':np.asarray(y_ids,np.uint16),'Y_values':np.asarray(y_values,np.float64),
        'truth_offsets':np.asarray(ty_offsets,np.uint32),'truth_ids':np.asarray(ty_ids,np.uint16),'truth_values':np.asarray(ty_values,np.float64),
        'target_ids':pred['target_ids'],'query_gene_ids':pred['query_gene_ids'],
        'arm_names':np.asarray(names),'k':np.asarray(25),'groups':np.asarray(5),
        **{'moment_'+k:v for k,v in moments.items()}}
    reconstructed=support.reconstruct_weights(arrays)
    for name in names:assert np.array_equal(reconstructed[name],pred['weight_'+name])
    np.savez_compressed(ROOT/f'data/{task}_support.npz',**arrays)
    shutil.copy2(query_path,ROOT/f'data/{task}_query_context.npz')
    report={
        'status':'PASS','task':task,'original_effect_entries':int(y.size),'retained_effect_entries':len(y_values),
        'retained_fraction':len(y_values)/y.size,'support_size_bytes':(ROOT/f'data/{task}_support.npz').stat().st_size,
        'retained_candidates_percentiles':np.percentile(row_counts,[0,25,50,75,100]).tolist(),
        'all_original_weights_exact':True,'all_signed_memberships_exact':True,'all_independent_full_sort_checks_passed':True,
        'prediction_membership_values_verified':len(names)*5*2*nq*nt,
        'truth_membership_values_verified':2*nq*nt,
        'all_boundary_ties_preserved':True,'float_quantization':False,'original_inputs_modified':False,
        'prediction_membership_sha256':fullhash.hexdigest(),
        'truth_membership_sha256':hashlib.sha256(tp.tobytes()).hexdigest(),
        'original_array_digest':support.array_digest(pred),
        'original_truth_array_digest':support.array_digest(ref),
        'source_files':{str(p.relative_to(source)):sha(p) for p in (pred_path,ref_path,query_path)},
        'scalar_moment_source_files':moment_sources,'elapsed_seconds':time.monotonic()-start}
    dump(ROOT/f'audit/{task}_COMPACTION.json',report)
    print(json.dumps({k:report[k] for k in ['task','status','retained_fraction','support_size_bytes']}),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',required=True,type=Path);args=p.parse_args()
    source=args.source_run.resolve()
    for name in SOURCES:shutil.copy2(source/'src'/name,ROOT/'frozen'/name)
    sys.path.insert(0,str(ROOT/'frozen'))
    import compact_support as support
    import stats_frozen_exact_score as exact
    import frozen_engine as engine
    changes=module_adapter(source/'evaluation_investigator_directed/stim_evaluate.py')
    reports={task:build_task(source,task,exact,support,engine) for task in ('main','xguide')}
    for rel,dest in [
        ('input/alignment_pretruth.npz','data/alignment_pretruth.npz'),
        ('input/alignment_pretruth_report.json','data/alignment_pretruth_report.json'),
        ('input/crossguide_pretruth_report.json','data/crossguide_pretruth_report.json'),
        ('evaluation/registered_scores.npz','expected/registered_scores.npz'),
        ('evaluation/EVALUATION_REPORT.json','expected/EVALUATION_REPORT.json'),
        ('input/TRADE_NOT_EVALUABLE.json','audit/TRADE_NOT_EVALUABLE.json'),
        ('audit/TRADE_COMPARATOR_ADJUDICATION.json','audit/TRADE_COMPARATOR_ADJUDICATION.json'),
        ('audit/trade_failure_0008_0016/INDEPENDENT_TRADE_FAILURE_AUDIT.json','audit/INDEPENDENT_TRADE_FAILURE_AUDIT.json'),
        ('audit/INDEPENDENT_FINAL_EVALUATION_AUDIT.json','audit/ORIGINAL_FINAL_EVALUATION_AUDIT.json'),
        ('INVESTIGATOR_DIRECTED_EVALUATION_RELEASE.json','audit/INVESTIGATOR_DIRECTED_EVALUATION_RELEASE.json'),
        ('evaluation_investigator_directed/stim_evaluate.py','audit/original_stim_evaluate.py'),
    ]:shutil.copy2(source/rel,ROOT/dest)
    private_receipt=source/'registration/INVESTIGATOR_DIRECTED_EVALUATION_RECEIPT.json'
    receipt=json.loads(private_receipt.read_text())
    receipt['browser_observations'].pop('prepared_draft',None)
    receipt['public_export']={
        'original_sha256':sha(private_receipt),
        'redacted_fields':['browser_observations.prepared_draft'],
        'reason':'Private registration draft URL is unnecessary for result reproduction; scientific and procedural assertions are unchanged.'}
    dump(ROOT/'audit/INVESTIGATOR_DIRECTED_EVALUATION_RECEIPT_PUBLIC.json',receipt)
    original=json.loads((source/'output/prediction_lock_export/PREDICTION_LOCK_MANIFEST.json').read_text())
    # Only the fixed unavailable-arm declaration is needed by numerical replay.
    dump(ROOT/'data/unavailable_arms.json',original['unavailable_arms'])
    dump(ROOT/'audit/COMPACTION_BUILD.json',{
        'status':'PASS','identity':'post-result exact fixed-method top-25 sufficient input compaction',
        'prediction_lock_zip_sha256':'8f6f231006f2a93773f6c092466458014a6b443ff38b951b9732579b448902d1',
        'scientific_source_files':{name:sha(source/'src'/name) for name in SOURCES},
        'original_evaluator_sha256':sha(source/'evaluation_investigator_directed/stim_evaluate.py'),
        'orchestration_replacements_only':changes,
        'tasks':{task:{k:v for k,v in r.items() if k!='scalar_moment_source_files'} for task,r in reports.items()},
        'prospective_lock_replaced':False,'network_used':False,'raw_cells_reread':False})

if __name__=='__main__':main()
