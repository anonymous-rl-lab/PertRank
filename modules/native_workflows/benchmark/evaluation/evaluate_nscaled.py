#!/usr/bin/env python3
"""Prespecified FR effective-penalty sensitivity; the primary scorer is unchanged.

Same native fitter, fixed D3 winning rank, matched cells, frozen weights and
scoring/bootstrap functions. No sensitivity result selects the primary model
or replaces primary resource gates. All sensitivity intervals are descriptive.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import evaluate_six_arm as e


def policy(args):
    amendment = json.loads(args.amendment.read_text())
    if amendment['primary_protocol_sha256'] != e.sha(args.protocol):
        raise ValueError('Sensitivity amendment does not match primary FR protocol')
    winner = e.check_lock(args.winner_lock, args.protocol)
    return amendment, winner


def evaluate(args):
    amendment, winner = policy(args)
    if args.dataset not in ('D4_Rest','D4_Stim8hr'):
        raise ValueError('N-scaled sensitivity is restricted to frozen D4 tasks')
    protocol = json.loads(args.protocol.read_text())
    primary = json.loads(args.primary_report.read_text())
    if primary['status'] != 'COMPLETE' or primary['dataset'] != args.dataset or primary['kind'] != args.kind:
        raise ValueError('Matching completed primary task report is required')
    if primary.get('parameter_policy') == 'N_scaled_effective_penalty':
        raise ValueError('Primary report cannot itself be a sensitivity report')
    if primary['provenance']['winner_lock_sha256'] != e.sha(args.winner_lock):
        raise ValueError('Primary report used a different D3 winner lock')
    primary_compact_path = args.primary_report.with_name('compact.npz')
    primary_arrays_path = args.primary_report.with_name('evaluation_arrays.npz')
    if e.sha(primary_compact_path) != primary['compact_sha256'] or e.sha(primary_arrays_path) != primary['evaluation_arrays_sha256']:
        raise ValueError('Primary task compact or arrays changed')
    baseline = e.load_npz(primary_compact_path)
    baseline_arrays = e.load_npz(primary_arrays_path)
    task_key = 'D4_Stim_independent' if args.kind == 'independent' else args.dataset
    reps = amendment['input_reps'][task_key]
    if len(args.fr_effects) != len(reps):
        raise ValueError('Provide exactly the frozen input repetitions in order')
    if args.kind == 'pooled':
        task_path = e.ROOT/'pooled_data'/args.dataset/'frozen_task.npz'
        selection_path = task_path.with_name('selection.npz')
        z, reference = e.assemble_pooled(task_path, reps)
        sources = dict(task_path=str(task_path.resolve()),task_sha256=e.sha(task_path))
    else:
        if args.dataset != 'D4_Stim8hr': raise ValueError('Independent task must be D4 Stim')
        source = e.ROOT/'independent/execution/D4_Stim8hr'
        prediction_path, reference_path = source/'prediction.npz',source/'reference.npz'
        selection_path = source.parent/'selection_D4_Stim8hr.npz'
        z, reference = e.assemble_independent(prediction_path, reference_path)
        sources = dict(prediction_path=str(prediction_path.resolve()),prediction_sha256=e.sha(prediction_path),
                       reference_path=str(reference_path.resolve()),reference_sha256=e.sha(reference_path))
    selection = e.load_npz(selection_path)
    ntc = len(selection['NTC_input'])
    target_depth = selection['input_cells'].shape[-1] if 'input_cells' in selection else selection['A_input'].shape[-1]
    total_cells = ntc + len(selection['target_ids'])*target_depth
    if total_cells != amendment['task_cells'][task_key] or ntc != amendment['n_control_all_tasks'] or target_depth != amendment['n_target_all_tasks']:
        raise ValueError('Actual task N/cell depths disagree with sensitivity amendment')
    config = next(g.copy() for g in protocol['fr_grid'] if g['id'] == winner['config_id'])
    nominal_alpha = float(config['lasso_alpha'])
    # Preserve operation order shared with the sensitivity native-fit runner.
    config['lasso_alpha'] = nominal_alpha*(amendment['N_D3']/amendment['n_D3_target'])/(total_cells/target_depth)
    fr, provenance = [], []
    for rep,path in zip(reps,args.fr_effects):
        effect, source = e.native_fr(path,z['candidates'],z['queries'],rep,config,protocol,winner=winner,selection=selection)
        fit = json.loads(path.with_name('fit_report.json').read_text())
        if fit['input']['n_cells'] != total_cells or fit['input']['n_ntc'] != ntc:
            raise ValueError('Native sensitivity fit N differs from prescribed N')
        fr.append(effect); provenance.append(source)
    z['fr_scores'] = np.stack(fr)
    e.validate_compact(z,args.dataset,args.kind)
    baseline_checks = {}
    for key in ('raw_scores','reference_scores','w0','wm','queries','candidates','blocks','input_reps','own_target'):
        baseline_checks[key] = bool(np.array_equal(z[key],baseline[key]))
        if not baseline_checks[key]: raise ValueError('Sensitivity changed frozen baseline '+key)
    args.out.mkdir(parents=True,exist_ok=False)
    begin = time.monotonic()
    results, arrays, rseed = e.score_arrays(z,draws=2000,reference=reference)
    for key,value in arrays.items():
        if key.startswith('six_arm_'):
            a,b = value[:,:3],baseline_arrays[key][:,:3]
        elif key.startswith('query_only_bootstrap_') or key.startswith('joint_bootstrap_'):
            a,b = value[:,:3],baseline_arrays[key][:,:3]
        else:
            a,b = value,baseline_arrays[key]
        if not np.array_equal(a,b):
            raise AssertionError('Sensitivity baseline or resampling differs bitwise: '+key)
        baseline_checks[key] = True
    for k in ('25','10'):
        for item in results[k]['contrasts'].values(): item['in_prespecified_five_contrast_family'] = False
    z['six_arm_p25'],z['six_arm_p10'] = arrays['six_arm_p25'],arrays['six_arm_p10']
    np.savez_compressed(args.out/'compact.npz',**z)
    np.savez_compressed(args.out/'evaluation_arrays.npz',**arrays)
    report = dict(status='COMPLETE',created_utc=e.now(),dataset=args.dataset,kind=args.kind,
        parameter_policy='N_scaled_effective_penalty',config_id=winner['config_id'],config=config,
        input_reps=reps,arms=e.ARMS,bootstrap_draws=2000,query_seed=e.SEED,reference_seed=rseed,
        reference_batch_size=16 if rseed else None,shape=primary['shape'],primary_K=25,sensitivity_K=10,
        nominal_D3_alpha=nominal_alpha,actual_task_cells=total_cells,actual_target_depth=target_depth,
        effective_target_threshold=config['lasso_alpha']*total_cells/target_depth,
        D3_target_threshold=nominal_alpha*amendment['N_D3']/amendment['n_D3_target'],
        results=results,elapsed_seconds=time.monotonic()-begin,baseline_bitwise_checks=baseline_checks,
        scope=primary['scope']+' Prespecified effective-penalty sensitivity, descriptive inference only.',
        simultaneous_intervals='No simultaneous significance claim for sensitivity. Saved 95% and 99% intervals are descriptive, and no correction across primary/sensitivity families is asserted.',
        replaces_primary_resource_gates=False,
        provenance={**sources,'selection_path':str(selection_path.resolve()),'selection_sha256':e.sha(selection_path),
            'protocol_path':str(args.protocol.resolve()),'protocol_sha256':e.sha(args.protocol),
            'sensitivity_amendment_path':str(args.amendment.resolve()),'sensitivity_amendment_sha256':e.sha(args.amendment),
            'evaluator_sha256':e.sha(e.__file__),'sensitivity_wrapper_sha256':e.sha(__file__),
            'score_primitive_sha256':e.sha(e.OLD_PATH),'fr':provenance,
            'winner_lock_path':str(args.winner_lock.resolve()),'winner_lock_sha256':e.sha(args.winner_lock),
            'primary_report_path':str(args.primary_report.resolve()),'primary_report_sha256':e.sha(args.primary_report)},
        compact_sha256=e.sha(args.out/'compact.npz'),evaluation_arrays_sha256=e.sha(args.out/'evaluation_arrays.npz'))
    e.save_json(args.out/'evaluation_report.json',report)
    print(json.dumps({'report':str(args.out/'evaluation_report.json'),'alpha':config['lasso_alpha'],'P25':results['25']}))


def combine(args):
    amendment,winner = policy(args)
    reports=[json.loads(p.read_text()) for p in args.reports]
    if any(r.get('parameter_policy')!='N_scaled_effective_penalty' or r['provenance']['sensitivity_amendment_sha256']!=e.sha(args.amendment) for r in reports):
        raise ValueError('Only two matching sensitivity reports can be combined here')
    e.combine(args)
    path=args.out/'equal_state_report.json'
    result=json.loads(path.read_text())
    result.update(parameter_policy='N_scaled_effective_penalty',sensitivity_amendment_sha256=e.sha(args.amendment),
        replaces_primary_resource_gates=False,development_resource_gates_are_binding=False,
        gate_note='Gate formulas reproduced descriptively for sensitivity; they do not replace or reverse the primary resource decision.',
        simultaneous_intervals='All sensitivity intervals descriptive; no additional simultaneous claim or correction across primary and sensitivity families.',
        sensitivity_wrapper_sha256=e.sha(__file__))
    for k in ('25','10'):
        for item in result['results'][k]['contrasts'].values(): item['in_prespecified_five_contrast_family']=False
    e.save_json(path,result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('evaluate')
    s.add_argument('--dataset',choices=['D4_Rest','D4_Stim8hr'],required=True)
    s.add_argument('--kind',choices=['pooled','independent'],default='pooled')
    s.add_argument('--primary-report',type=Path,required=True)
    s.add_argument('--fr-effects',nargs='+',type=Path,required=True)
    c=sub.add_parser('combine');c.add_argument('--reports',nargs='+',type=Path,required=True)
    for child in (s,c):
        child.add_argument('--protocol',type=Path,default=e.ROOT/'AMENDMENT_03_FR_SEARCH_21.json')
        child.add_argument('--amendment',type=Path,default=e.ROOT/'AMENDMENT_04_FR_EFFECTIVE_PENALTY.json')
        child.add_argument('--winner-lock',type=Path,default=e.ROOT/'FR_D3_WINNER_LOCK.json')
        child.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); {'evaluate':evaluate,'combine':combine}[a.command](a)


if __name__=='__main__':main()
