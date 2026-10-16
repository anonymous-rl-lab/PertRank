#!/usr/bin/env python3
"""Replay saved cell-bootstrap summaries and the completed native Mixscale scores.

No cell bootstrap, weight refit, native model fit or source-cell read is run.
"""
from pathlib import Path
import argparse, csv, json, time
import numpy as np

ROOT = Path(__file__).resolve().parent

def summary(x):
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)),
                empirical_q025_q975=np.quantile(x, [.025, .975]).tolist(),
                fraction_gt_zero=float(np.mean(x > 0)))

def check(actual, expected):
    for key, val in actual.items():
        np.testing.assert_allclose(val, expected[key], rtol=1e-12, atol=1e-12)

def membership(effect, own):
    nq, nt = effect.shape
    v = np.stack([effect, -effect], axis=1).reshape(2*nq, nt).copy()
    mask = np.repeat(np.arange(nq), 2)[:, None] == own[None, :]
    v[mask] = -np.inf
    cut = np.sort(v, axis=1)[:, -25, None]
    hi, tie = v > cut, v == cut
    p = hi.astype(float) + tie * ((25-hi.sum(1))/tie.sum(1))[:, None]
    assert np.allclose(p.sum(1), 25) and not np.any(p[mask])
    return p

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    a = ap.parse_args();out = a.output_dir.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True);start=time.monotonic()
    report = {'status':'RUNNING', 'scope':'Saved-draw summary replay and completed-coefficient scoring; no new experiment or refit.', 'draw_summaries':{}, 'mixscale':{}}
    try:
        for tag in ['conditional_bootstrap_D1','conditional_bootstrap_D2','conditional_bootstrap_D3','refit_D3']:
            expected=json.loads((ROOT/'data'/(tag+('_results.json' if tag=='refit_D3' else '.json'))).read_text())
            with np.load(ROOT/'data'/(tag+'_draws.npz'), allow_pickle=False) as z:
                assert z['mean_scores'].shape==(200,2,3)
                assert z['list_overlaps'].shape==(200,2,3)
                assert list(z['arms'])==['raw','w0y','W1']
                results={};checks=0
                for i,panel in enumerate(z['panels'].astype(str)):
                    source=expected['panels'][panel];r={}
                    for field,array in [('score_distributions','mean_scores'),('overlap_with_original_list','list_overlaps')]:
                        r[field]={}
                        for j,arm in enumerate(z['arms'].astype(str)):
                            value=summary(z[array][:,i,j]);check(value,source[field][arm]);checks+=4;r[field][arm]=value
                    r['comparison_distributions']={}
                    for j,arm in enumerate(['raw','w0y']):
                        value=summary(z['mean_scores'][:,i,2]-z['mean_scores'][:,i,j]);label='W1 - '+arm
                        check(value,source['comparison_distributions'][label]);checks+=4
                        assert abs(source['comparison_distributions'][label]['original_difference']-(source['original_scores']['W1']-source['original_scores'][arm]))<1e-12
                        r['comparison_distributions'][label]=value
                    results[panel]=r
                report['draw_summaries'][tag]={'status':'PASS','draws':200,'numeric_summary_fields_checked':checks,'panels':results,
                    'interpretation':'Empirical draw distribution, not donor-population confidence interval; original negative refit findings retained.'}
        with (ROOT/'data/paper_figS1_refit.csv').open() as f:
            rows=list(csv.DictReader(f))
        original_refit=json.loads((ROOT/'data/refit_D3_results.json').read_text())
        display_checks=0
        for row in rows:
            panel=row['panel'].lower();contrast='W1 - '+('w0y' if 'Euclidean' in row['contrast'] else 'raw')
            refit=report['draw_summaries']['refit_D3']['panels'][panel]['comparison_distributions'][contrast]
            values={'refit_mean':refit['mean'],'empirical_low':refit['empirical_q025_q975'][0],'empirical_high':refit['empirical_q025_q975'][1],
                    'original':original_refit['panels'][panel]['comparison_distributions'][contrast]['original_difference']}
            if row['fixed_weight_mean']:
                values['fixed_weight_mean']=report['draw_summaries']['conditional_bootstrap_D3']['panels'][panel]['comparison_distributions'][contrast]['mean']
            for key,value in values.items():
                assert abs(float(row[key])-value)<=0.000005,(panel,contrast,key,value)
                display_checks+=1
        report['paper_display']={'status':'PASS','source':'Supplementary Figure 1 archived display CSV','rows':len(rows),'rounded_values_checked':display_checks,'rounding_absolute_tolerance':0.000005}
        with np.load(ROOT/'data/mixscale_task.npz',allow_pickle=False) as c, np.load(ROOT/'data/mixscale_prediction_and_scores.npz',allow_pickle=False) as saved:
            expected=json.loads((ROOT/'data/mixscale_D3_results.json').read_text())
            np.testing.assert_array_equal(c['target_names'],saved['target_names'])
            own=c['own_query_index'];truth=membership(c['reference'],own)
            effects={'raw':c['raw_effect'],'w0y':c['raw_effect']*c['weight_Euclidean'],
                     'W1':c['raw_effect']*c['weight_PertRank'],'Mixscale_beta_weight':saved['coefficient']}
            scores={};errors={}
            for arm,effect in effects.items():
                sc=(membership(effect,own)*truth).sum(1).reshape(300,2)/25
                errors[arm]=float(np.max(abs(sc-saved['score_'+arm])))
                assert errors[arm]<1e-12,(arm,errors[arm]);scores[arm]=sc
                assert abs(sc.mean()-expected['methods_means'][arm])<1e-12
            tested=saved['tested'];coverage={'query_target_pairs_fitted':int(tested.sum()),'query_target_pairs_total':int(tested.size),'fraction_fitted':float(tested.mean()),'targets_with_no_query_returned':int((tested.sum(0)==0).sum())}
            assert coverage==expected['native_filter_coverage']
            assert np.all(saved['coefficient'][~tested]==0)
            labels=np.unique(c['blocks']);counts=np.array([(c['blocks']==b).sum() for b in labels]);draw=np.random.default_rng(2026).choice(len(labels),size=(2000,len(labels)),replace=True)
            contrasts={}
            for left,right in [('W1','Mixscale_beta_weight'),('w0y','Mixscale_beta_weight'),('Mixscale_beta_weight','raw')]:
                d=(scores[left]-scores[right]).mean(1);sums=np.array([d[c['blocks']==b].sum() for b in labels]);values=sums[draw].sum(1)/counts[draw].sum(1)
                got={'diff':float(d.mean()),'ci95':np.quantile(values,[.025,.975]).tolist(),'p_one_sided':float(np.mean(values<=0))}
                label=left+' - '+right;check(got,expected['comparisons'][label]);contrasts[label]=got
            np.savez_compressed(out/'mixscale_recomputed_scores.npz',**scores)
            report['mixscale']={'status':'PASS','targets':1200,'queries':300,'four_arm_score_values_checked':2400,'score_max_abs_errors':errors,'means':{k:float(v.mean()) for k,v in scores.items()},'coverage':coverage,'contrasts':contrasts,'scope':expected['status'],'native_refit_run':False,'query_bootstrap_draws':2000}
        report['status']='PASS'
    except Exception as exc:
        report['status']='FAIL';report['error']=repr(exc);raise
    finally:
        report['seconds']=time.monotonic()-start
        (out/'REPLAY_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'seconds':report['seconds'],'receipt':str(out/'REPLAY_REPORT.json')}))

if __name__=='__main__':main()
