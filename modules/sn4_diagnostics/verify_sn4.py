"""Recompute SN4.2–SN4.4 summaries from original target-level tables.

No raw-cell refitting or new analysis selection. Requires numpy and scipy.
Run: python verify_sn4.py
"""
from pathlib import Path
import csv,gzip,hashlib,io,json
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent
def load_table(name):
    data=gzip.decompress((ROOT/(name+'.gz')).read_bytes())
    expected=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())[name]['sha256_uncompressed']
    assert hashlib.sha256(data).hexdigest()==expected
    return list(csv.DictReader(io.StringIO(data.decode())))
def arr(rows,key):return np.array([float(r[key]) if r[key] else np.nan for r in rows])
def bins(rows):
    active=np.flatnonzero((arr(rows,'w0y')>0)|(arr(rows,'W1')>0))
    ordered=active[np.argsort(arr(rows,'d')[active],kind='stable')]
    return np.array_split(ordered,5)
def ci(num,den,idx):
    rng=np.random.default_rng(2026);v=[]
    for _ in range(2000):
        pick=idx[rng.integers(0,len(idx),len(idx))];v.append(num[pick].sum()/den[pick].sum())
    return np.percentile(v,[2.5,97.5]).tolist()

a9=load_table('w1_target_explanation_targets.csv');a10=load_table('w1_direction_replication_targets.csv')
reference=json.loads((ROOT/'a10_archived_summary.json').read_text())
assert hashlib.sha256((ROOT/'replicate_w1_directions.py').read_bytes()).hexdigest()==reference['integrity']['script_sha256']
out={'scope':'Reaggregation of the original target tables; original model predictions and candidate lists are unchanged.','sources':{}}
max_diff=0.0
for donor in ['D3','D1','D2']:
    r=[x for x in a9 if x['source']==donor];q=[x for x in a10 if x['source']==donor]
    assert [x['target'] for x in r]==[x['target'] for x in q]
    grouped=bins(r);expected_bin=np.zeros(len(r),int)
    for i,idx in enumerate(grouped,1):expected_bin[idx]=i
    assert np.array_equal(expected_bin,arr(q,'d_bin'))
    rates=[]
    for idx in grouped:
        add=arr(r,'own_add')[idx].sum();supported=arr(r,'own_add_sup')[idx].sum()
        rates.append({'n':len(idx),'additions':float(add),'supported_additions':float(supported),'support_rate':float(supported/add)})
    removals={}
    for pool in ['own','D3','D1','D2']:
        rem=arr(r,pool+'_rem');total=np.nansum(rem)
        if total>0:removals[pool]=float(np.nansum(rem[grouped[0]])/total)
    # Recompute primary and secondary ratios and target-bootstrap intervals.
    summaries={}
    for label,idx in [('bin 1',grouped[0]),('all targets',np.arange(len(q)))]:
        den=arr(q,'A0');num=arr(q,'C');other=arr(q,'A1');ref=reference['donors'][donor]['groups'][label]
        vals={'r_C_over_A0':float(num[idx].sum()/den[idx].sum()),'A1_over_A0':float(other[idx].sum()/den[idx].sum()),'r_C_over_A0_ci95':ci(num,den,idx),'A1_over_A0_ci95':ci(other,den,idx)}
        for k,v in vals.items():
            diff=float(np.max(np.abs(np.asarray(v)-np.asarray(ref[k]))));max_diff=max(max_diff,diff);assert diff<1e-12,(donor,label,k,diff)
        summaries[label]=vals
    ref=reference['donors'][donor]['groups']['bin 1'];first=ref['per_direction'][0]
    share=(first['sum_A0']-first['sum_C'])/(ref['sum_A0']-ref['sum_C'])
    out['sources'][donor]={'all_targets':len(r),'active_targets':sum(map(len,grouped)),'spearman_d_vs_w0y':float(spearmanr(arr(r,'d'),arr(r,'w0y')).statistic),'own_addition_support_by_bin':rates,'lowest_bin_removal_share':removals,'a10':summaries,'direction1_share_of_lowest_bin_reduction':share}
for d in ['D3','D1']:assert np.all(np.diff([r['support_rate'] for r in out['sources'][d]['own_addition_support_by_bin']])>0)
rates=out['sources']['D2']['own_addition_support_by_bin'];assert rates[3]['support_rate']>rates[4]['support_rate']
out['max_absolute_difference_from_archived_A10_point_estimates_and_intervals']=max_diff
out['status']='passed'
(ROOT/'verification_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':'passed','matched_original_target_rows':len(a9),'max_A10_difference':max_diff,'spearman':[out['sources'][d]['spearman_d_vs_w0y'] for d in ['D3','D1','D2']]},indent=2))
