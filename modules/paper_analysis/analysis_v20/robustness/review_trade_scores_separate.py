"""Separately written sort-based tie-expectation and block-ratio verification.

Uses no functions from the production scoring implementation.
"""
from pathlib import Path
import csv,json,hashlib
import numpy as np
P=Path(__file__).resolve().parent
z=np.load(P/'TRADE_SENSITIVITY_REPLAY_INPUTS.npz')
saved=np.load(P/'TRADE_SENSITIVITY_SCORES.npz')
expected=json.loads((P/'TRADE_SENSITIVITY_RESULT.json').read_text())
y=z['effects'];ref=z['reference'];targets=z['target_ids'].astype(str);queries=z['query_gene_ids'].astype(str)
K=25

def sorted_membership(values,eligible):
    ids=np.flatnonzero(eligible)
    order=ids[np.argsort(values[ids],kind='stable')[::-1]]
    boundary=values[order[K-1]]
    better=ids[values[ids]>boundary]
    tied=ids[values[ids]==boundary]
    p=np.zeros(len(values));p[better]=1.;p[tied]=(K-len(better))/len(tied)
    assert np.isclose(p.sum(),K,atol=1e-12) and not p[~eligible].any()
    return p

out=np.empty((2,5,len(queries),2));arms=['TRADE_scalar','PertRank']
for qi,qid in enumerate(queries):
    eligible=targets!=qid
    for di,sign in enumerate([1.,-1.]):
        t=sorted_membership(sign*ref[:,qi],eligible)
        for ai,arm in enumerate(arms):
            for g in range(5):
                p=sorted_membership(sign*y[g,:,qi]*z['weight_'+arm][g],eligible)
                out[ai,g,qi,di]=np.dot(p,t)/K
report={'status':'PASS','implementation':'separate sort-based threshold/membership implementation; no production scoring imports','queries':len(queries),'groups':5,'directions':2,'scores_per_arm':int(out[0].size),'excluded_own_target_queries':int(sum(q in set(targets) for q in queries)),'score_maxabs':{},'panels':{}}
for ai,arm in enumerate(arms):
    si=list(saved['arm_names']).index(arm)
    err=float(abs(out[ai]-saved['scores'][si]).max());assert err<1e-14
    report['score_maxabs'][arm]=err
for panel,key in [('responsive','responsive_positions'),('terciles','tercile_positions')]:
    pos=z[key];means={arm:float(out[ai][:,pos,:].mean()) for ai,arm in enumerate(arms)}
    for arm,v in means.items():assert abs(v-expected['panels'][panel]['means'][arm])<1e-14
    d=(out[1][:,pos,:]-out[0][:,pos,:]).mean(axis=(0,2))
    blocks=z['blocks_'+panel];labels=np.unique(blocks)
    block_sums=np.array([d[blocks==b].sum() for b in labels]);block_counts=np.array([sum(blocks==b) for b in labels])
    rng=np.random.default_rng(2026);resamples=rng.choice(len(labels),size=(2000,len(labels)),replace=True)
    draws=block_sums[resamples].sum(1)/block_counts[resamples].sum(1)
    interval=np.quantile(draws,[.025,.975]);point=float(d.mean())
    e=expected['panels'][panel]['contrasts']['PertRank_minus_TRADE_scalar']
    err=max(abs(point-e['difference']),float(abs(interval-e['conditional_query_block_95_interval']).max()));assert err<1e-14
    report['panels'][panel]={'means':means,'difference':point,'conditional_95_interval':interval.tolist(),'bootstrap_nonpositive_count':int(sum(draws<=0)),'result_maxabs':err,'block_count':len(labels),'draws':2000,'seed':2026}
rows=list(csv.DictReader((P/'TRADE_SENSITIVITY_FIT_MOMENTS.csv').open()))
assert len(rows)==2805 and len({(r['target'],int(r['replicate'])) for r in rows})==2805
assert all(np.isfinite(float(r['loglik'])) for r in rows)
assert all(int(r['n_genes'])==13394 for r in rows)
report['fit_diagnostics']={'fit_rows':len(rows),'unique_target_group_pairs':len(rows),'all_loglik_finite':True,'all_gene_counts_13394':True}
report['source_hashes']={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['TRADE_SENSITIVITY_REPLAY_INPUTS.npz','TRADE_SENSITIVITY_SCORES.npz','TRADE_SENSITIVITY_RESULT.json']}
(P/'TRADE_SEPARATE_SCORE_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
