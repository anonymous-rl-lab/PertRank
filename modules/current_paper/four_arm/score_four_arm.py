#!/usr/bin/env python3
"""New fixed four-arm A19-R readout analysis on recovered sufficient statistics."""
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
from score_native import membership, ordered_indices

ROOT=Path(__file__).resolve().parent
SEED=2026100502
B=2000
K=10
PAIRS={
    'readout_original_minus_compressed_at_wM':('wM_y','wM_Ay'),
    'readout_original_minus_compressed_at_w0':('w0_y','w0_Ay'),
    'geometry_on_original':('wM_y','w0_y'),
    'geometry_on_compressed':('wM_Ay','w0_Ay'),
}

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out.resolve();assert not out.exists();out.mkdir(parents=True)
    ref=load(ROOT/'query_reference.npz')
    recovery=json.loads((ROOT/'four_arm_inputs/input_recovery_validation.json').read_text())
    assert recovery['status']=='PASS'
    reports={}
    for d in ('D3','D1','D2'):
        z=load(ROOT/f'predictions_{d}.npz'); pp=ROOT/f'four_arm_inputs/{d}_projection.npz'
        assert sha(pp)==recovery['sources'][d]['compact_projection_sha256']
        p=load(pp);q=z['query_ids'];t=z['target_ids'];blocks=z['query_blocks'];own=z['own_mask']
        assert np.array_equal(p['query_ids'],q) and np.array_equal(p['target_ids'],t)
        assert float(p['beta'])==.9 and int(p['k'])==5
        original=z['raw_effect']
        compressed=original-(1-np.sqrt(1-.9))*(p['U_query']@p['effect_projection'].T)
        predictions={'w0_y':original*z['w0y'],'wM_y':original*z['W1'],
                     'w0_Ay':compressed*z['w0y'],'wM_Ay':compressed*z['W1']}
        truth=ref['mean_effect'][np.ix_(ordered_indices(ref['query_ids'],q),ordered_indices(ref['target_symbols'],z['target_symbols']))]
        truth_p=membership(truth,own)
        probs={k:membership(v,own) for k,v in predictions.items()}
        for new,old in [('w0_y','w0y'),('wM_y','W1')]:
            assert np.array_equal(probs[new],z['top10_probability'][list(z['arm_names']).index(old)])
        scores={k:(v*truth_p).sum(1).reshape(2,len(q)).T/K for k,v in probs.items()}
        labels=np.unique(blocks);members=[np.flatnonzero(blocks==b) for b in labels]
        rng=np.random.default_rng(SEED);weights=np.zeros((B,len(q)))
        for i in range(B):
            ix=np.concatenate([members[j] for j in rng.integers(len(labels),size=len(labels))])
            weights[i]=np.bincount(ix,minlength=len(q))/len(ix)
        contrasts={};account={};contrast_arrays={}
        for name,(a,b) in PAIRS.items():
            per_query=(scores[a]-scores[b]).mean(1);point=float(per_query.mean())
            draws=weights@per_query;contrast_arrays[name]=per_query
            contrasts[name]={'P10_difference':point,'members_per_10_difference':K*point,
                'conditional_95_interval':np.quantile(draws,[.025,.975]).tolist(),
                'conditional_99_interval':np.quantile(draws,[.005,.995]).tolist()}
            delta=probs[a]-probs[b];added=np.maximum(delta,0);removed=np.maximum(-delta,0)
            assert np.allclose(added.sum(1),removed.sum(1),atol=1e-12,rtol=0)
            a_support=float((added*truth_p).sum(1).mean());r_support=float((removed*truth_p).sum(1).mean())
            assert abs(a_support-r_support-K*point)<1e-12
            account[name]={'mean_slots_moved':float(added.sum(1).mean()),'mean_reference_members_added':a_support,
                'mean_reference_members_removed':r_support,'net_members':a_support-r_support}
        interaction=contrast_arrays['geometry_on_original']-contrast_arrays['geometry_on_compressed']
        draws=weights@interaction
        contrasts['interaction_original_minus_compressed_geometry']={'P10_difference':float(interaction.mean()),
            'members_per_10_difference':K*float(interaction.mean()),'conditional_95_interval':np.quantile(draws,[.025,.975]).tolist(),
            'conditional_99_interval':np.quantile(draws,[.005,.995]).tolist()}
        report={'analysis_identity':'NEW_POST_HOC_FOUR_ARM_ON_PREVIOUSLY_ANALYZED_EXTERNAL_REFERENCE',
            'source':d+'_Rest','n_queries':len(q),'n_targets':len(t),'query_blocks':len(labels),'K':K,'input_cells':20,
            'beta':.9,'k':5,'bootstrap_seed':SEED,'bootstrap_draws':B,
            'means':{k:float(v.mean()) for k,v in scores.items()},'contrasts':contrasts,'nomination_accounting':account,
            'old_original_readout_memberships_exact':True,'reference_sensitivity':'Scored separately by score_reference_sensitivity.py; see four_arm_reference_sensitivity/reference_sensitivity.json in the completed pilot',
            'input_sha256':{str(x.relative_to(ROOT)):sha(x) for x in [ROOT/f'predictions_{d}.npz',pp,ROOT/'query_reference.npz']}}
        reports[d]=report
        (out/(d+'_report.json')).write_text(json.dumps(report,indent=2)+'\n')
        np.savez_compressed(out/(d+'_scores.npz'),arm_names=np.array(list(scores)),scores=np.stack(list(scores.values())),
            predictions=np.stack(list(predictions.values())),memberships=np.stack(list(probs.values())),reference_membership=truth_p,
            query_ids=q,target_ids=t,query_blocks=blocks,own_mask=own)
    summary={'sources':reports,'equal_source_descriptive_mean':{
        'means':{arm:float(np.mean([reports[d]['means'][arm] for d in reports])) for arm in next(iter(reports.values()))['means']},
        'contrasts':{c:float(np.mean([reports[d]['contrasts'][c]['P10_difference'] for d in reports])) for c in next(iter(reports.values()))['contrasts']}},
        'scope':'Three source inputs share one independent RNA experiment. Intervals are conditional query-block intervals; no pooled source-level CI or independent-confirmation claim. No model refits, query changes, or altered candidate pools.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'means':summary['equal_source_descriptive_mean'],'contrasts':{d:r['contrasts'] for d,r in reports.items()}},indent=2))

if __name__=='__main__':main()
