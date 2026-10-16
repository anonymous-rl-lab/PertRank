#!/usr/bin/env python3
"""Fixed IL2 readout-score to comparison replay. Upstream raw-count parsing and guide-overlap classification are retained intermediates, not rerun."""
import argparse
import csv
import gzip
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DIRECTIONS = ['increasing_expression', 'decreasing_expression']
SIGN = {'increasing_expression': 1, 'decreasing_expression': -1}


def rc(seq):
    return seq.translate(str.maketrans('ACGT', 'TGCA'))[::-1]


def tokens(seq):
    return {s[i:i+19] for s in (seq, rc(seq)) for i in range(len(s)-18)}


def read_gz(name):
    return pd.read_csv(ROOT / 'data' / name, sep='\t', keep_default_na=False)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(x) for x in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results')
    parser.add_argument('--bootstrap', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20261011)
    args = parser.parse_args()
    start = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    archive = np.load(ROOT/'data/guide_readout_scores.npz', allow_pickle=False)
    ratios = archive['ratios'].copy()
    raw = pd.DataFrame({k:archive[k] for k in archive.files if k!='ratios'})
    mapping = pd.read_csv(ROOT/'data/candidate_mapping.tsv', sep='\t', keep_default_na=False)
    lists = read_gz('il2_all_nominations.tsv.gz')
    assert len(raw)==4330 and len(mapping)==561 and set(lists['query'])=={'IL2'}
    assert len(lists.groupby(['input_group','direction']))==10
    for _, block in lists.groupby(['input_group','direction']):
        assert block.euclidean_membership.sum()==25 and block.pertrank_membership.sum()==25
        assert set(block.membership_delta)<={-1.,0.,1.}
    libs=raw.library_set.to_numpy(); centers,ntc_ratios={},{}
    for lib in ['A','B']:
        ix=(libs==lib)&raw.ntc.to_numpy()&raw.both_donors_observed.to_numpy()
        ntc_ratios[lib]=ratios[ix];centers[lib]=np.median(ratios[ix],axis=0)
    centered=ratios-np.array([centers[lib] for lib in libs])
    raw['D1_ntc_centered_lfc']=centered[:,0];raw['D2_ntc_centered_lfc']=centered[:,1]
    label_indexes = {k: np.array(v) for k, v in raw.groupby('gene', sort=False).groups.items()}
    mapped = mapping.set_index('target_id')
    effects = []
    index_sets = {}
    for _, row in mapping.iterrows():
        indexes = label_indexes.get(row.external_design_label, np.array([], dtype=int))
        for mode, flag in [('nonoverlap', 'eligible_primary'), ('all_guides', 'eligible_all_guides')]:
            ix = indexes[raw.loc[indexes, flag].to_numpy(bool)]
            assert len({raw.loc[i, 'sequence'] for i in ix}) == len(ix), 'duplicate target sequences'
            index_sets[(row.target_id, mode)] = ix
            effect = np.median(centered[ix], axis=0) if len(ix) else np.array([np.nan, np.nan])
            effects.append(dict(target_id=row.target_id, gene_symbol=row.frozen_gene_symbol,
                                external_design_label=row.external_design_label, mode=mode,
                                n_guides=len(ix), n_A=int(sum(libs[ix] == 'A')), n_B=int(sum(libs[ix] == 'B')),
                                D1_lfc=effect[0], D2_lfc=effect[1], mean_lfc=np.mean(effect),
                                donor_sign_agreement=bool(effect[0]*effect[1] > 0) if len(ix) else None))
    effect_df = pd.DataFrame(effects)
    effect_df.to_csv(args.out/'target_external_effects.tsv', sep='\t', index=False)
    relevant_labels = set(mapping.external_design_label) | {'NO-TARGET'}
    raw[raw.gene.isin(relevant_labels)].to_csv(args.out/'candidate_and_control_guide_effects.tsv.gz',
                                             sep='\t', index=False, compression={'method':'gzip','mtime':0})
    list_rows, method_rows, occurrence_rows, scorecards = [], [], [], {}
    changed = lists[lists.membership_delta != 0].copy()
    for mode in ['nonoverlap', 'all_guides']:
        g = effect_df[effect_df['mode'] == mode].set_index('target_id')
        for min_guides in ([1, 2] if mode == 'nonoverlap' else [1]):
            analysis = f'{mode}_min{min_guides}'
            rows = lists.merge(g.reset_index()[['target_id','n_guides','D1_lfc','D2_lfc','mean_lfc']], on='target_id')
            rows['observed'] = rows.n_guides >= min_guides
            rows['sign'] = rows.direction.map(SIGN)
            rows['oriented_effect'] = rows.sign * rows.mean_lfc
            rows['binary_support'] = rows.observed & (rows.sign*rows.D1_lfc >= 0.5) & (rows.sign*rows.D2_lfc >= 0.5)
            rows['continuous_contribution'] = np.where(rows.observed, rows.membership_delta*rows.oriented_effect, 0)
            rows['binary_contribution'] = rows.membership_delta*rows.binary_support.astype(int)
            occurrence_rows.extend(rows.assign(analysis=analysis).to_dict('records'))
            for (group, direction), block in rows.groupby(['input_group','direction'], sort=True):
                observed = block[block.observed]
                missing = block[~block.observed]
                low = missing.loc[missing.membership_delta < 0, 'membership_delta'].sum()
                high = missing.loc[missing.membership_delta > 0, 'membership_delta'].sum()
                out = dict(analysis=analysis, input_group=int(group), direction=direction,
                           D_obs=block.continuous_contribution.sum()/25,
                           D1_obs=(observed.membership_delta*observed.sign*observed.D1_lfc).sum()/25,
                           D2_obs=(observed.membership_delta*observed.sign*observed.D2_lfc).sum()/25,
                           binary_net_supported=block.binary_contribution.sum(),
                           binary_unknown_lower=low, binary_unknown_upper=high)
                list_rows.append(out)
                for method, col in [('euclidean','euclidean_membership'), ('pertrank','pertrank_membership')]:
                    method_rows.append(dict(analysis=analysis, input_group=int(group), direction=direction, method=method,
                                            observed_nominations=(block[col]*block.observed).sum(),
                                            unknown_nominations=(block[col]*(~block.observed)).sum(),
                                            supported_nominations=(block[col]*block.binary_support).sum(),
                                            observed_oriented_lfc_sum=(observed[col]*observed.oriented_effect).sum()))
            ch = rows[rows.membership_delta != 0]
            donor = [(ch.loc[ch.observed,'membership_delta']*ch.loc[ch.observed,'sign']*
                      ch.loc[ch.observed,f'D{d}_lfc']).sum()/250 for d in (1,2)]
            missing = ch[~ch.observed]
            card = dict(D_obs=ch.continuous_contribution.sum()/250, donor_D_obs=donor,
                        direction_D_obs={direction: block.continuous_contribution.sum()/125 for direction,block in ch.groupby('direction')},
                        binary_net_supported_occurrences=float(ch.binary_contribution.sum()),
                        binary_net_supported_per_list=float(ch.binary_contribution.sum()/10),
                        binary_delta_support_fraction_observed=float(ch.binary_contribution.sum()/250),
                        binary_delta_support_fraction_lower=float((ch.binary_contribution.sum()+missing.loc[missing.membership_delta <0,'membership_delta'].sum())/250),
                        binary_delta_support_fraction_upper=float((ch.binary_contribution.sum()+missing.loc[missing.membership_delta >0,'membership_delta'].sum())/250),
                        changed_targets_observed=int(ch.loc[ch.observed,'target_id'].nunique()),
                        changed_targets_total=int(ch.target_id.nunique()),
                        swaps={kind:dict(total=len(block),observed=int(block.observed.sum()),
                                         supported=int(block.binary_support.sum()))for kind,block in ch.groupby('change')})
            contributions = ch.groupby(['target_id','gene_symbol']).continuous_contribution.sum()/250
            card['leave_one_target_out_D_obs_range'] = [float(card['D_obs']-contributions.max()),float(card['D_obs']-contributions.min())]
            card['largest_absolute_target_contributions'] = [dict(target_id=tid,gene_symbol=name,contribution=float(contributions.loc[(tid,name)]))
                         for tid,name in contributions.abs().sort_values(ascending=False).head(10).index]
            scorecards[analysis] = card
    pd.DataFrame(list_rows).to_csv(args.out/'list_comparisons.tsv',sep='\t',index=False)
    pd.DataFrame(method_rows).to_csv(args.out/'method_support_per_list.tsv',sep='\t',index=False)
    pd.DataFrame(occurrence_rows).to_csv(args.out/'nomination_evidence.tsv',sep='\t',index=False)
    # Paired guide bootstrap: preserve both donors and target/library strata.
    rng = np.random.default_rng(args.seed)
    B = args.bootstrap
    boot_centers = {}
    for lib in ['A','B']:
        nr = ntc_ratios[lib]
        bc = np.empty((B,2))
        for start_b in range(0,B,500):
            end_b = min(start_b+500,B)
            picks = rng.integers(0,len(nr),size=(end_b-start_b,len(nr)))
            bc[start_b:end_b] = np.median(nr[picks],axis=1)
        boot_centers[lib] = bc
    boot_d = np.zeros((B,2)); boot_binary = np.zeros(B)
    boot_d2 = np.zeros((B,2)); boot_binary2 = np.zeros(B)
    target_contributions = []
    for tid, ch in changed.groupby('target_id',sort=True):
        ix = index_sets[(tid,'nonoverlap')]
        if not len(ix):
            continue
        pieces=[]
        for lib in ['A','B']:
            lix = ix[libs[ix] == lib]
            if len(lix):
                picks = rng.integers(0,len(lix),size=(B,len(lix)))
                pieces.append(ratios[lix][picks]-boot_centers[lib][:,None,:])
        boot_effect = np.median(np.concatenate(pieces,axis=1),axis=1)
        coefficient = float((ch.membership_delta*ch.direction.map(SIGN)).sum())
        boot_d += coefficient*boot_effect/250
        cb = np.zeros(B)
        for direction, block in ch.groupby('direction'):
            coeff = float(block.membership_delta.sum())
            support = np.all(SIGN[direction]*boot_effect >= 0.5,axis=1)
            cb += coeff*support/250
        boot_binary += cb
        if len(ix) >= 2:
            boot_d2 += coefficient*boot_effect/250
            boot_binary2 += cb
        point = effect_df[(effect_df.target_id == tid)&(effect_df['mode'] == 'nonoverlap')].iloc[0]
        target_contributions.append(dict(target_id=tid,gene_symbol=point.gene_symbol,
            signed_occurrence_coefficient=coefficient,n_guides=len(ix),
            D_obs_contribution=coefficient*point.mean_lfc/250,
            D1_contribution=coefficient*point.D1_lfc/250,D2_contribution=coefficient*point.D2_lfc/250))
    for key, bd, bb in [('nonoverlap_min1',boot_d,boot_binary),('nonoverlap_min2',boot_d2,boot_binary2)]:
        scorecards[key]['conditional_guide_bootstrap'] = dict(
            draws=B, seed=args.seed,
            D_obs_percentile95=np.quantile(bd.mean(axis=1),[.025,.975]),
            donor_D_obs_percentile95=np.quantile(bd,[.025,.975],axis=0).T,
            binary_observed_delta_percentile95=np.quantile(bb,[.025,.975]),
            interpretation='Paired guide/control resampling, conditional on these two donors and fixed lists; no population coverage claim; missing contributions excluded.')
    pd.DataFrame(target_contributions).to_csv(args.out/'changed_target_contributions.tsv',sep='\t',index=False)
    pd.DataFrame({'D1_obs':boot_d[:,0],'D2_obs':boot_d[:,1],'D_obs':boot_d.mean(axis=1),
                  'binary_delta_observed':boot_binary,'min2_D_obs':boot_d2.mean(axis=1),
                  'min2_binary_delta_observed':boot_binary2}).to_csv(args.out/'conditional_bootstrap_draws.tsv.gz',
                    sep='\t',index=False,compression={'method':'gzip','mtime':0})
    metadata = dict(version='1.2.0', retained_candidate_and_control_guide_rows=len(raw), total_eligible_primary=int(raw.eligible_primary.sum()),
                    eligible_targeting=int((raw.eligible_primary&~raw.ntc).sum()),eligible_ntc=int((raw.eligible_primary&raw.ntc).sum()),
                    ntc_centers=centers, ntc_count_by_library={lib:len(x) for lib,x in ntc_ratios.items()},
                    scorecards=scorecards, runtime_seconds=time.time()-start,
                    inputs_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in sorted((ROOT/'data').iterdir()) if f.is_file()})
    (args.out/'PRIMARY_RESULTS.json').write_text(json.dumps(clean(metadata),indent=2)+'\n')
    print(json.dumps(clean(scorecards),indent=2),flush=True)


if __name__ == '__main__':
    main()
