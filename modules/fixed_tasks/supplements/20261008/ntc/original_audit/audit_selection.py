#!/usr/bin/env python3
"""Independent, read-only metadata audit of frozen NTC and target selections."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN / 'ntc_guide'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    s = np.load(ROOT / 'selection.npz', allow_pickle=False)
    m = np.load(ROOT / 'metadata/obs_metadata.npz', allow_pickle=False)
    r = json.loads((ROOT / 'selection_report.json').read_text())
    for key, path in [('selection_sha256','selection.npz'),
                      ('metadata_sha256','metadata/obs_metadata.npz'),
                      ('script_sha256','select_guides.py')]:
        assert r[key] == sha(ROOT/path), key
    nt = s['ntc_cells']; tc = s['target_cells']
    allids = np.concatenate([nt.ravel(),tc.ravel()])
    assert np.unique(allids).size == allids.size
    assert np.array_equal(np.sort(allids),s['cell_ids'])
    assert np.array_equal(s['cell_ids'][s['ntc_positions']],nt)
    assert np.array_equal(s['cell_ids'][s['target_positions']],tc)
    assert not np.any(m['low_quality'][allids])
    assert np.all(m['guide_codes'][allids] >= 0)
    assert np.all(m['lane_codes'][allids] >= 0)
    assert np.all(m['total_counts'][allids] > 0)
    assert np.array_equal(m['total_counts'][s['cell_ids']],s['total_counts'])
    assert float(s['scale']) == 8261.5 and len(s['keep']) == 13390
    assert len(np.unique(s['keep'])) == 13390
    types = m['type_categories'].astype(str)
    guide = m['guide_categories'].astype(str)
    genes = m['gene_categories'].astype(str)
    ntcode = int(np.flatnonzero(types == 'non-targeting')[0])
    tcode = int(np.flatnonzero(types == 'targeting')[0])
    ntqc = ((~m['low_quality']) & (m['gene_codes']>=0) &
            (m['type_codes']==ntcode))
    codes, counts = np.unique(m['guide_codes'][ntqc],return_counts=True)
    exp_guides = set(guide[codes[counts>=40]])
    assert set(s['ntc_guides'].astype(str)) == exp_guides
    assert np.all(m['type_codes'][nt] == ntcode)
    assert np.all(guide[m['guide_codes'][nt]] == s['ntc_guides'][:,None,None])
    assert np.all(m['type_codes'][tc] == tcode)
    assert np.all(guide[m['guide_codes'][tc]] == s['target_guides'][:,:,None,None])
    assert np.all(genes[m['gene_codes'][tc]] == s['target_names'][:,None,None,None])
    assert len(set(s['target_names'].astype(str))) == 64
    assert np.all(s['target_guides'][:,0] != s['target_guides'][:,1])
    nl = len(m['lane_categories'])
    def lane_counts(cells):
        ls=m['lane_codes'][cells]
        return np.eye(nl,dtype=int)[ls].sum(axis=-2)
    nlc=lane_counts(nt);tlc=lane_counts(tc)
    assert np.array_equal(nlc,s['ntc_lane_counts'])
    # TV compares proportions and ranges from zero (same) to one (disjoint).
    guide_lane_tv=np.abs(tlc[:,0].sum(axis=1)-tlc[:,1].sum(axis=1)).sum(axis=1)/80
    n_half_tv=np.abs(nlc[:,0]-nlc[:,1]).sum(axis=1)/40
    t_half_tv=np.abs(tlc[:,:,0]-tlc[:,:,1]).sum(axis=2)/40
    out={
      'status':'PASS', 'selection_sha256':sha(ROOT/'selection.npz'),
      'ntc_guides':len(nt),'target_pairs':len(tc),'selected_cells':int(allids.size),
      'unique_cells':True,'no_half_or_arm_overlap':True,'all_QC_pass':True,
      'all_eligible_NTC_guides_included':True,'guide_and_target_identities_match':True,
      'gene_axis':13390,'normalization_scale':8261.5,
      'ntc_perfect_lane_half_balance':int((n_half_tv==0).sum()),
      'ntc_max_half_lane_TV':float(n_half_tv.max()),
      'target_guide_perfect_lane_half_balance':int((t_half_tv==0).sum()),
      'target_guide_max_half_lane_TV':float(t_half_tv.max()),
      'target_pair_between_guide_lane_TV':{'median':float(np.median(guide_lane_tv)),
         'min':float(guide_lane_tv.min()),'max':float(guide_lane_tv.max())},
      'inference_scope':'Conditional on fixed gene axis, scale, QC, selected cells, and donor/state. Lane-associated expression differences remain possible.',
    }
    (HERE/'selection_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    main()
