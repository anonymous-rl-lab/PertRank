#!/usr/bin/env python3
"""Descriptive equal-source mean with the same shared reference draw per source."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

DONORS=['D3_Rest','D1_Rest','D2_Rest']

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scores-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists();a.out.mkdir(parents=True)
    reports={d:json.loads((a.scores_root/d/'report.json').read_text()) for d in DONORS}
    common=set.intersection(*[set(x['means']) for x in reports.values()])
    assert {'raw','Euclidean','PertRank'}<=common
    means={arm:{'P10':float(np.mean([reports[d]['means'][arm]['P10'] for d in DONORS])),
                'expected_members_per_10':float(np.mean([reports[d]['means'][arm]['expected_members_per_10'] for d in DONORS]))} for arm in sorted(common)}
    contrasts=set.intersection(*[set(x['contrasts']) for x in reports.values()])
    delta={c:{'P10_difference':float(np.mean([reports[d]['contrasts'][c]['P10_difference'] for d in DONORS])),
              'expected_members_difference':float(np.mean([reports[d]['contrasts'][c]['expected_members_difference'] for d in DONORS]))} for c in sorted(contrasts)}
    sensitivity={};provenance={str(a.scores_root/d/'report.json'):sha(a.scores_root/d/'report.json') for d in DONORS}
    for scheme in ['shared_labels','batch_separate']:
        files=[a.scores_root/d/('reference_propagation_'+scheme+'.npz') for d in DONORS]
        if not all(f.exists() for f in files):continue
        refs=[np.load(f,allow_pickle=False) for f in files]
        names=[str(x) for x in refs[0]['arm_names'] if str(x) in common]
        v=np.mean([r['expected_members'][:,[list(r['arm_names'].astype(str)).index(k) for k in names]] for r in refs],axis=0)
        assert v.shape==(500,len(names))
        sr={}
        for c in sorted(contrasts):
            aa,bb=c.split('_minus_');dd=v[:,names.index(aa)]-v[:,names.index(bb)]
            sr[c]={'median':float(np.median(dd)),'empirical_2p5_97p5':np.quantile(dd,[.025,.975]).tolist(),'fraction_positive':float(np.mean(dd>0))}
        sensitivity[scheme]=sr
        np.savez_compressed(a.out/('equal_source_reference_'+scheme+'.npz'),arm_names=np.array(names),expected_members=v)
        provenance.update({str(f):sha(f) for f in files})
        for r in refs:r.close()
    result={'status':'COMPLETE','identity':'equal-source descriptive summary; not pooled independent donor confirmation',
      'donors':DONORS,'reference_shared_between_sources':True,'pooled_confidence_interval':None,
      'means':means,'contrasts':delta,'reference_sensitivity':sensitivity,
      'reference_sensitivity_units':'expected externally supported members per10-name list; empirical reference uncertainty ranges, not population confidence intervals',
      'unevaluated':{d:reports[d]['unevaluated'] for d in DONORS},'source_sha256':provenance}
    (a.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'COMPLETE','means':means,'contrasts':delta},indent=2))

if __name__=='__main__':main()
