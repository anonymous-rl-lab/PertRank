#!/usr/bin/env python3
"""Match predeclared common-arm fingerprints against both original FR depth rows."""
from pathlib import Path
import argparse,datetime,hashlib,json
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--depth',type=int,choices=[20,50,100],required=True);ap.add_argument('--ash',type=Path,required=True);a=ap.parse_args();r=json.loads((a.ash/'evaluation_report.json').read_text());assert r['status']=='COMPLETE_FROZEN_POSTHOC_DEPTH_ASH'and r['depth']==a.depth
keys=['raw_scores','reference_scores','w0','wm','queries','candidates','blocks','three_arm_query_p25','three_arm_query_p10'];checks=[]
for spec in ['A_native_default_penalty','B_effective_penalty_sensitivity']:
 f=ROOT/'posthoc_depth_20261014/evaluation'/spec/f'n{a.depth}'/'evaluation_report.json';q=json.loads(f.read_text())
 assert q['status'].startswith('COMPLETE')and q['depth']==a.depth
 for key in keys:assert r['common_three_arm_array_sha256'][key]==q['common_three_arm_array_sha256'][key],(spec,key)
 checks.append(dict(FR_specification=spec,FR_report_path=str(f),FR_report_sha256=sha(f),all_nine_common_arm_fingerprints_match=True))
out=dict(status='DEPTH_COMMON_THREE_ARMS_BOTH_FR_SPECIFICATIONS_MATCH',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),depth=a.depth,main_benchmark=False,ash_report_sha256=sha(a.ash/'evaluation_report.json'),matched_keys=keys,checks=checks,source_sha256=sha(__file__),scope='Read-only identity check on the same separate 250-target universe; no scoring or ASH parameter changes.')
(a.ash/'COMMON_THREE_ARM_MATCH.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(status=out['status'],depth=a.depth,matched_fingerprints=18)),flush=True)
