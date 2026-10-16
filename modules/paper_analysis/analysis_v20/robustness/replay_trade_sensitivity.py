"""Recompute all scalar-competitor rankings/intervals from compact local inputs."""
from pathlib import Path
import sys,json,numpy as np
p=Path(__file__).resolve().parent;sys.path.insert(0,str(p/'vendor'))
import stim_stats as st
z=np.load(p/'TRADE_SENSITIVITY_REPLAY_INPUTS.npz');stored=np.load(p/'TRADE_SENSITIVITY_SCORES.npz');expected=json.loads((p/'TRADE_SENSITIVITY_RESULT.json').read_text())
weights={k[7:]:z[k] for k in z.files if k.startswith('weight_')}
scores,names=st.score_predictions(z['effects'],z['reference'],weights,z['query_gene_ids'],z['target_ids']);assert np.array_equal(np.array(names),stored['arm_names']);err=float(np.max(abs(scores-stored['scores'])));assert err<1e-14
for panel,key in [('responsive','responsive_positions'),('terciles','tercile_positions')]:
 ss=scores[:,:,z[key],:]
 for comparison in expected['panels'][panel]['contrasts']:
  a,b=comparison.split('_minus_');d=(ss[list(names).index(a)]-ss[list(names).index(b)]).mean((0,2));r=st.paired_block_interval(d,z['blocks_'+panel]);assert r==expected['panels'][panel]['contrasts'][comparison]
print(json.dumps({'status':'PASS','maximum_score_difference':err,'panels':2,'contrasts':sum(len(v['contrasts']) for v in expected['panels'].values())}))
