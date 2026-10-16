from pathlib import Path
import shutil,subprocess,os,sys,json,numpy as np
R=Path('/workspace/scratch/9bb77e8faf2b');src=R/'code_v150_release/PertRank_GitHub_Repro_v1.5.0/modules/corrected_evidence/guide';out=R/'code_v151_release/staging/corrected_intermediates/guide';work=R/'code_v151_release/qa/guide_export';work.mkdir(parents=True,exist_ok=True)
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
for mode in ['legacy','corrected']:
 w=work/mode;shutil.copytree(src/'canonical',w,dirs_exist_ok=True);stage=w/'same_budget_weight/stage3/data'
 for d in ['D3_Rest','D1_Rest','D2_Rest']:
  n=f'a15_{d}_guide_split.npz';z=np.load(src/'data'/n,allow_pickle=True);a={k:z[k]for k in z.files}
  if mode=='legacy':
   p=np.load(src/'data'/f'{d}_legacy_patch.npz',allow_pickle=True);a.update({k:p[k]for k in p.files})
  np.savez_compressed(stage/n,**a)
 for script in ['evaluate_a15_guide_split.py','a15a_guide_switch_cost.py','a15b_did_and_lane.py','a17_four_arm_ablation.py']:
  cap=out/'capture'/mode/script[:-3];res=work/'results'/mode;res.mkdir(parents=True,exist_ok=True)
  with (work/(mode+'_'+script+'.log')).open('w')as f:
   subprocess.run([sys.executable,'/tmp/capture_guide_intermediates.py',str(w),str(src/'data'),script,str(res),str(cap),str(src/'offline_execute.py')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 shutil.copytree(src/'expected'/mode,out/'archive'/mode,dirs_exist_ok=True)
# Lane raw integer counts are already tiny; preserve them for genuine lane recalculation.
for d in ['D3_Rest','D1_Rest','D2_Rest']:shutil.copy2(src/'data'/f'{d}_lane_counts.npz',out/f'{d}_lane_counts.npz')
shutil.copy2(src/'canonical/pr10/bootstrap.py',out/'bootstrap.py')
s=(src/'canonical/same_budget_weight/stage3/sizing_depth_ladder.py').read_text();fn=s[s.index('def donor_mean_bootstrap'):s.index('\ndef main()',s.index('def donor_mean_bootstrap'))]
(out/'donor_bootstrap.py').write_text('import numpy as np\n\n'+fn)
print('GUIDE_INTERMEDIATE_EXPORT_COMPLETE')
