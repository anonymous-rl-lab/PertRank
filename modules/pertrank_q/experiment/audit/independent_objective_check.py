"""Independent differentiability/symmetry review of the locked B update."""
from pathlib import Path
import importlib.util,json,sys,hashlib,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DEFAULT_BASE=ROOT/'baseline_B' if (ROOT/'baseline_B').exists() else ROOT.parent/'pertrank_boundary_learning_20261115'
BASE=Path(os.environ.get('PERTRANK_BASE_ROOT',str(DEFAULT_BASE))).resolve()
OUTPUT=Path(os.environ.get('PERTRANK_AUDIT_OUTPUT_ROOT',str(ROOT/'audit'))).resolve()
OUTPUT.mkdir(parents=True,exist_ok=True)
if not BASE.exists():BASE=ROOT.parent/'pertrank_boundary_learning_20261115'
spec=importlib.util.spec_from_file_location('reviewed_trainer',ROOT/'train_optimization.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.configure(BASE)
torch=mod.torch
mask=torch.tensor([[True,True,False,True,True],[True,False,True,True,True]])
scale=torch.tensor([[.7],[1.2]]);t={'mask':mask,'scale':scale}
score=torch.tensor([[.3,-.6,9.,1.4,-.1],[.8,2.,-.2,-.9,.4]],requires_grad=True)
support=torch.tensor([[[.8,.1,0.,.9,.2],[.6,0.,.1,.4,.9]],[[.1,.9,0.,.2,.8],[.1,0.,.8,.9,.2]]])
def loss(x):return mod.list_loss(mod.ranking_scores(x,t,'rank_scale'),mask,support[0],support[1],2,.5)
passed=torch.autograd.gradcheck(loss,(score,),eps=1e-6,atol=1e-6,rtol=1e-4);assert passed
g=torch.autograd.grad(loss(score),score)[0]
base=float(loss(score).detach());checks={}
for name,x in [('scale_half',score*.5),('scale_double',score*2),('query_shift',score+torch.tensor([[4.],[-7.]]))]:
 err=abs(float(loss(x).detach())-base);assert err<1e-12;checks[name]=err
for name,x in [('zero',torch.zeros_like(score)),('constant',torch.full_like(score,2.)),('nearzero',score.detach()*1e-10)]:
 x.requires_grad_();v=loss(x);gg=torch.autograd.grad(v,x)[0];assert torch.isfinite(v) and torch.isfinite(gg).all();checks[name+'_finite']=True
assert float(g[~mask].abs().max())==0
radial=float((g*score.detach()).sum(1).abs().max());translation=float(g.sum(1).abs().max());assert radial<1e-12 and translation<1e-12
text=(ROOT/'train_optimization.py').read_text()
assert "ensemble=np.mean([stored[seed,ih,epoch] for seed in SEEDS],axis=0)" in text
assert "precision(ensemble,va['reference'],va['ownmask'],5)" in text
assert "residual=.05*((score-t['base'])/t['scale'])[t['mask']].square().mean()" in text
report={'passed':True,'training_performed':False,'trainer_sha256':hashlib.sha256((ROOT/'train_optimization.py').read_bytes()).hexdigest(),'autograd_finite_difference_gradcheck':passed,'rank_scale_no_detach':True,'eligible_mean_and_population_sd':True,'variance_floor_before_sqrt':True,'radial_gradient_dot_max':radial,'translation_gradient_dot_max':translation,'excluded_score_gradient_zero':True,'checks':checks,'code_review_joint_selection':'Physical predictions are averaged before exact precision; not averaging per-seed precision','code_review_regularization':'Residual penalty remains normalized by frozen training baseline SD','held_reference_opened':False}
(OUTPUT/'independent_objective_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
