from pathlib import Path
import json,datetime,hashlib
ROOT=Path(__file__).resolve().parent
spec={
 'title':'PertRank versus native FR-Perturb and fixed reliability combinations',
 'status':'OPEN_DATA_DEVELOPMENT_PROTOCOL_NOT_PUBLIC_PREREGISTRATION',
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'new_D4_FR_scores_seen':False,
 'allowed_expression':['D3_Rest','D4_Rest','D4_Stim8hr'],
 'prohibited_expression':['D1_Stim8hr','D2_Stim8hr'],
 'arms':['raw','euclidean','pertrank','fr','fr_euclidean','fr_pertrank'],
 'formulas':{'raw':'y','euclidean':'w0*y','pertrank':'wM*y','fr':'F','fr_euclidean':'w0*F','fr_pertrank':'wM*F'},
 'weights':'Reuse original matched-input frozen w0 and wM. No weight re-estimation on F, no mixing coefficient, no post-result subgroup selection.',
 'fr_commit':'24e653b6a5c050ab17f3600ebaf222bef32e2d25',
 'fr_native':'Authors sparse PCA/SPAMS and Lasso algorithm, TP10K/log1p, control centering, native scaling. Full precision serialization; no external atlas.',
 'fr_gene_axis':'All original raw genes, input-only native filtering. Query outputs aligned by exact gene identifier. No reference-driven gene selection.',
 'fr_grid':[
 {'id':'default','rank':20,'spca_alpha':0.1,'lasso_alpha':0.0001},
 {'id':'rank10','rank':10,'spca_alpha':0.1,'lasso_alpha':0.0001},
 {'id':'rank50','rank':50,'spca_alpha':0.1,'lasso_alpha':0.0001},
 {'id':'lasso_low','rank':20,'spca_alpha':0.1,'lasso_alpha':0.00001},
 {'id':'lasso_high','rank':20,'spca_alpha':0.1,'lasso_alpha':0.001},
 {'id':'rank50_lasso_high','rank':50,'spca_alpha':0.1,'lasso_alpha':0.001}],
 'development':{'dataset':'D3_Rest','input_rep':0,'candidates':1200,'depth':20,'objective':'highest native FR P@25 over original responsive query panel, mean of signed directions','ties':'Within 1e-12 choose official default if tied, else lowest listed index','failed_runs':'Retain errors, no zero imputation. Numerical or resource failure is not a method loss. All grid runs required for strongest-in-grid claim.'},
 'transfer':{'datasets':['D4_Rest','D4_Stim8hr'],'input_reps':[0,1,2,3,4],'configuration':'single D3 native-FR winner, frozen before any D4 FR scores','reference':'original task reference; no target or query changes','analysis_scope':'opened data fixed-configuration transfer, not new blinded donor confirmation'},
 'independent_guide':{'dataset':'D4_Stim8hr','candidates':395,'depth':20,'reference_depth':200,'input_reps':[0],'configuration':'same D3 winner','reference_quality':'retain prior report; this analysis does not overturn original metric-increment stop','analysis_scope':'opened independent-guide task, not independent laboratory'},
 'sampling_and_scores':{'K_primary':25,'K_sensitivity':10,'own_target':'exclude by stable gene identity before top K','ties':'exact independent boundary-membership expectation','queries':'original frozen panels and 30 original blocks','controls':'same cells available to every method','minimum_candidates':250},
 'randomness':{'numpy_seed_base':20261012,'fit_seed':'base+input_rep, equal across configurations; record native solver randomness limitations','bootstrap_seed':20261012,'bootstrap_draws':2000},
 'uncertainty':{'pooled':'mean five inputs within query; paired original query-block bootstrap; conditional on observed inputs/reference. Report all input repetitions separately. Reference uncertainty only if actual reference-cell resampling executed.','independent':'joint target-by-lane reference and paired query-block resampling using existing archived cell-level reference; same reference and NTC draws across arms','multiple_contrasts':'95% paired intervals descriptive; 99% Bonferroni intervals for five prespecified contrasts if simultaneous claims are made'},
 'contrasts':['fr-raw','fr-euclidean','pertrank-fr','fr_pertrank-fr','fr_pertrank-fr_euclidean'],
 'development_gates':{'interpretation':'resource allocation only, not confirmatory success','direct':'PR-F positive in both D4 states; equal-state mean >=0.01; paired conditional 95% lower bound >0; at least 4/5 equal-state input-rep differences positive','complementarity':'wM*F-F positive in both states, equal-state mean >=0.005 with paired conditional 95% lower bound >0; wM*F-w0*F positive in both states with equal-state paired conditional 95% lower bound >0','stop':'Pause expansion if neither route passes; preserve all runs. No claim of universal plug-in layer from one external method.'},
 'engineering_gates':{'smoke':'<=10 minutes full-gene synthetic and 32-target real-data compute check, no D4 scoring','small':'<=1 hour engineering pilot then extrapolate full runtime and memory','failed_or_nonconverged':'record as failed/not evaluable, never score as all-zero or count as a win','resource_limit':'do not alter native inference silently; document before-result amendments if technically necessary'},
 'additional_external':{'ASH':'official ashr posterior mean, normal and halfuniform prior configurations selected only on D3','TRADE_posterior':'official halfuniform fit core full posterior mean adaptation, not scalar; explicit native mixEM numerical sensitivity below','optimizer_rule':'mixsqp primary. Official mixEM retry only on explicit numerical failure/nonconvergence with fixed 10000 iterations; log both. Keep solver-specific results and no winner selection by D4 performance. A failed full arm remains unavailable without valid complete outputs.','GPerturb':'only after FR milestone; prepare explicit convergence/resource budget before training; lack of GPU or incomplete training is not a loss'},
 'reference_sensitivity':'Keep original log-effect reference primary. A separately specified Pearson-reference implementation is secondary only; fit parameters on designated controls and verify rankings change. No sensitive reference is substituted for original.'
}
p=ROOT/'EXECUTION_SPEC.json'
assert not p.exists(),'Protocol exists; never overwrite silently'
p.write_text(json.dumps(spec,indent=2,ensure_ascii=False)+'\n')
h=hashlib.sha256(p.read_bytes()).hexdigest();(ROOT/'EXECUTION_SPEC.sha256').write_text(h+'  EXECUTION_SPEC.json\n')
print(json.dumps({'path':str(p),'sha256':h,'created_utc':spec['created_utc']}))
