"""Descriptive replay from saved target energies, control directions and case-gene effects."""
from pathlib import Path
import csv,json,types,numpy as np
from scipy.stats import pearsonr,spearmanr
ROOT=Path(__file__).resolve().parent

def run(out):
 z=np.load(ROOT/'data/reliability_target_energy.npz',allow_pickle=False)
 def rel(a,b):return {'pearson':round(float(pearsonr(a,b)[0]),3),'spearman':round(float(spearmanr(a,b).correlation),3)}
 s20,s30,sA,sB=[z[k] for k in ['s20','s30','sA','sB']]
 r={'T cell':{'s20 vs s30 (independent 20 and 30 cells)':rel(s20,s30),'sA vs sB (independent 100 and 100 cells)':rel(sA,sB),'s20 vs sA':rel(s20,sA)},'KOLF matched':{'s20 vs s30':rel(z['kolf_s20'],z['kolf_s30'])},'Replogle':{'s20 vs s30':rel(z['e1_s20'],z['e1_s30'])}}
 d2=(s20-s30)**2
 for key,x in [('energy',(20*s20+30*s30)/50),('vb',z['vb20'])]:
  groups=np.digitize(x,np.quantile(x,[.2,.4,.6,.8]));r['T cell'][f'd2 by {key} quintile / overall mean']=[round(float(d2[groups==i].mean()/d2.mean()),2) for i in range(5)]
 results={'stage1_reliability.json':r}
 # Execute the archived Q1 scientific body with stored directions replacing full-state loading.
 src=ROOT/'provenance/scripts/same_budget_weight/stage3/direction1_q1_similarity.py'
 text=src.read_text().replace('import a6_common as A','').replace('import evaluate_stage2 as es  # noqa: E402','')
 ns={'__name__':'stored_q1','__file__':str(src)};exec(compile(text,str(src),'exec'),ns)
 dn=np.load(ROOT/'data/direction_control_states.npz',allow_pickle=False)
 ns['donors']=lambda:{d:tuple(dn[d+'_'+k] for k in ['genes','U','cm','lam','trace']) for d in ['D3','D1','D2']}
 def pair(ga,Ua,gb,Ub,nrep=5):
  a=next(d for d in ['D3','D1','D2'] if np.array_equal(ga,dn[d+'_genes']) and np.array_equal(Ua,dn[d+'_U']))
  b=next(d for d in ['D3','D1','D2'] if np.array_equal(gb,dn[d+'_genes']) and np.array_equal(Ub,dn[d+'_U']))
  key=a+'_'+b;cos=dn[key+'_cos'];sab=[float((dn[key+'_QbTa'][:,i]**2).sum()) for i in range(nrep)];sba=[float((dn[key+'_QaTb'][:,j]**2).sum()) for j in range(nrep)]
  offa=float(np.abs(dn[key+'_gram_a']-np.eye(dn[key+'_gram_a'].shape[0])).max());offb=float(np.abs(dn[key+'_gram_b']-np.eye(dn[key+'_gram_b'].shape[0])).max())
  gap=max(max(abs(sab[i]-float((cos[i,:]**2).sum())) for i in range(nrep)),max(abs(sba[j]-float((cos[:,j]**2).sum())) for j in range(nrep)))
  return len(set(ga)&set(gb)),cos,sab,sba,(offa,offb,gap)
 ns['pair']=pair
 expected=json.loads((ROOT/'archived_outputs/direction1_similarity.json').read_text())
 ns['A']=types.SimpleNamespace(HERE=str(ROOT),BUNDLE=str(ROOT),arg=lambda *a:str(out),checked_sha=lambda *a:expected['integrity']['amendment_A11_sha256'],sha256=lambda *a:expected['integrity']['script_sha256'])
 ns['main']();results['direction1_similarity.json']=json.loads((out/'direction1_similarity.json').read_text())
 # Q2: recompute grouping, agreement, energy shares and relation support from rows.
 rows=list(csv.DictReader((ROOT/'data/direction1_cases_D1_genes.csv').open()));relations=json.loads((ROOT/'data/direction1_case_relations.json').read_text())
 expected=json.loads((ROOT/'archived_outputs/direction1_cases_D1.json').read_text());result={k:v for k,v in expected.items() if k!='targets'};result['targets']={}
 def agree(x,y):
  m=(x!=0)&(y!=0);return {'genes':int(m.sum()),'sign_agreement':float((np.sign(x[m])==np.sign(y[m])).mean()) if m.sum() else None,'spearman':float(spearmanr(x[m],y[m]).correlation) if m.sum()>2 else None}
 for target in ['TP53','RFX5']:
  rr=[r for r in rows if r['target']==target];a={k:np.array([float(x[k]) if x[k] else np.nan for x in rr]) for k in ['block0','direction1_component','block1','D1_truth','D3_truth','D2_truth','loading_direction_1']};y=a['block0'];comp=a['direction1_component'];along=(np.abs(comp)>=.5*np.abs(y))&(np.sign(comp)==np.sign(y))&(y!=0)
  assert np.array_equal(along,[x['along_direction_1']=='True' for x in rr])
  nz=a['loading_direction_1']!=0;c1=float(np.median(comp[nz]/a['loading_direction_1'][nz]));info={'c1 (block-0 projection on direction 1)':c1,'share of block-0 responsive energy carried by direction 1':float((comp**2).sum()/(y**2).sum()),'groups':{}}
  for key,mask in [('along direction 1',along),('not along direction 1',~along)]:
   g={'genes':int(mask.sum()),'share of block-0 responsive energy':float((y[mask]**2).sum()/(y**2).sum()),'block 1 (independent cells)':agree(y[mask],a['block1'][mask]),'D1 deep truth':agree(y[mask],a['D1_truth'][mask])}
   for donor in ['D3','D2']:
    use=mask&~np.isnan(a[donor+'_truth']);g[donor+' truth']=agree(y[use],a[donor+'_truth'][use])
   info['groups'][key]=g
  along_set={x['gene'] for x,b in zip(rr,along) if b};split={}
  for r in relations:
   if r['target']!=target:continue
   key='along direction 1' if r['gene'] in along_set else 'not along direction 1';s=split.setdefault(key,{'relations':0,'D2 rank support':0,'D2 effect support':0,'other donor measurable':0,'other donor support':0});s['relations']+=1;s['D2 rank support']+=int(r['D2_rank_support']);s['D2 effect support']+=int(r['D2_effect_support']);s['other donor measurable']+=int(r['other_donor_support'] is not None);s['other donor support']+=int(bool(r['other_donor_support']))
  for s in split.values():
   s['D2 rank support rate']=s['D2 rank support']/s['relations'];s['D2 effect support rate']=s['D2 effect support']/s['relations'];s['other donor support rate']=s['other donor support']/s['other donor measurable'] if s['other donor measurable'] else None
  info['w0y-specific relations split']=split;ix=np.flatnonzero(along);order=ix[np.argsort(-np.abs(y[ix]))];top=[]
  for i in order[:25]:
   q={'symbol':rr[i]['symbol']}
   for key,col in [('block0','block0'),('component','direction1_component'),('block1','block1'),('D1 truth','D1_truth'),('D3 truth','D3_truth'),('D2 truth','D2_truth')]:q[key]=None if np.isnan(a[col][i]) else round(float(a[col][i]),3)
   top.append(q)
  info['largest responses along direction 1']=top;result['targets'][target]=info
 results['direction1_cases_D1.json']=result
 return results
