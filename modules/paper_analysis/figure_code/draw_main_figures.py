"""Render six main figures from archived summaries. Version 25 editorial display; no new statistical analysis."""
from pathlib import Path
import json
import csv
import hashlib
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--only", nargs="+", choices=[str(i) for i in range(1,7)], help="Render only selected main-figure numbers.")
ARGS=parser.parse_args()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R=Path(__file__).resolve().parents[1]; D=json.load(open(R/'source_data/figures/figure_values.json')); O=R/'manuscript/figures';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
C={'pr':'#087F8C','raw':'#7F8996','eu':'#415D8A','fr':'#8556A5','ash':'#D48232','ink':'#243246','light':'#EEF4F7','red':'#B54A51'}
def title(ax,letter,t):
 ax.set_title(t,loc='left',pad=12,fontweight='bold')
 ax.text(-.11,1.045,letter,transform=ax.transAxes,fontweight='bold',fontsize=14,va='bottom')
def forest(ax,rows,color=None,labels=True,zero=True):
 for i,r in enumerate(rows):
  v=r['value']; cc=color[i] if isinstance(color,list) else (color or C['pr']); y=len(rows)-1-i
  if len(v)==3:ax.errorbar(v[0],y,xerr=np.array([[v[0]-v[1]],[v[2]-v[0]]]),fmt='o',color=cc,ms=5,capsize=2,lw=1.3)
  else:ax.plot(v[0],y,'D',color=cc,ms=5)
 if zero:ax.axvline(0,color='#A9AFB5',ls='--',lw=.8)
 ax.set_yticks(range(len(rows)));ax.set_yticklabels([r['setting'] for r in rows][::-1] if labels else []);ax.set_ylim(-.6,len(rows)-.4);ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
def box(ax,xy,wh,text,color=C['light'],fs=10):
 x,y=xy;w,h=wh;ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.025',fc=color,ec='#CED8E0',lw=.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=C['ink'])
def arr(ax,p,q,color=C['ink']):ax.annotate('',xy=q,xytext=p,arrowprops={'arrowstyle':'-|>','lw':1.3,'color':color})
def save(fig,name):
 if ARGS.only and name[3] not in ARGS.only:
  plt.close(fig);return
 for ext in ['png','pdf','svg']:fig.savefig(O/(name+'.'+ext),dpi=300,bbox_inches='tight',pad_inches=.12)
 plt.close(fig)
def bench(group,arm):return float(next(r['point'] for r in D['benchmark_scores'] if r['group']==group and r['arm']==arm and r['K']=='25'))
# Figure 1: control geometry placement, matched 2 x 2 example, and external utility.
# The 2 x 2 values are exact archived means, displayed to four decimal places.
B1=json.load(open(R/'source_data/figures/archive/b1_design_ablation.json'))
design_2x2=[dict(weight=weight,readout=readout,P25=B1['cells']['E4 D3 responsive 20']['means'][arm],source='b1_design_ablation.json/cells/E4 D3 responsive 20/means/'+arm)
 for weight,readout,arm in [('Euclidean w0','Original y','w0y'),('Euclidean w0','Compressed Ay','w0y_AY'),('Control geometry wM','Original y','W1'),('Control geometry wM','Compressed Ay','wM_AY')]]
fig=plt.figure(figsize=(10.2,9.4))
a=fig.add_axes([.14,.775,.83,.165]);a.set_axis_off();a.set_xlim(0,1);a.set_ylim(0,1)
title(a,'a','Where should control covariance enter candidate ranking?')
box(a,(.02,.32),(.20,.38),'Matched controls\nCovariance directions',fs=9)
box(a,(.36,.61),(.24,.26),'Reliability calculation\nEuclidean $w_0$ → geometry $w_M$',color='#E1F0F1',fs=8.8)
box(a,(.36,.10),(.24,.26),'Gene-level readout\nOriginal $y$ → compressed $Ay$',fs=8.8)
box(a,(.76,.32),(.22,.38),'Combine weight\nand readout\nRank candidates',color='#E1F0F1',fs=9)
arr(a,(.22,.58),(.36,.73),C['pr']);arr(a,(.22,.43),(.36,.23),C['eu'])
arr(a,(.60,.73),(.76,.58),C['pr']);arr(a,(.60,.23),(.76,.43),C['eu'])
a.text(.31,-.065,'The same covariance directions can enter either step or both.',ha='left',va='top',fontsize=8.6,color=C['ink'])

b=fig.add_axes([.14,.425,.83,.245]);b.set_axis_off();b.set_xlim(0,1);b.set_ylim(0,1)
title(b,'b','Development example: D3, 20 cells per target')
b.text(.305,.98,'Original readout $y$',ha='center',va='top',fontsize=10,color=C['ink'])
b.text(.715,.98,'Compressed readout $Ay$',ha='center',va='top',fontsize=10,color=C['ink'])
b.text(.015,.72,'Euclidean\nweight $w_0$',ha='left',va='center',fontsize=9.5,color=C['ink'])
b.text(.015,.27,'Geometry\nweight $w_M$',ha='left',va='center',fontsize=9.5,color=C['ink'])
for x,y,value,is_pr in [(.18,.62,.2758,False),(.59,.62,.2553,False),(.18,.17,.3114,True),(.59,.17,.2853,False)]:
 box(b,(x,y),(.25,.20),f'{value:.4f}',color='#D8EEEE' if is_pr else '#EEF4F7',fs=14)
# Vertical effects isolate covariance geometry in reliability; horizontal effects hold weight fixed.
arr(b,(.305,.59),(.305,.40),C['pr']);b.text(.315,.495,'+0.0356',ha='left',va='center',fontsize=10,color=C['pr'],fontweight='bold')
arr(b,(.715,.59),(.715,.40),C['pr']);b.text(.725,.495,'+0.0300',ha='left',va='center',fontsize=10,color=C['pr'],fontweight='bold')
arr(b,(.445,.72),(.575,.72),C['red']);b.text(.51,.80,'−0.0205',ha='center',va='bottom',fontsize=9,color=C['red'])
arr(b,(.445,.27),(.575,.27),C['red']);b.text(.51,.35,'−0.0261',ha='center',va='bottom',fontsize=9,color=C['red'])
b.text(.18,.025,'P@25; responsive-query development task; archived four-decimal display precision.',fontsize=8.4,ha='left',va='center',color=C['ink'])

c=fig.add_axes([.14,.105,.32,.225])
methods=[('Unweighted','g002','raw',C['raw']),('ASH','g018','ash',C['ash']),('FR rank 100','g002','fr',C['fr']),('Euclidean','g002','euclidean',C['eu']),('PertRank','g002','pertrank',C['pr'])]
vals=[bench(g,m) for _,g,m,_ in methods]
c.barh(range(5),vals,color=[cc for *_,cc in methods],height=.60)
c.set_yticks(range(5),[x[0] for x in methods]);c.invert_yaxis();c.set_xlim(0,.45)
c.set_xlabel('P@25 · pooled-guide deep RNA reference',fontsize=8.7);title(c,'c','D4: native method comparison')
for y,v in enumerate(vals):c.text(v+.006,y,f'{v:.4f}',va='center',fontsize=9)
c.text(0,-.30,'Rest and stimulated states, equally weighted\n20 cells/target; five inputs/state',transform=c.transAxes,fontsize=8.3,va='top',color=C['ink'])

d=fig.add_axes([.65,.105,.32,.225]);eq=D['rna']['equal_source']
rna_methods=[('Unweighted','raw',C['raw']),('ASH','ASH',C['ash']),('FR rank 300','FR',C['fr']),('Euclidean','Euclidean',C['eu']),('PertRank','PertRank',C['pr'])]
vals_rna=[eq['means'][key]['P10'] for _,key,_ in rna_methods]
d.barh(range(5),vals_rna,color=[col for _,_,col in rna_methods],height=.60)
d.set_yticks(range(5),[lab for lab,_,_ in rna_methods]);d.invert_yaxis();d.set_xlim(0,.30)
d.set_xlabel('P@10 · independent arrayed KO RNA reference',fontsize=8.7);title(d,'d','Independent RNA support')
for y,v in enumerate(vals_rna):d.text(v+.004,y,f'{v:.4f}',va='center',fontsize=9)
d.text(0,-.30,'Three input sources, equally weighted\nOne external experiment; descriptive means',transform=d.transAxes,fontsize=8.3,va='top',color=C['ink'])
overview_methods=methods; overview_rna_methods=rna_methods; overview_eq=eq
save(fig,'fig1_overview_v25')

# Figure 2: source-specific independent RNA evidence.
fig=plt.figure(figsize=(10.7,8.0));gs=fig.add_gridspec(2,2,width_ratios=[1.02,1],height_ratios=[1.1,1],hspace=.58,wspace=.60)
a=fig.add_subplot(gs[0,0]);a.set_axis_off();a.set_xlim(-.025,1.025);a.set_ylim(0,1);title(a,'a','Independent perturbation evidence')
box(a,(.02,.66),(.43,.28),'Rest inputs: 20 cells/target\nTargets: 52 / 59 / 63\n(D3 / D1 / D2)',fs=7.7);box(a,(.55,.66),(.41,.28),'One independent\narrayed KO RNA\nstudy',fs=9);box(a,(.02,.22),(.43,.25),'Seven methods\nCandidate rankings',fs=9);box(a,(.55,.22),(.41,.25),'Top-10 support\nMatched candidate\npools',fs=9);arr(a,(.25,.66),(.25,.47));arr(a,(.75,.66),(.75,.47));arr(a,(.45,.345),(.55,.345))
b=fig.add_subplot(gs[0,1]);ns=['D3_Rest','D1_Rest','D2_Rest'];ks=['raw','Euclidean','PertRank','FR','FR_x_Euclidean','FR_x_PertRank','ASH'];labs=['Unweighted','Euclidean','PertRank','FR rank 300','FR × Euclidean','FR × PertRank','ASH'];mat=np.array([[D['rna'][n]['means'][k]['P10'] for n in ns] for k in ks]);im=b.imshow(mat,cmap='YlGnBu',vmin=.215,vmax=.275,aspect='auto');b.set_xticks(range(3),['D3','D1','D2']);b.set_yticks(range(7),labs);title(b,'b','All seven methods, P@10')
for i in range(7):
 for j in range(3):b.text(j,i,f'{mat[i,j]:.4f}',ha='center',va='center',fontsize=8,color='white' if mat[i,j]>.253 else C['ink'])
c=fig.add_subplot(gs[1,0]);rows=[];colors=[]
for k,col,lab in [('PertRank_minus_FR',C['fr'],'PR − FR'),('PertRank_minus_ASH',C['ash'],'PR − ASH')]:
 for n in ns:
  x=D['rna'][n]['contrasts'][k];rows.append(dict(setting=n[:2]+'  '+lab,value=[x['P10_difference']]+x['conditional_query_block_95_interval']));colors.append(col)
forest(c,rows,colors);c.set_xlabel('ΔP@10, conditional 95% intervals');c.set_xlim(-.009,.05);title(c,'c','Comparison with native estimators')
d=fig.add_subplot(gs[1,1]);eq=D['rna']['equal_source'];keys=['PertRank_minus_FR','PertRank_minus_ASH','PertRank_minus_Euclidean','FR_x_PertRank_minus_FR_x_Euclidean'];labs=['PR − FR','PR − ASH','Geometry on raw','Geometry on FR'];vs=[eq['contrasts'][k]['expected_members_difference'] for k in keys];d.barh(range(4),vs,color=[C['fr'],C['ash'],C['eu'],C['pr']],height=.55);d.set_yticks(range(4),labs);d.invert_yaxis();d.set_xlim(0,.32);d.set_xlabel('Additional reference members per 10');title(d,'d','Contributions to the shortlist')
for i,v in enumerate(vs):d.text(v+.006,i,f'+{v:.3f}',va='center',fontsize=8)
fig.subplots_adjust(left=.16,right=.98,bottom=.12,top=.94);fig.text(.16,.024,'Equal-source summaries are descriptive. Sources share one external experiment; query intervals hold that reference fixed.',fontsize=8);save(fig,'fig2_independent_rna_v25')

# Figure 3: coordinate design and estimator-by-weight factorial.
fig=plt.figure(figsize=(10.2,8.7));gs=fig.add_gridspec(2,2,height_ratios=[2.25,1],hspace=.68,wspace=.55)
labels=[r['setting'].replace('E4 ','').replace('E5 ','').replace('E1 ','').replace('E3 ','').replace('responsive','resp.').replace('terciles','terc.') for r in D['design']]
a=fig.add_subplot(gs[0,0]);forest(a,[{'setting':lab,'value':r['coordinate']} for lab,r in zip(labels,D['design'])]);title(a,'a','Preserving the readout coordinates');a.set_xlabel('Original − transformed readout, ΔP@25\n(same control-geometry weight)');a.set_xlim(-.004,.051)
b=fig.add_subplot(gs[0,1]);forest(b,[{'setting':lab,'value':r['geometry']} for lab,r in zip(labels,D['design'])],C['eu'],labels=False);title(b,'b','Control geometry at fixed readout');b.set_xlabel('Geometry − Euclidean weight, ΔP@25\n(same original coordinates)');b.set_xlim(-.004,.051);b.set_xticks([0,.01,.02,.03,.04,.05])
c=fig.add_subplot(gs[1,:]);xx=np.arange(3)
for armkeys,color,lab,offset in [(['raw','euclidean','pertrank'],C['eu'],'Original effects',-.17),(['fr','fr_euclidean','fr_pertrank'],C['fr'],'FR effects (rank 300)',.17)]:
 vv=[bench('g005',m) for m in armkeys]
 c.bar(xx+offset,vv,width=.31,color=color,label=lab)
 for x,v in zip(xx+offset,vv):c.text(x,v+.007,f'{v:.4f}',ha='center',fontsize=8.5)
c.set_xticks(xx,['None','Euclidean','Control geometry']);c.set_ylim(0,.62);c.set_ylabel('P@25');c.set_xlabel('Reliability weight');title(c,'c','Reuse the reliability calculation across effect estimators')
c.legend(frameon=False,fontsize=8.5,loc='upper left',ncol=2);c.set_xlim(-.55,2.55)
z=D['interactions']['A_geometry_interaction']['equal_state']['25']
for key,ypos,color,label in [('raw_geometry',.94,C['eu'],'Original'),('FR_geometry',.81,C['fr'],'FR')]:
 q=z[key];lo,hi=q['descriptive_conditional_95_interval']
 c.text(.98,ypos,f"{label}: geometry Δ {q['point']:+.4f} [{lo:.4f}, {hi:.4f}]",transform=c.transAxes,ha='right',va='top',fontsize=8,color=color)
fig.subplots_adjust(left=.22,right=.98,bottom=.105,top=.94)
fig.text(.22,.025,'a,b: Four RNA sources, 14 correlated single-input settings; conditional 95% intervals.\nc: D4 states equally weighted, five inputs/state; FR rank-expansion sensitivity; conditional 95% intervals.',fontsize=8,color=C['ink'])
save(fig,'fig3_design_v25')

# Figure 4: reliability and alignment diagnostics.
fig,axs=plt.subplots(2,2,figsize=(10.2,7.5));fig.subplots_adjust(left=.23,right=.98,wspace=.8,hspace=.70,bottom=.10)
a=axs[0,0];forest(a,D['energy'],[C['pr'],C['raw']]);title(a,'a','Reliability is a performance bottleneck');a.set_xlabel('Gain over Euclidean, ΔP@25');a.set_xlim(0,.064);a.text(0,-.40,'Deep energy is an independent diagnostic.\nWeight agreement (Pearson): 0.619 → 0.736',transform=a.transAxes,fontsize=8,va='top')
b=axs[0,1];forest(b,D['direction']);title(b,'b','Leading directions carry the benefit');b.set_xlabel('D3 gain over Euclidean, ΔP@25');b.set_xlim(-.021,.055)
c=axs[1,0];forest(c,D['nonquery'],[C['pr'],C['pr'],C['eu']]);title(c,'c','Non-query and suppression checks');c.set_xlabel('Matched geometry increment, ΔP@25');c.set_xlim(0,.065);c.text(0,-.38,'Suppression subset: 291 targets; its full-pool\ncontribution accounts for 68.5% of the gain.',transform=c.transAxes,fontsize=8,va='top')
d=axs[1,1];al=D['alignment'];yy=np.arange(len(al));d.barh(yy,[r['gains'] for r in al],height=.50,color=C['pr'],label='Supported gains');d.barh(yy,[-r['losses'] for r in al],height=.50,color=C['red'],label='Supported losses');d.scatter([r['value'][0] for r in al],yy,c=C['ink'],s=22,zorder=4,label='Net contribution');d.set_yticks(yy,[r['setting'] for r in al]);d.invert_yaxis();d.axvline(0,c='#A9AFB5',lw=.8);d.set_xlabel('Reference members per 25 nominations');title(d,'d','Alignment reveals gains and costs');d.legend(frameon=False,fontsize=7,loc='lower right');d.set_xlim(-.40,.61)
save(fig,'fig4_mechanism_v25')
# Figure 5: fixed settings, depth and input equivalence.
fig=plt.figure(figsize=(10.2,7.1));gs=fig.add_gridspec(2,2,hspace=.62,wspace=.58)
a=fig.add_subplot(gs[0,0]);forest(a,D['donors']);title(a,'a','Fixed method across donors and states');a.set_xlabel('PertRank − Euclidean, ΔP@25');a.set_xlim(0,.045);a.set_xticks([0,.01,.02,.03,.04])
b=fig.add_subplot(gs[0,1]);forest(b,D['equivalence'],zero=False);b.axvline(20,color=C['raw'],ls='--',lw=1);title(b,'b','Ranking-input cell equivalents');b.set_xlabel('Unweighted cells matching PertRank20');b.set_xlim(18,45)
c=fig.add_subplot(gs[1,0]);r=D['d4_depth'];depths=[20,25,30,35,40,45,50];c.plot(depths,r['raw_curve'],'o-',color=C['raw'],label='Unweighted curve');
for method,color in [('Euclidean',C['eu']),('PertRank',C['pr'])]:
 v=r['methods'][method];c.hlines(v['target_score'],20,v['n_equivalent'],color=color,lw=1.4,ls='--');c.plot(v['n_equivalent'],v['target_score'],'o',color=color,label=f'{method} 20-cell level')
c.set_xlabel('Ranking cells per target');c.set_ylabel('P@25');title(c,'c','D4: measured depth calibration');c.legend(frameon=False,fontsize=7,loc='lower right');c.text(.02,.95,'38.87 unweighted cells ≈ PertRank20\n7.07 extra cells from geometry',transform=c.transAxes,va='top',fontsize=8)
d=fig.add_subplot(gs[1,1]);z=D['interactions']['B_depth_interaction']['25'];xx=[20,50,100];vals=[z['n'+str(n)]['point'] for n in xx];cis=[z['n'+str(n)]['descriptive_conditional_95_interval'] for n in xx];d.errorbar(xx,vals,yerr=np.array([[v-ci[0] for v,ci in zip(vals,cis)],[ci[1]-v for v,ci in zip(vals,cis)]]),fmt='o-',color=C['pr'],capsize=3);d.axhline(0,c=C['raw'],ls='--',lw=.8);d.set_xticks(xx);d.set_ylim(-.021,.036);d.set_xlabel('Ranking cells per target');d.set_ylabel('PertRank − ASH, ΔP@25');title(d,'d','The advantage changes with depth');d.text(.03,.98,'20 − 100 interaction: +0.0341\n95% interval [0.0271, 0.0405]',transform=d.transAxes,va='top',fontsize=8)
fig.subplots_adjust(left=.16,right=.98,bottom=.12,top=.93);save(fig,'fig5_efficiency_v25')
# Figure 6: fixed query-panel transfer.
fig=plt.figure(figsize=(10.2,7.2));gs=fig.add_gridspec(2,2,height_ratios=[1,1.4],hspace=.65,wspace=.65)
a=fig.add_subplot(gs[0,:]);a.set_axis_off();a.set_xlim(-.025,1.025);a.set_ylim(0,1);title(a,'a','Reuse an existing response panel in a new cell state')
box(a,(.00,.53),(.24,.33),'Existing D3 Rest panel\n300 fixed queries');box(a,(.00,.02),(.24,.33),'D4 Stim measurements\nControls + 20 cells/target');box(a,(.40,.27),(.22,.39),'Fixed PertRank\n561 candidates');box(a,(.79,.27),(.20,.39),'Query-specific\ncandidate lists',color='#E1F0F1');arr(a,(.24,.70),(.40,.54));arr(a,(.24,.18),(.40,.39));arr(a,(.62,.47),(.79,.47));a.text(.395,.04,'Queries fixed by the source panel',fontsize=9,color=C['pr']);a.text(.80,.10,'100-cell reference\nfor evaluation only',fontsize=8,ha='left')
b=fig.add_subplot(gs[1,0]);t=D['transfer'];labs=['Unweighted','Euclidean','Pearson metric','PertRank'];keys=['raw','Euclidean','PR_M','PertRank'];vals=[t['means_P25'][k] for k in keys];b.barh(range(4),vals,color=[C['raw'],C['eu'],'#A6B2C4',C['pr']],height=.57);b.set_yticks(range(4),labs);b.invert_yaxis();b.set_xlim(0,.385);b.set_xlabel('Transferred-panel P@25');title(b,'b','All methods use the same panel')
for y,v in enumerate(vals):b.text(v+.006,y,f'{v:.4f}',va='center',fontsize=8)
c=fig.add_subplot(gs[1,1]);rows=[dict(setting=lab,value=[t['contrasts'][k]['extra_members_per25']]+t['contrasts'][k]['members_interval']) for k,lab in [('PertRank-raw','Over unweighted'),('PertRank-Euclidean','Over Euclidean'),('PertRank-PR_M','Over Pearson metric')]];forest(c,rows);c.set_xlabel('Additional reference members per 25');c.set_xlim(0,2.45);title(c,'c','Quantified value of panel reuse')
fig.subplots_adjust(left=.17,right=.98,bottom=.13,top=.93);fig.text(.17,.025,'Previously opened source and target conditions; five fixed inputs. Intervals condition on the retained source query blocks.',fontsize=8);save(fig,'fig6_panel_transfer_v25')
# Display-level source mapping and overview values; no inferential calculations.
source=R/'source_data/figures'
with (source/'v25_design_2x2.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['task','weight','readout','P25_display_value','provenance','precision'])
 for row in design_2x2:
  writer.writerow(['D3 responsive 20 cells',row['weight'],row['readout'],f"{row['P25']:.4f}",row['source'],'four-decimal archived display'])
with (source/'v25_overview_display.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['panel','task','estimator','weight','value','ci95_lower','ci95_upper','source_key'])
 for row in design_2x2:
  writer.writerow(['1b','D3 responsive 20 cells P@25',row['readout'],row['weight'],f"{row['P25']:.4f}",'','',row['source']])
 for lab,group,arm,_ in overview_methods:
  writer.writerow(['1c','D4 equal-state P@25',lab,'as labelled',bench(group,arm),'','','benchmark_scores/'+group+'/'+arm+'/25'])
 for lab,key,_ in overview_rna_methods:
  writer.writerow(['1d','Independent RNA equal-source P@10',lab,'as labelled',overview_eq['means'][key]['P10'],'','','rna/equal_source/means/'+key+'/P10'])
manifest={
 'version':'25',
 'scope':'Editorial reorganisation and plotting of archived results; no new experiment, fit, bootstrap or significance test.',
 'source_file':'source_data/figures/figure_values.json',
 'source_sha256':hashlib.sha256((source/'figure_values.json').read_bytes()).hexdigest(),
 'figures':{
  'fig1_overview_v25':'Control-geometry placement schematic; matched 2x2 D3 illustration; native D4 comparison; descriptive independent RNA comparison.',
  'fig2_independent_rna_v25':'v21/v22 Fig6, retained source-specific seven-arm scores and conditional intervals.',
  'fig3_design_v25':'Fixed-geometry-weight readout contrast; pure geometry increment at original readout using archived W1 - w0y paired intervals (corrected D1); effect-estimator by weight comparison with archived geometry intervals.',
  'fig4_mechanism_v25':'v21/v22 Fig2, unchanged scientific content.',
  'fig5_efficiency_v25':'v21/v22 Fig3, unchanged scientific content.',
  'fig6_panel_transfer_v25':'v21/v22 Fig5, unchanged scientific content.'},
 'design_2x2':'D3 responsive 20-cell single-input development task. Exact archived mean scores displayed to four decimals; no confidence interval or interaction test inferred from rounded values.',
 'uncertainty':'Archived conditional intervals only. Equal-source independent RNA summaries have no pooled confidence intervals.',
 'FR_identity':'Figure1c: rank100 primary native configuration. Figure1d and Figure2: rank300 independent RNA configuration. Figure3c: rank300 expansion sensitivity.',
 'design_geometry_provenance':D['design_geometry_provenance'],
 'factorial_source':'source_data/figures/archive/b1_design_ablation.json contains the full archived factorial results.'
}
(source/'v25_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Rendered v25 figures '+(', '.join(ARGS.only) if ARGS.only else '1–6')+' as PNG, PDF and SVG; archived-result display map exported.')
