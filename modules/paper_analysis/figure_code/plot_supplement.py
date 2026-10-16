"""Redraw archived Supplementary Figures 1–4; no scientific analysis.
Run: python figure_code/plot_supplement.py from the paper package root.
Requires matplotlib and numpy. CSV input provenance is in source_data/figures.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'source_data'/'figures';OUT=ROOT/'manuscript'/'figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42})
BLUE='#2477AD';ORANGE='#D97918';TEAL='#248A78';PURPLE='#8464A3';GRAY='#72777C'
def read(name):return list(csv.DictReader((DATA/name).open()))
def val(r,k):return float(r[k])if r[k]!=''else None
def finish(fig,name):
 for ext in ['png','pdf']:
  fig.savefig(OUT/(name+'.'+ext),dpi=300,facecolor='white')
 plt.close(fig)
def panel(ax,letter,title):ax.set_title(f'{letter}  {title}',loc='left',fontweight='bold',pad=10)
def point(ax,v,y,l=None,h=None,c=ORANGE,marker='o',label=None):
 if l is None or h is None:ax.plot(v,y,marker,color=c,markersize=5,label=label)
 else:ax.errorbar(v,y,xerr=np.array([[v-l],[h-v]]),fmt=marker,color=c,capsize=2,ms=4.5,lw=1,label=label)
def zero(ax):ax.axvline(0,color='#AAAAAA',lw=.7,zorder=0);ax.grid(axis='x',color='#E8E8E8',lw=.5);ax.set_axisbelow(True)
# S1: empirical percentiles, never labelled confidence intervals.
rows=read('figS1_refit.csv');fig,axs=plt.subplots(1,2,figsize=(7.09,3.0),sharex=True)
for ax,panelname,letter in zip(axs,['Responsive','Terciles'],['a','b']):
 rr=[r for r in rows if r['panel']==panelname];zero(ax)
 for j,r in enumerate(rr):
  point(ax,val(r,'original'),j+.19,c=ORANGE,marker='D')
  point(ax,val(r,'refit_mean'),j-.02,val(r,'empirical_low'),val(r,'empirical_high'),c=TEAL)
  if val(r,'fixed_weight_mean')is not None:point(ax,val(r,'fixed_weight_mean'),j-.23,c=PURPLE,marker='s')
 ax.set_yticks([0,1],['vs Euclidean','vs Unweighted']);ax.set_ylim(1.5,-.5);ax.set_xlim(-.013,.041);ax.set_xlabel('PertRank difference in P@25');panel(ax,letter,panelname)
handles=[Line2D([0],[0],color=c,marker=m,lw=0,label=l)for c,m,l in [(ORANGE,'D','Original fixed cells'),(TEAL,'o','Refitted weights; empirical range'),(PURPLE,'s','Fixed-weight draw mean')]]
fig.legend(handles=handles,loc='lower center',ncol=1,frameon=False,bbox_to_anchor=(.52,.005));fig.subplots_adjust(left=.15,right=.98,top=.85,bottom=.34,wspace=.62)
finish(fig,'figS1_refit_sensitivity')
# S2 native Mixscale adaptation.
rows=read('figS2_mixscale.csv');fig,axs=plt.subplots(1,2,figsize=(7.09,2.9),gridspec_kw={'width_ratios':[1,1.12]})
r=[x for x in rows if x['kind']=='score']
for i,x in enumerate(r):point(axs[0],val(x,'value'),i,c=ORANGE if x['label']=='PertRank'else BLUE)
axs[0].set_yticks(range(4),[x['label']for x in r]);axs[0].invert_yaxis();axs[0].set_xlim(0,.35);axs[0].set_xlabel('P@25');axs[0].grid(axis='x',alpha=.2);panel(axs[0],'a','D3 nomination scores')
r=[x for x in rows if x['kind']=='contrast']
for i,x in enumerate(r):point(axs[1],val(x,'value'),i,val(x,'low'),val(x,'high'),c=ORANGE)
axs[1].set_yticks(range(3),[x['label'].replace(' - ','\n− ')for x in r]);axs[1].set_ylim(2.6,-.6);zero(axs[1]);axs[1].set_xlim(-.29,.32);axs[1].set_xlabel('Paired difference in P@25');panel(axs[1],'b','Matched contrasts')
fig.subplots_adjust(left=.2,right=.98,bottom=.2,top=.85,wspace=.9);finish(fig,'figS2_mixscale_adaptation')
# S3 mechanism diagnostics; all paired ranges are archived intervals.
fig=plt.figure(figsize=(7.09,6.3));gs=fig.add_gridspec(2,2,height_ratios=[1,1.35],wspace=.62,hspace=.63)
gsa=gs[0,0].subgridspec(2,1,hspace=.85);a=fig.add_subplot(gsa[0]);a2=fig.add_subplot(gsa[1]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,0]);d=fig.add_subplot(gs[1,1])
r=read('figS3a_energy.csv')
rr=[x for x in r if x['measure']=='P@25 gain over Euclidean']
for i,x in enumerate(rr):point(a,val(x,'value'),i,c=ORANGE if x['arm']=='PertRank'else GRAY)
a.set_yticks(range(2),[x['arm']for x in rr]);a.set_ylim(1.6,-.6);a.set_xlim(0,.06);a.set_xlabel('P@25 gain over Euclidean');panel(a,'a','Energy and reliability')
for j,metric in enumerate(['Pearson correlation','Spearman correlation']):
 for arm,off,col in [('Euclidean',-.12,BLUE),('PertRank',.12,ORANGE)]:
  x=next(x for x in r if x['measure']==metric and x['arm']==arm);point(a2,val(x,'value'),j+off,c=col)
a2.set_yticks([0,1],['Pearson','Spearman']);a2.set_xlim(.4,.85);a2.set_ylim(1.5,-.5);a2.set_xlabel('Agreement with deep-energy weight')
rr=read('figS3b_directions.csv')
for i,x in enumerate(rr):point(b,val(x,'value'),i,val(x,'low'),val(x,'high'))
b.set_yticks(range(5),[x['directions']for x in rr]);b.set_ylim(4.5,-.6);b.set_xlim(-.02,.055);zero(b);b.set_xlabel('P@25 gain over Euclidean');b.set_ylabel('Discounted control directions');panel(b,'b','Direction specificity')
rr=read('figS3c_nonquery.csv');settings=['Replogle, 20','Replogle, 50','D3, 20']
for j,s in enumerate(settings):
 for k,col,off in [('gene_set_change',BLUE,-.2),('metric_change',ORANGE,0),('reported_sensitivity',PURPLE,.2)]:
  x=next(x for x in rr if x['setting']==s and x['contrast']==k);point(c,val(x,'value'),j+off,val(x,'low'),val(x,'high'),c=col)
c.set_yticks(range(3),settings);c.set_ylim(2.6,-.6);c.set_xlim(-.035,.066);zero(c);c.set_xlabel('Difference in P@25');panel(c,'c','Non-query information')
c.legend(handles=[Line2D([0],[0],color=col,marker='o',lw=1,label=label)for col,label in [(BLUE,'Gene-set change'),(ORANGE,'Matched metric change'),(PURPLE,'Combined change')]],loc='upper left',bbox_to_anchor=(-.02,-.27),frameon=False)
rr=read('figS3d_adjustments.csv');settings=[(donor,guide)for donor in ['D3','D1','D2']for guide in ['A','B']]
for j,(donor,guide)in enumerate(settings):
 for arm,col,off in [('Up only',PURPLE,-.2),('Down only',TEAL,0),('PertRank',ORANGE,.2)]:
  x=next(x for x in rr if x['donor']==donor and x['guide']==guide and x['arm']==arm);point(d,val(x,'value'),j+off,val(x,'low'),val(x,'high'),c=col)
d.set_yticks(range(6),[f'{donor} {g}→{g}'for donor,g in settings]);d.set_ylim(5.6,-.6);d.set_xlim(0,.038);d.grid(axis='x',color='#E8E8E8');d.set_xlabel('P@25 gain over Euclidean');panel(d,'d','Weight adjustments')
d.legend(handles=[Line2D([0],[0],color=col,marker='o',lw=1,label=label)for col,label in [(PURPLE,'Up only'),(TEAL,'Down only'),(ORANGE,'PertRank')]],loc='upper left',bbox_to_anchor=(-.02,-.27),frameon=False)
fig.subplots_adjust(left=.17,right=.98,top=.93,bottom=.18);finish(fig,'figS3_mechanism_v16')
# S4: measured depth curves and archived inverse estimates.
curves=read('figS4_curves.csv');eq=read('figS4_equivalence.csv');fig,axs=plt.subplots(2,2,figsize=(7.09,5.25));axs=axs.ravel()
for ax,donor,letter in zip(axs,['D3','D2','D4'],['a','b','c']):
 rr=[r for r in curves if r['donor']==donor];x=[val(r,'depth')for r in rr]
 for k in range(1,6):ax.plot(x,[val(r,f'anchor{k}')for r in rr],color='#C5C8CB',lw=.7)
 ax.plot(x,[val(r,'raw_mean')for r in rr],color=GRAY,lw=1.8,marker='o',ms=3)
 for method,col in [('Euclidean',BLUE),('PertRank',ORANGE)]:
  y=val(rr[0],method);ax.axhline(y,color=col,ls='--',lw=1)
  e=next(r for r in eq if r['donor']==donor and r['method']==method)
  if val(e,'value')is not None:ax.plot(val(e,'value'),y,'D',color=col,ms=5)
 ax.set_xlabel('Input cells per target');ax.set_ylabel('P@25');ax.set_xticks([20,30,40,50]);panel(ax,letter,donor+' measured depth curve')
ax=axs[3]
for j,donor in enumerate(['D3','D2','D4']):
 for method,col,off in [('Euclidean',BLUE,-.13),('PertRank',ORANGE,.13)]:
  r=next(r for r in eq if r['donor']==donor and r['method']==method);v=val(r,'value');l=val(r,'low');h=val(r,'high')
  if v is None:
   ax.annotate('',xy=(19.1,j+off),xytext=(h,j+off),arrowprops=dict(arrowstyle='->',color=col,lw=1.2));ax.plot(20,j+off,'<',c=col,ms=5)
  else:point(ax,v,j+off,l,h,c=col,marker='D')
ax.set_yticks([0,1,2],['D3','D2','D4']);ax.set_ylim(2.6,-.6);ax.set_xlim(18,44);ax.axvline(20,color='#CCC',lw=.7);ax.set_xlabel('Equivalent Unweighted input cells');panel(ax,'d','Ranking-input equivalents')
fig.legend(handles=[Line2D([0],[0],color=GRAY,lw=1.8,label='Unweighted mean'),Line2D([0],[0],color=BLUE,ls='--',label='Euclidean, 20 cells'),Line2D([0],[0],color=ORANGE,ls='--',label='PertRank, 20 cells')],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.01))
fig.subplots_adjust(left=.1,right=.98,top=.92,bottom=.15,wspace=.38,hspace=.6);finish(fig,'figS4_cell_equivalence_v18')
print('Rendered four SI figures from included archived-value CSVs.')
