"""Render main and retained supplementary figures from verified summaries only.

No fitting, rescoring, resampling, or new inferential calculation is performed.
Run: python figure_code/draw_main_figures.py
"""
from pathlib import Path
import json, csv, hashlib, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
from export_plot_values import export_plot_values

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
D=json.loads((HERE/'figure_values.json').read_text())
B=json.loads((HERE/'design_ablation.json').read_text())
N=json.loads((HERE/'new_figure_values.json').read_text())
V=json.loads((HERE/'program_ranking_values.json').read_text())
RV={r['task']:r for r in V['rvalues']}
P=V['program_external']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10.5,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
C={'pr':'#087F8C','raw':'#7D8791','eu':'#3E5F8A','fr':'#8655A2','ash':'#CB7B27','ink':'#253347','light':'#EDF2F5','red':'#B65258','rv':'#8B6A50'}
DISPLAY=[]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--figures', nargs='+', choices=['1','2','3','4','5','6','S6','S7'], default=['1','2','3','4','5','6','S6','S7'])
SELECT=set(parser.parse_args().figures)
sources=['D3','D1','D2']
def title(ax,letter,txt):
    ax.set_title(txt,loc='left',pad=12,fontweight='bold')
    ax.text(-.12,1.035,letter,transform=ax.transAxes,fontsize=14,fontweight='bold',va='bottom')
def forest(ax,rows,colors=None,labels=True,zero=True):
    for i,r in enumerate(rows):
        v=r['value'];y=len(rows)-1-i;c=colors[i] if isinstance(colors,list) else (colors or C['pr'])
        if len(v)==3:
            assert v[1]<=v[0]<=v[2]
            ax.errorbar(v[0],y,xerr=np.array([[v[0]-v[1]],[v[2]-v[0]]]),fmt='o',color=c,ms=4.8,capsize=2,lw=1.3)
        else:ax.plot(v[0],y,'D',color=c,ms=5)
    ax.set_yticks(range(len(rows)),[r['setting'] for r in rows][::-1] if labels else [])
    ax.set_ylim(-.6,len(rows)-.4);ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
    if zero:ax.axvline(0,color='#AAB2BA',lw=.8,ls='--')
def box(ax,xy,wh,txt,color=None,fs=9):
    x,y=xy;w,h=wh
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.023',fc=color or C['light'],ec='#D1D9DF',lw=.8))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color=C['ink'])
def arrow(ax,p,q,color=None):ax.annotate('',xy=q,xytext=p,arrowprops={'arrowstyle':'-|>','lw':1.2,'color':color or C['ink']})
def save(fig,name):
    # Dense main figures are placed on full pages; retain legible final-size labels.
    minimum={'fig1':10.5,'fig2':10.5,'fig3':11.0,'fig4':11.0,'fig5':10.5,'fig6':10.0}.get(name)
    if minimum:
        for ax in fig.axes:ax.tick_params(axis='both',labelsize=minimum)
        fig.canvas.draw()
        for txt in fig.findobj(match=Text):
            if txt.get_text():txt.set_fontsize(max(minimum,txt.get_fontsize()))
    for ext in ['png','pdf','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=300,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
def bench(group,arm):return float(next(r['point'] for r in D['benchmark_scores'] if r['group']==group and r['arm']==arm and r['K']=='25'))
def shortlabel(s):return s.replace('E4 ','').replace('E5 ','').replace('E1 ','').replace('E3 ','').replace('responsive','resp.').replace('terciles','terc.')

# 1: Method and controlled design. D3 is an illustrative development setting.
if '1' in SELECT:
    fig=plt.figure(figsize=(10.4,10.8))
    a=fig.add_axes([.20,.83,.77,.115]);a.axis('off');a.set_xlim(-.03,1.03);a.set_ylim(0,1)
    title(a,'a','Use control directions to assess shallow responses')
    box(a,(.00,.28),(.22,.43),'Shallow responses\nMatched controls',fs=8.5)
    box(a,(.35,.51),(.31,.45),'Reliability weight\nEffect and noise energies\nin control geometry',color='#DFEFF0',fs=10.5)
    box(a,(.35,.01),(.31,.37),'Query readout\nGene or fixed program effect',fs=10.5)
    box(a,(.80,.28),(.19,.43),'Weighted score\nCandidate list',color='#DFEFF0',fs=8.5)
    arrow(a,(.23,.58),(.34,.73),C['pr']);arrow(a,(.23,.40),(.34,.20),C['eu'])
    arrow(a,(.67,.73),(.79,.57),C['pr']);arrow(a,(.67,.20),(.79,.42),C['eu'])
    b=fig.add_axes([.20,.57,.77,.18]);b.axis('off');b.set_xlim(0,1);b.set_ylim(0,1)
    title(b,'b','Matched 2 × 2 illustration: D3, 20 cells per target')
    m=B['cells']['E4 D3 responsive 20']['means']
    b.text(.34,1,'Original readout $y$',ha='center',va='top');b.text(.78,1,'Compressed readout $Ay$',ha='center',va='top')
    b.text(.00,.71,'Euclidean\nweight $w_0$',va='center',fontsize=9)
    b.text(.00,.23,'Geometry\nweight $w_M$',va='center',fontsize=9)
    for x,y,key in [(.22,.60,'w0y'),(.66,.60,'w0y_AY'),(.22,.12,'W1'),(.66,.12,'wM_AY')]:
        box(b,(x,y),(.24,.22),f'{m[key]:.3f}',color='#D6EDEF' if key=='W1' else None,fs=13)
    for x,k1,k0 in [(.34,'W1','w0y'),(.78,'wM_AY','w0y_AY')]:
        arrow(b,(x,.58),(x,.37),C['pr']);b.text(x+.015,.475,f"{m[k1]-m[k0]:+.4f}",color=C['pr'],va='center',fontweight='bold')
    for y,k1,k0 in [(.71,'w0y_AY','w0y'),(.23,'wM_AY','W1')]:
        arrow(b,(.48,y),(.64,y),C['red']);b.text(.56,y+.09,f"{m[k1]-m[k0]:+.4f}",color=C['red'],ha='center',fontsize=9)
    b.text(.22,-.025,'P@25; arrows show matched differences',fontsize=8,color=C['ink'])
    labels=[shortlabel(r['setting']) for r in D['design']]
    c=fig.add_axes([.20,.10,.32,.37]);d=fig.add_axes([.65,.10,.32,.37])
    forest(c,[{'setting':l,'value':r['geometry']} for l,r in zip(labels,D['design'])],C['eu'])
    title(c,'c','Geometry at the same readout');c.set_xlabel('Geometry − Euclidean, ΔP@25');c.set_xlim(-.004,.052)
    forest(d,[{'setting':l,'value':r['coordinate']} for l,r in zip(labels,D['design'])],labels=False)
    title(d,'d','Original at the same weight');d.set_xlabel('Original − compressed, ΔP@25');d.set_xlim(-.004,.052)
    fig.text(.20,.025,'Four RNA sources, 14 related settings; paired conditional 95% intervals.',fontsize=8,color=C['ink'])
    save(fig,'fig1')

# 2: Fixed-setting and effect-estimator reuse comparisons.
if '2' in SELECT:
    fig=plt.figure(figsize=(10.4,8.0));gs=fig.add_gridspec(2,2,hspace=.65,wspace=.72)
    a=fig.add_subplot(gs[0,0]);forest(a,D['donors'][-2:]);title(a,'a','Fixed-setting geometry gains')
    a.set_xlabel('PertRank − Euclidean, ΔP@25');a.set_xlim(0,.032)
    for i,r in enumerate(D['donors'][-2:]):a.text(r['value'][0],1-i+.22,f"{r['value'][0]:+.4f}",ha='center',fontsize=8)
    a.text(0,-.26,'D4: one held-out donor, two states\n20 cells/target; five inputs/state',transform=a.transAxes,fontsize=8,va='top')
    b=fig.add_subplot(gs[0,1]);methods=[('Unweighted','g002','raw',C['raw']),('rvalues',None,'rvalues',C['rv']),('ASH','g018','ash',C['ash']),('FR rank 100','g002','fr',C['fr']),('Euclidean','g002','euclidean',C['eu']),('PertRank','g002','pertrank',C['pr'])]
    vals=[(np.mean([float(RV[t]['rvalues']) for t in ['D4 Rest','D4 Stim']]) if k=='rvalues' else bench(g,k)) for _,g,k,_ in methods];b.barh(range(len(methods)),vals,color=[cc for *_,cc in methods],height=.58)
    b.set_yticks(range(len(methods)),[z[0] for z in methods]);b.invert_yaxis();b.set_xlim(0,.455);b.set_xlabel('P@25');title(b,'b','Complete-workflow comparison')
    for i,v in enumerate(vals):b.text(v+.006,i,f'{v:.3f}',va='center',fontsize=9)
    c=fig.add_subplot(gs[1,0]);xx=np.arange(3)
    for keys,col,lab,off in [(['raw','euclidean','pertrank'],C['eu'],'Original',-.17),(['fr','fr_euclidean','fr_pertrank'],C['fr'],'FR rank 300',.17)]:
        vv=[bench('g005',k) for k in keys];c.bar(xx+off,vv,width=.31,color=col,label=lab)
        for x,v in zip(xx+off,vv):c.text(x,v+(.009 if off<0 else .037),f'{v:.3f}',ha='center',fontsize=10.5)
    c.set_ylim(0,.49);c.set_xticks(xx,['None','Euclidean','Geometry']);c.set_xlabel('Reliability weight');c.set_ylabel('P@25');c.legend(frameon=False,fontsize=10.5,ncol=2,loc='upper left',columnspacing=.8,handlelength=1.2);title(c,'c','Reuse across effect estimators')
    d=fig.add_subplot(gs[1,1]);g=D['interactions']['A_geometry_interaction']['equal_state']['25']
    rows=[]
    for key,lab in [('raw_geometry','Original effects'),('FR_geometry','FR rank 300 effects')]:rows.append({'setting':lab,'value':[g[key]['point']]+g[key]['descriptive_conditional_95_interval']})
    forest(d,rows,[C['eu'],C['fr']]);title(d,'d','Matched geometry increments');d.set_xlabel('Geometry − Euclidean, ΔP@25');d.set_xlim(0,.027)
    for i,r in enumerate(rows):d.text(r['value'][0],1-i+.22,f"{r['value'][0]:+.4f}",ha='center',fontsize=8)
    fig.subplots_adjust(left=.17,right=.98,top=.93,bottom=.12)
    fig.text(.17,.025,'b–d: Rest and stimulated states equally weighted. a,d: paired conditional 95% intervals.',fontsize=8)
    save(fig,'fig2')

# 3: Independent RNA, native estimates and all-source four-arm readout results.
if '3' in SELECT:
    fig=plt.figure(figsize=(10.4,9.5));gs=fig.add_gridspec(2,2,hspace=.65,wspace=.72)
    sources=['D3','D1','D2'];ns=[s+'_Rest' for s in sources]
    a=fig.add_subplot(gs[0,0]);keys=['raw','Euclidean','PertRank','FR','FR_x_Euclidean','FR_x_PertRank','ASH','rvalues'];labs=['Unweighted','Euclidean','PertRank','FR rank 300','FR × Euclidean','FR × PertRank','ASH','rvalues']
    mat=np.array([[(float(RV['External RNA '+n[:2]]['rvalues']) if k=='rvalues' else D['rna'][n]['means'][k]['P10']) for n in ns] for k in keys]);a.imshow(mat,cmap='YlGnBu',vmin=.21,vmax=.28,aspect='auto')
    a.set_xticks(range(3),['D3 (52)','D1 (59)','D2 (63)']);a.set_yticks(range(len(keys)),labs);a.set_xlabel('Input source (candidate count)');title(a,'a','Complete candidate pools: P@10')
    for i in range(len(keys)):
        for j in range(3):a.text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',fontsize=8.5,color='white' if mat[i,j]>.257 else C['ink'])
    b=fig.add_subplot(gs[0,1]);rows=[];cols=[]
    for key,col,label in [('PertRank_minus_FR',C['fr'],'PR − FR'),('PertRank_minus_ASH',C['ash'],'PR − ASH'),('rvalues',C['rv'],'PR − rvalues')]:
        for n in ns:
            
            if key=='rvalues':
                q=RV['External RNA '+n[:2]];val=[float(q[z]) for z in ['difference','ci95_lower','ci95_upper']]
            else:
                q=D['rna'][n]['contrasts'][key];val=[q['P10_difference']]+q['conditional_query_block_95_interval']
            rows.append({'setting':n[:2]+'  '+label,'value':val});cols.append(col)
    forest(b,rows,cols);title(b,'b','Complete-workflow differences');b.set_xlabel('ΔP@10');b.set_xlim(-.008,.05)
    c=fig.add_subplot(gs[1,0]);rows=[]
    for s in sources:
        q=N['four_arm'][s]['contrasts']['readout_original_minus_compressed_at_wM'];rows.append({'setting':s,'value':[q['P10_difference']]+q['conditional_95_interval']})
    forest(c,rows,[C['pr'],C['red'],C['pr']]);title(c,'c','Readout comparison at fixed weight');c.set_xlabel('Original − compressed, ΔP@10');c.set_xlim(-.018,.01);c.set_xticks([-.015,-.005,.005])
    d=fig.add_subplot(gs[1,1]);rows=[];cols=[]
    for key,col,label in [('geometry_on_original',C['eu'],'Original'),('geometry_on_compressed',C['pr'],'Compressed')]:
        for s in sources:
            q=N['four_arm'][s]['contrasts'][key];rows.append({'setting':s+'  '+label,'value':[q['P10_difference']]+q['conditional_95_interval']});cols.append(col)
    forest(d,rows,cols);title(d,'d','Geometry at each readout');d.set_xlabel('Geometry − Euclidean, ΔP@10');d.set_xlim(-.008,.025);d.set_xticks([-.005,.005,.015,.025])
    fig.subplots_adjust(left=.17,right=.98,top=.93,bottom=.13)
    fig.text(.17,.025,'Three input sources share one arrayed knockout RNA experiment.\nAll intervals are paired conditional query-block 95% intervals; c,d use the complete four-arm analysis.',fontsize=8)
    save(fig,'fig3')

# 4: Directional reliability diagnostics; guide response accounting appears in Supplementary Fig. 6.
if '4' in SELECT:
    fig=plt.figure(figsize=(10.4,7.9));gs=fig.add_gridspec(2,2,hspace=.75,wspace=.80)
    a=fig.add_subplot(gs[0,0]);forest(a,D['energy'],[C['pr'],C['raw']]);title(a,'a','Independent energy diagnostic');a.set_xlabel('Gain over Euclidean, ΔP@25');a.set_xlim(0,.065)
    b=fig.add_subplot(gs[0,1]);forest(b,D['direction']);title(b,'b','Direction-specific interventions');b.set_xlabel('D3 gain over Euclidean, ΔP@25');b.set_xlim(-.021,.055)
    c=fig.add_subplot(gs[1,0]);forest(c,D['nonquery'],[C['pr'],C['pr'],C['eu']]);title(c,'c','Query exclusion and suppression');c.set_xlabel('Matched geometry increment, ΔP@25');c.set_xlim(0,.065)
    c.text(0,-.29,'Suppression subset: 291 targets\n68.5% of the full-pool increment',transform=c.transAxes,fontsize=8,va='top')
    d=fig.add_subplot(gs[1,1]);al=D['alignment'];yy=np.arange(len(al));d.barh(yy,[r['gains'] for r in al],height=.5,color=C['pr'],label='Supported gains');d.barh(yy,[-r['losses'] for r in al],height=.5,color=C['red'],label='Supported losses');d.scatter([r['value'][0] for r in al],yy,c=C['ink'],s=20,zorder=4,label='Net')
    d.set_yticks(yy,[r['setting'] for r in al]);d.invert_yaxis();d.axvline(0,color='#AAB2BA',lw=.8);d.set_xlim(-.4,.65);d.set_xlabel('Reference members per 25');title(d,'d','Gains and costs by alignment');d.legend(frameon=False,fontsize=11,loc='upper right',bbox_to_anchor=(1.03,-.24),handlelength=1.2,labelspacing=.35)
    fig.subplots_adjust(left=.24,right=.98,top=.92,bottom=.16)
    save(fig,'fig4')

# 5: Frozen program nomination and the matched component contrasts.
if '5' in SELECT:
    fig=plt.figure(figsize=(10.4,12.2))
    a=fig.add_axes([.13,.80,.83,.145]);a.axis('off');a.set_xlim(0,1);a.set_ylim(0,1)
    title(a,'a','Nominate drugs for six predefined transcriptional programs')
    a.text(.00,.95,'sci-Plex3 · 24 h · 1,000 nM · K = 10',fontsize=10,color=C['ink'],va='top')
    box(a,(.00,.36),(.22,.40),'A549: 91 candidates\nK562: 130 candidates\nMCF7: 174 candidates',fs=9)
    box(a,(.31,.51),(.27,.25),'Culture 1 input (20 cells)',color='#DFEFF0',fs=9)
    box(a,(.31,.08),(.27,.25),'Culture 2 input (20 cells)',color='#DFEFF0',fs=9)
    box(a,(.72,.51),(.27,.25),'Culture 2 reference (60)',fs=9)
    box(a,(.72,.08),(.27,.25),'Culture 1 reference (60)',fs=9)
    arrow(a,(.23,.63),(.30,.63));arrow(a,(.23,.43),(.30,.20))
    arrow(a,(.59,.64),(.71,.64),C['pr']);arrow(a,(.59,.20),(.71,.20),C['pr'])
    a.text(.645,.79,'Rank → evaluate',ha='center',fontsize=8.5)
    a.text(.00,-.10,'Matched vehicle controls; six fixed programs:\nIFN-γ, TNFA/NFKB, IL2/STAT5, E2F, G2M and OXPHOS.',fontsize=8.6)

    b=fig.add_axes([.21,.53,.31,.205]);title(b,'b','All six constructions')
    armkeys=['raw_one','raw_w0','raw_wM','ash_one','ash_w0','ash_wM']
    armlabs=['Original · none','Original · Euclidean','Original · geometry','ASH · none','ASH · Euclidean','ASH · geometry']
    vals=[P['method_results'][k]['estimate'] for k in armkeys]
    cols=[C['raw'],C['eu'],C['pr'],C['raw'],C['eu'],C['pr']]
    b.barh(range(6),vals,color=cols,height=.58);b.set_yticks(range(6),armlabs);b.invert_yaxis()
    b.set_xlim(0,.187);b.set_xlabel('Mean reference response, U')
    for i,v in enumerate(vals):b.text(v+.003,i,f'{v:.3f}',va='center',fontsize=8.4)
    

    c=fig.add_axes([.69,.53,.27,.205]);title(c,'c','Application differences')
    ap=P['application']; labels=['IFN-γ','TNFA/NFKB','IL2/STAT5','E2F','G2M','OXPHOS']
    rows=[{'setting':'Overall (97.5%)','value':[ap['overall']['estimate']]+ap['overall']['conditional_interval']}]
    rows += [{'setting':l,'value':[ap['by_program'][k]['estimate']]+ap['by_program'][k]['conditional_interval']} for k,l in zip(P['programs'],labels)]
    rows += [{'setting':k,'value':[ap['by_cell_line'][k]['estimate']]+ap['by_cell_line'][k]['conditional_interval']} for k in P['cell_lines']]
    forest(c,rows,[C['ink']]+[C['pr']]*6+[C['eu']]*3)
    c.axhline(2.5,color='#D1D9DF',lw=.9);c.axhline(8.5,color='#D1D9DF',lw=.9);c.set_xlim(-.035,.100);c.set_xticks([-.025,0,.025,.05,.075,.1]);c.tick_params(axis='x',labelsize=7.5)
    c.set_xlabel('PertRank − native ASH, ΔU')
    

    d=fig.add_axes([.21,.285,.75,.145]);title(d,'d','Geometry increments depend on the readout and source')
    rows=[]
    for key,label in [('original_geometry','D3 · original (95%)'),('ASH_geometry','D3 · ASH (95%)')]:
        q=V['program_D3'][key];rows.append({'setting':label,'value':[q['delta_utility']]+q['conditional_interval']})
    for key,label in [('geometry_on_raw_readout','sci-Plex3 · original (95%)'),('geometry_on_ASH_readout','sci-Plex3 · ASH (97.5%)')]:
        q=P['geometry'][key];rows.append({'setting':label,'value':[q['estimate']]+q['conditional_interval']})
    forest(d,rows,[C['eu'],C['pr'],C['eu'],C['pr']]);d.axhline(1.5,color='#D1D9DF',lw=.9)
    d.set_xlim(-.019,.039);d.set_xlabel('Geometry − Euclidean, ΔU')
    for i,row in enumerate(rows):d.text(.037,3-i,f"{row['value'][0]:+.3g}",ha='right',va='center',fontsize=8.5)
    d.text(0,-.30,'D3: development analysis. sci-Plex3: evaluation frozen before reference scoring.',transform=d.transAxes,fontsize=8,va='top')

    e=fig.add_axes([.21,.075,.75,.125]);title(e,'e','Which candidates enter and leave the list?')
    keys=['application_raw_PertRank_minus_native_ASH','geometry_on_ASH_readout']
    labs=['PertRank − native ASH','ASH: geometry − Euclidean'];yy=np.array([1.,0.])
    for off,field,label,col in [(.15,'mass_weighted_entering_reference_mean','Entering',C['pr']),(-.15,'mass_weighted_leaving_reference_mean','Leaving',C['red'])]:
        v=[P['replacement'][k][field] for k in keys];e.barh(yy+off,v,height=.26,color=col,label=label)
        for y,x in zip(yy+off,v):e.text(x+.0015,y,f'{x:.4f}',va='center',fontsize=8)
    e.set_yticks(yy,labs);e.set_xlim(0,.125);e.set_xlabel('Replacement-weighted mean reference response')
    e.legend(frameon=False,ncol=2,loc='upper right',fontsize=8,bbox_to_anchor=(1,1.30))
    for y,k in zip(yy,keys):
        q=P['replacement'][k]
        e.text(.123,y,f"Replaced: {q['mean_replacement_mass']:.3g}/10\nNet ΔU: {q['mean_utility_difference']:+.4f}",ha='right',va='center',fontsize=8.3)
    fig.text(.13,-.015,'Response units: input-control program standard deviations.\nIntervals condition on fixed inputs, lists and cultures; primary comparisons use 97.5%.\nFull task results and P@10 stability ranges: Supplementary Note SN26.',fontsize=8)
    save(fig,'fig5')

# 6: Input efficiency, inspectable nomination and reuse in the target screen.
if '6' in SELECT:
    fig=plt.figure(figsize=(10.4,11.5));gs=fig.add_gridspec(3,2,hspace=.85,wspace=.76)
    a=fig.add_subplot(gs[0,0]);forest(a,D['equivalence'],zero=False);a.axvline(20,c=C['raw'],ls='--',lw=.9);a.set_xlim(18,45)
    a.set_xlabel('Unweighted cells matching\n20-cell PertRank');title(a,'a','Ranking-input equivalents')
    b=fig.add_subplot(gs[0,1]);r=D['d4_depth'];xx=r['depths'];b.plot(xx,r['raw_curve'],'o-',color=C['raw'],ms=4,label='Unweighted')
    for meth,col in [('Euclidean',C['eu']),('PertRank',C['pr'])]:
        q=r['methods'][meth];b.hlines(q['target_score'],20,q['n_equivalent'],color=col,lw=1.2,ls='--');b.plot(q['n_equivalent'],q['target_score'],'o',color=col,label=meth+'20')
    b.set_xlabel('Ranking cells per target');b.set_ylabel('P@25');b.legend(frameon=False,fontsize=8,loc='lower right');title(b,'b','D4 Rest depth calibration')
    b.text(.03,.97,'PertRank20 ≈ 38.9 cells\nGeometry adds 7.07\ncell equivalents',transform=b.transAxes,va='top',fontsize=8.5)
    c=fig.add_subplot(gs[1,0]);z=D['interactions']['B_depth_interaction']['25'];xx=[20,50,100];vs=[z['n'+str(x)]['point'] for x in xx];cis=[z['n'+str(x)]['descriptive_conditional_95_interval'] for x in xx]
    c.errorbar(xx,vs,yerr=np.array([[v-ci[0] for v,ci in zip(vs,cis)],[ci[1]-v for v,ci in zip(vs,cis)]]),fmt='o-',color=C['pr'],capsize=3);c.axhline(0,c=C['raw'],lw=.8,ls='--');c.set_xticks(xx);c.set_ylim(-.02,.035)
    c.set_xlabel('Ranking cells per target');c.set_ylabel('PertRank − ASH, ΔP@25');title(c,'c','Depth dependence in D3')
    d=fig.add_subplot(gs[1,1]);xx=np.arange(2);width=.33
    for key,label,col,off in [('Euclidean_P25','Euclidean',C['eu'],-width/2),('PertRank_P25','PertRank',C['pr'],width/2)]:
        vv=[float(r[key]) for r in N['CTLA4']];d.bar(xx+off,vv,width,color=col,label=label)
        for x,v in zip(xx+off,vv):d.text(x,v+.015,f'{v:.2f}',ha='center',fontsize=9)
    d.set_xticks(xx,['CTLA4 up','CTLA4 down']);d.set_ylim(0,.84);d.set_ylabel('P@25');d.legend(frameon=False,fontsize=8,ncol=2,loc='upper right');title(d,'d','Inspect a nominated list')
    d.text(0,-.28,'First fixed input; all five inputs and both\ndirections are reported in Source Data.',transform=d.transAxes,fontsize=8,va='top')
    e=fig.add_subplot(gs[2,0]);e.axis('off');e.set_xlim(-.01,1.01);e.set_ylim(-.06,1);title(e,'e','From screen to candidate table')
    box(e,(.02,.73),(.96,.22),'D4 Stim: controls + 20 cells/target\n561 candidate perturbations',fs=8.5)
    box(e,(.02,.35),(.43,.23),'Query: CTLA4\nUp / down',fs=8.5)
    box(e,(.57,.35),(.41,.23),'PertRank score\n25 candidates',color='#DFEFF0',fs=8.5)
    box(e,(.02,-.03),(.96,.23),'Effect · weight · reference support\nTarget expression and suppression',fs=8.5)
    arrow(e,(.23,.72),(.23,.59));arrow(e,(.46,.46),(.56,.46));arrow(e,(.78,.34),(.78,.21))
    f=fig.add_subplot(gs[2,1]);t=D['transfer'];rows=[{'setting':lab,'value':[t['contrasts'][k]['delta_P25']]+t['contrasts'][k]['conditional_95pct_interval']} for k,lab in [('PertRank-raw','Over Unweighted'),('PertRank-Euclidean','Over Euclidean'),('PertRank-PR_M','Over Pearson')]]
    forest(f,rows,[C['raw'],C['eu'],C['pr']]);f.set_xlim(0,.10);f.set_xlabel('Transferred-panel difference, ΔP@25');title(f,'f','Reuse a fixed query panel')
    f.text(0,-.22,'300 Rest queries → D4 Stim\nFive inputs; 561 candidates; K = 25',transform=f.transAxes,fontsize=8,va='top')
    fig.subplots_adjust(left=.19,right=.98,top=.94,bottom=.10)
    
    save(fig,'fig6')


# S6: Descriptive guide diagnostic formerly shown as main Figure 4e.
if 'S6' in SELECT:
    fig=plt.figure(figsize=(9.6,3.5));a=fig.add_axes([.04,.16,.93,.74]);a.axis('off')
    a.set_title('Response strength and guide agreement are distinct quantities',loc='left',pad=15,fontweight='bold',fontsize=12)
    a.text(.00,.68,r'$\mathcal{C}=\mathcal{B}-\mathcal{D}/4$',fontsize=21,color=C['pr'],va='center')
    a.text(.31,.80,r'$\mathcal{B}=\|(\theta_1+\theta_2)/2\|_M^2/G$   average-response energy',fontsize=11)
    a.text(.31,.51,r'$\mathcal{D}=\|\theta_1-\theta_2\|_M^2/G$   guide-difference energy',fontsize=11)
    a.text(.31,.22,r'$\mathcal{C}=\langle\theta_1,\theta_2\rangle_M/G$   cross-guide agreement',fontsize=11)
    g=N['guide_diagnostic']
    a.text(.02,-.14,f"Both guides had positive held-out response-energy point estimates in {g['both_guide_heldout_energy_positive_points']}/{g['n_measurable']} measurable\nzero-weight entries from the earlier cross-guide construction.",fontsize=9,color=C['ink'])
    save(fig,'figS6_guide_response')

# S7: Retained supervised development result formerly shown as main Figure 6d.
if 'S7' in SELECT:
    fig=plt.figure(figsize=(7.8,4.2));a=fig.add_axes([.16,.34,.78,.49]);rows=[]
    for source in sources:
        q=N['Q'][source]['contrast'];rows.append({'setting':source,'value':[q['delta']]+q['conditional95']})
    forest(a,rows);a.set_xlim(-.002,.0054);a.set_xticks([-.002,0,.002,.004]);a.ticklabel_format(style='plain',axis='x')
    a.set_xlabel('PertRank-Q − PertRank, ΔP@10')
    a.set_title('Reference-trained query refinement',loc='left',pad=15,fontweight='bold',fontsize=12)
    fig.text(.16,.14,'Whole-target holdouts within candidate subpools; K = 10\nD3: 26/26 candidates; D1: 32/27; D2: 35/28',fontsize=9)
    fig.text(.16,.045,'Paired query-block 95% intervals condition on the retained models\nand shared external RNA reference.',fontsize=9)
    save(fig,'figS7_query_refinement')

# Export exact plotted values. CSV rows correspond to existing inputs; no new interval is produced.
m=B['cells']['E4 D3 responsive 20']['means']
for key in ['w0y','w0y_AY','W1','wM_AY']:DISPLAY.append(['1b',key,m[key],'',''])
for i,r in enumerate(D['design']):
    for ckey in ['geometry','coordinate']:DISPLAY.append(['1'+('c' if ckey=='geometry' else 'd'),r['setting'],*r[ckey]])
for s in sources:
    for key,q in N['four_arm'][s]['contrasts'].items():DISPLAY.append(['3',s+'/'+key,q['P10_difference'],*q['conditional_95_interval']])
    q=N['Q'][s]['contrast'];DISPLAY.append(['S7',s,q['delta'],*q['conditional95']])
with (HERE/'plotted_new_contrasts.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['panel','comparison','estimate','conditional95_lower','conditional95_upper']);w.writerows(DISPLAY)
manifest={'scope':'Existing verified results only; no new fit, bootstrap, rescore, or hypothesis test.','inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'figure_values.json',HERE/'design_ablation.json',HERE/'new_figure_values.json',HERE/'program_ranking_values.json']},'figures':[f'fig{i}.pdf' for i in range(1,7)]+['figS6_guide_response.pdf','figS7_query_refinement.pdf'], 'new_exact_values_csv':'plotted_values.csv', 'source_map':'PANEL_SOURCE_MAP.csv'}
manifest['new_exact_plot_value_rows']=export_plot_values(V,HERE)
manifest['figure_typography']={'fig1_to_2_minimum_source_pt':10.5,'fig3_to_4_minimum_source_pt':11.0,'fig5_minimum_source_pt':10.5,'fig6_minimum_source_pt':10.0,'placement':'Figures 5 and 6 occupy separate full-width manuscript pages; legends follow on the next page.'}
manifest['script_sha256']={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in [HERE/'draw_main_figures.py',HERE/'prepare_plot_inputs.py',HERE/'export_plot_values.py']}
(HERE/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Rendered selected figures as PNG, PDF and SVG: '+', '.join(sorted(SELECT)))
