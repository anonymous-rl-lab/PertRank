"""Plot completed D3 rank comparisons; development intervals are descriptive."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'report_assets'
OUT.mkdir(exist_ok=True)
winner_path = HERE / 'expanded_rank_transfer/POSTHOC_EXPANDED_RANK_WINNER.json'
winner = json.loads(winner_path.read_text())
paths = [ROOT / 'evaluation/D3_Rest/rank100_default_lasso/evaluation_report.json']
for evidence in winner['candidate_evidence']:
    p = Path(evidence['score_file'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == evidence['score_sha256']
    assert evidence['audit_pass']
    paths.append(p)
reports = [json.loads(p.read_text()) for p in paths]
ranks = np.array([r['config']['rank'] for r in reports])
assert ranks.tolist() == [100, 150, 200, 300]
arms = [r['results']['25']['arms'] for r in reports]
for arm in ['raw', 'euclidean', 'pertrank']:
    assert max(a[arm]['point'] for a in arms)-min(a[arm]['point'] for a in arms) < 1e-12
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'svg.fonttype':'none','savefig.facecolor':'white'})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5,4.45), gridspec_kw={'width_ratios':[1.4,1]})
colors = {'raw':'#60717E','euclidean':'#AFB7BE','pertrank':'#007C78','fr':'#2864B0','fr_pertrank':'#CB8035'}
labels = {'raw':'Raw','euclidean':'Euclidean (internal)','pertrank':'PertRank','fr':'Native FR','fr_pertrank':'FR × PertRank weight'}
for arm in ['pertrank','raw','euclidean']:
    ax.axhline(arms[0][arm]['point'], color=colors[arm], ls='--' if arm != 'pertrank' else '-', lw=1.5, label=labels[arm])
for arm in ['fr','fr_pertrank']:
    values = [a[arm]['point'] for a in arms]
    ax.plot(ranks,values,'o-',color=colors[arm],lw=2,label=labels[arm])
ax.set(xlabel='FR rank',ylabel='P@25',xticks=ranks,ylim=(.25,.314), title='A  D3 Rest: matched single input')
ax.legend(loc='lower center',frameon=False,ncol=2,fontsize=8)
ax.grid(axis='y',color='#DFE5E9',lw=.6)
points=[];lo=[];hi=[]
for r in reports:
    c=r['results']['25']['contrasts']['pertrank_minus_fr'];points.append(c['point']);lo.append(c['query_only_95_interval'][0]);hi.append(c['query_only_95_interval'][1])
points=np.array(points)
bx.errorbar(points,np.arange(4),xerr=[points-np.array(lo),np.array(hi)-points],fmt='o',color=colors['pertrank'],capsize=3,lw=1.8)
bx.axvline(0,color='#9DAAB2',lw=1)
bx.set(yticks=np.arange(4),yticklabels=[f'Rank {k}' for k in ranks],xlabel='PertRank − native FR (P@25)',xlim=(-.003,.06),title='B  Paired development differences')
bx.invert_yaxis();bx.grid(axis='x',color='#DFE5E9',lw=.6)
fig.subplots_adjust(bottom=.28,wspace=.35,top=.85,left=.08,right=.98)
fig.text(.08,.08,'1,200 candidates • 300 queries • 20 cells per target • one fixed D3 input\n95% query-block intervals are descriptive after development/selection; rank 300 remains a search boundary.',fontsize=9,color='#405262',linespacing=1.6)
for suffix in ['png','svg']:
    fig.savefig(OUT / f'D3_rank_boundary_20261014.{suffix}',dpi=200)
source={'figure':'D3_rank_boundary_20261014','sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'ranks':ranks.tolist(),'native_fr_p25':[a['fr']['point'] for a in arms],
        'pertRank_minus_fr_p25':points.tolist(),'inference':'Descriptive development/selection comparisons; no independent confirmation or global optimum claimed.'}
(OUT / 'D3_rank_boundary_20261014.json').write_text(json.dumps(source,indent=2)+'\n')
print(OUT / 'D3_rank_boundary_20261014.png')
