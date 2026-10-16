#!/usr/bin/env python3
"""Assemble a dated evidence snapshot from named, completed result files.
No model execution, expression-matrix access, or evaluation recomputation.
"""
from pathlib import Path
import json, hashlib, sys, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
RES=HERE.parent
ROOT=RES.parent
OUT=RES/'report_assets'; OUT.mkdir(exist_ok=True)
SOURCES=[]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,label=None):
 p=Path(p); d=json.loads(p.read_text()); SOURCES.append({'label':label or p.name,'path':str(p),'sha256':sha(p)}); return d
def check_sha(p,h):
 assert sha(p)==h, f'Changed source {p}'
def f(x):return f'{x:.4f}'
def signed(x):return f'{x:+.4f}'
def ci(v):return f'[{v[0]:+.4f}, {v[1]:+.4f}]'
def p25(d,a):return d['results']['25']['arms'][a]['point']
ARMS=['raw','euclidean','pertrank','fr','fr_euclidean','fr_pertrank']
NAMES={'raw':'Raw（不加权）','euclidean':'Euclidean（内部消融）','pertrank':'PertRank','fr':'原生 FR-Perturb','fr_euclidean':'FR × Euclidean 权重','fr_pertrank':'FR × PertRank 权重','ash':'ASH（官方默认）'}
# Six depth reports: score owner, hash binding, independent audit and relative contrasts.
DEPTH={}; REL={}
for curve in ['A_native_default_penalty','B_effective_penalty_sensitivity']:
 DEPTH[curve]={}; REL[curve]={}
 for n in [20,50,100]:
  q=ROOT/f'posthoc_depth_20261014/evaluation/{curve}/n{n}'
  o=read(q/'SCORE_OWNER_COMPLETION.json',f'Depth {curve} n={n}: owner')
  assert o['owner_exit_code']==0 and o['npz_crc']=='PASS'
  check_sha(q/'evaluation_report.json',o['files_sha256']['evaluation_report.json'])
  r=read(q/'evaluation_report.json',f'Depth {curve} n={n}')
  assert r['status']=='COMPLETE_INDEPENDENT_AUDIT_PASS'
  assert all(x['status']=='PASS' for x in r['independent_audit'].values())
  rr=read(q/'descriptive_relative_to_raw.json',f'Depth {curve} n={n}: method minus raw')
  check_sha(q/'evaluation_report.json',rr['report_sha256'])
  check_sha(q/'evaluation_arrays.npz',rr['arrays_sha256'])
  DEPTH[curve][n]=r; REL[curve][n]=rr
for n in [20,50,100]:
 a,b=[DEPTH[c][n] for c in DEPTH]
 for arm in ['raw','euclidean','pertrank']:assert p25(a,arm)==p25(b,arm)
# Scientific plotting uses absolute scores and keeps the A/B sensitivity panels separate.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
colors={'raw':'#697885','euclidean':'#A2A9B0','pertrank':'#007C78','fr':'#2864B0','fr_euclidean':'#9B70A6','fr_pertrank':'#CB8035'}
labels={'raw':'Raw','euclidean':'Euclidean (internal)','pertrank':'PertRank','fr':'Native FR','fr_euclidean':'FR × Euclidean weight','fr_pertrank':'FR × PertRank weight'}
fig,axes=plt.subplots(1,2,figsize=(11.5,5.25),sharey=True)
for ax,(curve,items),title in zip(axes,DEPTH.items(),['A  Fixed native penalty','B  Effective-penalty sensitivity']):
 for arm in ARMS:
  ax.plot([20,50,100],[p25(items[n],arm) for n in [20,50,100]],marker={'raw':'s','euclidean':'v','pertrank':'o','fr':'D','fr_euclidean':'^','fr_pertrank':'P'}[arm],ms=5.5,lw=2 if arm in ['pertrank','fr'] else 1.5,linestyle='--' if arm in ['euclidean','fr_euclidean'] else '-',color=colors[arm],label=labels[arm],alpha=.9)
 ax.set(xlabel='Input cells per target',xticks=[20,50,100],ylim=(.345,.57),title=title)
 ax.grid(axis='y',color='#E2E7EB',lw=.6)
axes[0].set_ylabel('P@25')
handles,labels2=axes[0].get_legend_handles_labels()
fig.legend(handles,labels2,loc='lower center',bbox_to_anchor=(.5,.115),ncol=3,frameon=False,fontsize=9)
fig.subplots_adjust(left=.07,right=.98,bottom=.30,top=.87,wspace=.18)
fig.text(.07,.045,'250 D3 targets • 300 queries • one nested 20/50/100-cell chain • rank 100\nA: Lasso α = 10⁻⁴. B: αN/n = 0.13. Dependent depths; post hoc diagnostic, not biological replication.',fontsize=9,color='#405262',linespacing=1.5)
for ext in ['png','svg']:fig.savefig(OUT/f'D3_depth_native_FR_20261014.{ext}',dpi=200)
plt.close(fig)
fig,ax=plt.subplots(figsize=(9.8,4.8))
for j,arm in enumerate(['pertrank','fr','fr_pertrank']):
 x=np.array([20,50,100],float)+(j-1)*1.7
 vals=[REL['A_native_default_penalty'][n]['results']['25'][arm+'_minus_raw'] for n in [20,50,100]]
 pts=np.array([v['point'] for v in vals]); lo=np.array([v['query_only_95_interval'][0] for v in vals]);hi=np.array([v['query_only_95_interval'][1] for v in vals])
 ax.errorbar(x,pts,yerr=[pts-lo,hi-pts],fmt='o-',capsize=3,ms=5,lw=1.8,color=colors[arm],label=labels[arm]+' − Raw')
ax.axhline(0,color='#697885',lw=1)
ax.set(xlabel='Input cells per target (points offset for clarity)',ylabel='Paired P@25 difference',xticks=[20,50,100],title='Depth diagnostic A: relative to raw estimates')
ax.grid(axis='y',color='#E2E7EB',lw=.6);ax.legend(frameon=False,loc='lower left',fontsize=9)
fig.subplots_adjust(left=.10,right=.98,bottom=.26,top=.86)
fig.text(.10,.06,'95% descriptive query-block intervals, conditional on fixed input chain and reference.\nNo depth-interaction test or study-wide multiplicity adjustment. At n=100, PertRank − Raw 99% interval spans zero.',fontsize=9,color='#405262',linespacing=1.5)
for ext in ['png','svg']:fig.savefig(OUT/f'D3_depth_relative_to_raw_20261014.{ext}',dpi=200)
plt.close(fig)
# D3 ranks and reconciliation.
REC=read(RES/'D3_TASK_RECONCILIATION.json','D3 matched input reconciliation');assert REC['status']=='PASS'
R100=read(ROOT/'evaluation/D3_Rest/rank100_default_lasso/evaluation_report.json','Original D3 rank 100 restored report')
rank_exec=read(RES/'RANK_BOUNDARY_EXECUTION.json','New D3 rank extension execution')
RANK=[R100]
for e in rank_exec['fits']:
 assert e['status']=='COMPLETE' and e['audit_pass'] and e['score_owner_exit_code']==0
 p=Path(e['evaluation_directory'])/'evaluation_report.json';check_sha(p,e['score_report_sha256']);RANK.append(read(p,f'New D3 rank {e["config"]["rank"]}'))
assert [r['config']['rank'] for r in RANK]==[100,150,200,300]
ASH_D3=read(ROOT/'ash_work/resume_20261014/D3_DEFAULT_REPLAY_CONFIRMATION.json','D3 default ASH restored posterior confirmation')
assert ASH_D3['status']=='DEFAULT_REPLAY_CONFIRMED_HISTORICAL_FIXED_CHOICE' and ASH_D3['replay_minus_historical_P25']==0
# Old D4 comparisons: this is restored output, never an additional replication.
D4={}
for state in ['D4_Rest','D4_Stim8hr']:
 q=RES/'evaluation'/state;r=read(q/'evaluation_report.json',state+' original FR restored result');audit=read(q/'audit/AUDIT_RESULT.json',state+' score audit')
 assert r['status']=='COMPLETE' and audit['status']=='PASS';D4[state]=r
EQ=read(RES/'evaluation/equal_state/equal_state_report.json','Restored original FR equal-state summary')
o=read(RES/'evaluation/equal_state/OWNER_COMPLETION.json','Equal-state summary owner')
assert o['owner_exit_code']==0
check_sha(RES/'evaluation/equal_state/equal_state_report.json',o['outputs_relative_to_parent']['equal_state/equal_state_report.json'])
EX=read(RES/'evaluation/equal_state/requested_extra_contrasts.json','Descriptive PertRank minus weighted FR')
GP=read(RES/'GPerturb_REAL_D3_RESOURCE_GATE_RESULT.json','New GPerturb full-input resource gate')
assert GP['status']=='RESOURCE_BLOCKED_NOT_A_METHOD_PERFORMANCE_FAILURE'
# ASH: only explicitly complete, published and post-owner verified tasks enter tables.
ASH={}
for task in ['D4_Rest_pooled','D4_Stim8hr_pooled']:
 q=ROOT/'ash_work/resume_20261014/D4_scores'/task
 if not (q/'POST_EXEC_VERIFIED.json').exists():continue
 v=read(q/'POST_EXEC_VERIFIED.json',task+' ASH post-owner integrity')
 assert v['status']=='PASS' and v['owner_exit_code']==0
 r=read(q/'evaluation_report.json',task+' completed ASH comparison')
 match=read(q/'COMMON_THREE_ARM_MATCH.json',task+' common-arm identity')
 assert match['status']=='MATCHED_COMMON_THREE_ARMS_BITWISE_PASS'
 check_sha(q/'evaluation_report.json',match['ash_report_sha256'])
 assert all(c['exact_array_equal'] for c in match['checks'].values())
 ASH[task]=r
q=ROOT/'ash_work/resume_20261014/D4_scores/D4_pooled_combined'
if (q/'POST_EXEC_VERIFIED.json').exists():
 v=read(q/'POST_EXEC_VERIFIED.json','ASH two-state summary post-owner integrity');assert v['status']=='PASS' and v['owner_exit_code']==0
 r=read(q/'equal_state_report.json','ASH completed equal-state comparison')
 match=read(q/'COMMON_THREE_ARM_MATCH.json','ASH equal-state common-arm identity')
 assert match['status']=='TASK_STRATIFIED_COMMON_THREE_ARMS_BITWISE_PASS' and all(c['bitwise_equal'] for c in match['checks'].values())
 check_sha(q/'equal_state_report.json',match['ASH_report_sha256']);ASH['两状态等权']=r
EXPANDED={}
expanded_sources=read(HERE/'EXPANDED_RESULT_SOURCES.json','Final verified expanded-rank result sources')
assert expanded_sources['status']=='COMPLETE_BOTH_STATES_AND_COMBINED'
for key in ['D4_Rest','D4_Stim8hr','equal_state']:
 e=expanded_sources[key];q=Path(e['directory']);o=read(q/'OWNER_COMPLETION.json',f'FR rank 300 {key} owner')
 check_sha(q/'OWNER_COMPLETION.json',e['owner_sha256'])
 assert o['owner_exit_code']==0 and o['independent_audit_pass'] and o['separate_tool_call_after_owner_exit_zero'] and o['npz_crc']=='PASS'
 check_sha(e['report'],o['report_sha256']);r=read(e['report'],f'New FR rank 300 {key}')
 for name,v in o['npz_files'].items():check_sha(q/name,v['sha256'])
 if key!='equal_state':
  m=read(q/'COMMON_THREE_ARMS_VS_ORIGINAL.json',f'FR rank 300 {key} common-arm identity')
  assert m['status']=='PASS' and all(m['checks'].values())
  for sp,h in m['sources_sha256'].items():check_sha(sp,h)
 EXPANDED[key]=r
FR_LEDGER=read(RES/'FR_WORK_CLASSIFICATION_AND_PAUSE_LEDGER.json','FR recovery, new work and paused queue classification')
IND=None
q=ROOT/'ash_work/resume_20261014/D4_scores/D4_Stim8hr_independent'
if (q/'POST_EXEC_VERIFIED.json').exists():
 v=read(q/'POST_EXEC_VERIFIED.json','Independent-guide ASH post-owner integrity');assert v['status']=='PASS' and v['owner_exit_code']==0
 IND=read(q/'evaluation_report.json','New independent-guide ASH comparison')
 m=read(q/'COMMON_THREE_ARM_MATCH.json','Independent-guide frozen scalar identity')
 assert m['status']=='FROZEN_SCALAR_COMMON_THREE_ARMS_BITWISE_PASS' and all(c['exact_array_equal'] for c in m['checks'].values())
 check_sha(q/'evaluation_report.json',m['ASH_report_sha256'])
 check_sha(m['frozen_scalar_report_path'],m['frozen_scalar_report_sha256'])
 assert m['historical_quality_gates_unchanged']['cross_guide_signal_pass'] is False
ASH_DEPTH={}
for n in [20,50,100]:
 q=ROOT/f'ash_work/resume_20261014/depth_evaluation/n{n}'
 if not (q/'POST_EXEC_VERIFIED.json').exists():continue
 v=read(q/'POST_EXEC_VERIFIED.json',f'ASH depth {n} post-owner integrity');assert v['status']=='PASS' and v['owner_exit_code']==0
 r=read(q/'evaluation_report.json',f'ASH depth {n}')
 m=read(q/'COMMON_THREE_ARM_MATCH.json',f'ASH depth {n} common-arm identity')
 assert m['status']=='DEPTH_COMMON_THREE_ARMS_BOTH_FR_SPECIFICATIONS_MATCH' and all(c['all_nine_common_arm_fingerprints_match'] for c in m['checks'])
 check_sha(q/'evaluation_report.json',m['ash_report_sha256'])
 ASH_DEPTH[n]=r
if len(ASH_DEPTH)==3:
 fig,(ax,bx)=plt.subplots(1,2,figsize=(11.5,5.2),gridspec_kw={'width_ratios':[1.05,1]})
 for arm,col,mkr in [('raw',colors['raw'],'s'),('euclidean',colors['euclidean'],'v'),('pertrank',colors['pertrank'],'o'),('ash','#AD4261','D')]:
  ax.plot([20,50,100],[p25(ASH_DEPTH[n],arm) for n in [20,50,100]],marker=mkr,color=col,lw=2,label='ASH (native default)' if arm=='ash' else labels[arm],ls='--' if arm=='euclidean' else '-')
 ax.set(xlabel='Input cells per target',ylabel='P@25',xticks=[20,50,100],title='A  ASH and scalar estimates')
 ax.legend(frameon=False,fontsize=9,loc='upper left');ax.grid(axis='y',color='#E2E7EB',lw=.6)
 for j,(key,label,col) in enumerate([('pertrank_minus_ash','PertRank − ASH',colors['pertrank']),('ash_minus_raw','ASH − Raw','#AD4261')]):
  vals=[ASH_DEPTH[n]['results']['25']['contrasts'][key] for n in [20,50,100]]
  pts=np.array([v['point'] for v in vals]);lo=np.array([v['query_only_95_interval'][0] for v in vals]);hi=np.array([v['query_only_95_interval'][1] for v in vals])
  bx.errorbar(np.array([20,50,100])+(j-.5)*2.5,pts,yerr=[pts-lo,hi-pts],fmt='o-',color=col,capsize=3,label=label,lw=1.8)
 bx.axhline(0,color='#697885',lw=1);bx.set(xlabel='Input cells per target (points offset)',ylabel='Paired P@25 difference',xticks=[20,50,100],title='B  Descriptive paired differences')
 bx.legend(frameon=False,fontsize=9,loc='upper right');bx.grid(axis='y',color='#E2E7EB',lw=.6)
 fig.subplots_adjust(left=.07,right=.98,bottom=.26,top=.87,wspace=.34)
 fig.text(.07,.06,'250 targets • one nested 20/50/100-cell chain • fixed ASH default\n95% query-block intervals are conditional and descriptive. No formal depth-interaction or donor-population inference.',fontsize=9,color='#405262',linespacing=1.6)
 for ext in ['png','svg']:fig.savefig(OUT/f'D3_depth_native_ASH_20261014.{ext}',dpi=200)
 plt.close(fig)
ledger=[['已有结果直接恢复','D3 原生 FR 21 个配置','保留原选择；不增加实验或复现次数'],['恢复 / 核查计算','D4 原配置 FR；D3 默认 ASH','D4 有历史数值摘要，本轮完成并核查；未证明所有历史原始输出完整保留。D3 ASH 后验 SHA 精确复现。均非新增独立验证'],['本轮新增分析','D3 FR rank 150/200/300；FR 六条与 ASH 三条深度配置；D3 数字对账','描述性敏感性和解释核查'],['本轮补齐原未完成比较','D4 默认 ASH：两状态五组、等权汇总和独立 guide','按已锁定配置补齐；不声称新 donor 复现'],['本轮新增、已完成','D4 rank 300：两状态各五个输入及等权汇总','post hoc 敏感性；原主比较和继续门不改写'],['本轮新增资源测试','GPerturb 全输入资源门','触发内存 guard；未取得性能结果'],['暂停未启动旧队列','D3 ASH 两个旧备选先验；FR 固定队列余下 10 项','FR：9 项 Nscaled（历史完成状态未知）和 1 项 rank 100 独立 guide；已完成 2 项 Rest Nscaled 不据此拼成完整任务。旧独立 guide 完整 FR 报告为 rank 20 pilot']]
rank_rows=[]
for r in RANK:
 k=r['config']['rank'];v=r['results']['25']['contrasts']['pertrank_minus_fr'];rank_rows.append([str(k)+'（旧）' if k==100 else str(k)+'（新增）',f(p25(r,'fr')),f(p25(r,'fr_pertrank')),signed(v['point']),ci(v['query_only_95_interval'])])
main_rows=[]
for arm in ARMS:
 main_rows.append([NAMES[arm],f(p25(D4['D4_Rest'],arm)),f(p25(D4['D4_Stim8hr'],arm)),f(p25(EQ,arm))])
ash_rows=[]; ash_contrast_rows=[]
for task,r in ASH.items():
 v=r['results']['25']['contrasts']['pertrank_minus_ash'];ash_rows.append([task.replace('_pooled',''),f(p25(r,'raw')),f(p25(r,'euclidean')),f(p25(r,'pertrank')),f(p25(r,'ash')),signed(v['point']),ci(v.get('query_only_95_interval',v.get('task_stratified_query_95_interval')))])
for task,r in ASH.items():
 for contrast,label in [('ash_minus_raw','ASH − Raw'),('ash_minus_euclidean','ASH − Euclidean'),('pertrank_minus_ash','PertRank − ASH')]:
  v=r['results']['25']['contrasts'][contrast];ash_contrast_rows.append([task.replace('_pooled',''),label,signed(v['point']),ci(v.get('query_only_95_interval',v.get('task_stratified_query_95_interval')))])
if 'D4_Stim8hr_pooled' not in ASH:ash_rows.append(['D4_Stim8hr','待完整结果','待完整结果','待完整结果','待完整结果','待完整结果','待完整结果'])
depth_rows=[]
for c,items in DEPTH.items():
 for n,r in items.items():depth_rows.append(['A' if c.startswith('A') else 'B',n]+[f(p25(r,a)) for a in ARMS])
rec_rows=[[name,f(s['native_benchmark_rep0_P25']),f(s['five_rep_average_P25'])] for name,s in REC['scores'].items()]
newest=max([r['created_utc'] for r in ASH.values()] +[rank_exec.get('created_utc','2026-10-14T01:32:00+00:00')])
manifest={'title':'PertRank v20：外部基线与深度诊断证据报告','date':'2026-10-14','status':'本轮实验与制稿完成；暂停项目及未评估项目单列','scope':['已完成的旧结果恢复与本轮新增实验分开标识。恢复 / 核查计算不构成额外独立复现证据；纯补档及历史状态未明的剩余旧队列已暂停。','本报告仅汇总已完成并核对的输出，不修改 v20，不读取 D1/D2 Stim 表达。本轮未读取不等于已经证明历史从未读取。'],
'sections':[
 {'title':'当前结论','text':['D3 扩展搜索改善了原生 FR，但在同一输入组上 PertRank 仍领先；rank 300 仍是搜索边界，不能宣称已经找到最优秩或性能饱和。','恢复的 D4 原配置结果支持：PertRank 优于原生 FR；FR 加 PertRank 权重后，P@25 与完整 PertRank 的差异很小且区间跨零。因此当前证据支持权重在这些任务中的可叠加价值，不能证明保留原始坐标普遍优于 FR 重构。','新增深度诊断显示，FR 绝对分数随细胞数提高，但相对 raw 的差距扩大。ASH 则在 20 细胞低于 PertRank、50 细胞差异接近零、100 细胞 P@25 高于 PertRank；100 细胞时 PertRank 也略低于 raw。因此不能笼统宣称外部方法深输入失效，优势应限定于低细胞数任务。']},
 {'title':'哪些是旧结果，哪些是新增','text':'已有完整结果的重放只补回材料；只有历史摘要的项目标为本轮恢复 / 核查，不假定历史原始输出均曾完整保存。','tables':[{'columns':['类别','任务','科学解释'],'rows':ledger}]},
 {'title':'D3 的 0.3051 与 0.3131 已对账','text':'32 项核查通过。区别是第一组 20 细胞输入与五组不重叠输入的均值；候选、query、reference、controls、gene axes 与权重定义一致。外部方法单组结果必须与 PertRank 单组对比。','tables':[{'columns':['方法','开发基准：第 1 组','论文主表：五组均值'],'rows':rec_rows},{'title':'D3 单组匹配比较','columns':['方法 / 配置','P@25','角色'],'rows':[['Raw',f(p25(R100,'raw')),'不加权基线'],['Euclidean',f(p25(R100,'euclidean')),'内部消融'],['PertRank',f(p25(R100,'pertrank')),'完整方法'],['原生 FR rank 100',f(p25(R100,'fr')),'原开发选择，恢复结果'],['原生 FR rank 300',f(p25(RANK[-1],'fr')),'新增边界扩展，post hoc'],['ASH 默认',f(ASH_D3['default_result']['P25']),'恢复后验，SHA 与历史精确一致']]}]},
 {'title':'新增：D3 FR 搜索边界扩展','text':'同一 D3 Rest 开发输入：1,200 个候选，300 个 queries，每靶点 20 个细胞。Raw=0.2771，Euclidean=0.2762，PertRank=0.3051。三项新增配置全部报告。rank 300 相对原 rank 100 提升 0.0081；仍位于上界。全表达维度配置超出当前内存预算，未测试满秩，也未证明 raw 是 FR 的性能上限。','tables':[{'columns':['FR rank','原生 FR P@25','FR × PertRank 权重','PertRank − 原生 FR','95% 描述性区间'],'rows':rank_rows}], 'figures':[{'path':str(OUT/'D3_rank_boundary_20261014.png'),'caption':'开发数据上的秩敏感性。区间在开发、配置选择后作描述性解释；不能视为独立确认。'}]},
 {'title':'D4 原配置 FR：本轮完成并核查的五输入结果','text':'两种状态来自同一 D4 donor；每状态五个固定、不重叠的 20 细胞输入组。下表为本轮完成并与历史数值记录核对的原配置计算。旧摘要不足以证明每次历史运行都已完成；本轮结果只计一次，不是新增独立验证，也不声称旧模型输出已逐项恢复。Rest 与 Stim8hr 分别有 589、561 个候选和各自 300 个 query；两状态等权汇总。','tables':[{'columns':['方法','Rest P@25','Stim8hr P@25','两状态等权 P@25'],'rows':main_rows},{'columns':['两状态对比','P@25 差值','95% 区间','解释'],'rows':[['原生 FR − Raw',signed(EQ['results']['25']['contrasts']['fr_minus_raw']['point']),ci(EQ['results']['25']['contrasts']['fr_minus_raw']['task_stratified_query_95_interval']),'外部方法相对不加权'],['原生 FR − Euclidean',signed(EQ['results']['25']['contrasts']['fr_minus_euclidean']['point']),ci(EQ['results']['25']['contrasts']['fr_minus_euclidean']['task_stratified_query_95_interval']),'外部方法相对内部收缩'],['PertRank − 原生 FR',signed(EQ['results']['25']['contrasts']['pertrank_minus_fr']['point']),ci(EQ['results']['25']['contrasts']['pertrank_minus_fr']['task_stratified_query_95_interval']),'原 FR 比较；99% 区间为 [0.0272, 0.0488]'],['FR × PertRank 权重 − 原生 FR',signed(EQ['results']['25']['contrasts']['fr_pertrank_minus_fr']['point']),ci(EQ['results']['25']['contrasts']['fr_pertrank_minus_fr']['task_stratified_query_95_interval']),'这些低细胞数任务支持可叠加增益'],['PertRank − FR × PertRank 权重',signed(EX['results']['25']['pertrank_minus_fr_pertrank']['point']),ci(EX['results']['25']['pertrank_minus_fr_pertrank']['task_stratified_query_95_interval']),'新增描述性对比；未检出差异不等于等效']]}]},
 {'title':'本轮补齐：D4 ASH 比较','text':'此前未完成的外部方法比较；默认官方 ashr 后验均值，固定配置不变。下列完整任务的 raw / Euclidean / PertRank 输入、评分与 FR 基准逐元素一致。ASH 对比均为描述性，不属于原 FR 五对比族。低分不能单独证明标准误不稳定是原因；FR 的分数也不能单独识别低秩丢失信息这一机制。','tables':[{'columns':['任务','Raw','Euclidean','PertRank','ASH','PertRank − ASH','95% 描述性区间'],'rows':ash_rows},{'title':'ASH 三层对比','columns':['任务','对比','P@25 差值','95% 描述性区间'],'rows':ash_contrast_rows}]},
 {'title':'新增：20 / 50 / 100 细胞深度诊断','text':['250 个 D3 候选由元数据确定，三种深度使用一个嵌套输入链、同一 reference 与 controls。A 固定原生 Lasso α=10⁻⁴；B 保持 αN/n=0.13，只作有效惩罚敏感性。两条曲线均固定 rank 100。','在 A 中，PertRank − raw 的 P@25 为 +0.0169（20 细胞）、+0.0090（50 细胞）和 −0.0077（100 细胞）。50 细胞的描述性 95% 区间跨零；100 细胞的 95% 区间为 [−0.0172, −0.0005]，99% 区间仍跨零。不能把该条件分析外推为所有高深度任务。','FR 随深度的绝对改善不能验证它已优于 raw，也不能单凭曲线确定低秩重构或标准误估计是差异的原因。250 候选的绝对 P@25 不能直接与 1,200 候选主任务比较。'], 'tables':[{'columns':['曲线','细胞数']+[NAMES[a] for a in ARMS],'rows':depth_rows}], 'figures':[{'path':str(OUT/'D3_depth_native_FR_20261014.png'),'caption':'A 与 B 分开显示，防止敏感性曲线覆盖原生固定参数诊断。所有深度来自同一嵌套输入链，不是独立重复。'},{'path':str(OUT/'D3_depth_relative_to_raw_20261014.png'),'caption':'A 中相对 raw 的配对差值；95% 区间仅反映固定输入、reference 下的 query-block 变异，所有对比均为描述性。'}]},
 {'title':'新增资源测试：GPerturb 未取得性能结果','text':[f'真实 D3 配置包含 26,000 个细胞、1,200 个靶点、15,616 个输入阳性基因、26 个技术协变量，width=700。worker 峰值 RSS 为 {GP["worker_peak_RSS_GiB"]:.3f} GiB，超过预设 6.5 GiB guard 后受控退出；这不是 OS OOM，也不是方法性能失败。','未完成完整资源门，没有 P@25、收敛结论或完整 epoch 时间预测；触发时阶段未持久化，不能声称已完成 20 个 batch 或全部 validation。width=1000 未运行。','此全基因与协变量适配是本次明确配置。GPerturb 允许其他基因 panel 或协变量设置；本次资源阻塞不能推出该方法整体不可评估。']},
 {'title':'独立 guide：仅保留历史摘要','status':'历史摘要；本轮未独立恢复验证','text':'此前摘要显示几何相对 Euclidean 的增量接近零；已找到的独立 guide 原生 FR 完整报告属于 rank 20 pilot，不能替代锁定 rank 100 的比较。本报告不把历史摘要当成本轮新增结果，也不据此断言控制几何只在同 guide 情形有效。该特定任务最多限制跨 guide 增益主张；与合并 guide 的显著性不同不是正式交互检验。'},
 {'title':'论文主张建议与完成范围','text':['可写：外部方法原生比较与内部 Euclidean 消融分开；D3 数字差异由输入重复定义解释；FR 秩扩展缩小但未消除 D3 差距；D4 原配置权重组合有明显增益。','须收窄：不能宣称优于所有最强外部方法、原始坐标始终更优、PertRank 在深输入仍普遍优胜、或跨 guide 几何已被确认。','本轮已完成：D3 秩扩展、D4 rank 300 转移、原配置 D4 FR 核查、全部 ASH 比较、FR/ASH 深度诊断、D3 数字对账与 GPerturb 资源门。原冻结主比较保留；新增敏感性均为 post hoc。']}
],
'methods':['P@25 是主要提名指标，P@10 为次要敏感性；正负方向分别提名并平均，排除自身靶点，沿用原始边界 ties 规则。','2,000 次 query-block bootstrap；D4 先平均五个固定输入，两状态按各自 query panel 分层重采样再等权汇总，不能按 query 数组位置配对。','原 FR 每个任务与 K 的五个指定对比可用边际 99% 区间作近似 Bonferroni 解释，前提是边际区间本身有效；此覆盖范围不含所有方法、rank、depth、states 或后加对比。','ASH、秩扩展、深度及 PertRank − 加权 FR 对比均作描述性分析。'],
'limitations':['只有一个 D4 donor 的两种状态；五组输入不是五个生物学重复，区间不反映 donor 总体不确定性。','P@25 衡量与既定深参考排名的一致性，不是经独立实验验证的生物真阳性率。深参考并非完整生物真值；本轮未新增独立 RNA 参考或 Pearson 残差等替代参考尺度敏感性。','rank 300 仍在搜索边界；native factorization 固定 50 次迭代不代表字典收敛。','GPerturb 无性能结论；不将计算环境阻塞计为算法落败。','新敏感性有本地时间记录，但未声称 OSF 公共注册。历史 D1/D2 表达访问状态尚未独立核实。'],
'sources':SOURCES}
if IND is not None:
 for i,section in enumerate(manifest['sections']):
  if section.get('title')=='独立 guide：仅保留历史摘要':
   contrasts=[]
   for key,label in [('ash_minus_raw','ASH − Raw'),('ash_minus_euclidean','ASH − Euclidean'),('pertrank_minus_ash','PertRank − ASH'),('pertrank_minus_euclidean','PertRank − Euclidean')]:
    v=IND['results']['25']['contrasts'][key];contrasts.append([label,signed(v['point']),ci(v['joint_95_interval'])])
   manifest['sections'][i]={'title':'本轮补齐：独立 guide 的 ASH 比较与几何核查','status':'ASH 与冻结 scalar 任务完整匹配；FR rank 100 历史完成状态未知','text':['固定的 395 靶点任务：guide A 输入 20 细胞，guide B 参考 200 细胞。raw、Euclidean、PertRank 以及原 reference-cell / query-block 联合 bootstrap 与冻结 scalar 结果通过 12 项逐元素核查，因此本节几何结论现有本轮核查支持。','PertRank 相对 ASH 的 P@25 差值约 +0.0026，联合 95% 区间跨零；P@10 同样不支持明确优势。相对 Euclidean 的增量约 −0.0002，联合 95% 区间也跨零。本任务不支持额外几何增益或 PertRank 明确优于 ASH。','原参考稳定性门通过，但跨 guide 生物信号门及继续门未通过，质量门未被改写。本结果受该特定任务信号分辨率限制；不能外推为所有独立 guide 情形无增益，也不能把 pooled 与 independent 显著性差异当作正式交互检验。','根据现有记录，无法确定锁定 rank 100 的独立 guide 比较此前是否完成；本轮未重跑，也未独立恢复该比较。已找到的旧完整报告仅为 rank 20 pilot，不能代替。'], 'tables':[{'columns':['方法','P@25'],'rows':[[NAMES[a],f(p25(IND,a))] for a in ['raw','euclidean','pertrank','ash']]},{'columns':['对比','P@25 差值','联合 95% 描述性区间'],'rows':contrasts}]}
ash_depth_rows=[]
for n in [20,50,100]:
 if n not in ASH_DEPTH:ash_depth_rows.append([n]+['待完整结果']*6);continue
 r=ASH_DEPTH[n];v=r['results']['25']['contrasts']['pertrank_minus_ash']
 ash_depth_rows.append([n]+[f(p25(r,a)) for a in ['raw','euclidean','pertrank','ash']]+[signed(v['point']),ci(v['query_only_95_interval'])])
manifest['sections'].insert(-1,{'title':'新增：ASH 深度诊断','status':f'{len(ASH_DEPTH)}/3 个深度完整','text':['与原生 FR 深度任务使用相同的 250 候选、nested cells、reference、controls、queries 和权重；每个已完成深度的共同三臂均与 A/B 两个 FR 配置匹配。默认 ASH 参数不随深度重新选择，所有区间为描述性。','ASH 的 P@25 从 0.3819（20 细胞）升至 0.4912（50 细胞）和 0.5552（100 细胞）。PertRank − ASH 的差值分别为 +0.0222、+0.0005、−0.0119；50 细胞的 95% 区间跨零，100 细胞的区间为 [−0.0172, −0.0062]。在这个特定 post hoc 深度任务中，100 细胞 P@25 支持 ASH 更优。','100 细胞 P@10 的 PertRank − ASH 为 −0.0064，但其 95% 区间跨零。上述模式与较深输入减少估计噪声的解释相容，却不能证明标准误不稳是已识别机制，也不是正式深度交互检验。'],'tables':[{'columns':['细胞数','Raw','Euclidean','PertRank','ASH','PertRank − ASH','95% 描述性区间'],'rows':ash_depth_rows}]})
if len(ASH_DEPTH)==3:
 manifest['sections'][-2]['figures']=[{'path':str(OUT/'D3_depth_native_ASH_20261014.png'),'caption':'ASH 与共同 scalar 三臂单独成图；右图为配对差值和描述性 95% 区间。深度依赖的观察模式不等于已完成正式交互检验。'}]
manifest['sections'][-1]['text']=[x.replace('ASH 独立 guide 与深度曲线','ASH 剩余深度曲线') for x in manifest['sections'][-1]['text']]
x=EXPANDED['equal_state'];cx=x['results']['25']['contrasts']
expanded_rows=[[NAMES[a],f(p25(EXPANDED['D4_Rest'],a)),f(p25(EXPANDED['D4_Stim8hr'],a)),f(p25(x,a))] for a in ARMS]
expanded_contrasts=[]
for key,label in [('fr_minus_raw','FR rank 300 − Raw'),('fr_minus_euclidean','FR rank 300 − Euclidean'),('pertrank_minus_fr','PertRank − FR rank 300'),('fr_pertrank_minus_fr','FR rank 300 × PertRank 权重 − 原生 FR'),('fr_pertrank_minus_fr_euclidean','FR rank 300 × PertRank 权重 − FR × Euclidean 权重'),('pertrank_minus_fr_pertrank','PertRank − FR rank 300 × PertRank 权重')]:
 v=cx[key];expanded_contrasts.append([label,signed(v['point']),ci(v['task_stratified_query_95_interval'])])
manifest['sections'].insert(5,{'title':'新增完成：D4 rank 300 转移敏感性','status':'两状态各五个输入与等权汇总均完成并独立核验','text':['rank 300 按 D3 扩展结果和事先写定的选择规则确定，再用于已打开的 D4 任务；全部为 post hoc 敏感性，不替换原 rank 100 主比较。','PertRank 对原生 FR rank 300 的两状态等权 P@25 优势仍为约 +0.0361。FR 加 PertRank 权重后得分为 0.3900，完整 PertRank 为 0.3865；差值 −0.0036 的描述性 95% 区间跨零。因此不支持完整 PertRank 在主要 P@25 指标上优于组合，也不能以此认定两者等效。','P@10 的 PertRank − 组合差值为 +0.0058，描述性 95% 区间 [0.0007, 0.0131]，但 99% 区间跨零。这是次要指标，不替代 P@25 的结论。两种状态仍来自同一 donor。'], 'tables':[{'columns':['方法','Rest P@25','Stim8hr P@25','两状态等权 P@25'],'rows':expanded_rows},{'columns':['两状态等权对比','P@25 差值','95% 描述性区间'],'rows':expanded_contrasts}]})
manifest['sections'][0]['text'].insert(1,'完成的 D4 rank 300 敏感性中，PertRank 对原生 FR 仍领先 0.0361 P@25；对 ASH 的两状态等权优势为 0.0638。加权 FR 的 P@25 为 0.3900，完整 PertRank 为 0.3865，二者差值区间跨零。')
manifest['sections'][-1]['text'].extend(['建议论文主张：“在浅输入的合并 guide 提名任务中，利用控制细胞几何估计可靠性，可改善效应排序，并可叠加于外部效应估计。”此处的支持范围是已评估的 D3/D4 任务与固定配置，不泛化为所有 donor、深度或独立 guide 任务。','不再用 D3 单次开发结果推论“保留原始坐标普遍优于 FR 重构”。D4 rank 100 与 rank 300 的加权组合结果均不支持这一普遍结论。','暂停而非完成：D3 ASH 两个旧备选先验的补档重放；9 项 Nscaled FR 与 1 项 rank 100 独立 guide 队列。尚未评估：GPerturb 性能、替代参考尺度敏感性、D1/D2 确认。GPerturb 已完成的是受资源约束的可行性测试。'])
manifest['sources']=SOURCES
(HERE/'evidence_manifest_20261014.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
if len(ASH_DEPTH)==3:
 (OUT/'D3_depth_native_ASH_20261014.json').write_text(json.dumps({'sources':[v for v in SOURCES if v['label'].startswith('ASH depth')], 'scores':{str(n):{a:p25(r,a) for a in ['raw','euclidean','pertrank','ash']} for n,r in ASH_DEPTH.items()}, 'contrasts':{str(n):r['results']['25']['contrasts'] for n,r in ASH_DEPTH.items()}, 'scope':'Same 250-target nested input chain; descriptive conditional query-block intervals; no formal interaction test.'},indent=2)+'\n')
(OUT/'D3_depth_native_FR_20261014.json').write_text(json.dumps({'sources':[s for s in SOURCES if s['label'].startswith('Depth')], 'scores':{c:{str(n):{a:p25(r,a) for a in ARMS} for n,r in items.items()} for c,items in DEPTH.items()},'scope':'Post hoc; 250 candidates, one nested chain; A/B not selected by performance.'},indent=2)+'\n')
(HERE/'ASSEMBLY_VERIFICATION.json').write_text(json.dumps({'status':'SOURCE_BINDING_AND_COMPLETION_CHECKS_PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':SOURCES,'models_run':False,'expression_matrices_read':False,'ash_tasks_included':list(ASH),'depth_reports':6,'rank_reports':4,'independent_ASH_included':IND is not None,'ASH_depths_included':sorted(ASH_DEPTH),'pending':[],'expanded_rank_300_tasks_included':list(EXPANDED),'paused':['D3 ASH alternative prior archival replays','9 Nscaled FR fits','1 rank100 independent-guide FR fit'],'not_evaluated':['GPerturb performance','alternative-reference scale','D1/D2 confirmation'],'manuscript_v20_modified':False},indent=2)+'\n')
print(HERE/'evidence_manifest_20261014.json')
