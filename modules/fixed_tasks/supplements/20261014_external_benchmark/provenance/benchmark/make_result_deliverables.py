#!/usr/bin/env python3
"""Render finalized benchmark reports without recomputing or selecting results."""
from pathlib import Path
import argparse, json, hashlib
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS=['raw','euclidean','pertrank','fr','fr_euclidean','fr_pertrank']
LABELS=['Raw','Euclidean','PertRank','Native FR','Euclidean × FR','PertRank × FR']
CONTRASTS=['pertrank_minus_fr','fr_pertrank_minus_fr','fr_pertrank_minus_fr_euclidean']
CLABELS=['PertRank − native FR','wM·FR − native FR','wM·FR − w0·FR']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reports',nargs=3,type=Path,required=True)
    ap.add_argument('--equal-state',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--sensitivity-reports',nargs=3,type=Path)
    a=ap.parse_args();root=Path(__file__).resolve().parent
    lock=json.loads((root/'FR_D3_WINNER_LOCK.json').read_text())
    reports=[json.loads(p.read_text()) for p in a.reports]
    assert all(r['status']=='COMPLETE' for r in reports)
    assert [r['kind'] for r in reports]==['pooled','pooled','independent']
    assert [r['dataset'] for r in reports[:2]]==['D4_Rest','D4_Stim8hr']
    eq=json.loads(a.equal_state.read_text());a.out.mkdir(exist_ok=True,parents=True)
    titles=['D4 Rest · pooled guides','D4 Stim8hr · pooled guides','D4 Stim8hr · independent guide']
    colors=['#246891','#BF6C25','#5C477D']
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.6),sharey=True)
    for ax,r,title in zip(axes,reports,titles):
        interval='joint_95_interval' if r['kind']=='independent' else 'query_only_95_interval'
        xs=[];ends=[]
        for i,(key,color) in enumerate(zip(CONTRASTS,colors)):
            d=r['results']['25']['contrasts'][key];x=d['point']*100;lo,hi=np.array(d[interval])*100
            # Percentile bootstrap intervals need not contain the point estimate.
            ax.plot([lo,hi],[2-i,2-i],color=color,lw=2)
            ax.plot([lo,lo],[2-i-.045,2-i+.045],color=color,lw=1)
            ax.plot([hi,hi],[2-i-.045,2-i+.045],color=color,lw=1)
            ax.scatter([x],[2-i],s=42,color=color,zorder=3)
            xs.append(x);ends.extend([lo,hi,x])
        ax.axvline(0,color='#777777',lw=.9,ls='--',zorder=0)
        ax.set_title(title,fontsize=10,pad=14)
        span=max(ends)-min(ends);pad=max(.12,span*.14)
        ax.set_xlim(min(0,min(ends))-pad,max(0,max(ends))+pad)
        ax.set_xlabel('P@25 difference (percentage points)',fontsize=9,labelpad=10)
        ax.set_ylim(-.6,2.6);ax.set_yticks([2,1,0],CLABELS,fontsize=10)
        ax.tick_params(axis='both',labelsize=9)
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
        ax.grid(axis='x',alpha=.14)
    fig.suptitle('Primary comparison: D3-selected native FR with fixed nominal alpha',fontsize=13,y=.97)
    fig.text(.02,.035,'95% paired intervals. Pooled tasks: query-block resampling conditional on five fixed inputs and reference.\nIndependent-guide task: joint reference-cell and query-block resampling. All tasks use one observed donor.',fontsize=8,color='#444444')
    fig.subplots_adjust(left=.19,right=.985,top=.80,bottom=.27,wspace=.28)
    fig.savefig(a.out/'PertRank_FR_Contrasts_20261012.png',dpi=240,facecolor='white')
    fig.savefig(a.out/'PertRank_FR_Contrasts_20261012.svg',facecolor='white');plt.close(fig)
    lines=['PertRank 对原生 FR-Perturb：执行结果',datetime.now(timezone.utc).isoformat(),'',
        '研究范围：已打开的 D3 Rest 开发集与 D4 两状态。不是新盲法供者确认。',
        '本轮未读取 D1/D2 Stim 表达；历史是否从未读取尚未证实。',
        f'D3 原生 FR 选择：{lock["config_id"]}；完整配置、输入和评分哈希见 FR_D3_WINNER_LOCK.json。',
        'FR 在 D3 从21个预先列出的原生配置中按自身P@25择优；PertRank保持历史固定参数。',
        '双方历史开发目标和总预算并不完全相同，不能声称严格同等历史调参。','',
        'P@25（正负方向分别取Top25，先排除自身靶点，再按原查询平均）',
        '方法\tD4 Rest\tD4 Stim8hr\tD4 Stim8hr 独立guide']
    for key,label in zip(ARMS,LABELS):
        lines.append(label+'\t'+'\t'.join(f'{r["results"]["25"]["arms"][key]["point"]:.8f}' for r in reports))
    lines += ['', '主要配对差值及95%区间（P@25原始比例单位）']
    for r,title in zip(reports,titles):
        lines.append(title);interval='joint_95_interval' if r['kind']=='independent' else 'query_only_95_interval'
        for key,label in zip(CONTRASTS,CLABELS):
            d=r['results']['25']['contrasts'][key];lo,hi=d[interval]
            lines.append(f'  {label}: {d["point"]:+.8f} [{lo:+.8f}, {hi:+.8f}]')
    if a.sensitivity_reports:
        sensitivity=[json.loads(p.read_text()) for p in a.sensitivity_reports]
        lines += ['', '保持每组有效惩罚强度的原生FR敏感性（预结果amendment 04）',
            'alpha_sensitivity = alpha_D3 × 26000 / N_task；D3所选rank和其余参数不变。',
            '以下95%区间仅描述性；不得按D4成绩替换主分析或改写原开发继续门。']
        for s,r,title in zip(sensitivity,reports,titles):
            for arm in ['raw','euclidean','pertrank']:
                assert s['results']['25']['arms'][arm]['point']==r['results']['25']['arms'][arm]['point']
            interval='joint_95_interval' if r['kind']=='independent' else 'query_only_95_interval'
            lines.append(title)
            for key,label in zip(CONTRASTS,CLABELS):
                d=s['results']['25']['contrasts'][key];lo,hi=d[interval]
                lines.append(f'  {label} (N-scaled): {d["point"]:+.8f} [{lo:+.8f}, {hi:+.8f}]')
            primary=r['results']['25']['contrasts']['pertrank_minus_fr']['point']
            secondary=s['results']['25']['contrasts']['pertrank_minus_fr']['point']
            if primary>0 and secondary<=0:lines.append('  主比较的PertRank点优势在惩罚标度敏感性下反转；不能称稳健胜过FR。')
    else: lines += ['', '有效惩罚强度敏感性尚未纳入本报告；不声称优势对该处理稳健。']
    lines += ['','预定开发继续门（计算资源分配标准，非确认性成功判定）',
        json.dumps(eq['development_resource_gates'],ensure_ascii=False,indent=2),'',
        '解释边界：',
        '1. Euclidean 是内部前身/消融，不计作外部竞争者。',
        '2. pooled任务的区间以固定输入和参考为条件；两状态不是两个独立供者。',
        '3. 两状态原始查询面板不同，各自分块抽样后等权汇总，不配对无关查询行。',
        '4. 组合直接复用原始输入的冻结标量权重乘原生FR输出，没有再估计权重或事后重定中心。组合只说明此实现的经验表现。',
        '5. 独立guide任务的旧生物信号/度量增量门结果保留；此比较不把它改写成新盲法验证。',
        '6. 主参考仍为原冻结log效应参考；本报告未声称完成Pearson残差参考敏感性。',
        '7. 95%区间为描述性；五项预定比较的99%区间、P@10、五次输入重复详见机器可读报告。',
        '8. 原生FR使用官方固定50次字典训练；没有原生字典收敛标志，不能把无警告等同于已证明收敛。','',
        '来源报告与SHA256：']
    source=[]
    for p in [*a.reports,*(a.sensitivity_reports or []),a.equal_state,root/'FR_D3_WINNER_LOCK.json']:
        h=hashlib.sha256(p.read_bytes()).hexdigest();lines.append(f'{p}\n  {h}');source.append({'path':str(p),'sha256':h})
    (a.out/'PertRank_FR_Results_20261012.txt').write_text('\n'.join(lines)+'\n')
    (a.out/'DELIVERABLE_SOURCES.json').write_text(json.dumps(source,indent=2)+'\n')
    print(json.dumps({'out':str(a.out),'files':[p.name for p in a.out.iterdir()]}))

if __name__=='__main__':main()
