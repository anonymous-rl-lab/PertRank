#!/usr/bin/env python3
"""Chinese factual report from completed fixed native comparisons; no thresholds changed."""
from pathlib import Path
import json

P=Path(__file__).resolve().parent
def main():
    ds=['D3_Rest','D1_Rest','D2_Rest']
    rows={d:json.loads((P/'scores'/d/'report.json').read_text()) for d in ds}
    combined=json.loads((P/'scores/equal_source/report.json').read_text())
    arms=['raw','Euclidean','PertRank','FR','FR_x_Euclidean','FR_x_PertRank','ASH']
    s=['A19-R 原生外部方法补充比较｜2026-10-14','',
      '身份：在既有、已打开的独立 arrayed knockout RNA 参考上新增原生 FR-Perturb 与 ASH 比较。方案设计时已知此前外部参考结果。本轮没有新采集或新打开外部生物实验，不属于前瞻盲确认。',
      '任务：完整保留 D3/D1/D2 Rest 的52/59/63候选、293/289/289查询、20个浅输入细胞/靶点、原2000输入对照、原权重与K=10；正负两个方向等权，自身基因排除和边界期望破并列均保持。旧raw/Euclidean/PertRank预测直接复用。',
      '原生设置：FR rank300 / spca alpha0.1 / lasso alpha0.0001 / 50迭代 / seed20261012 / 2线程；ASH ashr2.2-63官方默认，每靶点全原始控制表达基因轴拟合，无SE下限补丁。','',
      '一、每份10名名单的期望外部参考支持成员数（除以10即P@10）',
      '方法\tD3 Rest\tD1 Rest\tD2 Rest\t来源等权描述性均值']
    for arm in arms:
        s.append(arm+'\t'+'\t'.join(f"{rows[d]['means'][arm]['expected_members_per_10']:.6f}" for d in ds)+f"\t{combined['means'][arm]['expected_members_per_10']:.6f}")
    s+=['','二、主要配对差值：单位同为每10名增加的参考成员；括号为固定参考下的查询块95%条件区间']
    keys=['PertRank_minus_FR','PertRank_minus_ASH','FR_x_PertRank_minus_FR_x_Euclidean','FR_minus_raw','FR_minus_Euclidean','PertRank_minus_FR_x_PertRank','PertRank_minus_Euclidean']
    for key in keys:
        s.append(key)
        for d in ds:
            x=rows[d]['contrasts'][key];ci=[10*v for v in x['conditional_query_block_95_interval']]
            s.append(f"  {d}: {x['expected_members_difference']:+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}]")
        s.append(f"  三来源等权点差值: {combined['contrasts'][key]['expected_members_difference']:+.6f}；不构造独立供者合并置信区间。")
    s+=['','三、共享外部参考的不确定性传播',
      '原两种500轮参考抽样（shared_labels、batch_separate）完整复用；每轮同一参考同时用于三个来源。下列2.5–97.5百分位为经验敏感性范围，不是总体95%置信区间。']
    for scheme,result in combined['reference_sensitivity'].items():
        s.append(scheme+'（来源等权）')
        for key in keys:
            x=result[key];lo,hi=x['empirical_2p5_97p5'];s.append(f"  {key}: median {x['median']:+.6f}; [{lo:+.6f}, {hi:+.6f}]; fraction_positive {x['fraction_positive']:.3f}")
    s+=['','四、解释范围',
      '外部参考支持指与另一次实验的基因响应排序一致，不直接等同于已验证生物学真阳性。三个来源共享参考与部分查询，不视为三次独立外部生物验证。',
      '所有新增外部比较均为已打开任务上的描述性分析；没有依据新结果调参数、更换候选池或筛选查询。条件区间不包含浅输入细胞抽样、控制方向估计或面板选择的不确定性。',
      'FR的固定50迭代及原生LASSO数值审计不等于字典全局收敛或充分调参证明。ASH后验收缩诊断只描述结果，不建立标准误不稳导致排序变化的因果结论。',
      '', '五、执行完整性',
      '成功输出共6个固定原生拟合（3源×FR/ASH）。D1的第一次FR启动失去owner且停在SPAMS阶段记录；D2第一次FR启动只留下等待共享锁记录。两次均无最终效应输出，不计算法性能或新增验证。保留原记录后，根任务分别持有同配置第二次执行并独立验证。',
      '原生后验、效应、原权重、全部查询评分、bootstrap和候选加入/移除支持分解均完整保存。共享大参考抽样由精确归档ID、SHA与成员路径恢复，不重复训练。']
    (P/'RESULTS_CN.txt').write_text('\n'.join(s)+'\n')
    print(P/'RESULTS_CN.txt')

if __name__=='__main__':main()
