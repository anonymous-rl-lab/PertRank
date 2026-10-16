# A19-R：文稿设计的独立重建与外部参考误差传播（开发复核）

2026-10-05。用户授权有限补强。原A19的完整source锁定预测、原细胞seed与完整代码链尚缺，本轮不称原A19精确复现，也不称独立确认。

## 固定规则

1. 只读D3_Rest、D1_Rest、D2_Rest。使用原read_a15_guide_split.py同样的metadata过滤：非low_quality，guide_type=targeting，基因非multi/NTC/缺失。先对外部84KO基因按总合格细胞数≥40纳入。原报告候选52/59/63仅作核对，不作为需凑出的选择目标；若不符先查并报告。
2. 候选按官方symbol字典序；每靶点细胞先按源行号排，再以 `sha256('pertrank-a19r-20261005|'+donor+'|'+target)` 前16位十六进制转整数为NumPy default_rng seed置换。取前20作输入，随后min(100,n−20)作独立deep。**这是本轮新细胞选择，不声称与原A19相同。**
3. 使用各source已有冻结300响应查询，按外部13,284表达基因交集筛选。预计293/289/289，固定query和原blocks。全基因权重基因集、log1p(count×原scale/原totalcounts)、input NTC均值、control covariance directions、trace和投影方差均沿用原source状态；k=5、beta=.9不变，不重学控制方向。
4. 每target保留input全权重基因求和/平方和、前20方向投影求和/平方和、query sum；deep query sum和n。权重公式沿用原代码。own-target排除按source Ensembl标识与query核对，不读取坏的全−1旧self_row。
5. 外部参考主口径仍为已恢复且SHA一致的311样本counts：log2(CPM+1)，同batch同external donor减AAVS1，对三donor平均。不是作者原模型的复现。每source候选池内top10，双方向，边界并列用概率期望；deep gate仅作为诊断，不根据结果换池或规则。
6. 先每source三个hash选target读数smoke，核行数、总UMI、输入/deep不重叠、全基因与query投影朴素求和一致，再全量。控制gene-axis必须匹配存档。

## 分析

- 每source报告raw、Euclidean、PertRank及deep与外部参考top10的期望重合；PertRank−raw与PertRank−Euclidean两层同时保留。
- paired query-block bootstrap 2000次（seed2026100502）仅作固定cell/candidate/reference条件区间；三source均值描述性，不平均端点冒充joint CI。
- 传播外部参考误差：使用先前写定的500轮external donor multiplier +同batch/donor control重抽；同一轮参考供三source共同使用。query/candidate/sourcecell固定，报告经验敏感性范围，不称总体CI。主跨batch共享同号donor multiplier，副batch×donor独立；两个假设都报。
- float64生成reference效应用于重新排序；若读取已保存float32 draws，必须先完整核对边界排序与float64重建一致，不能让量化新并列进入评价。
- 另报删作者R脚本列出的3个bad AAVS1 controls的点估计敏感性，保持查询/过滤/候选不变。
- 不以新seed选择结果好坏，不新增参数网格、不改主方法、不读取封存D4/HEK293T/Stim。所有A19-R结果与原A19分列。

## 来源与判断限度

84个KO来自原研究两批24/60预选靶点，同实验室体系、外部是激活Cas9/bulk RNA、本项目是Rest CRISPRi/single-cell；新source与原A19都属于已公开资源的开发后支持性检验。本轮只建立实际重建细胞上的结果，不能填上原A19的出处哈希缺口。每batch只有3external donor且每KO/donor只有一个well，参考重抽仍不能涵盖一切实验误差。
