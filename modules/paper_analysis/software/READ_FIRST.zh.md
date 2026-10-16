# PertRank 0.2.0rc2 使用说明

本版将可安装软件与 v25.2 论文的核心设计对齐：**把表达谱层面的对照信息用于扰动可靠性评估，保留与生物学问题一致的基因读出，并把同一可靠性计算复用到外部效应估计。** 软件仍使用固定的五个对照方向和 `beta=0.9`。

当前为匿名私有评审版本。公开 PyPI 发布、scverse 登记和 pertpy 合并是独立的后续步骤；本地可安装包不代表已被这些平台收录。

## 安装与运行

在源代码目录运行：

```bash
python -m pip install .
python tutorials/00_anndata_quickstart.py
```

如需从源代码构建 wheel：

```bash
python -m pip wheel --no-deps . --wheel-dir dist
```

`00_anndata_quickstart.py` 用合成数据演示 AnnData 到候选表的软件流程。自己的数据须明确输入尺度、完整基因轴、靶点身份和匹配对照，具体见 [README](README.md) 和 [输入要求](docs/input_contracts.md)。

## 实际用途

1. **从浅层细胞输入提名候选：** `pr.rank(...)` 计算可靠性、原始效应及每个查询的上调/下调名单。
2. **为已有估计器复用可靠性：** `result.rank_effects(...)` 接收带靶点和基因标签的外部效应矩阵，保持原可靠性计算固定，重新生成候选名单。
3. **对照独立参考评价名单：** `pr.evaluate_reference(...)` 在相同候选、查询、名单长度和自身靶点排除规则下，计算逐查询、逐方向的精确并列期望重合数及 precision。参考仅用于评分，区间和跨来源汇总由论文完整复现链处理。

两个入口都返回 Unweighted、Euclidean 和 PertRank 三臂。PertRank 对 Unweighted 衡量完整加权流程；PertRank 对 Euclidean 才隔离新增控制几何。外部估计器复用后，这三臂均使用同一外部效应。权重是轮廓可靠性的估计，不是生物学验证概率。

外部复用结果中的 `effects` 是当前排名所用效应，`statistics` 始终保留原始可靠性输入，避免把外部效应和原始能量混成一套统计量。结果元数据记录软件版本、效应来源、基因轴与计算配置。

## 两篇真实数据教程

| 教程 | 归档内容 | 提供形式 |
|---|---|---|
| Replogle K562 | 全部 2,000 个候选、300 个查询；可靠性统计量来自 8,202 个基因 | `tutorials/01_replogle.ipynb` 及 HTML |
| T 细胞 D3 Rest | 全部 52 个候选、293 个查询；可靠性统计量来自 13,390 个基因，附已存 FR 输出 | `tutorials/02_tcell.ipynb` 及 HTML |

教程离线读取真实的已存充分统计量，重新计算权重和候选表；保留完整候选池，演示查询按 ID 字典序选取。每套数据均附来源、哈希和处理层级。这是统计量到排名的复现，逐细胞数据入口由 AnnData 示例和软件测试单独覆盖。

```bash
python scripts/check_tutorials.py
```

该命令运行两篇教程的所有代码单元。Jupyter 可交互查看 notebook，HTML 供直接阅读，候选表导出到 `outputs/`。

## 验证与完整复现

本版的实际检查范围、运行环境及结果见 [验证记录](docs/validation.md)。软件检查覆盖输入契约、存储格式、匹配对照、ID 对齐、并列排名、自身靶点排除、外部效应复用和冻结公式一致性。`requirements-tested.txt` 记录本版核验环境。

软件教程负责使用与数值回归；论文总代码包另含参考构建、性能评分和配对区间等科学复现链。本轮软件修订不增加生物实验或科学性能比较。

## 后续社区接入

[社区接入说明](docs/community_integration.md) 保留了 scverse 登记及 pertpy 贡献的准备材料。公开阶段须补齐实际维护者、仓库、作者选定的许可与引用信息，再完成发布和社区审核。
