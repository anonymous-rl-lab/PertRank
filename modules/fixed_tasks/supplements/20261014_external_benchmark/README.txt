PertRank external benchmark: anonymous review replay

运行（Python >=3.11；核验环境 Python 3.12 / NumPy 2.3.5）：
  python reproduce.py --output-dir ../PertRank_external_benchmark_reproduced
仅核查完整性：
  python reproduce.py --verify-only
输出目录必须位于包外且尚不存在。

本模块从保留的逐查询分数、固定候选成员概率、抽样设计和联合参考抽样结果，
重新计算既定方法的 P@25/P@10、配对差值和区间。数值精度及并列规则保持不变。
没有重新拟合模型或读取原始单细胞表达。

范围：
1. D3 的 21 个原生 FR 配置、秩扩展与默认 ASH 比较。
2. D4 Rest/Stim 的 FR rank 100/rank 300、五组输入及等权状态汇总。
3. D4 ASH 和独立 guide 比较。
4. D3 在 20/50/100 细胞下的深度诊断。
5. D3 单组输入与五组平均的数值对账。

目录：
  replay.py / scores.npz       主要比较与深度诊断
  d3_replay.py / d3/           D3 开发比较
  d3_reconciliation.py         单组与五组比较
  original_reports/           已完成比较的数值核对目标
  provenance/                 来源、配置及执行身份
  PACKAGE_MANIFEST.json        当前审稿副本校验表

当前报告文件的校验值对应匿名化副本；匿名化仅改变路径或来源文字，不改变
数值比较目标、评分算法或容差。未提供的原始文件哈希仍为来源记录，不重写为
本地文件校验值。GPerturb 仅保留资源试验记录，没有性能结果；未完成的比较不计入
成功复现范围。更改候选、查询、K 或重新训练需要相应完整输入和环境。

结果写入 numerical/、d3_development/、d3_reconciliation/，
总报告为 BENCHMARK_REPRODUCTION.json。
