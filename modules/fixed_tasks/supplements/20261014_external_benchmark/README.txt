PertRank External Benchmark — compact reproduction package v1.4.0

本模块保留完整固定任务结果及其离线复现；历史HTML展示报告已移入独立私有来源档。
此新增模块单独ZIP小于10,000,000字节；总代码包无10MB大小限制。

运行（Python >=3.11；已核对环境 Python 3.12 / NumPy 2.3.5）
  python reproduce.py --output-dir ../PertRank_external_benchmark_reproduced
仅核查文件完整性：
  python reproduce.py --verify-only
输出目录须为包外的新目录。离线运行，无需另外下载评分数据。

复现范围
1. D3 FR全部21个原开发配置与秩150/200/300扩展、默认ASH比较。
2. D4 Rest/Stim两状态的FR rank100与rank300比较，每状态五个固定输入及两状态等权汇总。
3. D4默认ASH合并guide比较和395候选独立guide比较。
4. D3的FR两种惩罚配置、ASH在20/50/100细胞下的深度诊断。
5. D3第1组输入与五组输入均值的数字对账。
6. 原报告、原始代码、配置和执行记录；GPerturb只保存资源测试事实。

数据和可复现性
本包自包含地重建已完成报告的固定任务数值，包括P@25、P@10、配对差值、
原查询块bootstrap区间、独立guide联合参考重抽样区间，以及指定等权汇总。
保存逐输入/逐查询原评分、原抽样设计、原联合抽样输出；可逆去重压缩保持
原数组精度。D3开发比较使用固定top-K成员概率表示，保留边界并列规则。
程序重新计算汇总与区间并对照原始结果；不是简单打印报告中的最终数值。

这是完整的“固定任务报告数值复现”，不是从原始单细胞表达重新训练模型。
大型原始counts、完整全基因后验和FR分解不重复塞进10MB模块；原拟合与
数据准备源码、配置、文件身份及来源清单保留，完整大数据归档仍单独保留。
换候选、查询、K或重新拟合模型需要相应完整原始归档。本入口不运行拟合。

目录
  reproduce.py                 统一离线入口和完整性校验
  replay.py / scores.npz       主要比较与深度诊断数值重建
  d3_replay.py / d3/           D3开发比较的排名充分数据与重建
  d3_reconciliation.py         D3第1组/五组均值对账
  original_reports/           原评分结果，用于数值核对
  provenance/                 原代码、配置、日志、64项报告来源绑定
  validation/                 本次打包的离线复现核验记录
  PACKAGE_MANIFEST.json        本模块全部文件的SHA-256

输出
  numerical/                  各方法分数、对比CSV、完整JSON与核验结果
  d3_development/             D3开发比较结果与核验
  d3_reconciliation/          单组/五组数字及核验
  BENCHMARK_REPRODUCTION.json 总核验结果

版本与科学状态
并入v1.3.0总代码包后升级为v1.4.0。此前独立knockout RNA和两项交互分析
继续保留在supplements/20261014_upgrade，未被本模块替换。
原报告为历史快照；其中“暂停/待完成”不构成继续执行指令。作者已永久停止
后续实验。D1/D2 Stim扩展及其预注册方案已弃用，已完成的一组输入预测和
零组参考评价记录仍由原总包保存，不计作确认结果。
GPerturb未获得性能结果；独立guide FR rank100完整比较未建立。
所有离线核对只复现既有结果；此次没有新模型拟合或表达读取。

本次离线复现核验：主要比较3,668个数值、D3开发2,521个数值、D3对账24个数值，
合计6,213个原始数值全部通过1e-12容差。64项报告来源哈希全部匹配。
仅验证本新增模块；未再次执行原总代码包的全部历史复现流程。
