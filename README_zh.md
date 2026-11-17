# PertRank 科研复现包

本包包含主方法、当前论文表图来源、独立 RNA 四臂与 guide 诊断、可安装软件及 PertRank-Q。审稿包不包含项目作者姓名、邮箱、单位、可识别作者身份的项目链接或 Git 历史。

## 三个文档的准确位置

- [科学设计说明](modules/paper_analysis/software/docs/scientific_design.md)
- [API 说明](modules/paper_analysis/software/docs/api.md)
- [Replogle 实际数据教程](modules/paper_analysis/software/tutorials/01_replogle.ipynb)

以上三个相对链接指向随包文件，在仓库内或解压后均可使用。可安装软件源码版本为 0.2.0rc3；科研计算材料完整包含在一个总包中。

## 使用

详细命令见 [英文 README](README.md)。先安装 `requirements.txt`，再执行：

```bash
python reproduce.py --verify-only
python reproduce.py --output-dir ../pertrank-reproduced
```

输出目录必须在包外且尚不存在。默认执行九项科研工作流，覆盖主方法、独立 RNA、guide 诊断、rvalues 比较与程序提名。Q 需要额外依赖，可单独运行或加入总流程：

```bash
python -m pip install -r modules/pertrank_q/requirements-q.txt --extra-index-url https://download.pytorch.org/whl/cpu
python reproduce.py --with-q --output-dir ../pertrank-all-reproduced
```

软件测试使用独立环境，避免与固定数值复现环境的 SciPy 版本冲突。不要直接在包内执行 pip 构建；复制软件目录到包外后安装。

## 怎样理解“复现”

各模块从明确标注的中间量开始。Q 从冻结检查点重建预测，不重新训练；图表从对应 Source Data 重绘，不等于上游模型重新拟合。原始输入未恢复的项目保留缺口说明，不纳入已通过复现的数量。


Windows 可使用 `py -3.12 -m venv ..\pertrank-env` 建立环境，再用 `..\pertrank-env\Scripts\python.exe` 执行对应的安装与复现命令。无需启用环境激活脚本。

## 程序提名、排序比较与原生拟合材料

排序比较与程序提名位于 `modules/new_results/`，包括 rvalues 固定补充比较和程序提名（D3、sci-Plex3 冻结验证及单独标注的探索分支）。总包同时包含原生 rvalues 拟合输入、主方法、独立 RNA、guide、Q 及不利结果，无需另外下载实验附件。

```bash
python reproduce.py --only new_results --output-dir ../pertrank-new-results
python reproduce.py --only new_results_exploratory --output-dir ../pertrank-program-exploratory
python reproduce.py --only sn4_diagnostics --output-dir ../pertrank-sn4
python reproduce.py --only rvalues_native --output-dir ../pertrank-rvalues-materials
```

`rvalues_native` 入口还原并校验原生拟合材料；实际 R 重拟合另按该模块 README 执行。图表渲染与数值复现分别报告。

## 无损存储

大数组存放在包内的 `storage/`，运行根入口时自动还原到输出目录的 `_source/`。还原逐文件核对原始 SHA256；数值精度、名单规则和重采样设置不变。运行需要额外磁盘空间，不会修改收到的源码目录。

本次清理范围与验证结果见 [审计报告](AUDIT_REPORT_CN.md)。
