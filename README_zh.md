# PertRank v1.5.4 精简复现包

本版同步升级软件为 **0.2.0rc2**：修复外部效应复用，提供精确并列 P@K 评分，并对齐 v25.2 的方法定位。科学数据与表格不变；本轮验证软件安装与接口，不新增科学实验。

本版为私有匿名审稿复现包：自有注册标识使用角色别名，原始对应关系另存私有档。交付身份及可用于主文的代码可得性说明见 `docs/REVIEW_IDENTITY.md` 和 `docs/CODE_AVAILABILITY.txt`。

本版整理科学代码及可重算论文数值和区间的中间数据，清除重复交付文件。整理后保留科学代码与数值数组，来源及文件迁移见 `docs/RELEASE_MIGRATION.csv`。本版只提供完整总包，ZIP 体积控制在 100 MB 以内，按十进制字节验收。

主要删除：新增部分的逐细胞原始 counts、整套预测张量、可由现有中间量重建的数组、重复数据和可重画的图片。原始实验与已完成结果没有被删除或改写；它们的来源和必要身份保留在说明中。

保留：逐查询评分、分块和随机种子、必要靶点/guide读出、参考成员与模型输出、重抽样中间量、完整图表Source Data、脚本及校验记录。压缩不降低浮点精度。部分模块从“细胞counts→预测”改为“中间评分→均值、差值与区间”；这是用户指定的精简复现层级，不等于从全部原始UMI重训。

## 使用完整包

解压后，在包根目录执行：

```bash
python -m pip install -r requirements.txt
python reproduce.py --verify-only
python reproduce.py --output-dir ../pertrank-reproduced
```

建议 Python 3.12 和独立环境。输出目录必须是包外的新目录。查看 `REPRODUCTION_REPORT.json` 获取每个模块的实际核验状态。

## 可选软件与真实数据教程

AnnData 软件使用独立环境，并先复制源码到总包以外。原地 pip 安装会在源码目录生成 `build/` 和 `*.egg-info/`，使总包严格清单检查报错。请在总包根目录执行：

```bash
python -c "from pathlib import Path; import shutil; shutil.copytree(Path('modules/paper_analysis/software'), Path('../pertrank-software-source'))"
python3.12 -m venv ../pertrank-software-env
../pertrank-software-env/bin/python -m pip install -r requirements-software.txt
python reproduce.py --verify-only
python reproduce.py --only software --software-python ../pertrank-software-env/bin/python --output-dir ../pertrank-software-check
```

复制命令要求目标目录尚不存在，避免覆盖已有文件。`requirements-software.txt` 指向包外副本；执行安装命令时应保持当前目录为总包根目录。调用 `reproduce.py` 时仍使用数值复现环境，通过 `--software-python` 指定独立软件环境。软件、构建产物和输出均不写回总包。

论文与各模块对应关系见 `docs/PAPER_COVERAGE.csv`。历史A14缺原逐细胞输入，仍明确作为存档结果保留；原字节未找回的历史重建项继续单独注明。默认命令不新增实验、不重新训练外部模型。

## GitHub Desktop

本版已初始化本地 Git 仓库。解压后直接用 **File → Add Local Repository** 选择同时包含 `README.md`、`modules`、`docs` 的 `PertRank` 根目录。详见 [Windows 操作说明](docs/GITHUB_DESKTOP_zh.md)。
