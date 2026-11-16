# PertRank 双匿名审稿复现包

本包包含主方法、当前论文表图来源、补齐的独立 RNA 四臂与 guide 诊断、可安装软件及 PertRank-Q。审稿包不包含项目作者姓名、邮箱、单位、可识别作者身份的项目链接或 Git 历史。

## 三个文档的准确位置

- [科学设计说明](https://github.com/anonymous-rl-lab/PertRank/blob/main/modules/paper_analysis/software/docs/scientific_design.md)
- [API 说明](https://github.com/anonymous-rl-lab/PertRank/blob/main/modules/paper_analysis/software/docs/api.md)
- [Replogle 实际数据教程](https://github.com/anonymous-rl-lab/PertRank/blob/main/modules/paper_analysis/software/tutorials/01_replogle.ipynb)

以上三个链接通过 HTTPS 打开匿名 GitHub 仓库中的对应文件；同一文件仍包含在源码中，可离线阅读。PyPI 不会把源码包内部路径自动变成在线文档页面，生成的 PyPI 描述保留这三个 HTTPS 链接，并将其余本地链接转换为明确的源码归档路径。新软件源码为审稿修订版；本次修改没有向 PyPI 或 GitHub 发布。

## 使用

详细命令见 [英文 README](README.md)。先安装 `requirements.txt`，再执行：

```bash
python reproduce.py --verify-only
python reproduce.py --output-dir ../pertrank-reproduced
```

输出目录必须在包外且尚不存在。默认运行八个科学工作流。Q 需要额外依赖，可单独运行或加入总流程：

```bash
python -m pip install -r modules/pertrank_q/requirements-q.txt --extra-index-url https://download.pytorch.org/whl/cpu
python reproduce.py --with-q --output-dir ../pertrank-all-reproduced
```

软件测试使用独立环境，避免与固定数值复现环境的 SciPy 版本冲突。不要直接在包内执行 pip 构建；复制软件目录到包外后安装。

## 怎样理解“复现”

各模块从明确标注的中间量开始。Q 从冻结检查点重建预测，不重新训练；图表从对应 Source Data 重绘，不等于上游模型重新拟合。原始输入未恢复的项目保留缺口说明，不纳入已通过复现的数量。

当前数据和代码导航见 [覆盖表](docs/CURRENT_PAPER_COVERAGE.md)，本轮检查结果见 [审计报告](docs/AUDIT_zh.md)。删除的是过时交付材料和重复副本，未删除当前论文依赖的阴性证据。第三方许可及独立数据来源继续保留。

Windows 可使用 `py -3.12 -m venv ..\pertrank-env` 建立环境，再用 `..\pertrank-env\Scripts\python.exe` 执行对应的安装与复现命令。无需启用环境激活脚本。
