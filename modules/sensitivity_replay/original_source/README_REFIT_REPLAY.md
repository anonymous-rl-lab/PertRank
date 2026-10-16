# D3 B200 重估权重：独立目录重跑说明

这份说明与 `replay_refit_fresh.py` 只补齐执行和打包层。冻结的统计脚本、B=200、种子 `202610053`、细胞、查询、对照、方向、评分与参数全部不改。包装器不是新实验，也不会执行官方 Mixscale。

**当前完成范围：D3 本轮已经完成固定方向下的完整权重重估；D1、D2 只有固定权重分析，未作权重重估。** 旧固定权重报告中“完整重估另需 Gram”描述的是当时状态；本说明覆盖该进度表述，原历史报告与冻结输出保留不改。

## 需要哪两个包

1. `PertRank_Refit_Cell_Resampling_20261005.zip`：冻结脚本、Gram 充分统计量、四份存档结果，以及本说明和执行包装器。
2. `PertRank_Shared_Reproduction_Inputs_20261005.zip`：原复现树，共 453 个文件，保留 `restored/working_v9_with_v6_inputs/code/repro_v10` 相对路径。

**两包缺一不可。** 只取得第一个包，可以读结果，不能完成这次重跑。重新计算 B200 不需要联网，也不需要另取完整原始 D3 计数包；逐细胞核查 Gram 的独立审计另有存档，不由这次包装器重新执行。

## 环境和资源

使用 Python 3.10 或更新版本，并安装 NumPy 和 SciPy；正式重跑的包装器与 `--python` 应使用同一个兼容解释器。包装器会记录这三个版本、精确命令、退出码和耗时，并把常用 BLAS/OpenMP 线程数设为 1。准备阶段只使用 Python 标准库，不导入计数矩阵，不运行任何重抽。

输入 ZIP 通过逐文件流式复制，避免把整个包读入内存。正式统计脚本仍会加载查询逐细胞矩阵，运行时应留出数 GB 内存；不要与 Mixscale 全量拟合同时执行。需要为输入树和新结果预留磁盘空间。此包装器不删除既有文件。

## 第一步：仅检查解包和路径

先从重抽包取出本说明及 `strengthening_neighbor/replay_refit_fresh.py`。将下列示例中的 ZIP 路径改为本机路径。`--workdir` 必须不存在或完全为空。

```bash
python strengthening_neighbor/replay_refit_fresh.py \
  --refit-package /path/to/PertRank_Refit_Cell_Resampling_20261005.zip \
  --shared-package /path/to/PertRank_Shared_Reproduction_Inputs_20261005.zip \
  --workdir /path/to/refit_path_check \
  --prepare-only
```

它验证共享包 453 个文件和必要重抽文件的 SHA256/字节数、原科学冻结记录、Gram 与存档结果的绑定、语法和必要路径。成功标记为 **`PREPARED_ONLY_NOT_REPLAYED`**，不能据此声称已经重算 B200。

## 第二步：完整重跑

指定另一个空目录，不复用第一步的非空目录；这样原始输入和第一次检查都完整保留。

```bash
python strengthening_neighbor/replay_refit_fresh.py \
  --refit-package /path/to/PertRank_Refit_Cell_Resampling_20261005.zip \
  --shared-package /path/to/PertRank_Shared_Reproduction_Inputs_20261005.zip \
  --workdir /path/to/refit_full_replay
```

包装器创建以下相对布局，冻结脚本可以原样使用：

- `strengthening_neighbor/`：原冻结脚本、诊断脚本及 `d3_prepared/d3_fixed_control_gram.npz`。
- `restored/working_v9_with_v6_inputs/code/repro_v10/`：共享输入树。
- `expected/`：只读用途的四份存档结果，与新写出的结果分开。
- `replay_audit/`：输入清单、命令、日志、运行环境和逐字段比较。

实际调用的两个统计命令为：

```bash
python strengthening_neighbor/refit_cell_bootstrap.py \
  --gram strengthening_neighbor/d3_prepared/d3_fixed_control_gram.npz
python strengthening_neighbor/weight_refit_diagnostics.py
```

没有使用 `--smoke`，没有减少重抽次数。第二条诊断也使用原 B200 种子和原逐靶点抽样次数；它只描述权重分布，不产生新方法或重新选择参数。

## 完成如何认定

必须同时满足：两个统计命令正常退出；四份 JSON/NPZ 结果齐全；所有 JSON 字段以及所有 NPZ 数组的键、形状、dtype 和数值与存档相符。

`replay_audit/field_comparison.json` 逐文件记录精确与非精确数值个数、最大绝对/相对差，以及所有出现非零差的字段或数组。浮点跨环境对照容差预先固定为 `abs(new-old) <= 1e-12 + 1e-10*abs(old)`；字符串、结构、布尔值及 NPZ 非数值数组须完全一致。任何超差使进程失败；不自动扩大容差。

整数计数和整数数组必须精确一致，不套浮点容差。科学冻结文件的核查使用包内原冻结记录；包里保留的历史记录不被当前执行包装器改写或覆盖。

- `REPLAY_PASS_EXACT`：逐字段数值精确相同。
- `REPLAY_PASS_WITHIN_TOLERANCE`：存在已列出的浮点差，但全部在上述容差内，不能写“逐值完全相同”。
- `FAILED`：命令失败或结果不符，查看对应日志。

NPZ 是容器，ZIP 时间戳可能让整个文件 SHA256 不同；包装器另外记录新旧文件哈希，并对每个数组做真正的数值比较，不把容器哈希差误判成科学数值差。

## 结果的统计身份

这次 B200 是 **D3 已有每靶点 20 个细胞的经验重抽敏感性分析**。每轮对同一个抽样重新计算均值、欧氏权重和 W1 权重；固定输入 NTC 均值、协方差方向、查询、候选池和深层参考。它不是包括全部来源变异的联合 95% 置信区间，也不是新抽到独立 20 个细胞时的直接性能分布。

存档中的负结果必须保留：响应查询 W1−w0y 的重抽平均为约 −0.00344，经验 2.5%–97.5% 分位范围约 [−0.00759, −0.00032]。同一经验细胞分布的均值会保留原样本偏移，再重抽会附加波动；这限制了外推解释，但不抹去本次“经验重抽并重估权重”出现的敏感性。主文原条件 query-block 区间与这项分析回答不同的问题，应并列报告。

这里只复核既有 B200 和权重诊断，不重算原始 Gram 构造、不修改已冻结主结果，也不声称完成全流程 bootstrap。
