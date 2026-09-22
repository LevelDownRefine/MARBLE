# MARBLE 复现记录

更新日期：2026-09-22。已完成 Fig. 5c–d 的作者模型重评估，以及 Achilles 单动物、三个种子的 MARBLE 从头训练。新训练使用官方稳定版 PyTorch 2.14.0、CUDA 13.0 和 uv 独立环境；尚未完成全论文复现。

- [从头训练报告：3 个种子 × 100 轮](reproduction/TRAINING_REPORT.md)
- [CUDA 13 环境与复运行命令](reproduction/gpu/README.md)
- [首轮作者权重重评估](reproduction/README.md)
- [逐步阅读：数据如何变成位置预测](reproduction/READING.md)
- 从头训练主评估 MAE：9.72、9.48、8.77 cm，均值 ± 种子间标准差为 9.32 ± 0.49 cm。作者权重本地重评估为 9.05 cm；所有主指标使用相同的 1,999 个测试时间点。
- 10 项 unittest、3 项原仓库 pytest、ruff 均通过；GPU 训练结果已用本 fork 的原始 MARBLE 源码回读核对。

## 仓库与论文

- 工作仓库：<https://github.com/LevelDownRefine/MARBLE>，本地远程名 `origin`。
- 作者仓库：<https://github.com/Dynamics-of-Neural-Systems-Lab/MARBLE>，本地远程名 `upstream`。
- 当前起点：`09491ec077bf153b7dece8518cc65ddaa20eb26d`，分支 `main`。
- 论文：*MARBLE: interpretable representations of neural population dynamics using geometric deep learning*，Nature Methods，2025。<https://doi.org/10.1038/s41592-024-02582-2>

## 第一个目标：Fig. 5c–d，大鼠位置解码

使用 Achilles 的海马记录，沿“神经活动 → 预处理 → MARBLE 表征 → 位置解码”重做一个完整分析链。

1. 先读 [decoding.ipynb](examples/rat_hippocampus/decoding.ipynb)，明确输入、数据划分、预处理和评价指标。
2. 使用作者检查点重新计算解码结果，与 CEBRA-time、CEBRA-behaviour 比较。这一步属于预训练模型重评估。
3. 再按固定配置从头训练 MARBLE，保存配置、日志、检查点和结果。这一步单独记录，不能与第 2 步混称。
4. 第一轮完成后，再扩展到跨动物一致性和参数敏感性。

官方另有 [训练 notebook](examples/rat_hippocampus/run_marble_and_cebra.ipynb)、[跨动物 notebook](examples/rat_hippocampus/multianimal_consistency.ipynb) 和 [示例说明](examples/rat_hippocampus/README.md)。

## 数据与模型

以下四个文件的链接来自当前提交的官方 `decoding.ipynb`。本轮从已核验的本地副本复制到 `reproduction/data/rat/`，并再次核对大小和 SHA-256；新流程不依赖相邻的旧项目。

| 文件 | 字节数 | 作者下载地址 | 用途 |
| --- | ---: | --- | --- |
| `rat_data.pkl` | 22,055,089 | <https://dataverse.harvard.edu/api/access/datafile/7609512> | 整理后的真实神经活动与行为标签 |
| `marble_achilles_32D.pth` | 251,915 | <https://dataverse.harvard.edu/api/access/datafile/7659512> | MARBLE 作者检查点 |
| `cebra_time_achilles_32D.pt` | 584,362 | <https://dataverse.harvard.edu/api/access/datafile/7609517> | CEBRA-time 作者检查点 |
| `cebra_behaviour_achilles_32D.pt` | 584,490 | <https://dataverse.harvard.edu/api/access/datafile/7609520> | CEBRA-behaviour 作者检查点 |

数据入口是处理后的神经记录，不是原始采集仪器文件。检查点是已训练模型，不能当成训练数据。

官方 notebook 读取 `achilles` 下的 `neural` 和 `continuous_index`，按时间前 80% / 后 20% 划分训练与测试数据。MARBLE 预处理在训练集拟合 PCA=20，并将同一个 PCA 应用于测试集。

## 需要固定的比较条件

- 数据文件的来源、大小、SHA-256，以及代码提交。
- Python、PyTorch、scikit-learn、CEBRA 和 MARBLE 依赖版本。
- 时间划分、PCA 求解器与符号约定、图构建参数和随机种子。
- 加载作者权重还是从头训练，以及推断时的 `train` / `eval` 模式。
- 解码器设置，位置 MAE 的单位（米）和位置 R²。

旧环境记录表明 PCA 版本可能影响作者权重重评估。应先还原预处理，再判断模型性能差异。官方脉冲转换实现与物理时间单位的对应也需要单独核查；任何修正必须另列实验，不能静默改变复现条件。

首轮交付应包含配置、命令、环境版本、解码轨迹、误差图、数值指标和运行日志。单只动物的时间点不能当作独立动物重复；首轮结果只支持所选记录与划分下的比较。

## 初始材料核验记录（首次准备阶段）

相邻的 `../MARBLE-reproduction/` 已有独立 CPU 环境、复现脚本、数据和历史报告，其内的作者源码与本 fork 当前提交一致。它尚未迁入本仓库；原环境的包导入仍关联旧源码目录，不能据此假定本 fork 的后续修改已经生效。

首次准备阶段实际执行：

- 克隆用户 fork，并配置 `origin` 和 `upstream`。
- 核对 29 个已有数据／模型文件，共 1,860,225,828 字节；大小和 SHA-256 与本地收据一致。
- 核对已有 `results/paper/manifest.json` 中 24 个代码、锁文件及指标文件；哈希全部一致。
- 在原 `MARBLE-reproduction` 环境重新执行 `tools/check.ps1`：10 项 unittest、3 项上游 pytest 通过，ruff 通过；pytest 仍有依赖弃用和稀疏张量提示。

这些初始检查证明文件与记录一致、已有测试通过，不等于独立重做论文实验。此后已另建独立环境并完成本页开头链接的本轮重评估；本轮记录和历史报告分别保留。

历史报告仅作为待核对的对照材料：`../MARBLE-reproduction/results/paper/REPORT.md`。其中 `rat-retrained/metrics.json` 同时存在 `marble_retrained=true` 和指向作者检查点的 `source` 文本；该来源描述不一致，需在新实验中明确记录，不能直接照抄为已验证结论。

## 接续顺序

先沿 [READING.md](reproduction/READING.md) 读懂输入、预处理、局部动力学特征和解码指标，再看已完成的 [从头训练报告](reproduction/TRAINING_REPORT.md)。后续可扩展其他动物、在相同划分下训练 CEBRA，以及单独检验预处理时间单位的影响。
