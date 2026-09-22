# Fig. 5c–d：从海马活动预测大鼠位置

这一步要回答：海马神经元的群体活动，经过 MARBLE 表征以后，能否让一个简单解码器更准确地预测大鼠在轨道上的位置？

当前范围是 Achilles 单只动物的作者模型重评估；表示模型没有重新训练，位置解码器会在训练段重新拟合。依据是论文 Fig. 5 和 Methods / Linear maze navigation，以及当前提交的 [decoding.ipynb](../examples/rat_hippocampus/decoding.ipynb)。

## 数据经过哪些步骤

| 步骤 | 数据含义与处理 | 代码出处 |
| --- | --- | --- |
| 读取 | `rat_data.pkl` 中选择 `achilles`；`neural` 是按时间排列的神经活动，`continuous_index` 保存位置和运动方向标签 | notebook 的数据读取单元 |
| 划分 | 前 80% 时间用于拟合，后 20% 用于解码评价；先划分再预处理 | notebook / `split_data` |
| 发放率 | 官方函数将非零脉冲 bin 转成事件，并平滑为发放率曲线 | [rat_utils.py / convert_spikes_to_rates](../examples/rat_hippocampus/rat_utils.py) |
| 神经状态 | 只在训练段拟合 PCA=20，测试段使用同一个变换，得到状态 x(t) | 同上 |
| 状态变化 | 计算 x(t+1)−x(t)，作为每个状态上的向量，因此每段少一个点 | 同上 / `np.diff` |
| 邻域与局部变化 | 以神经状态构造 CkNN 图（k=15、delta=1.5），计算局部梯度算子 | [preprocessing.py / construct_dataset](../MARBLE/preprocessing.py) |
| MARBLE 表征 | 作者的 32 维检查点把状态与局部变化特征映射到表征空间 | [main.py / net.forward、transform](../MARBLE/main.py) |
| 位置预测 | 用训练段的表征和真实位置拟合 36 邻居、余弦距离的 kNN，再预测测试段位置 | rat_utils.py / `decoding_pos_dir`；本次只取位置任务 |
| 比较 | 用相同测试时间点比较 MARBLE、CEBRA-time、CEBRA-behaviour 的位置绝对误差和 R² | [src/rat_decoding.py](src/rat_decoding.py) |

注意两个不同的“位置”：PyG 数据的 `data.pos` 是 **PCA 后的神经状态坐标**，并非大鼠在轨道上的物理位置。检查点的 `include_positions=True` 指前者，不表示把待预测的大鼠位置直接输入 MARBLE。真实行为位置用于下游解码器拟合及评价。

PCA 输出 20 维、MARBLE 输出 32 维也不矛盾：此处 MARBLE 编码的是局部动力学特征，不是再次对 20 维坐标做一个线性压缩。这个检查点使用一阶局部梯度、保留中心向量并拼接神经状态，特征数为 20+20×20+20=440，再由 MLP 映射到 32 维。[实现](../MARBLE/main.py)

## 为什么要并列两个运行模式

官方 `decoding.ipynb` 加载模型后直接 `transform`；当前实现的 `transform` 关闭梯度，但不自动执行 `eval()`，所以 batch normalization 仍采用当前批次统计量。本次保留这条路径，标记为 `MARBLE-notebook`。

另外加载一份全新的同一检查点，显式调用 `eval()`，标记为 `MARBLE-eval`。它使用检查点保存的归一化统计量。两次运行独立加载权重和缓冲区，防止第一条路径更新缓冲区后影响第二条。[实现](../MARBLE/main.py)

这两个模式需要分别报告；`torch.no_grad()` 和 `model.eval()` 的作用不同。

## 必须保留的协议差异

- 论文 Methods 描述 PCA=5，解码 notebook 使用 PCA=20。本轮以 notebook 和 32 维检查点为准，并固定 scikit-learn 1.3.2 的 PCA 行为。
- 论文图注使用“linearly decoded”的表述，但 Methods 和代码明确使用 36 邻居的 kNN。本轮实现 kNN。
- 官方预处理把非零 bin 的索引作为毫秒输入平滑函数，而记录中的 bin 是 25 ms；还会丢弃单个 bin 内多次脉冲的计数。本轮保留作者实现以对齐权重，物理时间解释和修正后的新模型属于后续独立实验。
- MARBLE 的差分少一个测试点，CEBRA 原生输出不减少。本轮主指标把 CEBRA 最后一个点也去掉，统一时间点；额外保留 CEBRA 的完整原生窗口指标。
- 作者模型配置为 `diffusion=False`。为避免计算不使用的完整谱，本轮只计算一个 Laplacian 特征对，并用小图回归测试验证这项改动不改变无扩散模型的输出。

## 怎样理解结果

位置 MAE 为平均 |预测位置−真实位置|，单位是米，越小越好；同时报告中位绝对误差和位置 R²。模型采用作者检查点，本次不能据此声称已经复现从头训练。

测试图使用整个测试窗口中的神经活动构建，因此这是离线解码；时间前后划分也不意味着各时间点相互独立。本轮不以逐时间点 p 值宣称跨动物的统计显著性。

论文链接：<https://doi.org/10.1038/s41592-024-02582-2>。
