# CUDA 13 训练环境

按 2026-09-22 的官方发布记录，使用 **PyTorch 2.14.0+cu130**（稳定版），CUDA 运行时为 **13.0**。环境由 uv 管理，`pyproject.toml` 声明依赖，`uv.lock` 固定实际解析版本。安装源为 PyTorch 官方 `https://download.pytorch.org/whl/cu130`，不是 nightly。

本机 GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU，12 GB 显存。已实测 CUDA 张量的前向、反向传播。`nvidia-smi` 显示的驱动 CUDA 支持版本与 `torch.version.cuda` 所示的运行时版本是不同信息。

在仓库根目录运行：

```powershell
Set-Location reproduction
.\tools\setup_gpu.ps1 -Python 'C:\path\to\python.exe'
.\tools\train_gpu.ps1
```

Python 使用 3.12。本机已安装，无需再次安装才能训练。首次克隆还需按上一级 README 建立 CPU 预处理环境并获取已校验的数据。`train_gpu.ps1` 顺序运行原代码预处理、GPU 训练、位置解码和原代码回读校验；默认三个种子 0、1、2，每个 100 轮。输出始终写入新的时间戳目录。

## 为什么保留两个环境

论文代码依赖 PyG 2.1 和旧版 `torch-cluster / torch-sparse / torch-scatter`。PyG 为 PyTorch 2.14 提供的 Windows 安装包没有这三个旧扩展。因此，原始图构建与邻居采样在已验证的 CPU 环境中运行，训练中的图特征计算、编码器、损失和反向传播在新的 CUDA 13 环境中运行。

GPU 适配器仅支持本次 Achilles 配置：一阶梯度、无扩散、完整邻居、无可训练内积变换、20 维神经状态、64 维隐藏层和 32 维输出。遇到不支持的参数会明确报错，不作为 MARBLE 全部实验的通用替代实现。

原代码计算 `K[target_ids, :][:, sampled_ids] @ x[sampled_ids]`。GPU 使用 PyTorch 原生稀疏乘法计算 `K @ (bincount(sampled_ids) * x)`，再选取目标行。两者代数等价，保留采样中的重复节点。所有原始采样节点、随机初始化权重和梯度核均保存到实验目录；未加载作者权重开始训练。

验证分三层：

1. 小图上与原代码比较特征、编码器、损失和连续三次 SGD 更新，覆盖重复节点。
2. 每次正式运行前，在真实 Achilles 训练图上比较连续三次 SGD 更新；任一数值检查失败即停止。
3. 训练后将每个种子的最佳模型重新加载到未经修改的 MARBLE 源码，核对测试表征。

浮点加法顺序可能不同；比较采用明确容差，CUDA 稀疏计算不保证逐位确定性。关闭 TF32。所有实验参数与限制另存于 `protocol.json`，检查误差另存于 `gpu_validation.json`。这不等于原论文所有实验已复现。

官方依据：[PyTorch 2.14 发布](https://github.com/pytorch/pytorch/releases/tag/v2.14.0)、[CUDA 13.0 安装包](https://download.pytorch.org/whl/cu130/torch/)、[PyG 对应安装包](https://data.pyg.org/whl/torch-2.14.0%2Bcu130.html)。
