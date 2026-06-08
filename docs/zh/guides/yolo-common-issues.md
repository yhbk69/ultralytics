---
comments: true
description: 全面的 YOLO26 常见问题排查指南，涵盖从安装错误到模型训练挑战的各种问题。借助我们的专业技巧提升您的 Ultralytics 项目。
keywords: YOLO, YOLO26, 故障排查, 安装错误, 模型训练, GPU 问题, Ultralytics, 人工智能, 计算机视觉, 深度学习, Python, CUDA, PyTorch, 调试
---

# YOLO 常见问题排查

<p align="center">
  <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yolo-common-issues.avif" alt="YOLO 常见训练和部署问题">
</p>

## 简介

本指南旨在全面帮助您排查在使用 YOLO26 进行 Ultralytics 项目时遇到的常见问题。在正确的指导下，这些问题都能迎刃而解，确保您的项目按计划推进，避免不必要的延误。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/TG9exsBlkDE"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> Ultralytics YOLO26 常见问题 | 安装错误、模型训练问题
</p>

## 常见问题

### 安装错误

安装错误可能由多种原因引起，如版本不兼容、缺少依赖项或环境配置不正确。首先，请确认您是否做到了以下几点：

- 使用推荐的 Python 3.8 或更高版本。
- 确保已安装正确版本的 [PyTorch](https://www.ultralytics.com/glossary/pytorch)（1.8 或更高版本）。
- 考虑使用虚拟环境以避免冲突。
- 按照[官方安装指南](../quickstart.md)逐步操作。

此外，以下是一些用户常见的安装问题及其解决方案：

- 导入错误或依赖问题 - 如果您在导入 YOLO26 时遇到错误，或遇到与依赖相关的问题，请考虑以下排查步骤：
    - **全新安装**：有时，全新安装可以解决意外问题。特别是对于 Ultralytics 这类库，更新可能会引入文件树结构或功能的变化。

    - **定期更新**：确保您使用的是库的最新版本。旧版本可能与最近的更新不兼容，从而导致潜在的冲突或问题。

    - **检查依赖项**：验证所有必需的依赖项是否已正确安装且版本兼容。

    - **审查变更**：如果您最初克隆或安装了旧版本，请注意重大更新可能会影响库的结构或功能。请始终参考官方文档或更新日志以了解任何重大变化。

    - 请记住，保持库和依赖项处于最新状态对于流畅、无差错的体验至关重要。

- 在 GPU 上运行 YOLO26 - 如果您在 GPU 上运行 YOLO26 时遇到问题，请考虑以下排查步骤：
    - **验证 CUDA 兼容性和安装**：确保您的 GPU 兼容 CUDA 且 CUDA 已正确安装。使用 `nvidia-smi` 命令检查 NVIDIA GPU 状态和 CUDA 版本。

    - **检查 PyTorch 与 CUDA 集成**：在 Python 终端中运行 `import torch; print(torch.cuda.is_available())` 以确认 PyTorch 可以使用 CUDA。如果返回 'True'，则 PyTorch 已配置为使用 CUDA。

    - **检查 GPU 兼容性**：自 cuDNN 9.11.0 起，对 Turing 架构之前且计算能力 (SM) < 7.5 的 GPU 架构的支持已被放弃。因此，如果您有较旧的 GPU（如 1080Ti），可能需要使用基于较旧版本 CUDA/cuDNN 构建的 PyTorch 版本。您可以运行以下代码检查：`import torch; cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0); cudnn = torch.backends.cudnn.version() or 0; ok = "not compatible" if cudnn >= 91100 and (cap[0] < 7 or (cap[0] == 7 and cap[1] < 5)) else "should be ok"; print(f"Compute capability: SM {cap[0]}.{cap[1]}, cuDNN: {cudnn} => {ok}")`

    - **环境激活**：确保您在安装了所有必需软件包的正确环境中。

    - **更新软件包**：过时的软件包可能与您的 GPU 不兼容，请保持更新。

    - **程序配置**：检查程序或代码是否指定了 GPU 使用。在 YOLO26 中，这可能位于设置或配置中。

### 模型训练问题

本节将讨论训练过程中遇到的常见问题及其相应的解释和解决方案。

#### 验证配置设置

**问题**：您不确定 `.yaml` 文件中的配置设置在模型训练期间是否正确应用。

**解决方案**：使用 `model.train()` 函数时，`.yaml` 文件中的配置设置应当会被应用。为确保这些设置被正确应用，请按以下步骤操作：

- 确认 `.yaml` 配置文件的路径正确。
- 确保在调用 `model.train()` 时将 `.yaml` 文件的路径作为 `data` 参数传入，如下所示：

    ```python
    model.train(data="/path/to/your/data.yaml", batch=4)
    ```

#### 使用多 GPU 加速训练

**问题**：单 GPU 训练速度较慢，您希望使用多个 GPU 加速该过程。

**解决方案**：增加[批量大小](https://www.ultralytics.com/glossary/batch-size)可以加速训练，但必须考虑 GPU 内存容量。要使用多个 GPU 加速训练，请按以下步骤操作：

- 确保您有多个可用的 GPU。
- 修改 `.yaml` 配置文件以指定要使用的 GPU 数量，例如 `gpus: 4`。
- 相应增加批量大小以充分利用多个 GPU，同时不超过内存限制。
- 修改训练命令以使用多个 GPU：

    ```python
    # 根据需要调整批量大小和其他设置以优化训练速度
    model.train(data="/path/to/your/data.yaml", batch=32)
    ```

#### 持续监控参数

**问题**：您想知道在训练过程中，除了损失值之外还应持续监控哪些参数。

**解决方案**：虽然损失值是需要监控的关键指标，但跟踪其他指标以优化模型性能也同样重要。训练期间需要监控的一些关键指标包括：

- 精确率（Precision）
- 召回率（Recall）
- [平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map)（mAP）

您可以从训练日志中访问这些指标，或使用 TensorBoard、wandb 等工具进行可视化。基于这些指标实施早停策略可以帮助您获得更好的结果。

#### 追踪训练进度的工具

**问题**：您在寻找追踪训练进度的工具建议。

**解决方案**：要追踪和可视化训练进度，您可以考虑使用以下工具：

- [TensorBoard](https://www.tensorflow.org/tensorboard)：TensorBoard 是可视化训练指标的热门选择，包括损失值、[准确率](https://www.ultralytics.com/glossary/accuracy)等。您可以将其与 YOLO26 训练流程集成。
- [Comet](https://bit.ly/yolov8-readme-comet)：Comet 为实验追踪和对比提供了丰富的工具集。它允许您追踪指标、超参数甚至模型权重。与 YOLO 模型的集成也很简单，为您提供实验周期的完整概览。
- [Ultralytics 平台](https://platform.ultralytics.com/)：Ultralytics 平台为追踪 YOLO 模型提供了专用环境，为您提供一站式平台来管理指标、数据集，甚至与团队协作。由于其对 YOLO 的专注，它提供了更定制化的追踪选项。

这些工具各有优势，您在做选择时可以结合项目的具体需求来考虑。

#### 如何检查训练是否在 GPU 上进行

**问题**：训练日志中的 'device' 值为 'null'，您不确定训练是否在 GPU 上进行。

**解决方案**：'device' 值为 'null' 通常意味着训练流程设置为自动使用可用的 GPU，这是默认行为。要确保训练在特定 GPU 上进行，您可以在 .yaml 配置文件中手动将 'device' 值设置为 GPU 索引（例如，第一个 GPU 为 '0'）：

```yaml
device: 0
```

这将显式地将训练流程分配给指定的 GPU。如果您希望在 CPU 上训练，请将 'device' 设置为 'cpu'。

请关注 'runs' 文件夹中的日志和指标，以有效监控训练进度。

#### 有效模型训练的关键考量

如果您遇到模型训练相关问题，以下几点需要牢记。

**数据集格式和标签**

- 重要性：任何[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型的基础在于其训练数据的质量和格式。
- 建议：确保您的自定义数据集及其关联的标签符合预期格式。验证标注是否准确且高质量至关重要。不正确或低质量的标注可能会扰乱模型的学习过程，导致不可预测的结果。

**模型收敛**

- 重要性：实现模型收敛意味着模型已从[训练数据](https://www.ultralytics.com/glossary/training-data)中充分学习。
- 建议：在"从头开始"训练模型时，确保模型达到令人满意的收敛水平至关重要。与微调现有模型相比，这可能需要更长的训练时间和更多的[训练轮次](https://www.ultralytics.com/glossary/epoch)。

**[学习率](https://www.ultralytics.com/glossary/learning-rate)和批量大小**

- 重要性：这些超参数在决定模型在训练期间如何更新权重方面起着关键作用。
- 建议：定期评估所选的学习率和批量大小是否适合您的特定数据集。与数据集特征不匹配的参数可能会影响模型性能。

**类别分布**

- 重要性：数据集中类别的分布会影响模型的预测倾向。
- 建议：定期评估数据集中的类别分布。如果存在类别不平衡，模型有可能会对更普遍的类别产生偏差。这种偏差可以在混淆矩阵中体现出来，模型可能会主要预测多数类。

**与预训练权重交叉验证**

- 重要性：利用预训练权重可以为模型训练提供一个坚实的起点，尤其是在数据有限的情况下。
- 建议：作为诊断步骤，考虑使用相同的数据但以预训练权重初始化来训练模型。如果此方法能产生良好的混淆矩阵，则可能表明"从头开始"的模型需要进一步训练或调整。

### 模型预测相关问题

本节将讨论模型预测过程中遇到的常见问题。

#### 使用 YOLO26 自定义模型获取边界框预测

**问题**：使用自定义 YOLO26 模型运行预测时，边界框坐标的格式和可视化存在挑战。

**解决方案**：

- 坐标格式：YOLO26 以绝对像素值提供边界框坐标。要将这些转换为相对坐标（范围从 0 到 1），需要除以图像尺寸。例如，假设您的图像尺寸为 640x640，则可以执行以下操作：

    ```python
    # 将绝对坐标转换为相对坐标
    x1 = x1 / 640  # x 坐标除以图像宽度
    x2 = x2 / 640
    y1 = y1 / 640  # y 坐标除以图像高度
    y2 = y2 / 640
    ```

- 文件名：要获取正在预测的图像的文件名，在预测循环中直接从结果对象中访问图像文件路径。

#### 在 YOLO26 预测中过滤对象

**问题**：在使用 Ultralytics 库运行 YOLO26 时，如何过滤并仅显示预测结果中的特定对象。

**解决方案**：要检测特定类别，使用 classes 参数指定您希望包含在输出中的类别。例如，仅检测汽车（假设 'cars' 的类别索引为 2）：

```bash
yolo task=detect mode=segment model=yolo26n-seg.pt source='path/to/car.mp4' show=True classes=2
```

#### 理解 YOLO26 中的精度指标

**问题**：对 YOLO26 中边界框精度、掩码精度和[混淆矩阵](https://www.ultralytics.com/glossary/confusion-matrix)精度之间的差异感到困惑。

**解决方案**：边界框精度使用 IoU（交并比）作为指标，衡量预测边界框与实际真实边界框之间的准确度。掩码精度评估预测分割掩码与真实掩码在像素级对象分类中的一致性。而混淆矩阵精度则关注所有类别的整体分类准确率，不考虑预测的几何精度。需要指出的是，即使类别预测错误，[边界框](https://www.ultralytics.com/glossary/bounding-box)也可能几何上准确（真阳性），这导致了边界框精度与混淆矩阵精度之间的差异。这些指标评估模型性能的不同方面，反映了不同任务需要不同评估指标的必要性。

#### 在 YOLO26 中提取对象尺寸

**问题**：在 YOLO26 中检索检测到的对象的长度和高度时遇到困难，尤其是当图像中检测到多个对象时。

**解决方案**：要检索边界框尺寸，首先使用 Ultralytics YOLO26 模型预测图像中的对象，然后从预测结果中提取边界框的宽度和高度信息。

```python
from ultralytics import YOLO

# 加载预训练的 YOLO26 模型
model = YOLO("yolo26n.pt")

# 指定源图像
source = "https://ultralytics.com/images/bus.jpg"

# 进行预测
results = model.predict(source, save=True, imgsz=320, conf=0.25)

# 提取边界框尺寸
boxes = results[0].boxes.xywh.cpu()
for box in boxes:
    x, y, w, h = box
    print(f"边界框宽度: {w}, 边界框高度: {h}")
```

### 部署挑战

#### GPU 部署问题

**问题：** 在多 GPU 环境中部署模型有时可能导致意外行为，如异常的内存使用、不同 GPU 间结果不一致等。

**解决方案：** 检查默认 GPU 初始化。某些框架（如 PyTorch）可能会在转换到指定 GPU 之前在默认 GPU 上初始化 CUDA 操作。为避免意外的默认初始化，在部署和预测期间直接指定 GPU。然后，使用工具实时监控 GPU 利用率和内存使用情况，以识别任何异常。同时，确保使用最新版本的框架或库。

#### 模型转换/导出问题

**问题：** 在将机器学习模型转换或导出为不同格式或平台的过程中，用户可能会遇到错误或意外行为。

**解决方案：**

- 兼容性检查：确保您使用的库和框架版本彼此兼容。版本不匹配可能导致转换期间的意外错误。
- 环境重置：如果您使用 Jupyter 或 Colab 等交互式环境，考虑在进行重大更改或安装后重启环境。全新启动有时可以解决潜在问题。
- 官方文档：始终参考您用于转换的工具或库的官方文档。文档中通常包含模型导出的具体指南和最佳实践。
- 社区支持：查看库或框架的官方仓库，寻找其他用户报告的类似问题。维护者或社区可能在讨论线程中提供了解决方案或变通方法。
- 定期更新：确保使用最新版本的工具或库。开发人员经常发布更新以修复已知错误或改进功能。
- 逐步测试：在进行完整转换之前，先用较小的模型或数据集测试流程，以便及早发现潜在问题。

## 社区与支持

与志同道合的社区互动可以显著提升您使用 YOLO26 的体验和成功率。以下是一些可能有用的渠道和资源。

### 获取帮助的论坛和渠道

**GitHub Issues：** GitHub 上的 YOLO26 仓库有一个 [Issues 标签页](https://github.com/ultralytics/ultralytics/issues)，您可以在其中提问、报告错误和提出新功能建议。社区和维护者在这里都很活跃，是获取具体问题帮助的好地方。

**Ultralytics Discord 服务器：** Ultralytics 拥有一个 [Discord 服务器](https://discord.com/invite/ultralytics)，您可以在其中与其他用户和开发人员交流。

### 官方文档和资源

**Ultralytics YOLO26 文档**：[官方文档](../index.md)提供了 YOLO26 的全面概述，以及安装、使用和故障排查指南。

这些资源应为您排查和改进 YOLO26 项目提供坚实的基础，并帮助您与 YOLO26 社区中的其他人建立联系。

## 总结

故障排查是任何开发过程中不可或缺的一部分，具备正确的知识可以显著减少解决问题所花费的时间和精力。本指南旨在解决 Ultralytics 生态中 YOLO26 模型用户面临的最常见挑战。通过理解和解决这些常见问题，您可以确保项目进展更顺利，并在[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务中取得更好的结果。

请记住，Ultralytics 社区是宝贵的资源。与其他开发人员和专家交流可以提供标准文档中可能未涵盖的额外见解和解决方案。始终保持学习、实验和分享经验的热情，为社区的集体知识做出贡献。

## 常见问题解答

### 如何解决 YOLO26 的安装错误？

安装错误通常由兼容性问题或缺少依赖项引起。确保您使用 Python 3.8 或更高版本，并安装了 PyTorch 1.8 或更高版本。使用虚拟环境有助于避免冲突。有关逐步安装指南，请参阅我们的[官方安装指南](../quickstart.md)。如果遇到导入错误，请尝试全新安装或将库更新到最新版本。

### 为什么我的 YOLO26 模型在单 GPU 上训练很慢？

单 GPU 训练可能因批量大小过大或内存不足而变慢。要加速训练，请使用多个 GPU。确保您的系统有多个可用 GPU，并调整 `.yaml` 配置文件以指定 GPU 数量，例如 `gpus: 4`。相应增加批量大小以充分利用 GPU 而不超过内存限制。示例命令：

```python
model.train(data="/path/to/your/data.yaml", batch=32)
```

### 如何确保 YOLO26 模型在 GPU 上训练？

如果训练日志中的 'device' 值显示 'null'，通常意味着训练流程设置为自动使用可用的 GPU。要显式指定特定 GPU，请在 `.yaml` 配置文件中设置 'device' 值。例如：

```yaml
device: 0
```

这会将训练流程设置为使用第一个 GPU。使用 `nvidia-smi` 命令确认 CUDA 配置。

### 如何监控和追踪 YOLO26 模型的训练进度？

通过 [TensorBoard](https://www.tensorflow.org/tensorboard)、[Comet](https://bit.ly/yolov8-readme-comet) 和 [Ultralytics 平台](https://platform.ultralytics.com/) 等工具可以高效地追踪和可视化训练进度。这些工具允许您记录和可视化损失值、[精确率](https://www.ultralytics.com/glossary/precision)、[召回率](https://www.ultralytics.com/glossary/recall)和 mAP 等指标。基于这些指标实施[早停](#持续监控参数)策略也有助于获得更好的训练效果。

### 如果 YOLO26 无法识别我的数据集格式该怎么办？

确保您的数据集和标签符合预期格式。验证标注准确且高质量。如果您遇到任何问题，请参阅[数据收集与标注](https://docs.ultralytics.com/guides/data-collection-and-annotation)指南了解最佳实践。如需更多数据集相关的指导，请查看文档中的[数据集](https://docs.ultralytics.com/datasets)章节。
