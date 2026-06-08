---
comments: true
description: 学习训练计算机视觉模型的最佳实践，包括批量大小优化、混合精度训练、早停法以及优化器选择，以提高效率和准确率。
keywords: 模型训练 机器学习, AI 模型训练, 训练轮数, 如何训练机器学习模型, 机器学习最佳实践, 什么是模型训练
---

# 模型训练的机器学习最佳实践与技巧

## 简介

在开展[计算机视觉项目](./steps-of-a-cv-project.md)时，最重要的步骤之一就是模型训练。在进入这一步之前，你需要先[明确目标](./defining-project-goals.md)并[收集和标注数据](./data-collection-and-annotation.md)。在[对标注数据进行预处理](./preprocessing_annotated_data.md)以确保数据干净且一致之后，就可以开始训练模型了。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GIrFEoR5PoU"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>模型训练技巧 | 如何处理大型数据集 | 批量大小、GPU 利用率与<a href="https://www.ultralytics.com/glossary/mixed-precision">混合精度</a>
</p>

那么，什么是[模型训练](../modes/train.md)？模型训练是让模型学会识别视觉模式并根据你的数据做出预测的过程。它直接影响到应用的性能和准确率。在本指南中，我们将介绍最佳实践、优化技巧和故障排除建议，帮助你高效地训练计算机视觉模型。

## 如何训练机器学习模型

计算机视觉模型通过调整其内部参数来最小化误差。一开始，模型会被输入大量带标签的图像。它会对这些图像的内容做出预测，然后将预测结果与实际标签或内容进行比较以计算误差。这些误差反映了模型预测与真实值之间的差距。

在训练过程中，模型通过一种称为[反向传播](https://www.ultralytics.com/glossary/backpropagation)的过程，反复进行预测、计算误差并更新参数。在此过程中，模型调整其内部参数（权重和偏置）以减少误差。通过多次重复这个循环，模型的准确率逐步提高。随着时间的推移，它学会识别复杂的模式，如形状、颜色和纹理。

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/backpropagation-diagram.avif" alt="什么是反向传播？">
</p>

这个学习过程使[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型能够执行各种[任务](../tasks/index.md)，包括[目标检测](../tasks/detect.md)、[实例分割](../tasks/segment.md)和[图像分类](../tasks/classify.md)。最终目标是创建一个能够将其学习成果泛化到新的、未见过的图像上的模型，从而在真实世界应用中准确理解视觉数据。

了解了训练模型时背后发生的事情之后，让我们来看看训练模型时需要考虑的几个要点。

## 在大型数据集上训练

当你计划使用大型数据集训练模型时，有几个不同的方面需要考虑。例如，你可以调整批量大小、控制 GPU 利用率、选择使用多尺度训练等。让我们逐一详细了解这些选项。

### 批量大小与 GPU 利用率

在大型数据集上训练模型时，高效利用 GPU 是关键。批量大小是一个重要因素，它指的是机器学习模型在单次训练迭代中处理的数据样本数量。
使用 GPU 支持的最大批量大小，可以充分发挥其能力，减少模型训练所需的时间。但需要注意避免 GPU 内存溢出。如果遇到内存错误，应逐步减小批量大小，直到模型能够顺利训练。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Gxl6Bbpcxs0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 Ultralytics YOLO26 中使用批量推理 | 在 Python 中加速目标检测 🎉
</p>

对于 YOLO26，你可以在[训练配置](../modes/train.md)中设置 `batch_size` 参数来匹配你的 GPU 容量。此外，在训练脚本中设置 `batch=-1` 会根据你的设备能力自动确定能够高效处理的[批量大小](https://www.ultralytics.com/glossary/batch-size)。通过微调批量大小，你可以充分利用 GPU 资源，改善整体训练过程。

### 子集训练

子集训练是一种明智的策略，即在一个能够代表整个数据集的较小数据子集上训练模型。它可以节省时间和资源，尤其是在模型开发的初始阶段和测试阶段。如果你时间紧迫或正在尝试不同的模型配置，子集训练是一个不错的选择。

对于 YOLO26，你可以通过使用 `fraction` 参数轻松实现子集训练。该参数允许你指定用于训练的数据集比例。例如，设置 `fraction=0.1` 将在 10% 的数据上训练模型。在投入完整数据集训练之前，你可以使用此技术进行快速迭代和模型调优。子集训练有助于你快速推进并尽早发现潜在问题。

### 多尺度训练

多尺度训练是一种通过在多种尺寸的图像上训练模型来提高其泛化能力的技术。你的模型可以学习在不同尺度和距离上检测物体，从而变得更加鲁棒。

例如，在训练 YOLO26 时，你可以通过设置 `scale` 参数来启用多尺度训练。该参数按指定因子调整训练图像的尺寸，模拟不同距离下的物体。例如，设置 `scale=0.5` 会在训练期间以 0.5 到 1.5 之间的随机因子缩放训练图像。配置此参数可以让模型体验多种图像尺度，提高其在不同物体大小和场景下的检测能力。

Ultralytics 还通过 `multi_scale` 参数支持图像尺寸多尺度训练。与 `scale` 缩放图像后再填充/裁剪回 `imgsz` 不同，`multi_scale` 在每个批次直接改变 `imgsz` 本身（按模型步长取整）。例如，设置 `imgsz=640` 和 `multi_scale=0.25`，训练尺寸会按步长从 480 到 800 进行采样（如 480, 512, 544, ..., 800），而 `multi_scale=0.0` 则保持固定尺寸。

### 缓存

缓存是提高机器学习模型训练效率的重要技术。通过在内存中存储预处理后的图像，缓存减少了 GPU 等待从磁盘加载数据的时间。模型可以持续接收数据，而不会因磁盘 I/O 操作而延迟。

训练 YOLO26 时，可以通过 `cache` 参数控制缓存：

- _`cache=True`_：将数据集图像存储在 RAM 中，提供最快的访问速度，但会增加内存使用量。
- _`cache='disk'`_：将图像存储在磁盘上，比 RAM 慢，但比每次加载新数据更快。
- _`cache=False`_：禁用缓存，完全依赖磁盘 I/O，这是最慢的选项。

### 混合精度训练

混合精度训练同时使用 16 位 (FP16) 和 32 位 (FP32) 浮点类型。通过利用 FP16 进行更快的计算、FP32 在需要时保持精度，充分发挥了两者的优势。[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)的大部分操作在 FP16 中完成，以享受更快的计算速度和更低的内存使用量。然而，模型权重的主副本以 FP32 保存，以确保权重更新步骤中的精度。你可以在相同的硬件限制下处理更大的模型或更大的批量大小。

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mixed-precision-training-overview.avif" alt="混合精度 FP16 训练的优势">
</p>

要实现混合精度训练，你需要修改训练脚本并确保你的硬件（如 GPU）支持它。许多现代[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)框架（如 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 和 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow)）提供了内置的混合精度支持。

在使用 YOLO26 时，混合精度训练非常简单。你可以在训练配置中使用 `amp` 标志。设置 `amp=True` 即可启用自动混合精度 (AMP) 训练。混合精度训练是一种简单而有效的优化模型训练过程的方法。

### 预训练权重

使用预训练权重是加速模型训练过程的一种明智方法。预训练权重来自已经在大规模数据集上训练过的模型，为你的模型提供了一个良好的起点。[迁移学习](https://www.ultralytics.com/glossary/transfer-learning)将预训练模型适配到新的相关任务上。微调预训练模型是指从这些权重开始，然后在你的特定数据集上继续训练。这种训练方法可以缩短训练时间，并且通常能获得更好的性能，因为模型从一开始就具备对基本特征的良好理解。

`pretrained` 参数让 YOLO26 中的迁移学习变得简单。设置 `pretrained=True` 将使用默认的预训练权重，或者你也可以指定自定义预训练模型的路径。使用预训练权重和迁移学习可以有效提升模型能力并降低训练成本。

### 处理大型数据集时需要考虑的其他技巧

处理大型数据集时，还有几个其他技巧值得考虑：

- **[学习率](https://www.ultralytics.com/glossary/learning-rate)调度器**：使用学习率调度器可以在训练过程中动态调整学习率。经过良好调优的学习率可以防止模型越过最小值，提高稳定性。在训练 YOLO26 时，`lrf` 参数通过将最终学习率设置为初始学习率的一个比例来帮助管理学习率调度。
- **分布式训练**：对于处理大型数据集，分布式训练可能是一个改变游戏规则的方法。通过将训练工作负载分布到多个 GPU 或机器上，你可以减少训练时间。这种方法对于拥有大量计算资源的企业级项目尤其有价值。

## 训练的轮数

在训练模型时，[epoch（轮次）](https://www.ultralytics.com/glossary/epoch)指的是对完整训练数据集的一次遍历。在一个 epoch 中，模型处理训练集中每个样本一次，并根据学习算法更新其参数。通常需要多个 epoch 来让模型随时间学习和优化其参数。

一个常见的问题是如何确定训练的 epoch 数量。一个不错的起点是 300 个 epoch。如果模型出现过早[过拟合](https://www.ultralytics.com/glossary/overfitting)的现象，可以减少 epoch 数量。如果 300 个 epoch 后未出现过拟合，可以将训练延长到 600、1200 或更多个 epoch。

然而，理想的 epoch 数量可能因数据集大小和项目目标而异。较大的数据集可能需要更多 epoch 以便模型有效学习，而较小的数据集可能需要较少的 epoch 以避免过拟合。对于 YOLO26，你可以在训练脚本中设置 `epochs` 参数。

## 早停法

早停法是一种优化模型训练的重要技术。通过监控验证性能，你可以在模型停止改进时停止训练。这可以节省计算资源并防止过拟合。

该过程涉及设置一个耐心（patience）参数，该参数决定了在停止训练之前等待验证指标改进的 epoch 数量。如果模型的性能在这些 epoch 内没有改进，则停止训练以避免浪费时间和资源。

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/early-stopping-overview.avif" alt="早停法防止模型过拟合">
</p>

对于 YOLO26，你可以通过在训练配置中设置 patience 参数来启用早停法。例如，`patience=5` 表示如果连续 5 个 epoch 验证指标没有改进，训练将停止。使用这种方法可以确保训练过程保持高效，并达到最佳性能而不消耗过多的计算资源。

## 选择云端训练还是本地训练

训练模型有两种选择：云端训练和本地训练。

云端训练提供可扩展性和强大的硬件，非常适合处理大型数据集和复杂模型。像 [Google Cloud](https://cloud.google.com/)、[AWS](https://aws.amazon.com/) 和 [Azure](https://azure.microsoft.com/) 这样的平台提供按需访问高性能 GPU 和 TPU 的能力，加快训练速度，支持更大模型的实验。然而，云端训练可能很昂贵，特别是长时间使用时，数据传输也会增加成本和延迟。

本地训练提供更大的控制权和自定义能力，让你可以根据特定需求定制环境，并避免持续的云成本。对于长期项目来说，它可能更经济，而且由于数据留在本地，安全性更高。然而，本地硬件可能存在资源限制并需要维护，这可能导致大型模型训练时间更长。

## 选择优化器

优化器是一种调整神经网络权重以最小化[损失函数](https://www.ultralytics.com/glossary/loss-function)的算法，损失函数衡量模型的表现。简单来说，优化器通过调整参数来减少误差，从而帮助模型学习。选择合适的优化器直接影响模型学习的快慢和准确程度。

你还可以微调优化器参数来提高模型性能。调整学习率决定了更新参数时步长的大小。为了稳定性，你可以从适中的学习率开始，并随时间逐渐降低以改善长期学习效果。此外，设置动量决定了过去更新对当前更新的影响程度。动量的常用值约为 0.9，通常能提供良好的平衡。

### 常见优化器

不同的优化器有各自的优缺点。让我们简要了解几种常见的优化器。

- **SGD（随机梯度下降）**：
    - 使用损失函数相对于参数的梯度来更新模型参数。
    - 简单高效，但收敛速度可能较慢，并且可能陷入局部最小值。

- **[Adam](https://www.ultralytics.com/glossary/adam-optimizer)（自适应矩估计）**：
    - 结合了带动量的 SGD 和 RMSProp 的优点。
    - 基于梯度一阶矩和二阶矩的估计，为每个参数自适应调整学习率。
    - 非常适合噪声数据和稀疏梯度。
    - 高效且通常需要较少的调参，是 YOLO26 推荐的优化器。

- **RMSProp（均方根传播）**：
    - 通过将梯度除以最近梯度幅值的移动平均值，为每个参数自适应调整学习率。
    - 有助于处理梯度消失问题，对[循环神经网络](https://www.ultralytics.com/glossary/recurrent-neural-network-rnn)有效。

- **MuSGD（Muon + SGD 混合）**：
    - 结合了 SGD 风格的更新和 Muon 启发的行为，提高大规模训练的稳定性。
    - 当你希望获得类似 SGD 的泛化能力但需要比普通 SGD 更平滑的收敛时，是一个不错的选择。
    - 与 YOLO26 训练方案特别相关；如果不确定，可以从 `optimizer=auto` 开始，并在你的数据集上与 MuSGD 进行比较。

对于 YOLO26，`optimizer` 参数允许你从多种优化器中选择，包括 SGD、MuSGD、Adam、AdamW、NAdam、RAdam 和 RMSProp，也可以将其设置为 `auto` 以根据模型配置自动选择。

```bash
yolo train model=yolo26n.pt data=coco8.yaml optimizer=MuSGD
```

## 与社区建立联系

加入计算机视觉爱好者社区可以帮助你解决问题并更快学习。以下是连接、获取帮助和分享想法的一些途径。

### 社区资源

- **GitHub Issues：**访问 [YOLO26 GitHub 仓库](https://github.com/ultralytics/ultralytics/issues)，使用 Issues 选项卡提问、报告 bug 和建议新功能。社区和维护者非常活跃，随时愿意提供帮助。
- **Ultralytics Discord 服务器：**加入 [Ultralytics Discord 服务器](https://discord.com/invite/ultralytics)与其他用户和开发者交流、获取支持并分享经验。

### 官方文档

- **Ultralytics YOLO26 文档：**查阅[官方 YOLO26 文档](./index.md)获取各种计算机视觉项目的详细指南和实用技巧。

利用这些资源可以帮助你解决挑战，并跟上计算机视觉社区的最新趋势和实践。

## 关键要点

训练计算机视觉模型需要遵循良好实践、优化策略并随时解决问题。调整批量大小、混合[精度](https://www.ultralytics.com/glossary/precision)训练以及从预训练权重开始等技术可以让你的模型效果更好、训练更快。子集训练和早停法等方法可以帮助你节省时间和资源。与社区保持联系并跟上新趋势，将帮助你不断提升模型训练技能。

## FAQ

### 如何在使用 Ultralytics YOLO 训练大型数据集时提高 GPU 利用率？

要提高 GPU 利用率，请将训练配置中的 `batch_size` 参数设置为 GPU 支持的最大值。这可以确保充分利用 GPU 的能力，减少训练时间。如果遇到内存错误，请逐步减小批量大小，直到训练顺畅运行。对于 YOLO26，在训练脚本中设置 `batch=-1` 将自动确定高效处理的最佳批量大小。更多信息请参阅[训练配置](../modes/train.md)。

### 什么是混合精度训练，如何在 YOLO26 中启用它？

混合精度训练同时使用 16 位 (FP16) 和 32 位 (FP32) 浮点类型，以平衡计算速度和精度。这种方法可以加快训练速度并减少内存使用，同时不牺牲模型[准确率](https://www.ultralytics.com/glossary/accuracy)。要在 YOLO26 中启用混合精度训练，请在训练配置中将 `amp` 参数设置为 `True`。这将激活自动混合精度 (AMP) 训练。关于此优化技术的更多细节，请参阅[训练配置](../modes/train.md)。

### 多尺度训练如何提高 YOLO26 模型性能？

多尺度训练通过在多种尺寸的图像上训练来增强模型性能，使模型能够更好地泛化到不同尺度和距离。在 YOLO26 中，你可以通过在训练配置中设置 `scale` 参数来启用多尺度训练。例如，`scale=0.5` 会在 0.5 到 1.5 之间采样缩放因子，然后填充/裁剪回 `imgsz`。这种技术模拟不同距离下的物体，使模型在各种场景下更加鲁棒。有关设置和更多细节，请参阅[训练配置](../modes/train.md)。

### 如何在 YOLO26 中使用预训练权重加速训练？

使用预训练权重可以大大加速训练并提高模型准确率，因为它利用了一个已经熟悉基本视觉特征的模型。在 YOLO26 中，只需在训练配置中将 `pretrained` 参数设置为 `True`，或提供自定义预训练权重的路径即可。这种方法称为迁移学习，可以让在大规模数据集上训练过的模型有效适配你的特定应用。在[训练配置指南](../modes/train.md)中了解更多关于如何使用预训练权重及其优势的信息。

### 训练模型的推荐 epoch 数量是多少，如何在 YOLO26 中设置？

epoch 数量指的是模型训练过程中对训练数据集的完整遍历次数。一个典型的起点是 300 个 epoch。如果模型出现过早过拟合，可以减少数量。反之，如果未观察到过拟合，可以将训练延长到 600、1200 或更多个 epoch。要在 YOLO26 中设置，请使用训练脚本中的 `epochs` 参数。关于确定理想 epoch 数量的更多建议，请参阅[训练轮数](#训练的轮数)部分。