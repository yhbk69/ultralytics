---
comments: true
description: 学习如何使用 YOLO26 高效训练目标检测模型，包含设置、数据增强和硬件利用的全面说明。
keywords: Ultralytics, YOLO26, 模型训练, 深度学习, 目标检测, GPU 训练, 数据集增强, 超参数调优, 模型性能, Apple Silicon 训练
---

# 使用 Ultralytics YOLO 进行模型训练

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO 生态系统与集成">

## 简介

训练一个[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型涉及向其提供数据并调整其参数，使其能够做出准确的预测。Ultralytics YOLO26 中的训练模式专为高效训练目标检测模型而设计，充分利用现代硬件能力。本指南旨在涵盖您开始使用 YOLO26 强大功能集训练自己的模型所需了解的所有细节。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何在 Google Colab 中训练自定义数据集的 YOLO 模型。
</p>

## 为什么选择 Ultralytics YOLO 进行训练？

以下是选择 YOLO26 训练模式的一些令人信服的理由：

- **高效性：** 无论您使用单 GPU 设置还是跨多个 GPU 扩展，都能充分利用您的硬件。
- **多功能性：** 除了现成的数据集（如 COCO、VOC 和 ImageNet）外，还可以在自定义数据集上训练。
- **用户友好：** 简单而强大的 CLI 和 Python 接口，提供直接的训练体验。
- **超参数灵活性：** 广泛的可自定义超参数，用于微调模型性能。为了更深层次的控制，您可以[自定义训练器](../guides/custom-trainer.md)本身。
- **云端训练：** 通过 [Ultralytics Platform](https://platform.ultralytics.com) 在云端 GPU 上训练，提供实时指标和自动检查点保存。

### 训练模式的主要特性

以下是 YOLO26 训练模式的一些显著特性：

- **自动数据集下载：** 标准数据集（如 COCO、VOC 和 ImageNet）在首次使用时自动下载。
- **多 GPU 支持：** 无缝跨多个 GPU 扩展训练工作，以加速过程。
- **超参数配置：** 通过 YAML 配置文件或 CLI 参数修改超参数的选项。
- **可视化与监控：** 实时跟踪训练指标并可视化学习过程，以获得更好的洞察。

!!! tip

    * YOLO26 数据集如 COCO、VOC、ImageNet 等在首次使用时自动下载，例如 `yolo train data=coco.yaml`

## 使用示例

在 COCO8 数据集上训练 YOLO26n 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，图像大小为 640。训练设备可以使用 `device` 参数指定。如果未传递参数，当 GPU 可用时将使用 `device=0`；否则将使用 `device='cpu'`。有关训练参数的完整列表，请参阅下面的参数部分。

!!! warning "Windows 多进程错误"

    在 Windows 上，当以脚本形式启动训练时，您可能会收到 `RuntimeError`。在训练代码前添加 `if __name__ == "__main__":` 块来解决此问题。

!!! example "单 GPU 和 CPU 训练示例"

    设备自动确定。如果有可用的 GPU，将使用 GPU（默认 CUDA 设备 0）；否则训练将在 CPU 上开始。

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.yaml")  # 从 YAML 构建新模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）
        model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # 从 YAML 构建并转移权重

        # 训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从 YAML 构建新模型并从零开始训练
        yolo detect train data=coco8.yaml model=yolo26n.yaml epochs=100 imgsz=640

        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640

        # 从 YAML 构建新模型，将预训练权重转移到它并开始训练
        yolo detect train data=coco8.yaml model=yolo26n.yaml pretrained=yolo26n.pt epochs=100 imgsz=640
        ```

### 多 GPU 训练

多 GPU 训练通过将训练负载分布在多个 GPU 上，可以更高效地利用可用硬件资源。此功能可通过 Python API 和命令行界面使用。要启用多 GPU 训练，请指定您希望使用的 GPU 设备 ID。

!!! example "多 GPU 训练示例"

    要使用 2 个 GPU（CUDA 设备 0 和 1）进行训练，请使用以下命令。根据需要扩展到更多 GPU。

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 使用 2 个 GPU 训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[0, 1])

        # 使用两个最空闲的 GPU 训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[-1, -1])
        ```

    === "CLI"

        ```bash
        # 使用 GPU 0 和 1 从预训练的 *.pt 模型开始训练
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640 device=0,1

        # 使用两个最空闲的 GPU
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640 device=-1,-1
        ```

!!! note "使用自定义代码进行多 GPU 训练"

    当您指定多个设备（例如 `device=[0, 1]`）时，Ultralytics 会在内部生成一个新的训练器实例并执行 `torch.distributed.run`。这对于标准的 CLI 使用和未修改的 Python 脚本可以无缝工作。

    但是，如果您的脚本包含自定义组件（例如自定义训练器、验证器、数据集或增强管道），这些对象无法自动序列化并传输到 DDP 子进程。在这种情况下，您必须直接使用 `torch.distributed.run` 启动脚本：

    ```bash
    python -m torch.distributed.run --nproc_per_node 2 your_training_script.py
    ```

### 空闲 GPU 训练

空闲 GPU 训练能够自动选择多 GPU 系统中利用率最低的 GPU，无需手动选择 GPU 即可优化资源使用。此功能根据利用率指标和 VRAM 可用性来识别可用 GPU。

!!! example "空闲 GPU 训练示例"

    要自动选择并使用最空闲的 GPU 进行训练，请使用 `-1` 设备参数。这在共享计算环境或具有多个用户的服务器中特别有用。

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 使用单个最空闲的 GPU 进行训练
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=-1)

        # 使用两个最空闲的 GPU 进行训练
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[-1, -1])
        ```

    === "CLI"

        ```bash
        # 使用单个最空闲的 GPU 开始训练
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640 device=-1

        # 使用两个最空闲的 GPU 开始训练
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640 device=-1,-1
        ```

自动选择算法优先考虑以下 GPU：

1. 当前利用率百分比较低
2. 可用内存较高（空闲 VRAM）
3. 温度和功耗较低

此功能在共享计算环境或跨不同模型运行多个训练作业时特别有价值。它会自动适应变化的系统条件，无需手动干预即可确保最佳资源分配。

### Apple Silicon MPS 训练

随着 Ultralytics YOLO 模型集成了对 Apple Silicon 芯片的支持，现在可以在使用强大的 Metal Performance Shaders (MPS) 框架的设备上训练模型。MPS 提供了一种在 Apple 定制芯片上执行计算和图像处理任务的高性能方式。

要在 Apple Silicon 芯片上启用训练，您应在启动训练过程时将 'mps' 指定为您的设备。以下是如何在 Python 和命令行中执行此操作的示例：

!!! example "MPS 训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 使用 MPS 训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="mps")
        ```

    === "CLI"

        ```bash
        # 使用 MPS 从预训练的 *.pt 模型开始训练
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640 device=mps
        ```

在利用 Apple Silicon 芯片的计算能力的同时，这可以实现更高效的训练任务处理。有关更详细的指导和高级配置选项，请参阅 [PyTorch MPS 文档](https://docs.pytorch.org/docs/stable/notes/mps.html)。

### 恢复中断的训练

从先前保存的状态恢复训练是使用深度学习模型时的一个关键功能。这在各种场景中都非常有用，例如训练过程意外中断时，或者当您希望使用新数据继续训练模型或进行更多轮次训练时。

恢复训练时，Ultralytics YOLO 会加载最后保存的模型的权重，并恢复优化器状态、[学习率](https://www.ultralytics.com/glossary/learning-rate)调度器和轮次数。这使您可以从上次中断的地方无缝继续训练过程。

您可以通过在调用 `train` 方法时将 `resume` 参数设置为 `True`，并指定包含部分训练模型权重的 `.pt` 文件路径，轻松地在 Ultralytics YOLO 中恢复训练。

以下是使用 Python 和命令行恢复中断训练的示例：

!!! example "恢复训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/last.pt")  # 加载部分训练的模型

        # 恢复训练
        results = model.train(resume=True)
        ```

    === "CLI"

        ```bash
        # 恢复中断的训练
        yolo train resume model=path/to/last.pt
        ```

通过设置 `resume=True`，`train` 函数将从上次中断的地方继续训练，使用存储在 'path/to/last.pt' 文件中的状态。如果省略 `resume` 参数或将其设置为 `False`，`train` 函数将开始新的训练会话。

请记住，检查点默认在每个轮次结束时保存，或使用 `save_period` 参数按固定间隔保存，因此您必须至少完成 1 个轮次才能恢复训练运行。

## 训练设置

YOLO 模型的训练设置包括训练过程中使用的各种超参数和配置。这些设置影响模型的性能、速度和[准确度](https://www.ultralytics.com/glossary/accuracy)。关键的训练设置包括批大小、学习率、动量和权重衰减。此外，优化器、[损失函数](https://www.ultralytics.com/glossary/loss-function)和训练数据集组成的选择也会影响训练过程。仔细调整和实验这些设置对于优化性能至关重要。

### MuSGD 优化器

在 YOLO26 中，**MuSGD** 是一种混合优化器，结合了标准的 **SGD** 更新和 **Muon 风格的正交化更新**。

**推荐用于较长的 YOLO26 训练运行和较大的数据集**，其中正交化的 Muon 更新有助于稳定优化。

只有 `param.ndim >= 2` 的参数（例如卷积权重）会与 SGD 一起接收 Muon 风格更新，而较低维度的参数（如批量归一化层和偏置项）则保持标准 SGD。

当使用 `optimizer=auto` 时，Ultralytics 会自动为较长的训练运行（通常当迭代次数 > 10000 时）选择 **MuSGD**。对于较短的运行，训练器会回退到 **AdamW**。

使用示例：

```bash
yolo train model=yolo26n.pt data=coco8.yaml optimizer=MuSGD
```

有关实现，请参阅 `ultralytics/optim/muon.py` 和 `BaseTrainer.build_optimizer` 中的优化器自动选择逻辑。

{% include "macros/train-args.md" %}

!!! info "关于批大小设置的说明"

    `batch` 参数可以通过三种方式配置：

    - **固定[批大小](https://www.ultralytics.com/glossary/batch-size)：** 设置整数值（例如 `batch=16`），直接指定每批图像的数量。
    - **自动模式（60% GPU 内存）：** 使用 `batch=-1` 自动调整批大小，以达到约 60% 的 CUDA 内存利用率。
    - **带利用率分数的自动模式：** 设置分数值（例如 `batch=0.70`），基于指定的 GPU 内存使用分数调整批大小。
    - **OOM 自动重试：** 如果在第一个轮次期间发生 CUDA 内存不足错误，训练器会自动将批大小减半并重试（最多 3 次）。这仅适用于单 GPU 训练；多 GPU (DDP) 训练将立即引发错误。

## 增强设置和超参数

增强技术对于提高 YOLO 模型的鲁棒性和性能至关重要，它通过向[训练数据](https://www.ultralytics.com/glossary/training-data)引入可变性，帮助模型更好地泛化到未见过的数据。下表概述了每个增强参数的目的和效果：

{% include "macros/augmentation-args.md" %}

这些设置可以根据数据集和手头任务的具体要求进行调整。尝试不同的值有助于找到导致最佳模型性能的最佳增强策略。

!!! info

    有关训练增强操作的更多信息，请参阅[参考部分](../reference/data/augment.md)。

## 日志记录

在训练 YOLO26 模型时，您可能会发现跟踪模型随时间变化的性能非常有价值。这就是日志记录发挥作用的地方。Ultralytics YOLO 支持三种类型的日志记录器 - [Comet](../integrations/comet.md)、[ClearML](../integrations/clearml.md) 和 [TensorBoard](../integrations/tensorboard.md)。

要使用日志记录器，请从上面的代码片段下拉菜单中选择一个并运行它。选择的日志记录器将被安装和初始化。

### Comet

[Comet](../integrations/comet.md) 是一个平台，允许数据科学家和开发人员跟踪、比较、解释和优化实验和模型。它提供实时指标、代码差异和超参数跟踪等功能。

要使用 Comet：

!!! example

    === "Python"

        ```python
        # pip install comet_ml
        import comet_ml

        comet_ml.init()
        ```

请记住在 Comet 网站上登录您的 Comet 帐户并获取您的 API 密钥。您需要将其添加到环境变量或脚本中以记录您的实验。

### ClearML

[ClearML](https://clear.ml/) 是一个开源平台，可自动化实验跟踪并帮助高效共享资源。它旨在帮助团队更高效地管理、执行和复现其 ML 工作。

要使用 ClearML：

!!! example

    === "Python"

        ```python
        # pip install clearml
        import clearml

        clearml.browser_login()
        ```

运行此脚本后，您需要在浏览器中登录您的 ClearML 帐户并验证您的会话。

### TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) 是 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 的可视化工具包。它允许您可视化 TensorFlow 图、绘制有关图执行的定量指标，并显示通过它的其他数据（如图像）。

要在 [Google Colab](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb) 中使用 TensorBoard：

!!! example

    === "CLI"

        ```bash
        load_ext tensorboard
        tensorboard --logdir ultralytics/runs # 替换为 'runs' 目录
        ```

要在本地使用 TensorBoard，请运行以下命令并在 `localhost:6006` 查看结果。

!!! example

    === "CLI"

        ```bash
        tensorboard --logdir ultralytics/runs # 替换为 'runs' 目录
        ```

这将加载 TensorBoard 并将其定向到保存训练日志的目录。

设置好日志记录器后，您可以继续模型训练。所有训练指标将自动记录在您选择的平台中，您可以访问这些日志来监控模型随时间变化的性能、比较不同模型并确定需要改进的领域。

## 常见问题解答

### 我可以在没有本地 GPU 的情况下训练吗？

可以。[Ultralytics Platform](https://platform.ultralytics.com) 支持云端训练，并提供免费积分供您开始。上传您的数据集，选择模型和 GPU，然后直接从浏览器进行训练。详情请参阅[云端训练指南](../platform/train/cloud-training.md)。

### 如何使用 Ultralytics YOLO26 训练[目标检测](https://www.ultralytics.com/glossary/object-detection)模型？

要使用 Ultralytics YOLO26 训练目标检测模型，您可以使用 Python API 或 CLI。以下是两者的示例：

!!! example "单 GPU 和 CPU 训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

更多细节，请参阅[训练设置](#train-settings)部分。

### Ultralytics YOLO26 训练模式的主要特性是什么？

Ultralytics YOLO26 训练模式的主要特性包括：

- **自动数据集下载：** 自动下载标准数据集，如 COCO、VOC 和 ImageNet。
- **多 GPU 支持：** 跨多个 GPU 扩展训练以加快处理速度。
- **超参数配置：** 通过 YAML 文件或 CLI 参数自定义超参数。
- **可视化与监控：** 实时跟踪训练指标以获得更好的洞察。

这些特性使训练高效且可根据您的需求自定义。更多细节，请参阅[训练模式的主要特性](#key-features-of-train-mode)部分。

### 如何在 Ultralytics YOLO26 中恢复中断会话的训练？

要恢复中断会话的训练，请将 `resume` 参数设置为 `True`，并指定最后保存的检查点路径。

!!! example "恢复训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载部分训练的模型
        model = YOLO("path/to/last.pt")

        # 恢复训练
        results = model.train(resume=True)
        ```

    === "CLI"

        ```bash
        yolo train resume model=path/to/last.pt
        ```

更多信息，请查看[恢复中断的训练](#resuming-interrupted-trainings)部分。

### 如何在类别不平衡的数据集上训练模型？

类别不平衡发生在某些类别的训练样本数量明显少于其他类别时。这可能导致模型在稀有类别上表现不佳。Ultralytics YOLO 通过 `cls_pw` 参数支持类别加权来解决此问题。

`cls_pw` 参数根据逆类别频率控制类别加权强度：

- `cls_pw=0.0`（默认）：禁用类别加权
- `cls_pw=1.0`：应用完整的逆频率加权
- 介于 `0.0` 和 `1.0` 之间的值：为中度不平衡提供部分加权

类别权重计算为 `(1.0 / class_counts) ^ cls_pw` 并归一化，使其平均值等于 1.0。

!!! example "在不平衡数据集上训练"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo26n.pt")

        # 对严重不平衡数据使用完整类别加权进行训练
        results = model.train(data="custom.yaml", epochs=100, imgsz=640, cls_pw=1.0)

        # 或对中度不平衡使用部分加权 (0.25)
        results = model.train(data="custom.yaml", epochs=100, imgsz=640, cls_pw=0.25)
        ```

    === "CLI"

        ```bash
        # 使用完整逆频率加权进行训练
        yolo detect train data=custom.yaml model=yolo26n.pt epochs=100 imgsz=640 cls_pw=1.0

        # 对中度不平衡使用部分加权进行训练
        yolo detect train data=custom.yaml model=yolo26n.pt epochs=100 imgsz=640 cls_pw=0.25
        ```

!!! tip

    对于中度不平衡的数据集，从 `cls_pw=0.25` 开始，如果稀有类别仍然表现不佳，则增加到 `1.0`。您可以在训练日志中检查计算出的类别权重以验证权重分布。

### 我可以在 Apple Silicon 芯片上训练 YOLO26 模型吗？

可以，Ultralytics YOLO26 支持在 Apple Silicon 芯片上利用 Metal Performance Shaders (MPS) 框架进行训练。将 'mps' 指定为您的训练设备。

!!! example "MPS 训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo26n.pt")

        # 在 Apple Silicon 芯片（M1/M2/M3/M4）上训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="mps")
        ```

    === "CLI"

        ```bash
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640 device=mps
        ```

更多细节，请参阅[Apple Silicon MPS 训练](#apple-silicon-mps-training)部分。

### 常见的训练设置有哪些，如何配置它们？

Ultralytics YOLO26 允许您通过参数配置各种训练设置，如批大小、学习率、轮次数等。以下是一个简要概述：

| 参数 | 默认值 | 描述 |
| -------- | ------- | ---------------------------------------------------------------------- |
| `model` | `None` | 用于训练的模型文件路径。 |
| `data` | `None` | 数据集配置文件路径（例如 `coco8.yaml`）。 |
| `epochs` | `100` | 训练总轮次数。 |
| `batch` | `16` | 批大小，可调整为整数或自动模式。 |
| `imgsz` | `640` | 训练的目标图像大小。 |
| `device` | `None` | 训练计算设备，如 `cpu`、`0`、`0,1` 或 `mps`。 |
| `save` | `True` | 启用训练检查点和最终模型权重的保存。 |

有关训练设置的深入指南，请查看[训练设置](#train-settings)部分。