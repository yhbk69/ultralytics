---
comments: true
description: 了解如何为 YOLO 分类任务构建数据集。详细的文件夹结构和使用示例，助力高效训练。
keywords: YOLO, 图像分类, 数据集结构, CIFAR-10, Ultralytics, 机器学习, 训练数据, 模型评估
---

# 图像分类数据集概览

## YOLO 分类任务的数据集结构

对于 [Ultralytics](https://www.ultralytics.com/) YOLO 分类任务，数据集必须在 `root` 目录下按照特定的分割目录结构组织，以便于进行正确的训练、测试和可选的验证过程。该结构包括用于训练（`train`）和测试（`test`）阶段的独立目录，以及一个可选的验证（`val`）目录。

每个目录应包含数据集中每个类别的一个子目录。子目录以相应的类别命名，并包含该类别下的所有图像。确保每个图像文件命名唯一，并以常见格式（如 JPEG 或 PNG）存储。

### 文件夹结构示例

以 [CIFAR-10](cifar10.md) 数据集为例，文件夹结构应如下所示：

```
cifar-10-/
|
|-- train/
|   |-- airplane/
|   |   |-- 10008_airplane.png
|   |   |-- 10009_airplane.png
|   |   |-- ...
|   |
|   |-- automobile/
|   |   |-- 1000_automobile.png
|   |   |-- 1001_automobile.png
|   |   |-- ...
|   |
|   |-- bird/
|   |   |-- 10014_bird.png
|   |   |-- 10015_bird.png
|   |   |-- ...
|   |
|   |-- ...
|
|-- test/
|   |-- airplane/
|   |   |-- 10_airplane.png
|   |   |-- 11_airplane.png
|   |   |-- ...
|   |
|   |-- automobile/
|   |   |-- 100_automobile.png
|   |   |-- 101_automobile.png
|   |   |-- ...
|   |
|   |-- bird/
|   |   |-- 1000_bird.png
|   |   |-- 1001_bird.png
|   |   |-- ...
|   |
|   |-- ...
|
|-- val/ (可选)
|   |-- airplane/
|   |   |-- 105_airplane.png
|   |   |-- 106_airplane.png
|   |   |-- ...
|   |
|   |-- automobile/
|   |   |-- 102_automobile.png
|   |   |-- 103_automobile.png
|   |   |-- ...
|   |
|   |-- bird/
|   |   |-- 1045_bird.png
|   |   |-- 1046_bird.png
|   |   |-- ...
|   |
|   |-- ...
```

这种结构化的方法确保模型能够在训练阶段从组织良好的类别中有效学习，并在测试和验证阶段准确评估性能。

## 使用方法

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="path/to/dataset", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=path/to/data model=yolo26n-cls.pt epochs=100 imgsz=640
        ```

!!! tip

    大多数内置数据集名称（例如 `cifar10`、`imagenette` 或 `mnist160`）将在首次引用时自动下载并缓存数据。仅当您整理了自定义数据集时，才将 `data` 指向文件夹路径。

## 支持的数据集

Ultralytics 支持以下可自动下载的数据集：

- [Caltech 101](caltech101.md)：一个包含 101 个对象类别图像的数据集，用于[图像分类](https://www.ultralytics.com/glossary/image-classification)任务。
- [Caltech 256](caltech256.md)：Caltech 101 的扩展版本，包含 256 个对象类别和更具挑战性的图像。
- [CIFAR-10](cifar10.md)：一个包含 6 万张 32x32 彩色图像的数据集，分为 10 个类别，每类 6000 张图像。
- [CIFAR-100](cifar100.md)：CIFAR-10 的扩展版本，包含 100 个对象类别，每类 600 张图像。
- [Fashion-MNIST](fashion-mnist.md)：一个包含 70000 张灰度图像的数据集，涵盖 10 个时尚类别，用于图像分类任务。
- [ImageNet](imagenet.md)：一个用于[目标检测](https://www.ultralytics.com/glossary/object-detection)和图像分类的大规模数据集，包含超过 1400 万张图像和 20000 个类别。
- [ImageNet-10](imagenet10.md)：ImageNet 的一个较小子集，包含 10 个类别，用于更快速的实验和测试。
- [Imagenette](imagenette.md)：ImageNet 的一个较小子集，包含 10 个易于区分的类别，用于更快速的训练和测试。
- [Imagewoof](imagewoof.md)：ImageNet 的一个更具挑战性的子集，包含 10 个犬种类别，用于图像分类任务。
- [MNIST](mnist.md)：一个包含 70000 张手写数字灰度图像的数据集，用于图像分类任务。
- [MNIST160](mnist.md)：MNIST 数据集中每个类别的前 8 张图像，总共包含 160 张图像。

### 添加您自己的数据集

如果您有自己的数据集，并希望将其用于 Ultralytics YOLO 训练分类模型，请确保其遵循上述"数据集结构"中指定的格式，然后在初始化训练脚本时将 `data` 参数指向数据集目录。

## 常见问题

### 如何为 YOLO 分类任务构建数据集？

要为 Ultralytics YOLO 分类任务构建数据集，您应遵循特定的分割目录格式。将数据集组织为 `train`、`test` 和可选的 `val` 独立目录。每个目录应包含以每个类别命名的子目录，其中存放相应的图像。这有助于顺利进行训练和评估过程。以 [CIFAR-10](cifar10.md) 数据集格式为例：

```
cifar-10-/
|-- train/
|   |-- airplane/
|   |-- automobile/
|   |-- bird/
|   ...
|-- test/
|   |-- airplane/
|   |-- automobile/
|   |-- bird/
|   ...
|-- val/ (可选)
|   |-- airplane/
|   |-- automobile/
|   |-- bird/
|   ...
```

更多详情，请参阅 [YOLO 分类任务的数据集结构](#yolo-分类任务的数据集结构)部分。

### Ultralytics YOLO 支持哪些图像分类数据集？

Ultralytics YOLO 支持自动下载多个图像分类数据集，包括 [Caltech 101](caltech101.md)、[Caltech 256](caltech256.md)、[CIFAR-10](cifar10.md)、[CIFAR-100](cifar100.md)、[Fashion-MNIST](fashion-mnist.md)、[ImageNet](imagenet.md)、[ImageNet-10](imagenet10.md)、[Imagenette](imagenette.md)、[Imagewoof](imagewoof.md) 和 [MNIST](mnist.md)。这些数据集的结构使其易于与 YOLO 配合使用。每个数据集的页面提供了有关其结构和应用的更多详细信息。

### 如何添加自己的数据集用于 YOLO 图像分类？

要使用自己的数据集与 Ultralytics YOLO 配合，请确保其遵循分类任务所需的指定目录格式，即包含独立的 `train`、`test` 和可选的 `val` 目录，以及每个类别包含相应图像的子目录。数据集结构正确后，在初始化训练脚本时将 `data` 参数指向数据集的根目录。以下是 Python 示例：

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="path/to/your/dataset", epochs=100, imgsz=640)
```

更多详情请参阅[添加您自己的数据集](#添加您自己的数据集)部分。

### 为什么应该使用 Ultralytics YOLO 进行图像分类？

Ultralytics YOLO 为图像分类提供了多项优势，包括：

- **预训练模型**：加载如 `yolo26n-cls.pt` 的预训练模型，快速启动训练过程。
- **易于使用**：简单的 API 和 CLI 命令用于训练和评估。
- **高性能**：最先进的[准确率](https://www.ultralytics.com/glossary/accuracy)和速度，非常适合实时应用。
- **多数据集支持**：与各种流行数据集（如 [CIFAR-10](cifar10.md)、[ImageNet](imagenet.md) 等）无缝集成。
- **社区与支持**：可访问广泛的文档和活跃的社区，用于故障排除和改进。

如需更多见解和实际应用，您可以探索 [Ultralytics YOLO](https://www.ultralytics.com/yolo)。

### 如何使用 Ultralytics YOLO 训练模型？

使用 Ultralytics YOLO 训练模型可以通过 Python 和 CLI 轻松完成。示例如下：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model

        # Train the model
        results = model.train(data="path/to/dataset", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=path/to/data model=yolo26n-cls.pt epochs=100 imgsz=640
        ```

这些示例展示了使用任一方法训练 YOLO 模型的简单过程。更多信息，请参阅[使用方法](#使用方法)部分和分类任务的[训练](https://docs.ultralytics.com/tasks/classify#train)页面。