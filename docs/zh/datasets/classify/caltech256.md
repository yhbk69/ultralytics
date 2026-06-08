---
comments: true
description: 探索 Caltech-256 数据集，包含 257 个类别约 30000 张图像，非常适合训练和测试目标识别算法。
keywords: Caltech-256 数据集, 目标分类, 图像数据集, 机器学习, 计算机视觉, 深度学习, YOLO, 训练数据集
---

# Caltech-256 数据集

[Caltech-256](https://data.caltech.edu/records/nyy15-4j048) 数据集是一个用于目标分类任务的广泛图像集合。它包含约 30000 张图像，分为 257 个类别（256 个对象类别和 1 个背景类别）。图像经过精心挑选和标注，为目标识别算法提供了一个具有挑战性和多样性的基准。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Y7cfNkqSdMg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO26 通过 Caltech-256 数据集训练<a href="https://www.ultralytics.com/glossary/image-classification">图像分类</a>模型
</p>

!!! note "自动数据分割"

    提供的 Caltech-256 数据集没有预定义的训练/验证分割。但是，当您使用下面使用示例中的训练命令时，Ultralytics 框架将自动为您分割数据集。默认分割为训练集 80%，验证集 20%。

## 关键特性

- Caltech-256 数据集包含约 30000 张彩色图像，分为 257 个类别。
- 每个类别至少包含 80 张图像。
- 类别涵盖各种真实世界的对象，包括动物、车辆、家居用品和人物。
- 图像尺寸和分辨率各异。
- Caltech-256 广泛用于机器学习领域的训练和测试，特别是目标识别任务。

## 数据集结构

与 [Caltech-101](../classify/caltech101.md) 类似，Caltech-256 数据集没有训练集和测试集之间的正式分割。用户通常根据自身需求自行创建分割。常见的做法是随机选择一部分图像用于训练，其余图像用于测试。

## 应用

Caltech-256 数据集广泛用于训练和评估目标识别任务中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，例如[卷积神经网络](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn)（CNN）、[支持向量机](https://www.ultralytics.com/glossary/support-vector-machine-svm)（SVM）以及各种其他机器学习算法。其多样化的类别和高质量的图像使其成为机器学习和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域研究和开发的宝贵数据集。

## 使用方法

要在 Caltech-256 数据集上训练 YOLO 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="caltech256", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=caltech256 model=yolo26n-cls.pt epochs=100 imgsz=416
        ```

## 示例图像与标注

Caltech-256 数据集包含各种对象的高质量彩色图像，为目标识别任务提供了全面的数据集。以下是数据集中图像的一些示例（[来源](https://ml4a.github.io/demos/tsne_viewer.html)）：

![Caltech-256 图像分类数据集样本](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/caltech256-sample-image.avif)

该示例展示了 Caltech-256 数据集中对象的多样性和复杂性，强调了多样化数据集对于训练鲁棒目标识别模型的重要性。

## 引用与致谢

如果您在研究或开发工作中使用了 Caltech-256 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{griffin2007caltech,
                 title={Caltech-256 object category dataset},
                 author={Griffin, Gregory and Holub, Alex and Perona, Pietro},
                 year={2007}
        }
        ```

我们要感谢 Gregory Griffin、Alex Holub 和 Pietro Perona 创建并维护 Caltech-256 数据集，为[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和计算机视觉研究社区提供了宝贵的资源。有关 Caltech-256 数据集及其创建者的更多信息，请访问 [Caltech-256 数据集网站](https://data.caltech.edu/records/nyy15-4j048)。

## 常见问题

### Caltech-256 数据集是什么？为什么它对机器学习很重要？

[Caltech-256](https://data.caltech.edu/records/nyy15-4j048) 数据集是一个主要用于机器学习和计算机视觉中目标分类任务的大型图像数据集。它包含约 30000 张彩色图像，分为 257 个类别，涵盖广泛的真实世界对象。数据集的多样化和高质量图像使其成为评估目标识别算法的优秀基准，这对于开发鲁棒的机器学习模型至关重要。

### 如何使用 Python 或 CLI 在 Caltech-256 数据集上训练 YOLO 模型？

要在 Caltech-256 数据集上训练 YOLO 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，可以使用以下代码片段。有关其他选项，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model

        # Train the model
        results = model.train(data="caltech256", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=caltech256 model=yolo26n-cls.pt epochs=100 imgsz=416
        ```

### Caltech-256 数据集最常见的用例是什么？

Caltech-256 数据集广泛用于各种目标识别任务，例如：

- 训练卷积[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)（CNN）
- 评估支持向量机（SVM）的性能
- 基准测试新的深度学习算法
- 使用 Ultralytics YOLO 等框架开发[目标检测](https://www.ultralytics.com/glossary/object-detection)模型

其多样性和全面的标注使其成为机器学习和计算机视觉研究和开发的理想选择。

### Caltech-256 数据集的结构如何？如何分割用于训练和测试？

Caltech-256 数据集没有预定义的训练和测试分割。用户通常根据自身需求自行创建分割。常见的方法是随机选择一部分图像用于训练，其余图像用于测试。这种灵活性使用户能够根据特定项目需求和实验设置调整数据集。

### 为什么应该使用 Ultralytics YOLO 在 Caltech-256 数据集上训练模型？

Ultralytics YOLO 模型在 Caltech-256 数据集上训练具有多项优势：

- **高准确率**：YOLO 模型以其在目标检测任务中的最先进性能而闻名。
- **速度快**：提供实时推理能力，适合需要快速预测的应用。
- **易于使用**：通过 [Ultralytics Platform](https://platform.ultralytics.com)，用户无需大量编码即可训练、验证和部署模型。
- **预训练模型**：从预训练模型（如 `yolo26n-cls.pt`）开始，可以显著减少训练时间并提高模型[准确率](https://www.ultralytics.com/glossary/accuracy)。

更多详情，请探索我们的[全面训练指南](../../modes/train.md)，并了解使用 Ultralytics YOLO 进行[图像分类](../../tasks/classify.md)。