---
comments: true
description: 探索用于目标检测和分割的 COCO 数据集。了解其结构、用法、预训练模型和关键特性。
keywords: COCO 数据集, 目标检测, 分割, 基准测试, 计算机视觉, 姿态估计, YOLO 模型, COCO 标注
---

# COCO 数据集

[COCO](https://cocodataset.org/#home)（Common Objects in Context）数据集是一个大规模的目标检测、分割和字幕数据集。它旨在鼓励对各种对象类别的研究，通常用于对[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型进行基准测试。对于从事目标检测、分割和姿态估计任务的研究人员和开发者来说，它是一个必不可少的数据集。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/uDrn9QZJ2lk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics COCO 数据集概览
</p>

## COCO 预训练模型

{% include "macros/yolo-det-perf.md" %}

## 关键特性

- COCO 包含 33 万张图像，其中 20 万张图像具有用于目标检测、分割和字幕任务的标注。
- 数据集包含 80 个对象类别，包括汽车、自行车和动物等常见对象，以及雨伞、手提包和运动器材等更具体的类别。
- 标注包括每个图像的目标边界框、分割掩码和字幕。
- COCO 提供标准化的评估指标，如目标检测的[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map)（mAP）和分割任务的平均[召回率](https://www.ultralytics.com/glossary/recall)均值（mAR），使其适合比较模型性能。

## 数据集结构

COCO 数据集分为三个子集：

1. **Train2017**：该子集包含 11.8 万张图像，用于训练目标检测、分割和字幕模型。
2. **Val2017**：该子集有 5000 张图像，用于模型训练期间的验证目的。
3. **Test2017**：该子集包含 2 万张图像，用于测试和基准测试训练后的模型。该子集的地面真实标注不公开，结果需提交到 [COCO 评估服务器](https://codalab.lisn.upsaclay.fr/competitions/7384)进行性能评估。

## 应用

COCO 数据集广泛用于训练和评估目标检测（如 [Ultralytics YOLO](../../models/yolo26.md)、[Faster R-CNN](https://arxiv.org/abs/1506.01497) 和 [SSD](https://arxiv.org/abs/1512.02325)）、[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)（如 [Mask R-CNN](https://arxiv.org/abs/1703.06870)）和关键点检测（如 [OpenPose](https://arxiv.org/abs/1812.08008)）中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型。该数据集多样化的对象类别、大量的标注图像和标准化的评估指标使其成为计算机视觉研究人员和从业者的重要资源。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含数据集路径、类别和其他相关信息。对于 COCO 数据集，`coco.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)。

!!! example "ultralytics/cfg/datasets/coco.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco.yaml"
    ```

## 使用方法

要在 COCO 数据集上训练 YOLO26n 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，图像大小为 640，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## 示例图像与标注

COCO 数据集包含具有各种对象类别和复杂场景的多样化图像集。以下是数据集中图像的一些示例及其对应的标注：

![COCO 数据集拼接训练批次（目标检测）](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-coco-dataset-sample.avif)

- **拼接图像**：此图像展示了一个由拼接数据集图像组成的训练批次。拼接是训练中使用的一种技术，将多张图像合并为单张图像，以增加每个训练批次中对象和场景的多样性。这有助于提高模型泛化到不同对象大小、宽高比和上下文的能力。

该示例展示了 COCO 数据集中图像的多样性和复杂性，以及在训练过程中使用拼接技术的好处。

## 引用与致谢

如果您在研究或开发工作中使用 COCO 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lin2015microsoft,
              title={Microsoft COCO: Common Objects in Context},
              author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Dollár},
              year={2015},
              eprint={1405.0312},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

我们要感谢 COCO 联盟创建并维护了这一对计算机视觉社区有价值的资源。有关 COCO 数据集及其创建者的更多信息，请访问 [COCO 数据集网站](https://cocodataset.org/#home)。

## 常见问题

### COCO 数据集是什么？为什么它对计算机视觉很重要？

[COCO 数据集](https://cocodataset.org/#home)（Common Objects in Context）是一个用于[目标检测](https://www.ultralytics.com/glossary/object-detection)、分割和字幕的大规模数据集。它包含 33 万张图像，涵盖 80 个对象类别的详细标注，使其对于计算机视觉模型的基准测试和训练至关重要。研究人员使用 COCO 是因为其多样化的类别和标准化的评估指标，如平均[精度](https://www.ultralytics.com/glossary/precision)均值（mAP）。

### 如何使用 COCO 数据集训练 YOLO 模型？

要使用 COCO 数据集训练 YOLO26 模型，可以使用以下代码片段：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

有关可用参数的更多详细信息，请参阅[训练页面](../../modes/train.md)。

### COCO 数据集的关键特性是什么？

COCO 数据集包括：

- 33 万张图像，其中 20 万张具有用于目标检测、分割和字幕的标注。
- 80 个对象类别，从汽车和动物等常见物品到手提包和运动器材等具体物品。
- 目标检测的标准化评估指标（mAP）和分割的评估指标（平均召回率均值，mAR）。
- 训练批次中的**拼接**技术，以增强模型在各种对象大小和上下文中的泛化能力。

### 在哪里可以找到在 COCO 数据集上训练的 YOLO26 预训练模型？

在 COCO 数据集上训练的 YOLO26 预训练模型可以从文档中提供的链接下载。示例包括：

- [YOLO26n](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt)
- [YOLO26s](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt)
- [YOLO26m](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt)
- [YOLO26l](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt)
- [YOLO26x](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt)

这些模型在大小、mAP 和推理速度方面各不相同，为不同的性能和资源需求提供了选择。

### COCO 数据集是如何结构的？如何使用它？

COCO 数据集分为三个子集：

1. **Train2017**：11.8 万张图像用于训练。
2. **Val2017**：5000 张图像用于训练期间的验证。
3. **Test2017**：2 万张图像用于基准测试训练后的模型。结果需提交到 [COCO 评估服务器](https://codalab.lisn.upsaclay.fr/competitions/7384)进行性能评估。

数据集的 YAML 配置文件可在 [coco.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 获取，该文件定义了路径、类别和数据集详细信息。