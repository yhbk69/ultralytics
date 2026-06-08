---
comments: true
description: 探索 COCO-Seg 数据集，COCO 的扩展版本，具有详细的分割标注。了解如何使用 COCO-Seg 训练 YOLO 模型。
keywords: COCO-Seg, 数据集, YOLO 模型, 实例分割, 目标检测, COCO 数据集, YOLO26, 计算机视觉, Ultralytics, 机器学习
---

# COCO-Seg 数据集

[COCO-Seg](https://cocodataset.org/#home) 数据集是 COCO（Common Objects in Context）数据集的扩展版本，专为辅助目标[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)研究而设计。它使用与 COCO 相同的图像，但引入了更详细的分割标注。该数据集是从事实例分割任务的研究人员和开发人员的重要资源，特别适用于训练 [Ultralytics YOLO](https://docs.ultralytics.com/models) 模型。

## COCO-Seg 预训练模型

{% include "macros/yolo-seg-perf.md" %}

## 主要特点

- COCO-Seg 保留了 COCO 的原始 33 万张图像。
- 数据集包含与原始 COCO 数据集相同的 80 个目标类别。
- 标注现在包含图像中每个目标更详细的实例分割掩码。
- COCO-Seg 提供标准化评估指标，如目标检测的[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map)（mAP）和实例分割任务的平均[召回率](https://www.ultralytics.com/glossary/recall)均值（mAR），从而能够有效比较模型性能。

## 数据集结构

COCO-Seg 数据集被划分为三个子集：

1. **Train2017**：118K 张图像，用于训练实例分割模型。
2. **Val2017**：5K 张图像，用于模型开发过程中的验证。
3. **Test2017**：20K 张图像，用于基准测试。该子集的真实标注不公开提供，因此预测结果必须提交到 [COCO 评估服务器](https://codalab.lisn.upsaclay.fr/competitions/7383) 进行评分。

## 应用场景

COCO-Seg 广泛用于训练和评估实例分割中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，例如 YOLO 模型。大量标注图像、多样化的目标类别以及标准化的评估指标使其成为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)研究人员和从业者不可或缺的资源。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置，包含数据集路径、类别及其他相关信息。COCO-Seg 数据集的 `coco.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)。

!!! example "ultralytics/cfg/datasets/coco.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco.yaml"
    ```

## 使用方法

要在 COCO-Seg 数据集上以图像大小 640 训练 YOLO26n-seg 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，您可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-seg.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo segment train data=coco.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
        ```

## 示例图像和标注

COCO-Seg 与其前身 COCO 一样，包含各种目标类别和复杂场景的多样化图像。然而，COCO-Seg 为图像中的每个目标引入了更详细的实例分割掩码。以下是来自数据集的一些示例图像及其对应的实例分割掩码：

![COCO 分割数据集马赛克训练批次](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch-3.avif)

- **马赛克图像**：此图像展示了由马赛克数据集图像组成的训练批次。[马赛克](https://docs.ultralytics.com/guides/hyperparameter-tuning)是训练过程中使用的一种技术，将多张图像合并成一张图像，以增加每个训练批次中目标和场景的多样性。这有助于模型对不同目标大小、宽高比和上下文的泛化能力。

该示例展示了 COCO-Seg 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克技术的好处。

## 引用和致谢

如果您在研究或开发工作中使用了 COCO-Seg 数据集，请引用原始 COCO 论文并致谢对 COCO-Seg 的扩展：

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

我们向 COCO 联盟为计算机视觉社区创建和维护这一宝贵资源致以诚挚的感谢。有关 COCO 数据集及其创建者的更多信息，请访问 [COCO 数据集网站](https://cocodataset.org/#home)。

## 常见问题

### COCO-Seg 数据集是什么，它与原始 COCO 数据集有何不同？

[COCO-Seg](https://cocodataset.org/#home) 数据集是原始 COCO（Common Objects in Context）数据集的扩展版本，专为实例分割任务而设计。虽然它使用与 COCO 数据集相同的图像，但 COCO-Seg 包含更详细的分割标注，使其成为专注于[目标实例分割](https://docs.ultralytics.com/tasks/segment)的研究人员和开发人员的强大资源。

### 如何使用 COCO-Seg 数据集训练 YOLO26 模型？

要在 COCO-Seg 数据集上以图像大小 640 训练 YOLO26n-seg 模型 100 个轮次，您可以使用以下代码片段。有关可用参数的详细列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-seg.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo segment train data=coco.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
        ```

### COCO-Seg 数据集有哪些主要特点？

COCO-Seg 数据集包含以下几个主要特点：

- 保留了 COCO 数据集的原始 33 万张图像。
- 标注与原始 COCO 中相同的 80 个目标类别。
- 为每个目标提供更详细的实例分割掩码。
- 使用标准化评估指标，如[目标检测](https://www.ultralytics.com/glossary/object-detection)的平均[精确率](https://www.ultralytics.com/glossary/precision)均值（mAP）和实例分割任务的平均召回率均值（mAR）。

### COCO-Seg 有哪些可用的预训练模型，其性能指标是什么？

COCO-Seg 数据集支持多种预训练 YOLO26 分割模型，具有不同的性能指标。以下是可用模型及其关键指标的摘要：

{% include "macros/yolo-seg-perf.md" %}

这些模型从轻量级的 YOLO26n-seg 到更强大的 YOLO26x-seg，在速度和精度之间提供了不同的权衡，以满足各种应用需求。有关模型选择的更多信息，请访问 [Ultralytics 模型页面](https://docs.ultralytics.com/models)。

### COCO-Seg 数据集的结构是怎样的，包含哪些子集？

COCO-Seg 数据集被划分为三个子集，用于特定的训练和评估需求：

1. **Train2017**：包含 118K 张图像，主要用于训练实例分割模型。
2. **Val2017**：包含 5K 张图像，用于训练过程中的验证。
3. **Test2017**：包含 20K 张图像，用于测试和对已训练模型进行基准测试。请注意，该子集的真实标注不公开提供，性能结果需提交到 [COCO 评估服务器](https://codalab.lisn.upsaclay.fr/competitions/7383) 进行评估。

如需较小规模的实验，您也可以考虑使用 [COCO8-seg 数据集](https://docs.ultralytics.com/datasets/segment/coco8-seg)，这是一个仅包含 COCO train 2017 集中 8 张图像的紧凑版本。
