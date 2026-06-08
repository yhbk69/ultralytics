---
comments: true
description: 了解 PASCAL VOC 数据集，目标检测、分割和分类的核心资源。了解主要特性、应用和使用技巧。
keywords: PASCAL VOC, VOC 数据集, 目标检测, 分割, 分类, YOLO, Faster R-CNN, Mask R-CNN, 图像标注, 计算机视觉
---

# VOC 数据集

[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)（Visual Object Classes）数据集是一个著名的目标检测、分割和分类数据集。它旨在鼓励对各种物体类别的研究，通常用于计算机视觉模型的基准测试。对于从事目标检测、分割和分类任务的研究人员和开发者来说，它是一个必不可少的数据集。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yrHzL8RyY6g"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 Pascal VOC 数据集上训练 Ultralytics YOLO26 | 目标检测 🚀
</p>

## 主要特性

- VOC 数据集包含两个主要挑战：VOC2007 和 VOC2012。
- 数据集包含 20 个物体类别，包括汽车、自行车和动物等常见物体，以及船只、沙发和餐桌等更具体的类别。
- 标注包括用于目标检测和分类任务的物体边界框和类别标签，以及用于分割任务的分割掩码。
- VOC 提供了标准化的评估指标，如目标检测和分类的[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map)（mAP），适合比较模型性能。

## 数据集结构

VOC 数据集分为三个子集：

1. **训练集**：该子集包含用于训练目标检测、分割和分类模型的图像。
2. **验证集**：该子集包含在模型训练期间用于验证的图像。
3. **测试集**：该子集包含用于测试和基准测试训练模型的图像。该子集的真实标注不公开，结果历史上提交至 PASCAL VOC 评估服务器进行性能评估。

## 应用场景

VOC 数据集广泛用于训练和评估[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，包括目标检测（如 [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo26)、[Faster R-CNN](https://arxiv.org/abs/1506.01497) 和 [SSD](https://arxiv.org/abs/1512.02325)）、[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)（如 [Mask R-CNN](https://arxiv.org/abs/1703.06870)）以及[图像分类](https://www.ultralytics.com/glossary/image-classification)。数据集中多样化的物体类别、大量的标注图像和标准化的评估指标使其成为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)研究人员和实践者的重要资源。

## 数据集 YAML

YAML 文件用于定义数据集配置，包含数据集路径、类别和其他相关信息。VOC 数据集的 `VOC.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VOC.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VOC.yaml)。

!!! example "ultralytics/cfg/datasets/VOC.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/VOC.yaml"
    ```

## 使用方法

要在 VOC 数据集上以 640 的图像大小训练 YOLO26n 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型的[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="VOC.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=VOC.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## 示例图像与标注

VOC 数据集包含多样化的图像集合，涵盖各种物体类别和复杂场景。以下是数据集中的一些图像示例及其对应标注：

![Pascal VOC 数据集马赛克训练批次](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-voc-dataset-sample.avif)

- **马赛克图像**：该图像展示了一个由马赛克数据集图像组成的训练批次。马赛克是训练中使用的一种技术，将多张图像合并为一张，以增加每个训练批次中物体和场景的多样性。这有助于提高模型在不同物体大小、宽高比和上下文中的泛化能力。

该示例展示了 VOC 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克技术的好处。

## 引用与致谢

如果你在研究或开发工作中使用 VOC 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{everingham2010pascal,
              title={The PASCAL Visual Object Classes (VOC) Challenge},
              author={Mark Everingham and Luc Van Gool and Christopher K. I. Williams and John Winn and Andrew Zisserman},
              year={2010},
              eprint={0909.5206},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

我们感谢 PASCAL VOC 联盟创建和维护这一对[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)社区有价值的资源。有关 VOC 数据集及其创建者的更多信息，请访问 [PASCAL VOC 数据集网站](http://host.robots.ox.ac.uk/pascal/VOC/)。

## 常见问题

### PASCAL VOC 数据集是什么，为什么它对计算机视觉任务很重要？

[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)（Visual Object Classes）数据集是计算机视觉中[目标检测](https://www.ultralytics.com/glossary/object-detection)、分割和分类的著名基准。它包含 20 个不同物体类别的全面标注，如边界框、类别标签和分割掩码。研究人员广泛使用它来评估 Faster R-CNN、YOLO 和 Mask R-CNN 等模型的性能，因为它提供了平均精度均值（mAP）等标准化评估指标。

### 如何使用 VOC 数据集训练 YOLO26 模型？

要使用 VOC 数据集训练 YOLO26 模型，需要在 YAML 文件中配置数据集。以下是训练 YOLO26n 模型 100 个 epoch、图像大小为 640 的示例：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="VOC.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=VOC.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

### VOC 数据集包含哪些主要挑战？

VOC 数据集包含两个主要挑战：VOC2007 和 VOC2012。这些挑战测试 20 个不同物体类别的目标检测、分割和分类。每张图像都经过精心标注，包含边界框、类别标签和分割掩码。这些挑战提供了 mAP 等标准化指标，便于比较和基准测试不同的计算机视觉模型。

### PASCAL VOC 数据集如何增强模型基准测试和评估？

PASCAL VOC 数据集通过其详细的标注和平均[精度](https://www.ultralytics.com/glossary/precision)均值（mAP）等标准化指标来增强模型基准测试和评估。这些指标对于评估目标检测和分类模型的性能至关重要。数据集中多样化和复杂的图像确保了在各种真实世界场景中对模型的全面评估。

### 如何在 YOLO 模型中使用 VOC 数据集进行[语义分割](https://www.ultralytics.com/glossary/semantic-segmentation)？

要使用 VOC 数据集进行 YOLO 模型的语义分割任务，需要在 YAML 文件中正确配置数据集。YAML 文件定义了训练分割模型所需的路径和类别。有关详细设置，请查看 [VOC.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VOC.yaml) 中的 VOC 数据集 YAML 配置文件。对于分割任务，应使用分割专用模型如 `yolo26n-seg.pt`，而非检测模型。
