---
comments: true
description: 探索 Ultralytics 的多功能且易于管理的 COCO8-Seg 数据集，非常适合测试和调试分割模型或新检测方法。
keywords: COCO8-Seg, Ultralytics, 分割数据集, YOLO26, COCO 2017, 模型训练, 计算机视觉, 数据集配置
---

# COCO8-Seg 数据集

## 简介

[Ultralytics](https://www.ultralytics.com/) COCO8-Seg 是一个小型但功能多样的[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)数据集，由 COCO train 2017 集的前 8 张图像组成，其中 4 张用于训练，4 张用于验证。该数据集非常适合测试和调试分割模型，或者尝试新的检测方法。8 张图像的规模小到易于管理，但又足够多样，可以测试训练流水线是否存在错误，并在训练更大的数据集之前进行合理性检查。

## 数据集结构

- **图像**：共 8 张（4 张训练 / 4 张验证）。
- **类别**：80 个 COCO 类别。
- **标签**：YOLO 格式的多边形标注，存储在与每个图像文件对应的 `labels/{train,val}` 目录下。

该数据集旨在与 [Ultralytics 平台](https://platform.ultralytics.com/)和 [YOLO26](https://github.com/ultralytics/ultralytics) 配合使用。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置，包含数据集路径、类别及其他相关信息。COCO8-Seg 数据集的 `coco8-seg.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-seg.yaml)。

!!! example "ultralytics/cfg/datasets/coco8-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8-seg.yaml"
    ```

## 使用方法

要在 COCO8-Seg 数据集上以图像大小 640 训练 YOLO26n-seg 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，您可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-seg.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo segment train data=coco8-seg.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
        ```

## 示例图像和标注

以下是来自 COCO8-Seg 数据集的一些示例图像及其对应的标注：

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch-2.avif" alt="COCO8-seg 实例分割数据集马赛克" width="800">

- **马赛克图像**：此图像展示了由马赛克数据集图像组成的训练批次。马赛克是训练过程中使用的一种技术，将多张图像合并成一张图像，以增加每个训练批次中目标和场景的多样性。这有助于提升模型对不同目标大小、宽高比和上下文的泛化能力。

该示例展示了 COCO8-Seg 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克技术的好处。

## 引用和致谢

如果您在研究或开发工作中使用了 COCO 数据集，请引用以下论文：

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

我们向 COCO 联盟为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)社区创建和维护这一宝贵资源致以诚挚的感谢。有关 COCO 数据集及其创建者的更多信息，请访问 [COCO 数据集网站](https://cocodataset.org/#home)。

## 常见问题

### COCO8-Seg 数据集是什么，如何在 Ultralytics YOLO26 中使用它？

**COCO8-Seg 数据集**是 Ultralytics 的一个紧凑型实例分割数据集，由 COCO train 2017 集的前 8 张图像组成——4 张用于训练，4 张用于验证。该数据集专为测试和调试分割模型或尝试新检测方法而设计。它特别适合与 Ultralytics [YOLO26](https://github.com/ultralytics/ultralytics) 和[平台](https://platform.ultralytics.com/)配合使用，在扩展到更大数据集之前进行快速迭代和流水线错误检查。有关详细用法，请参阅模型[训练](../../modes/train.md)页面。

### 如何使用 COCO8-Seg 数据集训练 YOLO26n-seg 模型？

要在 COCO8-Seg 数据集上以图像大小 640 训练 **YOLO26n-seg** 模型 100 个轮次，您可以使用 Python 或 CLI 命令。以下是一个快速示例：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-seg.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo segment train data=coco8-seg.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
        ```

有关可用参数和配置选项的详细说明，请参阅[训练](../../modes/train.md)文档。

### COCO8-Seg 数据集对模型开发和调试有何重要意义？

**COCO8-Seg 数据集**提供了一个紧凑但多样化的 8 张图像集，非常适合快速测试和调试分割模型或尝试新的检测技术。其小规模允许快速进行合理性检查和早期流水线验证，有助于在扩展到更大数据集之前识别问题。在 [Ultralytics 分割数据集指南](https://docs.ultralytics.com/datasets/segment)中了解更多支持的数据集格式。

### 在哪里可以找到 COCO8-Seg 数据集的 YAML 配置文件？

**COCO8-Seg 数据集**的 YAML 配置文件在 Ultralytics 仓库中可用。您可以直接在 <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-seg.yaml> 访问该文件。YAML 文件包含模型训练和验证所需的有关数据集路径、类别和配置设置的基本信息。

### 在使用 COCO8-Seg 数据集训练时，使用马赛克技术有哪些好处？

在训练中使用**马赛克技术**有助于增加每个训练批次中目标和场景的多样性。该技术将多张图像合并成一张复合图像，增强了模型对场景中不同目标大小、宽高比和上下文的泛化能力。马赛克技术有助于提高模型的鲁棒性和[精度](https://www.ultralytics.com/glossary/accuracy)，特别是在处理像 COCO8-Seg 这样的小数据集时。有关马赛克图像的示例，请参阅[示例图像和标注](#示例图像和标注)部分。
