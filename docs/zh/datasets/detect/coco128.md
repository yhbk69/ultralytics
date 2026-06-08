---
comments: true
description: 探索 Ultralytics COCO128 数据集，一个由 128 张图像组成的通用且易于管理的数据集，非常适合测试目标检测模型和训练流程。
keywords: COCO128, Ultralytics, 数据集, 目标检测, YOLO26, 训练, 验证, 机器学习, 计算机视觉
---

# COCO128 数据集

## 简介

[Ultralytics](https://www.ultralytics.com/) COCO128 是一个小型但通用的[目标检测](https://www.ultralytics.com/glossary/object-detection)数据集，由 COCO train 2017 集的前 128 张图像组成。该数据集非常适合测试和调试目标检测模型，或用于尝试新的检测方法。128 张图像足够小，易于管理，但又足够多样化，可以测试训练流程中的错误，并在训练更大的数据集之前作为健全性检查。

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

该数据集旨在与 [Ultralytics Platform](https://platform.ultralytics.com/) 和 [YOLO26](https://github.com/ultralytics/ultralytics) 一起使用。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含数据集路径、类别和其他相关信息。对于 COCO128 数据集，`coco128.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml)。

!!! example "ultralytics/cfg/datasets/coco128.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco128.yaml"
    ```

## 使用方法

要在 COCO128 数据集上训练 YOLO26n 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，图像大小为 640，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco128.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco128.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## 示例图像与标注

以下是 COCO128 数据集中图像的一些示例及其对应的标注：

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch-1.avif" alt="COCO128 目标检测数据集拼接训练批次" width="800">

- **拼接图像**：此图像展示了一个由拼接数据集图像组成的训练批次。拼接是训练中使用的一种技术，将多张图像合并为单张图像，以增加每个训练批次中对象和场景的多样性。这有助于提高模型泛化到不同对象大小、宽高比和上下文的能力。

该示例展示了 COCO128 数据集中图像的多样性和复杂性，以及在训练过程中使用拼接技术的好处。

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

我们要感谢 COCO 联盟创建并维护了这一对[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)社区有价值的资源。有关 COCO 数据集及其创建者的更多信息，请访问 [COCO 数据集网站](https://cocodataset.org/#home)。

## 常见问题

### Ultralytics COCO128 数据集有什么用途？

Ultralytics COCO128 数据集是一个紧凑的子集，包含 COCO train 2017 数据集的前 128 张图像。它主要用于测试和调试[目标检测](https://www.ultralytics.com/glossary/object-detection)模型、尝试新的检测方法，以及在扩展到更大的数据集之前验证训练流程。其可管理的大小使其非常适合快速迭代，同时仍提供足够的多样性以成为有意义的测试用例。

### 如何使用 COCO128 数据集训练 YOLO26 模型？

要在 COCO128 数据集上训练 YOLO26 模型，可以使用 Python 或 CLI 命令。方法如下：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n.pt")

        # Train the model
        results = model.train(data="coco128.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        yolo detect train data=coco128.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

有关更多训练选项和参数，请参阅[训练](../../modes/train.md)文档。

### 在 COCO128 中使用拼接增强有什么好处？

如示例图像所示，拼接增强将多张训练图像合并为单张复合图像。这种技术在 COCO128 训练中提供了几个好处：

- 增加每个训练批次中对象和上下文的多样性
- 提高模型在不同对象大小和宽高比下的泛化能力
- 增强模型对各种尺度对象的检测性能
- 通过创建更多样化的训练样本来最大化小数据集的效用

这种技术对 COCO128 等较小的数据集尤其有价值，有助于模型从有限的数据中学习更鲁棒的特征。

### COCO128 与其他 COCO 数据集变体相比如何？

COCO128（128 张图像）在大小上介于 [COCO8](../detect/coco8.md)（8 张图像）和完整的 [COCO](../detect/coco.md) 数据集（11.8 万+ 张图像）之间：

- **COCO8**：仅包含 8 张图像（4 张训练，4 张验证）— 适合快速测试和调试
- **COCO128**：包含 128 张图像 — 在大小和多样性之间取得平衡
- **完整 COCO**：包含 11.8 万+ 张训练图像 — 全面但资源密集

COCO128 提供了一个良好的中间地带，比 COCO8 提供更多样性，同时比完整的 COCO 数据集更容易管理，适合实验和初始模型开发。

### COCO128 可以用于目标检测以外的任务吗？

虽然 COCO128 主要为目标检测设计，但数据集的标注可以适应其他计算机视觉任务：

- **实例分割**：使用标注中提供的分割掩码
- **关键点检测**：对于包含具有关键点标注的人物图像
- **迁移学习**：作为微调模型以适应自定义任务的起点

对于[分割](../../tasks/segment.md)等专门任务，考虑使用包含相应标注的专用变体，如 [COCO8-seg](../segment/coco8-seg.md)。