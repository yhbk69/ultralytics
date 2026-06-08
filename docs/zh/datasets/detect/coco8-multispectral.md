---
comments: true
description: 探索 Ultralytics COCO8-Multispectral 数据集，这是一个增强版的 COCO8，具有插值光谱通道，非常适合测试多光谱目标检测模型和训练流程。
keywords: COCO8-Multispectral, Ultralytics, 数据集, 多光谱, 目标检测, YOLO26, 训练, 验证, 机器学习, 计算机视觉
---

# COCO8-Multispectral 数据集

## 简介

[Ultralytics](https://www.ultralytics.com/) COCO8-Multispectral 数据集是原始 COCO8 数据集的高级变体，旨在促进多光谱目标检测模型的实验。它由 COCO train 2017 集中相同的 8 张图像组成 —— 4 张用于训练，4 张用于验证 —— 但每张图像被转换为 10 通道多光谱格式。通过超越标准 RGB 通道，COCO8-Multispectral 使得开发和评估能够利用更丰富光谱信息的模型成为可能。

<p align="center">
  <img width="640" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/coco8-multispectral-overview.avif" alt="目标检测的多光谱成像">
</p>

COCO8-Multispectral 完全兼容 [Ultralytics Platform](https://platform.ultralytics.com/) 和 [YOLO26](../../models/yolo26.md)，确保无缝集成到您的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)工作流程中。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yw2Fo6qjJU4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在多光谱数据集上训练 Ultralytics YOLO26 | 多通道 VisionAI 🚀
</p>

## 数据集生成

COCO8-Multispectral 中的多光谱图像是通过在可见光谱范围内的 10 个均匀分布的光谱通道上对原始 RGB 图像进行插值而创建的。该过程包括：

- **波长分配**：为 RGB 通道分配标称波长 —— 红：650 nm，绿：510 nm，蓝：475 nm。
- **插值**：使用线性插值估计 450 nm 到 700 nm 之间中间波长的像素值，产生 10 个光谱通道。
- **外推**：使用 SciPy 的 `interp1d` 函数进行外推，估计原始 RGB 波长之外的值，确保完整的光谱表示。

这种方法模拟了多光谱成像过程，为模型训练和评估提供了更多样化的数据集。有关多光谱成像的进一步阅读，请参阅[多光谱成像维基百科文章](https://en.wikipedia.org/wiki/Multispectral_imaging)。

## 数据集 YAML

COCO8-Multispectral 数据集使用 YAML 文件配置，该文件定义了数据集路径、类别名称和基本元数据。您可以在 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-multispectral.yaml)中查看官方的 `coco8-multispectral.yaml` 文件。

!!! example "ultralytics/cfg/datasets/coco8-multispectral.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8-multispectral.yaml"
    ```

!!! note

    请准备 `(channel, height, width)` 顺序的 TIFF 图像，使用 `.tiff` 或 `.tif` 扩展名保存，并确保它们是 `uint8` 类型以便与 Ultralytics 一起使用：

    ```python
    import cv2
    import numpy as np

    # Create and write 10-channel TIFF
    image = np.ones((10, 640, 640), dtype=np.uint8)  # CHW-order
    cv2.imwritemulti("example.tiff", image)

    # Read TIFF
    success, frames_list = cv2.imreadmulti("example.tiff")
    image = np.stack(frames_list, axis=2)
    print(image.shape)  # (640, 640, 10)  HWC-order for training and inference
    ```

## 使用方法

要在 COCO8-Multispectral 数据集上训练 YOLO26n 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，图像大小为 640，请使用以下示例。有关完整的训练选项列表，请参阅 [YOLO 训练文档](../../modes/train.md)。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO26n model
        model = YOLO("yolo26n.pt")

        # Train the model on COCO8-Multispectral
        results = model.train(data="coco8-multispectral.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train YOLO26n on COCO8-Multispectral using the command line
        yolo detect train data=coco8-multispectral.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

有关模型选择和最佳实践的更多详细信息，请参阅 [Ultralytics YOLO 模型文档](../../models/yolo26.md) 和 [YOLO 模型训练技巧指南](https://docs.ultralytics.com/guides/model-training-tips)。

## 示例图像与标注

以下是 COCO8-Multispectral 数据集中拼接训练批次的示例：

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/coco8-multispectral-mosaic-batch.avif" alt="COCO8 多光谱数据集拼接训练批次" width="800">

- **拼接图像**：此图像展示了一个训练批次，其中通过[拼接增强](https://docs.ultralytics.com/reference/data/augment)将多张数据集图像组合在一起。拼接增强增加了每个批次中对象和场景的多样性，帮助模型更好地泛化到各种对象大小、宽高比和背景。

这种技术对 COCO8-Multispectral 等小数据集尤其有价值，因为它最大化了训练期间每张图像的效用。

## 引用与致谢

如果您在研究或开发中使用 COCO 数据集，请引用以下论文：

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

特别感谢 [COCO 联盟](https://cocodataset.org/#home) 对[计算机视觉社区](https://www.ultralytics.com/blog/a-history-of-vision-models)的持续贡献。

## 常见问题

### Ultralytics COCO8-Multispectral 数据集有什么用途？

Ultralytics COCO8-Multispectral 数据集专为[多光谱目标检测](https://www.ultralytics.com/glossary/object-detection)模型的快速测试和调试而设计。仅有 8 张图像（4 张训练，4 张验证），非常适合验证您的 [YOLO26](../../models/yolo26.md) 训练流程，并确保在扩展到更大数据集之前一切正常运行。有关更多可供实验的数据集，请访问 [Ultralytics 数据集目录](https://docs.ultralytics.com/datasets)。

### 多光谱数据如何改善目标检测？

多光谱数据提供了标准 RGB 之外的额外光谱信息，使模型能够基于不同波长上反射率的细微差异来区分对象。这可以提高检测精度，特别是在具有挑战性的场景中。了解更多关于[多光谱成像](https://en.wikipedia.org/wiki/Multispectral_imaging)及其在[高级计算机视觉](https://www.ultralytics.com/blog/ai-in-aviation-a-runway-to-smarter-airports)中的应用。

### COCO8-Multispectral 是否与 Ultralytics Platform 和 YOLO 模型兼容？

是的，COCO8-Multispectral 完全兼容 [Ultralytics Platform](https://platform.ultralytics.com/) 和所有 [YOLO 模型](../../models/yolo26.md)，包括最新的 YOLO26。这使您可以轻松地将数据集集成到训练和验证工作流程中。

### 在哪里可以找到有关数据增强技术的更多信息？

要更深入地了解数据增强方法（如拼接）及其对模型性能的影响，请参阅 [YOLO 数据增强指南](https://docs.ultralytics.com/guides/yolo-data-augmentation)和 [Ultralytics 数据增强博客](https://www.ultralytics.com/blog/the-ultimate-guide-to-data-augmentation-in-2025)。

### 我可以将 COCO8-Multispectral 用于基准测试或教育目的吗？

当然可以！COCO8-Multispectral 的小尺寸和多光谱特性使其非常适合基准测试、教育演示和新模型架构的原型设计。有关更多基准测试数据集，请参见 [Ultralytics 基准数据集集合](https://docs.ultralytics.com/datasets)。