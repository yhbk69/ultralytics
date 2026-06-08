---
comments: true
description: 了解 Facebook AI Research 发布的 LVIS 数据集，这是一个包含大量多样化词汇的目标检测和实例分割基准。学习如何使用它。
keywords: LVIS 数据集, 目标检测, 实例分割, Facebook AI Research, YOLO, 计算机视觉, 模型训练, LVIS 示例
---

# LVIS 数据集

[LVIS 数据集](https://www.lvisdataset.org/) 是由 Facebook AI Research (FAIR) 开发并发布的大规模、细粒度词汇级标注数据集。它主要用作具有大量类别词汇表的[目标检测](https://www.ultralytics.com/glossary/object-detection)和[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)研究基准，旨在推动计算机视觉领域的进一步发展。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/cfTKj96TjSE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 LVIS 数据集的 YOLO World 训练工作流程
</p>

<p align="center">
    <img width="640" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/lvis-dataset-example-images.avif" alt="LVIS 大规模词汇实例分割数据集">
</p>

## 主要特性

- LVIS 包含 160k 张图像和 2M 个实例标注，用于目标检测、分割和描述任务。
- 数据集包含 1,203 个物品类别，包括汽车、自行车和动物等常见物品，以及雨伞、手提包和运动器材等更具体的类别。
- 标注包括物体边界框、分割掩码和每张图像的描述。
- LVIS 提供标准化的评估指标，如用于目标检测的[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map)（mAP）和用于分割任务的平均[召回率](https://www.ultralytics.com/glossary/recall)均值（mAR），适合比较模型性能。
- LVIS 使用与 [COCO](./coco.md) 数据集完全相同的图像，但具有不同的拆分和不同的标注。

## 数据集结构

LVIS 数据集分为四个子集：

1. **Train**：该子集包含 100k 张图像，用于训练目标检测、分割和描述模型。
2. **Val**：该子集包含 20k 张图像，用于模型训练期间的验证。
3. **Minival**：该子集与 COCO val2017 集完全相同，包含 5k 张图像，用于模型训练期间的验证。
4. **Test**：该子集包含 20k 张图像，用于测试和基准测试训练好的模型。该子集的地面真实标注不公开，结果需提交到 [LVIS 评估服务器](https://eval.ai/web/challenges/challenge-page/675/overview)进行性能评估。

## 应用场景

LVIS 数据集广泛用于训练和评估[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，用于目标检测（如 [YOLO](../../models/yolo26.md)、[Faster R-CNN](https://arxiv.org/abs/1506.01497) 和 [SSD](https://arxiv.org/abs/1512.02325)）、实例分割（如 [Mask R-CNN](https://arxiv.org/abs/1703.06870)）。数据集多样化的物品类别、大量的标注图像和标准化的评估指标使其成为计算机视觉研究人员和实践者的必备资源。

## 数据集 YAML

YAML 文件用于定义数据集配置，包含数据集路径、类别和其他相关信息。LVIS 数据集的 `lvis.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml)。

!!! example "ultralytics/cfg/datasets/lvis.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/lvis.yaml"
    ```

## 使用方法

要在 LVIS 数据集上以 640 的图像大小训练 YOLO26n 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="lvis.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=lvis.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## 示例图像与标注

LVIS 数据集包含各种物品类别和复杂场景的多样化图像。以下是数据集中图像的一些示例及其对应标注：

![LVIS 大规模词汇实例分割数据集马赛克](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/lvis-mosaiced-training-batch.avif)

- **马赛克图像**：该图像展示了一个由马赛克数据集图像组成的训练批次。马赛克是训练过程中使用的一种技术，将多张图像合并为一张图像，以增加每个训练批次中物品和场景的多样性。这有助于提高模型对不同物体大小、宽高比和上下文的泛化能力。

该示例展示了 LVIS 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克技术的好处。

## 引用与致谢

如果你在研究或开发工作中使用了 LVIS 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{gupta2019lvis,
          title={LVIS: A Dataset for Large Vocabulary Instance Segmentation},
          author={Gupta, Agrim and Dollar, Piotr and Girshick, Ross},
          booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition},
          year={2019}
        }
        ```

我们感谢 LVIS 联盟为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)社区创建和维护这一宝贵资源。有关 LVIS 数据集及其创建者的更多信息，请访问 [LVIS 数据集网站](https://www.lvisdataset.org/)。

## 常见问题

### LVIS 数据集是什么，如何在计算机视觉中使用？

[LVIS 数据集](https://www.lvisdataset.org/) 是由 Facebook AI Research (FAIR) 开发的大规模数据集，具有细粒度词汇级标注。它主要用于目标检测和实例分割，包含超过 1,203 个物品类别和 200 万个实例标注。研究人员和实践者使用它来训练和基准测试 Ultralytics YOLO 等模型，以完成高级计算机视觉任务。该数据集庞大的规模和多样性使其成为推动检测和分割领域模型性能的重要资源。

### 如何使用 LVIS 数据集训练 YOLO26n 模型？

要在 LVIS 数据集上以 640 的图像大小训练 YOLO26n 模型 100 个 epoch，请参考以下示例。该流程使用 Ultralytics 框架，该框架提供全面的训练功能。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="lvis.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=lvis.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

有关详细的训练配置，请参阅[训练](../../modes/train.md)文档。

### LVIS 数据集与 COCO 数据集有何不同？

LVIS 数据集中的图像与 [COCO 数据集](./coco.md)中的图像相同，但两者在拆分和标注方面有所不同。与 COCO 的 80 个类别相比，LVIS 提供了更大、更详细的词汇表，包含 1,203 个物品类别。此外，LVIS 侧重于标注的完整性和多样性，旨在通过提供更细致和全面的数据来推动[目标检测](https://www.ultralytics.com/glossary/object-detection)和实例分割模型的极限。

### 为什么应该使用 Ultralytics YOLO 在 LVIS 数据集上训练？

Ultralytics YOLO 模型（包括最新的 YOLO26）针对实时目标检测进行了优化，具有最先进的[准确度](https://www.ultralytics.com/glossary/accuracy)和速度。它们支持广泛的标注格式，包括 LVIS 数据集提供的细粒度标注，使其成为高级计算机视觉应用的理想选择。此外，Ultralytics 提供与各种[训练](../../modes/train.md)、[验证](../../modes/val.md)和[预测](../../modes/predict.md)模式的无缝集成，确保高效的模型开发和部署。

### 可以查看 LVIS 数据集的一些示例标注吗？

可以，LVIS 数据集包含各种物品类别和复杂场景的多样化图像。以下是示例图像及其标注：

![LVIS 大规模词汇实例分割数据集马赛克](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/lvis-mosaiced-training-batch.avif)

这张马赛克图像展示了一个由多张数据集图像合并而成的训练批次。马赛克增加了每个训练批次中物品和场景的多样性，增强了模型在不同上下文中的泛化能力。有关 LVIS 数据集的更多详情，请参阅 [LVIS 数据集文档](#key-features)。
