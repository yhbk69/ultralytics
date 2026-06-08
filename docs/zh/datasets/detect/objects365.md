---
comments: true
description: 探索包含 200 万张图像和 3000 万个边界框、涵盖 365 个类别的 Objects365 数据集。使用多样化、高质量的数据增强你的目标检测模型。
keywords: Objects365 数据集, 目标检测, 机器学习, 深度学习, 计算机视觉, 标注图像, 边界框, YOLO26, 高分辨率图像, 数据集配置
---

# Objects365 数据集

[Objects365](https://www.objects365.org/) 是一个大规模、高质量的数据集，旨在推动目标检测研究，重点关注野外环境中的多样化物体。该数据集由 [Megvii](https://en.megvii.com/) 研究团队创建，提供了广泛的高分辨率图像，并配有涵盖 365 个物品类别的全面标注边界框集。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/J-RH22rwx1A"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics 在 Objects365 数据集上训练 Ultralytics YOLO26 | 200 万标注 🚀
</p>

## 主要特性

- Objects365 包含 365 个物品类别，200 万张图像和超过 3000 万个边界框。
- 数据集包含各种场景中的多样化物体，为目标检测任务提供了丰富且具有挑战性的基准。
- 标注包含物体的边界框，适合训练和评估目标检测模型。
- Objects365 预训练模型显著优于 ImageNet 预训练模型，在各种任务上具有更好的泛化能力。

## 数据集结构

Objects365 数据集由一组图像及其对应标注组成：

- **图像**：数据集包含 200 万张高分辨率图像，每张图像包含 365 个类别中的各种物体。
- **标注**：图像标注了超过 3000 万个边界框，为[目标检测](https://docs.ultralytics.com/tasks/detect)任务提供了全面的地面真实信息。

## 应用场景

Objects365 数据集广泛用于训练和评估目标检测任务中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型。数据集多样化的物品类别和高质量的标注使其成为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域研究人员和实践者的宝贵资源。

## 数据集 YAML

YAML 文件用于定义数据集配置，包含数据集路径、类别和其他相关信息。Objects365 数据集的 `Objects365.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Objects365.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Objects365.yaml)。

!!! example "ultralytics/cfg/datasets/Objects365.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/Objects365.yaml"
    ```

## 使用方法

要在 Objects365 数据集上以 640 的图像大小训练 YOLO26n 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="Objects365.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=Objects365.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## 示例数据与标注

Objects365 数据集包含来自 365 个类别的高分辨率图像多样化集，为[目标检测](https://www.ultralytics.com/glossary/object-detection)任务提供了丰富的上下文。以下是数据集中图像的一些示例：

![Objects365 数据集多样化物体标注示例](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/objects365-sample-image.avif)

- **Objects365**：该图像展示了目标检测的示例，物体以边界框标注。数据集提供了广泛的图像，以促进用于此任务的模型开发。

该示例展示了 Objects365 数据集中数据的多样性和复杂性，并突显了精准目标检测对计算机视觉应用的重要性。

## 引用与致谢

如果你在研究或开发工作中使用了 Objects365 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{shao2019objects365,
          title={Objects365: A Large-scale, High-quality Dataset for Object Detection},
          author={Shao, Shuai and Li, Zeming and Zhang, Tianyuan and Peng, Chao and Yu, Gang and Li, Jing and Zhang, Xiangyu and Sun, Jian},
          booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
          pages={8425--8434},
          year={2019}
        }
        ```

我们感谢创建并维护 Objects365 数据集的研究团队，该数据集是计算机视觉研究社区的宝贵资源。有关 Objects365 数据集及其创建者的更多信息，请访问 [Objects365 数据集网站](https://www.objects365.org/)。

## 常见问题

### Objects365 数据集用于什么？

[Objects365 数据集](https://www.objects365.org/) 专为[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和计算机视觉中的目标检测任务设计。它提供了大规模、高质量的数据集，包含 200 万张标注图像和 3000 万个边界框，涵盖 365 个类别。利用这样多样化的数据集有助于提高目标检测模型的性能和泛化能力，对该领域的研究和开发具有不可估量的价值。

### 如何在 Objects365 数据集上训练 YOLO26 模型？

要使用 Objects365 数据集训练 YOLO26n 模型 100 个 epoch，图像大小为 640，请按照以下说明操作：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="Objects365.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=Objects365.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

有关可用参数的完整列表，请参阅[训练](../../modes/train.md)页面。

### 为什么应该使用 Objects365 数据集进行目标检测项目？

Objects365 数据集为目标检测任务提供了以下几个优势：

1. **多样性**：包含 200 万张图像，涵盖 365 个类别中各种场景的物体。
2. **高质量标注**：超过 3000 万个边界框提供全面的地面真实数据。
3. **性能**：在 Objects365 上预训练的模型显著优于在 [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet) 等数据集上训练的模型，具有更好的泛化能力。

### 在哪里可以找到 Objects365 数据集的 YAML 配置文件？

Objects365 数据集的 YAML 配置文件位于 [Objects365.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Objects365.yaml)。该文件包含设置训练环境所必需的关键信息，如数据集路径和类别标签。

### Objects365 的数据集结构如何增强目标检测建模？

[Objects365 数据集](https://www.objects365.org/) 由 200 万张高分辨率图像和超过 3000 万个边界框的全面标注组成。这种结构确保了用于训练目标检测[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型的鲁棒数据集，提供了各种物体和场景的广泛种类。这种多样性和体量有助于开发更精准、能更好地泛化到真实世界应用的模型。更多数据集结构详情，请参阅[数据集 YAML](#dataset-yaml) 部分。
