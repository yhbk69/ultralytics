---
comments: true
description: 探索 VisDrone 数据集，一个用于无人机图像和视频分析的大规模基准，包含超过 260 万个行人、车辆等物体的标注。
keywords: VisDrone, 无人机数据集, 计算机视觉, 目标检测, 目标追踪, 人群计数, 机器学习, 深度学习
---

# VisDrone 数据集

[VisDrone 数据集](https://github.com/VisDrone/VisDrone-Dataset)是由中国天津大学[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)与数据挖掘实验室的 AISKYEYE 团队创建的大规模基准数据集。它包含了精心标注的真实标注数据，用于与无人机图像和视频分析相关的各种计算机视觉任务。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/9ymyH4H1fG4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 VisDrone 数据集上训练 Ultralytics YOLO26 | 航拍检测 | 完整教程 🚀
</p>

VisDrone 由 288 个视频片段（261,908 帧）和 10,209 张静态图像组成，由各种无人机搭载的相机拍摄。数据集涵盖了广泛的方面，包括地点（中国 14 个不同城市）、环境（城市和乡村）、物体（行人、车辆、自行车等）和密度（稀疏和拥挤场景）。数据集使用不同的无人机平台在各种场景以及不同的天气和光照条件下采集。这些帧经过手动标注，包含超过 260 万个边界框，目标包括行人、汽车、自行车和三轮车。还提供了场景可见性、物体类别和遮挡等属性，以便更好地利用数据。

## 数据集结构

VisDrone 数据集分为五个主要子集，每个子集专注于一个特定任务：

1. **任务 1**：图像目标检测
2. **任务 2**：视频目标检测
3. **任务 3**：单目标追踪
4. **任务 4**：[多目标追踪](../index.md#multi-object-tracking)
5. **任务 5**：人群计数

## 应用场景

VisDrone 数据集广泛用于训练和评估无人机[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务（如目标检测、目标追踪和人群计数）中的深度学习模型。数据集中多样化的传感器数据、物体标注和属性使其成为无人机计算机视觉领域研究人员和实践者的宝贵资源。

## 数据集 YAML

YAML 文件用于定义数据集配置，包含数据集路径、类别和其他相关信息。Visdrone 数据集的 `VisDrone.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VisDrone.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VisDrone.yaml)。

!!! example "ultralytics/cfg/datasets/VisDrone.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/VisDrone.yaml"
    ```

## 使用方法

要在 VisDrone 数据集上以 640 的图像大小训练 YOLO26n 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型的[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=VisDrone.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## 示例数据与标注

VisDrone 数据集包含由无人机搭载相机拍摄的多样化图像和视频。以下是数据集中的一些数据示例及其对应标注：

![VisDrone 数据集航拍无人机图像目标检测](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/visdrone-object-detection-sample.avif)

- **任务 1**：图像[目标检测](https://www.ultralytics.com/glossary/object-detection)——该图像展示了图像目标检测的示例，物体用[边界框](https://www.ultralytics.com/glossary/bounding-box)标注。数据集提供了从不同地点、环境和密度拍摄的各种图像，以促进此任务模型的开发。

该示例展示了 VisDrone 数据集中数据的多样性和复杂性，并突显了高质量传感器数据对无人机计算机视觉任务的重要性。

## 引用与致谢

如果你在研究或开发工作中使用 VisDrone 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @ARTICLE{9573394,
          author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          title={Detection and Tracking Meet Drones Challenge},
          year={2021},
          volume={},
          number={},
          pages={1-1},
          doi={10.1109/TPAMI.2021.3119563}}
        ```

我们感谢中国天津大学机器学习与[数据挖掘](https://www.ultralytics.com/glossary/data-mining)实验室的 AISKYEYE 团队创建和维护 VisDrone 数据集，该数据集是无人机计算机视觉研究社区的宝贵资源。有关 VisDrone 数据集及其创建者的更多信息，请访问 [VisDrone 数据集 GitHub 仓库](https://github.com/VisDrone/VisDrone-Dataset)。

## 常见问题

### VisDrone 数据集是什么，它有哪些主要特性？

[VisDrone 数据集](https://github.com/VisDrone/VisDrone-Dataset)是由中国天津大学 AISKYEYE 团队创建的大规模基准数据集，专为无人机图像和视频分析相关的各种计算机视觉任务设计。主要特性包括：

- **组成**：288 个视频片段，261,908 帧和 10,209 张静态图像。
- **标注**：超过 260 万个边界框，用于行人、汽车、自行车和三轮车等物体。
- **多样性**：在 14 个城市采集，涵盖城市和乡村环境，不同的天气和光照条件。
- **任务**：分为五个主要任务——图像和视频目标检测、单目标和多目标追踪以及人群计数。

### 如何使用 Ultralytics 在 VisDrone 数据集上训练 YOLO26 模型？

要在 VisDrone 数据集上以 640 的图像大小训练 YOLO26 模型 100 个 epoch，可以按以下步骤操作：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo26n.pt")

        # 训练模型
        results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=VisDrone.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

有关其他配置选项，请参阅模型[训练](../../modes/train.md)页面。

### VisDrone 数据集的主要子集及其应用是什么？

VisDrone 数据集分为五个主要子集，每个子集针对特定的计算机视觉任务：

1. **任务 1**：图像目标检测。
2. **任务 2**：视频目标检测。
3. **任务 3**：单目标追踪。
4. **任务 4**：多目标追踪。
5. **任务 5**：人群计数。

这些子集广泛用于训练和评估无人机应用（如监控、交通监控和公共安全）中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型。

### 在哪里可以找到 Ultralytics 中 VisDrone 数据集的配置文件？

VisDrone 数据集的配置文件 `VisDrone.yaml` 可以在 Ultralytics 仓库的以下链接找到：
[VisDrone.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VisDrone.yaml)。

### 如果在研究中使用 VisDrone 数据集，应如何引用？

如果你在研究或开发工作中使用 VisDrone 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @ARTICLE{9573394,
          author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          title={Detection and Tracking Meet Drones Challenge},
          year={2021},
          volume={},
          number={},
          pages={1-1},
          doi={10.1109/TPAMI.2021.3119563}
        }
        ```
