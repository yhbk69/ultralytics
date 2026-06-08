---
comments: true
description: 探索 Ultralytics KITTI 数据集，这是一个用于计算机视觉任务（如 3D 目标检测、深度估计和自动驾驶感知）的基准数据集。
keywords: kitti, Ultralytics, 数据集, 目标检测, 3D 视觉, YOLO26, 训练, 验证, 自动驾驶, 计算机视觉
---

# KITTI 数据集

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-kitti-detection-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开 KITTI 数据集"></a>

KITTI 数据集是自动驾驶和计算机视觉领域最具影响力的基准数据集之一。由卡尔斯鲁厄理工学院和芝加哥丰田技术研究所发布，包含从真实驾驶场景中采集的立体相机、LiDAR 和 GPS/IMU 数据。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/NNeDlTbq9pA"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 KITTI 数据集上训练 Ultralytics YOLO26 🚀
</p>

它广泛用于评估目标检测、深度估计、光流和视觉里程计等算法。该数据集与 Ultralytics YOLO26 完全兼容，可用于 2D 目标检测任务，并可轻松集成到 Ultralytics 平台中进行训练和评估。

## 数据集结构

!!! warning

    此处不包括 KITTI 原始测试集，因为它不包含真实标注。

该数据集共包含 7,481 张图像，每张图像都配有详细的标注，涵盖汽车、行人、骑行者和其他道路元素等目标。数据集分为两个主要子集：

- **训练集：** 包含 5,985 张带标注标签的图像，用于模型训练。
- **验证集：** 包含 1,496 张带对应标注的图像，用于性能评估和基准测试。

## 应用场景

KITTI 数据集推动了自动驾驶和机器人技术的进步，支持以下任务：

- **自动驾驶车辆感知**：训练模型检测和跟踪车辆、行人和障碍物，以确保自动驾驶系统的安全导航。
- **3D 场景理解**：支持深度估计、立体视觉和 3D 目标定位，帮助机器理解空间环境。
- **光流与运动预测**：支持运动分析以预测目标移动并改善动态环境中的轨迹规划。
- **计算机视觉基准测试**：作为标准基准，用于评估目标检测和跟踪等多项视觉任务的性能。

## 数据集 YAML

Ultralytics 使用 YAML 文件定义 KITTI 数据集配置。该文件指定了数据集路径、类别标签和训练所需的元数据。配置文件位于 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml)。

!!! example "ultralytics/cfg/datasets/kitti.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/kitti.yaml"
    ```

## 使用方法

要在 KITTI 数据集上以图像大小 640 训练 YOLO26n 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，请使用以下命令。更多详情请参阅[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO26 模型
        model = YOLO("yolo26n.pt")

        # 在 kitti 数据集上训练
        results = model.train(data="kitti.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=kitti.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

您还可以使用相同的配置文件，通过命令行或 Python API 直接执行评估、[推理](../../modes/predict.md)和[导出](../../modes/export.md)任务。

## 示例图像与标注

KITTI 数据集提供了多样化的驾驶场景。每张图像都包含用于 2D 目标检测任务的边界框标注。以下示例展示了该数据集丰富的多样性，使模型能够在多种真实场景中实现稳健的泛化。

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/kitti-dataset-sample.avif" alt="KITTI 数据集车辆检测示例" width="800">

## 引用与致谢

如果您在研究中使用 KITTI 数据集，请引用以下论文：

!!! quote

    === "BibTeX"

        ```bibtex
        @article{Geiger2013IJRR,
          author = {Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun},
          title = {Vision meets Robotics: The KITTI Dataset},
          journal = {International Journal of Robotics Research (IJRR)},
          year = {2013}
        }
        ```

我们感谢 KITTI 视觉基准套件提供了这一全面的数据集，持续推动计算机视觉、机器人技术和自主系统的进步。更多信息请访问 [KITTI 网站](https://www.cvlibs.net/datasets/kitti/)。

## 常见问题

### KITTI 数据集用于什么？

KITTI 数据集主要用于自动驾驶领域的计算机视觉研究，支持目标检测、深度估计、光流和 3D 定位等任务。

### KITTI 数据集包含多少张图像？

数据集包含 5,985 张带标注的训练图像和 1,496 张验证图像，涵盖城市、乡村和高速公路场景。此处不包括原始测试集，因为它不包含真实标注。

### 数据集中标注了哪些目标类别？

KITTI 包含对汽车、行人、骑行者、卡车、有轨电车和其他道路使用者等目标的标注。

### 可以使用 KITTI 数据集训练 Ultralytics YOLO26 模型吗？

可以，KITTI 与 Ultralytics YOLO26 完全兼容。您可以直接使用提供的 YAML 配置文件进行[训练](../../modes/train.md)和[验证](../../modes/val.md)。

### 在哪里可以找到 KITTI 数据集的配置文件？

您可以在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml) 访问 YAML 文件。
