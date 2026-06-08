---
comments: true
description: 探索 Argo AI 提供的全面 Argoverse 数据集，用于自动驾驶研究中的 3D 跟踪、运动预测和立体深度估计。
keywords: Argoverse 数据集, 自动驾驶, 3D 跟踪, 运动预测, 立体深度估计, Argo AI, LiDAR 点云, 高分辨率图像, HD 地图
---

# Argoverse 数据集

[Argoverse](https://www.argoverse.org/) 数据集是一个旨在支持自动驾驶任务研究的数据集合，如 3D 跟踪、运动预测和立体深度估计。该数据集由 Argo AI 开发，提供广泛的高质量传感器数据，包括高分辨率图像、LiDAR 点云和地图数据。

!!! note

    训练所需的 Argoverse 数据集 `*.zip` 文件在 Ford 关闭 Argo AI 后已从 Amazon S3 中移除，但我们已将其提供在 [Google Drive](https://drive.google.com/file/d/1st9qW3BeIwQsnR0t8mRpvbsSWIo16ACi/view?usp=drive_link) 上供手动下载。

## 关键特性

- Argoverse 包含超过 29 万个标注的 3D 对象轨迹和 500 万个对象实例，涵盖 1263 个不同场景。
- 数据集包括高分辨率相机图像、LiDAR 点云和丰富标注的 HD 地图。
- 标注包括对象的 3D 边界框、对象轨迹和轨迹信息。
- Argoverse 为不同任务提供多个子集，如 3D 跟踪、运动预测和立体深度估计。

## 数据集结构

Argoverse 数据集分为三个主要子集：

1. **Argoverse 3D 跟踪**：该子集包含 113 个场景，超过 29 万个标注的 3D 对象轨迹，专注于 3D 对象跟踪任务。包括 LiDAR 点云、相机图像和传感器校准信息。
2. **Argoverse 运动预测**：该子集包含从 60 小时驾驶数据中收集的 32.4 万条车辆轨迹，适用于运动预测任务。
3. **Argoverse 立体深度估计**：该子集专为立体深度估计任务设计，包含超过 1 万对立体图像及对应的 LiDAR 点云，用于真实深度估计。

## 应用

Argoverse 数据集广泛用于训练和评估自动驾驶任务中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，如 3D 对象跟踪、运动预测和立体深度估计。该数据集多样化的传感器数据、对象标注和地图信息使其成为自动驾驶领域研究人员和从业者的宝贵资源。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含数据集路径、类别和其他相关信息。对于 Argoverse 数据集，`Argoverse.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Argoverse.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Argoverse.yaml)。

!!! example "ultralytics/cfg/datasets/Argoverse.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/Argoverse.yaml"
    ```

## 使用方法

要在 Argoverse 数据集上训练 YOLO26n 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，图像大小为 640，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="Argoverse.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=Argoverse.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## 样本数据与标注

Argoverse 数据集包含多样化的传感器数据，包括相机图像、LiDAR 点云和 HD 地图信息，为自动驾驶任务提供丰富的上下文。以下是数据集中数据的一些示例及其对应的标注：

![Argoverse 数据集 3D 跟踪样本（含车辆标注）](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/argoverse-3d-tracking-sample.avif)

- **Argoverse 3D 跟踪**：此图像展示了 3D 对象跟踪的示例，其中对象使用 3D 边界框进行标注。数据集提供 LiDAR 点云和相机图像，以促进此任务模型的开发。

该示例展示了 Argoverse 数据集中数据的多样性和复杂性，并强调了高质量传感器数据对自动驾驶任务的重要性。

## 引用与致谢

如果您在研究或开发工作中使用了 Argoverse 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{chang2019argoverse,
          title={Argoverse: 3D Tracking and Forecasting with Rich Maps},
          author={Chang, Ming-Fang and Lambert, John and Sangkloy, Patsorn and Singh, Jagjeet and Bak, Slawomir and Hartnett, Andrew and Wang, Dequan and Carr, Peter and Lucey, Simon and Ramanan, Deva and others},
          booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
          pages={8748--8757},
          year={2019}
        }
        ```

我们要感谢 Argo AI 创建并维护 Argoverse 数据集，为自动驾驶研究社区提供了宝贵的资源。有关 Argoverse 数据集及其创建者的更多信息，请访问 [Argoverse 数据集网站](https://www.argoverse.org/)。

## 常见问题

### Argoverse 数据集是什么？它有哪些关键特性？

[Argoverse](https://www.argoverse.org/) 数据集由 Argo AI 开发，支持自动驾驶研究。它包含超过 29 万个标注的 3D 对象轨迹和 500 万个对象实例，涵盖 1263 个不同场景。数据集提供高分辨率相机图像、LiDAR 点云和标注的 HD 地图，使其对 3D 跟踪、运动预测和立体深度估计等任务非常有价值。

### 如何使用 Argoverse 数据集训练 Ultralytics YOLO 模型？

要使用 Argoverse 数据集训练 YOLO26 模型，请使用提供的 YAML 配置文件和以下代码：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="Argoverse.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=Argoverse.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

有关参数的详细说明，请参阅模型[训练](../../modes/train.md)页面。

### Argoverse 数据集提供哪些类型的数据和标注？

Argoverse 数据集包括各种传感器数据类型，如高分辨率相机图像、LiDAR 点云和 HD 地图数据。标注包括 3D 边界框、对象轨迹和轨迹信息。这些全面的标注对于 3D 对象跟踪、运动预测和立体深度估计等任务中的精确模型训练至关重要。

### Argoverse 数据集是如何结构的？

数据集分为三个主要子集：

1. **Argoverse 3D 跟踪**：包含 113 个场景，超过 29 万个标注的 3D 对象轨迹，专注于 3D 对象跟踪任务。包括 LiDAR 点云、相机图像和传感器校准信息。
2. **Argoverse 运动预测**：包含从 60 小时驾驶数据中收集的 32.4 万条车辆轨迹，适用于运动预测任务。
3. **Argoverse 立体深度估计**：包含超过 1 万对立体图像及对应的 LiDAR 点云，用于真实深度估计。

### 既然 Argoverse 数据集已从 Amazon S3 移除，在哪里可以下载？

之前可在 Amazon S3 上获取的 Argoverse 数据集 `*.zip` 文件，现在可以从 [Google Drive](https://drive.google.com/file/d/1st9qW3BeIwQsnR0t8mRpvbsSWIo16ACi/view?usp=drive_link) 手动下载。

### Argoverse 数据集使用的 YAML 配置文件是什么？

YAML 文件包含数据集的路径、类别和其他基本信息。对于 Argoverse 数据集，配置文件 `Argoverse.yaml` 可在以下链接找到：[Argoverse.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Argoverse.yaml)。

有关 YAML 配置的更多信息，请参阅我们的[数据集](../index.md)指南。