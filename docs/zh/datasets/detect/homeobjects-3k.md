---
comments: true
description: 了解 HomeObjects-3K，一个包含床、沙发、电视、笔记本电脑等 12 个类别的丰富室内目标检测数据集。适用于智能家居、机器人和 AR 领域的计算机视觉应用。
keywords: HomeObjects-3K, 室内数据集, 家居物品, 目标检测, 计算机视觉, YOLO26, 智能家居 AI, 机器人数据集
---

# HomeObjects-3K 数据集

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-homeobjects-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开 HomeObjects-3K 数据集"></a>

HomeObjects-3K 数据集是一个精选的常见家居物品图像集合，专为训练、测试和[基准测试](../../modes/benchmark.md)[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型而设计。该数据集包含约 3,000 张图像和 12 个不同的物品类别，非常适合室内场景理解、智能家居设备、[机器人](https://www.ultralytics.com/glossary/robotics)和增强现实领域的研究与应用。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/v3iqOYoRBFQ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 HomeObjects-3K 数据集上训练 Ultralytics YOLO26 | 检测、验证与 ONNX 导出 🚀
</p>

## 数据集结构

HomeObjects-3K 数据集分为以下子集：

- **训练集**：包含 2,285 张已标注图像，涵盖沙发、椅子、桌子、灯具等物品。
- **验证集**：包含 404 张已标注图像，用于评估模型性能。

每张图像使用与 [Ultralytics YOLO](../detect/index.md#what-is-the-ultralytics-yolo-dataset-format-and-how-to-structure-it) 格式一致的边界框进行标注。室内光照、物体大小和方向的多样性使其在真实部署场景中具有鲁棒性。

## 物品类别

该数据集支持 12 个日常物品类别，涵盖家具、电子产品和装饰品。这些类别经过挑选，反映了室内家居环境中常见的物品，支持[目标检测](../../tasks/detect.md)和[目标追踪](../../modes/track.md)等视觉任务。

!!! Tip "HomeObjects-3K 类别"

    0. bed（床）
    1. sofa（沙发）
    2. chair（椅子）
    3. table（桌子）
    4. lamp（灯具）
    5. tv（电视）
    6. laptop（笔记本电脑）
    7. wardrobe（衣柜）
    8. window（窗户）
    9. door（门）
    10. potted plant（盆栽）
    11. photo frame（相框）

## 应用场景

HomeObjects-3K 支持室内计算机视觉领域的广泛应用，涵盖研究和实际产品开发：

- **室内目标检测**：使用 [Ultralytics YOLO26](../../models/yolo26.md) 等模型在图像中查找和定位床、椅子、灯具和笔记本电脑等常见家居物品，有助于实时理解室内场景。

- **场景布局解析**：在机器人和智能家居系统中，帮助设备理解房间的布局方式，了解门、窗和家具的位置，以便安全导航并与其环境正确交互。

- **AR 应用**：为使用增强现实的应用提供[目标识别](http://ultralytics.com/glossary/image-recognition)功能。例如，检测电视或衣柜并在其上显示额外信息或效果。

- **教育与研究**：为学生和研究人员提供即用型数据集，用于使用真实世界示例练习室内目标检测，支持学习和学术项目。

- **家庭库存与资产追踪**：自动检测和列出照片或视频中的家居物品，可用于管理财产、组织空间或在房地产中可视化家具。

## 数据集 YAML

HomeObjects-3K 数据集的配置通过 YAML 文件提供。该文件概述了关键信息，如训练和验证目录的图像路径以及物品类别列表。
你可以直接从 Ultralytics 仓库访问 `HomeObjects-3K.yaml` 文件：[https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/HomeObjects-3K.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/HomeObjects-3K.yaml)

!!! example "ultralytics/cfg/datasets/HomeObjects-3K.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/HomeObjects-3K.yaml"
    ```

## 使用方法

你可以使用 640 的图像大小在 HomeObjects-3K 数据集上训练 YOLO26n 模型 100 个 epoch。以下示例展示了如何开始。更多训练选项和详细设置请查看[训练](../../modes/train.md)指南。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo26n.pt")

        # 在 HomeObjects-3K 数据集上训练模型
        model.train(data="HomeObjects-3K.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=HomeObjects-3K.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## 示例图像与标注

该数据集包含丰富的室内场景图像，捕捉了自然家居环境中各种家居物品。以下是数据集中的示例视觉图像，每张都配有相应的标注，以展示物品位置、大小和空间关系。

![HomeObjects-3K 数据集家居物品示例](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/homeobjects-3k-dataset-sample.avif)

## 许可证与归属

HomeObjects-3K 由 **[Ultralytics 团队](https://www.ultralytics.com/about)** 开发并发布，采用 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)，支持开源研究和在适当归属下的商业使用。

如果你在研究中使用此数据集，请使用以下详细信息进行引用：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @dataset{Jocher_Ultralytics_Datasets_2025,
            author = {Jocher, Glenn and Rizwan, Muhammad},
            license = {AGPL-3.0},
            month = {May},
            title = {Ultralytics Datasets: HomeObjects-3K Detection Dataset},
            url = {https://docs.ultralytics.com/datasets/detect/homeobjects-3k/},
            version = {1.0.0},
            year = {2025}
        }
        ```

## 常见问题

### HomeObjects-3K 数据集的设计目的是什么？

HomeObjects-3K 专为推动 AI 对室内场景的理解而设计。它专注于检测日常家居物品——如床、沙发、电视和灯具——非常适合智能家居、机器人、增强现实和室内监控系统中的应用。无论你是在为实时边缘设备还是学术研究训练模型，该数据集都提供了一个均衡的基础。

### 包含哪些物品类别，为什么选择这些类别？

数据集包含 12 种最常见的家居物品：床、沙发、椅子、桌子、灯具、电视、笔记本电脑、衣柜、窗户、门、盆栽和相框。这些物品的选择反映了真实的室内环境，并支持多用途任务，如机器人导航或 AR/VR 应用中的场景生成。

### 如何使用 HomeObjects-3K 数据集训练 YOLO 模型？

要训练 YOLO26n 等 YOLO 模型，你只需要 `HomeObjects-3K.yaml` 配置文件和[预训练模型](../../models/index.md)权重。无论是使用 Python 还是 CLI，训练都可以通过一条命令启动。你可以根据目标性能和硬件配置自定义参数，如 epoch 数、图像大小和批量大小。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo26n.pt")

        # 在 HomeObjects-3K 数据集上训练模型
        model.train(data="HomeObjects-3K.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=HomeObjects-3K.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

### 该数据集适合初学者级别的项目吗？

当然。凭借清晰的标注和标准化的 YOLO 兼容标注格式，HomeObjects-3K 是学生和爱好者探索室内场景真实目标检测的绝佳切入点。它也可以很好地扩展到商业环境中更复杂的应用。

### 在哪里可以找到标注格式和 YAML？

请参阅[数据集 YAML](#dataset-yaml) 部分。格式是标准 YOLO 格式，与大多数目标检测流水线兼容。
