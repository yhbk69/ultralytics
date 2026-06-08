---
comments: true
description: 探索 xView 数据集，一个包含 100 万+ 物体实例的高分辨率卫星图像丰富资源。增强检测、学习效率等。
keywords: xView 数据集, 俯视图像, 卫星图像, 目标检测, 高分辨率, 边界框, 计算机视觉, TensorFlow, PyTorch, 数据集结构
---

# xView 数据集

[xView](http://xviewdataset.org/) 数据集是最大的公开俯视图像数据集之一，包含来自世界各地复杂场景的图像，使用边界框进行标注。xView 数据集的目标是加速四个[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)前沿领域的进展：

1. 降低检测所需的最低分辨率。
2. 提高学习效率。
3. 发现更多物体类别。
4. 改进细粒度类别的检测。

xView 建立在 [Common Objects in Context (COCO)](../detect/coco.md) 等挑战的成功基础上，旨在利用计算机视觉分析来自太空的日益增长的可用图像，以新的方式理解视觉世界，并应对一系列重要应用。

!!! warning "需要手动下载"

    xView 数据集**不会**由 Ultralytics 脚本自动下载。你**必须**先从官方来源手动下载数据集：

    - **来源：**美国国家地理空间情报局（NGA）的 DIUx xView 2018 挑战赛
    - **URL：** [https://challenge.xviewdataset.org](https://challenge.xviewdataset.org)

    **重要提示：**下载必要文件（如 `train_images.tif`、`val_images.tif`、`xView_train.geojson`）后，你需要将其解压并放入正确的目录结构中（通常预期在 `datasets/xView/` 文件夹下），**然后**才能运行下面提供的训练命令。确保按照挑战赛的说明正确设置数据集。

## 主要特性

- xView 包含超过 100 万个物体实例，涵盖 60 个类别。
- 数据集分辨率为 0.3 米，提供比大多数公开卫星图像数据集更高分辨率的图像。
- xView 具有多样化的小型、稀有、细粒度和多类型物体集合，并带有[边界框](https://www.ultralytics.com/glossary/bounding-box)标注。
- 附带使用 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 目标检测 API 的预训练基线模型和 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 示例。

## 数据集结构

xView 数据集由 WorldView-3 卫星以 0.3 米地面采样距离采集的卫星图像组成。它在超过 1,400 平方公里的图像中包含超过 100 万个物体，涵盖 60 个类别。该数据集对于[遥感](https://www.ultralytics.com/blog/using-computer-vision-to-analyze-satellite-imagery)应用和环境监测特别有价值。

## 应用场景

xView 数据集广泛用于训练和评估俯视图像目标检测的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型。数据集中多样化的物体类别和高分辨率图像使其成为计算机视觉领域（尤其是卫星图像分析）研究人员和实践者的宝贵资源。应用包括：

- 军事和国防侦察
- 城市规划与发展
- 环境监测
- 灾害响应与评估
- 基础设施测绘与管理

## 数据集 YAML

YAML 文件用于定义数据集配置，包含数据集路径、类别和其他相关信息。xView 数据集的 `xView.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/xView.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/xView.yaml)。

!!! example "ultralytics/cfg/datasets/xView.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/xView.yaml"
    ```

## 使用方法

要在 xView 数据集上以 640 的图像大小训练模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型的[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="xView.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=xView.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## 示例数据与标注

xView 数据集包含高分辨率卫星图像，具有使用边界框标注的多样化物体集合。以下是数据集中的一些数据示例及其对应标注：

![xView 数据集俯视卫星图像目标检测](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/overhead-imagery-object-detection.avif)

- **俯视图像**：该图像展示了俯视图像中[目标检测](https://www.ultralytics.com/glossary/object-detection)的示例，物体用边界框标注。数据集提供了高分辨率卫星图像，以促进此任务模型的开发。

该示例展示了 xView 数据集中数据的多样性和复杂性，并突显了高质量卫星图像对目标检测任务的重要性。

## 相关数据集

如果你正在处理卫星图像，可能也对以下相关数据集感兴趣：

- [DOTA-v2](../obb/dota-v2.md)：用于航拍图像旋转目标检测的数据集
- [VisDrone](../detect/visdrone.md)：用于无人机拍摄图像中目标检测和追踪的数据集
- [Argoverse](../detect/argoverse.md)：具有 3D 追踪标注的自动驾驶数据集

## 引用与致谢

如果你在研究或开发工作中使用 xView 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lam2018xview,
              title={xView: Objects in Context in Overhead Imagery},
              author={Darius Lam and Richard Kuzma and Kevin McGee and Samuel Dooley and Michael Laielli and Matthew Klaric and Yaroslav Bulatov and Brendan McCord},
              year={2018},
              eprint={1802.07856},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

我们感谢[国防创新单元](https://www.diu.mil/)（DIU）和 xView 数据集的创建者对计算机视觉研究社区的宝贵贡献。有关 xView 数据集及其创建者的更多信息，请访问 [xView 数据集网站](http://xviewdataset.org/)。

## 常见问题

### xView 数据集是什么，它如何促进计算机视觉研究？

[xView](http://xviewdataset.org/) 数据集是最大的公开高分辨率俯视图像数据集之一，包含超过 100 万个物体实例，涵盖 60 个类别。它旨在增强计算机视觉研究的各个方面，如降低检测所需的最低分辨率、提高学习效率、发现更多物体类别以及推进细粒度目标检测。

### 如何使用 Ultralytics YOLO 在 xView 数据集上训练模型？

要使用 [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo26) 在 xView 数据集上训练模型，请按以下步骤操作：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="xView.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=xView.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

有关详细的参数和设置，请参阅模型[训练](../../modes/train.md)页面。

### xView 数据集的主要特性是什么？

xView 数据集因其全面的特性而脱颖而出：

- 超过 100 万个物体实例，涵盖 60 个不同类别。
- 0.3 米的高分辨率图像。
- 多样化的物体类型，包括小型、稀有和细粒度物体，均使用边界框标注。
- 提供预训练基线模型以及 TensorFlow 和 PyTorch 示例。

### xView 的数据集结构是怎样的，它是如何标注的？

xView 数据集包含由 WorldView-3 卫星以 0.3 米地面采样距离采集的高分辨率卫星图像，在约 1,400 平方公里的标注图像中涵盖超过 100 万个物体，涉及 60 个不同类别。每个物体都用边界框标注，使数据集非常适合训练和评估俯视目标检测的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型。详细分类请参阅[数据集结构部分](#dataset-structure)。

### 如何在研究中引用 xView 数据集？

如果你在研究中使用 xView 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lam2018xview,
            title={xView: Objects in Context in Overhead Imagery},
            author={Darius Lam and Richard Kuzma and Kevin McGee and Samuel Dooley and Michael Laielli and Matthew Klaric and Yaroslav Bulatov and Brendan McCord},
            year={2018},
            eprint={1802.07856},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
        }
        ```

有关 xView 数据集的更多信息，请访问官方 [xView 数据集网站](http://xviewdataset.org/)。
