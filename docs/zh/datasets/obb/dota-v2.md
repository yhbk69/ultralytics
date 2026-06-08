---
comments: true
description: 探索用于航拍图像目标检测的 DOTA 数据集，包含 170 万个旋转边界框，涵盖 18 个类别。非常适合航拍图像分析。
keywords: DOTA 数据集, 目标检测, 航拍图像, 旋转边界框, OBB, DOTA v1.0, DOTA v1.5, DOTA v2.0, 多尺度检测, Ultralytics
---

# 带 OBB 的 DOTA 数据集

[DOTA](https://captain-whu.github.io/DOTA/index.html) 是一个专门的数据集，专注于航拍图像中的[目标检测](https://www.ultralytics.com/glossary/object-detection)。它源自 DOTA 系列数据集，提供带有[旋转边界框 (OBB)](https://docs.ultralytics.com/datasets/obb)的标注图像，捕捉了多样化的航拍场景。

![DOTA 数据集航拍检测物体类别](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/dota-classes-visual.avif)

## 主要特性

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/JjQ-URE0LJE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 Google Colab 中的 DOTA 数据集上训练 Ultralytics YOLO26 进行旋转边界框检测
</p>

- 从各种传感器和平台采集，图像大小范围从 800 × 800 到 20,000 × 20,000 像素。
- 包含超过 170 万个旋转边界框，涵盖 18 个类别。
- 由于每张图像中物体尺寸分布广泛，涵盖了多尺度目标检测。
- 实例由专家使用任意（8 自由度）四边形进行标注，捕捉不同尺度、方向和形状的物体。

## 数据集版本

### DOTA-v1.0

- 包含 15 个常见类别。
- 包含 2,806 张图像，188,282 个实例。
- 拆分比例：1/2 用于训练，1/6 用于验证，1/3 用于测试。

### DOTA-v1.5

- 包含与 DOTA-v1.0 相同的图像。
- 微小实例（小于 10 像素）也被标注。
- 新增类别："container crane"（集装箱起重机）。
- 共 403,318 个实例。
- 为 [DOAI Challenge 2019 on Object Detection in Aerial Images](https://captain-whu.github.io/DOAI2019/challenge.html) 发布。

### DOTA-v2.0

- 来自 Google Earth、GF-2 卫星和其他航拍图像的采集。
- 包含 18 个常见类别。
- 包含 11,268 张图像，惊人的 1,793,658 个实例。
- 新增类别："airport"（机场）和 "helipad"（直升机停机坪）。
- 图像拆分：
    - 训练：1,830 张图像，268,627 个实例。
    - 验证：593 张图像，81,048 个实例。
    - 测试开发集：2,792 张图像，353,346 个实例。
    - 测试挑战集：6,053 张图像，1,090,637 个实例。

## 数据集结构

DOTA 展示了一个为 OBB 目标检测挑战量身定制的结构化布局：

- **图像**：大量高分辨率航拍图像，捕捉了多样化的地形和结构。
- **旋转边界框**：以旋转矩形形式的标注，无论物体方向如何都能包裹物体，非常适合捕捉飞机、船只和建筑物等物体。

## 应用场景

DOTA 作为专门为航拍图像分析量身定制的模型的训练和评估基准。通过包含 OBB 标注，它提供了独特的挑战，使得能够开发适合航拍图像细微差别的专用[目标检测](https://docs.ultralytics.com/tasks/detect)模型。该数据集在遥感、监控和环境监测应用中特别有价值。

## 数据集 YAML

数据集 YAML 文件指定图像/标签根目录、类别名称和其他重要元数据。Ultralytics 为两个最常用的版本维护官方 YAML 文件：

- [`DOTAv1.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml)
- [`DOTAv1.5.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.5.yaml)

使用与你下载的版本匹配的 YAML，或者如果你使用 DOTA-v2 或其他衍生版本，自行编写自定义 YAML。

!!! example "DOTAv1.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/DOTAv1.yaml"
    ```

## 拆分 DOTA 图像

原始图像的边长通常超过 10,000 像素，因此在将数据输入 YOLO 之前需要进行切片。使用以下辅助工具将原始图像以多尺度切片为重叠的 1024 × 1024 裁剪块，同时保持标注同步。

!!! example "拆分图像"

    === "Python"

        ```python
        from ultralytics.data.split_dota import split_test, split_trainval

        # 拆分训练和验证集，带标签。
        split_trainval(
            data_root="path/to/DOTAv1.0/",
            save_dir="path/to/DOTAv1.0-split/",
            rates=[0.5, 1.0, 1.5],  # 多尺度
            gap=500,
        )
        # 拆分测试集，不带标签。
        split_test(
            data_root="path/to/DOTAv1.0/",
            save_dir="path/to/DOTAv1.0-split/",
            rates=[0.5, 1.0, 1.5],  # 多尺度
            gap=500,
        )
        ```

!!! tip

    保持输出目录以标准的 YOLO 布局组织（`images/train`、`labels/train` 等），以便你可以直接从数据集 YAML 引用它。

## 使用方法

要在 DOTA v1 数据集上训练模型，可以使用以下代码片段。始终参阅模型文档以获取完整的可用参数列表。对于那些希望先尝试较小子集的人，可以考虑使用仅包含 8 张图像用于快速测试的 [DOTA8 数据集](https://docs.ultralytics.com/datasets/obb/dota8)。

!!! warning

    请注意，DOTAv1 数据集中的所有图像和相关标注可用于学术目的，但禁止商业使用。感谢你对数据集创建者意愿的理解和尊重！

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 从头创建新的 YOLO26n-OBB 模型
        model = YOLO("yolo26n-obb.yaml")

        # 在 DOTAv1 数据集上训练模型
        results = model.train(data="DOTAv1.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        # 在 DOTAv1 数据集上训练新的 YOLO26n-OBB 模型
        yolo obb train data=DOTAv1.yaml model=yolo26n-obb.pt epochs=100 imgsz=1024
        ```

## 示例数据与标注

查看数据集可以展示其深度：

![DOTA 数据集带旋转边界框标注](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/instances-DOTA.avif)

- **DOTA 示例**：此快照强调了航拍场景的复杂性以及旋转[边界框](https://www.ultralytics.com/glossary/bounding-box)标注的重要性，能够以自然方向捕捉物体。

数据集的丰富性为航拍图像独有的目标检测挑战提供了宝贵的洞察。[DOTA-v2.0 数据集](https://www.ultralytics.com/blog/exploring-the-best-computer-vision-datasets-in-2025)因其全面的标注和多样化的物体类别，在遥感和航拍监控项目中特别受欢迎。

## 引用与致谢

如果你在工作中使用 DOTA，请引用相关研究论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{9560031,
          author={Ding, Jian and Xue, Nan and Xia, Gui-Song and Bai, Xiang and Yang, Wen and Yang, Michael and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          title={Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges},
          year={2021},
          volume={},
          number={},
          pages={1-1},
          doi={10.1109/TPAMI.2021.3117983}
        }
        ```

特别感谢 DOTA 数据集背后的团队在整理此数据集方面所做的值得称赞的努力。要全面了解数据集及其细微差别，请访问 [DOTA 官方网站](https://captain-whu.github.io/DOTA/index.html)。

## 常见问题

### DOTA 数据集是什么，为什么它对航拍图像中的目标检测很重要？

[DOTA 数据集](https://captain-whu.github.io/DOTA/index.html)是一个专注于航拍图像目标检测的专门数据集。它采用旋转边界框 (OBB)，提供来自多样化航拍场景的标注图像。DOTA 在其 170 万个标注和 18 个类别中的物体方向、尺度和形状的多样性，使其成为开发和评估专门为航拍图像分析量身定制的模型的理想选择，例如用于监控、环境监测和灾害管理的模型。

### DOTA 数据集如何处理图像中的不同尺度和方向？

DOTA 使用旋转边界框 (OBB) 进行标注，由旋转矩形表示，无论物体方向如何都能包裹物体。此方法确保物体无论大小或角度如何都能被准确捕捉。数据集的多尺度图像范围从 800 × 800 到 20,000 × 20,000 像素，进一步允许有效检测小型和大型物体。这种方法对于物体以各种角度和尺度出现的航拍图像特别有价值。

### 如何使用 DOTA 数据集训练模型？

要使用 DOTA 数据集训练模型，可以使用以下 [Ultralytics YOLO](https://docs.ultralytics.com/tasks/obb) 示例：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 从头创建新的 YOLO26n-OBB 模型
        model = YOLO("yolo26n-obb.yaml")

        # 在 DOTAv1 数据集上训练模型
        results = model.train(data="DOTAv1.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        # 在 DOTAv1 数据集上训练新的 YOLO26n-OBB 模型
        yolo obb train data=DOTAv1.yaml model=yolo26n-obb.pt epochs=100 imgsz=1024
        ```

有关如何拆分和预处理 DOTA 图像的更多详细信息，请参阅[拆分 DOTA 图像部分](#拆分-dota-图像)。

### DOTA-v1.0、DOTA-v1.5 和 DOTA-v2.0 之间有什么区别？

- **DOTA-v1.0**：包含 15 个常见类别，2,806 张图像，188,282 个实例。数据集分为训练、验证和测试集。
- **DOTA-v1.5**：在 DOTA-v1.0 的基础上标注了微小实例（小于 10 像素）并新增"集装箱起重机"类别，共 403,318 个实例。
- **DOTA-v2.0**：通过 Google Earth 和 GF-2 卫星的标注进一步扩展，包含 11,268 张图像和 1,793,658 个实例。新增"机场"和"直升机停机坪"类别。

详细比较和更多具体信息，请查看[数据集版本部分](#数据集版本)。

### 如何准备高分辨率 DOTA 图像进行训练？

DOTA 图像可能非常大，需拆分为较小分辨率以便管理训练。以下是拆分图像的 Python 代码片段：

!!! example

    === "Python"

        ```python
        from ultralytics.data.split_dota import split_test, split_trainval

        # 拆分训练和验证集，带标签。
        split_trainval(
            data_root="path/to/DOTAv1.0/",
            save_dir="path/to/DOTAv1.0-split/",
            rates=[0.5, 1.0, 1.5],  # 多尺度
            gap=500,
        )
        # 拆分测试集，不带标签。
        split_test(
            data_root="path/to/DOTAv1.0/",
            save_dir="path/to/DOTAv1.0-split/",
            rates=[0.5, 1.0, 1.5],  # 多尺度
            gap=500,
        )
        ```

此过程有助于提高训练效率和模型性能。详细说明请访问[拆分 DOTA 图像部分](#拆分-dota-图像)。
