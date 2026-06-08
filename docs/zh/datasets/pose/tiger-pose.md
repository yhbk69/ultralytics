---
comments: true
description: 探索包含 263 张多样化图像的 Ultralytics Tiger-Pose 数据集。非常适合测试、训练和细化姿态估计算法。
keywords: Ultralytics, Tiger-Pose, 数据集, 姿态估计, YOLO26, 训练数据, 机器学习, 神经网络
---

# Tiger-Pose 数据集

## 介绍

[Ultralytics](https://www.ultralytics.com/) 推出 Tiger-Pose 数据集，这是一个专为姿态估计任务设计的多功能集合。该数据集包含从 [YouTube 视频](https://www.youtube.com/watch?v=MIBAT6BGE6U&pp=ygUbVGlnZXIgd2Fsa2luZyByZWZlcmVuY2UubXA0)中提取的 263 张图像，其中 210 张用于训练，53 张用于验证。它是测试和排查姿态估计算法问题的优质资源。

尽管其训练集仅包含 210 张图像，Tiger-Pose 数据集仍提供了足够的多样性，适合评估训练管道、识别潜在错误，并在使用更大数据集进行[姿态估计](https://docs.ultralytics.com/tasks/pose)之前作为一个有价值的准备步骤。

此数据集适用于 [Ultralytics 平台](https://platform.ultralytics.com/) 和 [YOLO26](https://github.com/ultralytics/ultralytics)。

## 数据集结构

- **图像总数**：263 张（210 张训练 / 53 张验证）。
- **关键点**：每只老虎 12 个关键点（无可见性标志）。
- **目录布局**：YOLO 格式关键点存储在 `labels/{train,val}` 下，图像存储在 `images/{train,val}` 下。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Gc6K5eKrTNQ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 使用 Ultralytics 平台在 Tiger-Pose 数据集上训练 YOLO26 姿态模型
</p>

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于指定数据集的配置详情。它包含文件路径、类别定义和其他相关信息等关键数据。具体到 `tiger-pose.yaml` 文件，你可以查看 [Ultralytics Tiger-Pose 数据集配置文件](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/tiger-pose.yaml)。

!!! example "ultralytics/cfg/datasets/tiger-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/tiger-pose.yaml"
    ```

## 使用方法

要在 Tiger-Pose 数据集上以 640 的图像尺寸训练 YOLO26n-pose 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，可使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-pose.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="tiger-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练 *.pt 模型开始训练
        yolo pose train data=tiger-pose.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
        ```

## 示例图像与标注

以下是一些来自 Tiger-Pose 数据集的图像示例及其对应标注：

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch-4.avif" alt="Tiger 姿态估计数据集马赛克训练批次" width="100%">

- **马赛克图像**：此图像展示了一个由马赛克数据集图像组成的训练批次。马赛克是一种训练中使用的技术，将多张图像组合成一张图像，以增加每个训练批次中对象和场景的多样性。这有助于提高模型对不同对象大小、宽高比和上下文的泛化能力。

该示例展示了 Tiger-Pose 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克的好处。

## 推理示例

!!! example "推理示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/best.pt")  # 加载 tiger-pose 训练好的模型

        # 运行推理
        results = model.predict(source="https://youtu.be/MIBAT6BGE6U", show=True)
        ```

    === "CLI"

        ```bash
        # 使用 tiger-pose 训练好的模型运行推理
        yolo pose predict source="https://youtu.be/MIBAT6BGE6U" show=True model="path/to/best.pt"
        ```

## 引用与致谢

该数据集以 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)发布。

## 常见问题

### Ultralytics Tiger-Pose 数据集用于什么？

Ultralytics Tiger-Pose 数据集专为姿态估计任务设计，包含从 [YouTube 视频](https://www.youtube.com/watch?v=MIBAT6BGE6U&pp=ygUbVGlnZXIgd2Fsa2luZyByZWZlcmVuY2UubXA0)中提取的 263 张图像。该数据集分为 210 张训练图像和 53 张验证图像。它特别适用于使用 [Ultralytics 平台](https://platform.ultralytics.com/) 和 [YOLO26](https://github.com/ultralytics/ultralytics) 测试、训练和细化姿态估计算法。

### 如何在 Tiger-Pose 数据集上训练 YOLO26 模型？

要在 Tiger-Pose 数据集上以 640 的图像尺寸训练 YOLO26n-pose 模型 100 个 epoch，请使用以下代码片段。更多详情请访问[训练](../../modes/train.md)页面：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-pose.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="tiger-pose.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # 从预训练 *.pt 模型开始训练
        yolo pose train data=tiger-pose.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
        ```

### `tiger-pose.yaml` 文件包含哪些配置？

`tiger-pose.yaml` 文件用于指定 Tiger-Pose 数据集的配置详情。它包括文件路径和类别定义等关键数据。要查看确切的配置，可以查看 [Ultralytics Tiger-Pose 数据集配置文件](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/tiger-pose.yaml)。

### 如何使用 Tiger-Pose 数据集上训练的 YOLO26 模型运行推理？

要使用在 Tiger-Pose 数据集上训练的 YOLO26 模型执行推理，可以使用以下代码片段。详细指南请访问[预测](../../modes/predict.md)页面：

!!! example "推理示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/best.pt")  # 加载 tiger-pose 训练好的模型

        # 运行推理
        results = model.predict(source="https://youtu.be/MIBAT6BGE6U", show=True)
        ```


    === "CLI"

        ```bash
        # 使用 tiger-pose 训练好的模型运行推理
        yolo pose predict source="https://youtu.be/MIBAT6BGE6U" show=True model="path/to/best.pt"
        ```

### 使用 Tiger-Pose 数据集进行姿态估计有哪些好处？

尽管 Tiger-Pose 数据集规模可控（仅 210 张训练图像），但它提供了一组多样化的图像集合，非常适合测试姿态估计管道。该数据集有助于识别潜在错误，并在使用更大数据集之前作为准备步骤。此外，它还支持使用 [Ultralytics 平台](https://platform.ultralytics.com/) 和 [YOLO26](https://github.com/ultralytics/ultralytics) 等先进工具训练和细化姿态估计算法，从而提升模型性能和[准确率](https://www.ultralytics.com/glossary/accuracy)。