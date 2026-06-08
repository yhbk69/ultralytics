---
comments: true
description: 探索用于高级姿态估计的手部关键点估计数据集。了解数据集、预训练模型、评估指标以及使用 YOLO 进行训练的应用。
keywords: 手部关键点, 姿态估计, 数据集, 关键点, MediaPipe, YOLO, 深度学习, 计算机视觉
---

# 手部关键点数据集

## 介绍

手部关键点数据集包含 26,768 张标注有关键点的手部图像，适用于训练 Ultralytics YOLO 等模型进行姿态估计任务。标注使用 Google MediaPipe 库生成，确保了高[准确率](https://www.ultralytics.com/glossary/accuracy)和一致性，且该数据集与 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) 格式兼容。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/fd6u1TW_AGY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 使用 Ultralytics YOLO26 进行手部关键点估计 | 人手姿态估计教程
</p>

## 手部关键点

![包含 21 个点的手部关键点标注图](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/hand_landmarks.jpg)

## 关键点

该数据集包含用于手部检测的关键点。关键点标注如下：

1. 手腕
2. 拇指（4 个点）
3. 食指（4 个点）
4. 中指（4 个点）
5. 无名指（4 个点）
6. 小指（4 个点）

每只手共有 21 个关键点。

## 关键特性

- **大型数据集**：26,768 张带有手部关键点标注的图像。
- **YOLO26 兼容**：标签以 YOLO 关键点格式提供，可直接用于 YOLO26 模型。
- **21 个关键点**：详细的手部姿态表示，涵盖手腕和每根手指的四个点。

## 数据集结构

手部关键点数据集分为两个子集：

1. **训练集**：此子集包含来自手部关键点数据集的 18,776 张图像，标注用于训练姿态估计模型。
2. **验证集**：此子集包含 7,992 张图像，可用于模型训练期间的验证。

## 应用

手部关键点可用于[手势识别](https://www.ultralytics.com/blog/enhancing-hand-keypoints-estimation-with-ultralytics-yolo11)、[AR/VR 控制](https://docs.ultralytics.com/tasks/pose)、机器人操控以及医疗保健中的手部运动分析。它们还可应用于运动捕捉动画和生物特征认证安全系统。手指位置的详细跟踪可以实现与虚拟对象的精确交互和非接触式控制界面。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含数据集的路径、类别和其他相关信息。对于手部关键点数据集，`hand-keypoints.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/hand-keypoints.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/hand-keypoints.yaml)。

!!! example "ultralytics/cfg/datasets/hand-keypoints.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/hand-keypoints.yaml"
    ```

## 使用方法

要在手部关键点数据集上以 640 的图像尺寸训练 YOLO26n-pose 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，可使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-pose.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练 *.pt 模型开始训练
        yolo pose train data=hand-keypoints.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
        ```

## 示例图像与标注

手部关键点数据集包含一组多样化的人手图像，标注有关键点。以下是一些数据集图像示例及其对应标注：

![手部关键点姿态估计数据集样本](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/human-hand-pose.avif)

- **马赛克图像**：此图像展示了一个由马赛克数据集图像组成的训练批次。马赛克是一种训练中使用的技术，将多张图像组合成一张图像，以增加每个训练批次中对象和场景的多样性。这有助于提高模型对不同对象大小、宽高比和上下文的泛化能力。

该示例展示了手部关键点数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克的好处。

## 引用与致谢

如果你在研究或开发工作中使用手部关键点数据集，请致谢以下来源：

!!! quote ""

    === "致谢"

    我们要感谢以下来源提供本数据集中使用的图像：

    - [11k Hands](https://sites.google.com/view/11khands)
    - [2000 Hand Gestures](https://www.kaggle.com/datasets/ritikagiridhar/2000-hand-gestures)
    - [Gesture Recognition](https://www.kaggle.com/datasets/imsparsh/gesture-recognition)

    这些图像是根据各平台提供的相应许可证收集的，并根据 [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) 分发。

我们还要感谢本数据集的创建者 [Rion Dsilva](https://www.linkedin.com/in/rion-dsilva-043464229/)，感谢他对 Vision AI 研究的巨大贡献。

## 常见问题

### 如何在手部关键点数据集上训练 YOLO26 模型？

要在手部关键点数据集上训练 YOLO26 模型，可以使用 Python 或命令行界面（CLI）。以下是以 640 的图像尺寸训练 YOLO26n-pose 模型 100 个 epoch 的示例：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-pose.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练 *.pt 模型开始训练
        yolo pose train data=hand-keypoints.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
        ```

有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

### 手部关键点数据集的关键特性是什么？

手部关键点数据集专为高级[姿态估计](https://docs.ultralytics.com/datasets/pose)任务而设计，包含以下关键特性：

- **大型数据集**：包含 26,768 张带有手部关键点标注的图像。
- **YOLO26 兼容**：可直接用于 YOLO26 模型。
- **21 个关键点**：详细的手部姿态表示，包括手腕和手指关节。

更多详情，请参阅[手部关键点数据集](#介绍)部分。

### 哪些应用可以从手部关键点数据集中受益？

手部关键点数据集可应用于多个领域，包括：

- **手势识别**：增强人机交互。
- **AR/VR 控制**：改善增强现实和虚拟现实中的用户体验。
- **机器人操控**：实现机器人手的精确控制。
- **医疗保健**：分析手部运动用于医疗诊断。
- **动画**：捕捉动作用于逼真动画。
- **生物特征认证**：增强安全系统。

更多信息，请参阅[应用](#应用)部分。

### 手部关键点数据集的结构是怎样的？

手部关键点数据集分为两个子集：

1. **训练集**：包含 18,776 张图像，用于训练姿态估计模型。
2. **验证集**：包含 7,992 张图像，用于模型训练期间的验证。

这种结构确保了全面的训练和验证过程。更多详情，请参阅[数据集结构](#数据集结构)部分。

### 如何使用数据集 YAML 文件进行训练？

数据集配置定义在 YAML 文件中，包括路径、类别和其他相关信息。`hand-keypoints.yaml` 文件可在 [hand-keypoints.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/hand-keypoints.yaml) 找到。

要使用此 YAML 文件进行训练，请如上述训练示例所示在训练脚本或 CLI 命令中指定它。更多详情，请参阅[数据集 YAML](#数据集-yaml) 部分。