---
comments: true
description: 探索用于汽车 AI 应用的汽车零部件分割数据集。使用 Ultralytics YOLO 和丰富的标注数据提升您的分割模型。
keywords: 汽车零部件分割数据集, 计算机视觉, 汽车 AI, 车辆维护, Ultralytics, YOLO, 分割模型, 深度学习, 目标分割
---

# 汽车零部件分割数据集

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-carparts-segmentation-dataset.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开汽车零部件分割数据集"></a>

汽车零部件分割数据集是专为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)应用设计的图像和视频精选集，特别专注于[分割任务](https://docs.ultralytics.com/tasks/segment)。该数据集提供了从多个角度拍摄的多样化视觉内容，为训练和测试分割模型提供了宝贵的[标注](https://www.ultralytics.com/glossary/data-labeling)示例。

无论您是从事[汽车研究](https://www.ultralytics.com/solutions/ai-in-automotive)、开发车辆维护 AI 解决方案，还是探索计算机视觉应用，汽车零部件分割数据集都是提升 [Ultralytics YOLO](../../models/yolo26.md) 等模型项目精度和效率的宝贵资源。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/FvWl00sD4rc"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics 平台分割汽车零部件 | 训练、部署与推理 | Ultralytics YOLO26 🚀
</p>

## 数据集结构

汽车零部件分割数据集的数据分布如下：

- **训练集**：包含 3156 张图像，每张图像都附有对应的标注。该集合用于[训练](https://www.ultralytics.com/glossary/training-data)[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)[模型](https://www.ultralytics.com/glossary/foundation-model)。
- **测试集**：包含 276 张图像，每张图像均与其对应的标注配对。该集合用于使用[测试数据](https://www.ultralytics.com/glossary/test-data)评估训练后的模型性能。
- **验证集**：包含 401 张图像，每张图像都有对应的标注。该集合在训练过程中用于调整[超参数](https://docs.ultralytics.com/guides/hyperparameter-tuning)并使用[验证数据](https://www.ultralytics.com/glossary/validation-data)防止[过拟合](https://www.ultralytics.com/glossary/overfitting)。

## 应用场景

汽车零部件分割在多个领域有广泛应用，包括：

- **汽车质量控制**：在制造过程中识别汽车零部件的缺陷或不一致之处（[制造业 AI](https://www.ultralytics.com/solutions/ai-in-manufacturing)）。
- **汽车维修**：协助技术人员识别需要维修或更换的零部件。
- **电商目录**：在网店中自动标记和分类汽车零部件，用于[电商](https://en.wikipedia.org/wiki/E-commerce)平台。
- **交通监控**：分析交通监控视频中的车辆组件。
- **自动驾驶汽车**：增强[自动驾驶汽车](https://www.ultralytics.com/blog/ai-in-self-driving-cars)感知系统，更好地理解周围车辆。
- **保险理赔处理**：通过识别受损汽车零部件来自动化保险理赔的损失评估。
- **回收利用**：对车辆组件进行分类，以实现高效的回收流程。
- **智慧城市建设**：为[智慧城市](https://en.wikipedia.org/wiki/Smart_city)内的城市规划和交通管理系统提供数据支持。

通过准确识别和分类不同的车辆组件，汽车零部件分割简化了流程，并为这些行业提高效率和自动化水平做出了贡献。

## 数据集 YAML

[YAML](https://www.ultralytics.com/glossary/yaml)（Yet Another Markup Language）文件定义数据集配置，包括路径、类别名称和其他重要细节。汽车零部件分割数据集的 `carparts-seg.yaml` 文件位于 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/carparts-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/carparts-seg.yaml)。您可以在 [yaml.org](https://yaml.org/) 了解更多关于 YAML 格式的信息。

!!! example "ultralytics/cfg/datasets/carparts-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/carparts-seg.yaml"
    ```

## 使用方法

要在汽车零部件分割数据集上以图像大小 640 训练 [Ultralytics YOLO26](../../models/yolo26.md) 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，请使用以下代码片段。参阅模型[训练指南](../../modes/train.md)获取可用参数的完整列表，并浏览[模型训练技巧](https://docs.ultralytics.com/guides/model-training-tips)了解最佳实践。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练分割模型，如 YOLO26n-seg
        model = YOLO("yolo26n-seg.pt")  # 加载预训练模型（推荐用于训练）

        # 在汽车零部件分割数据集上训练模型
        results = model.train(data="carparts-seg.yaml", epochs=100, imgsz=640)

        # 训练后，您可以在验证集上验证模型性能
        results = model.val()

        # 或对新图像或视频执行预测
        results = model.predict("path/to/your/image.jpg")
        ```

    === "CLI"

        ```bash
        # 使用命令行界面从预训练的 *.pt 模型开始训练
        # 指定数据集配置文件、模型、轮次数和图像大小
        yolo segment train data=carparts-seg.yaml model=yolo26n-seg.pt epochs=100 imgsz=640

        # 使用验证集验证已训练的模型
        yolo segment val data=carparts-seg.yaml model=path/to/best.pt

        # 使用已训练的模型对特定图像源进行预测
        yolo segment predict model=path/to/best.pt source=path/to/your/image.jpg
        ```

## 示例数据和标注

汽车零部件分割数据集包含从不同角度拍摄的多样化图像和视频。以下是展示数据及其对应标注的示例：

![汽车零部件分割数据集示例图像](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/carparts-seg-sample.avif)

- 该图像展示了汽车图像样本中的[目标分割](https://docs.ultralytics.com/tasks/segment)。带有掩码的标注[边界框](https://www.ultralytics.com/glossary/bounding-box)突出显示了已识别的汽车零部件（如大灯、格栅）。
- 数据集包含在不同条件下（地点、光照、目标密度）拍摄的各种图像，为训练鲁棒的汽车零部件分割模型提供了全面的资源。
- 该示例强调了数据集的复杂性以及[高质量数据](https://www.ultralytics.com/blog/the-importance-of-high-quality-computer-vision-datasets)在计算机视觉任务中的重要性，特别是在汽车组件分析等专业领域。[数据增强](https://www.ultralytics.com/glossary/data-augmentation)等技术可以进一步提升模型的泛化能力。

## 引用和致谢

如果您在研究或开发工作中使用了汽车零部件分割数据集，请引用原始来源：

!!! quote ""

    === "BibTeX"

        ```bibtex
           @misc{ car-seg-un1pm_dataset,
                title = { car-seg Dataset },
                type = { Open Source Dataset },
                author = { Gianmarco Russo },
                url = { https://universe.roboflow.com/gianmarco-russo-vt9xr/car-seg-un1pm },
                year = { 2023 },
                month = { nov },
                note = { visited on 2024-01-24 },
            }
        ```

我们感谢 Gianmarco Russo 和 Roboflow 团队为计算机视觉社区创建和维护这一宝贵数据集。更多数据集，请访问 [Ultralytics 数据集合集](https://docs.ultralytics.com/datasets)。

## 常见问题

### 汽车零部件分割数据集是什么？

汽车零部件分割数据集是一个专门的图像和视频集合，用于训练计算机视觉模型对汽车零部件执行[分割](https://docs.ultralytics.com/tasks/segment)。它包含具有详细标注的多样化视觉内容，适用于汽车 AI 应用。

### 如何使用汽车零部件分割数据集与 Ultralytics YOLO26 配合？

您可以使用此数据集训练 [Ultralytics YOLO26](../../models/yolo26.md) 分割模型。加载预训练模型（例如 `yolo26n-seg.pt`），并使用提供的 Python 或 CLI 示例启动训练，引用 `carparts-seg.yaml` 配置文件。详细说明请查阅[训练指南](../../modes/train.md)。

!!! example "训练示例片段"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-seg.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="carparts-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo segment train data=carparts-seg.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
        ```

### 汽车零部件分割有哪些应用场景？

汽车零部件分割在以下领域非常有用：

- **汽车质量控制**：确保零部件符合标准（[制造业 AI](https://www.ultralytics.com/solutions/ai-in-manufacturing)）。
- **汽车维修**：识别需要维修的零部件。
- **电商**：在线目录化零部件。
- **自动驾驶汽车**：改善车辆感知能力（[汽车 AI](https://www.ultralytics.com/solutions/ai-in-automotive)）。
- **保险**：自动评估车辆损失。
- **回收利用**：高效分类零部件。

### 汽车零部件分割的数据集配置文件在哪里？

数据集配置文件 `carparts-seg.yaml`（包含数据集路径和类别的详细信息）位于 Ultralytics GitHub 仓库中：[carparts-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/carparts-seg.yaml)。

### 为什么要使用汽车零部件分割数据集？

该数据集提供丰富的标注数据，对于开发用于汽车应用的精准[分割模型](https://docs.ultralytics.com/tasks/segment)至关重要。其多样性有助于提升模型在真实场景中（如自动车辆检测、安全系统增强和自动驾驶技术支持）的鲁棒性和性能。使用此类高质量的领域专用数据集可以加速 AI 开发。
