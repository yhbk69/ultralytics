---
comments: true
description: 探索 Google 发布的全面 Open Images V7 数据集。了解其标注、应用，并使用 YOLO26 预训练模型进行计算机视觉任务。
keywords: Open Images V7, Google 数据集, 计算机视觉, YOLO26 模型, 目标检测, 图像分割, 视觉关系, AI 研究, Ultralytics
---

# Open Images V7 数据集

[Open Images V7](https://storage.googleapis.com/openimages/web/index.html) 是由 Google 主导的一个多功能且庞大的数据集。旨在推动[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域的研究，它拥有大量图像集合，并标注了丰富的数据，包括图像级标签、物体边界框、物体分割掩码、视觉关系和局部化叙述。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/u3pLlgzUeV8"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 OpenImagesV7 预训练模型进行<a href="https://www.ultralytics.com/glossary/object-detection">目标检测</a>
</p>

## Open Images V7 预训练模型

| 模型                                                                                       | 大小<br><sup>(像素)</sup> | mAP<sup>val<br>50-95</sup> | 速度<br><sup>CPU ONNX<br>(ms)</sup> | 速度<br><sup>A100 TensorRT<br>(ms)</sup> | 参数量<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ------------------------------------------------------------------------------------------ | ------------------------- | -------------------------- | ----------------------------------- | ---------------------------------------- | ------------------------ | ----------------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-oiv7.pt) | 640                       | 18.4                       | 142.4                               | 1.21                                     | 3.5                      | 10.5                    |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-oiv7.pt) | 640                       | 27.7                       | 183.1                               | 1.40                                     | 11.4                     | 29.7                    |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m-oiv7.pt) | 640                       | 33.6                       | 408.5                               | 2.26                                     | 26.2                     | 80.6                    |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8l-oiv7.pt) | 640                       | 34.9                       | 596.9                               | 2.43                                     | 44.1                     | 167.4                   |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x-oiv7.pt) | 640                       | 36.3                       | 860.6                               | 3.56                                     | 68.7                     | 260.6                   |

你可以按如下方式使用这些预训练模型进行推理或微调。

!!! example "预训练模型使用示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 Open Images Dataset V7 预训练的 YOLOv8n 模型
        model = YOLO("yolov8n-oiv7.pt")

        # 运行预测
        results = model.predict(source="image.jpg")

        # 从预训练检查点开始训练
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 使用 Open Images Dataset V7 预训练模型进行预测
        yolo detect predict source=image.jpg model=yolov8n-oiv7.pt

        # 从 Open Images Dataset V7 预训练检查点开始训练
        yolo detect train data=coco8.yaml model=yolov8n-oiv7.pt epochs=100 imgsz=640
        ```

![Open Images V7 类别可视化](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/open-images-v7-classes-visual.avif)

## 主要特性

- 包含约 900 万张以多种方式标注的图像，适用于多种计算机视觉任务。
- 在 190 万张图像中拥有惊人的 1600 万个边界框，涵盖 600 个物体类别。这些边界框主要由专家手工绘制，确保了高[精度](https://www.ultralytics.com/glossary/precision)。
- 提供总计 330 万个视觉关系标注，详细描述了 1,466 个独特的关系三元组、物体属性和人类活动。
- V5 版本为 350 个类别的 280 万个物体引入了分割掩码。
- V6 版本引入了 67.5 万个局部化叙述，融合了语音、文本和鼠标轨迹，突出显示所描述的物体。
- V7 版本在 140 万张图像上引入了 6640 万个点级标签，涵盖 5,827 个类别。
- 包含 6140 万个图像级标签，涵盖 20,638 个不同类别。
- 为[图像分类](https://www.ultralytics.com/glossary/image-classification)、目标检测、关系检测、[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)和多模态图像描述提供了统一平台。

## 数据集结构

Open Images V7 由多个组件构成，满足不同的计算机视觉挑战：

- **图像**：约 900 万张图像，通常展示复杂场景，平均每张图像有 8.3 个物体。
- **边界框**：超过 1600 万个边界框，标定 600 个类别的物体。
- **分割掩码**：详细描述 350 个类别中 280 万个物体的精确边界。
- **视觉关系**：330 万个标注，指示物体关系、属性和动作。
- **局部化叙述**：67.5 万个描述，结合语音、文本和鼠标轨迹。
- **点级标签**：140 万张图像上的 6640 万个标签，适用于零样本/少样本[语义分割](https://www.ultralytics.com/glossary/semantic-segmentation)。

## 应用场景

Open Images V7 是训练和评估各种计算机视觉任务中最先进模型的基石。该数据集广泛的范围和高质量的标注使其成为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域研究人员和开发者的必备资源。

一些关键应用包括：

- **高级目标检测**：训练模型以高精度识别和定位复杂场景中的多个物体。
- **语义理解**：开发理解物体之间视觉关系的系统。
- **图像分割**：为物体创建精确的像素级掩码，实现详细的场景分析。
- **多模态学习**：将视觉数据与文本描述结合，实现更丰富的 AI 理解。
- **零样本学习**：利用广泛的类别覆盖来识别训练中未见的物体。

## 数据集 YAML

Ultralytics 维护一个 `open-images-v7.yaml` 文件，指定了训练所需的数据集路径、类别名称和其他配置详情。

!!! example "OpenImagesV7.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/open-images-v7.yaml"
    ```

## 使用方法

要在 Open Images V7 数据集上以 640 的图像大小训练 YOLO26n 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! warning

    完整的 Open Images V7 数据集包含 1,743,042 张训练图像和 41,620 张验证图像，下载后需要约 **561 GB 的存储空间**。

    执行以下命令将触发完整数据集的自动下载（如果本地尚未存在）。在运行以下示例之前，务必：

    - 验证你的设备是否有足够的存储容量。
    - 确保有稳定且快速的互联网连接。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 在 Open Images V7 数据集上训练模型
        results = model.train(data="open-images-v7.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 在 Open Images V7 数据集上训练 COCO 预训练的 YOLO26n 模型
        yolo detect train data=open-images-v7.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## 示例数据与标注

数据集的图示有助于了解其丰富性：

![Open Images V7 数据集边界框标注示例](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/oidv7-all-in-one-example-ab.avif)

- **Open Images V7**：该图像展示了可用标注的深度和细节，包括边界框、关系和分割掩码。

研究人员可以深入了解该数据集所应对的各种计算机视觉挑战，从基本的目标检测到复杂的关系识别。标注的多样性使 Open Images V7 对于开发能够理解复杂视觉场景的模型特别有价值。

## 引用与致谢

对于在工作中使用 Open Images V7 的人员，建议引用相关论文并致谢创建者：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{OpenImages,
          author = {Alina Kuznetsova and Hassan Rom and Neil Alldrin and Jasper Uijlings and Ivan Krasin and Jordi Pont-Tuset and Shahab Kamali and Stefan Popov and Matteo Malloci and Alexander Kolesnikov and Tom Duerig and Vittorio Ferrari},
          title = {The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale},
          year = {2020},
          journal = {IJCV}
        }
        ```

衷心感谢 Google AI 团队创建和维护 Open Images V7 数据集。要深入了解该数据集及其内容，请访问 [Open Images V7 官方网站](https://storage.googleapis.com/openimages/web/index.html)。

## 常见问题

### Open Images V7 数据集是什么？

Open Images V7 是由 Google 创建的一个庞大且多功能的数据集，旨在推动计算机视觉研究。它包含图像级标签、物体边界框、物体分割掩码、视觉关系和局部化叙述，非常适合目标检测、分割和关系检测等各种计算机视觉任务。

### 如何在 Open Images V7 数据集上训练 YOLO26 模型？

要在 Open Images V7 数据集上训练 YOLO26 模型，可以使用 Python 和 CLI 命令。以下是训练 YOLO26n 模型 100 个 epoch、图像大小为 640 的示例：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 在 Open Images V7 数据集上训练模型
        results = model.train(data="open-images-v7.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # 在 Open Images V7 数据集上训练 COCO 预训练的 YOLO26n 模型
        yolo detect train data=open-images-v7.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

有关参数和设置的更多详情，请参阅[训练](../../modes/train.md)页面。

### Open Images V7 数据集有哪些主要特性？

Open Images V7 数据集包含约 900 万张图像，具有多种标注：

- **边界框**：600 个物体类别的 1600 万个边界框。
- **分割掩码**：350 个类别中 280 万个物体的掩码。
- **视觉关系**：330 万个标注，指示关系、属性和动作。
- **局部化叙述**：675,000 个描述，结合语音、文本和鼠标轨迹。
- **点级标签**：140 万张图像上的 6640 万个标签。
- **图像级标签**：20,638 个类别的 6140 万个标签。

### Open Images V7 数据集有哪些可用的预训练模型？

Ultralytics 为 Open Images V7 数据集提供了多个 YOLOv8 预训练模型，每个模型具有不同的大小和性能指标：

| 模型                                                                                       | 大小<br><sup>(像素)</sup> | mAP<sup>val<br>50-95</sup> | 速度<br><sup>CPU ONNX<br>(ms)</sup> | 速度<br><sup>A100 TensorRT<br>(ms)</sup> | 参数量<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ------------------------------------------------------------------------------------------ | ------------------------- | -------------------------- | ----------------------------------- | ---------------------------------------- | ------------------------ | ----------------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-oiv7.pt) | 640                       | 18.4                       | 142.4                               | 1.21                                     | 3.5                      | 10.5                    |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-oiv7.pt) | 640                       | 27.7                       | 183.1                               | 1.40                                     | 11.4                     | 29.7                    |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m-oiv7.pt) | 640                       | 33.6                       | 408.5                               | 2.26                                     | 26.2                     | 80.6                    |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8l-oiv7.pt) | 640                       | 34.9                       | 596.9                               | 2.43                                     | 44.1                     | 167.4                   |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x-oiv7.pt) | 640                       | 36.3                       | 860.6                               | 3.56                                     | 68.7                     | 260.6                   |

### Open Images V7 数据集可用于哪些应用？

Open Images V7 数据集支持多种计算机视觉任务，包括：

- **[图像分类](https://www.ultralytics.com/glossary/image-classification)**
- **目标检测**
- **实例分割**
- **视觉关系检测**
- **多模态图像描述**

其全面的标注和广泛的范围使其适合训练和评估高级[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型，如我们在[应用场景](#applications)部分中详述的实际用例所示。
