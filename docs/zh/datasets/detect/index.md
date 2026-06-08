---
comments: true
description: 了解与 Ultralytics YOLO 兼容的数据集格式，用于鲁棒的目标检测。探索支持的数据集并学习如何转换格式。
keywords: Ultralytics, YOLO, 目标检测数据集, 数据集格式, COCO, 数据集转换, 训练数据集
---

# 目标检测数据集概览

训练一个鲁棒且准确的[目标检测](https://www.ultralytics.com/glossary/object-detection)模型需要一个全面的数据集。本指南介绍与 Ultralytics YOLO 模型兼容的各种数据集格式，并提供有关其结构、使用方法以及如何在不同格式之间转换的见解。

## 支持的数据集格式

### Ultralytics YOLO 格式

Ultralytics YOLO 格式是一种数据集配置格式，允许您定义数据集根目录、训练/验证/测试图像目录或包含图像路径的 `*.txt` 文件的相对路径，以及类别名称的字典。以下是一个示例：

!!! example "ultralytics/cfg/datasets/coco8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8.yaml"
    ```

此格式的标签应导出为 YOLO 格式，每张图像一个 `*.txt` 文件。如果图像中没有对象，则不需要 `*.txt` 文件。`*.txt` 文件应按 `class x_center y_center width height` 格式每行一个对象。边界框坐标必须采用**归一化 xywh** 格式（0 到 1）。如果您的边界框以像素为单位，应将 `x_center` 和 `width` 除以图像宽度，将 `y_center` 和 `height` 除以图像高度。类别编号应从零开始索引（从 0 开始）。

<p align="center"><img width="750" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/two-persons-tie.avif" alt="带有人物和领带边界框的 YOLO 标注图像"></p>

上述图像对应的标签文件包含 2 个 person（类别 `0`）和一个 tie（类别 `27`）：

<p align="center"><img width="428" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/two-persons-tie-1.avif" alt="带有归一化坐标的 YOLO 格式标签文件"></p>

使用 Ultralytics YOLO 格式时，请按照下面 [COCO8 数据集](coco8.md)示例组织您的训练和验证图像及标签。

<p align="center"><img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/two-persons-tie-2.avif" alt="带有 train 和 val 文件夹的 YOLO 数据集目录结构"></p>

#### 使用示例

以下是使用 YOLO 格式数据集训练模型的方法：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

### Ultralytics NDJSON 格式

NDJSON（换行分隔 JSON）格式提供了为 Ultralytics YOLO 模型定义数据集的替代方式。该格式将数据集元数据和标注存储在单个文件中，其中每行包含一个独立的 JSON 对象。

一个 NDJSON 数据集文件包含：

1. **数据集记录**（第一行）：包含数据集元数据，包括任务类型、类别名称和通用信息
2. **图像记录**（后续行）：包含单个图像数据，包括尺寸、标注和文件路径

!!! example "NDJSON 示例"

    === "数据集记录（第 1 行）"

        ```json
        {
            "type": "dataset",
            "task": "detect",
            "name": "Example",
            "description": "COCO NDJSON example dataset",
            "url": "https://app.ultralytics.com/user/datasets/example",
            "class_names": { "0": "person", "1": "bicycle", "2": "car" },
            "bytes": 426342,
            "version": 0,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z"
        }
        ```

    === "检测 (Detect)"

        ```json
        {
            "type": "image",
            "file": "image1.jpg",
            "url": "https://www.url.com/path/to/image1.jpg",
            "width": 640,
            "height": 480,
            "split": "train",
            "annotations": {
                "boxes": [
                    [0, 0.525, 0.376, 0.284, 0.418],
                    [1, 0.735, 0.298, 0.193, 0.337]
                ]
            }
        }
        ```

        格式：`[class_id, x_center, y_center, width, height]`

    === "分割 (Segment)"

        ```json
        {
            "type": "image",
            "file": "image1.jpg",
            "url": "https://www.url.com/path/to/image1.jpg",
            "width": 640,
            "height": 480,
            "split": "train",
            "annotations": {
                "segments": [
                    [0, 0.681, 0.485, 0.670, 0.487, 0.676, 0.487, 0.688, 0.515],
                    [1, 0.422, 0.315, 0.438, 0.330, 0.445, 0.328, 0.450, 0.320]
                ]
            }
        }
        ```

        格式：`[class_id, x1, y1, x2, y2, x3, y3, ...]`

    === "姿态 (Pose)"

        ```json
        {
            "type": "image",
            "file": "image1.jpg",
            "url": "https://www.url.com/path/to/image1.jpg",
            "width": 640,
            "height": 480,
            "split": "train",
            "annotations": {
                "pose": [
                    [0, 0.523, 0.376, 0.283, 0.418, 0.374, 0.169, 2, 0.364, 0.178, 2],
                    [0, 0.735, 0.298, 0.193, 0.337, 0.412, 0.225, 2, 0.408, 0.231, 2]
                ]
            }
        }
        ```

        格式：`[class_id, x_center, y_center, width, height, x1, y1, v1, x2, y2, v2, ...]`

        关键点以 `(x, y, v)` 三元组的形式跟随在边界框之后，其中 `v` 为可见性：0=未标注，1=已标注但被遮挡，2=已标注且可见。关键点数量因数据集而异（例如，COCO 姿态有 17 个关键点 = 边界框后有 51 个值）。

    === "旋转检测 (OBB)"

        ```json
        {
            "type": "image",
            "file": "image1.jpg",
            "url": "https://www.url.com/path/to/image1.jpg",
            "width": 640,
            "height": 480,
            "split": "train",
            "annotations": {
                "obb": [
                    [0, 0.480, 0.352, 0.568, 0.356, 0.572, 0.400, 0.484, 0.396],
                    [1, 0.711, 0.274, 0.759, 0.278, 0.755, 0.322, 0.707, 0.318]
                ]
            }
        }
        ```

        格式：`[class_id, x1, y1, x2, y2, x3, y3, x4, y4]`

        四个角点按顺时针顺序定义旋转边界框，从左上角开始。所有坐标已归一化（0-1）。

    === "分类 (Classify)"

        ```json
        {
            "type": "image",
            "file": "image1.jpg",
            "url": "https://www.url.com/path/to/image1.jpg",
            "width": 640,
            "height": 480,
            "split": "train",
            "annotations": {
                "classification": [0]
            }
        }
        ```

        格式：`[class_id]`

#### 使用示例

要使用 NDJSON 数据集训练 YOLO26，只需指定 `.ndjson` 文件的路径：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")

        # Train using NDJSON dataset
        results = model.train(data="path/to/dataset.ndjson", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training with NDJSON dataset
        yolo detect train data=path/to/dataset.ndjson model=yolo26n.pt epochs=100 imgsz=640
        ```

#### NDJSON 格式的优势

- **单文件**：所有数据集信息包含在一个文件中
- **流式处理**：可以逐行处理大型数据集，无需将全部内容加载到内存中
- **云集成**：支持远程图像 URL，用于基于云的训练
- **可扩展**：易于添加自定义元数据字段
- **版本控制**：单文件格式与 git 和版本控制系统配合良好

## 支持的数据集

以下是支持的数据集列表及每个数据集的简要描述：

- [African-wildlife](african-wildlife.md)：包含非洲野生动物图像的数据集，包括水牛、大象、犀牛和斑马。
- [Argoverse](argoverse.md)：包含来自城市环境的 3D 跟踪和运动预测数据的数据集，具有丰富的标注信息。
- [Brain-tumor](brain-tumor.md)：用于检测脑肿瘤的数据集，包含 MRI 或 CT 扫描图像，详细描述了肿瘤的存在、位置和特征。
- [COCO](coco.md)：Common Objects in Context (COCO) 是一个大规模[目标检测](https://www.ultralytics.com/glossary/object-detection)、分割和图像描述数据集，包含 80 个对象类别。
- [COCO8](coco8.md)：COCO 训练集和验证集各前 4 张图像的较小子集，适合快速测试。
- [COCO8-Grayscale](coco8-grayscale.md)：通过将 RGB 转换为灰度创建的 COCO8 灰度版本，适用于单通道模型评估。
- [COCO8-Multispectral](coco8-multispectral.md)：通过插值 RGB 波长创建的 10 通道多光谱 COCO8 版本，适用于光谱感知模型评估。
- [COCO12-Formats](coco12-formats.md)：包含 12 张图像、覆盖所有支持的图像格式（AVIF、BMP、DNG、HEIC、JP2、JPEG、JPG、MPO、PNG、TIF、TIFF、WebP）的测试数据集，用于验证图像加载管线。
- [COCO128](coco128.md)：COCO 训练集和验证集各前 128 张图像的较小子集，适合测试。
- [Construction-PPE](construction-ppe.md)：包含建筑工地工人图像的数据集，标注了安全装备如头盔、背心、手套、靴子和护目镜，包括缺失装备标注如 no_helmet、no_googles，用于现实世界的合规监测。
- [Global Wheat 2020](globalwheat2020.md)：包含麦穗图像的数据集，用于 Global Wheat Challenge 2020。
- [HomeObjects-3K](homeobjects-3k.md)：包含床、椅子、电视等室内家居物品的数据集，非常适合智能家居自动化、机器人、增强现实和房间布局分析等应用。
- [KITTI](kitti.md)：包含真实驾驶场景的数据集，具有立体视觉、LiDAR 和 GPS/IMU 数据，此处用于**2D 目标检测**任务，如识别城市、乡村和高速公路环境中的汽车、行人和骑行者。
- [LVIS](lvis.md)：大规模目标检测、分割和图像描述数据集，包含 1203 个对象类别。
- [Medical-pills](medical-pills.md)：包含药丸图像的数据集，标注用于药品质量保证、药丸分类和合规监管等应用。
- [Objects365](objects365.md)：高质量大规模目标检测数据集，包含 365 个对象类别和超过 60 万张标注图像。
- [OpenImagesV7](open-images-v7.md)：Google 提供的综合数据集，包含 170 万张训练图像和 4.2 万张验证图像。
- [Roboflow 100](roboflow-100.md)：多样化的目标检测基准，包含 100 个数据集，涵盖七个图像领域，用于全面的模型评估。
- [Signature](signature.md)：包含各种文档图像的数据集，带有标注的签名，支持文档验证和欺诈检测研究。
- [SKU-110K](sku-110k.md)：包含零售环境中密集目标检测的数据集，超过 1.1 万张图像和 170 万个[边界框](https://www.ultralytics.com/glossary/bounding-box)。
- [TT100K](tt100k.md)：探索 Tsinghua-Tencent 100K (TT100K) 交通标志数据集，包含 10 万张街景图像和 3 万多个标注的交通标志，用于鲁棒的检测和分类。
- [VisDrone](visdrone.md)：包含无人机拍摄图像的目标检测和多目标跟踪数据集，超过 1 万张图像和视频序列。
- [VOC](voc.md)：Pascal Visual Object Classes (VOC) 数据集，用于目标检测和分割，包含 20 个对象类别和超过 1.1 万张图像。
- [xView](xview.md)：用于俯视图像目标检测的数据集，包含 60 个对象类别和超过 100 万个标注对象。

### 添加您自己的数据集

如果您有自己的数据集并希望将其用于 Ultralytics YOLO 格式训练检测模型，请确保它遵循上面"Ultralytics YOLO 格式"下指定的格式。将您的标注转换为所需格式，并在 YAML 配置文件中指定路径、类别数量和类别名称。

## 移植或转换标签格式

### COCO 数据集格式转换为 YOLO 格式

您可以使用以下代码片段轻松地将流行的 [COCO 数据集](coco.md)格式标签转换为 YOLO 格式：

!!! example

    === "Python"

        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco(labels_dir="path/to/coco/annotations/")
        ```

此转换工具可用于将 COCO 数据集或任何 COCO 格式的数据集转换为 Ultralytics YOLO 格式。该过程将基于 JSON 的 COCO 标注转换为更简单的基于文本的 YOLO 格式，使其与 [Ultralytics YOLO 模型](../../models/yolo26.md)兼容。

请务必仔细检查您要使用的数据集是否与您的模型兼容，并遵循必要的格式约定。格式正确的数据集对于成功训练目标检测模型至关重要。

## 常见问题

### Ultralytics YOLO 数据集格式是什么？如何构建它？

Ultralytics YOLO 格式是一种结构化的配置，用于在训练项目中定义数据集。它涉及设置训练、验证和测试图像及对应标签的路径。例如：

```yaml
--8<-- "ultralytics/cfg/datasets/coco8.yaml"
```

标签保存在 `*.txt` 文件中，每张图像一个文件，格式为 `class x_center y_center width height`，坐标已归一化。有关详细指南，请参见 [COCO8 数据集示例](coco8.md)。

### 如何将 COCO 数据集转换为 YOLO 格式？

您可以使用 [Ultralytics 转换工具](../../reference/data/converter.md)将 COCO 数据集转换为 YOLO 格式。以下是一个快速方法：

```python
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="path/to/coco/annotations/")
```

此代码将把您的 COCO 标注转换为 YOLO 格式，实现与 Ultralytics YOLO 模型的无缝集成。有关更多详细信息，请访问[移植或转换标签格式](#移植或转换标签格式)部分。

### Ultralytics YOLO 支持哪些目标检测数据集？

Ultralytics YOLO 支持广泛的数据集，包括：

- [Argoverse](argoverse.md)
- [COCO](coco.md)
- [LVIS](lvis.md)
- [COCO8](coco8.md)
- [Global Wheat 2020](globalwheat2020.md)
- [Objects365](objects365.md)
- [OpenImagesV7](open-images-v7.md)

每个数据集页面提供有关结构和用法的详细信息，专为高效的 YOLO26 训练而定制。在[支持的数据集](#支持的数据集)部分探索完整列表。

### 如何使用我的数据集开始训练 YOLO26 模型？

要开始训练 YOLO26 模型，请确保您的数据集格式正确并在 YAML 文件中定义了路径。使用以下脚本开始训练：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")  # Load a pretrained model
        results = model.train(data="path/to/your_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=path/to/your_dataset.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

有关使用不同模式（包括 CLI 命令）的更多详细信息，请参阅[使用示例](#使用示例)部分。

### 在哪里可以找到使用 Ultralytics YOLO 进行目标检测的实用示例？

Ultralytics 为 YOLO26 在多样化应用中的使用提供了众多示例和实用指南。要获得全面概述，请访问 [Ultralytics 博客](https://www.ultralytics.com/blog)，您可以找到案例研究、详细教程和社区故事，展示 YOLO26 在目标检测、分割等方面的应用。有关具体示例，请查看文档中的[使用](../../modes/predict.md)部分。