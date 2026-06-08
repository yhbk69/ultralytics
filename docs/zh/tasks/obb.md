---
comments: true
description: 了解如何使用 YOLO26 OBB 模型进行带旋转角度的更高精度目标检测。轻松学习、训练、验证和导出 OBB 模型。
keywords: 定向边界框, OBB, 目标检测, YOLO26, Ultralytics, DOTAv1, 模型训练, 模型导出, AI, 机器学习
model_name: yolo26n-obb
---

# 定向边界框 [目标检测](https://www.ultralytics.com/glossary/object-detection)

<!-- obb task poster -->

定向目标检测比标准目标检测更进一步，通过引入额外的角度信息来更精确地在图像中定位目标。

定向目标检测器的输出是一组旋转边界框，它们精确地包围图像中的目标，同时输出每个框的类别标签和置信度分数。当目标以不同角度出现时，定向边界框尤其有用，例如在航拍图像中，传统的轴对齐边界框可能会包含不必要的背景区域。

<!-- youtube video link for obb task -->

!!! tip

    YOLO26 OBB 模型使用 `-obb` 后缀，即 `yolo26n-obb.pt`，并在 [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml) 数据集上进行预训练。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/128JhhR2DlM"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO26 定向边界框 (OBB) 检测和跟踪目标 | 船舶跟踪 🚢
</p>

## 可视化示例

|                                               使用 OBB 进行船舶检测                                                |                                                使用 OBB 进行车辆检测                                                |
| :-------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: |
| ![使用 OBB 进行船舶检测](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ships-detection-using-obb.avif) | ![使用 OBB 进行车辆检测](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/vehicle-detection-using-obb.avif) |

## [模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

以下是 YOLO26 预训练的 OBB 模型，这些模型在 [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml) 数据集上进行了预训练。

[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)会在首次使用时从最新的 Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)自动下载。

{% include "macros/yolo-obb-perf.md" %}

- **mAP<sup>test</sup>** 值为单模型多尺度在 [DOTAv1](https://captain-whu.github.io/DOTA/index.html) 数据集上的结果。<br>可通过 `yolo val obb data=DOTAv1.yaml device=0 split=test` 复现，并将合并结果提交至 [DOTA 评估](https://captain-whu.github.io/DOTA/evaluation.html)。
- **速度** 是在 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例上对 DOTAv1 验证集图像取平均值得到的结果。<br>可通过 `yolo val obb data=DOTAv1.yaml batch=1 device=0|cpu` 复现。
- **参数量** 和 **FLOPs** 值为经过 `model.fuse()` 融合后的模型结果，该操作会合并 Conv 和 BatchNorm 层，对于端到端模型还会移除辅助的一对多检测头。预训练检查点保留完整的训练架构，可能显示更高的数值。

## 训练

在 DOTA8 数据集上训练 YOLO26n-obb 模型，训练 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640。完整的可用参数列表请参见 [配置](../usage/cfg.md) 页面。

!!! note

    OBB 角度被限制在 **0–90 度**范围内（不含 90 度）。不支持 90 度及以上的角度。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-obb.yaml")  # 从 YAML 构建新模型
        model = YOLO("yolo26n-obb.pt")  # 加载预训练模型（推荐用于训练）
        model = YOLO("yolo26n-obb.yaml").load("yolo26n-obb.pt")  # 从 YAML 构建并迁移权重

        # 训练模型
        results = model.train(data="dota8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从 YAML 构建新模型并从头开始训练
        yolo obb train data=dota8.yaml model=yolo26n-obb.yaml epochs=100 imgsz=640

        # 从预训练的 *.pt 模型开始训练
        yolo obb train data=dota8.yaml model=yolo26n-obb.pt epochs=100 imgsz=640

        # 从 YAML 构建新模型，将预训练权重迁移到该模型并开始训练
        yolo obb train data=dota8.yaml model=yolo26n-obb.yaml pretrained=yolo26n-obb.pt epochs=100 imgsz=640
        ```

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/uZ7SymQfqKI"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics 平台在 DOTA 数据集上训练 Ultralytics YOLO-OBB（定向边界框）模型
</p>

### 数据集格式

OBB 数据集格式的详细信息可以在 [数据集指南](../datasets/obb/index.md) 中找到。YOLO OBB 格式通过四个角点来定义边界框，坐标归一化到 0 到 1 之间，遵循以下结构。[Ultralytics 平台](https://platform.ultralytics.com) 支持使用专用的定向边界框绘制工具进行 OBB 标注：

```
class_index x1 y1 x2 y2 x3 y3 x4 y4
```

在内部，YOLO 以 `xywhr` 格式处理损失和输出，该格式表示[边界框](https://www.ultralytics.com/glossary/bounding-box)的中心点 (xy)、宽度、高度和旋转角度。

## 验证

在 DOTA8 数据集上验证已训练的 YOLO26n-obb 模型的[精度](https://www.ultralytics.com/glossary/accuracy)。无需额外参数，因为 `model` 会将训练时的 `data` 和参数作为模型属性保留。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-obb.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 验证模型
        metrics = model.val(data="dota8.yaml")  # 无需参数，数据集和设置已记录
        metrics.box.map  # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps  # 包含每个类别 mAP50-95(B) 的列表
        metrics.box.image_metrics  # 每张图像的指标字典，包含精确率、召回率、F1、TP、FP 和 FN
        ```

    === "CLI"

        ```bash
        yolo obb val model=yolo26n-obb.pt data=dota8.yaml         # 验证官方模型
        yolo obb val model=path/to/best.pt data=path/to/data.yaml # 验证自定义模型
        ```

## 预测

使用已训练的 YOLO26n-obb 模型对图像进行预测。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-obb.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 使用模型进行预测
        results = model("https://ultralytics.com/images/boats.jpg")  # 对图像进行预测

        # 访问结果
        for result in results:
            xywhr = result.obb.xywhr  # 中心 x, 中心 y, 宽度, 高度, 角度（弧度）
            xyxyxyxy = result.obb.xyxyxyxy  # 4 点多边形格式
            names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # 每个框的类别名称
            confs = result.obb.conf  # 每个框的置信度分数
        ```

    === "CLI"

        ```bash
        yolo obb predict model=yolo26n-obb.pt source='https://ultralytics.com/images/boats.jpg'  # 使用官方模型预测
        yolo obb predict model=path/to/best.pt source='https://ultralytics.com/images/boats.jpg' # 使用自定义模型预测
        ```

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/5XYdm5CYODA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO-OBB 检测和跟踪储油罐 | 定向边界框 | DOTA
</p>

完整的 `predict` 模式详情请参见 [预测](../modes/predict.md) 页面。

## 导出

将 YOLO26n-obb 模型导出为不同格式，如 ONNX、CoreML 等。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-obb.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义训练的模型

        # 导出模型
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-obb.pt format=onnx  # 导出官方模型
        yolo export model=path/to/best.pt format=onnx # 导出自定义训练的模型
        ```

YOLO26-obb 可用的导出格式如下表所示。您可以使用 `format` 参数导出为任何格式，例如 `format='onnx'` 或 `format='engine'`。您可以直接对导出的模型进行预测或验证，例如 `yolo predict model=yolo26n-obb.onnx`。导出完成后会显示模型的使用示例。

{% include "macros/export-table.md" %}

完整的 `export` 详情请参见 [导出](../modes/export.md) 页面。

## 实际应用

YOLO26 的 OBB 检测在各行各业有着广泛的实际应用：

- **海事与港口管理**：检测不同角度的船舶和船只，用于[船队管理](https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-obb-object-detection)和监控。
- **城市规划**：从航拍图像中分析建筑和基础设施。
- **农业**：通过无人机影像监测农作物和农业设备。
- **能源行业**：检测不同朝向的太阳能板和风力涡轮机。
- **交通运输**：从不同视角跟踪道路和停车场中的车辆。

这些应用受益于 OBB 能够以任意角度精确拟合目标，相比传统边界框提供更准确的检测结果。

## 常见问题

### 什么是定向边界框 (OBB)，它与普通边界框有何不同？

定向边界框 (OBB) 包含额外的角度信息，以提高图像中目标定位的精度。与轴对齐矩形的普通边界框不同，OBB 可以旋转以更好地贴合目标的朝向。这对于需要精确定位的应用特别有用，例如航拍或卫星图像（[数据集指南](../datasets/obb/index.md)）。

### 如何使用自定义数据集训练 YOLO26n-obb 模型？

要使用自定义数据集训练 YOLO26n-obb 模型，请参考以下 Python 或 CLI 示例：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo26n-obb.pt")

        # 训练模型
        results = model.train(data="path/to/custom_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo obb train data=path/to/custom_dataset.yaml model=yolo26n-obb.pt epochs=100 imgsz=640
        ```

更多训练参数请参见 [配置](../usage/cfg.md) 章节。

### 训练 YOLO26-OBB 模型可以使用哪些数据集？

YOLO26-OBB 模型在 [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml) 等数据集上进行了预训练，但您可以使用任何按 OBB 格式标注的数据集。关于 OBB 数据集格式的详细信息，请参见 [数据集指南](../datasets/obb/index.md)。

### 如何将 YOLO26-OBB 模型导出为 ONNX 格式？

使用 Python 或 CLI 将 YOLO26-OBB 模型导出为 ONNX 格式非常简单：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-obb.pt")

        # 导出模型
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-obb.pt format=onnx
        ```

更多导出格式和详情请参见 [导出](../modes/export.md) 页面。

### 如何验证 YOLO26n-obb 模型的精度？

要验证 YOLO26n-obb 模型，可以使用以下 Python 或 CLI 命令：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-obb.pt")

        # 验证模型
        metrics = model.val(data="dota8.yaml")
        ```

    === "CLI"

        ```bash
        yolo obb val model=yolo26n-obb.pt data=dota8.yaml
        ```

完整的验证详情请参见 [验证](../modes/val.md) 章节。
