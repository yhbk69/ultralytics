---
comments: true
description: 了解 YOLO26 端到端无 NMS 检测的工作原理、对部署流程的影响、支持的导出格式以及如何从 YOLOv8 或 YOLO11 迁移。
keywords: YOLO26, 端到端检测, 无 NMS 推理, 模型导出, 部署指南, 目标检测, Ultralytics, YOLOv8 迁移, YOLO11 迁移, ONNX, TensorRT, CoreML, 后处理, 计算机视觉
---

# 理解 Ultralytics YOLO26 中的端到端检测

## 简介

如果你正在从 [YOLOv8](../models/yolov8.md) 或 [YOLO11](../models/yolo11.md) 等早期模型升级到 [YOLO26](../models/yolo26.md)，你会注意到最大的变化之一是移除了[非极大值抑制](https://www.ultralytics.com/glossary/non-maximum-suppression-nms)（NMS）。传统的 YOLO 模型会产生数千个重叠的预测，需要额外的 NMS 后处理步骤来筛选出最终的检测结果。这会增加延迟、使导出图变得复杂，并可能在不同硬件平台上表现出不一致的行为。

YOLO26 采用了不同的方法。它直接从模型输出最终检测结果——无需外部过滤。这被称为**端到端[目标检测](https://www.ultralytics.com/glossary/object-detection)**，在所有 YOLO26 模型中默认启用。其结果是更简单的部署流程、更低的延迟，以及**在 CPU 上高达 43% 的推理加速**。

本指南将介绍其中的变化、是否需要更新代码、哪些导出格式支持端到端推理，以及如何从旧版 YOLO 模型平滑迁移。

关于这一架构转变背后的动因，请参阅 [Ultralytics 关于 YOLO26 移除 NMS 的博客文章](https://www.ultralytics.com/blog/why-ultralytics-yolo26-removes-nms-and-how-that-changes-deployment)。

!!! summary "快速摘要"

    - **使用 Ultralytics API 或 CLI？** 无需任何更改——只需将模型名称替换为 `yolo26n.pt`。
    - **使用自定义推理代码（ONNX Runtime、TensorRT 等）？** 更新你的后处理逻辑——检测输出现在为 `(N, 300, 6)` 的 `xyxy` 格式，无需 NMS。其他任务会附加额外的数据（掩码系数、关键点或角度）。
    - **导出？** 大多数格式原生支持端到端输出。但是，少数格式（NCNN、RKNN、PaddlePaddle、ExecuTorch、IMX 和 Edge TPU）由于不支持相关算子（如 `torch.topk`），会自动回退到传统输出。

## 端到端检测的工作原理

YOLO26 在[训练](../modes/train.md)期间使用**双头架构**。两个头共享相同的骨干网络和颈部网络，但以不同的方式产生输出：

| 头                         | 用途                   | 检测输出            | 后处理               |
| -------------------------- | ---------------------- | ------------------- | -------------------- |
| **一对一**（默认）         | 端到端推理             | `(N, 300, 6)`       | 仅置信度阈值过滤     |
| **一对多**                 | 传统 YOLO 输出         | `(N, nc + 4, 8400)` | 需要 NMS             |

以上形状适用于[检测](../tasks/detect.md)任务。其他任务在一对一输出上按每个检测附加额外数据：

| 任务                               | 端到端输出                                   | 额外数据                              |
| ---------------------------------- | -------------------------------------------- | ------------------------------------- |
| [检测](../tasks/detect.md)         | `(N, 300, 6)`                                | —                                     |
| [分割](../tasks/segment.md)        | `(N, 300, 6 + nm)` + 原型 `(N, nm, H, W)`     | `nm` 个掩码系数（默认 32）            |
| [姿态](../tasks/pose.md)           | `(N, 300, 57)`                               | 17 个关键点 × 3（x, y, 可见性）       |
| [OBB](../tasks/obb.md)             | `(N, 300, 7)`                                | 旋转角度                              |

在训练期间，两个头同时运行——一对多头提供更丰富的学习信号，而一对一头则学习产生干净、无重叠的预测。在[推理](../modes/predict.md)和[导出](../modes/export.md)期间，默认仅激活**一对一头**，每张图像最多产生 300 个检测结果，格式为 `[x1, y1, x2, y2, confidence, class_id]`。

当你调用 `model.fuse()` 时，它会融合 Conv + BatchNorm 层以加快推理速度，并且在端到端模型上还会移除一对多头——从而减小模型大小和 FLOPs。有关双头架构的更多细节，请参见 [YOLO26 模型页面](../models/yolo26.md)。

## 我需要更改代码吗？

### 使用 Ultralytics Python API 或 CLI

**无需更改。** 如果你使用标准的 [Ultralytics Python API](../usage/python.md) 或 [CLI](../usage/cli.md)，一切都会自动运行——[预测](../modes/predict.md)、[验证](../modes/val.md)和[导出](../modes/export.md)都开箱即用地支持端到端模型。

!!! example "使用 Ultralytics API 无需代码更改"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO26 模型
        model = YOLO("yolo26n.pt")

        # 预测——无需 NMS 步骤，无需代码更改
        results = model.predict("image.jpg")
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo26n.pt source=image.jpg
        ```

### 使用自定义推理代码

**是的，输出格式不同。** 如果你为 [YOLOv8](../models/yolov8.md) 或 [YOLO11](../models/yolo11.md) 编写了自定义后处理逻辑（例如，在使用 [ONNX Runtime](../integrations/onnx.md) 或 [TensorRT](../integrations/tensorrt.md) 进行推理时），你需要更新它以处理新的输出形状：

|                      | YOLOv8 / YOLO11                            | YOLO26（端到端）                                                |
| -------------------- | ------------------------------------------ | --------------------------------------------------------------- |
| **检测输出**         | `(N, nc + 4, 8400)`                        | `(N, 300, 6)`                                                   |
| **边界框格式**       | `xywh`（中心 x, 中心 y, 宽度, 高度）        | `xyxy`（左上角 x, 左上角 y, 右下角 x, 右下角 y）                 |
| **布局**             | 每个锚点的边界框坐标 + 类别分数             | `[x1, y1, x2, y2, conf, class_id]`                              |
| **需要 NMS**         | 是                                         | 否                                                              |
| **后处理**           | NMS + 置信度过滤                            | 仅置信度过滤                                                    |

对于[分割](../tasks/segment.md)、[姿态](../tasks/pose.md)和 [OBB](../tasks/obb.md) 任务，YOLO26 为每个检测附加任务特定数据——参见上方的[输出形状表](#端到端检测的工作原理)。

其中 `N` 是[批次大小](https://www.ultralytics.com/glossary/batch-size)，`nc` 是类别数量（例如，[COCO](../datasets/detect/coco.md) 为 80）。

使用端到端模型，后处理变得简单得多——例如，使用 [ONNX Runtime](../integrations/onnx.md) 时：

```python
import onnxruntime as ort

# 加载并运行导出的端到端模型
session = ort.InferenceSession("yolo26n.onnx")
output = session.run(None, {session.get_inputs()[0].name: input_tensor})

# 端到端输出：(batch, 300, 6) → [x1, y1, x2, y2, confidence, class_id]
detections = output[0][0]  # 批次中的第一张图像
detections = detections[detections[:, 4] > conf_threshold]  # 置信度过滤——就这么简单！
```

### 切换到一对多头

如果你需要传统的 YOLO 输出格式（例如，为了复用现有的基于 NMS 的后处理代码），你可以随时通过设置 `end2end=False` 切换到一对多头：

!!! example "使用一对多头获取传统的基于 NMS 的输出"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # 使用 NMS 进行预测（传统行为）
        results = model.predict("image.jpg", end2end=False)

        # 使用 NMS 进行验证
        metrics = model.val(data="coco.yaml", end2end=False)

        # 不使用端到端导出
        model.export(format="onnx", end2end=False)
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo26n.pt source=image.jpg end2end=False
        yolo val model=yolo26n.pt data=coco.yaml end2end=False
        yolo export model=yolo26n.pt format=onnx end2end=False
        ```

## 导出格式兼容性

大多数[导出格式](../modes/export.md#export-formats)开箱即用地支持端到端推理，包括 [ONNX](../integrations/onnx.md)、[TensorRT](../integrations/tensorrt.md)、[CoreML](../integrations/coreml.md)、[OpenVINO](../integrations/openvino.md)、[TFLite](../integrations/tflite.md)、[TF.js](../integrations/tfjs.md) 和 [MNN](../integrations/mnn.md)。

以下格式**不支持**端到端，并会自动回退到一对多头：[NCNN](../integrations/ncnn.md)、[RKNN](../integrations/rockchip-rknn.md)、[PaddlePaddle](../integrations/paddlepaddle.md)、[ExecuTorch](../integrations/executorch.md)、[IMX](../integrations/sony-imx500.md) 和 [Edge TPU](../integrations/edge-tpu.md)。

!!! tip "当端到端不受支持时会发生什么"

    当你导出到这些格式时，Ultralytics 会自动切换到一对多头并记录一条警告——无需手动干预。这意味着对于这些格式，**你的推理流程中需要 NMS**，就像使用 [YOLOv8](../models/yolov8.md) 或 [YOLO11](../models/yolo11.md) 时一样。

!!! note "TensorRT + INT8"

    [TensorRT](../integrations/tensorrt.md) 支持端到端，但在 TensorRT ≤10.3.0 上使用 `int8=True` 导出时会**自动禁用**。

## 精度与速度的权衡

端到端检测提供了显著的部署优势，对[精度](https://www.ultralytics.com/glossary/accuracy)的影响极小：

| 指标                     | 端到端（默认）            | 一对多 + NMS (`end2end=False`)  |
| ------------------------ | ------------------------- | ------------------------------- |
| **CPU 推理速度**         | 最高**提升 43%**          | 基准                            |
| **mAP 影响**             | 约低 0.5 mAP              | 达到或超过 YOLO11               |
| **后处理**               | 仅置信度过滤              | 完整的 NMS 流程                 |
| **部署复杂度**           | 最小                      | 需要实现 NMS                    |

对于大多数实际应用，约 0.5 [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) 的差异可以忽略不计，尤其是考虑到速度和简洁性上的收益。如果最高精度是你的首要目标，你始终可以使用 `end2end=False` 回退到一对多头。

有关所有模型规模（n, s, m, l, x）的详细基准测试，请参见 [YOLO26 性能指标](../models/yolo26.md#performance-metrics)。

## 从 YOLOv8 或 YOLO11 迁移

如果你正在将现有项目升级到 YOLO26，以下是一份快速检查清单以确保顺利过渡：

- **Ultralytics API / CLI 用户：** 无需更改——只需将模型名称更新为 `yolo26n.pt`（或 `yolo26n-seg.pt`、`yolo26n-pose.pt`、`yolo26n-obb.pt`）
- **自定义后处理代码：** 更新以处理新的输出形状——检测为 `(N, 300, 6)`，以及[分割](../tasks/segment.md)、[姿态](../tasks/pose.md)和 [OBB](../tasks/obb.md) 的任务特定数据。还需注意边界框格式从 `xywh` 变为 `xyxy`
- **导出流程：** 针对你的目标格式检查上方的[格式兼容性](#导出格式兼容性)部分
- **TensorRT + INT8：** 验证你的 TensorRT 版本 >10.3.0 以支持端到端
- **FP16 导出：** 如果你需要所有输出都为 FP16，请使用 `end2end=False` 导出——参见[为什么使用 half=True 和 end2end=True 导出时 output0 仍为 FP32](../modes/export.md#why-is-output0-fp32-when-exporting-with-halftrue-and-end2endtrue)
- **iOS / CoreML：** 端到端完全支持。如果你需要 Xcode Preview 支持，请使用 `end2end=False` 配合 `nms=True`
- **边缘设备（NCNN、RKNN）：** 这些格式会自动回退到一对多，因此请在设备端推理流程中包含 NMS

## 常见问题

### 我可以同时使用 end2end=True 和 nms=True 吗？

不可以。这些选项是互斥的。如果你在[导出](../modes/export.md)期间在端到端模型上设置 `nms=True`，它会被自动强制为 `nms=False` 并发出警告。端到端头已经在内部处理了重复过滤，因此不需要外部 NMS。

然而，`end2end=False` 结合 `nms=True` 是一个有效的配置——它将传统的 NMS 嵌入到导出图中。这对于 [CoreML](../integrations/coreml.md) 导出非常有用，因为它允许你直接在 Xcode 中使用 Preview 功能配合检测模型。

### max_det 参数在端到端模型中控制什么？

`max_det` 参数（默认值：300）设置一对一头每张图像最多可以输出的检测数量。你可以在推理或导出时调整它：

```python
model.predict("image.jpg", max_det=100)  # 更少的检测，稍快一些
model.export(format="onnx", max_det=500)  # 更多检测，适用于密集场景
```

请注意，默认的 YOLO26 检查点是在 `max_det=300` 下训练的。虽然你可以增加该值，但一对一头在训练期间被优化为最多产生 300 个干净的检测，因此超过该限制的检测质量可能较低。如果你每张图像需要超过 300 个检测，请考虑使用更高的 `max_det` 值重新训练。

### 我导出的 ONNX 模型输出 (1, 300, 6)——这是正确的吗？

是的，这是检测任务预期的端到端输出格式：[批次大小](https://www.ultralytics.com/glossary/batch-size)为 1，最多 300 个检测，每个检测包含 6 个值 `[x1, y1, x2, y2, confidence, class_id]`。只需按置信度阈值过滤即可——无需 NMS。

对于其他任务，输出形状不同：

| 任务         | 输出形状                              | 描述                                                               |
| ------------ | ------------------------------------- | ------------------------------------------------------------------ |
| 检测         | `(1, 300, 6)`                         | `[x1, y1, x2, y2, conf, class_id]`                                 |
| 分割         | `(1, 300, 38)` + `(1, 32, 160, 160)`  | 6 个边界框值 + 32 个掩码系数，外加一个原型掩码张量                  |
| 姿态         | `(1, 300, 57)`                        | 6 个边界框值 + 17 个关键点 × 3（x, y, 可见性）                     |
| OBB          | `(1, 300, 7)`                         | 6 个边界框值 + 1 个旋转角度                                        |

### 如何检查我导出的模型是否为端到端？

你可以使用 Ultralytics Python API 或直接检查导出的 ONNX 模型元数据来验证：

!!! example "检查模型是否为端到端"

    === "Python API"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.onnx")
        model.predict(verbose=False)  # 先运行预测以设置预测器
        print(model.predictor.model.end2end)  # 如果端到端已启用则为 True
        ```

    === "ONNX Runtime"

        ```python
        import onnxruntime as ort

        session = ort.InferenceSession("yolo26n.onnx")
        metadata = session.get_modelmeta().custom_metadata_map
        print(metadata.get("end2end"))  # 如果端到端已启用则为 'True'
        ```

或者，检查输出形状——端到端检测模型输出 `(1, 300, 6)`，而传统模型输出 `(1, nc + 4, 8400)`。对于其他任务的形状，请参见[输出形状常见问题](#我导出的-onnx-模型输出-1-300-6这是正确的吗)。

### 端到端是否支持分割、姿态和 OBB 任务？

是的。所有 YOLO26 任务变体——[检测](../tasks/detect.md)、[分割](../tasks/segment.md)、[姿态估计](../tasks/pose.md)和[旋转目标检测 (OBB)](../tasks/obb.md)——默认都支持端到端推理。`end2end=False` 回退选项同样适用于所有任务。

每个任务在基础检测输出之上附加任务特定数据：

| 任务         | 模型              | 端到端输出                                   |
| ------------ | ----------------- | -------------------------------------------- |
| 检测         | `yolo26n.pt`      | `(N, 300, 6)`                                |
| 分割         | `yolo26n-seg.pt`  | `(N, 300, 38)` + 原型 `(N, 32, 160, 160)`     |
| 姿态         | `yolo26n-pose.pt` | `(N, 300, 57)`                               |
| OBB          | `yolo26n-obb.pt`  | `(N, 300, 7)`                                |