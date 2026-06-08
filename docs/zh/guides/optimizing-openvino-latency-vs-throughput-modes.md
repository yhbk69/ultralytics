---
comments: true
description: 探索如何使用 Intel OpenVINO 工具包提升 Ultralytics YOLO 模型性能，高效优化延迟与吞吐量。
keywords: Ultralytics YOLO, OpenVINO 优化, 深度学习, 模型推理, 吞吐量优化, 延迟优化, AI 部署, Intel OpenVINO, 性能调优
---

# YOLO 的 OpenVINO 推理优化

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/openvino-ecosystem.avif" alt="OpenVINO Ecosystem">

## 简介

在部署[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型时，尤其是像 Ultralytics YOLO 这类[目标检测](https://www.ultralytics.com/glossary/object-detection)模型，实现最优性能至关重要。本指南将深入探讨如何利用 [Intel OpenVINO 工具包](https://docs.ultralytics.com/integrations/openvino)优化推理，重点关注延迟与吞吐量。无论你是在开发消费级应用还是大规模部署，理解并应用这些优化策略都能确保你的模型在各种设备上高效运行。

## 优化延迟

延迟优化对于需要单个输入得到单个模型即时响应的应用（典型消费场景）至关重要，其目标是最小化输入与推理结果之间的时延。然而，实现低延迟需要审慎考量，尤其是在运行并发推理或管理多个模型时。

### 延迟优化的关键策略：

- **每设备单次推理：** 实现低延迟最简单的方法是将每个设备同时运行的推理限制为一次。额外的并发往往会导致延迟增加。
- **利用子设备：** 如多路 CPU 或多 Tile GPU 这类设备，可以利用其内部子设备在延迟增幅极小的情况下执行多个请求。
- **OpenVINO 性能提示：** 在模型编译期间使用 OpenVINO 的 `ov::hint::PerformanceMode::LATENCY` 作为 `ov::hint::performance_mode` 属性，可以简化性能调优，提供一种与设备无关且面向未来的方法。

### 管理首次推理延迟：

- **模型缓存：** 为减少模型加载与编译时间对延迟的影响，应尽可能使用模型缓存。在无法使用缓存的场景下，CPU 通常提供最快的模型加载时间。
- **模型映射与读取：** 为缩短加载时间，OpenVINO 用映射替代了模型读取。但如果模型位于可移动或网络驱动器上，可考虑使用 `ov::enable_mmap(false)` 切换回读取模式。
- **AUTO 设备选择：** 该模式先在 CPU 上开始推理，待加速器就绪后无缝切换，从而降低首次推理延迟。

## 优化吞吐量

吞吐量优化对于同时处理大量推理请求、最大化[资源利用率](https://www.ultralytics.com/blog/measuring-ai-performance-to-weigh-the-impact-of-your-innovations)而又不过分牺牲单请求性能的场景至关重要。

### 吞吐量优化方法：

1. **OpenVINO 性能提示：** 使用性能提示跨设备提升吞吐量的高层次、面向未来的方法。

    ```python
    import openvino.properties.hint as hints

    config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
    compiled_model = core.compile_model(model, "GPU", config)
    ```

2. **显式批处理与流：** 一种更细粒度的方法，涉及显式批处理和使用流进行高级性能调优。

### 设计面向吞吐量的应用：

为最大化吞吐量，应用应当：

- 并行处理输入，充分利用设备能力。
- 将数据流分解为并发推理请求，调度为并行执行。
- 使用带回调的异步 API 保持效率，避免设备空闲。

### 多设备执行：

OpenVINO 的多设备模式可自动在多个设备间均衡推理请求，无需应用层进行设备管理，从而简化吞吐量扩展。

## 实际性能提升

对 Ultralytics YOLO 模型实施 OpenVINO 优化可以带来显著的性能提升。如[基准测试](https://docs.ultralytics.com/integrations/openvino#openvino-yolov8-benchmarks)所示，在 Intel CPU 上推理速度可达原来的 3 倍，在 Intel 全系列硬件（包括集成 GPU、独立 GPU 和 VPU）上还可获得更大的加速效果。

例如，在 Intel Xeon CPU 上运行 YOLOv8 模型时，OpenVINO 优化版本在每张图像推理时间上持续优于 PyTorch 版本，且不牺牲[准确率](https://www.ultralytics.com/glossary/accuracy)。

## 实践操作

要将 Ultralytics YOLO 模型导出并优化为 OpenVINO 格式，可使用[导出](https://docs.ultralytics.com/modes/export)功能：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo26n.pt")

# 将模型导出为 OpenVINO 格式
model.export(format="openvino", half=True)  # 使用 FP16 精度导出
```

导出后，可使用优化后的模型运行推理：

```python
# 加载 OpenVINO 模型
ov_model = YOLO("yolo26n_openvino_model/")

# 使用延迟性能提示运行推理
results = ov_model("path/to/image.jpg", verbose=True)
```

## 结论

使用 OpenVINO 针对延迟和吞吐量优化 Ultralytics YOLO 模型，可以显著提升应用性能。通过审慎应用本指南中概述的策略，开发者可以确保模型高效运行，满足各种部署场景的需求。请记住，选择优化延迟还是吞吐量，取决于你的具体应用需求以及部署环境的特性。

如需更详细的技术信息和最新更新，请参阅 [OpenVINO 文档](https://docs.openvino.ai/2024/index.html)和 [Ultralytics YOLO 仓库](https://github.com/ultralytics/ultralytics)。这些资源提供了深入的指南、教程和社区支持，帮助你充分发挥深度学习模型的潜力。

---

确保模型达到最佳性能不仅仅是调整配置，更在于理解应用需求并做出明智决策。无论你是为[实时响应](https://www.ultralytics.com/blog/real-time-inferences-in-vision-ai-solutions-are-making-an-impact)优化，还是为大规模处理最大化吞吐量，Ultralytics YOLO 模型与 OpenVINO 的强强联合都为开发者部署高性能 AI 解决方案提供了强大的工具包。

## 常见问题

### 如何使用 OpenVINO 优化 Ultralytics YOLO 模型的低延迟？

优化 Ultralytics YOLO 模型的低延迟涉及以下几个关键策略：

1. **每设备单次推理：** 每个设备一次只进行一次推理，以最小化延迟。
2. **利用子设备：** 利用多路 CPU 或多 Tile GPU 等设备，它们能以最小的延迟增加处理多个请求。
3. **OpenVINO 性能提示：** 在模型编译期间使用 OpenVINO 的 `ov::hint::PerformanceMode::LATENCY`，实现简化的、与设备无关的调优。

有关优化延迟的更多实用技巧，请参阅本指南的[优化延迟部分](#优化延迟)。

### 为什么要使用 OpenVINO 优化 Ultralytics YOLO 的吞吐量？

OpenVINO 通过最大化设备资源利用率来提升 Ultralytics YOLO 模型的吞吐量，且不牺牲性能。主要优势包括：

- **性能提示：** 跨设备简单、高层次的性能调优。
- **显式批处理与流：** 针对高级性能的精细调优。
- **多设备执行：** 自动推理负载均衡，简化应用层管理。

配置示例：

```python
import openvino.properties.hint as hints

config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
compiled_model = core.compile_model(model, "GPU", config)
```

在详细指南的[吞吐量优化部分](#优化吞吐量)了解更多吞吐量优化内容。

### OpenVINO 减少首次推理延迟的最佳实践是什么？

为减少首次推理延迟，可考虑以下实践：

1. **模型缓存：** 使用模型缓存来减少加载和编译时间。
2. **模型映射与读取：** 默认使用映射（`ov::enable_mmap(true)`），但如果模型位于可移动或网络驱动器上，则切换为读取（`ov::enable_mmap(false)`）。
3. **AUTO 设备选择：** 利用 AUTO 模式先从 CPU 推理开始，再无缝切换到加速器。

关于管理首次推理延迟的详细策略，请参阅[管理首次推理延迟部分](#管理首次推理延迟)。

### 如何在使用 Ultralytics YOLO 和 OpenVINO 时平衡延迟与吞吐量的优化？

平衡延迟与吞吐量的优化需要理解你的应用需求：

- **延迟优化：** 适用于需要即时响应的实时应用（如消费级应用）。
- **吞吐量优化：** 最适合大量并发推理的场景，最大化资源利用（如大规模部署）。

使用 OpenVINO 的高层性能提示和多设备模式有助于找到合适的平衡。根据你的具体需求选择合适的 [OpenVINO 性能提示](https://docs.ultralytics.com/integrations/openvino#openvino-performance-hints)。

### 除了 OpenVINO，Ultralytics YOLO 模型还可以与其他 AI 框架配合使用吗？

可以，Ultralytics YOLO 模型高度通用，可与多种 AI 框架集成。可选方案包括：

- **TensorRT：** 用于 NVIDIA GPU 优化，请参阅 [TensorRT 集成指南](https://docs.ultralytics.com/integrations/tensorrt)。
- **CoreML：** 用于 Apple 设备，请参阅 [CoreML 导出说明](https://docs.ultralytics.com/integrations/coreml)。
- **[TensorFlow](https://www.ultralytics.com/glossary/tensorflow).js：** 用于 Web 和 Node.js 应用，请参阅 [TF.js 转换指南](https://docs.ultralytics.com/integrations/tfjs)。

在 [Ultralytics 集成页面](https://docs.ultralytics.com/integrations)探索更多集成方案。
