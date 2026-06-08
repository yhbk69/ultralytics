---
comments: true
description: 了解如何使用 NVIDIA DALI 为 Ultralytics YOLO 模型实现 GPU 加速预处理。通过在 GPU 上运行 letterbox 调整尺寸、填充和归一化来消除 CPU 瓶颈，从而加快 TensorRT 和 Triton 部署。
keywords: NVIDIA DALI, GPU 预处理, Ultralytics, YOLO, YOLO26, TensorRT, Triton 推理服务器, letterbox, 推理优化, 深度学习, 计算机视觉, 部署, 视频处理, 批量推理, DALI 管道, CV-CUDA
---

# 使用 NVIDIA DALI 进行 GPU 加速预处理

## 简介

在生产环境中部署 [Ultralytics YOLO](../models/index.md) 模型时，[预处理](https://www.ultralytics.com/glossary/data-preprocessing) 往往成为瓶颈。虽然 [TensorRT](../integrations/tensorrt.md) 可以在几毫秒内完成模型 [推理](../modes/predict.md)，但基于 CPU 的预处理（调整尺寸、填充、归一化）每张图像可能需要 2-10ms，尤其是在高分辨率下。[NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html)（数据加载库）通过将整个预处理管道移至 GPU 来解决这一问题。

本指南将引导你构建能够精确复现 Ultralytics YOLO 预处理的 DALI 管道，并将其与 `model.predict()` 集成、处理视频流，以及与 [Triton 推理服务器](triton-inference-server.md) 进行端到端部署。

!!! tip "本指南适合谁？"

    本指南面向在生产环境中部署 YOLO 模型的工程师——这些环境中 CPU 预处理已被证实为瓶颈——通常是 NVIDIA GPU 上的 [TensorRT](../integrations/tensorrt.md) 部署、高吞吐量视频管道或 [Triton 推理服务器](triton-inference-server.md) 配置。如果你使用 `model.predict()` 运行标准推理且没有预处理瓶颈，默认的 CPU 管道完全够用。

!!! summary "快速概览"

    - **构建 DALI 管道？** 使用 `fn.resize(mode="not_larger")` + `fn.crop(out_of_bounds_policy="pad")` + `fn.crop_mirror_normalize` 在 GPU 上复现 YOLO 的 letterbox 预处理。
    - **与 Ultralytics 集成？** 将 DALI 输出作为 `torch.Tensor` 传递给 `model.predict()` —— Ultralytics 会自动跳过图像预处理。
    - **使用 Triton 部署？** 将 DALI 后端与 TensorRT 集成模型结合使用，实现零 CPU 预处理。

## 为什么使用 DALI 进行 YOLO 预处理

在典型的 YOLO 推理管道中，预处理步骤在 CPU 上运行：

1. **解码** 图像（JPEG/PNG）
2. **调整尺寸** 同时保持宽高比
3. **填充** 到目标尺寸（letterbox）
4. **归一化** 像素值从 `[0, 255]` 到 `[0, 1]`
5. **转换** 布局从 HWC 到 CHW

使用 DALI，所有这些操作都在 GPU 上运行，消除了 CPU 瓶颈。这在以下场景中特别有价值：

| 场景                                                              | DALI 的价值                                                                                        |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **快速 GPU 推理**                                                 | [TensorRT](../integrations/tensorrt.md) 引擎具有亚毫秒级推理速度，使 CPU 预处理成为主要成本        |
| **高分辨率输入**                                                  | 1080p 和 4K 视频流需要昂贵的调整尺寸操作                                                           |
| **大 [批次大小](https://www.ultralytics.com/glossary/batch-size)** | 服务器端推理并行处理大量图像                                                                       |
| **有限的 CPU 核心数**                                             | 边缘设备如 [NVIDIA Jetson](nvidia-jetson.md)，或每个 GPU 对应较少 CPU 核心的高密度 GPU 服务器      |

## 前提条件

!!! warning "仅限 Linux"

    NVIDIA DALI 仅支持 **Linux**。它在 Windows 或 macOS 上不可用。

安装所需软件包：

=== "CUDA 12.x"

    ```bash
    pip install ultralytics
    pip install --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda120
    ```

=== "CUDA 11.x"

    ```bash
    pip install ultralytics
    pip install --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda110
    ```

**要求：**

- NVIDIA GPU（计算能力 5.0+ / Maxwell 或更新）
- CUDA 11.0+ 或 12.0+
- Python 3.10-3.14
- Linux 操作系统

## 理解 YOLO 预处理

在构建 DALI 管道之前，理解 Ultralytics 在预处理过程中究竟做了什么会很有帮助。关键类是 [`ultralytics/data/augment.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py) 中的 `LetterBox`：

```python
from ultralytics.data.augment import LetterBox

letterbox = LetterBox(
    new_shape=(640, 640),  # 目标尺寸
    center=True,  # 居中图像（两侧均匀填充）
    stride=32,  # 步幅对齐
    padding_value=114,  # 灰色填充 (114, 114, 114)
)
```

[`ultralytics/engine/predictor.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py) 中的完整预处理管道执行以下步骤：

| 步骤 | 操作                  | CPU 函数                       | DALI 等价物                                    |
| ---- | --------------------- | ------------------------------ | ---------------------------------------------- |
| 1    | Letterbox 调整尺寸    | `cv2.resize`                   | `fn.resize(mode="not_larger")`                 |
| 2    | 居中填充              | `cv2.copyMakeBorder`           | `fn.crop(out_of_bounds_policy="pad")`          |
| 3    | BGR → RGB             | `im[..., ::-1]`                | `fn.decoders.image(output_type=types.RGB)`     |
| 4    | HWC → CHW + 归一化 /255 | `np.transpose` + `tensor / 255` | `fn.crop_mirror_normalize(std=[255,255,255])` |

letterbox 操作通过以下方式保持宽高比：

1. 计算缩放比例：`r = min(target_h / h, target_w / w)`
2. 调整尺寸到 `(round(w * r), round(h * r))`
3. 用灰色 (`114`) 填充剩余空间以达到目标尺寸
4. 居中图像，使填充均匀分布在两侧

## YOLO 的 DALI 管道

使用下面的居中管道作为默认参考。它匹配 Ultralytics `LetterBox(center=True)` 的行为，即标准 YOLO 推理使用的方式。

### 居中管道（推荐，匹配 Ultralytics LetterBox）

此版本精确复现了默认的 Ultralytics 居中填充预处理，匹配 `LetterBox(center=True)`：

!!! example "带居中填充的 DALI 管道（推荐）"

    ```python
    import nvidia.dali as dali
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types


    @dali.pipeline_def(batch_size=8, num_threads=4, device_id=0)
    def yolo_dali_pipeline_centered(image_dir, target_size=640):
        """复现 YOLO 居中填充预处理的 DALI 管道。

        精确匹配 Ultralytics LetterBox(center=True) 的行为。
        """
        # 在 GPU 上读取和解码图像
        jpegs, _ = fn.readers.file(file_root=image_dir, random_shuffle=False, name="Reader")
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

        # 保持宽高比的调整尺寸
        resized = fn.resize(
            images,
            resize_x=target_size,
            resize_y=target_size,
            mode="not_larger",
            interp_type=types.INTERP_LINEAR,
            antialias=False,  # 匹配 cv2.INTER_LINEAR（无抗锯齿）
        )

        # 使用 fn.crop 配合 out_of_bounds_policy 实现居中填充
        # 当裁剪尺寸 > 图像尺寸时，fn.crop 居中图像并对称填充
        padded = fn.crop(
            resized,
            crop=(target_size, target_size),
            out_of_bounds_policy="pad",
            fill_values=114,  # YOLO 填充值
        )

        # 归一化并转换布局
        output = fn.crop_mirror_normalize(
            padded,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
        )
        return output
    ```

!!! note "什么时候 `fn.pad` 就足够了？"

    如果你不需要精确匹配 `LetterBox(center=True)`，可以通过使用 `fn.pad(...)` 替代 `fn.crop(..., out_of_bounds_policy="pad")` 来简化填充步骤。该变体仅在**右侧和底部**添加填充，对于自定义部署管道来说可能是可以接受的，但它不会精确匹配 Ultralytics 默认的居中 letterbox 行为。

!!! tip "为什么使用 `fn.crop` 实现居中填充？"

    DALI 的 `fn.pad` 操作符仅在**右侧和底部**边缘添加填充。要获得居中填充（匹配 Ultralytics `LetterBox(center=True)`），需使用 `fn.crop` 配合 `out_of_bounds_policy="pad"`。使用默认的 `crop_pos_x=0.5` 和 `crop_pos_y=0.5`，图像会自动居中并对称填充。

!!! warning "抗锯齿不匹配"

    DALI 的 `fn.resize` 默认启用抗锯齿（`antialias=True`），而 OpenCV 的 `cv2.resize` 使用 `INTER_LINEAR` 时**不会**应用抗锯齿。在 DALI 中务必设置 `antialias=False` 以匹配 CPU 管道。省略此项设置会导致细微的数值差异，可能影响 [模型准确率](https://www.ultralytics.com/glossary/accuracy)。

### 运行管道

!!! example "构建并运行 DALI 管道"

    ```python
    # 构建并运行管道
    pipe = yolo_dali_pipeline_centered(image_dir="/path/to/images", target_size=640)
    pipe.build()

    # 获取一批预处理后的图像
    (output,) = pipe.run()

    # 转换为 numpy 或 PyTorch 张量
    batch_np = output.as_cpu().as_array()  # 形状: (batch_size, 3, 640, 640)
    print(f"输出形状: {batch_np.shape}, 数据类型: {batch_np.dtype}")
    print(f"数值范围: [{batch_np.min():.4f}, {batch_np.max():.4f}]")
    ```

## 将 DALI 与 Ultralytics Predict 一起使用

你可以直接将预处理后的 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 张量传递给 `model.predict()`。当传递 `torch.Tensor` 时，Ultralytics **跳过图像预处理**（letterbox、BGR→RGB、HWC→CHW 和 /255 归一化），仅在将数据送入模型之前执行设备传输和数据类型转换。

由于在这种情况下 Ultralytics 无法获取原始图像尺寸，检测框坐标会以 640×640 letterbox 空间返回。要将它们映射回原始图像坐标，请使用 [`scale_boxes`](../reference/utils/ops.md) 来处理 `LetterBox` 使用的精确舍入逻辑：

```python
from ultralytics.utils.ops import scale_boxes

# boxes: 形状为 (N, 4) 的张量，xyxy 格式，在 640x640 letterbox 坐标中
# 将框从 letterbox (640, 640) 缩放回原始 (orig_h, orig_w)
boxes = scale_boxes((640, 640), boxes, (orig_h, orig_w))
```

这适用于所有外部预处理路径——直接张量输入、视频流和 Triton 部署。

!!! example "DALI + Ultralytics predict"

    ```python
    from nvidia.dali.plugin.pytorch import DALIGenericIterator

    from ultralytics import YOLO

    # 加载模型
    model = YOLO("yolo26n.pt")

    # 创建 DALI 迭代器
    pipe = yolo_dali_pipeline_centered(image_dir="/path/to/images", target_size=640)
    pipe.build()
    dali_iter = DALIGenericIterator(pipe, ["images"], reader_name="Reader")

    # 使用 DALI 预处理后的张量运行推理
    for batch in dali_iter:
        images = batch[0]["images"]  # 已在 GPU 上，形状 (B, 3, 640, 640)
        results = model.predict(images, verbose=False)
        for result in results:
            print(f"检测到 {len(result.boxes)} 个目标")
    ```

!!! tip "零预处理开销"

    当你将 `torch.Tensor` 传递给 `model.predict()` 时，图像预处理步骤大约只需 0.004ms（基本为零），而 CPU 预处理需要约 1-10ms。张量必须为 BCHW 格式、float32（或 float16）且归一化到 `[0, 1]`。Ultralytics 仍会自动处理设备传输和数据类型转换。

## DALI 与视频流

对于实时视频处理，使用 `fn.external_source` 从任何来源——[OpenCV](https://www.ultralytics.com/glossary/opencv)、GStreamer 或自定义采集库——送入帧：

!!! example "用于视频流预处理的 DALI 管道"

    === "管道定义"

        ```python
        import nvidia.dali as dali
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types


        @dali.pipeline_def(batch_size=1, num_threads=4, device_id=0)
        def yolo_video_pipeline(target_size=640):
            """处理来自外部源的视频帧的 DALI 管道。"""
            # 外部源，用于从 OpenCV、GStreamer 等送入帧
            frames = fn.external_source(device="cpu", name="input")
            frames = fn.reshape(frames, layout="HWC")

            # 移至 GPU 并预处理
            frames_gpu = frames.gpu()
            resized = fn.resize(
                frames_gpu,
                resize_x=target_size,
                resize_y=target_size,
                mode="not_larger",
                interp_type=types.INTERP_LINEAR,
                antialias=False,
            )
            padded = fn.crop(
                resized,
                crop=(target_size, target_size),
                out_of_bounds_policy="pad",
                fill_values=114,
            )
            output = fn.crop_mirror_normalize(
                padded,
                dtype=types.FLOAT,
                output_layout="CHW",
                mean=[0.0, 0.0, 0.0],
                std=[255.0, 255.0, 255.0],
            )
            return output
        ```

    === "推理循环（简单 OpenCV 回退）"

        ```python
        import cv2
        import numpy as np
        import torch

        from ultralytics import YOLO

        model = YOLO("yolo26n.engine")  # TensorRT 模型

        pipe = yolo_video_pipeline(target_size=640)
        pipe.build()

        cap = cv2.VideoCapture("video.mp4")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 送入 BGR 帧（转换为 RGB 用于 DALI）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pipe.feed_input("input", [np.array(frame_rgb)])
            (output,) = pipe.run()

            # 将 DALI 输出转换为 torch 张量用于推理。
            # 这是一个简单的回退路径：使用 feed_input() 配合 pipe.run() 会保留 GPU→CPU→GPU 拷贝。
            # 对于高吞吐量部署，推荐使用基于读取器的管道配合 DALIGenericIterator 将数据保留在 GPU 上。
            tensor = torch.tensor(output.as_cpu().as_array()).to("cuda")
            results = model.predict(tensor, verbose=False)
        ```

## 使用 DALI 的 Triton 推理服务器

对于生产部署，使用集成模型将 DALI 预处理与 [TensorRT](../integrations/tensorrt.md) 推理结合在 [Triton 推理服务器](triton-inference-server.md) 中。这完全消除了 CPU 预处理——原始 JPEG 字节进入，检测结果输出，全部在 GPU 上处理。

### 模型仓库结构

```
model_repository/
├── dali_preprocessing/
│   ├── 1/
│   │   └── model.dali
│   └── config.pbtxt
├── yolo_trt/
│   ├── 1/
│   │   └── model.plan
│   └── config.pbtxt
└── ensemble_dali_yolo/
    ├── 1/                  # 空目录（Triton 要求）
    └── config.pbtxt
```

### 步骤 1：创建 DALI 管道

为 Triton DALI 后端序列化 DALI 管道：

!!! example "为 Triton 序列化 DALI 管道"

    ```python
    import nvidia.dali as dali
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types


    @dali.pipeline_def(batch_size=8, num_threads=4, device_id=0)
    def triton_dali_pipeline():
        """用于 Triton 部署的 DALI 预处理管道。"""
        # 输入：来自 Triton 的原始编码图像字节
        images = fn.external_source(device="cpu", name="DALI_INPUT_0")
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)

        resized = fn.resize(
            images,
            resize_x=640,
            resize_y=640,
            mode="not_larger",
            interp_type=types.INTERP_LINEAR,
            antialias=False,
        )
        padded = fn.crop(
            resized,
            crop=(640, 640),
            out_of_bounds_policy="pad",
            fill_values=114,
        )
        output = fn.crop_mirror_normalize(
            padded,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
        )
        return output


    # 序列化管道到模型仓库
    pipe = triton_dali_pipeline()
    pipe.serialize(filename="model_repository/dali_preprocessing/1/model.dali")
    ```

### 步骤 2：导出 YOLO 到 TensorRT

!!! example "导出 YOLO 模型到 TensorRT 引擎"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    model.export(format="engine", imgsz=640, half=True, batch=8)
    # 将 .engine 文件复制到 model_repository/yolo_trt/1/model.plan
    ```

### 步骤 3：配置 Triton

**dali_preprocessing/config.pbtxt：**

```protobuf
name: "dali_preprocessing"
backend: "dali"
max_batch_size: 8
input [
  {
    name: "DALI_INPUT_0"
    data_type: TYPE_UINT8
    dims: [ -1 ]
  }
]
output [
  {
    name: "DALI_OUTPUT_0"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]
```

**yolo_trt/config.pbtxt：**

```protobuf
name: "yolo_trt"
platform: "tensorrt_plan"
max_batch_size: 8
input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]
output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 300, 6 ]
  }
]
```

**ensemble_dali_yolo/config.pbtxt：**

```protobuf
name: "ensemble_dali_yolo"
platform: "ensemble"
max_batch_size: 8
input [
  {
    name: "INPUT"
    data_type: TYPE_UINT8
    dims: [ -1 ]
  }
]
output [
  {
    name: "OUTPUT"
    data_type: TYPE_FP32
    dims: [ 300, 6 ]
  }
]
ensemble_scheduling {
  step [
    {
      model_name: "dali_preprocessing"
      model_version: -1
      input_map {
        key: "DALI_INPUT_0"
        value: "INPUT"
      }
      output_map {
        key: "DALI_OUTPUT_0"
        value: "preprocessed_image"
      }
    },
    {
      model_name: "yolo_trt"
      model_version: -1
      input_map {
        key: "images"
        value: "preprocessed_image"
      }
      output_map {
        key: "output0"
        value: "OUTPUT"
      }
    }
  ]
}
```

!!! info "集成模型映射如何工作"

    集成模型通过**虚拟张量名称**连接模型。DALI 步骤中的 `output_map` 值 `"preprocessed_image"` 与 TensorRT 步骤中的 `input_map` 值 `"preprocessed_image"` 匹配。这些是任意名称，用于将一个步骤的输出链接到下一个步骤的输入——它们不需要匹配任何模型的内部张量名称。

### 步骤 4：发送推理请求

!!! info "为什么使用 `tritonclient` 而不是 `YOLO(\"http://...\")`？"

    Ultralytics 具有 [内置的 Triton 支持](triton-inference-server.md#running-inference)，可自动处理预处理/后处理。但是，它不适用于 DALI 集成模型，因为 `YOLO()` 发送的是预处理后的 float32 张量，而集成模型期望接收原始 JPEG 字节。对于 DALI 集成模型，请直接使用 `tritonclient`；对于不带 DALI 的标准部署，请使用 [内置集成](triton-inference-server.md)。

!!! example "向 Triton 集成模型发送图像"

    ```python
    import numpy as np
    import tritonclient.http as httpclient

    client = httpclient.InferenceServerClient(url="localhost:8000")

    # 将图像加载为原始字节（JPEG/PNG 编码）
    image_data = np.fromfile("image.jpg", dtype="uint8")
    image_data = np.expand_dims(image_data, axis=0)  # 添加批次维度

    # 创建输入
    input_tensor = httpclient.InferInput("INPUT", image_data.shape, "UINT8")
    input_tensor.set_data_from_numpy(image_data)

    # 通过集成模型运行推理
    result = client.infer(model_name="ensemble_dali_yolo", inputs=[input_tensor])
    detections = result.as_numpy("OUTPUT")  # 形状: (1, 300, 6) -> [x1, y1, x2, y2, conf, class_id]

    # 按置信度过滤（无需 NMS——YOLO26 是端到端的）
    detections = detections[0]  # 第一张图像
    detections = detections[detections[:, 4] > 0.25]  # 置信度阈值
    print(f"检测到 {len(detections)} 个目标")
    ```

!!! tip "批量处理 JPEG 图像"

    向 Triton 发送一批 JPEG 图像时，将所有编码后的字节数组填充到相同长度（批次中的最大字节数）。Triton 要求输入张量的批次形状一致。

## 支持的任务

DALI 预处理适用于所有使用标准 `LetterBox` 管道的 YOLO 任务：

| 任务                                        | 支持情况 | 备注                                                    |
| ------------------------------------------- | -------- | ------------------------------------------------------- |
| [检测](../tasks/detect.md)                  | ✅       | 标准 letterbox 预处理                                   |
| [分割](../tasks/segment.md)                 | ✅       | 与检测相同预处理                                        |
| [姿态估计](../tasks/pose.md)                | ✅       | 与检测相同预处理                                        |
| [旋转检测 (OBB)](../tasks/obb.md)           | ✅       | 与检测相同预处理                                        |
| [分类](../tasks/classify.md)                | ❌       | 使用 torchvision 变换（中心裁剪），而非 letterbox       |

## 局限性

- **仅限 Linux**：DALI 不支持 Windows 或 macOS
- **需要 NVIDIA GPU**：无 CPU 回退方案
- **静态管道**：管道结构在构建时定义，无法动态更改
- **`fn.pad` 仅支持右侧/底部**：使用 `fn.crop` 配合 `out_of_bounds_policy="pad"` 实现居中填充
- **不支持 rect 模式**：DALI 管道产生固定尺寸输出（如 640×640）。产生可变尺寸输出（如 384×640）的 `auto=True` rect 模式不受支持。请注意，虽然 [TensorRT](../integrations/tensorrt.md) 确实支持动态输入形状，但固定尺寸的 DALI 管道天然适合与固定尺寸引擎配对以获得最大吞吐量
- **多实例的内存问题**：在 Triton 中使用 `instance_group` 且 `count` > 1 可能导致高内存使用。对 DALI 模型使用默认实例组

## 常见问题

### DALI 预处理与 CPU 预处理的速度对比如何？

收益取决于你的管道。当 GPU 推理已经很快时（使用 [TensorRT](../integrations/tensorrt.md)），2-10ms 的 CPU 预处理可能成为主要成本。DALI 通过在 GPU 上运行预处理来消除这一瓶颈。最大的收益体现在高分辨率输入（1080p、4K）、大 [批次大小](https://www.ultralytics.com/glossary/batch-size) 以及每个 GPU 对应较少 CPU 核心的系统上。

### 我可以将 DALI 与 PyTorch 模型（不仅仅是 TensorRT）一起使用吗？

可以。使用 `DALIGenericIterator` 获取预处理后的 `torch.Tensor` 输出，然后将其传递给 `model.predict()`。然而，性能收益在使用 [TensorRT](../integrations/tensorrt.md) 模型时最大，因为此时推理已经非常快而 CPU 预处理成为瓶颈。

### `fn.pad` 和 `fn.crop` 在填充方面有什么区别？

`fn.pad` 仅在**右侧和底部**边缘添加填充。`fn.crop` 配合 `out_of_bounds_policy="pad"` 会将图像居中并在所有边上对称添加填充，匹配 Ultralytics `LetterBox(center=True)` 的行为。

### DALI 是否产生与 CPU 预处理像素级相同的结果？

几乎相同。在 `fn.resize` 中设置 `antialias=False` 以匹配 OpenCV 的 `cv2.INTER_LINEAR`。由于 GPU 与 CPU 的算术差异，可能会出现微小的浮点差异（< 0.001），但这些对检测 [准确率](https://www.ultralytics.com/glossary/accuracy) 没有可测量的影响。

### CV-CUDA 作为 DALI 的替代方案如何？

[CV-CUDA](https://github.com/CVCUDA/CV-CUDA) 是另一个用于 GPU 加速视觉处理的 NVIDIA 库。它提供逐操作符控制（如 [OpenCV](https://www.ultralytics.com/glossary/opencv) 但在 GPU 上），而非 DALI 的管道方式。CV-CUDA 的 `cvcuda.copymakeborder()` 支持显式的每边填充，使居中 letterbox 实现变得简单。对于基于管道的工作流（尤其是与 [Triton](triton-inference-server.md) 配合），选择 DALI；对于自定义推理代码中的细粒度操作符级控制，选择 CV-CUDA。