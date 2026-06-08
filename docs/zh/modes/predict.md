---
comments: true
description: 利用 Ultralytics YOLO26 在各种数据源上实现实时高速推理。了解预测模式、关键功能及实际应用。
keywords: Ultralytics, YOLO26, 模型预测, 推理, 预测模式, 实时推理, 计算机视觉, 机器学习, 流式处理, 高性能
---

# 使用 Ultralytics YOLO 进行模型预测

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO 生态系统与集成">

## 简介

在[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域，理解视觉数据的过程通常称为推理或预测。Ultralytics YOLO26 提供了一个名为**预测模式**的强大功能，专为在各种数据源上进行高性能实时推理而设计。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/YKbBXWBJloY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何从 Ultralytics YOLO26 任务中提取结果用于自定义项目 🚀
</p>

## 实际应用

|                   制造业                   |                    体育                    |                   安全                    |
| :----------------------------------------: | :----------------------------------------: | :---------------------------------------: |
| ![车辆备件检测][car spare parts] | ![足球运动员检测][football player detect] | ![人员跌倒检测][human fall detect] |
|           车辆备件检测           |           足球运动员检测           |            人员跌倒检测            |

## 为什么选择 Ultralytics YOLO 进行推理？

以下是您应考虑使用 YOLO26 预测模式满足各种推理需求的原因：

- **多功能性：** 能够对图像、视频甚至实时流进行推理。
- **高性能：** 专为实时高速处理而设计，同时不牺牲[准确率](https://www.ultralytics.com/glossary/accuracy)。
- **易用性：** 直观的 Python 和 CLI 接口，便于快速部署和测试。
- **高度可定制：** 提供多种设置和参数，可根据具体需求调整模型的推理行为。
- **生产就绪：** 在 [Ultralytics 平台](https://platform.ultralytics.com)上将模型部署为 API 端点，支持自动扩缩容和监控，也可在本地运行推理。

### 预测模式的关键功能

YOLO26 的预测模式设计得稳健且多功能，具有以下特点：

- **多数据源兼容性：** 无论数据是单个图像、图像集合、视频文件还是实时视频流，预测模式都能胜任。
- **流式模式：** 使用流式功能可生成内存高效的 `Results` 对象生成器。在预测器的调用方法中设置 `stream=True` 即可启用。
- **批处理：** 在单个批次中处理多个图像或视频帧，进一步减少总推理时间。
- **集成友好：** 得益于其灵活的 API，可轻松与现有数据管道和其他软件组件集成。

Ultralytics YOLO 模型在推理时返回一个 Python 列表形式的 `Results` 对象列表，或者在传入 `stream=True` 时返回一个内存高效的 `Results` 对象生成器：

!!! example "预测"

    === "使用 `stream=False` 返回列表"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 预训练的 YOLO26n 模型

        # 对图像列表进行批量推理
        results = model(["image1.jpg", "image2.jpg"])  # 返回 Results 对象列表

        # 处理结果列表
        for result in results:
            boxes = result.boxes  # 用于边界框输出的 Boxes 对象
            masks = result.masks  # 用于分割掩码输出的 Masks 对象
            keypoints = result.keypoints  # 用于姿态输出的 Keypoints 对象
            probs = result.probs  # 用于分类输出的 Probs 对象
            obb = result.obb  # 用于 OBB 输出的 Oriented boxes 对象
            result.show()  # 显示到屏幕
            result.save(filename="result.jpg")  # 保存到磁盘
        ```

    === "使用 `stream=True` 返回生成器"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 预训练的 YOLO26n 模型

        # 对图像列表进行批量推理
        results = model(["image1.jpg", "image2.jpg"], stream=True)  # 返回 Results 对象生成器

        # 处理结果生成器
        for result in results:
            boxes = result.boxes  # 用于边界框输出的 Boxes 对象
            masks = result.masks  # 用于分割掩码输出的 Masks 对象
            keypoints = result.keypoints  # 用于姿态输出的 Keypoints 对象
            probs = result.probs  # 用于分类输出的 Probs 对象
            obb = result.obb  # 用于 OBB 输出的 Oriented boxes 对象
            result.show()  # 显示到屏幕
            result.save(filename="result.jpg")  # 保存到磁盘
        ```

## 推理源

YOLO26 可以处理不同类型的输入源进行推理，如下表所示。数据源包括静态图像、视频流以及各种数据格式。表格还标示了每种数据源是否可以在流式模式下使用 `stream=True` ✅ 参数。流式模式对于处理视频或实时流非常有用，因为它会创建一个结果生成器，而不是将所有帧加载到内存中。

!!! tip

    使用 `stream=True` 处理长视频或大型数据集可以高效管理内存。当 `stream=False` 时，所有帧或数据点的结果都存储在内存中，对于大型输入，这可能会迅速累积并导致内存不足错误。相反，`stream=True` 使用生成器，仅在内存中保留当前帧或数据点的结果，显著降低内存消耗并防止内存不足问题。

| 数据源                                                  | 示例                                        | 类型              | 备注                                                                                           |
| ------------------------------------------------------- | ------------------------------------------- | ----------------- | ---------------------------------------------------------------------------------------------- |
| 图像                                                    | `'image.jpg'`                               | `str` 或 `Path`   | 单个图像文件。                                                                                 |
| URL                                                     | `'https://ultralytics.com/images/bus.jpg'`  | `str`             | 图像的 URL。                                                                                   |
| 屏幕截图                                                | `'screen'`                                  | `str`             | 捕获屏幕截图。                                                                                 |
| PIL                                                     | `Image.open('image.jpg')`                   | `PIL.Image`       | RGB 通道的 HWC 格式。                                                                          |
| [OpenCV](https://www.ultralytics.com/glossary/opencv)   | `cv2.imread('image.jpg')`                   | `np.ndarray`      | BGR 通道的 HWC 格式 `uint8 (0-255)`。                                                          |
| NumPy                                                   | `np.zeros((640,1280,3))`                    | `np.ndarray`      | BGR 通道的 HWC 格式 `uint8 (0-255)`。                                                          |
| torch                                                   | `torch.zeros(16,3,320,640)`                 | `torch.Tensor`    | RGB 通道的 BCHW 格式 `float32 (0.0-1.0)`。                                                     |
| CSV                                                     | `'sources.csv'`                             | `str` 或 `Path`   | 包含图像、视频或目录路径的 CSV 文件。                                                           |
| 视频 ✅                                                 | `'video.mp4'`                               | `str` 或 `Path`   | MP4、AVI 等格式的视频文件。                                                                    |
| 目录 ✅                                                 | `'path/'`                                   | `str` 或 `Path`   | 包含图像或视频的目录路径。                                                                     |
| glob ✅                                                 | `'path/*.jpg'`                              | `str`             | 匹配多个文件的 glob 模式。使用 `*` 字符作为通配符。                                             |
| YouTube ✅                                              | `'https://youtu.be/LNwODJXcvt4'`            | `str`             | YouTube 视频的 URL。                                                                           |
| 流 ✅                                                   | `'rtsp://example.com/media.mp4'`            | `str`             | 流媒体协议的 URL，如 RTSP、RTMP、TCP 或 IP 地址。                                               |
| 多流 ✅                                                 | `'list.streams'`                            | `str` 或 `Path`   | `*.streams` 文本文件，每行一个流 URL，例如 8 个流将以 batch-size 8 运行。                       |
| 摄像头 ✅                                               | `0`                                         | `int`             | 已连接摄像头设备的索引，用于在其上运行推理。                                                      |

以下是每种数据源类型的代码示例：

!!! example "预测数据源"

    === "图像"

        对图像文件进行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 定义图像文件路径
        source = "path/to/image.jpg"

        # 对数据源进行推理
        results = model(source)  # Results 对象列表
        ```

    === "屏幕截图"

        将当前屏幕内容作为截图进行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 将当前屏幕截图作为数据源
        source = "screen"

        # 对数据源进行推理
        results = model(source)  # Results 对象列表
        ```

    === "URL"

        对通过 URL 远程托管的图像或视频进行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 定义远程图像或视频 URL
        source = "https://ultralytics.com/images/bus.jpg"

        # 对数据源进行推理
        results = model(source)  # Results 对象列表
        ```

    === "PIL"

        对使用 Python Imaging Library (PIL) 打开的图像进行推理。
        ```python
        from PIL import Image

        from ultralytics import YOLO

        # 加载预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 使用 PIL 打开图像
        source = Image.open("path/to/image.jpg")

        # 对数据源进行推理
        results = model(source)  # Results 对象列表
        ```

    === "OpenCV"

        对使用 OpenCV 读取的图像进行推理。
        ```python
        import cv2

        from ultralytics import YOLO

        # 加载预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 使用 OpenCV 读取图像
        source = cv2.imread("path/to/image.jpg")

        # 对数据源进行推理
        results = model(source)  # Results 对象列表
        ```

    === "NumPy"

        对表示为 NumPy 数组的图像进行推理。
        ```python
        import numpy as np

        from ultralytics import YOLO

        # 加载预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 创建随机 NumPy 数组，HWC 形状为 (640, 640, 3)，值域 [0, 255]，类型为 uint8
        source = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype="uint8")

        # 对数据源进行推理
        results = model(source)  # Results 对象列表
        ```

    === "torch"

        对表示为 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 张量的图像进行推理。
        ```python
        import torch

        from ultralytics import YOLO

        # 加载预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 创建随机 torch 张量，BCHW 形状为 (1, 3, 640, 640)，值域 [0, 1]，类型为 float32
        source = torch.rand(1, 3, 640, 640, dtype=torch.float32)

        # 对数据源进行推理
        results = model(source)  # Results 对象列表
        ```

    === "CSV"

        对列在 CSV 文件中的图像、URL、视频和目录集合进行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 定义包含图像、URL、视频和目录的 CSV 文件路径
        source = "path/to/file.csv"

        # 对数据源进行推理
        results = model(source)  # Results 对象列表
        ```

    === "视频"

        对视频文件进行推理。使用 `stream=True` 可以创建 Results 对象生成器以减少内存使用。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 定义视频文件路径
        source = "path/to/video.mp4"

        # 对数据源进行推理
        results = model(source, stream=True)  # Results 对象生成器
        ```

    === "目录"

        对目录中的所有图像和视频进行推理。如需包含子目录中的资源，请使用 glob 模式，如 `path/to/dir/**/*`。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 定义包含图像和视频的目录路径用于推理
        source = "path/to/dir"

        # 对数据源进行推理
        results = model(source, stream=True)  # Results 对象生成器
        ```

    === "glob"

        对匹配带有 `*` 字符的 glob 表达式的所有图像和视频进行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 定义对目录中所有 JPG 文件的 glob 搜索
        source = "path/to/dir/*.jpg"

        # 或者定义递归 glob 搜索，包括子目录中的所有 JPG 文件
        source = "path/to/dir/**/*.jpg"

        # 对数据源进行推理
        results = model(source, stream=True)  # Results 对象生成器
        ```

    === "YouTube"

        对 YouTube 视频进行推理。使用 `stream=True` 可以创建 Results 对象生成器以减少长视频的内存使用。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 将 YouTube 视频 URL 作为数据源
        source = "https://youtu.be/LNwODJXcvt4"

        # 对数据源进行推理
        results = model(source, stream=True)  # Results 对象生成器
        ```

    === "流"

        使用流模式对实时视频流进行推理，支持 RTSP、RTMP、TCP 或 IP 地址协议。如果提供单个流，模型以[批大小](https://www.ultralytics.com/glossary/batch-size) 1 运行推理。对于多流，可以使用 `.streams` 文本文件进行批量推理，批大小由提供的流数量决定（例如，8 个流的 batch-size 为 8）。

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 单流以 batch-size 1 推理
        source = "rtsp://example.com/media.mp4"  # RTSP、RTMP、TCP 或 IP 流地址

        # 对数据源进行推理
        results = model(source, stream=True)  # Results 对象生成器
        ```

        对于单流使用，批大小默认设置为 1，从而可以对视频流进行高效的实时处理。

    === "多流"

        要同时处理多个视频流，请使用 `.streams` 文本文件，每行一个数据源。模型将以批大小等于流数量的方式运行批量推理。此设置可实现对多个视频流的高效并发处理。

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 多流批量推理（例如，8 个流的 batch-size 为 8）
        source = "path/to/list.streams"  # *.streams 文本文件，每行一个流地址

        # 对数据源进行推理
        results = model(source, stream=True)  # Results 对象生成器
        ```

        `.streams` 文本文件示例：

        ```
        rtsp://example.com/media1.mp4
        rtsp://example.com/media2.mp4
        rtmp://example2.com/live
        tcp://192.168.1.100:554
        ...
        ```

        文件中的每一行代表一个流数据源，可让您同时监控和推理多个视频流。

    === "摄像头"

        您可以通过将特定摄像头的索引传递给 `source` 参数，对已连接的摄像头设备进行推理。

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 对数据源进行推理
        results = model(source=0, stream=True)  # Results 对象生成器
        ```

## 推理参数

`model.predict()` 接受多个参数，这些参数可以在推理时传入以覆盖默认值：

### 固定形状 vs 最小矩形 (`rect`)

默认情况下，predict 使用 **`rect=True`**，在可能的情况下启用**最小矩形**填充。图像被缩放以适应 `imgsz`，并仅填充到最近的步长倍数，因此最终张量可能**小于** `imgsz`。最小矩形填充仅在**批次中所有图像形状相同**且后端支持时（PyTorch `.pt`，或动态 ONNX / Triton）使用。否则，图像将被填充到**完整**的 `imgsz` 目标尺寸。

使用 **`rect=False`** 可始终填充到完整的 `imgsz` 目标尺寸。当需要固定输入大小以匹配导出模型（ONNX、TensorRT 等）时，建议使用此设置。

**整数 vs 元组 `imgsz`**

- **整数** `imgsz=640` 在经过步长舍入后变为方形目标 `(640, 640)`。
- **元组** `imgsz=(384, 672)` 设置矩形目标。在 `rect=True` 且 `auto=True` 的情况下，实际张量可能小于此目标。

**训练 vs 预测/导出**

训练仅接受单个整数 `imgsz`（`[h, w]` 列表会被强制转换为最大值）。预测和导出接受整数或 `(height, width)` 元组。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 对 'bus.jpg' 使用参数进行推理
        model.predict("https://ultralytics.com/images/bus.jpg", save=True, imgsz=320, conf=0.25)
        ```

    === "CLI"

        ```bash
        # 对 'bus.jpg' 进行推理
        yolo predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg'
        ```

推理参数：

{% include "macros/predict-args.md" %}

可视化参数：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table() }}

## 图像和视频格式

YOLO26 支持多种图像和视频格式，详见 [ultralytics/data/utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py)。请参阅下表了解有效的后缀名和预测命令示例。

### 图像

下表包含 Ultralytics 支持的有效图像格式。

!!! note

    HEIC/HEIF 格式需要 `pi-heif`，该库会在首次使用时自动安装。AVIF 由 Pillow 原生支持。

| 图像后缀 | 预测命令示例                     | 参考                                                                        |
| -------- | -------------------------------- | --------------------------------------------------------------------------- |
| `.avif`  | `yolo predict source=image.avif` | [AV1 图像文件格式](https://en.wikipedia.org/wiki/AVIF)                      |
| `.bmp`   | `yolo predict source=image.bmp`  | [Microsoft BMP 文件格式](https://en.wikipedia.org/wiki/BMP_file_format)     |
| `.dng`   | `yolo predict source=image.dng`  | [Adobe DNG](https://en.wikipedia.org/wiki/Digital_Negative)                 |
| `.heic`  | `yolo predict source=image.heic` | [高效图像文件格式](https://en.wikipedia.org/wiki/HEIF)                      |
| `.heif`  | `yolo predict source=image.heif` | [高效图像文件格式](https://en.wikipedia.org/wiki/HEIF)                      |
| `.jp2`   | `yolo predict source=image.jp2`  | [JPEG 2000](https://en.wikipedia.org/wiki/JPEG_2000)                        |
| `.jpeg`  | `yolo predict source=image.jpeg` | [JPEG](https://en.wikipedia.org/wiki/JPEG)                                  |
| `.jpg`   | `yolo predict source=image.jpg`  | [JPEG](https://en.wikipedia.org/wiki/JPEG)                                  |
| `.mpo`   | `yolo predict source=image.mpo`  | [Multi Picture Object](https://fileinfo.com/extension/mpo)                  |
| `.png`   | `yolo predict source=image.png`  | [便携式网络图形](https://en.wikipedia.org/wiki/PNG)                         |
| `.tif`   | `yolo predict source=image.tif`  | [标签图像文件格式](https://en.wikipedia.org/wiki/TIFF)                      |
| `.tiff`  | `yolo predict source=image.tiff` | [标签图像文件格式](https://en.wikipedia.org/wiki/TIFF)                      |
| `.webp`  | `yolo predict source=image.webp` | [WebP](https://en.wikipedia.org/wiki/WebP)                                  |

### 视频

下表包含 Ultralytics 支持的有效视频格式。

| 视频后缀 | 预测命令示例                     | 参考                                                                              |
| -------- | -------------------------------- | --------------------------------------------------------------------------------- |
| `.asf`   | `yolo predict source=video.asf`  | [高级系统格式](https://en.wikipedia.org/wiki/Advanced_Systems_Format)             |
| `.avi`   | `yolo predict source=video.avi`  | [音频视频交错](https://en.wikipedia.org/wiki/Audio_Video_Interleave)              |
| `.gif`   | `yolo predict source=video.gif`  | [图形交换格式](https://en.wikipedia.org/wiki/GIF)                                 |
| `.m4v`   | `yolo predict source=video.m4v`  | [MPEG-4 第 14 部分](https://en.wikipedia.org/wiki/M4V)                            |
| `.mkv`   | `yolo predict source=video.mkv`  | [Matroska](https://en.wikipedia.org/wiki/Matroska)                                |
| `.mov`   | `yolo predict source=video.mov`  | [QuickTime 文件格式](https://en.wikipedia.org/wiki/QuickTime_File_Format)         |
| `.mp4`   | `yolo predict source=video.mp4`  | [MPEG-4 第 14 部分 - Wikipedia](https://en.wikipedia.org/wiki/MPEG-4_Part_14)     |
| `.mpeg`  | `yolo predict source=video.mpeg` | [MPEG-1 第 2 部分](https://en.wikipedia.org/wiki/MPEG-1)                          |
| `.mpg`   | `yolo predict source=video.mpg`  | [MPEG-1 第 2 部分](https://en.wikipedia.org/wiki/MPEG-1)                          |
| `.ts`    | `yolo predict source=video.ts`   | [MPEG 传输流](https://en.wikipedia.org/wiki/MPEG_transport_stream)                |
| `.wmv`   | `yolo predict source=video.wmv`  | [Windows Media Video](https://en.wikipedia.org/wiki/Windows_Media_Video)          |
| `.webm`  | `yolo predict source=video.webm` | [WebM 项目](https://en.wikipedia.org/wiki/WebM)                                   |

## 处理结果

所有 Ultralytics `predict()` 调用都会返回一个 `Results` 对象列表：

!!! example "结果"

    ```python
    from ultralytics import YOLO

    # 加载预训练的 YOLO26n 模型
    model = YOLO("yolo26n.pt")

    # 对图像进行推理
    results = model("https://ultralytics.com/images/bus.jpg")
    results = model(
        [
            "https://ultralytics.com/images/bus.jpg",
            "https://ultralytics.com/images/zidane.jpg",
        ]
    )  # 批量推理
    ```

`Results` 对象具有以下属性：

| 属性          | 类型                  | 描述                                                         |
| ------------- | --------------------- | ------------------------------------------------------------ |
| `orig_img`    | `np.ndarray`          | 原始图像，以 NumPy 数组形式表示。                            |
| `orig_shape`  | `tuple`               | 原始图像形状，格式为 (height, width)。                       |
| `boxes`       | `Boxes, 可选`         | 包含检测边界框的 Boxes 对象。                                |
| `masks`       | `Masks, 可选`         | 包含检测掩码的 Masks 对象。                                  |
| `probs`       | `Probs, 可选`         | 包含分类任务每个类别概率的 Probs 对象。                      |
| `keypoints`   | `Keypoints, 可选`     | 包含每个对象检测到的关键点的 Keypoints 对象。                |
| `obb`         | `OBB, 可选`           | 包含旋转边界框的 OBB 对象。                                  |
| `speed`       | `dict`                | 每张图像的预处理、推理和后处理速度字典，单位为毫秒。         |
| `names`       | `dict`                | 将类别索引映射到类别名称的字典。                             |
| `path`        | `str`                 | 图像文件的路径。                                             |
| `save_dir`    | `str, 可选`           | 保存结果的目录。                                             |

`Results` 对象具有以下方法：

| 方法          | 返回类型               | 描述                                                                             |
| ------------- | ---------------------- | -------------------------------------------------------------------------------- |
| `update()`    | `None`                 | 使用新的检测数据（boxes、masks、probs、obb、keypoints）更新 Results 对象。       |
| `cpu()`       | `Results`              | 返回 Results 对象的副本，所有张量移至 CPU 内存。                                 |
| `numpy()`     | `Results`              | 返回 Results 对象的副本，所有张量转换为 NumPy 数组。                             |
| `cuda()`      | `Results`              | 返回 Results 对象的副本，所有张量移至 GPU 内存。                                 |
| `to()`        | `Results`              | 返回 Results 对象的副本，张量移至指定的设备和数据类型。                          |
| `new()`       | `Results`              | 创建新的 Results 对象，具有相同的图像、路径、名称和速度属性。                    |
| `plot()`      | `np.ndarray`           | 在输入 RGB 图像上绘制检测结果，并返回标注后的图像。                              |
| `show()`      | `None`                 | 显示带有推理结果标注的图像。                                                     |
| `save()`      | `str`                  | 将带有推理结果标注的图像保存到文件，并返回文件名。                               |
| `verbose()`   | `str`                  | 返回每个任务的日志字符串，详细说明检测和分类结果。                               |
| `save_txt()`  | `str`                  | 将检测结果保存到文本文件，并返回保存文件的路径。                                 |
| `save_crop()` | `None`                 | 将裁剪后的检测图像保存到指定目录。                                               |
| `summary()`   | `List[Dict[str, Any]]` | 将推理结果转换为带有可选归一化功能的汇总字典。                                   |
| `to_df()`     | `DataFrame`            | 将检测结果转换为 Polars DataFrame。                                              |
| `to_csv()`    | `str`                  | 将检测结果转换为 CSV 格式。                                                      |
| `to_json()`   | `str`                  | 将检测结果转换为 JSON 格式。                                                     |

更多详情请参阅 [`Results` 类文档](../reference/engine/results.md)。

### Boxes

`Boxes` 对象可用于索引、操作边界框，并将其转换为不同格式。

!!! example "Boxes"

    ```python
    from ultralytics import YOLO

    # 加载预训练的 YOLO26n 模型
    model = YOLO("yolo26n.pt")

    # 对图像进行推理
    results = model("https://ultralytics.com/images/bus.jpg")  # results 列表

    # 查看结果
    for r in results:
        print(r.boxes)  # 打印包含检测边界框的 Boxes 对象
    ```

以下是 `Boxes` 类的方法和属性表，包括名称、类型和描述：

| 名称      | 类型                      | 描述                                               |
| --------- | ------------------------- | -------------------------------------------------- |
| `cpu()`   | 方法                      | 将对象移至 CPU 内存。                              |
| `numpy()` | 方法                      | 将对象转换为 NumPy 数组。                          |
| `cuda()`  | 方法                      | 将对象移至 CUDA 内存。                             |
| `to()`    | 方法                      | 将对象移至指定设备。                               |
| `xyxy`    | 属性 (`torch.Tensor`)     | 返回 xyxy 格式的边界框。                           |
| `conf`    | 属性 (`torch.Tensor`)     | 返回边界框的置信度值。                             |
| `cls`     | 属性 (`torch.Tensor`)     | 返回边界框的类别值。                               |
| `id`      | 属性 (`torch.Tensor`)     | 返回边界框的跟踪 ID（如果可用）。                  |
| `xywh`    | 属性 (`torch.Tensor`)     | 返回 xywh 格式的边界框。                           |
| `xyxyn`   | 属性 (`torch.Tensor`)     | 返回按原始图像尺寸归一化的 xyxy 格式边界框。       |
| `xywhn`   | 属性 (`torch.Tensor`)     | 返回按原始图像尺寸归一化的 xywh 格式边界框。       |

更多详情请参阅 [`Boxes` 类文档](../reference/engine/results.md#ultralytics.engine.results.Boxes)。

### Masks

`Masks` 对象可用于索引、操作掩码，并将其转换为线段。

!!! example "Masks"

    ```python
    from ultralytics import YOLO

    # 加载预训练的 YOLO26n-seg 分割模型
    model = YOLO("yolo26n-seg.pt")

    # 对图像进行推理
    results = model("https://ultralytics.com/images/bus.jpg")  # results 列表

    # 查看结果
    for r in results:
        print(r.masks)  # 打印包含检测实例掩码的 Masks 对象
    ```

以下是 `Masks` 类的方法和属性表，包括名称、类型和描述：

| 名称      | 类型                      | 描述                                             |
| --------- | ------------------------- | ------------------------------------------------ |
| `cpu()`   | 方法                      | 返回 CPU 内存上的掩码张量。                      |
| `numpy()` | 方法                      | 将掩码张量返回为 NumPy 数组。                    |
| `cuda()`  | 方法                      | 返回 GPU 内存上的掩码张量。                      |
| `to()`    | 方法                      | 返回具有指定设备和数据类型的掩码张量。           |
| `xyn`     | 属性 (`torch.Tensor`)     | 表示为张量的归一化线段列表。                     |
| `xy`      | 属性 (`torch.Tensor`)     | 表示为张量的像素坐标线段列表。                   |

更多详情请参阅 [`Masks` 类文档](../reference/engine/results.md#ultralytics.engine.results.Masks)。

### Keypoints

`Keypoints` 对象可用于索引、操作和归一化坐标。

!!! example "Keypoints"

    ```python
    from ultralytics import YOLO

    # 加载预训练的 YOLO26n-pose 姿态模型
    model = YOLO("yolo26n-pose.pt")

    # 对图像进行推理
    results = model("https://ultralytics.com/images/bus.jpg")  # results 列表

    # 查看结果
    for r in results:
        print(r.keypoints)  # 打印包含检测到的关键点的 Keypoints 对象
    ```

以下是 `Keypoints` 类的方法和属性表，包括名称、类型和描述：

| 名称      | 类型                      | 描述                                                       |
| --------- | ------------------------- | ---------------------------------------------------------- |
| `cpu()`   | 方法                      | 返回 CPU 内存上的关键点张量。                              |
| `numpy()` | 方法                      | 将关键点张量返回为 NumPy 数组。                            |
| `cuda()`  | 方法                      | 返回 GPU 内存上的关键点张量。                              |
| `to()`    | 方法                      | 返回具有指定设备和数据类型的关键点张量。                   |
| `xyn`     | 属性 (`torch.Tensor`)     | 表示为张量的归一化关键点列表。                             |
| `xy`      | 属性 (`torch.Tensor`)     | 表示为张量的像素坐标关键点列表。                           |
| `conf`    | 属性 (`torch.Tensor`)     | 返回关键点的置信度值（如果可用），否则返回 None。          |

更多详情请参阅 [`Keypoints` 类文档](../reference/engine/results.md#ultralytics.engine.results.Keypoints)。

### Probs

`Probs` 对象可用于索引，获取分类的 `top1` 和 `top5` 索引及分数。

!!! example "Probs"

    ```python
    from ultralytics import YOLO

    # 加载预训练的 YOLO26n-cls 分类模型
    model = YOLO("yolo26n-cls.pt")

    # 对图像进行推理
    results = model("https://ultralytics.com/images/bus.jpg")  # results 列表

    # 查看结果
    for r in results:
        print(r.probs)  # 打印包含检测类别概率的 Probs 对象
    ```

以下是 `Probs` 类的方法和属性汇总表：

| 名称       | 类型                      | 描述                                                   |
| ---------- | ------------------------- | ------------------------------------------------------ |
| `cpu()`    | 方法                      | 返回 CPU 内存上 probs 张量的副本。                     |
| `numpy()`  | 方法                      | 将 probs 张量返回为 NumPy 数组的副本。                 |
| `cuda()`   | 方法                      | 返回 GPU 内存上 probs 张量的副本。                     |
| `to()`     | 方法                      | 返回具有指定设备和数据类型的 probs 张量副本。          |
| `top1`     | 属性 (`int`)              | Top 1 类别的索引。                                     |
| `top5`     | 属性 (`list[int]`)        | Top 5 类别的索引。                                     |
| `top1conf` | 属性 (`torch.Tensor`)     | Top 1 类别的置信度。                                   |
| `top5conf` | 属性 (`torch.Tensor`)     | Top 5 类别的置信度。                                   |

更多详情请参阅 [`Probs` 类文档](../reference/engine/results.md#ultralytics.engine.results.Probs)。

### OBB

`OBB` 对象可用于索引、操作旋转边界框，并将其转换为不同格式。

!!! example "OBB"

    ```python
    from ultralytics import YOLO

    # 加载预训练的 YOLO26n-obb 模型
    model = YOLO("yolo26n-obb.pt")

    # 对图像进行推理
    results = model("https://ultralytics.com/images/boats.jpg")  # results 列表

    # 查看结果
    for r in results:
        print(r.obb)  # 打印包含旋转检测边界框的 OBB 对象
    ```

以下是 `OBB` 类的方法和属性表，包括名称、类型和描述：

| 名称          | 类型                      | 描述                                                         |
| ------------- | ------------------------- | ------------------------------------------------------------ |
| `cpu()`       | 方法                      | 将对象移至 CPU 内存。                                        |
| `numpy()`     | 方法                      | 将对象转换为 NumPy 数组。                                    |
| `cuda()`      | 方法                      | 将对象移至 CUDA 内存。                                       |
| `to()`        | 方法                      | 将对象移至指定设备。                                         |
| `conf`        | 属性 (`torch.Tensor`)     | 返回边界框的置信度值。                                       |
| `cls`         | 属性 (`torch.Tensor`)     | 返回边界框的类别值。                                         |
| `id`          | 属性 (`torch.Tensor`)     | 返回边界框的跟踪 ID（如果可用）。                            |
| `xyxy`        | 属性 (`torch.Tensor`)     | 返回 xyxy 格式的水平边界框。                                 |
| `xywhr`       | 属性 (`torch.Tensor`)     | 返回 xywhr 格式的旋转边界框。                                |
| `xyxyxyxy`    | 属性 (`torch.Tensor`)     | 返回 xyxyxyxy 格式的旋转边界框。                             |
| `xyxyxyxyn`   | 属性 (`torch.Tensor`)     | 返回按图像尺寸归一化的 xyxyxyxy 格式旋转边界框。             |

更多详情请参阅 [`OBB` 类文档](../reference/engine/results.md#ultralytics.engine.results.OBB)。

## 绘制结果

`Results` 对象中的 `plot()` 方法通过将检测到的对象（如边界框、掩码、关键点和概率）叠加到原始图像上来实现预测的可视化。该方法将标注后的图像作为 NumPy 数组返回，便于显示或保存。

!!! example "绘图"

    ```python
    from PIL import Image

    from ultralytics import YOLO

    # 加载预训练的 YOLO26n 模型
    model = YOLO("yolo26n.pt")

    # 对 'bus.jpg' 进行推理
    results = model(["https://ultralytics.com/images/bus.jpg", "https://ultralytics.com/images/zidane.jpg"])  # results 列表

    # 可视化结果
    for i, r in enumerate(results):
        # 绘制结果图像
        im_bgr = r.plot()  # BGR 顺序的 NumPy 数组
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB 顺序的 PIL 图像

        # 将结果显示到屏幕（在受支持的环境中）
        r.show()

        # 将结果保存到磁盘
        r.save(filename=f"results{i}.jpg")
    ```

### `plot()` 方法参数

`plot()` 方法支持多种参数来自定义输出：

| 参数          | 类型                   | 描述                                                         | 默认值            |
| ------------- | ---------------------- | ------------------------------------------------------------ | ----------------- |
| `conf`        | `bool`                 | 包含检测置信度分数。                                         | `True`            |
| `line_width`  | `float`                | 边界框的线宽。如果为 `None`，则根据图像大小缩放。            | `None`            |
| `font_size`   | `float`                | 文本字体大小。如果为 `None`，则根据图像大小缩放。            | `None`            |
| `font`        | `str`                  | 文本标注的字体名称。                                         | `'Arial.ttf'`     |
| `pil`         | `bool`                 | 以 PIL Image 对象返回图像。                                  | `False`           |
| `img`         | `np.ndarray`           | 用于绘图的替代图像。如果为 `None`，则使用原始图像。          | `None`            |
| `im_gpu`      | `torch.Tensor`         | GPU 加速图像，用于更快的掩码绘制。形状：(1, 3, 640, 640)。   | `None`            |
| `kpt_radius`  | `int`                  | 绘制关键点的半径。                                           | `5`               |
| `kpt_line`    | `bool`                 | 用线段连接关键点。                                           | `True`            |
| `labels`      | `bool`                 | 在标注中包含类别标签。                                       | `True`            |
| `boxes`       | `bool`                 | 在图像上叠加边界框。                                         | `True`            |
| `masks`       | `bool`                 | 在图像上叠加掩码。                                           | `True`            |
| `probs`       | `bool`                 | 包含分类概率。                                               | `True`            |
| `show`        | `bool`                 | 使用默认图像查看器直接显示标注后的图像。                     | `False`           |
| `save`        | `bool`                 | 将标注后的图像保存到 `filename` 指定的文件。                 | `False`           |
| `filename`    | `str`                  | 如果 `save` 为 `True`，保存标注图像的文件路径和名称。         | `None`            |
| `color_mode`  | `str`                  | 指定颜色模式，例如 `'instance'` 或 `'class'`。               | `'class'`         |
| `txt_color`   | `tuple[int, int, int]` | 边界框和图像分类标签的 RGB 文本颜色。                        | `(255, 255, 255)` |

## 线程安全推理

当您在不同线程中并行运行多个 YOLO 模型时，确保推理过程中的线程安全至关重要。线程安全推理可以保证每个线程的预测是隔离的，互不干扰，从而避免竞争条件，确保输出结果一致且可靠。

在多线程应用中使用 YOLO 模型时，重要的是为每个线程实例化单独的模型对象，或使用线程本地存储来防止冲突：

!!! example "线程安全推理"

    在每个线程内部实例化一个单独的模型以实现线程安全推理：
    ```python
    from threading import Thread

    from ultralytics import YOLO


    def thread_safe_predict(model, image_path):
        """使用本地实例化的 YOLO 模型对图像进行线程安全预测。"""
        model = YOLO(model)
        results = model.predict(image_path)
        # 处理结果


    # 启动线程，每个线程都有自己的模型实例
    Thread(target=thread_safe_predict, args=("yolo26n.pt", "image1.jpg")).start()
    Thread(target=thread_safe_predict, args=("yolo26n.pt", "image2.jpg")).start()
    ```

要深入了解 YOLO 模型的线程安全推理及分步指导，请参阅我们的 [YOLO 线程安全推理指南](../guides/yolo-thread-safe-inference.md)。该指南将为您提供所有必要的信息，帮助您避免常见陷阱，确保多线程推理顺畅运行。

## 流式数据源 `for` 循环

以下是一个使用 OpenCV (`cv2`) 和 YOLO 对视频帧进行推理的 Python 脚本。该脚本假定您已安装了必要的包（`opencv-python` 和 `ultralytics`）。

!!! example "流式 for 循环"

    ```python
    import cv2

    from ultralytics import YOLO

    # 加载 YOLO 模型
    model = YOLO("yolo26n.pt")

    # 打开视频文件
    video_path = "path/to/your/video/file.mp4"
    cap = cv2.VideoCapture(video_path)

    # 循环遍历视频帧
    while cap.isOpened():
        # 从视频中读取一帧
        success, frame = cap.read()

        if success:
            # 对帧运行 YOLO 推理
            results = model(frame)

            # 在帧上可视化结果
            annotated_frame = results[0].plot()

            # 显示标注后的帧
            cv2.imshow("YOLO Inference", annotated_frame)

            # 按下 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 到达视频末尾时退出循环
            break

    # 释放视频捕获对象并关闭显示窗口
    cap.release()
    cv2.destroyAllWindows()
    ```

此脚本将对视频的每一帧运行预测，可视化结果，并在窗口中显示。按 'q' 键可以退出循环。

[car spare parts]: https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/car-parts-detection-for-predict.avif
[football player detect]: https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/football-players-detection.avif
[human fall detect]: https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/person-fall-detection.avif

## 常见问题

### 什么是 Ultralytics YOLO 及其用于实时推理的预测模式？

Ultralytics YOLO 是一个用于实时[目标检测](https://www.ultralytics.com/glossary/object-detection)、分割和分类的最先进模型。其**预测模式**允许用户对图像、视频和实时流等各种数据源进行高速推理。该模式专为性能和多用途而设计，还提供批处理和流式模式。有关其功能的更多详细信息，请参阅 [Ultralytics YOLO 预测模式](#预测模式的关键功能)。

### 如何使用 Ultralytics YOLO 对不同数据源进行推理？

Ultralytics YOLO 可以处理各种数据源，包括单个图像、视频、目录、URL 和流。您可以在 `model.predict()` 调用中指定数据源。例如，使用 `'image.jpg'` 表示本地图像，或使用 `'https://ultralytics.com/images/bus.jpg'` 表示 URL。请查看文档中不同[推理源](#推理源)的详细示例。

### 如何优化 YOLO 推理速度和内存使用？

要优化推理速度并高效管理内存，可以在预测器的调用方法中设置 `stream=True` 来使用流式模式。流式模式会生成一个内存高效的 `Results` 对象生成器，而不是将所有帧加载到内存中。对于处理长视频或大型数据集，流式模式尤为有用。了解更多关于[流式模式](#预测模式的关键功能)的信息。

### Ultralytics YOLO 支持哪些推理参数？

YOLO 中 `model.predict()` 方法支持多种参数，如 `conf`、`iou`、`imgsz`、`device` 等。这些参数允许您自定义推理过程，设置置信度阈值、图像大小和计算设备等参数。这些参数的详细描述可在[推理参数](#推理参数)部分找到。

### 如何可视化和保存 YOLO 预测结果？

使用 YOLO 进行推理后，`Results` 对象包含显示和保存标注图像的方法。您可以使用 `result.show()` 和 `result.save(filename="result.jpg")` 等方法来可视化和保存结果。文件名路径中的任何缺失父目录都会自动创建（例如 `result.save("path/to/result.jpg")`）。有关这些方法的完整列表，请参阅[处理结果](#处理结果)部分。
