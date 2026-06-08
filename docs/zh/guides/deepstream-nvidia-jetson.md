---
comments: true
description: 学习如何使用 TensorRT 和 DeepStream SDK 在 NVIDIA Jetson 设备上部署 Ultralytics YOLO26。探索性能基准并最大化 AI 能力。
keywords: Ultralytics, YOLO26, NVIDIA Jetson, JetPack, AI 部署, 嵌入式系统, 深度学习, TensorRT, DeepStream SDK, 计算机视觉
---

# 使用 DeepStream SDK 和 TensorRT 在 NVIDIA Jetson 上部署 Ultralytics YOLO26

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/hvGqrVT2wPg"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看视频：</strong>如何在 Jetson Orin NX 上使用 Ultralytics YOLO26 模型与 NVIDIA Deepstream 🚀
</p>

本综合指南提供了使用 DeepStream SDK 和 TensorRT 在 [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) 设备上部署 Ultralytics YOLO26 的详细步骤。这里我们使用 TensorRT 来最大化 Jetson 平台上的推理性能。

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/deepstream-nvidia-jetson.avif" alt="NVIDIA DeepStream SDK 在 Jetson 平台上">

!!! note

    本指南已通过以下设备测试：
    - 运行最新稳定版 JetPack [JP6.1](https://developer.nvidia.com/embedded/jetpack-sdk-61) 的 [NVIDIA Jetson Orin Nano 超级开发者套件](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit)
    - 基于 NVIDIA Jetson Orin NX 16GB 运行 JetPack [JP5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513) 的 [Seeed Studio reComputer J4012](https://www.seeedstudio.com/reComputer-J4012-p-5586.html)
    - 基于 NVIDIA Jetson Nano 4GB 运行 JetPack [JP4.6.4](https://developer.nvidia.com/jetpack-sdk-464) 的 [Seeed Studio reComputer J1020 v2](https://www.seeedstudio.com/reComputer-J1020-v2-p-5498.html)
    预计可在所有 NVIDIA Jetson 硬件系列（包括最新和旧款）上正常工作。

## 什么是 NVIDIA DeepStream？

[NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) 是一个基于 GStreamer 的完整流分析工具包，用于基于 AI 的多传感器处理、视频、音频和图像理解。它非常适合视觉 AI 开发者、软件合作伙伴、初创公司和 OEM 构建 IVA（智能视频分析）应用和服务。您现在可以创建包含[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)和其他复杂处理任务（如跟踪、视频编码/解码和视频渲染）的流处理管道。这些管道支持对视频、图像和传感器数据进行实时分析。DeepStream 的多平台支持让您能够更快速、更轻松地在本地、边缘和云端开发视觉 AI 应用和服务。

## 先决条件

在开始遵循本指南之前：

- 访问我们的文档 [快速入门指南：NVIDIA Jetson 与 Ultralytics YOLO26](nvidia-jetson.md)，设置您的 NVIDIA Jetson 设备与 Ultralytics YOLO26
- 根据 JetPack 版本安装 [DeepStream SDK](https://developer.nvidia.com/deepstream-getting-started)
    - 对于 JetPack 4.6.4，安装 [DeepStream 6.0.1](https://archive.docs.nvidia.com/metropolis/deepstream/6.0.1/dev-guide/text/DS_Quickstart.html)
    - 对于 JetPack 5.1.3，安装 [DeepStream 6.3](https://archive.docs.nvidia.com/metropolis/deepstream/6.3/dev-guide/text/DS_Quickstart.html)
    - 对于 JetPack 6.1，安装 [DeepStream 7.1](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_Overview.html)
    - 对于 JetPack 7.1，安装 [DeepStream 9.0](https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_Overview.html)

!!! tip

    在本指南中，我们使用了 Debian 包方法将 DeepStream SDK 安装到 Jetson 设备。您也可以访问 [DeepStream SDK on Jetson (Archived)](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived) 来获取旧版本的 DeepStream。

## YOLO26 的 DeepStream 配置

这里我们使用 [marcoslucianops/DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) GitHub 仓库，其中包含对 YOLO 模型的 NVIDIA DeepStream SDK 支持。我们感谢 marcoslucianops 的贡献！

1.  安装 Ultralytics 及其必要依赖

    ```bash
    cd ~
    pip install -U pip
    git clone https://github.com/ultralytics/ultralytics
    cd ultralytics
    pip install -e ".[export]" onnxslim
    ```

2.  克隆 DeepStream-Yolo 仓库

    ```bash
    cd ~
    git clone https://github.com/marcoslucianops/DeepStream-Yolo
    ```

3.  将 `export_yolo26.py` 文件从 `DeepStream-Yolo/utils` 目录复制到 `ultralytics` 文件夹

    ```bash
    cp ~/DeepStream-Yolo/utils/export_yolo26.py ~/ultralytics
    cd ultralytics
    ```

4.  从 [YOLO26 发布页面](https://github.com/ultralytics/assets/releases) 下载您选择的 Ultralytics YOLO26 检测模型 (.pt)。这里我们使用 [yolo26s.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt)。

    ```bash
    wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt
    ```

    !!! note

        您也可以使用 [自定义训练的 YOLO26 模型](https://docs.ultralytics.com/modes/train)。

5.  将模型转换为 ONNX

    ```bash
    python3 export_yolo26.py -w yolo26s.pt
    ```

    !!! note "向上述命令传递以下参数"

        对于 DeepStream 5.1，移除 `--dynamic` 参数并使用 `opset` 12 或更低版本。默认 `opset` 为 17。

        ```bash
        --opset 12
        ```

        更改推理尺寸（默认：640）

        ```bash
        -s SIZE
        --size SIZE
        -s HEIGHT WIDTH
        --size HEIGHT WIDTH
        ```

        1280 的示例：

        ```bash
        -s 1280
        或
        -s 1280 1280
        ```

        简化 ONNX 模型（DeepStream >= 6.0）

        ```bash
        --simplify
        ```

        使用动态批处理大小（DeepStream >= 6.1）

        ```bash
        --dynamic
        ```

        使用静态批处理大小（批处理大小 = 4 的示例）

        ```bash
        --batch 4
        ```

6.  将生成的 `.onnx` 模型文件和 `labels.txt` 文件复制到 `DeepStream-Yolo` 文件夹

    ```bash
    cp yolo26s.pt.onnx labels.txt ~/DeepStream-Yolo
    cd ~/DeepStream-Yolo
    ```

7.  根据安装的 JetPack 版本设置 CUDA 版本

    对于 JetPack 4.6.4：

    ```bash
    export CUDA_VER=10.2
    ```

    对于 JetPack 5.1.3：

    ```bash
    export CUDA_VER=11.4
    ```

    对于 JetPack 6.1：

    ```bash
    export CUDA_VER=12.6
    ```

8.  编译库

    ```bash
    make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
    ```

9.  根据您的模型编辑 `config_infer_primary_yolo26.txt` 文件（针对 80 个类别的 YOLO26s）

    ```bash
    [property]
    ...
    onnx-file=yolo26s.pt.onnx
    ...
    num-detected-classes=80
    ...
    ```

10. 编辑 `deepstream_app_config` 文件

    ```bash
    ...
    [primary-gie]
    ...
    config-file=config_infer_primary_yolo26.txt
    ```

11. 您还可以在 `deepstream_app_config` 文件中更改视频源。这里加载了一个默认视频文件

    ```bash
    ...
    [source0]
    ...
    uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
    ```

### 运行推理

```bash
deepstream-app -c deepstream_app_config.txt
```

!!! note

    在开始推理之前，生成 TensorRT 引擎文件需要很长时间。请耐心等待。

<div align=center><img width=1000 src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yolov8-with-deepstream.avif" alt="YOLO26 与 deepstream"></div>

!!! tip

    如果您想将模型转换为 FP16 精度，只需在 `config_infer_primary_yolo26.txt` 中设置 `model-engine-file=model_b1_gpu0_fp16.engine` 和 `network-mode=2`

## INT8 校准

如果您想使用 INT8 精度进行推理，需要按照以下步骤操作：

!!! note

    目前 INT8 不适用于 TensorRT 10.x。本指南的此部分已通过 TensorRT 8.x 测试，预计可以正常工作。

1.  设置 `OPENCV` 环境变量

    ```bash
    export OPENCV=1
    ```

2.  编译库

    ```bash
    make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
    ```

3.  对于 COCO 数据集，下载 [val2017](http://images.cocodataset.org/zips/val2017.zip)，解压并移动到 `DeepStream-Yolo` 文件夹

4.  为校准图像创建新目录

    ```bash
    mkdir calibration
    ```

5.  运行以下命令从 COCO 数据集中选择 1000 张随机图像进行校准

    ```bash
    for jpg in $(ls -1 val2017/*.jpg | sort -R | head -1000); do
      cp ${jpg} calibration/
    done
    ```

    !!! note

        NVIDIA 建议至少使用 500 张图像以获得良好的[准确率](https://www.ultralytics.com/glossary/accuracy)。在此示例中，选择 1000 张图像以获得更好的准确率（图像越多 = 准确率越高）。您可以从 **head -1000** 设置。例如，对于 2000 张图像，使用 **head -2000**。此过程可能需要很长时间。

6.  创建包含所有选定图像的 `calibration.txt` 文件

    ```bash
    realpath calibration/*jpg > calibration.txt
    ```

7.  设置环境变量

    ```bash
    export INT8_CALIB_IMG_PATH=calibration.txt
    export INT8_CALIB_BATCH_SIZE=1
    ```

    !!! note

        更高的 INT8_CALIB_BATCH_SIZE 值将带来更高的准确率和更快的校准速度。请根据您的 GPU 内存进行设置。

8.  更新 `config_infer_primary_yolo26.txt` 文件

    从

    ```bash
    ...
    model-engine-file=model_b1_gpu0_fp32.engine
    #int8-calib-file=calib.table
    ...
    network-mode=0
    ...
    ```

    改为

    ```bash
    ...
    model-engine-file=model_b1_gpu0_int8.engine
    int8-calib-file=calib.table
    ...
    network-mode=1
    ...
    ```

### 运行推理

```bash
deepstream-app -c deepstream_app_config.txt
```

## 多流设置

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/BpSuXSUzEYY"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看视频：</strong>如何在 Jetson Orin 上使用 NVIDIA DeepStream 运行多流推理与 Ultralytics YOLO26 🚀
</p>

要在单个 DeepStream 应用程序下设置多个流，请对 `deepstream_app_config.txt` 文件进行以下更改：

1. 根据您想要的流数量更改行和列以构建网格显示。例如，对于 4 个流，我们可以添加 2 行和 2 列。

    ```bash
    [tiled-display]
    rows=2
    columns=2
    ```

2. 设置 `num-sources=4` 并为所有四个流添加 `uri` 条目。

    ```bash
    [source0]
    enable=1
    type=3
    uri=path/to/video1.jpg
    uri=path/to/video2.jpg
    uri=path/to/video3.jpg
    uri=path/to/video4.jpg
    num-sources=4
    ```

### 运行推理

```bash
deepstream-app -c deepstream_app_config.txt
```

<div align=center><img width=1000 src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/multistream-setup.avif" alt="DeepStream 多摄像头流配置"></div>

## 基准测试结果

以下基准测试总结了 YOLO26 模型在 NVIDIA Jetson Orin NX 16GB 上以不同 TensorRT 精度级别和 640x640 输入尺寸的性能表现。

### 对比图表

<div align=center><img width=1000 src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/jetson-deepstream-benchmarks.avif" alt="NVIDIA Jetson DeepStream 性能基准测试"></div>

### 详细对比表

!!! tip "性能"

    === "YOLO11n"

        | 格式          | 状态 | 推理时间 (ms/图像) |
        |-----------------|--------|------------------------|
        | TensorRT (FP32) | ✅      | 8.64                   |
        | TensorRT (FP16) | ✅      | 5.27                   |
        | TensorRT (INT8) | ✅      | 4.54                   |

    === "YOLO11s"

        | 格式          | 状态 | 推理时间 (ms/图像) |
        |-----------------|--------|------------------------|
        | TensorRT (FP32) | ✅      | 14.53                  |
        | TensorRT (FP16) | ✅      | 7.91                   |
        | TensorRT (INT8) | ✅      | 6.05                   |

    === "YOLO11m"

        | 格式          | 状态 | 推理时间 (ms/图像) |
        |-----------------|--------|------------------------|
        | TensorRT (FP32) | ✅      | 32.05                  |
        | TensorRT (FP16) | ✅      | 15.55                  |
        | TensorRT (INT8) | ✅      | 10.43                  |

    === "YOLO11l"

        | 格式          | 状态 | 推理时间 (ms/图像) |
        |-----------------|--------|------------------------|
        | TensorRT (FP32) | ✅      | 39.68                  |
        | TensorRT (FP16) | ✅      | 19.88                  |
        | TensorRT (INT8) | ✅      | 13.64                  |

    === "YOLO11x"

        | 格式          | 状态 | 推理时间 (ms/图像) |
        |-----------------|--------|------------------------|
        | TensorRT (FP32) | ✅      | 80.65                  |
        | TensorRT (FP16) | ✅      | 39.06                  |
        | TensorRT (INT8) | ✅      | 22.83                  |

## 致谢

本指南最初由我们的朋友 Seeed Studio 的 Lakshantha 和 Elaine 创建。

## 常见问题解答

### 如何在 NVIDIA Jetson 设备上设置 Ultralytics YOLO26？

要在 [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) 设备上设置 Ultralytics YOLO26，首先需要安装与您的 JetPack 版本兼容的 [DeepStream SDK](https://developer.nvidia.com/deepstream-getting-started)。按照我们的[快速入门指南](nvidia-jetson.md)中的分步指南配置您的 NVIDIA Jetson 以进行 YOLO26 部署。

### 在 NVIDIA Jetson 上使用 TensorRT 与 YOLO26 有什么好处？

在 NVIDIA Jetson 设备上使用 TensorRT 与 YOLO26 可以优化模型推理，显著降低延迟并提高吞吐量。TensorRT 通过层融合、精度校准和内核自动调优提供高性能、低延迟的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)推理。这导致更快、更高效的执行，特别适用于视频分析和自主机器等实时应用。

### 我可以在不同的 NVIDIA Jetson 硬件上运行 Ultralytics YOLO26 与 DeepStream SDK 吗？

是的，使用 DeepStream SDK 和 TensorRT 部署 Ultralytics YOLO26 的指南与整个 NVIDIA Jetson 系列兼容。这包括像运行 [JetPack 5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513) 的 Jetson Orin NX 16GB 和运行 [JetPack 4.6.4](https://developer.nvidia.com/jetpack-sdk-464) 的 Jetson Nano 4GB 等设备。有关详细步骤，请参阅 [YOLO26 的 DeepStream 配置](#deepstream-configuration-for-yolo26) 部分。

### 如何将 YOLO26 模型转换为 ONNX 格式用于 DeepStream？

要将 YOLO26 模型转换为 ONNX 格式以用于 DeepStream 部署，请使用 [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) 仓库中的 `utils/export_yolo26.py` 脚本。

以下是示例命令：

```bash
python3 utils/export_yolo26.py -w yolo26s.pt --opset 12 --simplify
```

有关模型转换的更多详细信息，请查看我们的[模型导出部分](../modes/export.md)。

### YOLO 在 NVIDIA Jetson Orin NX 上的性能基准是什么？

YOLO26 模型在 NVIDIA Jetson Orin NX 16GB 上的性能根据 TensorRT 精度级别而有所不同。例如，YOLO26s 模型实现：

- **FP32 精度**：14.6 ms/图像，68.5 FPS
- **FP16 精度**：7.94 ms/图像，126 FPS
- **INT8 精度**：5.95 ms/图像，168 FPS

这些基准测试凸显了在 NVIDIA Jetson 硬件上使用 TensorRT 优化的 YOLO26 模型的效率和能力。更多详细信息，请参阅我们的[基准测试结果](#benchmark-results)部分。