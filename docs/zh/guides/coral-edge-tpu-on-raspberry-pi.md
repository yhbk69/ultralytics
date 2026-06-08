---
comments: true
description: 了解如何使用 Coral Edge TPU 配合 Ultralytics YOLO26 提升 Raspberry Pi 的机器学习性能。遵循我们的详细设置与安装指南。
keywords: Coral Edge TPU, Raspberry Pi, YOLO26, Ultralytics, TensorFlow Lite, 机器学习推理, 机器学习, AI, 安装指南, 设置教程
---

# 在 Raspberry Pi 上使用 Coral Edge TPU 配合 Ultralytics YOLO26 🚀

<p align="center">
  <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/edge-tpu-usb-accelerator-and-pi.avif" alt="Raspberry Pi 搭配 Edge TPU 加速器">
</p>

## 什么是 Coral Edge TPU？

Coral Edge TPU 是一款紧凑型设备，可为您的系统添加 Edge TPU 协处理器。它能够为 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) Lite 模型提供低功耗、高性能的机器学习推理。更多信息请参阅 [Coral Edge TPU 主页](https://developers.google.com/coral)。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/w4yHORvDBw0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Google Coral Edge TPU 在 Raspberry Pi 上运行推理
</p>

## 使用 Coral Edge TPU 提升 Raspberry Pi 模型性能

许多人希望在嵌入式或移动设备（如 Raspberry Pi）上运行模型，因为这些设备功耗极低，适用于多种不同场景。然而，即使使用 [ONNX](../integrations/onnx.md) 或 [OpenVINO](../integrations/openvino.md) 等格式，这些设备上的推理性能通常也不太理想。Coral Edge TPU 是解决这一问题的绝佳方案，因为它可以与 Raspberry Pi 配合使用，大幅提升推理性能。

## Raspberry Pi 上配合 TensorFlow Lite 使用 Edge TPU（新版）⭐

Coral 官方提供的关于如何在 Raspberry Pi 上使用 Edge TPU 的[现有指南](https://gweb-coral-full.uc.r.appspot.com/docs/accelerator/get-started/)已经过时，当前的 Coral Edge TPU 运行时构建版本已不再兼容最新的 TensorFlow Lite 运行时版本。此外，Google 似乎已完全放弃了 Coral 项目，在 2021 年至 2025 年间没有任何更新。本指南将向您展示如何在 Raspberry Pi 单板计算机（SBC）上，配合最新版本的 TensorFlow Lite 运行时和更新后的 Coral Edge TPU 运行时来使用 Edge TPU。

## 前提条件

- [Raspberry Pi 4B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/)（建议 2GB 或以上）或 [Raspberry Pi 5](https://www.raspberrypi.com/products/raspberry-pi-5/)（推荐）
- [Raspberry Pi OS](https://www.raspberrypi.com/software/) Bullseye/Bookworm（64 位）桌面版（推荐）
- [Coral USB 加速器](https://developers.google.com/coral)
- 一台非 ARM 架构的平台，用于导出 Ultralytics [PyTorch](https://www.ultralytics.com/glossary/pytorch) 模型

## 安装步骤

本指南假设您已成功安装 Raspberry Pi OS，并已安装 `ultralytics` 及其所有依赖项。如需安装 `ultralytics`，请先访问[快速入门指南](../quickstart.md)完成设置，再继续后续操作。

### 安装 Edge TPU 运行时

首先，我们需要安装 Edge TPU 运行时。有多种版本可供选择，您需要根据自己的操作系统选择正确的版本。
高频版本会以更高的时钟频率运行 Edge TPU，从而提升性能。但这可能导致 Edge TPU 热节流，因此建议配备一定的散热措施。

| Raspberry Pi OS | 高频模式 | 下载版本                                    |
| --------------- | :------: | ------------------------------------------ |
| Bullseye 32 位  |    否    | `libedgetpu1-std_ ... .bullseye_armhf.deb` |
| Bullseye 64 位  |    否    | `libedgetpu1-std_ ... .bullseye_arm64.deb` |
| Bullseye 32 位  |    是    | `libedgetpu1-max_ ... .bullseye_armhf.deb` |
| Bullseye 64 位  |    是    | `libedgetpu1-max_ ... .bullseye_arm64.deb` |
| Bookworm 32 位  |    否    | `libedgetpu1-std_ ... .bookworm_armhf.deb` |
| Bookworm 64 位  |    否    | `libedgetpu1-std_ ... .bookworm_arm64.deb` |
| Bookworm 32 位  |    是    | `libedgetpu1-max_ ... .bookworm_armhf.deb` |
| Bookworm 64 位  |    是    | `libedgetpu1-max_ ... .bookworm_arm64.deb` |

[从此处下载最新版本](https://github.com/feranick/libedgetpu/releases)。

下载文件后，使用以下命令安装：

```bash
sudo dpkg -i path/to/package.deb
```

安装运行时后，将 Coral Edge TPU 插入 Raspberry Pi 的 USB 3.0 端口，以使新的 `udev` 规则生效。

???+ warning "重要提示"

    如果您已经安装了 Coral Edge TPU 运行时，请使用以下命令卸载。

    ```bash
    # 如果安装的是标准版本
    sudo apt remove libedgetpu1-std

    # 如果安装的是高频版本
    sudo apt remove libedgetpu1-max
    ```

## 导出至 Edge TPU

要使用 Edge TPU，您需要将模型转换为兼容格式。建议在 Google Colab、x86_64 Linux 机器上、使用官方 [Ultralytics Docker 容器](docker-quickstart.md)或通过 [Ultralytics 平台](../platform/quickstart.md)来执行导出操作，因为 Edge TPU 编译器不支持 ARM 架构。可用参数详见[导出模式](../modes/export.md)。

!!! example "导出模型"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/model.pt")  # 加载官方模型或自定义模型

        # 导出模型
        model.export(format="edgetpu")
        ```

    === "CLI"

        ```bash
        yolo export model=path/to/model.pt format=edgetpu # 导出官方模型或自定义模型
        ```

导出的模型将保存在 `<模型名称>_saved_model/` 文件夹中，文件名为 `<模型名称>_full_integer_quant_edgetpu.tflite`。请确保文件名以 `_edgetpu.tflite` 为后缀，否则 Ultralytics 将无法识别您正在使用 Edge TPU 模型。

## 运行模型

在实际运行模型之前，您需要安装正确的库。

如果您已安装 TensorFlow，请使用以下命令卸载：

```bash
pip uninstall tensorflow tensorflow-aarch64
```

然后安装或更新 `tflite-runtime`：

```bash
pip install -U tflite-runtime
```

现在您可以使用以下代码运行推理：

!!! example "运行模型"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/<模型名称>_full_integer_quant_edgetpu.tflite")  # 加载官方模型或自定义模型

        # 运行预测
        model.predict("path/to/source.png")
        ```

    === "CLI"

        ```bash
        yolo predict model=path/to/MODEL_NAME_full_integer_quant_edgetpu.tflite source=path/to/source.png # 加载官方模型或自定义模型
        ```

完整的预测模式详细信息请参阅[预测](../modes/predict.md)页面。

!!! note "使用多个 Edge TPU 进行推理"

    如果您有多个 Edge TPU，可以使用以下代码选择特定的 TPU。

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/<模型名称>_full_integer_quant_edgetpu.tflite")  # 加载官方模型或自定义模型

        # 运行预测
        model.predict("path/to/source.png")  # 推理默认使用第一个 TPU

        model.predict("path/to/source.png", device="tpu:0")  # 选择第一个 TPU

        model.predict("path/to/source.png", device="tpu:1")  # 选择第二个 TPU
        ```

## 基准测试

!!! tip "基准测试"

    测试环境为 Raspberry Pi OS Bookworm 64 位系统，搭配 USB Coral Edge TPU。

    !!! note

        以下展示的是推理时间，不包括预处理/后处理时间。

    === "Raspberry Pi 4B 2GB"

        | 图像尺寸 | 模型    | 标准模式推理时间 (ms) | 高频模式推理时间 (ms) |
        |----------|---------|----------------------|----------------------|
        | 320      | YOLOv8n | 32.2                 | 26.7                 |
        | 320      | YOLOv8s | 47.1                 | 39.8                 |
        | 512      | YOLOv8n | 73.5                 | 60.7                 |
        | 512      | YOLOv8s | 149.6                | 125.3                |

    === "Raspberry Pi 5 8GB"

        | 图像尺寸 | 模型    | 标准模式推理时间 (ms) | 高频模式推理时间 (ms) |
        |----------|---------|----------------------|----------------------|
        | 320      | YOLOv8n | 22.2                 | 16.7                 |
        | 320      | YOLOv8s | 40.1                 | 32.2                 |
        | 512      | YOLOv8n | 53.5                 | 41.6                 |
        | 512      | YOLOv8s | 132.0                | 103.3                |

    平均而言：

    - Raspberry Pi 5 在标准模式下比 Raspberry Pi 4B 快 22%。
    - Raspberry Pi 5 在高频模式下比 Raspberry Pi 4B 快 30.2%。
    - 高频模式比标准模式快 28.4%。

## 常见问题

### 什么是 Coral Edge TPU？它如何提升 Raspberry Pi 配合 Ultralytics YOLO26 的性能？

Coral Edge TPU 是一款紧凑型设备，旨在为您的系统添加 Edge TPU 协处理器。该协处理器能够实现低功耗、高性能的[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)推理，特别针对 TensorFlow Lite 模型进行了优化。在 Raspberry Pi 上使用时，Edge TPU 可加速机器学习模型推理，显著提升性能，尤其适用于 Ultralytics YOLO26 模型。您可以在 Coral Edge TPU [主页](https://developers.google.com/coral)了解更多信息。

### 如何在 Raspberry Pi 上安装 Coral Edge TPU 运行时？

要在 Raspberry Pi 上安装 Coral Edge TPU 运行时，请从[此链接](https://github.com/feranick/libedgetpu/releases)下载对应您 Raspberry Pi OS 版本的 `.deb` 包。下载后，使用以下命令安装：

```bash
sudo dpkg -i path/to/package.deb
```

请务必按照[安装步骤](#安装步骤)部分的说明，先卸载之前可能存在的 Coral Edge TPU 运行时版本。

### 可以将 Ultralytics YOLO26 模型导出为兼容 Coral Edge TPU 的格式吗？

可以。您可以将 Ultralytics YOLO26 模型导出为兼容 Coral Edge TPU 的格式。建议在 Google Colab、x86_64 Linux 机器上或使用 [Ultralytics Docker 容器](docker-quickstart.md)来执行导出操作。您也可以使用 [Ultralytics 平台](../platform/quickstart.md)进行导出。以下是通过 Python 和 CLI 导出模型的方法：

!!! example "导出模型"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/model.pt")  # 加载官方模型或自定义模型

        # 导出模型
        model.export(format="edgetpu")
        ```

    === "CLI"

        ```bash
        yolo export model=path/to/model.pt format=edgetpu # 导出官方模型或自定义模型
        ```

更多信息请参阅[导出模式](../modes/export.md)文档。

### 如果 Raspberry Pi 上已安装 TensorFlow，但我想改用 tflite-runtime，该怎么办？

如果您的 Raspberry Pi 上已安装 TensorFlow，需要切换到 `tflite-runtime`，请先使用以下命令卸载 TensorFlow：

```bash
pip uninstall tensorflow tensorflow-aarch64
```

然后使用以下命令安装或更新 `tflite-runtime`：

```bash
pip install -U tflite-runtime
```

详细说明请参阅[运行模型](#运行模型)部分。

### 如何使用 Coral Edge TPU 在 Raspberry Pi 上运行导出的 YOLO26 模型进行推理？

将 YOLO26 模型导出为 Edge TPU 兼容格式后，您可以使用以下代码片段运行推理：

!!! example "运行模型"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/edgetpu_model.tflite")  # 加载官方模型或自定义模型

        # 运行预测
        model.predict("path/to/source.png")
        ```

    === "CLI"

        ```bash
        yolo predict model=path/to/edgetpu_model.tflite source=path/to/source.png # 加载官方模型或自定义模型
        ```

完整的预测模式功能详情请参阅[预测页面](../modes/predict.md)。