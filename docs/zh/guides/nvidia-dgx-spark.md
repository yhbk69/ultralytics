---
comments: true
description: 通过本详细指南学习在 NVIDIA DGX Spark 上部署 Ultralytics YOLO26。探索性能基准并在这款紧凑型桌面 AI 超级计算机上最大化 AI 能力。
keywords: Ultralytics, YOLO26, NVIDIA DGX Spark, AI 部署, 性能基准, 深度学习, TensorRT, 计算机视觉, GB10 Grace Blackwell
---

# 快速入门指南：NVIDIA DGX Spark 与 Ultralytics YOLO26

本综合指南提供了在 [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)（NVIDIA 的紧凑型桌面 AI 超级计算机）上部署 Ultralytics YOLO26 的详细步骤。此外，还展示了性能基准，以演示 YOLO26 在这款强大系统上的能力。

<p align="center">
  <img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/nvidia-dgx-spark.avif" alt="NVIDIA DGX Spark AI 工作站概览">
</p>

!!! note

    本指南已在运行基于 Ubuntu 的 DGX OS 的 NVIDIA DGX Spark Founders Edition 上测试通过。预计与最新的 DGX OS 版本兼容。

## 什么是 NVIDIA DGX Spark？

NVIDIA DGX Spark 是一款紧凑型桌面 AI 超级计算机，搭载 NVIDIA GB10 Grace Blackwell 超级芯片。它在 FP4 精度下提供高达 1 petaFLOP 的 AI 计算性能，非常适合需要桌面级强大 AI 能力的开发者、研究人员和数据科学家。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/VHGfpOrPh-s"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 NVIDIA DGX Spark 上使用 Ultralytics YOLO26 实现高达 1000 FPS | <a href="https://docs.ultralytics.com/integrations/tensorrt">TensorRT</a> 与批量推理
</p>

### 关键规格

| 规格         | 详情                                                                 |
| ------------ | -------------------------------------------------------------------- |
| AI 性能      | 高达 1 PFLOP (FP4)                                                   |
| GPU          | NVIDIA Blackwell 架构，搭载第五代 Tensor Core、第四代 RT Core           |
| CPU          | 20 核 Arm 处理器 (10 Cortex-X925 + 10 Cortex-A725)                   |
| 内存         | 128 GB LPDDR5x 统一系统内存，256 位接口，4266 MHz，273 GB/s 带宽       |
| 存储         | 1 TB 或 4 TB NVMe M.2，支持自加密                                    |
| 网络         | 1x RJ-45 (10 GbE)，ConnectX-7 智能网卡，Wi-Fi 7，蓝牙 5.4             |
| 连接性       | 4x USB Type-C，1x HDMI 2.1a，HDMI 多声道音频                         |
| 视频处理     | 1x NVENC，1x NVDEC                                                    |

### DGX OS

[NVIDIA DGX OS](https://docs.nvidia.com/dgx/dgx-os-7-user-guide/introduction.html) 是一个定制化的 Linux 发行版，为在 DGX 系统上运行 AI、机器学习和分析应用提供了稳定、经过测试且受支持的操作系统基础。它包括：

- 针对 AI 工作负载优化的稳健 Linux 基础
- 预配置的 NVIDIA 硬件驱动程序和系统设置
- 安全更新和系统维护能力
- 与更广泛的 NVIDIA 软件生态系统的兼容性

DGX OS 遵循定期发布计划，通常每年提供两次更新（约在 2 月和 8 月），并在主要版本之间提供额外的安全补丁。

### DGX 仪表板

DGX Spark 内置了 [DGX 仪表板](https://docs.nvidia.com/dgx/dgx-spark/dgx-dashboard.html)，提供以下功能：

- **实时系统监控**：系统当前运行指标的概览
- **系统更新**：可直接从仪表板应用更新
- **系统设置**：更改设备名称和其他配置
- **集成 JupyterLab**：访问本地 Jupyter Notebook 进行开发

<p align="center">
  <img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/nvidia-dgx-dashboard.avif" alt="NVIDIA DGX 管理仪表板界面">
</p>

#### 访问仪表板

=== "本地访问"

    点击 Ubuntu 桌面左下角的"显示应用程序"按钮，然后选择"DGX Dashboard"即可在浏览器中打开。

=== "通过 SSH 远程访问"

    ```bash
    # 打开 SSH 隧道
    ssh -L 11000:localhost:11000 username@spark-abcd.local

    # 然后在浏览器中打开
    # http://localhost:11000
    ```

=== "通过 NVIDIA Sync 远程访问"

    使用 NVIDIA Sync 连接后，点击"DGX Dashboard"按钮即可在 `http://localhost:11000` 打开仪表板。

!!! tip "集成 JupyterLab"

    仪表板包含一个集成的 JupyterLab 实例，启动时会自动创建虚拟环境并安装推荐的软件包。每个用户账户分配一个专用端口用于 JupyterLab 访问。

## 使用 Docker 快速入门

在 NVIDIA DGX Spark 上使用 Ultralytics YOLO26 的最快方式是运行预构建的 Docker 镜像。支持 Jetson AGX Thor（JetPack 7.0）的同一 Docker 镜像也可在 DGX Spark 上使用 DGX OS 运行。

```bash
t=ultralytics/ultralytics:latest-nvidia-arm64
sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia --gpus all $t
```

完成后，跳转到[在 NVIDIA DGX Spark 上使用 TensorRT](#在-nvidia-dgx-spark-上使用-tensorrt) 部分。

## 原生安装入门

如需不使用 Docker 的原生安装，请按以下步骤操作。

### 安装 Ultralytics 软件包

这里我们将在 DGX Spark 上安装带有可选依赖的 Ultralytics 软件包，以便将 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 模型导出为其他格式。我们将主要关注 [NVIDIA TensorRT 导出](../integrations/tensorrt.md)，因为 TensorRT 能确保我们从 DGX Spark 中获得最大性能。

1. 更新软件包列表，安装 pip 并升级到最新版本

    ```bash
    sudo apt update
    sudo apt install python3-pip -y
    pip install -U pip
    ```

2. 安装带有可选依赖的 `ultralytics` pip 软件包

    ```bash
    pip install ultralytics[export]
    ```

3. 重启设备

    ```bash
    sudo reboot
    ```

### 安装 PyTorch 和 Torchvision

上述 ultralytics 安装将安装 Torch 和 Torchvision。然而，通过 pip 安装的这些软件包可能未针对 DGX Spark 的 ARM64 架构和 CUDA 13 进行完全优化。因此，我们建议安装与 CUDA 13 兼容的版本：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

!!! info

    在 NVIDIA DGX Spark 上运行 PyTorch 2.9.1 时，初始化 CUDA 时（例如运行 `yolo checks`、`yolo predict` 等）可能会遇到以下 `UserWarning`：

    ```
    UserWarning: Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
    Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
    ```

    此警告可以安全忽略。为了永久解决此问题，已在 PyTorch PR [#164590](https://github.com/pytorch/pytorch/pull/164590) 中提交了修复，该修复将包含在 PyTorch 2.10 版本中。

### 安装 `onnxruntime-gpu`

PyPI 上托管的 [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/) 软件包不包含 ARM64 系统的 `aarch64` 二进制文件。因此我们需要手动安装此软件包。某些导出功能需要此软件包。

这里我们将下载并安装支持 `Python3.12` 的 `onnxruntime-gpu 1.24.0`。

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.24.0-cp312-cp312-linux_aarch64.whl
```

## 在 NVIDIA DGX Spark 上使用 TensorRT

在 Ultralytics 支持的所有模型导出格式中，TensorRT 在 NVIDIA DGX Spark 上提供最高的推理性能，因此是我们强烈推荐的部署选择。有关设置说明和高级用法，请参阅我们的 [TensorRT 集成指南](../integrations/tensorrt.md)。

### 将模型转换为 TensorRT 并运行推理

将 PyTorch 格式的 YOLO26n 模型转换为 TensorRT 格式，并使用导出的模型运行推理。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO26n PyTorch 模型
        model = YOLO("yolo26n.pt")

        # 将模型导出为 TensorRT 格式
        model.export(format="engine")  # 创建 'yolo26n.engine'

        # 加载导出的 TensorRT 模型
        trt_model = YOLO("yolo26n.engine")

        # 运行推理
        results = trt_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO26n PyTorch 模型导出为 TensorRT 格式
        yolo export model=yolo26n.pt format=engine # 创建 'yolo26n.engine'

        # 使用导出的模型运行推理
        yolo predict model=yolo26n.engine source='https://ultralytics.com/images/bus.jpg'
        ```

!!! note

    访问[导出页面](../modes/export.md#arguments)以获取导出模型到不同格式时的其他参数

## NVIDIA DGX Spark YOLO11 基准测试

YOLO11 基准测试由 Ultralytics 团队在多种模型格式上运行，测量速度和[准确率](https://www.ultralytics.com/glossary/accuracy)：PyTorch、TorchScript、ONNX、OpenVINO、TensorRT、TF SavedModel、TF GraphDef、TF Lite、MNN、NCNN、ExecuTorch。基准测试在 NVIDIA DGX Spark 上以 FP32 [精度](https://www.ultralytics.com/glossary/precision)运行，默认输入图像尺寸为 640。

### 详细对比表

下表展示了五种不同模型（YOLO11n、YOLO11s、YOLO11m、YOLO11l、YOLO11x）在多种格式下的基准测试结果，包括状态、大小、mAP50-95(B) 指标以及每种组合的推理时间。

!!! tip "性能"

    === "YOLO11n"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|------|---------------|-------------|------------------|
        | PyTorch         | ✅    | 5.4           | 0.5071      | 2.67             |
        | TorchScript     | ✅    | 10.5          | 0.5083      | 2.62             |
        | ONNX            | ✅    | 10.2          | 0.5074      | 5.92             |
        | OpenVINO        | ✅    | 10.4          | 0.5058      | 14.95            |
        | TensorRT (FP32) | ✅    | 12.8          | 0.5085      | 1.95             |
        | TensorRT (FP16) | ✅    | 7.0           | 0.5068      | 1.01             |
        | TensorRT (INT8) | ✅    | 18.6          | 0.4880      | 1.62             |
        | TF SavedModel   | ✅    | 25.7          | 0.5076      | 36.39            |
        | TF GraphDef     | ✅    | 10.3          | 0.5076      | 41.06            |
        | TF Lite         | ✅    | 10.3          | 0.5075      | 64.36            |
        | MNN             | ✅    | 10.1          | 0.5075      | 12.14            |
        | NCNN            | ✅    | 10.2          | 0.5041      | 12.31            |
        | ExecuTorch      | ✅    | 10.2          | 0.5075      | 27.61            |

    === "YOLO11s"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|------|---------------|-------------|------------------|
        | PyTorch         | ✅    | 18.4          | 0.5767      | 5.38             |
        | TorchScript     | ✅    | 36.5          | 0.5781      | 5.48             |
        | ONNX            | ✅    | 36.3          | 0.5784      | 8.17             |
        | OpenVINO        | ✅    | 36.4          | 0.5809      | 27.12            |
        | TensorRT (FP32) | ✅    | 39.8          | 0.5783      | 3.59             |
        | TensorRT (FP16) | ✅    | 20.1          | 0.5800      | 1.85             |
        | TensorRT (INT8) | ✅    | 17.5          | 0.5664      | 1.88             |
        | TF SavedModel   | ✅    | 90.8          | 0.5782      | 66.63            |
        | TF GraphDef     | ✅    | 36.3          | 0.5782      | 71.67            |
        | TF Lite         | ✅    | 36.3          | 0.5782      | 187.36           |
        | MNN             | ✅    | 36.2          | 0.5775      | 27.05            |
        | NCNN            | ✅    | 36.2          | 0.5806      | 26.26            |
        | ExecuTorch      | ✅    | 36.2          | 0.5782      | 54.73            |

    === "YOLO11m"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|------|---------------|-------------|------------------|
        | PyTorch         | ✅    | 38.8          | 0.6254      | 11.14            |
        | TorchScript     | ✅    | 77.3          | 0.6304      | 12.00            |
        | ONNX            | ✅    | 76.9          | 0.6304      | 13.83            |
        | OpenVINO        | ✅    | 77.1          | 0.6284      | 62.44            |
        | TensorRT (FP32) | ✅    | 79.9          | 0.6305      | 6.96             |
        | TensorRT (FP16) | ✅    | 40.6          | 0.6313      | 3.14             |
        | TensorRT (INT8) | ✅    | 26.6          | 0.6204      | 3.30             |
        | TF SavedModel   | ✅    | 192.4         | 0.6306      | 139.85           |
        | TF GraphDef     | ✅    | 76.9          | 0.6306      | 146.76           |
        | TF Lite         | ✅    | 76.9          | 0.6306      | 568.18           |
        | MNN             | ✅    | 76.8          | 0.6306      | 67.67            |
        | NCNN            | ✅    | 76.8          | 0.6308      | 60.49            |
        | ExecuTorch      | ✅    | 76.9          | 0.6306      | 120.37           |

    === "YOLO11l"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|------|---------------|-------------|------------------|
        | PyTorch         | ✅    | 49.0          | 0.6366      | 13.95            |
        | TorchScript     | ✅    | 97.6          | 0.6399      | 15.67            |
        | ONNX            | ✅    | 97.0          | 0.6399      | 16.62            |
        | OpenVINO        | ✅    | 97.3          | 0.6377      | 78.80            |
        | TensorRT (FP32) | ✅    | 99.2          | 0.6407      | 8.86             |
        | TensorRT (FP16) | ✅    | 50.8          | 0.6350      | 3.85             |
        | TensorRT (INT8) | ✅    | 32.5          | 0.6224      | 4.52             |
        | TF SavedModel   | ✅    | 242.7         | 0.6409      | 187.45           |
        | TF GraphDef     | ✅    | 97.0          | 0.6409      | 193.92           |
        | TF Lite         | ✅    | 97.0          | 0.6409      | 728.61           |
        | MNN             | ✅    | 96.9          | 0.6369      | 85.21            |
        | NCNN            | ✅    | 96.9          | 0.6373      | 77.62            |
        | ExecuTorch      | ✅    | 97.0          | 0.6409      | 153.56           |

    === "YOLO11x"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|------|---------------|-------------|------------------|
        | PyTorch         | ✅    | 109.3         | 0.6992      | 23.19            |
        | TorchScript     | ✅    | 218.1         | 0.6900      | 25.75            |
        | ONNX            | ✅    | 217.5         | 0.6900      | 27.43            |
        | OpenVINO        | ✅    | 217.8         | 0.6872      | 149.44           |
        | TensorRT (FP32) | ✅    | 222.7         | 0.6902      | 13.87            |
        | TensorRT (FP16) | ✅    | 111.1         | 0.6883      | 6.19             |
        | TensorRT (INT8) | ✅    | 62.9          | 0.6793      | 6.62             |
        | TF SavedModel   | ✅    | 543.9         | 0.6900      | 335.10           |
        | TF GraphDef     | ✅    | 217.5         | 0.6900      | 348.86           |
        | TF Lite         | ✅    | 217.5         | 0.6900      | 1578.66          |
        | MNN             | ✅    | 217.3         | 0.6874      | 168.95           |
        | NCNN            | ✅    | 217.4         | 0.6901      | 132.13           |
        | ExecuTorch      | ✅    | 217.4         | 0.6900      | 297.17           |

    使用 Ultralytics 8.3.249 进行基准测试

## 复现我们的结果

要复现上述 Ultralytics 在所有导出[格式](../modes/export.md)上的基准测试，请运行以下代码：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO26n PyTorch 模型
        model = YOLO("yolo26n.pt")

        # 在 COCO128 数据集上对所有导出格式进行 YOLO26n 速度和准确率基准测试
        results = model.benchmark(data="coco128.yaml", imgsz=640)
        ```

    === "CLI"

        ```bash
        # 在 COCO128 数据集上对所有导出格式进行 YOLO26n 速度和准确率基准测试
        yolo benchmark model=yolo26n.pt data=coco128.yaml imgsz=640
        ```

    请注意，基准测试结果可能因系统的具体硬件和软件配置以及运行基准测试时的系统当前工作负载而异。为获得最可靠的结果，请使用包含大量图像的数据集，例如 `data='coco.yaml'`（5000 张验证图像）。

## NVIDIA DGX Spark 最佳实践

在使用 NVIDIA DGX Spark 时，遵循以下最佳实践以最大化 YOLO26 的运行性能。

1. **监控系统性能**

    使用 NVIDIA 的监控工具跟踪 GPU 和 CPU 利用率：

    ```bash
    nvidia-smi
    ```

2. **优化内存使用**

    凭借 128GB 的统一内存，DGX Spark 可以处理大批量大小和大型模型。考虑增加批量大小以提高吞吐量：

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.engine")
    results = model.predict(source="path/to/images", batch=16)
    ```

3. **使用 FP16 或 INT8 精度的 TensorRT**

    为获得最佳性能，使用 FP16 或 INT8 精度导出模型：

    ```bash
    yolo export model=yolo26n.pt format=engine half=True # FP16
    yolo export model=yolo26n.pt format=engine int8=True # INT8
    ```

## 系统更新（Founders Edition）

保持 DGX Spark Founders Edition 的最新状态对于性能和安全至关重要。NVIDIA 提供了两种主要方法来更新系统操作系统、驱动程序和固件。

### 使用 DGX 仪表板（推荐）

[DGX 仪表板](https://docs.nvidia.com/dgx/dgx-spark/dgx-dashboard.html)是执行系统更新以确保兼容性的推荐方式。它允许你：

- 查看可用的系统更新
- 安装安全补丁和系统更新
- 管理 NVIDIA 驱动程序和固件更新

### 手动系统更新

对于高级用户，可以通过终端手动执行更新：

```bash
sudo apt update
sudo apt dist-upgrade
sudo fwupdmgr refresh
sudo fwupdmgr upgrade
sudo reboot
```

!!! warning

    在执行更新之前，请确保系统连接到稳定的电源，并已备份关键数据。

## 下一步

如需进一步学习和支持，请参阅 [Ultralytics YOLO26 文档](../index.md)。

## 常见问题

### 如何在 NVIDIA DGX Spark 上部署 Ultralytics YOLO26？

在 NVIDIA DGX Spark 上部署 Ultralytics YOLO26 非常简单。你可以使用预构建的 Docker 镜像进行快速设置，也可以手动安装所需的软件包。每种方法的详细步骤可在[使用 Docker 快速入门](#使用-docker-快速入门)和[原生安装入门](#原生安装入门)部分找到。

### 在 NVIDIA DGX Spark 上 YOLO26 能达到什么样的性能？

得益于 GB10 Grace Blackwell 超级芯片，YOLO26 模型在 DGX Spark 上提供出色的性能。TensorRT 格式提供最佳的推理性能。请查看[详细对比表](#详细对比表)部分，了解不同模型大小和格式的具体基准测试结果。

### 为什么应该在 DGX Spark 上使用 TensorRT 运行 YOLO26？

由于 TensorRT 能提供最佳性能，强烈推荐在 DGX Spark 上使用 TensorRT 部署 YOLO26 模型。它通过利用 Blackwell GPU 的能力来加速推理，确保最大的效率和速度。详情请参阅[在 NVIDIA DGX Spark 上使用 TensorRT](#在-nvidia-dgx-spark-上使用-tensorrt)部分。

### DGX Spark 与 Jetson 设备在 YOLO26 上的表现如何比较？

DGX Spark 提供远超 Jetson 设备的计算能力，高达 1 PFLOP 的 AI 性能和 128GB 统一内存，而 Jetson AGX Thor 为 2070 TFLOPS 和 128GB 内存。DGX Spark 被设计为桌面 AI 超级计算机，而 Jetson 设备是为边缘部署优化的嵌入式系统。

### 可以在 DGX Spark 和 Jetson AGX Thor 上使用相同的 Docker 镜像吗？

可以！`ultralytics/ultralytics:latest-nvidia-arm64` Docker 镜像同时支持 NVIDIA DGX Spark（使用 DGX OS）和 Jetson AGX Thor（使用 JetPack 7.0），因为两者都使用 ARM64 架构、CUDA 13 和类似的软件栈。