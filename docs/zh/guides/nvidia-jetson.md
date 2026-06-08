---
comments: true
description: 通过本详细指南，学习在 NVIDIA Jetson 设备上部署 Ultralytics YOLO26。探索性能基准测试，最大化 AI 能力。
keywords: Ultralytics, YOLO26, NVIDIA Jetson, JetPack, AI 部署, 性能基准, 嵌入式系统, 深度学习, TensorRT, 计算机视觉
---

# 快速入门指南：NVIDIA Jetson 与 Ultralytics YOLO26

本综合指南提供了在 [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) 设备上部署 Ultralytics YOLO26 的详细步骤。此外还展示了性能基准，以体现 YOLO26 在这些小巧而强大的设备上的能力。

!!! tip "新品支持"

    我们已更新本指南，加入最新的 [NVIDIA Jetson AGX Thor 开发者套件](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor)，该套件提供高达 2070 FP4 TFLOPS 的 AI 算力和 128 GB 内存，功耗可在 40 W 至 130 W 之间配置。其 AI 算力比 NVIDIA Jetson AGX Orin 高出 7.5 倍以上，能效提升 3.5 倍，可流畅运行最流行的 AI 模型。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/BPYkGt3odNk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何在 NVIDIA Jetson 设备上使用 Ultralytics YOLO26
</p>

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/nvidia-jetson-ecosystem.avif" alt="NVIDIA Jetson 生态系统">

!!! note

    本指南已在以下设备上测试通过：[NVIDIA Jetson AGX Thor 开发者套件 (Jetson T5000)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor) 运行最新稳定版 [JP7.0](https://developer.nvidia.com/embedded/jetpack/downloads)、[NVIDIA Jetson AGX Orin 开发者套件 (64GB)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin) 运行 [JP6.2](https://developer.nvidia.com/embedded/jetpack-sdk-62)、[NVIDIA Jetson Orin Nano Super 开发者套件](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit) 运行 [JP6.1](https://developer.nvidia.com/embedded/jetpack-sdk-61)、[Seeed Studio reComputer J4012](https://www.seeedstudio.com/reComputer-J4012-p-5586.html)（基于 NVIDIA Jetson Orin NX 16GB）运行 [JP6.0](https://developer.nvidia.com/embedded/jetpack-sdk-60)/[JP5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513)，以及 [Seeed Studio reComputer J1020 v2](https://www.seeedstudio.com/reComputer-J1020-v2-p-5498.html)（基于 NVIDIA Jetson Nano 4GB）运行 [JP4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-461)。预计本指南适用于所有 NVIDIA Jetson 硬件系列，包括最新和旧款设备。

## 什么是 NVIDIA Jetson？

NVIDIA Jetson 是一系列嵌入式计算板卡，旨在将加速 AI（人工智能）计算带到边缘设备。这些紧凑而强大的设备基于 NVIDIA 的 GPU 架构构建，可以直接在设备上运行复杂的 AI 算法和[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，无需依赖[云计算](https://www.ultralytics.com/glossary/cloud-computing)资源。Jetson 板卡通常用于机器人、自动驾驶车辆、工业自动化以及其他需要低延迟、高效率本地执行 AI 推理的应用场景。此外，这些板卡基于 ARM64 架构，功耗低于传统的 GPU 计算设备。

## NVIDIA Jetson 系列对比

[NVIDIA Jetson AGX Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/) 是 NVIDIA Jetson 系列基于 NVIDIA Blackwell 架构的最新迭代产品，相比前代产品带来了大幅提升的 AI 性能。下表对比了生态系统中的几款 Jetson 设备。

|                   | Jetson AGX Thor(T5000)                                           | Jetson AGX Orin 64GB                                              | Jetson Orin NX 16GB                                              | Jetson Orin Nano Super                                        | Jetson AGX Xavier                                           | Jetson Xavier NX                                              | Jetson Nano                                   |
| ----------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------- |
| AI 性能            | 2070 TFLOPS                                                      | 275 TOPS                                                          | 100 TOPS                                                         | 67 TOPS                                                       | 32 TOPS                                                     | 21 TOPS                                                       | 472 GFLOPS                                    |
| GPU               | 2560 核 NVIDIA Blackwell 架构 GPU，96 个 Tensor Core              | 2048 核 NVIDIA Ampere 架构 GPU，64 个 Tensor Core                  | 1024 核 NVIDIA Ampere 架构 GPU，32 个 Tensor Core                 | 1024 核 NVIDIA Ampere 架构 GPU，32 个 Tensor Core              | 512 核 NVIDIA Volta 架构 GPU，64 个 Tensor Core              | 384 核 NVIDIA Volta™ 架构 GPU，48 个 Tensor Core               | 128 核 NVIDIA Maxwell™ 架构 GPU               |
| GPU 最大频率        | 1.57 GHz                                                         | 1.3 GHz                                                           | 918 MHz                                                          | 1020 MHz                                                      | 1377 MHz                                                    | 1100 MHz                                                      | 921MHz                                        |
| CPU               | 14 核 Arm® Neoverse®-V3AE 64 位 CPU，1MB L2 + 16MB L3            | 12 核 NVIDIA Arm® Cortex A78AE v8.2 64 位 CPU，3MB L2 + 6MB L3   | 8 核 NVIDIA Arm® Cortex A78AE v8.2 64 位 CPU，2MB L2 + 4MB L3    | 6 核 Arm® Cortex®-A78AE v8.2 64 位 CPU，1.5MB L2 + 4MB L3     | 8 核 NVIDIA Carmel Arm®v8.2 64 位 CPU，8MB L2 + 4MB L3       | 6 核 NVIDIA Carmel Arm®v8.2 64 位 CPU，6MB L2 + 4MB L3         | 四核 Arm® Cortex®-A57 MPCore 处理器           |
| CPU 最大频率        | 2.6 GHz                                                          | 2.2 GHz                                                           | 2.0 GHz                                                          | 1.7 GHz                                                       | 2.2 GHz                                                     | 1.9 GHz                                                       | 1.43GHz                                       |
| 内存               | 128GB 256 位 LPDDR5X 273GB/s                                    | 64GB 256 位 LPDDR5 204.8GB/s                                     | 16GB 128 位 LPDDR5 102.4GB/s                                     | 8GB 128 位 LPDDR5 102 GB/s                                    | 32GB 256 位 LPDDR4x 136.5GB/s                               | 8GB 128 位 LPDDR4x 59.7GB/s                                   | 4GB 64 位 LPDDR4 25.6GB/s                     |

更详细的对比表请访问 [NVIDIA Jetson 官方页面](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems) 的 **对比规格** 部分。

## 什么是 NVIDIA JetPack？

为 Jetson 模块提供动力的 [NVIDIA JetPack SDK](https://developer.nvidia.com/embedded/jetpack) 是最全面的解决方案，提供了构建端到端加速 AI 应用的完整开发环境，缩短产品上市时间。JetPack 包含 Jetson Linux（含引导程序）、Linux 内核、Ubuntu 桌面环境，以及用于加速 GPU 计算、多媒体、图形和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)的完整库集合。它还包括示例、文档和面向主机及开发者套件的开发工具，并支持更高级别的 SDK，如用于流视频分析的 [DeepStream](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson)、用于机器人的 Isaac 和用于对话式 AI 的 Riva。

## 将 JetPack 烧录到 NVIDIA Jetson

拿到 NVIDIA Jetson 设备后的第一步是将 NVIDIA JetPack 烧录到设备中。有几种不同的方式可以烧录 NVIDIA Jetson 设备。

1. 如果您拥有官方 NVIDIA 开发者套件（如 Jetson AGX Thor 开发者套件），可以[下载镜像并准备一个可启动的 USB 盘，将 JetPack 烧录到内置 SSD](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html)。
2. 如果您拥有官方 NVIDIA 开发者套件（如 Jetson Orin Nano 开发者套件），可以[下载镜像并准备一张 SD 卡用于启动设备](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)。
3. 如果您拥有其他 NVIDIA 开发者套件，可以[使用 SDK Manager 将 JetPack 烧录到设备](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html)。
4. 如果您拥有 Seeed Studio reComputer J4012 设备，可以[将 JetPack 烧录到内置 SSD](https://wiki.seeedstudio.com/reComputer_J4012_Flash_Jetpack/)，如果您拥有 Seeed Studio reComputer J1020 v2 设备，可以[将 JetPack 烧录到 eMMC/SSD](https://wiki.seeedstudio.com/reComputer_J2021_J202_Flash_Jetpack/)。
5. 如果您拥有基于 NVIDIA Jetson 模块的其他第三方设备，建议遵循[命令行烧录方式](https://docs.nvidia.com/jetson/archives/r35.5.0/DeveloperGuide/IN/QuickStart.html)。

!!! note

    对于上述方法 1、4 和 5，烧录系统并启动设备后，请在设备终端输入 `sudo apt update && sudo apt install nvidia-jetpack -y` 以安装所有剩余的 JetPack 组件。

## 基于 Jetson 设备的 JetPack 支持

下表列出了不同 NVIDIA Jetson 设备支持的 NVIDIA JetPack 版本。

|                   | JetPack 4 | JetPack 5 | JetPack 6 | JetPack 7 |
| ----------------- | --------- | --------- | --------- | --------- |
| Jetson Nano       | ✅        | ❌        | ❌        | ❌        |
| Jetson TX2        | ✅        | ❌        | ❌        | ❌        |
| Jetson Xavier NX  | ✅        | ✅        | ❌        | ❌        |
| Jetson AGX Xavier | ✅        | ✅        | ❌        | ❌        |
| Jetson AGX Orin   | ❌        | ✅        | ✅        | ❌        |
| Jetson Orin NX    | ❌        | ✅        | ✅        | ❌        |
| Jetson Orin Nano  | ❌        | ✅        | ✅        | ❌        |
| Jetson AGX Thor   | ❌        | ❌        | ❌        | ✅        |

## 使用 Docker 快速入门

在 NVIDIA Jetson 上使用 Ultralytics YOLO26 的最快方式是使用预构建的 Jetson Docker 镜像。请参考上表，根据您拥有的 Jetson 设备选择 JetPack 版本。

=== "JetPack 4"

    ```bash
    t=ultralytics/ultralytics:latest-jetson-jetpack4
    sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t
    ```

=== "JetPack 5"

    ```bash
    t=ultralytics/ultralytics:latest-jetson-jetpack5
    sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t
    ```

=== "JetPack 6"

    ```bash
    t=ultralytics/ultralytics:latest-jetson-jetpack6
    sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t
    ```

=== "JetPack 7"

    ```bash
    t=ultralytics/ultralytics:latest-nvidia-arm64
    sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t
    ```

完成后，跳转到 [在 NVIDIA Jetson 上使用 TensorRT](#在-nvidia-jetson-上使用-tensorrt) 部分。

## 通过原生安装开始

如需不使用 Docker 的原生安装，请参考以下步骤。

### 在 JetPack 7.0 上运行

#### 安装 Ultralytics 包

我们将在 Jetson 上安装 Ultralytics 包及其可选依赖，以便将 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 模型导出为其他格式。我们将主要关注 [NVIDIA TensorRT 导出](../integrations/tensorrt.md)，因为 TensorRT 能确保我们在 Jetson 设备上获得最佳性能。

1. 更新软件包列表，安装 pip 并升级至最新

    ```bash
    sudo apt update
    sudo apt install python3-pip -y
    pip install -U pip
    ```

2. 安装 `ultralytics` pip 包及可选依赖

    ```bash
    pip install ultralytics[export]
    ```

3. 重启设备

    ```bash
    sudo reboot
    ```

#### 安装 PyTorch 和 Torchvision

上述 ultralytics 安装将安装 Torch 和 Torchvision。然而，通过 pip 安装的这两个包与搭载 JetPack 7.0 和 CUDA 13 的 Jetson AGX Thor 不兼容。因此，我们需要手动安装它们。

根据 JP7.0 安装 `torch` 和 `torchvision`

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

#### 安装 `onnxruntime-gpu`

PyPI 上托管的 [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/) 包没有适用于 Jetson 的 `aarch64` 二进制文件。因此我们需要手动安装此包。某些导出操作需要此包。

这里我们将下载并安装支持 `Python3.12` 的 `onnxruntime-gpu 1.24.0`。

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.24.0-cp312-cp312-linux_aarch64.whl
```

### 在 JetPack 6.1 上运行

#### 安装 Ultralytics 包

我们将在 Jetson 上安装 Ultralytics 包及其可选依赖，以便将 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 模型导出为其他格式。我们将主要关注 [NVIDIA TensorRT 导出](../integrations/tensorrt.md)，因为 TensorRT 能确保我们在 Jetson 设备上获得最佳性能。

1. 更新软件包列表，安装 pip 并升级至最新

    ```bash
    sudo apt update
    sudo apt install python3-pip -y
    pip install -U pip
    ```

2. 安装 `ultralytics` pip 包及可选依赖

    ```bash
    pip install ultralytics[export]
    ```

3. 重启设备

    ```bash
    sudo reboot
    ```

#### 安装 PyTorch 和 Torchvision

上述 ultralytics 安装将安装 Torch 和 Torchvision。然而，通过 pip 安装的这两个包与基于 ARM64 架构的 Jetson 平台不兼容。因此，我们需要手动安装预构建的 PyTorch pip wheel，并从源码编译或安装 Torchvision。

根据 JP6.1 安装 `torch 2.10.0` 和 `torchvision 0.25.0`

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.10.0-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl
```

!!! note

    访问 [PyTorch for Jetson 页面](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) 获取不同 JetPack 版本的所有 PyTorch 版本。有关 PyTorch、Torchvision 兼容性的更详细列表，请访问 [PyTorch 与 Torchvision 兼容性页面](https://github.com/pytorch/vision)。

安装 [`cuDSS`](https://developer.nvidia.com/cudss-downloads?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) 以解决 `torch 2.10.0` 的依赖问题

```bash
wget https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudss
```

#### 安装 `onnxruntime-gpu`

PyPI 上托管的 [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/) 包没有适用于 Jetson 的 `aarch64` 二进制文件。因此我们需要手动安装此包。某些导出操作需要此包。

您可以在 [Jetson Zoo ONNX Runtime 兼容性矩阵](https://elinux.org/Jetson_Zoo#ONNX_Runtime) 中找到所有可用的 `onnxruntime-gpu` 包，按 JetPack 版本、Python 版本和其他兼容性详细信息排列。

对于 **JetPack 6** 及 `Python 3.10` 支持，可以安装 `onnxruntime-gpu 1.23.0`：

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
```

或者安装 `onnxruntime-gpu 1.20.0`：

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
```

### 在 JetPack 5.1.2 上运行

#### 安装 Ultralytics 包

我们将在 Jetson 上安装 Ultralytics 包及其可选依赖，以便将 PyTorch 模型导出为其他格式。我们将主要关注 [NVIDIA TensorRT 导出](../integrations/tensorrt.md)，因为 TensorRT 能确保我们在 Jetson 设备上获得最佳性能。

1. 更新软件包列表，安装 pip 并升级至最新

    ```bash
    sudo apt update
    sudo apt install python3-pip -y
    pip install -U pip
    ```

2. 安装 `ultralytics` pip 包及可选依赖

    ```bash
    pip install ultralytics[export]
    ```

3. 重启设备

    ```bash
    sudo reboot
    ```

#### 安装 PyTorch 和 Torchvision

上述 ultralytics 安装将安装 Torch 和 Torchvision。然而，通过 pip 安装的这两个包与基于 ARM64 架构的 Jetson 平台不兼容。因此，我们需要手动安装预构建的 PyTorch pip wheel，并从源码编译或安装 Torchvision。

1. 卸载当前安装的 PyTorch 和 Torchvision

    ```bash
    pip uninstall torch torchvision
    ```

2. 根据 JP5.1.2 安装 `torch 2.1.0` 和 `torchvision 0.16.2`

    ```bash
    pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
    pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.16.2+c6f3977-cp38-cp38-linux_aarch64.whl
    ```

!!! note

    访问 [PyTorch for Jetson 页面](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) 获取不同 JetPack 版本的所有 PyTorch 版本。有关 PyTorch、Torchvision 兼容性的更详细列表，请访问 [PyTorch 与 Torchvision 兼容性页面](https://github.com/pytorch/vision)。

#### 安装 `onnxruntime-gpu`

PyPI 上托管的 [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/) 包没有适用于 Jetson 的 `aarch64` 二进制文件。因此我们需要手动安装此包。某些导出操作需要此包。

您可以在 [Jetson Zoo ONNX Runtime 兼容性矩阵](https://elinux.org/Jetson_Zoo#ONNX_Runtime) 中找到所有可用的 `onnxruntime-gpu` 包，按 JetPack 版本、Python 版本和其他兼容性详细信息排列。这里我们将下载并安装支持 `Python3.8` 的 `onnxruntime-gpu 1.17.0`。

```bash
wget https://nvidia.box.com/shared/static/zostg6agm00fb6t5uisw51qi6kpcuwzd.whl -O onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl
pip install onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl
```

!!! note

    `onnxruntime-gpu` 会自动将 NumPy 版本回退到最新版本。因此我们需要重新安装 `1.23.5` 版本的 NumPy 以修复一个问题，执行：

    `pip install numpy==1.23.5`

## 在 NVIDIA Jetson 上使用 TensorRT

在 Ultralytics 支持的所有模型导出格式中，TensorRT 在 NVIDIA Jetson 设备上提供最高的推理性能，是我们推荐的首选 Jetson 部署方案。有关设置说明和高级用法，请参阅我们的 [专用 TensorRT 集成指南](../integrations/tensorrt.md)。

### 将模型转换为 TensorRT 并运行推理

将 PyTorch 格式的 YOLO26n 模型转换为 TensorRT 格式，使用导出后的模型运行推理。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO26n PyTorch 模型
        model = YOLO("yolo26n.pt")

        # 将模型导出为 TensorRT 格式
        model.export(format="engine")  # 创建 'yolo26n.engine'

        # 加载导出后的 TensorRT 模型
        trt_model = YOLO("yolo26n.engine")

        # 运行推理
        results = trt_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO26n PyTorch 模型导出为 TensorRT 格式
        yolo export model=yolo26n.pt format=engine # 创建 'yolo26n.engine'

        # 使用导出后的模型运行推理
        yolo predict model=yolo26n.engine source='https://ultralytics.com/images/bus.jpg'
        ```

!!! note

    访问 [导出页面](../modes/export.md#arguments) 查看将模型导出为不同格式时的其他参数。

### 使用 NVIDIA 深度学习加速器 (DLA)

[NVIDIA 深度学习加速器 (DLA)](https://developer.nvidia.com/deep-learning-accelerator) 是内置于 NVIDIA Jetson 设备中的专用硬件组件，可优化深度学习推理的能效和性能。通过将任务从 GPU 卸载（释放 GPU 用于更密集的处理），DLA 使模型能够以更低的功耗运行，同时保持高吞吐量，非常适合嵌入式系统和实时 AI 应用。

以下 Jetson 设备配备了 DLA 硬件：

| Jetson 设备               | DLA 核心数 | DLA 最大频率 |
| ------------------------- | ---------- | ------------ |
| Jetson AGX Orin 系列       | 2          | 1.6 GHz      |
| Jetson Orin NX 16GB       | 2          | 614 MHz      |
| Jetson Orin NX 8GB        | 1          | 614 MHz      |
| Jetson AGX Xavier 系列    | 2          | 1.4 GHz      |
| Jetson Xavier NX 系列     | 2          | 1.1 GHz      |

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO26n PyTorch 模型
        model = YOLO("yolo26n.pt")

        # 将模型导出为启用 DLA 的 TensorRT 格式（仅支持 FP16 或 INT8）
        model.export(format="engine", device="dla:0", half=True)  # dla:0 或 dla:1 对应 DLA 核心

        # 加载导出后的 TensorRT 模型
        trt_model = YOLO("yolo26n.engine")

        # 运行推理
        results = trt_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO26n PyTorch 模型导出为启用 DLA 的 TensorRT 格式（仅支持 FP16 或 INT8）
        # 在导出时指定 DLA 核心号后，推理时将使用相同的核心
        yolo export model=yolo26n.pt format=engine device="dla:0" half=True # dla:0 或 dla:1 对应 DLA 核心

        # 在 DLA 上使用导出后的模型运行推理
        yolo predict model=yolo26n.engine source='https://ultralytics.com/images/bus.jpg'
        ```

!!! note

    使用 DLA 导出时，某些层可能不支持在 DLA 上运行，将回退到 GPU 执行。这种回退可能引入额外的延迟并影响整体推理性能。因此，DLA 的主要设计目的并非减少相对于完全在 GPU 上运行 TensorRT 的推理延迟，而是提高吞吐量并改善能效。

## NVIDIA Jetson YOLO11/YOLO26 基准测试

YOLO11/YOLO26 基准测试由 Ultralytics 团队在 11 种不同模型格式上运行，测量速度和[准确率](https://www.ultralytics.com/glossary/accuracy)：PyTorch、TorchScript、ONNX、OpenVINO、TensorRT、TF SavedModel、TF GraphDef、TF Lite、MNN、NCNN、ExecuTorch。基准测试在 NVIDIA Jetson AGX Thor 开发者套件、NVIDIA Jetson AGX Orin 开发者套件 (64GB)、NVIDIA Jetson Orin Nano Super 开发者套件以及基于 Jetson Orin NX 16GB 的 Seeed Studio reComputer J4012 设备上运行，使用 FP32 [精度](https://www.ultralytics.com/glossary/precision)，默认输入图像尺寸为 640。

### 对比图表

虽然所有模型导出格式都能在 NVIDIA Jetson 上运行，但在下方对比图表中我们仅包含 **PyTorch、TorchScript、TensorRT**，因为它们利用了 Jetson 上的 GPU，能够保证最佳效果。所有其他导出格式仅使用 CPU，性能不如上述三种。您可以在本图表后面的部分找到所有导出格式的基准测试。

#### NVIDIA Jetson AGX Thor 开发者套件

<figure style="text-align: center;">
    <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/jetson-agx-thor-benchmarks-coco128.avif" alt="Jetson AGX Thor 基准测试">
    <figcaption style="font-style: italic; color: gray;">使用 Ultralytics 8.3.226 进行基准测试</figcaption>
</figure>

#### NVIDIA Jetson AGX Orin 开发者套件 (64GB)

<figure style="text-align: center;">
    <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/jetson-agx-orin-benchmarks-coco128.avif" alt="Jetson AGX Orin 基准测试">
    <figcaption style="font-style: italic; color: gray;">使用 Ultralytics 8.4.32 进行基准测试</figcaption>
</figure>

#### NVIDIA Jetson Orin Nano Super 开发者套件

<figure style="text-align: center;">
    <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/jetson-orin-nano-super-benchmarks-coco128.avif" alt="Jetson Orin Nano Super 基准测试">
    <figcaption style="font-style: italic; color: gray;">使用 Ultralytics 8.4.33 进行基准测试</figcaption>
</figure>

#### NVIDIA Jetson Orin NX 16GB

<figure style="text-align: center;">
    <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/jetson-orin-nx-16-benchmarks-coco128.avif" alt="Jetson Orin NX 16GB 基准测试">
    <figcaption style="font-style: italic; color: gray;">使用 Ultralytics 8.4.33 进行基准测试</figcaption>
</figure>

### 详细对比表

下表展示了五种不同模型（YOLO11n、YOLO11s、YOLO11m、YOLO11l、YOLO11x）在 11 种不同格式（PyTorch、TorchScript、ONNX、OpenVINO、TensorRT、TF SavedModel、TF GraphDef、TF Lite、MNN、NCNN、ExecuTorch）下的基准测试结果，包含每种组合的状态、磁盘大小、mAP50-95(B) 指标和推理时间。

#### NVIDIA Jetson AGX Thor 开发者套件

!!! tip "性能"

    === "YOLO26n"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 5.3          | 0.4798     | 7.39            |
        | TorchScript     | ✅   | 9.8          | 0.4789     | 4.21            |
        | ONNX            | ✅   | 9.5          | 0.4767     | 6.58            |
        | OpenVINO        | ✅   | 10.1         | 0.4794     | 17.50           |
        | TensorRT (FP32) | ✅   | 13.9         | 0.4791     | 1.90            |
        | TensorRT (FP16) | ✅   | 7.6          | 0.4797     | 1.39            |
        | TensorRT (INT8) | ✅   | 6.5          | 0.4273     | 1.52            |
        | TF SavedModel   | ✅   | 25.7         | 0.4764     | 47.24           |
        | TF GraphDef     | ✅   | 9.5          | 0.4764     | 45.98           |
        | TF Lite         | ✅   | 9.9          | 0.4764     | 182.04          |
        | MNN             | ✅   | 9.4          | 0.4784     | 21.83           |

    === "YOLO26s"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 19.5         | 0.5738     | 7.99            |
        | TorchScript     | ✅   | 36.8         | 0.5664     | 6.01            |
        | ONNX            | ✅   | 36.5         | 0.5666     | 9.31            |
        | OpenVINO        | ✅   | 38.5         | 0.5656     | 35.56           |
        | TensorRT (FP32) | ✅   | 38.9         | 0.5664     | 2.95            |
        | TensorRT (FP16) | ✅   | 21.0         | 0.5650     | 1.77            |
        | TensorRT (INT8) | ✅   | 13.5         | 0.5010     | 1.75            |
        | TF SavedModel   | ✅   | 96.6         | 0.5665     | 88.87           |
        | TF GraphDef     | ✅   | 36.5         | 0.5665     | 89.20           |
        | TF Lite         | ✅   | 36.9         | 0.5665     | 604.25          |
        | MNN             | ✅   | 36.4         | 0.5651     | 53.75           |

    === "YOLO26m"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 42.2         | 0.6237     | 10.76           |
        | TorchScript     | ✅   | 78.5         | 0.6217     | 10.57           |
        | ONNX            | ✅   | 78.2         | 0.6211     | 14.91           |
        | OpenVINO        | ✅   | 82.2         | 0.6204     | 86.27           |
        | TensorRT (FP32) | ✅   | 82.2         | 0.6230     | 5.56            |
        | TensorRT (FP16) | ✅   | 41.6         | 0.6209     | 2.58            |
        | TensorRT (INT8) | ✅   | 24.3         | 0.5595     | 2.49            |
        | TF SavedModel   | ✅   | 205.8        | 0.6229     | 200.96          |
        | TF GraphDef     | ✅   | 78.2         | 0.6229     | 203.00          |
        | TF Lite         | ✅   | 78.6         | 0.6229     | 1867.12         |
        | MNN             | ✅   | 78.0         | 0.6176     | 142.00          |

    === "YOLO26l"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 50.7         | 0.6258     | 13.34           |
        | TorchScript     | ✅   | 95.5         | 0.6248     | 13.86           |
        | ONNX            | ✅   | 95.0         | 0.6247     | 18.44           |
        | OpenVINO        | ✅   | 99.9         | 0.6238     | 106.67          |
        | TensorRT (FP32) | ✅   | 99.0         | 0.6249     | 6.74            |
        | TensorRT (FP16) | ✅   | 50.3         | 0.6243     | 3.34            |
        | TensorRT (INT8) | ✅   | 29.0         | 0.5708     | 3.24            |
        | TF SavedModel   | ✅   | 250.0        | 0.6245     | 259.74          |
        | TF GraphDef     | ✅   | 95.0         | 0.6245     | 263.42          |
        | TF Lite         | ✅   | 95.4         | 0.6245     | 2367.83         |
        | MNN             | ✅   | 94.8         | 0.6272     | 174.39          |

    === "YOLO26x"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 113.2        | 0.6565     | 20.92           |
        | TorchScript     | ✅   | 213.5        | 0.6595     | 21.76           |
        | ONNX            | ✅   | 212.9        | 0.6590     | 26.72           |
        | OpenVINO        | ✅   | 223.6        | 0.6620     | 205.27          |
        | TensorRT (FP32) | ✅   | 217.2        | 0.6593     | 12.29           |
        | TensorRT (FP16) | ✅   | 112.1        | 0.6611     | 5.16            |
        | TensorRT (INT8) | ✅   | 58.9         | 0.5222     | 4.72            |
        | TF SavedModel   | ✅   | 559.2        | 0.6593     | 498.85          |
        | TF GraphDef     | ✅   | 213.0        | 0.6593     | 507.43          |
        | TF Lite         | ✅   | 213.3        | 0.6593     | 5134.22         |
        | MNN             | ✅   | 212.8        | 0.6625     | 347.84          |

    使用 Ultralytics 8.4.7 进行基准测试

    !!! note

        推理时间不包括前/后处理。

#### NVIDIA Jetson AGX Orin 开发者套件 (64GB)

!!! tip "性能"

    === "YOLO26n"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 5.3          | 0.4790     | 11.58           |
        | TorchScript     | ✅   | 9.8          | 0.4770     | 4.60            |
        | ONNX            | ✅   | 9.5          | 0.4770     | 9.87            |
        | OpenVINO        | ✅   | 9.6          | 0.4820     | 28.80           |
        | TensorRT (FP32) | ✅   | 11.5         | 0.0450     | 4.18            |
        | TensorRT (FP16) | ✅   | 7.9          | 0.0450     | 2.62            |
        | TensorRT (INT8) | ✅   | 5.4          | 0.4640     | 2.30            |
        | TF SavedModel   | ✅   | 24.6         | 0.4760     | 71.10           |
        | TF GraphDef     | ✅   | 9.5          | 0.4760     | 70.02           |
        | TF Lite         | ✅   | 9.9          | 0.4760     | 227.94          |
        | MNN             | ✅   | 9.4          | 0.4760     | 32.46           |
        | NCNN            | ✅   | 9.3          | 0.4810     | 29.93           |

    === "YOLO26s"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 20.0         | 0.5730     | 13.18           |
        | TorchScript     | ✅   | 36.8         | 0.5670     | 11.48           |
        | ONNX            | ✅   | 36.5         | 0.5660     | 13.47           |
        | OpenVINO        | ✅   | 36.7         | 0.5650     | 58.30           |
        | TensorRT (FP32) | ✅   | 38.5         | 0.5660     | 6.82            |
        | TensorRT (FP16) | ✅   | 21.9         | 0.5660     | 3.76            |
        | TensorRT (INT8) | ✅   | 12.5         | 0.5480     | 2.98            |
        | TF SavedModel   | ✅   | 92.2         | 0.5660     | 145.62          |
        | TF GraphDef     | ✅   | 36.5         | 0.5660     | 146.26          |
        | TF Lite         | ✅   | 36.9         | 0.5660     | 753.52          |
        | MNN             | ✅   | 36.4         | 0.5650     | 79.50           |
        | NCNN            | ✅   | 36.4         | 0.5700     | 58.73           |

    === "YOLO26m"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 43.0         | 0.6220     | 19.36           |
        | TorchScript     | ✅   | 78.5         | 0.6230     | 20.02           |
        | ONNX            | ✅   | 78.2         | 0.6230     | 25.40           |
        | OpenVINO        | ✅   | 78.3         | 0.6190     | 130.76          |
        | TensorRT (FP32) | ✅   | 80.2         | 0.6220     | 12.60           |
        | TensorRT (FP16) | ✅   | 42.5         | 0.6220     | 6.24            |
        | TensorRT (INT8) | ✅   | 23.4         | 0.5820     | 4.72            |
        | TF SavedModel   | ✅   | 196.3        | 0.6230     | 306.76          |
        | TF GraphDef     | ✅   | 78.2         | 0.6230     | 314.23          |
        | TF Lite         | ✅   | 78.5         | 0.6230     | 2331.63         |
        | MNN             | ✅   | 78.0         | 0.6220     | 206.93          |
        | NCNN            | ✅   | 78.0         | 0.6220     | 143.03          |

    === "YOLO26l"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 51.0         | 0.6230     | 23.53           |
        | TorchScript     | ✅   | 95.5         | 0.6250     | 24.23           |
        | ONNX            | ✅   | 95.0         | 0.6250     | 31.73           |
        | OpenVINO        | ✅   | 95.3         | 0.6240     | 162.80          |
        | TensorRT (FP32) | ✅   | 97.3         | 0.6250     | 15.90           |
        | TensorRT (FP16) | ✅   | 51.4         | 0.6240     | 7.93            |
        | TensorRT (INT8) | ✅   | 29.9         | 0.5920     | 5.97            |
        | TF SavedModel   | ✅   | 238.4        | 0.6250     | 394.30          |
        | TF GraphDef     | ✅   | 95.0         | 0.6250     | 398.63          |
        | TF Lite         | ✅   | 95.4         | 0.6250     | 2925.27         |
        | MNN             | ✅   | 94.8         | 0.6250     | 255.87          |
        | NCNN            | ✅   | 94.8         | 0.6320     | 177.70          |

    === "YOLO26x"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 114          | 0.6610     | 38.37           |
        | TorchScript     | ✅   | 213.5        | 0.6590     | 41.23           |
        | ONNX            | ✅   | 212.9        | 0.6590     | 52.03           |
        | OpenVINO        | ✅   | 213.2        | 0.6590     | 300.40          |
        | TensorRT (FP32) | ✅   | 215.2        | 0.6590     | 28.43           |
        | TensorRT (FP16) | ✅   | 110.3        | 0.6570     | 13.50           |
        | TensorRT (INT8) | ✅   | 59.9         | 0.6080     | 9.33            |
        | TF SavedModel   | ✅   | 533.3        | 0.6590     | 738.60          |
        | TF GraphDef     | ✅   | 212.9        | 0.6590     | 785.70          |
        | TF Lite         | ✅   | 217.6        | 0.6900     | 6476.80         |
        | MNN             | ✅   | 213.3        | 0.6590     | 519.77          |
        | NCNN            | ✅   | 212.8        | 0.6670     | 300.00          |

    使用 Ultralytics 8.4.32 进行基准测试

    !!! note

        推理时间不包括前/后处理。

#### NVIDIA Jetson Orin Nano Super 开发者套件

!!! tip "性能"

    === "YOLO26n"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 5.3          | 0.4790     | 15.60           |
        | TorchScript     | ✅   | 9.8          | 0.4770     | 12.60           |
        | ONNX            | ✅   | 9.5          | 0.4760     | 15.76           |
        | OpenVINO        | ✅   | 9.6          | 0.4820     | 56.23           |
        | TensorRT (FP32) | ✅   | 11.3         | 0.4770     | 7.53            |
        | TensorRT (FP16) | ✅   | 8.1          | 0.4800     | 4.57            |
        | TensorRT (INT8) | ✅   | 5.3          | 0.4490     | 3.80            |
        | TF SavedModel   | ✅   | 24.6         | 0.4760     | 118.33          |
        | TF GraphDef     | ✅   | 9.5          | 0.4760     | 116.30          |
        | TF Lite         | ✅   | 9.9          | 0.4760     | 286.00          |
        | MNN             | ✅   | 9.4          | 0.4760     | 68.77           |
        | NCNN            | ✅   | 9.3          | 0.4810     | 47.50           |

    === "YOLO26s"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 20.0         | 0.5730     | 22.83           |
        | TorchScript     | ✅   | 36.8         | 0.5670     | 21.83           |
        | ONNX            | ✅   | 36.5         | 0.5664     | 26.29           |
        | OpenVINO        | ✅   | 36.7         | 0.5653     | 127.09          |
        | TensorRT (FP32) | ✅   | 38.2         | 0.5664     | 13.60           |
        | TensorRT (FP16) | ✅   | 21.3         | 0.5649     | 7.17            |
        | TensorRT (INT8) | ✅   | 12.7         | 0.5468     | 5.25            |
        | TF SavedModel   | ✅   | 92.2         | 0.5665     | 263.69          |
        | TF GraphDef     | ✅   | 36.5         | 0.5665     | 268.21          |
        | TF Lite         | ✅   | 36.9         | 0.5665     | 949.63          |
        | MNN             | ✅   | 36.4         | 0.5644     | 184.68          |
        | NCNN            | ✅   | 36.4         | 0.5697     | 107.48          |

    === "YOLO26m"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 43.0         | 0.6220     | 44.43           |
        | TorchScript     | ✅   | 78.5         | 0.6230     | 44.00           |
        | ONNX            | ✅   | 78.2         | 0.6225     | 53.44           |
        | OpenVINO        | ✅   | 78.3         | 0.6186     | 303.26          |
        | TensorRT (FP32) | ✅   | 80.0         | 0.6217     | 28.19           |
        | TensorRT (FP16) | ✅   | 42.6         | 0.6225     | 13.59           |
        | TensorRT (INT8) | ✅   | 23.4         | 0.5817     | 9.30            |
        | TF SavedModel   | ✅   | 196.3        | 0.6229     | 636.03          |
        | TF GraphDef     | ✅   | 78.2         | 0.6229     | 659.57          |
        | TF Lite         | ✅   | 78.5         | 0.6229     | 2905.17         |
        | MNN             | ✅   | 78.0         | 0.6168     | 500.09          |
        | NCNN            | ✅   | 78.0         | 0.6224     | 332.39          |

    === "YOLO26l"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 51.0         | 0.6230     | 60.97           |
        | TorchScript     | ✅   | 95.5         | 0.6250     | 56.20           |
        | ONNX            | ✅   | 95.0         | 0.6247     | 68.12           |
        | OpenVINO        | ✅   | 95.3         | 0.6238     | 397.84          |
        | TensorRT (FP32) | ✅   | 97.1         | 0.6250     | 35.88           |
        | TensorRT (FP16) | ✅   | 51.4         | 0.6225     | 17.42           |
        | TensorRT (INT8) | ✅   | 30.0         | 0.5923     | 11.83           |
        | TF SavedModel   | ✅   | 238.4        | 0.6245     | 835.83          |
        | TF GraphDef     | ✅   | 95.0         | 0.6245     | 852.16          |
        | TF Lite         | ✅   | 95.4         | 0.6245     | 3650.85         |
        | MNN             | ✅   | 94.8         | 0.6257     | 612.37          |
        | NCNN            | ✅   | 94.8         | 0.6323     | 405.45          |

    === "YOLO26x"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 113.2        | 0.6561     | 98.44           |
        | TorchScript     | ✅   | 214.0        | 0.6593     | 98.0            |
        | ONNX            | ✅   | 212.9        | 0.6595     | 122.43          |
        | OpenVINO        | ✅   | 213.2        | 0.6592     | 760.72          |
        | TensorRT (FP32) | ✅   | 215.1        | 0.6593     | 67.17           |
        | TensorRT (FP16) | ✅   | 110.2        | 0.6637     | 32.60           |
        | TensorRT (INT8) | ✅   | 59.9         | 0.6170     | 19.99           |
        | TF SavedModel   | ✅   | 533.3        | 0.6593     | 1647.06         |
        | TF GraphDef     | ✅   | 212.9        | 0.6593     | 1670.30         |
        | TF Lite         | ✅   | 213.3        | 0.6590     | 8066.30         |
        | MNN             | ✅   | 212.8        | 0.6600     | 1227.90         |
        | NCNN            | ✅   | 212.8        | 0.6666     | 782.24          |

    使用 Ultralytics 8.4.33 进行基准测试

    !!! note

        推理时间不包括前/后处理。

#### NVIDIA Jetson Orin NX 16GB

!!! tip "性能"

    === "YOLO26n"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 5.3          | 0.4799     | 13.90           |
        | TorchScript     | ✅   | 9.8          | 0.4787     | 11.60           |
        | ONNX            | ✅   | 9.5          | 0.4763     | 14.18           |
        | OpenVINO        | ✅   | 9.6          | 0.4819     | 40.19           |
        | TensorRT (FP32) | ✅   | 11.4         | 0.4770     | 7.01            |
        | TensorRT (FP16) | ✅   | 8.0          | 0.4789     | 4.13            |
        | TensorRT (INT8) | ✅   | 5.5          | 0.4489     | 3.49            |
        | TF SavedModel   | ✅   | 24.6         | 0.4764     | 92.34           |
        | TF GraphDef     | ✅   | 9.5          | 0.4764     | 92.06           |
        | TF Lite         | ✅   | 9.9          | 0.4764     | 254.43          |
        | MNN             | ✅   | 9.4          | 0.4760     | 48.55           |
        | NCNN            | ✅   | 9.3          | 0.4805     | 34.31           |

    === "YOLO26s"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 19.5         | 0.5738     | 20.40           |
        | TorchScript     | ✅   | 36.8         | 0.5664     | 19.20           |
        | ONNX            | ✅   | 36.5         | 0.5664     | 24.35           |
        | OpenVINO        | ✅   | 36.7         | 0.5653     | 88.18           |
        | TensorRT (FP32) | ✅   | 38.5         | 0.5664     | 12.62           |
        | TensorRT (FP16) | ✅   | 21.5         | 0.5652     | 6.41            |
        | TensorRT (INT8) | ✅   | 12.6         | 0.5468     | 4.78            |
        | TF SavedModel   | ✅   | 92.2         | 0.5665     | 195.16          |
        | TF GraphDef     | ✅   | 36.5         | 0.5665     | 197.57          |
        | TF Lite         | ✅   | 36.9         | 0.5665     | 827.48          |
        | MNN             | ✅   | 36.4         | 0.5649     | 123.47          |
        | NCNN            | ✅   | 36.4         | 0.5697     | 74.04           |

    === "YOLO26m"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 42.2         | 0.6237     | 38.60           |
        | TorchScript     | ✅   | 78.5         | 0.6227     | 40.50           |
        | ONNX            | ✅   | 78.2         | 0.6225     | 48.87           |
        | OpenVINO        | ✅   | 78.3         | 0.6186     | 205.69          |
        | TensorRT (FP32) | ✅   | 80.1         | 0.6217     | 24.69           |
        | TensorRT (FP16) | ✅   | 42.6         | 0.6225     | 11.66           |
        | TensorRT (INT8) | ✅   | 23.4         | 0.5817     | 8.22            |
        | TF SavedModel   | ✅   | 196.3        | 0.6229     | 451.48          |
        | TF GraphDef     | ✅   | 78.2         | 0.6229     | 460.94          |
        | TF Lite         | ✅   | 78.5         | 0.6229     | 2555.53         |
        | MNN             | ✅   | 78.0         | 0.6217     | 333.33          |
        | NCNN            | ✅   | 78.0         | 0.6224     | 214.60          |

    === "YOLO26l"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 50.7         | 0.6258     | 48.60           |
        | TorchScript     | ✅   | 95.5         | 0.6249     | 51.60           |
        | ONNX            | ✅   | 95.0         | 0.6247     | 61.95           |
        | OpenVINO        | ✅   | 95.3         | 0.6238     | 272.47          |
        | TensorRT (FP32) | ✅   | 97.1         | 0.6250     | 31.64           |
        | TensorRT (FP16) | ✅   | 51.4         | 0.6225     | 14.77           |
        | TensorRT (INT8) | ✅   | 30.0         | 0.5923     | 10.49           |
        | TF SavedModel   | ✅   | 238.4        | 0.6245     | 596.46          |
        | TF GraphDef     | ✅   | 95.0         | 0.6245     | 606.10          |
        | TF Lite         | ✅   | 95.4         | 0.6245     | 3275.55         |
        | MNN             | ✅   | 94.8         | 0.6247     | 408.15          |
        | NCNN            | ✅   | 94.8         | 0.6323     | 262.99          |

    === "YOLO26x"

        | 格式            | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |----------------|------|--------------|------------|-----------------|
        | PyTorch         | ✅   | 113.2        | 0.6561     | 84.40           |
        | TorchScript     | ✅   | 213.5        | 0.6594     | 91.20           |
        | ONNX            | ✅   | 212.9        | 0.6595     | 109.34          |
        | OpenVINO        | ✅   | 213.2        | 0.6592     | 520.88          |
        | TensorRT (FP32) | ✅   | 215.1        | 0.6593     | 57.18           |
        | TensorRT (FP16) | ✅   | 109.7        | 0.6632     | 26.76           |
        | TensorRT (INT8) | ✅   | 60.0         | 0.6170     | 17.32           |
        | TF SavedModel   | ✅   | 533.3        | 0.6593     | 1170.50         |
        | TF GraphDef     | ✅   | 212.9        | 0.6593     | 1217.87         |
        | TF Lite         | ✅   | 213.3        | 0.6593     | 7247.11         |
        | MNN             | ✅   | 212.8        | 0.6591     | 820.90          |
        | NCNN            | ✅   | 212.8        | 0.6666     | 534.30          |

    使用 Ultralytics 8.4.33 进行基准测试

    !!! note

        推理时间不包括前/后处理。

[探索 Seeed Studio 在不同版本 NVIDIA Jetson 硬件上的更多基准测试](https://www.seeedstudio.com/blog/2023/03/30/yolov8-performance-benchmarks-on-nvidia-jetson-devices/)。

## 复现我们的结果

要复现上述所有导出[格式](../modes/export.md)的 Ultralytics 基准测试，运行以下代码：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11n PyTorch 模型
        model = YOLO("yolo11n.pt")

        # 在 COCO128 数据集上对所有导出格式进行 YOLO11n 速度和准确率基准测试
        results = model.benchmark(data="coco128.yaml", imgsz=640)
        ```

    === "CLI"

        ```bash
        # 在 COCO128 数据集上对所有导出格式进行 YOLO11n 速度和准确率基准测试
        yolo benchmark model=yolo11n.pt data=coco128.yaml imgsz=640
        ```

    请注意，基准测试结果可能因系统的确切硬件和软件配置以及运行基准测试时系统的当前工作负载而异。为了获得最可靠的结果，请使用包含大量图像的数据集，例如 `data='coco.yaml'`（5000 张验证图像）。

## 使用 NVIDIA Jetson 的最佳实践

在使用 NVIDIA Jetson 时，遵循以下最佳实践可以确保在运行 YOLO26 时获得最佳性能。

1. 启用 MAX Power 模式

    在 Jetson 上启用 MAX Power 模式将确保所有 CPU、GPU 核心都开启。

    ```bash
    sudo nvpmodel -m 0
    ```

2. 启用 Jetson Clocks

    启用 Jetson Clocks 将确保所有 CPU、GPU 核心以最高频率运行。

    ```bash
    sudo jetson_clocks
    ```

3. 安装 Jetson Stats 应用

    我们可以使用 jetson stats 应用来监控系统组件的温度，并查看其他系统详细信息，如 CPU、GPU、RAM 利用率、更改电源模式、设置最高时钟频率、查看 JetPack 信息等。

    ```bash
    sudo apt update
    sudo pip install jetson-stats
    sudo reboot
    jtop
    ```

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/jetson-stats-application.avif" alt="Jetson Stats">

## NVIDIA Jetson 内存优化技巧

可用内存通常是 Jetson 设备的限制因素，特别是在低内存版本上，如 Jetson Orin Nano (8 GB) 或 Orin NX 8 GB。以下提示是一些实用的低风险更改，可以共同释放数百兆字节内存，让您运行更大的 YOLO 模型或支持额外的并行工作负载。全面了解可参见 [NVIDIA 关于最大化 Jetson 内存效率的博客](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/)。

### 1. 切换到无头（无 GUI）启动

如果您的 Jetson 通过 SSH 连接或作为无显示器的生产设备运行，消除桌面环境和显示服务器可以回收多达 **865 MB** 的 RAM：

```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

稍后恢复桌面：

```bash
sudo systemctl set-default graphical.target
sudo reboot
```

### 2. 禁用未使用的系统服务

非必需的后台服务（蓝牙、连接管理器、未使用的硬件守护进程）合计消耗约 **32 MB**。列出活动服务并禁用部署不需要的任何服务：

```bash
# 列出运行中的服务
systemctl list-units --type=service --state=running

# 禁用某个服务
sudo systemctl disable <服务名>
```

### 3. 分析内存使用情况

在优化之前，先确定哪些进程实际消耗 RAM。`procrank` 按 PSS（按比例分配的内存集大小）对进程排序，这比 RSS（常驻内存集大小，即进程映射的物理 RAM 页面总数，包括与其他进程共享的页面）更准确地反映了真实的每进程内存占用：

```bash
git clone https://github.com/csimmonds/procrank_linux.git
cd procrank_linux && make
sudo ./procrank
```

要查看每进程的 GPU 和 NvMap（CUDA/视频管线）分配：

```bash
sudo cat /sys/kernel/debug/nvmap/iovmm/clients
```

### 4. 在生产环境中无显示运行推理

对于没有实时预览需求的推理管线，禁用显示相关组件（Tiler、OSD、DisplaySink）可以仅从管线中节省 **200+ MB**。使用 Ultralytics YOLO，抑制查看器并将结果写入磁盘：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.engine")

        # show=False 阻止任何显示窗口；save=True 将标注输出写入磁盘
        results = model.predict(source="video.mp4", show=False, save=True)
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo11n.engine source=video.mp4 show=False save=True
        ```

### 累积影响

| 优化措施                     | 大约节省的内存 |
| ---------------------------- | -------------- |
| 禁用桌面 GUI                 | ~865 MB        |
| 禁用未使用的操作系统服务     | ~32 MB         |
| 无头推理管线（无显示）       | ~200+ MB       |
| **总计（轻松实现的部分）**   | **~1 GB+**     |

在内存受限的设备上以 TensorRT INT8 模型为目标时，组合这些更改尤其有价值——这可能决定了能否将更大的模型变体装入内存。

## 后续步骤

如需进一步学习和支持，请参阅 [Ultralytics YOLO26 文档](../index.md)。

## 常见问题 (FAQ)

### 如何在 NVIDIA Jetson 设备上部署 Ultralytics YOLO26？

在 NVIDIA Jetson 设备上部署 Ultralytics YOLO26 是一个直接的过程。首先，使用 NVIDIA JetPack SDK 烧录 Jetson 设备。然后，使用预构建的 Docker 镜像快速设置，或手动安装所需的包。每种方法的详细步骤可在 [使用 Docker 快速入门](#使用-docker-快速入门) 和 [通过原生安装开始](#通过原生安装开始) 部分找到。

### NVIDIA Jetson 设备上 YOLO11 模型的性能基准如何？

YOLO11 模型已在各种 NVIDIA Jetson 设备上进行了基准测试，展示了显著的性能提升。例如，TensorRT 格式提供最佳的推理性能。[详细对比表](#详细对比表) 部分中的表格提供了不同模型格式的 mAP50-95 和推理时间等性能指标的综合视图。

### 为什么在 NVIDIA Jetson 上部署 YOLO26 应使用 TensorRT？

强烈建议在 NVIDIA Jetson 上使用 TensorRT 部署 YOLO26 模型，因为它具有最佳性能。它通过利用 Jetson 的 GPU 能力加速推理，确保最高效率和速度。在 [在 NVIDIA Jetson 上使用 TensorRT](#在-nvidia-jetson-上使用-tensorrt) 部分了解更多关于如何转换为 TensorRT 并运行推理的信息。

### 如何在 NVIDIA Jetson 上安装 PyTorch 和 Torchvision？

要在 NVIDIA Jetson 上安装 PyTorch 和 Torchvision，首先卸载可能已通过 pip 安装的现有版本。然后，手动安装适用于 Jetson ARM64 架构的兼容 PyTorch 和 Torchvision 版本。此过程的详细说明见 [安装 PyTorch 和 Torchvision](#安装-pytorch-和-torchvision) 部分。

### 在 NVIDIA Jetson 上使用 YOLO26 时最大化性能的最佳实践是什么？

要在 NVIDIA Jetson 上使用 YOLO26 最大化性能，请遵循以下最佳实践：

1. 启用 MAX Power 模式以利用所有 CPU 和 GPU 核心。
2. 启用 Jetson Clocks 以最高频率运行所有核心。
3. 安装 Jetson Stats 应用以监控系统指标。

有关命令和更多详细信息，请参阅 [使用 NVIDIA Jetson 的最佳实践](#使用-nvidia-jetson-的最佳实践) 部分。

### 如何释放 NVIDIA Jetson 上的内存以运行更大的 YOLO 模型？

可用 RAM 通常是低内存 Jetson 设备的瓶颈。三个轻松实现的措施合计可以回收超过 1 GB：

1. **切换到无头启动**（`sudo systemctl set-default multi-user.target`）以消除桌面 GUI（约节省 865 MB）。
2. **禁用未使用的服务**，如蓝牙或连接管理器（约节省 32 MB）。
3. **无显示运行推理**，在 YOLO `predict` 调用中设置 `show=False`，避免分配显示管线内存（约节省 200+ MB）。

使用 `procrank` 分析每进程 RAM 使用情况，使用 `sudo cat /sys/kernel/debug/nvmap/iovmm/clients` 检查 GPU 分配。完整详情见 [NVIDIA Jetson 内存优化技巧](#nvidia-jetson-内存优化技巧) 部分。

### 为什么在 JetPack 6 上我的 TensorRT INT8 导出会禁用 end2end？

JetPack 6 附带的 TensorRT 10.3.0 存在一个已知问题，当启用 `end2end=True` 时会阻止 INT8 引擎构建。当 Ultralytics 检测到此组合时，会自动禁用 end2end 分支以确保导出成功。

要恢复 end2end INT8 导出，请将 TensorRT 升级到较新版本（例如 10.7.0+）：

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y tensorrt
```

升级后，重新运行导出。更多详情请参见 [GitHub issue #23841](https://github.com/ultralytics/ultralytics/issues/23841)。
