---
comments: true
description: 通过本综合指南学习如何在树莓派上部署 Ultralytics YOLO26。包含性能基准测试、安装说明和最佳实践。
keywords: Ultralytics, YOLO26, 树莓派, 安装, 指南, 基准测试, 计算机视觉, 目标检测, NCNN, Docker, 摄像头模块
---

# 快速入门指南：在树莓派上使用 Ultralytics YOLO26

本综合指南详细介绍了如何在[树莓派](https://www.raspberrypi.com/)设备上部署 Ultralytics YOLO26。此外，还展示了性能基准测试，以体现 YOLO26 在这些小巧而强大的设备上的能力。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yul4gq_LrOI"
    title="Introducing Raspberry Pi 5" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>树莓派 5 的更新与改进。
</p>

!!! note

    本指南已在运行最新 [Raspberry Pi OS Bookworm (Debian 12)](https://www.raspberrypi.com/software/operating-systems/) 的树莓派 4 和树莓派 5 上测试通过。对于较旧的树莓派设备（如树莓派 3），只要安装了相同的 Raspberry Pi OS Bookworm，也应可以正常工作。

## 什么是树莓派？

树莓派是一款小巧、经济实惠的单板计算机。它在从爱好者的家庭自动化到工业用途的各类项目和应用中广受欢迎。树莓派主板能够运行多种操作系统，并提供 GPIO（通用输入/输出）引脚，便于与传感器、执行器及其他硬件组件集成。树莓派有多种规格不同的型号，但都秉承低成本、紧凑、多功能的基本设计理念。

## 树莓派系列对比

|                   | 树莓派 3                               | 树莓派 4                               | 树莓派 5                               |
| ----------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| CPU               | Broadcom BCM2837, Cortex-A53 64Bit SoC | Broadcom BCM2711, Cortex-A72 64Bit SoC | Broadcom BCM2712, Cortex-A76 64Bit SoC |
| CPU 最高频率      | 1.4GHz                                 | 1.8GHz                                 | 2.4GHz                                 |
| GPU               | Videocore IV                           | Videocore VI                           | VideoCore VII                          |
| GPU 最高频率      | 400MHz                                 | 500MHz                                 | 800MHz                                 |
| 内存              | 1GB LPDDR2 SDRAM                       | 1GB、2GB、4GB、8GB LPDDR4-3200 SDRAM    | 4GB、8GB LPDDR4X-4267 SDRAM            |
| PCIe              | 无                                     | 无                                     | 1x PCIe 2.0 接口                       |
| 最大功耗          | 2.5A@5V                                | 3A@5V                                  | 5A@5V（支持 PD）                       |

## 什么是 Raspberry Pi OS？

[Raspberry Pi OS](https://www.raspberrypi.com/software/)（前身为 Raspbian）是一款基于 Debian GNU/Linux 发行版的类 Unix 操作系统，由树莓派基金会为树莓派系列紧凑型单板计算机分发。Raspberry Pi OS 针对搭载 ARM CPU 的树莓派进行了高度优化，使用经过修改的 LXDE 桌面环境和 Openbox 堆叠窗口管理器。Raspberry Pi OS 正在积极开发中，重点在于尽可能提升更多 Debian 软件包在树莓派上的稳定性和性能。

## 将 Raspberry Pi OS 烧录到树莓派

拿到树莓派后，首先要做的就是将 Raspberry Pi OS 烧录到 micro-SD 卡中，插入设备并启动操作系统。请按照[树莓派官方入门文档](https://www.raspberrypi.com/documentation/computers/getting-started.html)的详细说明，为首次使用做好准备。

## 安装 Ultralytics

在树莓派上安装 Ultralytics 包以构建您的下一个[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)项目有两种方式，您可以任选其一。

- [使用 Docker 开始](#使用-docker-开始)
- [不使用 Docker 开始](#不使用-docker-开始)

### 使用 Docker 开始

在树莓派上使用 Ultralytics YOLO26 的最快方式是运行预构建的树莓派 Docker 镜像。

执行以下命令拉取 Docker 容器并在树莓派上运行。该镜像基于 [arm64v8/debian](https://hub.docker.com/r/arm64v8/debian) Docker 镜像，包含 Debian 12 (Bookworm) 和 Python3 环境。

```bash
t=ultralytics/ultralytics:latest-arm64
sudo docker pull $t && sudo docker run -it --ipc=host $t
```

完成之后，请跳转到[在树莓派上使用 NCNN 章节](#在树莓派上使用-ncnn)。

### 不使用 Docker 开始

#### 安装 Ultralytics 包

我们将在此步骤中安装 Ultralytics 包及其可选依赖项，以便将 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 模型导出为其他不同格式。

1. 更新软件包列表，安装 pip 并升级到最新版

    ```bash
    sudo apt update
    sudo apt install python3-pip -y
    pip install -U pip
    ```

2. 安装 `ultralytics` pip 包及可选依赖项

    ```bash
    pip install ultralytics[export]
    ```

3. 重启设备

    ```bash
    sudo reboot
    ```

## 在树莓派上使用 NCNN

在 Ultralytics 支持的所有模型导出格式中，[NCNN](https://docs.ultralytics.com/integrations/ncnn) 在树莓派设备上提供最佳的推理性能，因为 NCNN 针对移动端/嵌入式平台（如 ARM 架构）进行了高度优化。

## 将模型转换为 NCNN 并运行推理

将 PyTorch 格式的 YOLO26n 模型转换为 NCNN 格式，以使用导出的模型运行推理。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO26n PyTorch 模型
        model = YOLO("yolo26n.pt")

        # 将模型导出为 NCNN 格式
        model.export(format="ncnn")  # 创建 'yolo26n_ncnn_model'

        # 加载导出的 NCNN 模型
        ncnn_model = YOLO("yolo26n_ncnn_model")

        # 运行推理
        results = ncnn_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO26n PyTorch 模型导出为 NCNN 格式
        yolo export model=yolo26n.pt format=ncnn # 创建 'yolo26n_ncnn_model'

        # 使用导出的模型运行推理
        yolo predict model='yolo26n_ncnn_model' source='https://ultralytics.com/images/bus.jpg'
        ```

!!! tip

    有关支持的导出选项的更多详细信息，请访问 [Ultralytics 部署选项文档页面](https://docs.ultralytics.com/guides/model-deployment-options)。

## YOLO26 相比 YOLO11 的性能提升

YOLO26 专为在树莓派 5 等硬件受限设备上运行而设计。与 YOLO11n 相比，YOLO26n 在树莓派 5 上以 ONNX 导出模型在 640 输入尺寸下，FPS 提高了约 15%（6.79 → 7.79），同时 mAP 也更高（40.1 vs 39.5）。下表和图展示了这一对比。

<figure style="text-align: center;">
    <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yolo26-vs-yolo11-rpi5-onnx-benchmarks.avif" alt="YOLO26 在树莓派 5 上的基准测试">
    <figcaption style="font-style: italic; color: gray;">使用 Ultralytics 8.4.14 进行基准测试</figcaption>
</figure>

!!! tip "性能"

    === "YOLO26 (ONNX)"

        | 模型      | mAP50-95(B) | 推理时间 (ms/张) |
        |---------  |------------ |----------------- |
        | YOLO26n   | 40.1        | 128.42           |
        | YOLO26s   | 47.8        | 352.84           |
        | YOLO26m   | 52.5        | 993.78           |
        | YOLO26l   | 54.4        | 1259.46          |
        | YOLO26x   | 56.9        | 2636.26          |


    === "YOLO11 (ONNX)"

        | 模型      | mAP50-95(B) | 推理时间 (ms/张) |
        |---------  |------------ |----------------- |
        | YOLO11n   | 39.5        | 147.20           |
        | YOLO11s   | 47.0        | 366.83           |
        | YOLO11m   | 51.5        | 997.46           |
        | YOLO11l   | 53.4        | 1274.95          |
        | YOLO11x   | 54.7        | 2646.76          |

    使用 Ultralytics 8.4.14 进行基准测试。

## 树莓派 5 YOLO26 基准测试

YOLO26 基准测试由 Ultralytics 团队在十种不同模型格式上运行，测量速度和[准确率](https://www.ultralytics.com/glossary/accuracy)：PyTorch、TorchScript、ONNX、OpenVINO、TF SavedModel、TF GraphDef、TF Lite、MNN、NCNN、ExecuTorch。基准测试在树莓派 5 上以 FP32 [精度](https://www.ultralytics.com/glossary/precision)运行，默认输入图像尺寸为 640。

### 对比图

我们仅包含 YOLO26n 和 YOLO26s 模型的基准测试结果，因为其他模型尺寸过大，无法在树莓派上运行且性能不佳。

<figure style="text-align: center;">
    <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/raspberry-pi-yolo26-benchmarks.avif" alt="YOLO26 在树莓派 5 上的基准测试">
    <figcaption style="font-style: italic; color: gray;">使用 Ultralytics 8.4.1 进行基准测试</figcaption>
</figure>

### 详细对比表

下表展示了两种不同模型（YOLO26n、YOLO26s）在十种不同格式（PyTorch、TorchScript、ONNX、OpenVINO、TF SavedModel、TF GraphDef、TF Lite、MNN、NCNN、ExecuTorch）下，在树莓派 5 上运行的基准测试结果，包含每种组合的状态、磁盘大小、mAP50-95(B) 指标和推理时间。

!!! tip "性能"

    === "YOLO26n"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/张) |
        |---------------|------|--------------|-------------|-----------------|
        | PyTorch       | ✅    | 5.3          | 0.4798      | 302.15          |
        | TorchScript   | ✅    | 9.8          | 0.4764      | 357.58          |
        | ONNX          | ✅    | 9.5          | 0.4764      | 130.33          |
        | OpenVINO      | ✅    | 9.6          | 0.4818      | 70.74           |
        | TF SavedModel | ✅    | 24.6         | 0.4764      | 213.58          |
        | TF GraphDef   | ✅    | 9.5          | 0.4764      | 213.5           |
        | TF Lite       | ✅    | 9.9          | 0.4764      | 251.41          |
        | MNN           | ✅    | 9.4          | 0.4784      | 90.89           |
        | NCNN          | ✅    | 9.4          | 0.4805      | 67.69           |
        | ExecuTorch    | ✅    | 9.4          | 0.4764      | 148.36          |

    === "YOLO26s"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/张) |
        |---------------|------|--------------|-------------|-----------------|
        | PyTorch       | ✅    | 19.5         | 0.5740      | 836.54          |
        | TorchScript   | ✅    | 36.8         | 0.5665      | 1032.25         |
        | ONNX          | ✅    | 36.5         | 0.5665      | 351.96          |
        | OpenVINO      | ✅    | 36.7         | 0.5654      | 158.6           |
        | TF SavedModel | ✅    | 92.2         | 0.5665      | 507.6           |
        | TF GraphDef   | ✅    | 36.5         | 0.5665      | 525.64          |
        | TF Lite       | ✅    | 36.9         | 0.5665      | 805.3           |
        | MNN           | ✅    | 36.4         | 0.5644      | 236.47          |
        | NCNN          | ✅    | 36.4         | 0.5697      | 168.47          |
        | ExecuTorch    | ✅    | 36.5         | 0.5665      | 388.72          |

    使用 Ultralytics 8.4.1 进行基准测试。

    !!! note

        推理时间不包含前/后处理。

### 复现我们的结果

要复现上述所有[导出格式](../modes/export.md)的 Ultralytics 基准测试结果，请运行以下代码：

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

    请注意，基准测试结果可能因系统的具体硬件和软件配置以及运行基准测试时系统的当前负载而异。要获得最可靠的结果，请使用包含大量图像的数据集，例如 `data='coco.yaml'`（5000 张验证图像）。

## 使用树莓派摄像头

使用树莓派进行计算机视觉项目时，获取实时视频流进行推理可能至关重要。树莓派上的板载 MIPI CSI 接口允许您连接官方树莓派摄像头模块。在本指南中，我们使用[树莓派摄像头模块 3](https://www.raspberrypi.com/products/camera-module-3/) 来获取视频流并使用 YOLO26 模型进行推理。

!!! tip

    详细了解[树莓派提供的不同摄像头模块](https://www.raspberrypi.com/documentation/accessories/camera.html)以及[如何开始使用树莓派摄像头模块](https://www.raspberrypi.com/documentation/computers/camera_software.html#introducing-the-raspberry-pi-cameras)。

!!! note

    树莓派 5 使用比树莓派 4 更小的 CSI 接口（15 针 vs 22 针），因此您需要一根 [15 针转 22 针适配线](https://www.raspberrypi.com/products/camera-cable/)来连接树莓派摄像头。

### 测试摄像头

将摄像头连接到树莓派后，执行以下命令。您应该能看到约 5 秒钟的摄像头实时视频画面。

```bash
rpicam-hello
```

!!! tip

    详细了解 [rpicam-hello 在官方树莓派文档中的用法](https://www.raspberrypi.com/documentation/computers/camera_software.html#rpicam-hello)

### 使用摄像头进行推理

使用树莓派摄像头在 YOLO26 模型上运行推理有两种方法。

!!! usage

    === "方法 1"

        我们可以使用 Raspberry Pi OS 预装的 `picamera2` 来访问摄像头并在 YOLO26 模型上运行推理。

        !!! example

            === "Python"

                ```python
                import cv2
                from picamera2 import Picamera2

                from ultralytics import YOLO

                # 初始化 Picamera2
                picam2 = Picamera2()
                picam2.preview_configuration.main.size = (1280, 720)
                picam2.preview_configuration.main.format = "RGB888"
                picam2.preview_configuration.align()
                picam2.configure("preview")
                picam2.start()

                # 加载 YOLO26 模型
                model = YOLO("yolo26n.pt")

                while True:
                    # 逐帧捕获
                    frame = picam2.capture_array()

                    # 在帧上运行 YOLO26 推理
                    results = model(frame)

                    # 在帧上可视化结果
                    annotated_frame = results[0].plot()

                    # 显示结果帧
                    cv2.imshow("Camera", annotated_frame)

                    # 按下 'q' 键退出循环
                    if cv2.waitKey(1) == ord("q"):
                        break

                # 释放资源并关闭窗口
                cv2.destroyAllWindows()
                ```

    === "方法 2"

        我们需要使用 `rpicam-vid` 从已连接的摄像头启动 TCP 流，以便在之后推理时将该流 URL 作为输入使用。执行以下命令来启动 TCP 流。

        ```bash
        rpicam-vid -n -t 0 --inline --listen -o tcp://127.0.0.1:8888
        ```

        详细了解 [rpicam-vid 在官方树莓派文档中的用法](https://www.raspberrypi.com/documentation/computers/camera_software.html#rpicam-vid)

        !!! example

            === "Python"

                ```python
                from ultralytics import YOLO

                # 加载 YOLO26n PyTorch 模型
                model = YOLO("yolo26n.pt")

                # 运行推理
                results = model("tcp://127.0.0.1:8888")
                ```

            === "CLI"

                ```bash
                yolo predict model=yolo26n.pt source="tcp://127.0.0.1:8888"
                ```

!!! tip

    如果您想更改图像/视频输入类型，请查看我们的[推理来源文档](https://docs.ultralytics.com/modes/predict#inference-sources)

## 在树莓派上使用的最佳实践

以下是一些最佳实践，遵循这些实践可以在运行 YOLO26 的树莓派上获得最佳性能。

1. 使用 SSD

    当树莓派需要 7x24 小时持续运行时，建议使用 SSD 作为系统盘，因为 SD 卡无法承受持续写入，可能会损坏。借助树莓派 5 上的板载 PCIe 接口，您现在可以使用适配器（如 [树莓派 5 NVMe 底座](https://shop.pimoroni.com/products/nvme-base)）连接 SSD。

2. 安装无 GUI 版本的系统

    在烧录 Raspberry Pi OS 时，您可以选择不安装桌面环境（Raspberry Pi OS Lite），这可以为设备节省一些内存，为计算机视觉处理留出更多空间。

3. 超频树莓派

    如果您希望在使用树莓派 5 运行 Ultralytics YOLO26 模型时获得一些性能提升，可以将 CPU 从基础频率 2.4GHz 超频至 2.9GHz，将 GPU 从 800MHz 超频至 1GHz。如果系统变得不稳定或崩溃，请将超频值每次降低 100MHz。请确保有适当的散热措施，因为超频会增加发热量并可能导致热降频。

    a. 升级软件

    ```bash
    sudo apt update && sudo apt dist-upgrade
    ```

    b. 打开配置文件进行编辑

    ```bash
    sudo nano /boot/firmware/config.txt
    ```

    c. 在底部添加以下行

    ```bash
    arm_freq=3000
    gpu_freq=1000
    force_turbo=1
    ```

    d. 按 CTRL + X，然后按 Y，再按 ENTER 保存并退出

    e. 重启树莓派

## 下一步

您已成功在树莓派上设置好 YOLO。如需进一步学习和支持，请访问 [Ultralytics YOLO26 文档](../index.md) 和 [Kashmir World Foundation](https://www.kashmirworldfoundation.org/)。

## 致谢与引用

本指南最初由 Daan Eeltink 为 Kashmir World Foundation 创建，该组织致力于利用 YOLO 保护濒危物种。我们感谢他们在目标检测技术领域的开创性工作和教育重点。

有关 Kashmir World Foundation 活动的更多信息，请访问他们的[网站](https://www.kashmirworldfoundation.org/)。

## 常见问题

### 如何在不使用 Docker 的情况下在树莓派上安装 Ultralytics YOLO26？

要在不使用 Docker 的情况下在树莓派上安装 Ultralytics YOLO26，请按照以下步骤操作：

1. 更新软件包列表并安装 `pip`：
    ```bash
    sudo apt update
    sudo apt install python3-pip -y
    pip install -U pip
    ```
2. 安装 Ultralytics 包及可选依赖项：
    ```bash
    pip install ultralytics[export]
    ```
3. 重启设备以应用更改：
    ```bash
    sudo reboot
    ```

详细说明请参考[不使用 Docker 开始](#不使用-docker-开始)章节。

### 为什么应该在树莓派上使用 Ultralytics YOLO26 的 NCNN 格式进行 AI 任务？

Ultralytics YOLO26 的 NCNN 格式针对移动端和嵌入式平台进行了高度优化，非常适合在树莓派设备上运行 AI 任务。NCNN 通过利用 ARM 架构最大化推理性能，相比其他格式提供更快速、更高效的处理。有关支持的导出选项的更多详情，请访问 [Ultralytics 部署选项文档页面](https://docs.ultralytics.com/guides/model-deployment-options)。

### 如何将 YOLO26 模型转换为 NCNN 格式以在树莓派上使用？

您可以使用 Python 或 CLI 命令将 PyTorch YOLO26 模型转换为 NCNN 格式：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO26n PyTorch 模型
        model = YOLO("yolo26n.pt")

        # 将模型导出为 NCNN 格式
        model.export(format="ncnn")  # 创建 'yolo26n_ncnn_model'

        # 加载导出的 NCNN 模型
        ncnn_model = YOLO("yolo26n_ncnn_model")

        # 运行推理
        results = ncnn_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO26n PyTorch 模型导出为 NCNN 格式
        yolo export model=yolo26n.pt format=ncnn # 创建 'yolo26n_ncnn_model'

        # 使用导出的模型运行推理
        yolo predict model='yolo26n_ncnn_model' source='https://ultralytics.com/images/bus.jpg'
        ```

更多详情请参见[在树莓派上使用 NCNN](#在树莓派上使用-ncnn)章节。

### 树莓派 4 和树莓派 5 在运行 YOLO26 方面的硬件差异有哪些？

主要差异包括：

- **CPU**：树莓派 4 使用 Broadcom BCM2711, Cortex-A72 64 位 SoC，而树莓派 5 使用 Broadcom BCM2712, Cortex-A76 64 位 SoC。
- **CPU 最高频率**：树莓派 4 最高频率为 1.8GHz，树莓派 5 则达到 2.4GHz。
- **内存**：树莓派 4 提供最高 8GB LPDDR4-3200 SDRAM，而树莓派 5 配备 LPDDR4X-4267 SDRAM，提供 4GB 和 8GB 两种版本。

这些增强使得 YOLO26 模型在树莓派 5 上的性能基准测试结果优于树莓派 4。请参考[树莓派系列对比](#树莓派系列对比)表格了解更多详情。

### 如何设置树莓派摄像头模块以配合 Ultralytics YOLO26 使用？

设置树莓派摄像头进行 YOLO26 推理有两种方法：

1. **使用 `picamera2`**：

    ```python
    import cv2
    from picamera2 import Picamera2

    from ultralytics import YOLO

    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1280, 720)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    model = YOLO("yolo26n.pt")

    while True:
        frame = picam2.capture_array()
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Camera", annotated_frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
    ```

2. **使用 TCP 流**：

    ```bash
    rpicam-vid -n -t 0 --inline --listen -o tcp://127.0.0.1:8888
    ```

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    results = model("tcp://127.0.0.1:8888")
    ```

详细的设置说明请访问[使用摄像头进行推理](#使用摄像头进行推理)章节。
