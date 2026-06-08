---
comments: true
description: 学习如何轻松地在 Docker 中设置 Ultralytics，从安装到使用 CPU/GPU 支持运行。遵循我们的全面指南，获得无缝的容器体验。
keywords: Ultralytics, Docker, 快速入门指南, CPU 支持, GPU 支持, NVIDIA Docker, NVIDIA 容器工具包, 容器设置, Docker 环境, Docker Hub, Ultralytics 项目
---

# Ultralytics Docker 快速入门指南

<p align="center">
  <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-docker-package-visual.avif" alt="Ultralytics Docker 包可视化">
</p>

本指南全面介绍了如何为您的 Ultralytics 项目设置 Docker 环境。[Docker](https://www.docker.com/) 是一个用于在容器中开发、交付和运行应用程序的平台。它特别有益于确保软件无论在何处部署，都能始终以相同的方式运行。有关更多详细信息，请访问 [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics) 上的 Ultralytics Docker 仓库。

[![Docker Image Version](https://img.shields.io/docker/v/ultralytics/ultralytics?sort=semver&logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics)
[![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/ultralytics)](https://hub.docker.com/r/ultralytics/ultralytics)

## 您将学习什么

- 设置支持 NVIDIA 的 Docker
- 安装 Ultralytics Docker 镜像
- 在支持 CPU 或 GPU 的 Docker 容器中运行 Ultralytics
- 在 Docker 中使用显示服务器来显示 Ultralytics 检测结果
- 将本地目录挂载到容器中

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/IYWQZvtOy_Q"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何开始使用 Docker | 在 Docker 中使用 Ultralytics Python 包的实时演示 🎉
</p>

---

## 先决条件

- 确保您的系统已安装 Docker。如果未安装，可以从 [Docker 官网](https://www.docker.com/products/docker-desktop/) 下载并安装。
- 确保您的系统有 NVIDIA GPU 并已安装 NVIDIA 驱动程序。
- 如果您使用 NVIDIA Jetson 设备，请确保已安装适当的 JetPack 版本。有关更多详细信息，请参阅 [NVIDIA Jetson 指南](https://docs.ultralytics.com/guides/nvidia-jetson)。

---

## 设置支持 NVIDIA 的 Docker

首先，通过运行以下命令验证 NVIDIA 驱动程序是否已正确安装：

```bash
nvidia-smi
```

### 安装 NVIDIA 容器工具包

现在，让我们安装 [NVIDIA 容器工具包](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)，以在 Docker 容器中启用 GPU 支持：

=== "Ubuntu/Debian"

    ```bash
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    ```
    更新软件包列表并安装 nvidia-container-toolkit 软件包：

    ```bash
    sudo apt-get update
    ```

    安装最新版本的 `nvidia-container-toolkit`：

    ```bash
    sudo apt-get install -y nvidia-container-toolkit \
      nvidia-container-toolkit-base libnvidia-container-tools \
      libnvidia-container1
    ```

    ??? info "可选：安装指定版本的 nvidia-container-toolkit"

        您可以选择通过设置 `NVIDIA_CONTAINER_TOOLKIT_VERSION` 环境变量来安装特定版本的 nvidia-container-toolkit：

        ```bash
        export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
        sudo apt-get install -y \
          nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
          nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
          libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
          libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
        ```

    ```bash
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

=== "RHEL/CentOS/Fedora/Amazon Linux"

    ```bash
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
      | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
    ```

    更新软件包列表并安装 nvidia-container-toolkit 软件包：

    ```bash
    sudo dnf clean expire-cache
    sudo dnf check-update
    ```

    ```bash
    sudo dnf install \
      nvidia-container-toolkit \
      nvidia-container-toolkit-base \
      libnvidia-container-tools \
      libnvidia-container1
    ```


    ??? info "可选：安装指定版本的 nvidia-container-toolkit"

        您可以选择通过设置 `NVIDIA_CONTAINER_TOOLKIT_VERSION` 环境变量来安装特定版本的 nvidia-container-toolkit：

          ```bash
          export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
          sudo dnf install -y \
            nvidia-container-toolkit-${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
            nvidia-container-toolkit-base-${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
            libnvidia-container-tools-${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
            libnvidia-container1-${NVIDIA_CONTAINER_TOOLKIT_VERSION}
          ```

    ```bash
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

### 验证 Docker 的 NVIDIA 运行时

运行 `docker info | grep -i runtime` 以确保 `nvidia` 出现在运行时列表中：

```bash
docker info | grep -i runtime
```

---

## 安装 Ultralytics Docker 镜像

Ultralytics 提供了多个针对不同平台和用例优化的 Docker 镜像：

- **Dockerfile:** GPU 镜像，适合训练。
- **Dockerfile-arm64:** 适用于 ARM64 架构的设备，如 [Raspberry Pi](raspberry-pi.md)。
- **Dockerfile-cpu:** 仅 CPU 版本，适用于推理和非 GPU 环境。
- **Dockerfile-jetson-jetpack4:** 针对运行 [NVIDIA JetPack 4](https://developer.nvidia.com/embedded/jetpack-sdk-461) 的 [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson) 设备优化。
- **Dockerfile-jetson-jetpack5:** 针对运行 [NVIDIA JetPack 5](https://developer.nvidia.com/embedded/jetpack-sdk-512) 的 [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson) 设备优化。
- **Dockerfile-jetson-jetpack6:** 针对运行 [NVIDIA JetPack 6](https://developer.nvidia.com/embedded/jetpack-sdk-61) 的 [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson) 设备优化。
- **Dockerfile-jupyter:** 用于在浏览器中使用 JupyterLab 进行交互式开发。
- **Dockerfile-nvidia-arm64:** 适用于 NVIDIA ARM64 设备，如 Jetson AGX Thor 和 DGX Spark，支持 JetPack 7.0 和 DGX OS。
- **Dockerfile-python:** 适用于轻量级应用程序的最小 Python 环境。
- **Dockerfile-python-export:** 扩展了完整导出功能的最小 Python 镜像，用于 YOLO 模型转换。
- **Dockerfile-conda:** 包含 [Miniconda3](https://www.anaconda.com/docs/main) 和通过 Conda 安装的 Ultralytics 包。
- **Dockerfile-export:** 预装了所有导出格式依赖项的 GPU 镜像，用于模型转换和基准测试。

拉取最新镜像：

```bash
# 将镜像名称设置为变量
t=ultralytics/ultralytics:latest

# 从 Docker Hub 拉取最新的 Ultralytics 镜像
sudo docker pull $t
```

---

## 在 Docker 容器中运行 Ultralytics

以下是执行 Ultralytics Docker 容器的方法：

### 仅使用 CPU

```bash
# 不使用 GPU 运行
sudo docker run -it --ipc=host $t
```

### 使用 GPU

```bash
# 使用所有 GPU 运行
sudo docker run -it --ipc=host --runtime=nvidia --gpus all $t

# 运行并指定要使用的 GPU
sudo docker run -it --ipc=host --runtime=nvidia --gpus '"device=2,3"' $t
```

`-it` 标志分配一个伪 TTY 并保持 stdin 打开，允许您与容器交互。`--ipc=host` 标志启用主机 IPC 命名空间的共享，这对于进程间共享内存至关重要。`--gpus` 标志允许容器访问主机的 GPU。

### 关于文件可访问性的说明

要在容器内使用本地计算机上的文件，可以使用 Docker 卷：

```bash
# 将本地目录挂载到容器中
sudo docker run -it --ipc=host --runtime=nvidia --gpus all -v /path/on/host:/path/in/container $t
```

将 `/path/on/host` 替换为您本地计算机上的目录路径，将 `/path/in/container` 替换为 Docker 容器内的目标路径。

### 持久化训练输出

训练输出默认保存到容器内的 `/ultralytics/runs/<task>/<name>/`。如果不挂载主机目录，当容器被移除时，输出将丢失。

要持久化训练输出：

```bash
# 推荐：挂载工作空间并指定项目路径
sudo docker run --rm -it -v "$(pwd)":/w -w /w ultralytics/ultralytics:latest \
  yolo train model=yolo26n.pt data=coco8.yaml project=/w/runs
```

这将所有训练输出保存到您主机上的 `./runs` 目录。

## 在 Docker 容器中运行图形用户界面 (GUI) 应用程序

!!! danger "高度实验性 - 用户承担所有风险"

    以下说明是实验性的。与 Docker 容器共享 X11 套接字存在潜在的安全风险。因此，建议仅在受控环境中测试此解决方案。有关更多信息，请参阅这些关于如何使用 `xhost` 的资源<sup>[(1)](http://users.stat.umn.edu/~geyer/secure.html)[(2)](https://linux.die.net/man/1/xhost)</sup>。

Docker 主要用于容器化后台应用程序和 CLI 程序，但也可以运行图形程序。在 Linux 世界中，两个主要的图形服务器处理图形显示：[X11](https://www.x.org/wiki/)（也称为 X Window System）和 [Wayland](<https://en.wikipedia.org/wiki/Wayland_(protocol)>)。在开始之前，必须确定您当前正在使用哪个图形服务器。运行此命令来查找：

```bash
env | grep -E -i 'x11|xorg|wayland'
```

X11 或 Wayland 显示服务器的设置和配置超出了本指南的范围。如果上述命令没有返回任何内容，那么您需要先为您的系统配置其中之一，然后才能继续。

### 使用 GUI 运行 Docker 容器

!!! example

    ??? info "使用 GPU"
            如果您使用 [GPU](#using-gpus)，可以将 `--gpus all` 标志添加到命令中。

    ??? info "Docker 运行时标志"
            如果您的 Docker 安装默认不使用 `nvidia` 运行时，可以将 `--runtime=nvidia` 标志添加到命令中。

    === "X11"

        如果您使用 X11，可以运行以下命令以允许 Docker 容器访问 X11 套接字：

        ```bash
        xhost +local:docker && docker run -e DISPLAY=$DISPLAY \
          -v /tmp/.X11-unix:/tmp/.X11-unix \
          -v ~/.Xauthority:/root/.Xauthority \
          -it --ipc=host $t
        ```

        此命令将 `DISPLAY` 环境变量设置为主机的显示，挂载 X11 套接字，并将 `.Xauthority` 文件映射到容器。`xhost +local:docker` 命令允许 Docker 容器访问 X11 服务器。


    === "Wayland"

        对于 Wayland，使用以下命令：

        ```bash
        xhost +local:docker && docker run -e DISPLAY=$DISPLAY \
          -v $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:/tmp/$WAYLAND_DISPLAY \
          --net=host -it --ipc=host $t
        ```

        此命令将 `DISPLAY` 环境变量设置为主机的显示，挂载 Wayland 套接字，并允许 Docker 容器访问 Wayland 服务器。

### 使用带 GUI 的 Docker

现在您可以在 Docker 容器中显示图形应用程序。例如，您可以运行以下 [CLI 命令](../usage/cli.md) 来可视化 [YOLO26 模型](../models/yolo26.md) 的 [预测结果](../modes/predict.md)：

```bash
yolo predict model=yolo26n.pt show=True
```

??? info "测试"

    验证 Docker 组是否有权访问 X11 服务器的一个简单方法是运行带有 GUI 程序的容器，如 [`xclock`](https://www.x.org/archive/X11R6.8.1/doc/xclock.1.html) 或 [`xeyes`](https://www.x.org/releases/X11R7.5/doc/man/man1/xeyes.1.html)。或者，您也可以在 Ultralytics Docker 容器中安装这些程序来测试对 GNU-Linux 显示服务器的 X11 服务器的访问权限。如果遇到任何问题，请考虑设置环境变量 `-e QT_DEBUG_PLUGINS=1`。设置此环境变量可以启用调试信息的输出，有助于故障排除过程。

### 完成 Docker GUI 使用后

!!! warning "撤销访问权限"

    在这两种情况下，完成后不要忘记撤销 Docker 组的访问权限。

    ```bash
    xhost -local:docker
    ```

??? question "想直接在终端中查看图像结果吗？"

    请参阅以下关于 [使用终端查看图像结果](./view-results-in-terminal.md) 的指南

---

您现在已设置好使用 Docker 的 Ultralytics，并准备好利用其功能。有关其他安装方法，请参阅 [Ultralytics 快速入门文档](../quickstart.md)。

## 常见问题解答

### 如何使用 Docker 设置 Ultralytics？

要使用 Docker 设置 Ultralytics，首先确保您的系统已安装 Docker。如果您有 NVIDIA GPU，请安装 [NVIDIA 容器工具包](#installing-nvidia-container-toolkit) 以启用 GPU 支持。然后，使用以下命令从 Docker Hub 拉取最新的 Ultralytics Docker 镜像：

```bash
sudo docker pull ultralytics/ultralytics:latest
```

有关详细步骤，请参阅我们的 Docker 快速入门指南。

### 使用 Ultralytics Docker 镜像进行机器学习项目有什么好处？

使用 Ultralytics Docker 镜像可确保在不同机器上环境一致，复制相同的软件和依赖项。这对于 [跨团队协作](https://www.ultralytics.com/blog/how-ultralytics-integration-can-enhance-your-workflow)、在各种硬件上运行模型以及保持可重复性特别有用。对于基于 GPU 的训练，Ultralytics 提供了优化的 Docker 镜像，例如用于一般 GPU 使用的 `Dockerfile` 和用于 NVIDIA Jetson 设备的 `Dockerfile-jetson`。有关更多详细信息，请探索 [Ultralytics Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics)。

### 如何在支持 GPU 的 Docker 容器中运行 Ultralytics YOLO？

首先，确保已安装并配置 [NVIDIA 容器工具包](#installing-nvidia-container-toolkit)。然后，使用以下命令运行支持 GPU 的 Ultralytics YOLO：

```bash
sudo docker run -it --ipc=host --runtime=nvidia --gpus all ultralytics/ultralytics:latest # 所有 GPU
```

此命令设置了一个具有 GPU 访问权限的 Docker 容器。有关其他详细信息，请参阅 Docker 快速入门指南。

### 如何在带有显示服务器的 Docker 容器中可视化 YOLO 预测结果？

要在带有 GUI 的 Docker 容器中可视化 YOLO 预测结果，您需要允许 Docker 访问您的显示服务器。对于运行 X11 的系统，命令如下：

```bash
xhost +local:docker && docker run -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/.Xauthority:/root/.Xauthority \
  -it --ipc=host ultralytics/ultralytics:latest
```

对于运行 Wayland 的系统，使用：

```bash
xhost +local:docker && docker run -e DISPLAY=$DISPLAY \
  -v $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:/tmp/$WAYLAND_DISPLAY \
  --net=host -it --ipc=host ultralytics/ultralytics:latest
```

更多信息可以在 [在 Docker 容器中运行图形用户界面 (GUI) 应用程序](#run-graphical-user-interface-gui-applications-in-a-docker-container) 部分找到。

### 我可以将本地目录挂载到 Ultralytics Docker 容器中吗？

是的，您可以使用 `-v` 标志将本地目录挂载到 Ultralytics Docker 容器中：

```bash
sudo docker run -it --ipc=host --runtime=nvidia --gpus all -v /path/on/host:/path/in/container ultralytics/ultralytics:latest
```

将 `/path/on/host` 替换为您本地计算机上的目录，将 `/path/in/container` 替换为容器内的目标路径。此设置允许您在容器内使用本地文件。有关更多信息，请参阅 [关于文件可访问性的说明](#note-on-file-accessibility) 部分。