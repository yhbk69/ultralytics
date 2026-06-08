---
comments: true
description: 学习为 Ultralytics 项目设置 Conda 环境。按照我们的全面指南轻松完成安装和初始化。
keywords: Ultralytics, Conda, 设置, 安装, 环境, 指南, 机器学习, 数据科学
---

# Ultralytics Conda 快速入门指南

<p align="center">
  <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-conda-package-visual.avif" alt="Ultralytics Conda Package Visual">
</p>

本指南全面介绍了如何为您的 Ultralytics 项目设置 Conda 环境。Conda 是一个开源的包和环境管理系统，为安装包和依赖项提供了替代 pip 的优秀方案。其隔离环境特性使其特别适合数据科学和[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)工作。更多详情，请访问 [Anaconda](https://anaconda.org/conda-forge/ultralytics) 上的 Ultralytics Conda 包，并在 [GitHub](https://github.com/conda-forge/ultralytics-feedstock/) 上查看 Ultralytics feedstock 仓库以获取包更新信息。

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics?logo=condaforge)](https://anaconda.org/conda-forge/ultralytics)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)
[![Conda Recipe](https://img.shields.io/badge/recipe-ultralytics-green.svg)](https://anaconda.org/conda-forge/ultralytics)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)

## 您将学到

- 设置 Conda 环境
- 通过 Conda 安装 Ultralytics
- 在您的环境中初始化 Ultralytics
- 结合 Conda 使用 Ultralytics Docker 镜像

---

## 前提条件

- 您的系统上应已安装 Anaconda 或 Miniconda。如果尚未安装，请从 [Anaconda](https://www.anaconda.com/) 或 [Miniconda](https://www.anaconda.com/docs/main) 下载并安装。

---

## 设置 Conda 环境

首先，创建一个新的 Conda 环境。打开终端并运行以下命令：

```bash
conda create --name ultralytics-env python=3.11 -y
```

激活新环境：

```bash
conda activate ultralytics-env
```

---

## 安装 Ultralytics

您可以从 conda-forge 频道安装 Ultralytics 包。执行以下命令：

```bash
conda install -c conda-forge ultralytics
```

### 关于 CUDA 环境的说明

如果您在启用 CUDA 的环境中工作，建议将 `ultralytics`、`pytorch` 和 `pytorch-cuda` 一起安装以解决任何冲突：

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

---

## 使用 Ultralytics

安装 Ultralytics 后，您就可以开始使用其强大的功能进行[目标检测](https://www.ultralytics.com/glossary/object-detection)、[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)等任务。例如，要对图像进行预测，可以运行：

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # 初始化模型
results = model("path/to/image.jpg")  # 执行推理
results[0].show()  # 显示第一张图像的推理结果
```

---

## Ultralytics Conda Docker 镜像

如果您更喜欢使用 Docker，Ultralytics 提供了包含 Conda 环境的 Docker 镜像。您可以从 [DockerHub](https://hub.docker.com/r/ultralytics/ultralytics) 拉取这些镜像。

拉取最新的 Ultralytics 镜像：

```bash
# 将镜像名称设为变量
t=ultralytics/ultralytics:latest-conda

# 从 Docker Hub 拉取最新的 Ultralytics 镜像
sudo docker pull $t
```

运行镜像：

```bash
# 在支持 GPU 的容器中运行 Ultralytics 镜像
sudo docker run -it --ipc=host --runtime=nvidia --gpus all $t            # 所有 GPU
sudo docker run -it --ipc=host --runtime=nvidia --gpus '"device=2,3"' $t # 指定 GPU
```

## 使用 Libmamba 加速安装

如果您希望[加快 Conda 中的包安装速度](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community)，可以选择使用 `libmamba`，这是一个快速、跨平台且具有依赖感知能力的包管理器，可替代 Conda 的默认求解器。

### 如何启用 Libmamba

要启用 `libmamba` 作为 Conda 的求解器，可以执行以下步骤：

1. 首先，安装 `conda-libmamba-solver` 包。如果您的 Conda 版本为 4.11 或更高，可以跳过此步骤，因为 `libmamba` 已默认包含。

    ```bash
    conda install conda-libmamba-solver
    ```

2. 接下来，配置 Conda 使用 `libmamba` 作为求解器：

    ```bash
    conda config --set solver libmamba
    ```

这样就完成了！您的 Conda 安装现在将使用 `libmamba` 作为求解器，从而使包安装过程更加快速。

---

您已成功设置 Conda 环境，安装了 Ultralytics 包，现在可以开始探索其功能了。有关更多高级教程和示例，请参阅 [Ultralytics 文档](../index.md)。

## 常见问题解答

### 为 Ultralytics 项目设置 Conda 环境的流程是什么？

为 Ultralytics 项目设置 Conda 环境非常简单，可确保顺畅的包管理。首先，使用以下命令创建一个新的 Conda 环境：

```bash
conda create --name ultralytics-env python=3.11 -y
```

然后，使用以下命令激活新环境：

```bash
conda activate ultralytics-env
```

最后，从 conda-forge 频道安装 Ultralytics：

```bash
conda install -c conda-forge ultralytics
```

### 为什么在 Ultralytics 项目中应使用 Conda 而非 pip 来管理依赖项？

Conda 是一个强大的包和环境管理系统，相较于 pip 具有多项优势。它能高效地管理依赖项，并确保所有必要的库相互兼容。Conda 的隔离环境可防止包之间的冲突，这在数据科学和机器学习项目中至关重要。此外，Conda 支持二进制包分发，可加快安装过程。

### 能否在启用 CUDA 的环境中使用 Ultralytics YOLO 以获得更快的性能？

可以，通过使用启用 CUDA 的环境可以提升性能。确保将 `ultralytics`、`pytorch` 和 `pytorch-cuda` 一起安装以避免冲突：

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

此设置可启用 GPU 加速，这对于[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型训练和推理等密集型任务至关重要。有关更多信息，请访问 [Ultralytics 安装指南](../quickstart.md)。

### 使用带有 Conda 环境的 Ultralytics Docker 镜像有哪些好处？

使用 Ultralytics Docker 镜像可以确保一致且可复现的环境，消除"在我机器上能运行"的问题。这些镜像包含预配置的 Conda 环境，简化了设置流程。您可以使用以下命令拉取并运行最新的 Ultralytics Docker 镜像：

```bash
sudo docker pull ultralytics/ultralytics:latest-conda
sudo docker run -it --ipc=host --runtime=nvidia --gpus all ultralytics/ultralytics:latest-conda            # 所有 GPU
sudo docker run -it --ipc=host --runtime=nvidia --gpus '"device=2,3"' ultralytics/ultralytics:latest-conda # 指定 GPU
```

这种方式非常适合在生产环境中部署应用或运行复杂的工作流，无需手动配置。了解更多关于 [Ultralytics Conda Docker 镜像](../quickstart.md)的信息。

### 如何在 Ultralytics 环境中加速 Conda 包安装？

您可以使用 `libmamba`（Conda 的快速依赖求解器）来加速包安装过程。首先，安装 `conda-libmamba-solver` 包：

```bash
conda install conda-libmamba-solver
```

然后配置 Conda 使用 `libmamba` 作为求解器：

```bash
conda config --set solver libmamba
```

此设置可提供更快、更高效的包管理。有关优化环境的更多提示，请阅读 [libmamba 安装说明](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community)。