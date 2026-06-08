---
comments: true
description: 了解如何使用 pip、conda 或 Docker 安装 Ultralytics。按照我们的分步指南，轻松完成 Ultralytics YOLO 的安装配置。
keywords: Ultralytics, YOLO26, YOLO11, 安装 Ultralytics, pip, conda, Docker, GitHub, 机器学习, 目标检测
---

# 安装 Ultralytics

Ultralytics 提供多种安装方式，包括 pip、conda 和 Docker。你可以通过 `ultralytics` pip 包安装最新的稳定版 YOLO，也可以克隆 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics) 获取最新版本。Docker 也是一种选择，可在隔离容器中运行该包，无需本地安装。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/_a7cVL9hqnk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics YOLO 快速入门指南
</p>

!!! example "安装"

    ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)

    === "Pip 安装（推荐）"

        使用 pip 安装或更新 `ultralytics` 包，运行 `pip install -U ultralytics`。有关 `ultralytics` 包的更多详情，请访问 [Python Package Index (PyPI)](https://pypi.org/project/ultralytics/)。

        [![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://clickpy.clickhouse.com/dashboard/ultralytics)

        ```bash
        # 从 PyPI 安装或升级 ultralytics 包
        pip install -U ultralytics
        ```

        你也可以直接从 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics) 安装 `ultralytics`。如果你想要最新的开发版本，这会很有用。确保已安装 Git 命令行工具，然后运行：

        ```bash
        # 从 GitHub 安装 ultralytics 包
        pip install git+https://github.com/ultralytics/ultralytics.git@main
        ```

    === "Conda 安装"

        Conda 可作为 pip 的替代包管理器使用。更多详情请访问 [Anaconda](https://anaconda.org/conda-forge/ultralytics)。用于更新 conda 包的 Ultralytics feedstock 仓库可在 [GitHub](https://github.com/conda-forge/ultralytics-feedstock/) 上找到。

        [![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics?logo=condaforge)](https://anaconda.org/conda-forge/ultralytics) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Recipe](https://img.shields.io/badge/recipe-ultralytics-green.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)

        ```bash
        # 使用 conda 安装 ultralytics 包
        conda install -c conda-forge ultralytics
        ```

        !!! note

            如果你在 CUDA 环境中安装，最佳实践是在同一条命令中同时安装 `ultralytics`、`pytorch` 和 `pytorch-cuda`。这样 conda 包管理器可以解决依赖冲突。或者，在必要时最后安装 `pytorch-cuda` 来覆盖仅支持 CPU 的 `pytorch` 包。
            ```bash
            # 使用 conda 一次性安装所有包
            conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
            ```

        ### Conda Docker 镜像

        Ultralytics Conda Docker 镜像也可在 [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics) 上获取。这些镜像基于 [Miniconda3](https://www.anaconda.com/docs/main)，提供了一种在 Conda 环境中开始使用 `ultralytics` 的简便方式。

        ```bash
        # 将镜像名称设为变量
        t=ultralytics/ultralytics:latest-conda

        # 从 Docker Hub 拉取最新的 ultralytics 镜像
        sudo docker pull $t

        # 在容器中运行 ultralytics 镜像并启用 GPU 支持
        sudo docker run -it --ipc=host --runtime=nvidia --gpus all $t            # 所有 GPU
        sudo docker run -it --ipc=host --runtime=nvidia --gpus '"device=2,3"' $t # 指定 GPU
        ```

    === "Git 克隆"

        克隆 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics)，如果你有兴趣参与开发或希望尝试最新的源代码。克隆后，进入目录并使用 pip 以可编辑模式 `-e` 安装该包。

        [![GitHub last commit](https://img.shields.io/github/last-commit/ultralytics/ultralytics?logo=github)](https://github.com/ultralytics/ultralytics) [![GitHub commit activity](https://img.shields.io/github/commit-activity/t/ultralytics/ultralytics)](https://github.com/ultralytics/ultralytics)

        ```bash
        # 克隆 ultralytics 仓库
        git clone https://github.com/ultralytics/ultralytics

        # 进入克隆的目录
        cd ultralytics

        # 以可编辑模式安装包，用于开发
        pip install -e .
        ```

    === "Docker"

        使用 Docker 在隔离容器中运行 `ultralytics` 包，确保在各种环境中性能一致。从 [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics) 选择官方 `ultralytics` 镜像之一，你可以避免本地安装的复杂性，并获得经过验证的工作环境。Ultralytics 提供五种主要的受支持 Docker 镜像，每种都旨在提供高兼容性和高效率：

        [![Docker Image Version](https://img.shields.io/docker/v/ultralytics/ultralytics?sort=semver&logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics) [![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/ultralytics)](https://hub.docker.com/r/ultralytics/ultralytics)

        - **Dockerfile：** 推荐用于训练的 GPU 镜像。
        - **Dockerfile-arm64：** 针对 ARM64 架构优化，适用于 Raspberry Pi 和其他基于 ARM64 平台的设备部署。
        - **Dockerfile-cpu：** 基于 Ubuntu 的仅 CPU 版本，适用于推理和无 GPU 环境。
        - **Dockerfile-jetson：** 专为 [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson) 设备定制，集成了针对这些平台优化的 GPU 支持。
        - **Dockerfile-python：** 仅包含 Python 和必要依赖的最小镜像，非常适合轻量级应用和开发。
        - **Dockerfile-conda：** 基于 Miniconda3，使用 conda 安装 `ultralytics` 包。

        以下是获取最新镜像并执行它的命令：

        ```bash
        # 将镜像名称设为变量
        t=ultralytics/ultralytics:latest

        # 从 Docker Hub 拉取最新的 ultralytics 镜像
        sudo docker pull $t

        # 在容器中运行 ultralytics 镜像并启用 GPU 支持
        sudo docker run -it --ipc=host --runtime=nvidia --gpus all $t            # 所有 GPU
        sudo docker run -it --ipc=host --runtime=nvidia --gpus '"device=2,3"' $t # 指定 GPU
        ```

        上述命令使用最新的 `ultralytics` 镜像初始化一个 Docker 容器。`-it` 标志分配一个伪 TTY 并保持 stdin 打开，允许与容器交互。`--ipc=host` 标志将 IPC（进程间通信）命名空间设置为主机，这对于进程间共享内存至关重要。`--gpus all` 标志允许容器内访问所有可用的 GPU，对于需要 GPU 计算的任务至关重要。

        注意：要在容器内操作本地机器上的文件，请使用 Docker 卷将本地目录挂载到容器中：

        ```bash
        # 将本地目录挂载到容器内的目录
        sudo docker run -it --ipc=host --runtime=nvidia --gpus all -v /path/on/host:/path/in/container $t
        ```

        将 `/path/on/host` 替换为本地机器上的目录路径，将 `/path/in/container` 替换为 Docker 容器内期望的路径。

        有关高级 Docker 用法，请参阅 [Ultralytics Docker 指南](guides/docker-quickstart.md)。

参阅 `ultralytics` 的 [pyproject.toml](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml) 文件获取依赖列表。请注意，以上所有示例都会安装所有必需的依赖。

!!! tip

    [PyTorch](https://www.ultralytics.com/glossary/pytorch) 的要求因操作系统和 CUDA 需求而异，因此请先按照 [PyTorch](https://pytorch.org/get-started/locally/) 的说明安装 PyTorch。

    <a href="https://pytorch.org/get-started/locally/">
        <img width="800" alt="不同平台的 PyTorch 安装选择器" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/pytorch-installation-instructions.avif">
    </a>

## 无头服务器安装

对于没有显示器的服务器环境（例如云虚拟机、Docker 容器、CI/CD 流水线），请使用 `ultralytics-opencv-headless` 包。它与标准的 `ultralytics` 包相同，但依赖 `opencv-python-headless` 而不是 `opencv-python`，避免了不必要的 GUI 依赖和潜在的 `libGL` 错误。

!!! example "无头安装"

    ```bash
    pip install ultralytics-opencv-headless
    ```

两个包提供相同的功能和 API。无头版本仅排除了需要显示库的 OpenCV GUI 组件。

## 高级安装

虽然标准安装方法涵盖了大多数使用场景，但你可能需要为开发或自定义配置更量身定制的设置。

!!! example "高级方法"

    === "从 Fork 安装"

        如果你需要持久化的自定义修改，可以 fork Ultralytics 仓库，修改 `pyproject.toml` 或其他代码，然后从你的 fork 安装。

        1.  **Fork** [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics) 到你的 GitHub 账户。
        2.  **克隆** 你的 fork 到本地：
            ```bash
            git clone https://github.com/YOUR_USERNAME/ultralytics.git
            cd ultralytics
            ```
        3.  为你的修改 **创建新分支**：
            ```bash
            git checkout -b my-custom-branch
            ```
        4.  根据需要 **修改** `pyproject.toml` 或其他文件。
        5.  **提交并推送** 你的更改：
            ```bash
            git add .
            git commit -m "我的自定义修改"
            git push origin my-custom-branch
            ```
        6.  使用 `git+https` 语法通过 pip **安装**，指向你的分支：
            ```bash
            pip install git+https://github.com/YOUR_USERNAME/ultralytics.git@my-custom-branch
            ```

    === "本地克隆并安装"

        克隆仓库到本地，根据需要修改文件，然后以可编辑模式安装。

        1.  **克隆** Ultralytics 仓库：
            ```bash
            git clone https://github.com/ultralytics/ultralytics
            cd ultralytics
            ```
        2.  根据需要 **修改** `pyproject.toml` 或其他文件。
        3.  以可编辑模式（`-e`）**安装** 包。Pip 将使用你修改后的 `pyproject.toml` 来解析依赖：
            ```bash
            pip install -e .
            ```

        这种方法适用于开发或在提交前测试本地更改。

    === "使用 requirements.txt"

        在你的 `requirements.txt` 文件中指定自定义的 Ultralytics fork，以确保团队中安装的一致性。

        ```text title="requirements.txt"
        # 从指定的 git 分支安装 ultralytics
        git+https://github.com/YOUR_USERNAME/ultralytics.git@my-custom-branch

        # 其他项目依赖
        flask
        ```

        从文件安装依赖：
        ```bash
        pip install -r requirements.txt
        ```

## 使用 Ultralytics CLI

Ultralytics 命令行接口（CLI）允许使用简单的单行命令，无需 Python 环境。CLI 不需要自定义或编写 Python 代码；所有任务都可以通过 `yolo` 命令从终端运行。有关从命令行使用 YOLO 的更多信息，请参阅 [CLI 指南](usage/cli.md)。

!!! example

    === "语法"

        Ultralytics `yolo` 命令使用以下语法：
        ```bash
        yolo TASK MODE ARGS
        ```
        - `TASK`（可选）是以下之一：（[detect](tasks/detect.md)、[segment](tasks/segment.md)、[classify](tasks/classify.md)、[pose](tasks/pose.md)、[obb](tasks/obb.md)）
        - `MODE`（必填）是以下之一：（[train](modes/train.md)、[val](modes/val.md)、[predict](modes/predict.md)、[export](modes/export.md)、[track](modes/track.md)、[benchmark](modes/benchmark.md)）
        - `ARGS`（可选）是 `arg=value` 对，如 `imgsz=640`，用于覆盖默认值。

        在完整的 [配置指南](usage/cfg.md) 中查看所有 `ARGS`，或使用 `yolo cfg` CLI 命令。

    === "训练"

        使用初始学习率 0.01 训练检测模型 10 个 [epoch](https://www.ultralytics.com/glossary/epoch)：
        ```bash
        yolo train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01
        ```

    === "预测"

        使用预训练的分割模型对 YouTube 视频进行预测，图像尺寸为 320：
        ```bash
        yolo predict model=yolo26n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "验证"

        使用批量大小为 1、图像尺寸为 640 验证预训练检测模型：
        ```bash
        yolo val model=yolo26n.pt data=coco8.yaml batch=1 imgsz=640
        ```

    === "导出"

        将 YOLO26n 分类模型导出为 ONNX 格式，图像尺寸为 224x128（无需指定 TASK）：
        ```bash
        yolo export model=yolo26n-cls.pt format=onnx imgsz=224,128
        ```

    === "计数"

        使用 YOLO26 统计视频或实时流中的物体数量：
        ```bash
        yolo solutions count show=True

        yolo solutions count source="path/to/video.mp4" # 指定视频文件路径
        ```

    === "健身训练"

        使用 YOLO26 姿态模型监控健身训练动作：
        ```bash
        yolo solutions workout show=True

        yolo solutions workout source="path/to/video.mp4" # 指定视频文件路径

        # 使用关键点进行腹肌训练
        yolo solutions workout kpts="[5, 11, 13]" # 左侧
        yolo solutions workout kpts="[6, 12, 14]" # 右侧
        ```

    === "排队"

        使用 YOLO26 统计指定队列或区域内的物体数量：
        ```bash
        yolo solutions queue show=True

        yolo solutions queue source="path/to/video.mp4" # 指定视频文件路径

        yolo solutions queue region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]" # 配置队列坐标
        ```

    === "使用 Streamlit 推理"

        使用 [Streamlit](https://docs.ultralytics.com/reference/solutions/streamlit_inference) 在 Web 浏览器中执行目标检测、实例分割或姿态估计：
        ```bash
        yolo solutions inference

        yolo solutions inference model="path/to/model.pt" # 使用 Ultralytics Python 包微调的模型
        ```

    === "特殊命令"

        运行特殊命令查看版本、查看设置、运行检查等：
        ```bash
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        yolo solutions help
        ```

!!! warning

    参数必须以 `arg=value` 对的形式传递，用等号 `=` 分隔，用空格分隔。不要在参数之间使用 `--` 参数前缀或逗号 `,`。

    - `yolo predict model=yolo26n.pt imgsz=640 conf=0.25`  ✅
    - `yolo predict model yolo26n.pt imgsz 640 conf 0.25`  ❌（缺少 `=`）
    - `yolo predict model=yolo26n.pt, imgsz=640, conf=0.25`  ❌（不要使用 `,`）
    - `yolo predict --model yolo26n.pt --imgsz 640 --conf 0.25`  ❌（不要使用 `--`）
    - `yolo solution model=yolo26n.pt imgsz=640 conf=0.25` ❌（应使用 `solutions`，而非 `solution`）

[CLI 指南](usage/cli.md){ .md-button }

## 使用 Ultralytics Python 接口

Ultralytics YOLO Python 接口可无缝集成到 Python 项目中，轻松加载、运行和处理模型输出。Python 接口设计简洁，用户可以快速实现 [目标检测](https://www.ultralytics.com/glossary/object-detection)、分割和分类。这使得 YOLO Python 接口成为将这些功能融入 Python 项目的宝贵工具。

例如，用户只需几行代码即可加载模型、训练模型、评估其性能并将其导出为 ONNX 格式。请参阅 [Python 指南](usage/python.md) 了解有关在 Python 项目中使用 YOLO 的更多信息。

!!! example

    ```python
    from ultralytics import YOLO

    # 从零开始创建一个新的 YOLO 模型
    model = YOLO("yolo26n.yaml")

    # 加载预训练的 YOLO 模型（推荐用于训练）
    model = YOLO("yolo26n.pt")

    # 使用 'coco8.yaml' 数据集训练模型 3 个 epoch
    results = model.train(data="coco8.yaml", epochs=3)

    # 在验证集上评估模型的性能
    results = model.val()

    # 使用模型对图像进行目标检测
    results = model("https://ultralytics.com/images/bus.jpg")

    # 将模型导出为 ONNX 格式
    success = model.export(format="onnx")
    ```

[Python 指南](usage/python.md){.md-button .md-button--primary}

## Ultralytics 设置

Ultralytics 库包含一个 `SettingsManager`，用于对实验进行精细控制，允许用户轻松访问和修改设置。这些设置存储在该环境用户配置目录中的 JSON 文件中，可以在 Python 环境中或通过命令行接口（CLI）查看或修改。

### 查看设置

查看当前配置的设置：

!!! example "查看设置"

    === "Python"

        使用 Python 查看设置，从 `ultralytics` 模块导入 `settings` 对象。使用以下命令打印和返回设置：
        ```python
        from ultralytics import settings

        # 查看所有设置
        print(settings)

        # 返回特定设置
        value = settings["runs_dir"]
        ```

    === "CLI"

        命令行接口允许你通过以下方式检查设置：
        ```bash
        yolo settings
        ```

### 修改设置

Ultralytics 可以通过以下方式轻松修改设置：

!!! example "更新设置"

    === "Python"

        在 Python 中，使用 `settings` 对象的 `update` 方法：
        ```python
        from ultralytics import settings

        # 更新一项设置
        settings.update({"runs_dir": "/path/to/runs"})

        # 更新多项设置
        settings.update({"runs_dir": "/path/to/runs", "tensorboard": False})

        # 将设置重置为默认值
        settings.reset()
        ```

    === "CLI"

        使用命令行接口修改设置：
        ```bash
        # 更新一项设置
        yolo settings runs_dir='/path/to/runs'

        # 更新多项设置
        yolo settings runs_dir='/path/to/runs' tensorboard=False

        # 将设置重置为默认值
        yolo settings reset
        ```

### 设置说明

下表概述了 Ultralytics 中的可调整设置，包括示例值、数据类型和描述。

| 名称               | 示例值                | 数据类型 | 描述                                                                                      |
| ------------------ | --------------------- | -------- | ----------------------------------------------------------------------------------------- |
| `settings_version` | `'0.0.4'`             | `str`    | Ultralytics _settings_ 版本（与 Ultralytics [pip] 版本不同）                               |
| `datasets_dir`     | `'/path/to/datasets'` | `str`    | 数据集存储目录                                                                             |
| `weights_dir`      | `'/path/to/weights'`  | `str`    | 模型权重存储目录                                                                           |
| `runs_dir`         | `'/path/to/runs'`     | `str`    | 实验运行结果存储目录                                                                       |
| `uuid`             | `'a1b2c3d4'`          | `str`    | 当前设置的唯一标识符                                                                       |
| `sync`             | `True`                | `bool`   | 将分析和崩溃数据同步到 [Ultralytics Platform]                                              |
| `api_key`          | `''`                  | `str`    | [Ultralytics Platform] API 密钥                                                            |
| `clearml`          | `True`                | `bool`   | 是否使用 [ClearML] 日志记录                                                                |
| `comet`            | `True`                | `bool`   | 是否使用 [Comet ML] 进行实验跟踪和可视化                                                    |
| `dvc`              | `True`                | `bool`   | 是否使用 [DVC for experiment tracking] 进行实验跟踪和版本控制                               |
| `hub`              | `True`                | `bool`   | 是否使用 [Ultralytics Platform] 集成                                                        |
| `mlflow`           | `True`                | `bool`   | 是否使用 [MLFlow] 进行实验跟踪                                                              |
| `neptune`          | `True`                | `bool`   | 是否使用 [Neptune] 进行实验跟踪                                                             |
| `raytune`          | `True`                | `bool`   | 是否使用 [Ray Tune] 进行 [超参数调优](https://www.ultralytics.com/glossary/hyperparameter-tuning) |
| `tensorboard`      | `True`                | `bool`   | 是否使用 [TensorBoard] 进行可视化                                                           |
| `wandb`            | `True`                | `bool`   | 是否使用 [Weights & Biases] 日志记录                                                        |
| `vscode_msg`       | `True`                | `bool`   | 当检测到 VS Code 终端时，是否提示下载 [Ultralytics-Snippets] 扩展                            |

随着项目或实验的推进，请重新审视这些设置以确保最佳配置。

## 常见问题

### 如何使用 pip 安装 Ultralytics？

使用 pip 安装 Ultralytics：

```bash
pip install -U ultralytics
```

这将从 [PyPI](https://pypi.org/project/ultralytics/) 安装最新的稳定版 `ultralytics` 包。要从 GitHub 安装开发版本：

```bash
pip install git+https://github.com/ultralytics/ultralytics.git
```

确保系统上已安装 Git 命令行工具。

### 可以使用 conda 安装 Ultralytics YOLO 吗？

可以，使用 conda 安装 Ultralytics YOLO：

```bash
conda install -c conda-forge ultralytics
```

这种方法是 pip 的一个很好的替代方案，可以确保与其他包的兼容性。对于 CUDA 环境，请将 `ultralytics`、`pytorch` 和 `pytorch-cuda` 一起安装以解决冲突：

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

更多说明请参阅 [Conda 快速入门指南](guides/conda-quickstart.md)。

### 使用 Docker 运行 Ultralytics YOLO 有哪些优势？

Docker 为 Ultralytics YOLO 提供了一个隔离、一致的环境，确保在不同系统上性能稳定，并避免了本地安装的复杂性。官方 Docker 镜像可在 [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics) 上获取，包含 GPU、CPU、ARM64、[NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson) 和 Conda 等多种变体。拉取并运行最新镜像：

```bash
# 从 Docker Hub 拉取最新的 ultralytics 镜像
sudo docker pull ultralytics/ultralytics:latest

# 在容器中运行 ultralytics 镜像并启用 GPU 支持
sudo docker run -it --ipc=host --runtime=nvidia --gpus all ultralytics/ultralytics:latest
```

详细的 Docker 说明请参阅 [Docker 快速入门指南](guides/docker-quickstart.md)。

### 如何克隆 Ultralytics 仓库用于开发？

克隆 Ultralytics 仓库并设置开发环境：

```bash
# 克隆 ultralytics 仓库
git clone https://github.com/ultralytics/ultralytics

# 进入克隆的目录
cd ultralytics

# 以可编辑模式安装包，用于开发
pip install -e .
```

这样可以为项目贡献代码或尝试最新的源代码。详情请访问 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics)。

### 为什么应该使用 Ultralytics YOLO CLI？

Ultralytics YOLO CLI 简化了运行目标检测任务，无需编写 Python 代码，可以直接从终端使用单行命令进行训练、验证和预测。基本语法为：

```bash
yolo TASK MODE ARGS
```

例如，训练一个检测模型：

```bash
yolo train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01
```

在完整的 [CLI 指南](usage/cli.md) 中探索更多命令和用法示例。

<!-- Article Links -->

[Ultralytics Platform]: https://platform.ultralytics.com
[pip]: https://pypi.org/project/ultralytics/
[DVC for experiment tracking]: https://dvc.org/doc/dvclive/ml-frameworks/yolo
[Comet ML]: https://bit.ly/yolov8-readme-comet
[ClearML]: ./integrations/clearml.md
[MLFlow]: ./integrations/mlflow.md
[Neptune]: https://neptune.ai/
[Tensorboard]: ./integrations/tensorboard.md
[Ray Tune]: ./integrations/ray-tune.md
[Weights & Biases]: ./integrations/weights-biases.md
[Ultralytics-Snippets]: ./integrations/vscode.md