---
comments: true
description: 了解如何将 Ultralytics YOLO26 与 NVIDIA Triton Inference Server 集成，实现可扩展、高性能的 AI 模型部署。
keywords: Triton Inference Server, YOLO26, Ultralytics, NVIDIA, 深度学习, AI 模型部署, ONNX, 可扩展推理
---

# Triton Inference Server 与 Ultralytics YOLO26

[Triton Inference Server](https://developer.nvidia.com/dynamo-triton)（原名 TensorRT Inference Server）是 NVIDIA 开发的开源软件解决方案。它提供了针对 NVIDIA GPU 优化的云端推理解决方案。Triton 简化了 AI 模型在生产环境中的大规模部署。将 [Ultralytics YOLO26](../models/yolo26.md) 与 Triton Inference Server 集成，可以部署可扩展、高性能的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)推理工作负载。本指南提供了设置和测试该集成的步骤。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/NQDtfSi5QF4"
    title="NVIDIA Triton Inference Server 入门"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>NVIDIA Triton Inference Server 入门。
</p>

## 什么是 Triton Inference Server？

Triton Inference Server 专为在生产环境中部署多种 AI 模型而设计。它支持广泛的深度学习和[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)框架，包括 [PyTorch](https://www.ultralytics.com/glossary/pytorch)、[TensorFlow](../integrations/tf-savedmodel.md)、[ONNX](../integrations/onnx.md)、[OpenVINO](../integrations/openvino.md)、[TensorRT](../integrations/tensorrt.md) 等。其主要用例包括：

- 从单个服务器实例提供多个模型
- 无需重启服务器即可动态加载和卸载模型
- 集成推理，允许多个模型协同使用以获得结果
- 模型版本管理，支持 A/B 测试和滚动更新

## Triton Inference Server 的关键优势

将 Triton Inference Server 与 [Ultralytics YOLO26](../models/yolo26.md) 结合使用具有以下优势：

- **自动批处理**：在处理前将多个 AI 请求分组，降低延迟并提高推理速度
- **Kubernetes 集成**：云原生设计，可与 Kubernetes 无缝协作，用于管理和扩展 AI 应用
- **硬件专用优化**：充分利用 NVIDIA GPU 以获得最佳性能
- **框架灵活性**：支持多种 AI 框架，包括 [PyTorch](https://www.ultralytics.com/glossary/pytorch)、[TensorFlow](../integrations/tf-savedmodel.md)、[ONNX](../integrations/onnx.md)、[OpenVINO](../integrations/openvino.md) 和 [TensorRT](../integrations/tensorrt.md)
- **开源可定制**：可根据特定需求进行修改，确保各种 AI 应用的灵活性

## 前提条件

在继续之前，请确保满足以下前提条件：

- 机器上已安装 Docker 或 Podman
- 安装 `ultralytics`：
    ```bash
    pip install ultralytics
    ```
- 安装 `tritonclient`：
    ```bash
    pip install tritonclient[all]
    ```

## 设置 Triton Inference Server

运行以下完整设置代码块，将 [Ultralytics YOLO26](../models/yolo26.md) 导出为 [ONNX](../integrations/onnx.md)、构建 Triton 模型仓库并启动 Triton Inference Server：

!!! note

    使用脚本中的 `runtime` 开关选择容器引擎：

    - Docker 用户设置 `runtime = "docker"`
    - Podman 用户设置 `runtime = "podman"`

```python
import contextlib
import subprocess
import time
from pathlib import Path

from tritonclient.http import InferenceServerClient

from ultralytics import YOLO

runtime = "docker"  # 使用 Podman 时设置为 "podman"

# 1) 将 YOLO26 导出为 ONNX 格式

# 加载模型
model = YOLO("yolo26n.pt")  # 加载官方模型

# 在导出时获取元数据。元数据需添加到 config.pbtxt 中。参见下一节。
metadata = []


def export_cb(exporter):
    metadata.append(exporter.metadata)


model.add_callback("on_export_end", export_cb)

# 导出模型
onnx_file = model.export(format="onnx", dynamic=True)


# 2) 设置 Triton 模型仓库

# 定义路径
model_name = "yolo"
triton_repo_path = Path("tmp") / "triton_repo"
triton_model_path = triton_repo_path / model_name

# 创建目录
(triton_model_path / "1").mkdir(parents=True, exist_ok=True)

# 将 ONNX 模型移动到 Triton 模型路径
Path(onnx_file).rename(triton_model_path / "1" / "model.onnx")

# 创建配置文件
(triton_model_path / "config.pbtxt").touch()

data = """
# 添加元数据
parameters {
  key: "metadata"
  value {
    string_value: "%s"
  }
}

# （可选）启用 TensorRT 进行 GPU 推理
# 首次运行会因 TensorRT 引擎转换而较慢
optimization {
  execution_accelerators {
    gpu_execution_accelerator {
      name: "tensorrt"
      parameters {
        key: "precision_mode"
        value: "FP16"
      }
      parameters {
        key: "max_workspace_size_bytes"
        value: "3221225472"
      }
      parameters {
        key: "trt_engine_cache_enable"
        value: "1"
      }
      parameters {
        key: "trt_engine_cache_path"
        value: "/models/yolo/1"
      }
    }
  }
}
""" % metadata[0]  # noqa

with open(triton_model_path / "config.pbtxt", "w") as f:
    f.write(data)

# 3) 运行 Triton Inference Server

# 定义镜像 https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
tag = "nvcr.io/nvidia/tritonserver:26.02-py3"  # 16.17 GB（压缩大小）

subprocess.call(f"{runtime} pull {tag}", shell=True)

# Docker 和 Podman 的 GPU 参数不同
gpu_flags = "--device nvidia.com/gpu=all" if runtime == "podman" else "--runtime=nvidia --gpus all"

container_name = "triton_server"

# 注意：卷挂载中的 :z 标志对于使用 SELinux 的系统（如 Fedora/RHEL）是必需的
subprocess.call(
    f"{runtime} run -d --rm --name {container_name} {gpu_flags} -v {triton_repo_path.absolute()}:/models:z -p 8000:8000 {tag} tritonserver --model-repository=/models",
    shell=True,
)

# 等待 Triton 服务器启动
triton_client = InferenceServerClient(url="127.0.0.1:8000", verbose=False, ssl=False)

# 等待模型就绪
for _ in range(10):
    with contextlib.suppress(Exception):
        assert triton_client.is_model_ready(model_name)
        break
    time.sleep(1)
```

## 运行推理

使用 Triton Server 模型运行推理：

```python
from ultralytics import YOLO

# 加载 Triton Server 模型
model = YOLO("http://127.0.0.1:8000/yolo", task="detect")

# 在服务器上运行推理
results = model("path/to/image.jpg")
```

清理容器（`runtime` 和 `container_name` 在上面的设置代码块中已定义）：

```python
import subprocess

runtime = "docker"  # 使用 Podman 时设置为 "podman"
container_name = "triton_server"  # 终止指定名称的容器
subprocess.call(f"{runtime} kill {container_name}", shell=True)
```

## TensorRT 优化（可选）

为了获得更高的性能，你可以将 [TensorRT](../integrations/tensorrt.md) 与 Triton Inference Server 结合使用。TensorRT 是专为 NVIDIA GPU 构建的高性能深度学习优化器，可以显著提高推理速度。

将 [TensorRT](../integrations/tensorrt.md) 与 Triton 结合使用的主要优势包括：

- 相比未优化模型，推理速度最高可提升 36 倍
- 硬件专用优化，最大化 GPU 利用率
- 在保持精度的同时支持降低精度格式（INT8、FP16）
- 层融合以减少计算开销

要直接使用 TensorRT，你可以将 [Ultralytics YOLO26](../models/yolo26.md) 模型导出为 TensorRT 格式：

```python
from ultralytics import YOLO

# 加载 YOLO26 模型
model = YOLO("yolo26n.pt")

# 将模型导出为 TensorRT 格式
model.export(format="engine")  # 生成 'yolo26n.engine'
```

有关 TensorRT 优化的更多信息，请参阅 [TensorRT 集成指南](../integrations/tensorrt.md)。

---

通过以上步骤，你可以在 Triton Inference Server 上高效部署和运行 [Ultralytics YOLO26](../models/yolo26.md) 模型，为深度学习推理任务提供可扩展且高性能的解决方案。如果遇到任何问题或有进一步疑问，请参阅 [官方 Triton 文档](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) 或联系 Ultralytics 社区获取支持。

## 常见问题

### 如何将 Ultralytics YOLO26 与 NVIDIA Triton Inference Server 一起设置？

将 [Ultralytics YOLO26](../models/yolo26.md) 与 [NVIDIA Triton Inference Server](https://developer.nvidia.com/dynamo-triton) 一起设置涉及以下关键步骤：

1. **将 YOLO26 导出为 ONNX 格式**：

    ```python
    from ultralytics import YOLO

    # 加载模型
    model = YOLO("yolo26n.pt")  # 加载官方模型

    # 将模型导出为 ONNX 格式
    onnx_file = model.export(format="onnx", dynamic=True)
    ```

2. **设置 Triton 模型仓库**：

    ```python
    from pathlib import Path

    # 定义路径
    model_name = "yolo"
    triton_repo_path = Path("tmp") / "triton_repo"
    triton_model_path = triton_repo_path / model_name

    # 创建目录
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    Path(onnx_file).rename(triton_model_path / "1" / "model.onnx")
    (triton_model_path / "config.pbtxt").touch()
    ```

3. **运行 Triton 服务器**：

    ```python
    import contextlib
    import subprocess
    import time

    from tritonclient.http import InferenceServerClient

    # 定义镜像 https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
    tag = "nvcr.io/nvidia/tritonserver:26.02-py3"

    runtime = "docker"  # 使用 Podman 时设置为 "podman"
    subprocess.call(f"{runtime} pull {tag}", shell=True)

    # Docker 和 Podman 的 GPU 参数不同
    gpu_flags = "--device nvidia.com/gpu=all" if runtime == "podman" else "--runtime=nvidia --gpus all"

    container_name = "triton_server"
    subprocess.call(
        f"{runtime} run -d --rm --name {container_name} {gpu_flags} -v {triton_repo_path.absolute()}:/models:z -p 8000:8000 {tag} tritonserver --model-repository=/models",
        shell=True,
    )

    triton_client = InferenceServerClient(url="127.0.0.1:8000", verbose=False, ssl=False)

    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready(model_name)
            break
        time.sleep(1)
    ```

此设置可以帮助你在 Triton Inference Server 上高效地大规模部署 [Ultralytics YOLO26](../models/yolo26.md) 模型，实现高性能 AI 模型推理。

### 将 Ultralytics YOLO26 与 NVIDIA Triton Inference Server 结合使用有哪些优势？

将 [Ultralytics YOLO26](../models/yolo26.md) 与 [NVIDIA Triton Inference Server](https://developer.nvidia.com/dynamo-triton) 集成具有以下优势：

- **可扩展的 AI 推理**：Triton 允许从单个服务器实例提供多个模型，支持动态模型加载和卸载，使其对于多样化的 AI 工作负载具有高度可扩展性。
- **高性能**：针对 NVIDIA GPU 优化，Triton Inference Server 确保高速推理操作，非常适合[目标检测](https://www.ultralytics.com/glossary/object-detection)等实时应用。
- **集成与模型版本管理**：Triton 的集成模式支持组合多个模型以改善结果，其模型版本管理支持 A/B 测试和滚动更新。
- **自动批处理**：Triton 自动将多个推理请求分组，显著提高吞吐量并降低延迟。
- **简化部署**：逐步优化 AI 工作流，无需彻底改造系统，便于高效扩展。

有关设置和运行 [Ultralytics YOLO26](../models/yolo26.md) 与 Triton 的详细说明，请参阅**设置 Triton Inference Server** 和**运行推理**部分。

### 为什么在 Triton Inference Server 之前应将 YOLO26 模型导出为 ONNX 格式？

将 [Ultralytics YOLO26](../models/yolo26.md) 模型在部署到 [NVIDIA Triton Inference Server](https://developer.nvidia.com/dynamo-triton) 之前导出为 ONNX（Open Neural Network Exchange）格式具有以下关键优势：

- **互操作性**：ONNX 格式支持在不同深度学习框架（如 PyTorch、TensorFlow）之间转换，确保更广泛的兼容性。
- **优化**：包括 Triton 在内的许多部署环境都针对 ONNX 进行了优化，从而实现更快的推理和更好的性能。
- **易于部署**：ONNX 在各种框架和平台上得到广泛支持，简化了在不同操作系统和硬件配置中的部署过程。
- **框架无关性**：转换为 ONNX 后，模型不再绑定到原始框架，更具可移植性。
- **标准化**：ONNX 提供了标准化的表示方式，有助于克服不同 AI 框架之间的兼容性问题。

要导出模型，请使用：

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
onnx_file = model.export(format="onnx", dynamic=True)
```

你可以按照 [ONNX 集成指南](https://docs.ultralytics.com/integrations/onnx) 中的步骤完成此过程。

### 可以在 Triton Inference Server 上使用 Ultralytics YOLO26 模型运行推理吗？

可以，你可以在 [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) 上使用 [Ultralytics YOLO26](../models/yolo26.md) 模型运行推理。一旦模型在 Triton 模型仓库中设置完毕且服务器正在运行，你可以按如下方式加载并运行推理：

```python
from ultralytics import YOLO

# 加载 Triton Server 模型
model = YOLO("http://127.0.0.1:8000/yolo", task="detect")

# 在服务器上运行推理
results = model("path/to/image.jpg")
```

这种方法让你在利用 Triton 优化的同时，使用熟悉的 Ultralytics YOLO 接口。

### Ultralytics YOLO26 与 TensorFlow 和 PyTorch 模型在部署方面相比如何？

与 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 和 PyTorch 模型相比，[Ultralytics YOLO26](../models/yolo26.md) 在部署方面具有以下独特优势：

- **实时性能**：针对实时目标检测任务进行了优化，[Ultralytics YOLO26](../models/yolo26.md) 提供了最先进的[精度](https://www.ultralytics.com/glossary/accuracy)和速度，非常适合需要实时视频分析的应用。
- **易于使用**：[Ultralytics YOLO26](../models/yolo26.md) 与 Triton Inference Server 无缝集成，支持多种导出格式（[ONNX](../integrations/onnx.md)、[TensorRT](../integrations/tensorrt.md)），使其在各种部署场景中都很灵活。
- **高级功能**：[Ultralytics YOLO26](../models/yolo26.md) 包含动态模型加载、模型版本管理和集成推理等功能，这些对于可扩展且可靠的 AI 部署至关重要。
- **简化的 API**：Ultralytics API 在不同部署目标之间提供一致的接口，降低了学习曲线和开发时间。
- **边缘优化**：[Ultralytics YOLO26](../models/yolo26.md) 模型在设计时考虑了边缘部署，即使在资源受限的设备上也能提供出色的性能。

更多详细信息，请参阅[模型导出指南](../modes/export.md)中的部署选项对比。
