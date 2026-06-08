---
comments: true
description: 使用 Ultralytics 导出工具将任意 PyTorch 模型（timm、torchvision 或自定义模型）导出为 ONNX、OpenVINO、CoreML、TensorFlow SavedModel、TorchScript、NCNN、MNN、PaddlePaddle 和 ExecuTorch 格式。
keywords: 导出 PyTorch 模型, PyTorch 转 ONNX, PyTorch 转 CoreML, PyTorch 转 OpenVINO, PyTorch 转 TensorFlow, 非 YOLO 导出, timm 导出, torchvision 导出, TorchScript 导出, NCNN 导出, MNN 导出, PaddlePaddle 导出, ExecuTorch 导出, TFLite 导出, Ultralytics 导出工具, torch.nn.Module 导出, 模型转换, 模型部署, PyTorch 部署
---

# 如何使用 Ultralytics 导出非 YOLO 的 PyTorch 模型

将 PyTorch 模型部署到生产环境通常意味着要应付每个目标平台的不同导出器：ONNX 用 `torch.onnx.export`，Apple 设备用 `coremltools`，TensorFlow 用 `onnx2tf`，NCNN 用 `pnnx`，等等。每种工具都有自己的 API、依赖怪癖和输出规范。

Ultralytics 提供了独立的导出工具，将多个后端封装在一个统一接口之下。您可以将任何 `torch.nn.Module`（包括 [timm](https://github.com/huggingface/pytorch-image-models) 图像模型、[torchvision](https://docs.pytorch.org/vision/) 分类器和检测器，或您自己的自定义架构）导出为 [ONNX](../integrations/onnx.md)、[TorchScript](../integrations/torchscript.md)、[OpenVINO](../integrations/openvino.md)、[CoreML](../integrations/coreml.md)、[NCNN](../integrations/ncnn.md)、[PaddlePaddle](../integrations/paddlepaddle.md)、[MNN](../integrations/mnn.md)、[ExecuTorch](../integrations/executorch.md) 和 [TensorFlow SavedModel](../integrations/tf-savedmodel.md)，无需逐一学习每种后端。

## 为什么使用 Ultralytics 导出非 YOLO 模型？

- **一套 API 覆盖 10 种格式：** 学习一种调用规范，而不是十几种。
- **共享工具接口：** 导出辅助函数位于 `ultralytics.utils.export` 下，一旦安装了后端包，您就可以在不同格式之间保持相同的调用模式。
- **与 YOLO 导出相同的代码路径：** 同样的辅助函数驱动着每一个 Ultralytics YOLO 导出。
- **内置 FP16 和 INT8 量化**，适用于支持该功能的格式（OpenVINO、CoreML、MNN、NCNN）。
- **支持 CPU 运行：** 导出过程本身不需要 GPU，您可以在任何笔记本电脑上本地运行。

## 快速开始

最快的方式是两行代码导出为 [ONNX](../integrations/onnx.md)，不涉及 YOLO 代码，除了 `pip install ultralytics onnx timm` 外无需额外设置：

```python
import timm
import torch

from ultralytics.utils.export import torch2onnx

model = timm.create_model("resnet18", pretrained=True).eval()
torch2onnx(model, torch.randn(1, 3, 224, 224), output_file="resnet18.onnx")
```

## 支持的导出格式

`torch2*` 函数接受一个标准的 `torch.nn.Module` 和一个示例输入张量。MNN、TF SavedModel 和 TF Frozen Graph 通过中间的 ONNX 或 Keras 产物进行转换。两种方式都不需要 YOLO 特定的属性。

| 格式              | 函数                    | 安装方式                                                             | 输出                           |
| ----------------- | ----------------------- | -------------------------------------------------------------------- | ------------------------------ |
| ONNX              | `torch2onnx()`          | `pip install onnx`                                                   | `.onnx` 文件                   |
| TorchScript       | `torch2torchscript()`   | 随 PyTorch 附带                                                      | `.torchscript` 文件            |
| OpenVINO          | `torch2openvino()`      | `pip install openvino`                                               | `_openvino_model/` 目录        |
| CoreML            | `torch2coreml()`        | `pip install coremltools`                                            | `.mlpackage`                   |
| TF SavedModel     | `onnx2saved_model()`    | [详见下方要求](#导出为-tensorflow-savedmodel)                        | `_saved_model/` 目录           |
| TF Frozen Graph   | `keras2pb()`            | [详见下方要求](#导出为-tensorflow-savedmodel)                        | `.pb` 文件                     |
| NCNN              | `torch2ncnn()`          | `pip install ncnn pnnx`                                              | `_ncnn_model/` 目录            |
| MNN               | `onnx2mnn()`            | `pip install MNN`                                                    | `.mnn` 文件                    |
| PaddlePaddle      | `torch2paddle()`        | `pip install paddlepaddle x2paddle`                                  | `_paddle_model/` 目录          |
| ExecuTorch        | `torch2executorch()`    | `pip install executorch`                                             | `_executorch_model/` 目录      |

!!! note "ONNX 作为中间格式"

    [MNN](../integrations/mnn.md)、[TF SavedModel](../integrations/tf-savedmodel.md) 和 TF Frozen Graph 的导出以 ONNX 作为中间步骤。先导出为 ONNX，再进行转换。

!!! tip "嵌入元数据"

    多个导出函数接受可选的 `metadata` 字典（例如 `torch2torchscript(..., metadata={"author": "me"})`），在格式支持的情况下，将自定义键值对嵌入导出的产物中。

## 分步示例

以下每个示例都使用相同的设置——来自 timm 的预训练 ResNet-18，处于评估模式：

```python
import timm
import torch

model = timm.create_model("resnet18", pretrained=True).eval()
im = torch.randn(1, 3, 224, 224)
```

!!! warning "导出前务必调用 `model.eval()`"

    Dropout、[批量归一化](https://www.ultralytics.com/glossary/batch-normalization) 和其他仅训练层在推理时的行为不同。跳过 `.eval()` 会导致导出结果不正确。

### 导出为 ONNX

```python
from ultralytics.utils.export import torch2onnx

torch2onnx(model, im, output_file="resnet18.onnx")
```

如需动态 batch size，传入 `dynamic` 字典：

```python
torch2onnx(model, im, output_file="resnet18_dyn.onnx", dynamic={"images": {0: "batch_size"}})
```

默认 opset 为 `14`，默认输入名称为 `"images"`。可通过 `opset`、`input_names` 或 `output_names` 参数覆盖。

### 导出为 TorchScript

无需额外依赖。底层使用 `torch.jit.trace`。

```python
from ultralytics.utils.export import torch2torchscript

torch2torchscript(model, im, output_file="resnet18.torchscript")
```

### 导出为 OpenVINO

```python
from ultralytics.utils.export import torch2openvino

ov_model = torch2openvino(model, im, output_dir="resnet18_openvino_model")
```

目录中包含固定名称的 `model.xml` 和 `model.bin` 对：

```
resnet18_openvino_model/
├── model.xml
└── model.bin
```

传入 `dynamic=True` 以支持动态输入形状，`half=True` 启用 FP16，或 `int8=True` 启用 INT8 量化。INT8 还需额外提供 `calibration_dataset` 参数。

需要 `openvino>=2024.0.0`（macOS 15.4+ 上需 `>=2025.2.0`）和 `torch>=2.1`。

### 导出为 CoreML

```python
import coremltools as ct

from ultralytics.utils.export import torch2coreml

inputs = [ct.TensorType("input", shape=(1, 3, 224, 224))]
ct_model = torch2coreml(model, inputs, im, output_file="resnet18.mlpackage")
```

对于[图像分类](https://www.ultralytics.com/glossary/image-classification)模型，可将类别名称列表传入 `classifier_names`，为 CoreML 模型添加分类头。

需要 `coremltools>=9.0`、`torch>=1.11` 和 `numpy<=2.3.5`。不支持 Windows。

!!! warning "`BlobWriter not loaded` 错误"

    `coremltools>=9.0` 为 macOS 和 Linux 上的 Python 3.10–3.13 提供了 wheel 包。在较新的 Python 版本上，原生 C 扩展无法加载。请使用 Python 3.10–3.13 进行 CoreML 导出。

### 导出为 TensorFlow SavedModel

TF SavedModel 导出以 ONNX 作为中间步骤：

```python
from ultralytics.utils.export import onnx2saved_model, torch2onnx

torch2onnx(model, im, output_file="resnet18.onnx")
keras_model = onnx2saved_model("resnet18.onnx", output_dir="resnet18_saved_model")
```

该函数返回一个 Keras 模型，并在输出目录中生成 TFLite 文件（`.tflite`）：

```
resnet18_saved_model/
├── saved_model.pb
├── variables/
├── resnet18_float32.tflite
├── resnet18_float16.tflite
└── resnet18_int8.tflite
```

依赖要求：

- `tensorflow>=2.0.0,<=2.19.0`
- `onnx2tf>=1.26.3,<1.29.0`
- `tf_keras<=2.19.0`
- `sng4onnx>=1.0.1`
- `onnx_graphsurgeon>=0.3.26`（使用 `--extra-index-url https://pypi.ngc.nvidia.com` 安装）
- macOS 上需 `ai-edge-litert>=1.2.0,<1.4.0`（其他平台 `ai-edge-litert>=1.2.0`）
- `onnxslim>=0.1.71`
- `onnx>=1.12.0,<2.0.0`
- `protobuf>=5`

### 导出为 TensorFlow Frozen Graph

在上面的 SavedModel 导出基础上，将返回的 Keras 模型转换为冻结的 `.pb` 图：

```python
from pathlib import Path

from ultralytics.utils.export import keras2pb

keras2pb(keras_model, output_file=Path("resnet18_saved_model/resnet18.pb"))
```

### 导出为 NCNN

```python
from ultralytics.utils.export import torch2ncnn

torch2ncnn(model, im, output_dir="resnet18_ncnn_model")
```

目录中包含固定名称的 param 和 bin 文件以及一个 Python 封装：

```
resnet18_ncnn_model/
├── model.ncnn.param
├── model.ncnn.bin
└── model_ncnn.py
```

`torch2ncnn()` 首次使用时会检查 `ncnn` 和 `pnnx`。

### 导出为 MNN

MNN 导出需要 ONNX 文件作为输入。先导出为 ONNX，再进行转换：

```python
from ultralytics.utils.export import onnx2mnn, torch2onnx

torch2onnx(model, im, output_file="resnet18.onnx")
onnx2mnn("resnet18.onnx", output_file="resnet18.mnn")
```

支持 `half=True` 启用 FP16 和 `int8=True` 启用 INT8 量化。需要 `MNN>=2.9.6` 和 `torch>=1.10`。

### 导出为 PaddlePaddle

```python
from ultralytics.utils.export import torch2paddle

torch2paddle(model, im, output_dir="resnet18_paddle_model")
```

目录中包含 PaddlePaddle 模型和参数文件：

```
resnet18_paddle_model/
├── model.pdmodel
└── model.pdiparams
```

需要 `x2paddle` 和适合您平台的 PaddlePaddle 发行版：

- CUDA 平台上 `paddlepaddle-gpu>=3.0.0,<3.3.0`
- ARM64 CPU 上 `paddlepaddle==3.0.0`
- 其他 CPU 上 `paddlepaddle>=3.0.0,<3.3.0`

不支持 NVIDIA Jetson。

### 导出为 ExecuTorch

```python
from ultralytics.utils.export import torch2executorch

torch2executorch(model, im, output_dir="resnet18_executorch_model")
```

导出的 `.pte` 文件保存在输出目录中：

```
resnet18_executorch_model/
└── model.pte
```

需要 `torch>=2.9.0` 和匹配的 ExecuTorch 运行时（`pip install executorch`）。运行时用法请参见 [ExecuTorch 集成](../integrations/executorch.md)。

## 验证导出的模型

导出后，在交付前验证与原始 PyTorch 模型的数值一致性。使用 `ultralytics.nn.backends` 中的 `ONNXBackend` 进行快速冒烟测试，比较输出并及早发现 tracing 或量化错误：

```python
import numpy as np
import timm
import torch

from ultralytics.nn.backends import ONNXBackend

model = timm.create_model("resnet18", pretrained=True).eval()
im = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    pytorch_output = model(im).numpy()

onnx_model = ONNXBackend("resnet18.onnx", device=torch.device("cpu"))
onnx_output = onnx_model(im)[0]

diff = np.abs(pytorch_output - onnx_output).max()
print(f"Max difference: {diff:.6f}")  # 应 < 1e-5
```

!!! tip "预期差异"

    对于 FP32 导出，最大绝对差异应在 `1e-5` 以下。更大的差异表明存在不支持的操作、输入形状不正确或模型未处于 eval 模式。FP16 和 INT8 导出的容差较宽松。请使用真实数据而非随机张量进行验证。

对于其他运行时，输入张量名称可能不同。例如，OpenVINO 使用模型的前向参数名称（通用模型通常为 `x`），而 `torch2onnx` 默认使用 `"images"`。

## 已知限制

- **多输入支持不均衡：** `torch2onnx` 和 `torch2openvino` 接受示例张量的元组或列表以支持多输入模型。`torch2torchscript`、`torch2coreml`、`torch2ncnn`、`torch2paddle` 和 `torch2executorch` 假定只有一个输入张量。
- **ExecuTorch 需要 `flatc`：** ExecuTorch 运行时需要 FlatBuffers 编译器。macOS 上使用 `brew install flatbuffers` 安装，Ubuntu 上使用 `apt install flatbuffers-compiler`。
- **不支持通过 Ultralytics 进行推理：** 导出的非 YOLO 模型无法通过 `YOLO()` 加载进行推理。请使用每种格式的原生运行时（[ONNX Runtime](../integrations/onnx.md)、[OpenVINO Runtime](../integrations/openvino.md) 等）。
- **仅限 YOLO 的格式：** [Axelera](../integrations/axelera.md) 和 [Sony IMX500](../integrations/sony-imx500.md) 导出需要 YOLO 特定的模型属性，不适用于通用模型。
- **平台特定格式：** [TensorRT](../integrations/tensorrt.md) 需要 NVIDIA GPU。[RKNN](../integrations/rockchip-rknn.md) 需要 `rknn-toolkit2` SDK（仅限 Linux）。[Edge TPU](../integrations/edge-tpu.md) 需要 `edgetpu_compiler` 二进制文件（仅限 Linux）。

## 常见问题

### 我可以用 Ultralytics 导出哪些模型？

任何 `torch.nn.Module`。包括来自 timm、torchvision 的模型或任何自定义 PyTorch 模型。导出前模型必须处于评估模式（`model.eval()`）。ONNX 和 OpenVINO 还接受示例张量的元组以支持多输入模型。

### 哪些导出格式不需要 GPU？

所有支持的格式（TorchScript、ONNX、OpenVINO、CoreML、TF SavedModel、TF Frozen Graph、NCNN、PaddlePaddle、MNN、ExecuTorch）都可以在 CPU 上导出。导出过程本身不需要 GPU。TensorRT 是唯一需要 NVIDIA GPU 的格式。

### 我需要什么版本的 Ultralytics？

使用 Ultralytics `>=8.4.38`，该版本包含了 `ultralytics.utils.export` 模块和标准化的 `output_file`/`output_dir` 参数。

### 我可以将 torchvision 模型导出为 CoreML 用于 iOS 部署吗？

可以。torchvision 分类器、检测器和分割模型可通过 `torch2coreml` 导出为 `.mlpackage`。对于图像分类模型，可将类别名称列表传入 `classifier_names` 以嵌入分类头。导出需要在 macOS 或 Linux 上运行，CoreML 不支持 Windows。有关 iOS 部署细节，请参见 [CoreML 集成](../integrations/coreml.md)。

### 我可以将导出的模型量化为 INT8 或 FP16 吗？

可以，适用于多种格式。导出为 OpenVINO、CoreML、MNN 或 NCNN 时，传入 `half=True` 启用 FP16 或 `int8=True` 启用 INT8。OpenVINO 的 INT8 还需要额外提供 `calibration_dataset` 参数以进行[训练后量化](https://www.ultralytics.com/glossary/model-quantization)。请参见每种格式的集成页面了解量化权衡。