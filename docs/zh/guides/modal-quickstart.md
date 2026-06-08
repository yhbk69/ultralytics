---
comments: true
description: 学习如何配置 Modal 以在云端运行 Ultralytics YOLO26。跟随本指南轻松实现无服务器 GPU 推理与训练。
keywords: Ultralytics, Modal, YOLO26, 无服务器, 云计算, GPU, 机器学习, 推理, 训练
---

# Modal 快速入门指南（Ultralytics）

本指南全面介绍如何在 [Modal](https://modal.com/) 上运行 [Ultralytics YOLO26](../models/yolo26.md)，涵盖无服务器 GPU 推理和模型训练。

## Modal 是什么？

[Modal](https://modal.com/) 是一个面向 AI 和[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)工作负载的无服务器[云计算](https://www.ultralytics.com/glossary/cloud-computing)平台。它自动处理资源供应、扩缩容和执行——你在本地编写 Python 代码，Modal 在云端使用 GPU 运行。这使得运行 YOLO26 等[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型变得非常简单，无需管理基础设施。

## 你将学到什么

- 配置 Modal 并进行身份验证
- 在 Modal 上运行 YOLO26 推理
- 使用 GPU 加速推理
- 在 Modal 上训练 YOLO26 模型

## 前置条件

- 一个 Modal 账户（可在 [modal.com](https://modal.com/) 免费注册）
- 本地安装 Python 3.9 或更高版本

## 安装

安装 Modal Python 包并进行身份验证：

```bash
pip install modal
```

```bash
modal token new
```

!!! tip "身份验证"

    `modal token new` 命令将打开浏览器窗口以验证你的 Modal 账户。验证完成后，你可以在终端中运行 Modal 命令。

## 运行 YOLO26 推理

创建一个名为 `modal_yolo.py` 的新 Python 文件，内容如下：

```python
"""
Modal + Ultralytics YOLO26 快速入门
运行方式: modal run modal_yolo.py
"""

import modal

app = modal.App("ultralytics-yolo")

image = modal.Image.debian_slim(python_version="3.11").apt_install("libgl1", "libglib2.0-0").pip_install("ultralytics")


@app.function(image=image)
def predict(image_url: str):
    """在图片 URL 上运行 YOLO26 推理。"""
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    results = model(image_url)

    for r in results:
        print(f"检测到 {len(r.boxes)} 个目标:")
        for box in r.boxes:
            print(f"  - {model.names[int(box.cls)]}: {float(box.conf):.2f}")


@app.local_entrypoint()
def main():
    """使用示例图片测试推理。"""
    predict.remote("https://ultralytics.com/images/bus.jpg")
```

运行推理：

```bash
modal run modal_yolo.py
```

预期输出：

```
✓ Initialized. View run at https://modal.com/apps/your-username/main/ap-xxxxxxxx
✓ Created objects.
├── 🔨 Created mount modal_yolo.py
└── 🔨 Created function predict.
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt'...
Downloading https://ultralytics.com/images/bus.jpg to 'bus.jpg'...
image 1/1 /root/bus.jpg: 640x480 4 persons, 1 bus, 377.8ms
Speed: 5.8ms preprocess, 377.8ms inference, 0.3ms postprocess per image at shape (1, 3, 640, 480)

Detected 5 objects:
  - bus: 0.92
  - person: 0.91
  - person: 0.91
  - person: 0.87
  - person: 0.53
✓ App completed.
```

你可以在 Modal 控制台中监控函数执行情况：

<p align="center">
  <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@fda2504f1e5dab2f437097ac7b99a7da984e4243/docs/modal-dashboard-function-calls.avif" alt="Modal 控制台函数调用">
</p>

## 使用 GPU 加速推理

通过指定 `gpu` 参数为函数添加 GPU：

```python
@app.function(image=image, gpu="T4")  # 可选: "T4", "A10G", "A100", "H100"
def predict_gpu(image_url: str):
    """在 GPU 上运行 YOLO26 推理。"""
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    results = model(image_url)
    print(results[0].boxes)
```

| GPU  | 显存    | 最佳用途        |
| ---- | ------- | --------------- |
| T4   | 16 GB   | 推理、小模型训练 |
| A10G | 24 GB   | 中等规模训练     |
| A100 | 40 GB   | 大规模训练       |
| H100 | 80 GB   | 极致性能         |

## 在 Modal 上训练 YOLO26

训练时需使用 GPU 和 Modal [Volumes](https://modal.com/docs/guide/volumes)（持久化存储）。创建一个名为 `train_yolo.py` 的新 Python 文件：

```python
import modal

app = modal.App("ultralytics-training")

volume = modal.Volume.from_name("yolo-training-vol", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").apt_install("libgl1", "libglib2.0-0").pip_install("ultralytics")


@app.function(image=image, gpu="T4", timeout=3600, volumes={"/data": volume})
def train():
    """在 Modal 上训练 YOLO26 模型。"""
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    model.train(data="coco8.yaml", epochs=3, imgsz=640, project="/data/runs")


@app.local_entrypoint()
def main():
    train.remote()
```

运行训练：

```bash
modal run train_yolo.py
```

!!! tip "Volume 持久化"

    Modal Volumes 在函数运行之间持久化数据。训练好的权重保存在 `/data/runs/detect/train/weights/`。

恭喜！你已成功在 Modal 上配置好 Ultralytics YOLO26。进一步学习：

- 探索 [Ultralytics YOLO26 文档](../models/yolo26.md) 了解高级功能
- 学习如何使用自己的数据集[训练自定义模型](../modes/train.md)
- 访问 [Modal 文档](https://modal.com/docs) 了解平台高级功能

## 常见问题

### 如何为我的 YOLO26 工作负载选择合适的 GPU？

对于推理任务，NVIDIA T4（16 GB）通常足够且经济实惠。对于训练或 YOLO26x 等大型模型，建议使用 A10G 或 A100 GPU。

### 在 Modal 上运行 YOLO26 的费用是多少？

Modal 采用按秒计费。大致费率：CPU ~$0.05/小时，T4 ~$0.59/小时，A10G ~$1.10/小时，A100 ~$2.10/小时。请查看 [Modal 定价](https://modal.com/pricing)了解最新费率。

### 可以使用自己训练的自定义 YOLO 模型吗？

可以！从 Modal Volume 加载自定义模型：

```python
model = YOLO("/data/my_custom_model.pt")
```

有关训练自定义模型的更多信息，请参阅[训练指南](../modes/train.md)。