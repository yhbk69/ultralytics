---
comments: true
description: 学习在 Python 中集成 Ultralytics YOLO 进行目标检测、分割和分类。通过我们的综合指南轻松加载和训练模型，并进行预测。
keywords: YOLO, Python, 目标检测, 分割, 分类, 机器学习, AI, 预训练模型, 训练模型, 进行预测
---

# Python 用法

欢迎来到 Ultralytics YOLO Python 用法文档！本指南旨在帮助你无缝地将 Ultralytics YOLO 集成到你的 Python 项目中，用于[目标检测](https://www.ultralytics.com/glossary/object-detection)、[分割](https://docs.ultralytics.com/tasks/segment)和[分类](https://docs.ultralytics.com/tasks/classify)。在这里，你将学习如何加载和使用预训练模型、训练新模型以及对图像进行预测。易于使用的 Python 接口是任何希望将 YOLO 纳入其 Python 项目的人的宝贵资源，让你能够快速实现高级目标检测功能。让我们开始吧！

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=58"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>掌握 Ultralytics YOLO：Python
</p>

例如，用户只需几行代码就可以加载模型、训练模型、在验证集上评估其性能，甚至[将其导出为 ONNX 格式](../modes/export.md)。

!!! example "Python"

    ```python
    from ultralytics import YOLO

    # 从头创建一个新的 YOLO 模型
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

## 训练

[训练模式](../modes/train.md)用于在自定义数据集上训练 YOLO 模型。在此模式下，模型使用指定的数据集和超参数进行训练。训练过程涉及优化模型的参数，使其能够准确预测图像中对象的类别和位置。

!!! example "训练"

    === "从预训练开始（推荐）"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")  # 传入任何模型类型
        results = model.train(epochs=5)
        ```

    === "从头开始"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.yaml")
        results = model.train(data="coco8.yaml", epochs=5)
        ```

    === "恢复"

        ```python
        model = YOLO("last.pt")
        results = model.train(resume=True)
        ```

[训练示例](../modes/train.md){ .md-button }

## 验证

[验证模式](../modes/val.md)用于在训练后验证 YOLO 模型。在此模式下，模型在验证集上进行评估，以衡量其[准确度](https://www.ultralytics.com/glossary/accuracy)和泛化性能。此模式可用于调优模型的超参数以提高其性能。

!!! example "验证"

    === "训练后验证"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO 模型
        model = YOLO("yolo26n.yaml")

        # 训练模型
        model.train(data="coco8.yaml", epochs=5)

        # 在训练数据上验证
        model.val()
        ```

    === "在另一个数据集上验证"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO 模型
        model = YOLO("yolo26n.yaml")

        # 训练模型
        model.train(data="coco8.yaml", epochs=5)

        # 在单独的数据上验证
        model.val(data="path/to/separate/data.yaml")
        ```

[验证示例](../modes/val.md){ .md-button }

## 预测

[预测模式](../modes/predict.md)用于使用训练好的 YOLO 模型对新图像或视频进行预测。在此模式下，模型从检查点文件加载，用户可以提供图像或视频来执行推理。模型预测输入图像或视频中对象的类别和位置。

!!! example "预测"

    === "从源"

        ```python
        import cv2
        from PIL import Image

        from ultralytics import YOLO

        model = YOLO("model.pt")
        # 接受所有格式 - 图像/目录/路径/URL/视频/PIL/ndarray。0 表示摄像头
        results = model.predict(source="0")
        results = model.predict(source="folder", show=True)  # 显示预测结果。接受所有 YOLO 预测参数

        # 从 PIL
        im1 = Image.open("bus.jpg")
        results = model.predict(source=im1, save=True)  # 保存绘制的图像

        # 从 ndarray
        im2 = cv2.imread("bus.jpg")
        results = model.predict(source=im2, save=True, save_txt=True)  # 将预测保存为标签

        # 从 PIL/ndarray 列表
        results = model.predict(source=[im1, im2])
        ```

    === "结果使用"

        ```python
        # 默认情况下，results 是一个包含所有预测的 Results 对象列表
        # 但要注意，当图像很多时，它可能占用大量内存，
        # 尤其是分割任务。
        # 1. 作为列表返回
        results = model.predict(source="folder")

        # 通过设置 stream=True，results 将是一个对内存更友好的生成器
        # 2. 作为生成器返回
        results = model.predict(source=0, stream=True)

        for result in results:
            # 检测
            result.boxes.xyxy  # xyxy 格式的边界框，(N, 4)
            result.boxes.xywh  # xywh 格式的边界框，(N, 4)
            result.boxes.xyxyn  # xyxy 格式但归一化的边界框，(N, 4)
            result.boxes.xywhn  # xywh 格式但归一化的边界框，(N, 4)
            result.boxes.conf  # 置信度分数，(N, 1)
            result.boxes.cls  # 类别，(N, 1)

            # 分割
            result.masks.data  # 掩码，(N, H, W)
            result.masks.xy  # x,y 分割段（像素），List[segment] * N
            result.masks.xyn  # x,y 分割段（归一化），List[segment] * N

            # 分类
            result.probs  # 类别概率，(num_class, )

        # 默认情况下，每个结果由 torch.Tensor 组成，
        # 你可以轻松使用以下功能：
        result = result.cuda()
        result = result.cpu()
        result = result.to("cpu")
        result = result.numpy()
        ```

[预测示例](../modes/predict.md){ .md-button }

## 导出

[导出模式](../modes/export.md)用于将 YOLO 模型导出为可用于部署的格式。在此模式下，模型被转换为其他软件应用程序或硬件设备可以使用的格式。此模式在将模型部署到生产环境时非常有用。

!!! example "导出"

    === "导出为 ONNX"

        将官方 YOLO 模型导出为 [ONNX](https://www.ultralytics.com/glossary/onnx-open-neural-network-exchange)，支持动态批次大小和图像大小。
        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="onnx", dynamic=True)
        ```

    === "导出为 TensorRT"

        将官方 YOLO 模型导出为 [TensorRT](https://www.ultralytics.com/glossary/tensorrt)，在 `device=0` 上加速 CUDA 设备。
        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="engine", device=0)
        ```

[导出示例](../modes/export.md){ .md-button }

## 跟踪

[跟踪模式](../modes/track.md)用于使用 YOLO 模型实时跟踪对象。在此模式下，模型从检查点文件加载，用户可以提供实时视频流来执行实时对象跟踪。此模式适用于监控系统或[自动驾驶汽车](https://www.ultralytics.com/solutions/ai-in-automotive)等应用。

!!! example "跟踪"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载官方检测模型
        model = YOLO("yolo26n-seg.pt")  # 加载官方分割模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 使用模型进行跟踪
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")
        ```

[跟踪示例](../modes/track.md){ .md-button }

## 基准测试

[基准测试模式](../modes/benchmark.md)用于分析 YOLO 各种导出格式的速度和准确度。基准测试提供有关导出格式大小、`mAP50-95` 指标（用于目标检测和分割）或 `accuracy_top5` 指标（用于分类）以及各种导出格式（如 ONNX、[OpenVINO](https://docs.ultralytics.com/integrations/openvino)、TensorRT 等）每张图像的推理时间（毫秒）的信息。这些信息可以帮助用户根据速度和准确度要求选择适合其特定用例的最佳导出格式。

!!! example "基准测试"

    === "Python"

        在所有导出格式上对官方 YOLO 模型进行基准测试。
        ```python
        from ultralytics.utils.benchmarks import benchmark

        # 基准测试
        benchmark(model="yolo26n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
        ```

[基准测试示例](../modes/benchmark.md){ .md-button }

## 使用训练器

`YOLO` 模型类作为 Trainer 类的高级封装。每个 YOLO 任务都有自己的训练器，继承自 `BaseTrainer`。这种架构允许在[机器学习工作流](https://docs.ultralytics.com/guides/model-training-tips)中实现更大的灵活性和自定义。

!!! tip "检测训练器示例"

    ```python
    from ultralytics.models.yolo.detect import DetectionPredictor, DetectionTrainer, DetectionValidator

    # 训练器
    trainer = DetectionTrainer(overrides={})
    trainer.train()
    trained_model = trainer.best

    # 验证器
    val = DetectionValidator(args=...)
    val(model=trained_model)

    # 预测器
    pred = DetectionPredictor(overrides={})
    pred(source=SOURCE, model=trained_model)

    # 从最后一个权重恢复
    overrides["resume"] = trainer.last
    trainer = DetectionTrainer(overrides=overrides)
    ```

你可以轻松自定义训练器以支持自定义任务或探索研究和开发想法。Ultralytics YOLO 的模块化设计允许你将框架适应你的特定需求，无论你是在研究新颖的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务，还是微调现有模型以获得更好的性能。

[自定义教程](engine.md){ .md-button }

## FAQ

### 如何将 YOLO 集成到我的 Python 项目中进行目标检测？

将 Ultralytics YOLO 集成到你的 Python 项目中非常简单。你可以加载预训练模型或从头训练新模型。以下是如何开始：

```python
from ultralytics import YOLO

# 加载预训练的 YOLO 模型
model = YOLO("yolo26n.pt")

# 对图像进行目标检测
results = model("https://ultralytics.com/images/bus.jpg")

# 可视化结果
for result in results:
    result.show()
```

在我们的[预测模式](../modes/predict.md)章节中查看更多详细示例。

### YOLO 中有哪些不同的模式可用？

Ultralytics YOLO 提供了各种模式来满足不同的[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)工作流。这些包括：

- **[训练](../modes/train.md)**：使用自定义数据集训练模型。
- **[验证](../modes/val.md)**：在验证集上验证模型性能。
- **[预测](../modes/predict.md)**：对新图像或视频流进行预测。
- **[导出](../modes/export.md)**：将模型导出为 ONNX 和 TensorRT 等各种格式。
- **[跟踪](../modes/track.md)**：在视频流中实时跟踪对象。
- **[基准测试](../modes/benchmark.md)**：在不同配置下对模型性能进行基准测试。

每种模式旨在为[模型开发和部署](https://docs.ultralytics.com/guides/model-deployment-options)的不同阶段提供全面的功能。

### 如何使用我的数据集训练自定义 YOLO 模型？

要训练自定义 YOLO 模型，你需要指定数据集和其他[超参数](https://www.ultralytics.com/glossary/hyperparameter-tuning)。以下是一个快速示例：

```python
from ultralytics import YOLO

# 加载 YOLO 模型
model = YOLO("yolo26n.yaml")

# 使用自定义数据集训练模型
model.train(data="path/to/your/dataset.yaml", epochs=10)
```

有关训练的更多细节和示例用法链接，请访问我们的[训练模式](../modes/train.md)页面。

### 如何导出 YOLO 模型以进行部署？

使用 `export` 函数可以轻松地将 YOLO 模型导出为适合部署的格式。例如，你可以将模型导出为 ONNX 格式：

```python
from ultralytics import YOLO

# 加载 YOLO 模型
model = YOLO("yolo26n.pt")

# 将模型导出为 ONNX 格式
model.export(format="onnx")
```

有关各种导出选项，请参见[导出模式](../modes/export.md)文档。

### 是否可以在不同数据集上验证我的 YOLO 模型？

可以，在不同数据集上验证 YOLO 模型是可行的。训练后，你可以使用验证模式来评估性能：

```python
from ultralytics import YOLO

# 加载 YOLO 模型
model = YOLO("yolo26n.yaml")

# 训练模型
model.train(data="coco8.yaml", epochs=5)

# 在不同数据集上验证模型
model.val(data="path/to/separate/data.yaml")
```

查看[验证模式](../modes/val.md)页面以获取详细示例和用法。