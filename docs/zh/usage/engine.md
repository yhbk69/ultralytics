---
comments: true
description: 学习为特定任务自定义 Ultralytics YOLO 训练器。带有 Python 示例的分步说明，以实现最佳模型性能。
keywords: Ultralytics, YOLO, 训练器自定义, Python, 机器学习, AI, 模型训练, DetectionTrainer, 自定义模型
---

# 高级自定义

Ultralytics YOLO 的命令行和 Python 接口都是构建在基础引擎执行器之上的高级抽象。本指南重点介绍 `Trainer` 引擎，说明如何根据你的特定需求对其进行自定义。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=104"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>掌握 Ultralytics YOLO：高级自定义
</p>

!!! tip

    有关常见训练器自定义的实用示例——自定义指标、类别加权损失、模型保存、骨干网络冻结和逐层学习率——请参见[自定义训练器](../guides/custom-trainer.md)指南。

## BaseTrainer

`BaseTrainer` 类提供了一个通用的训练例程，可适应各种任务。通过覆盖特定函数或操作来自定义它，同时遵守所需的格式。例如，通过覆盖以下函数来集成你自己的自定义模型和数据加载器：

- `get_model(cfg, weights)`：构建要训练的模型。
- `get_dataloader()`：构建数据加载器。

有关更多细节和源代码，请参见 [`BaseTrainer` 参考](../reference/engine/trainer.md)。

## DetectionTrainer

以下是如何使用和自定义 Ultralytics YOLO `DetectionTrainer`：

```python
from ultralytics.models.yolo.detect import DetectionTrainer

trainer = DetectionTrainer(overrides={...})
trainer.train()
trained_model = trainer.best  # 获取最佳模型
```

### 自定义 DetectionTrainer

要训练一个不直接支持的自定义检测模型，重载现有的 `get_model` 功能：

```python
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """根据配置和权重文件加载自定义检测模型。"""
        ...


trainer = CustomTrainer(overrides={...})
trainer.train()
```

通过修改[损失函数](https://www.ultralytics.com/glossary/loss-function)或添加[回调函数](callbacks.md)来进一步自定义训练器，例如每 10 个 [epoch](https://www.ultralytics.com/glossary/epoch) 将模型上传到 Google Drive。以下是一个示例：

```python
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel


class MyCustomModel(DetectionModel):
    def init_criterion(self):
        """初始化损失函数并添加一个回调，每 10 个 epoch 将模型上传到 Google Drive。"""
        ...


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """返回一个使用指定配置和权重配置的自定义检测模型实例。"""
        return MyCustomModel(...)


# 回调函数用于上传模型权重
def log_model(trainer):
    """记录训练器使用的最后一个模型权重路径。"""
    last_weight_path = trainer.last
    print(last_weight_path)


trainer = CustomTrainer(overrides={...})
trainer.add_callback("on_train_epoch_end", log_model)  # 添加到现有回调中
trainer.train()
```

有关回调触发事件和入口点的更多信息，请参见[回调指南](../usage/callbacks.md)。

## 其他引擎组件

以类似的方式自定义其他组件，如 `Validators` 和 `Predictors`。有关更多信息，请参见 [Validators](../reference/engine/validator.md) 和 [Predictors](../reference/engine/predictor.md) 的文档。

## 使用 YOLO 与自定义训练器

`YOLO` 模型类为 Trainer 类提供了高级封装。你可以利用这种架构在机器学习工作流中获得更大的灵活性：

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


# 创建自定义训练器
class MyCustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """自定义代码实现。"""
        ...


# 初始化 YOLO 模型
model = YOLO("yolo26n.pt")

# 使用自定义训练器训练
results = model.train(trainer=MyCustomTrainer, data="coco8.yaml", epochs=3)
```

这种方法允许你在保持 YOLO 接口简单性的同时，自定义底层训练过程以满足你的特定需求。

## FAQ

### 如何为特定任务自定义 Ultralytics YOLO DetectionTrainer？

通过覆盖其方法来使 `DetectionTrainer` 适应你的自定义模型和数据加载器，从而为特定任务进行自定义。首先继承 `DetectionTrainer` 并重新定义 `get_model` 等方法以实现自定义功能。以下是一个示例：

```python
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """根据配置和权重文件加载自定义检测模型。"""
        ...


trainer = CustomTrainer(overrides={...})
trainer.train()
trained_model = trainer.best  # 获取最佳模型
```

有关进一步的自定义，例如更改[损失函数](https://www.ultralytics.com/glossary/loss-function)或添加[回调](https://www.ultralytics.com/glossary/callback)，请参见[回调指南](../usage/callbacks.md)。

### Ultralytics YOLO 中 BaseTrainer 的关键组件是什么？

`BaseTrainer` 作为训练例程的基础，可通过覆盖其通用方法进行自定义以适应各种任务。关键组件包括：

- `get_model(cfg, weights)`：构建要训练的模型。
- `get_dataloader()`：构建数据加载器。
- `preprocess_batch()`：在模型前向传播之前处理批次预处理。
- `set_model_attributes()`：根据数据集信息设置模型属性。
- `get_validator()`：返回用于模型评估的验证器。

有关自定义和源代码的更多细节，请参见 [`BaseTrainer` 参考](../reference/engine/trainer.md)。

### 如何向 Ultralytics YOLO DetectionTrainer 添加回调函数？

添加回调函数以监控和修改 `DetectionTrainer` 中的训练过程。以下是添加回调函数在每个训练 [epoch](https://www.ultralytics.com/glossary/epoch) 后记录模型权重的方法：

```python
from ultralytics.models.yolo.detect import DetectionTrainer


# 上传模型权重的回调
def log_model(trainer):
    """记录训练器使用的最后一个模型权重路径。"""
    last_weight_path = trainer.last
    print(last_weight_path)


trainer = DetectionTrainer(overrides={...})
trainer.add_callback("on_train_epoch_end", log_model)  # 添加到现有回调中
trainer.train()
```

有关回调事件和入口点的更多详情，请参见[回调指南](../usage/callbacks.md)。

### 为什么应使用 Ultralytics YOLO 进行模型训练？

Ultralytics YOLO 提供了强大的引擎执行器之上的高级抽象，使其非常适合快速开发和自定义。主要优点包括：

- **易用性**：命令行和 Python 接口简化了复杂任务。
- **性能**：针对实时[目标检测](https://www.ultralytics.com/glossary/object-detection)和各种视觉 AI 应用进行了优化。
- **可自定义性**：易于扩展以支持自定义模型、[损失函数](https://www.ultralytics.com/glossary/loss-function)和数据加载器。
- **模块化**：组件可以独立修改，而不会影响整个流程。
- **集成**：与 ML 生态系统中的流行框架和工具无缝协作。

通过浏览 [Ultralytics YOLO](https://www.ultralytics.com/yolo) 主页面了解更多关于 YOLO 功能的信息。

### 是否可以将 Ultralytics YOLO DetectionTrainer 用于非标准模型？

可以，`DetectionTrainer` 高度灵活且可自定义，适用于非标准模型。继承 `DetectionTrainer` 并重载方法以支持你特定模型的需求。以下是一个简单示例：

```python
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """加载自定义检测模型。"""
        ...


trainer = CustomDetectionTrainer(overrides={...})
trainer.train()
```

有关全面的说明和示例，请查看 [`DetectionTrainer` 参考](../reference/models/yolo/detect/train.md)。