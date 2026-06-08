---
comments: true
description: 探索 Ultralytics 回调函数，用于训练、验证、导出和预测。了解如何为你的 ML 模型使用和自定义它们。
keywords: Ultralytics, 回调, 训练, 验证, 导出, 预测, ML 模型, YOLO, Python, 机器学习
---

# 回调函数

Ultralytics 框架支持回调函数（callbacks），作为在 `train`、`val`、`export` 和 `predict` 模式中关键阶段的入口点。每个回调函数接受一个 `Trainer`、`Validator` 或 `Predictor` 对象，具体取决于操作类型。这些对象的所有属性在文档的[参考章节](../reference/cfg/__init__.md)中有详细说明。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ENQXiK7HF5o"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics 回调函数 | 预测、训练、验证和导出回调 | Ultralytics YOLO🚀
</p>

## 示例

### 在预测中返回额外信息

在此示例中，我们演示了如何随每个结果对象返回原始帧：

```python
from ultralytics import YOLO


def on_predict_batch_end(predictor):
    """将预测结果与对应的帧进行组合。"""
    _, image, _, _ = predictor.batch

    # 确保 image 是一个列表
    image = image if isinstance(image, list) else [image]

    # 将预测结果与对应的帧组合
    predictor.results = zip(predictor.results, image)


# 创建 YOLO 模型实例
model = YOLO("yolo26n.pt")

# 将自定义回调添加到模型中
model.add_callback("on_predict_batch_end", on_predict_batch_end)

# 遍历结果和帧
for result, frame in model.predict():  # 或 model.track()
    pass
```

### 使用 `on_model_save` 回调访问模型指标

此示例展示了如何在检查点保存后，使用 `on_model_save` 回调来获取训练详情，例如 best_fitness 分数、total_loss 和其他指标。

```python
from ultralytics import YOLO

# 加载 YOLO 模型
model = YOLO("yolo26n.pt")


def print_checkpoint_metrics(trainer):
    """在每个检查点保存后打印训练器指标和损失详情。"""
    print(
        f"模型详情\n"
        f"最佳适应度: {trainer.best_fitness}, "
        f"损失名称: {trainer.loss_names}, "  # 损失名称列表
        f"指标: {trainer.metrics}, "
        f"总损失: {trainer.tloss}"  # 总损失值
    )


if __name__ == "__main__":
    # 添加 on_model_save 回调。
    model.add_callback("on_model_save", print_checkpoint_metrics)

    # 在自定义数据集上运行模型训练。
    results = model.train(data="coco8.yaml", epochs=3)
```

## 所有回调函数

以下是所有支持的回调函数。更多细节请参阅回调函数[源码](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py)。

### Trainer 回调函数

| 回调函数                   | 描述                                                                        |
| -------------------------- | --------------------------------------------------------------------------- |
| `on_pretrain_routine_start` | 在预训练例程开始时触发，在数据加载和模型设置之前。                           |
| `on_pretrain_routine_end`   | 在预训练例程结束时触发，数据加载和模型设置完成后。                           |
| `on_train_start`            | 当训练开始时触发，在第一个[epoch](https://www.ultralytics.com/glossary/epoch)开始之前。 |
| `on_train_epoch_start`      | 在每个训练 [epoch](https://www.ultralytics.com/glossary/epoch) 开始时触发，在批次迭代开始之前。 |
| `on_train_batch_start`      | 在每个训练批次开始时触发，在前向传播之前。                                   |
| `optimizer_step`            | 在优化器步骤期间触发。保留用于自定义集成；不由默认训练循环调用。             |
| `on_before_zero_grad`       | 在梯度清零之前触发。保留用于自定义集成；不由默认训练循环调用。               |
| `on_train_batch_end`        | 在每个训练批次结束时触发，在反向传播之后。优化器步骤可能因梯度累积而延迟。   |
| `on_train_epoch_end`        | 在每个训练 epoch 结束时触发，所有批次处理完后但在**验证之前**。验证指标和适应度可能尚不可用。 |
| `on_model_save`             | 在模型检查点保存时触发，在验证之后。                                         |
| `on_fit_epoch_end`          | 在每个 fit epoch（训练+验证）结束时触发，在**验证和任何检查点保存之后**。验证指标可用，适应度可用于每个 epoch 的训练调用。此回调在最终最佳模型评估期间也会被调用，此时不会保存检查点，适应度可能不存在。 |
| `on_train_end`              | 当训练过程结束时触发，在最佳模型最终评估之后。                               |
| `on_params_update`          | 当模型参数更新时触发。保留用于自定义集成；不由默认训练循环调用。             |
| `teardown`                  | 当训练过程正在清理时触发。                                                   |

### Validator 回调函数

| 回调函数             | 描述                          |
| -------------------- | ----------------------------- |
| `on_val_start`       | 当验证开始时触发。            |
| `on_val_batch_start` | 在每个验证批次开始时触发。    |
| `on_val_batch_end`   | 在每个验证批次结束时触发。    |
| `on_val_end`         | 当验证结束时触发。            |

### Predictor 回调函数

| 回调函数                     | 描述                              |
| ---------------------------- | --------------------------------- |
| `on_predict_start`           | 当预测过程开始时触发。            |
| `on_predict_batch_start`     | 在每个预测批次开始时触发。        |
| `on_predict_postprocess_end` | 在预测后处理结束时触发。          |
| `on_predict_batch_end`       | 在每个预测批次结束时触发。        |
| `on_predict_end`             | 当预测过程结束时触发。            |

### Exporter 回调函数

| 回调函数          | 描述                          |
| ----------------- | ----------------------------- |
| `on_export_start` | 当导出过程开始时触发。        |
| `on_export_end`   | 当导出过程结束时触发。        |

## FAQ

### 什么是 Ultralytics 回调函数，如何使用它们？

Ultralytics 回调函数是专门的入口点，在模型操作（如训练、验证、导出和预测）的关键阶段被触发。这些回调函数允许在过程的特定点实现自定义功能，从而增强和修改工作流。每个回调函数接受一个 `Trainer`、`Validator` 或 `Predictor` 对象，具体取决于操作类型。这些对象的详细属性请参见[参考章节](../reference/cfg/__init__.md)。

要使用回调函数，先定义一个函数，然后使用 [`model.add_callback()`](../reference/engine/model.md#ultralytics.engine.model.Model.add_callback) 方法将其添加到模型中。以下是在预测期间返回额外信息的示例：

```python
from ultralytics import YOLO


def on_predict_batch_end(predictor):
    """通过将结果与对应帧组合来处理预测批次结束；修改 predictor 的结果。"""
    _, image, _, _ = predictor.batch
    image = image if isinstance(image, list) else [image]
    predictor.results = zip(predictor.results, image)


model = YOLO("yolo26n.pt")
model.add_callback("on_predict_batch_end", on_predict_batch_end)
for result, frame in model.predict():
    pass
```

### 如何使用回调函数自定义 Ultralytics 训练例程？

通过在训练过程的特定阶段注入逻辑来自定义 Ultralytics 训练例程。Ultralytics YOLO 提供了多种训练回调函数，如 `on_train_start`、`on_train_end` 和 `on_train_batch_end`，允许你添加自定义指标、处理或日志记录。

以下是如何在冻结层时使用回调函数冻结 BatchNorm 统计信息：

```python
from ultralytics import YOLO


# 添加回调函数将冻结层置于 eval 模式，以防止 BN 值发生变化
def put_in_eval_mode(trainer):
    n_layers = trainer.args.freeze
    if not isinstance(n_layers, int):
        return

    for i, (name, module) in enumerate(trainer.model.named_modules()):
        if name.endswith("bn") and int(name.split(".")[1]) < n_layers:
            module.eval()
            module.track_running_stats = False


model = YOLO("yolo26n.pt")
model.add_callback("on_train_epoch_start", put_in_eval_mode)
model.train(data="coco.yaml", epochs=10)
```

有关有效使用训练回调函数的更多细节，请参见[训练指南](../modes/train.md)。

### 为什么在 Ultralytics YOLO 验证期间应使用回调函数？

在验证期间使用回调函数可以通过启用自定义处理、日志记录或指标计算来增强模型评估。`on_val_start`、`on_val_batch_end` 和 `on_val_end` 等回调函数提供了注入自定义逻辑的入口点，确保详细全面的验证过程。

例如，绘制所有验证批次而不仅仅是前三个：

```python
import inspect

from ultralytics import YOLO


def plot_samples(validator):
    frame = inspect.currentframe().f_back.f_back
    v = frame.f_locals
    validator.plot_val_samples(v["batch"], v["batch_i"])
    validator.plot_predictions(v["batch"], v["preds"], v["batch_i"])


model = YOLO("yolo26n.pt")
model.add_callback("on_val_batch_end", plot_samples)
model.val(data="coco.yaml")
```

有关将回调函数纳入验证过程的更多见解，请参见[验证指南](../modes/val.md)。

### 如何在 Ultralytics YOLO 中为预测模式附加自定义回调函数？

要为预测模式附加自定义回调函数，定义一个回调函数并将其注册到预测过程中。常见的预测回调函数包括 `on_predict_start`、`on_predict_batch_end` 和 `on_predict_end`。这些允许修改预测输出并集成额外的功能，如数据日志记录或结果转换。

以下是一个示例，其中自定义回调根据特定类别的对象是否存在来保存预测结果：

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

class_id = 2


def save_on_object(predictor):
    r = predictor.results[0]
    if class_id in r.boxes.cls:
        predictor.args.save = True
    else:
        predictor.args.save = False


model.add_callback("on_predict_postprocess_end", save_on_object)
results = model("pedestrians.mp4", stream=True, save=True)

for results in results:
    pass
```

有关更全面的用法，请参见[预测指南](../modes/predict.md)，其中包括详细说明和其他自定义选项。

### 在 Ultralytics YOLO 中使用回调函数有哪些实际示例？

Ultralytics YOLO 支持各种实际的回调函数实现，以增强和自定义训练、验证和预测等不同阶段。一些实际示例包括：

- **记录自定义指标**：在不同阶段记录额外的指标，例如在训练或验证 [epoch](https://www.ultralytics.com/glossary/epoch) 结束时。
- **[数据增强](https://www.ultralytics.com/glossary/data-augmentation)**：在预测或训练批次期间实现自定义数据转换或增强。
- **中间结果**：保存中间结果，如预测或帧，以供进一步分析或可视化。

示例：使用 `on_predict_batch_end` 在预测期间将帧与预测结果组合：

```python
from ultralytics import YOLO


def on_predict_batch_end(predictor):
    """将预测结果与帧组合。"""
    _, image, _, _ = predictor.batch
    image = image if isinstance(image, list) else [image]
    predictor.results = zip(predictor.results, image)


model = YOLO("yolo26n.pt")
model.add_callback("on_predict_batch_end", on_predict_batch_end)
for result, frame in model.predict():
    pass
```

探索[回调函数源码](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py)以获取更多选项和示例。