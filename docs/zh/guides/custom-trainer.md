---
comments: true
description: 学习如何通过自定义指标、类别加权损失、自定义模型保存、主干网络冻结、逐层学习率、SyncBatchNorm 和梯度裁剪来自定义 Ultralytics YOLO 训练器。
keywords: Ultralytics, YOLO, 自定义训练器, DetectionTrainer, BaseTrainer, 自定义指标, F1 分数, 类别权重, 主干网络冻结, 逐层学习率, SyncBatchNorm, 梯度裁剪, 多 GPU 训练, 微调, 迁移学习
---

# 自定义训练器

Ultralytics 训练流程围绕 `BaseTrainer` 和特定任务训练器（如 `DetectionTrainer`）构建。这些类开箱即用地处理训练循环、验证、检查点和日志记录。当您需要更多控制时（如跟踪自定义指标、调整损失权重或实现学习率调度），可以子类化训练器并覆盖特定方法。

本指南将介绍七种常见的自定义方法：

1. 在每个 [epoch](https://www.ultralytics.com/glossary/epoch) 结束时[记录自定义指标（F1 分数）](#记录自定义指标)
2. [添加类别权重](#添加类别权重)以处理类别不平衡
3. [根据不同的指标保存最佳模型](#根据自定义指标保存最佳模型)
4. 在前 N 个 epoch [冻结主干网络](#冻结和解冻主干网络)，然后解冻
5. [指定逐层学习率](#逐层学习率)
6. 为多 GPU 训练[跨 GPU 同步 BatchNorm](#为多-gpu-训练同步-batchnorm)
7. 为稳定性调优[配置梯度裁剪](#可配置的梯度裁剪)

!!! tip "先决条件"

    在阅读本指南之前，请确保您熟悉 [训练 YOLO 模型](../modes/train.md)的基础知识以及 [高级自定义](../usage/engine.md)页面，该页面涵盖了 `BaseTrainer` 架构。

## 自定义训练器的工作原理

`YOLO` 模型类的 `train()` 方法接受 `trainer` 参数。这允许您传递自己的训练器类来扩展默认行为：

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomTrainer(DetectionTrainer):
    """一个自定义训练器，扩展 DetectionTrainer 并添加额外功能。"""

    pass  # 在此添加您的自定义内容


model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=10, trainer=CustomTrainer)
```

您的自定义训练器继承了 `DetectionTrainer` 的所有功能，因此您只需覆盖要自定义的特定方法。

## 记录自定义指标

[验证](../modes/val.md)步骤计算[精度](https://www.ultralytics.com/glossary/precision)、[召回率](https://www.ultralytics.com/glossary/recall)和[mAP](https://www.ultralytics.com/glossary/mean-average-precision-map)。如果您需要额外的指标（如每类的 [F1 分数](https://www.ultralytics.com/glossary/f1-score)），请覆盖 `validate()`：

```python
import numpy as np

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER


class MetricsTrainer(DetectionTrainer):
    """自定义训练器，在每个 epoch 结束时计算并记录 F1 分数。"""

    def validate(self):
        """运行验证并计算每类 F1 分数。"""
        metrics, fitness = super().validate()
        if metrics is None:
            return metrics, fitness

        if hasattr(self.validator, "metrics") and hasattr(self.validator.metrics, "box"):
            box = self.validator.metrics.box
            f1_per_class = box.f1
            class_indices = box.ap_class_index
            names = self.validator.names

            valid_f1 = f1_per_class[f1_per_class > 0]
            mean_f1 = np.mean(valid_f1) if len(valid_f1) > 0 else 0.0

            LOGGER.info(f"平均 F1 分数: {mean_f1:.4f}")
            per_class_str = [
                f"{names[i]}: {f1_per_class[j]:.3f}" for j, i in enumerate(class_indices) if f1_per_class[j] > 0
            ]
            LOGGER.info(f"每类 F1: {per_class_str}")

        return metrics, fitness


model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=5, trainer=MetricsTrainer)
```

这会在每次验证运行后记录所有类别的平均 F1 分数和每类细分。

!!! note "可用指标"

    验证器通过 `self.validator.metrics.box` 提供对许多指标的访问：

    | 属性 | 描述 |
    |---|---|
    | `f1` | 每类 F1 分数 |
    | `image_metrics` | 每张图像的指标字典，包含精度、召回率、F1、TP、FP 和 FN |
    | `p` | 每类精度 |
    | `r` | 每类召回率 |
    | `ap50` | IoU 0.5 时的每类 AP |
    | `ap` | IoU 0.5:0.95 时的每类 AP |
    | `mp`, `mr` | 平均精度和召回率 |
    | `map50`, `map` | 平均 AP 指标 |

## 添加类别权重

如果您的数据集存在类别不平衡（例如，制造检测中的罕见缺陷），您可以在[损失函数](https://www.ultralytics.com/glossary/loss-function)中为代表性不足的类别增加权重。这使得模型对稀有类别上的错误分类施加更重的惩罚。

要自定义损失，请子类化损失类、模型和训练器：

```python
import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss


class WeightedDetectionLoss(v8DetectionLoss):
    """应用类别权重的检测损失，用于 BCE 分类损失。"""

    def __init__(self, model, class_weights=None, tal_topk=10, tal_topk2=None):
        """使用可选的每类权重初始化 BCE 损失。"""
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        if class_weights is not None:
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=class_weights.to(self.device),
                reduction="none",
            )


class WeightedE2ELoss(E2ELoss):
    """YOLO26 的带类别权重的 E2E 损失。"""

    def __init__(self, model, class_weights=None):
        """使用加权检测损失初始化 E2E 损失。"""

        def weighted_loss_fn(model, tal_topk=10, tal_topk2=None):
            return WeightedDetectionLoss(model, class_weights=class_weights, tal_topk=tal_topk, tal_topk2=tal_topk2)

        super().__init__(model, loss_fn=weighted_loss_fn)


class WeightedDetectionModel(DetectionModel):
    """使用类别加权损失的检测模型。"""

    def init_criterion(self):
        """使用每类权重初始化加权损失准则。"""
        class_weights = torch.ones(self.nc)
        class_weights[0] = 2.0  # 增加类别 0 的权重
        class_weights[1] = 3.0  # 增加稀有类别 1 的权重
        return WeightedE2ELoss(self, class_weights=class_weights)


class WeightedTrainer(DetectionTrainer):
    """返回 WeightedDetectionModel 的训练器。"""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回一个 WeightedDetectionModel。"""
        model = WeightedDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model


model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=10, trainer=WeightedTrainer)
```

!!! tip "从数据集中计算权重"

    您可以根据数据集标签分布自动计算类别权重。一种常见的方法是逆频率加权：

    ```python
    import numpy as np

    # class_counts: 每类的实例数
    class_counts = np.array([5000, 200, 3000])
    # 逆频率：稀有类别获得更高权重
    class_weights = max(class_counts) / class_counts
    # 结果: [1.0, 25.0, 1.67]
    ```

## 根据自定义指标保存最佳模型

训练器根据适应度（fitness）保存 `best.pt`，默认适应度为 `0.9 × mAP@0.5:0.95 + 0.1 × mAP@0.5`。要使用不同的指标（如 `mAP@0.5` 或召回率），请覆盖 `validate()` 并将您选择的指标作为适应度值返回。内置的 `save_model()` 将自动使用它：

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomSaveTrainer(DetectionTrainer):
    """根据 mAP@0.5 而非默认适应度保存最佳模型的训练器。"""

    def validate(self):
        """覆盖适应度，使用 mAP@0.5 进行最佳模型选择。"""
        metrics, fitness = super().validate()
        if metrics:
            fitness = metrics.get("metrics/mAP50(B)", fitness)
            if self.best_fitness is None or fitness > self.best_fitness:
                self.best_fitness = fitness
        return metrics, fitness


model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=20, trainer=CustomSaveTrainer)
```

!!! note "可用指标"

    验证后 `self.metrics` 中可用的常见指标包括：

    | 键 | 描述 |
    |---|---|
    | `metrics/precision(B)` | 精度 |
    | `metrics/recall(B)` | 召回率 |
    | `metrics/mAP50(B)` | IoU 0.5 时的 mAP |
    | `metrics/mAP50-95(B)` | IoU 0.5:0.95 时的 mAP |

## 冻结和解冻主干网络

[迁移学习](https://www.ultralytics.com/glossary/transfer-learning)工作流程通常受益于在前 N 个 epoch 冻结预训练的主干网络，允许检测头适应，然后再[微调](https://www.ultralytics.com/glossary/fine-tuning)整个网络。Ultralytics 提供了一个 `freeze` 参数，用于在训练开始时冻结层，您可以使用[回调](../usage/callbacks.md)在 N 个 epoch 后解冻它们：

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER

FREEZE_EPOCHS = 5


def unfreeze_backbone(trainer):
    """在 FREEZE_EPOCHS 后解冻所有层的回调。"""
    if trainer.epoch == FREEZE_EPOCHS:
        LOGGER.info(f"Epoch {trainer.epoch}: 解冻所有层以进行微调")
        for name, param in trainer.model.named_parameters():
            if not param.requires_grad:
                param.requires_grad = True
                LOGGER.info(f"  已解冻: {name}")
        trainer.freeze_layer_names = [".dfl"]


class FreezingTrainer(DetectionTrainer):
    """在前 N 个 epoch 冻结主干网络的训练器。"""

    def __init__(self, *args, **kwargs):
        """初始化并注册解冻回调。"""
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", unfreeze_backbone)


model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=20, freeze=10, trainer=FreezingTrainer)
```

`freeze=10` 参数在训练开始时冻结前 10 层（通常是 YOLO 架构中的主干网络）。`on_train_epoch_start` 回调在每个 epoch 开始时触发，并在冻结期结束后解冻所有参数。

!!! tip "选择冻结内容"

    - `freeze=10` 冻结前 10 层（通常是 YOLO 架构中的主干网络）
    - `freeze=[0, 1, 2, 3]` 按索引冻结特定层
    - 较高的 `FREEZE_EPOCHS` 值让检测头在主干网络改变之前有更多时间适应

## 逐层学习率

网络的不同部分可以受益于不同的[学习率](https://www.ultralytics.com/glossary/learning-rate)。一种常见策略是对预训练主干网络使用较低的学习率以保留已学习的特征，同时允许检测头以较高的学习率更快地适应：

```python
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model


class PerLayerLRTrainer(DetectionTrainer):
    """主干网络和检测头使用不同学习率的训练器。"""

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """构建优化器，为主干网络和检测头设置不同的学习率。"""
        backbone_params = []
        head_params = []

        for k, v in unwrap_model(model).named_parameters():
            if not v.requires_grad:
                continue
            is_backbone = any(k.startswith(f"model.{i}.") for i in range(10))
            if is_backbone:
                backbone_params.append(v)
            else:
                head_params.append(v)

        backbone_lr = lr * 0.1

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr, "weight_decay": decay},
                {"params": head_params, "lr": lr, "weight_decay": decay},
            ],
        )

        LOGGER.info(
            f"逐层学习率优化器: 主干网络 ({len(backbone_params)} 个参数, lr={backbone_lr}) "
            f"| 检测头 ({len(head_params)} 个参数, lr={lr})"
        )
        return optimizer


model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=20, trainer=PerLayerLRTrainer)
```

### RT-DETR 变体

对于 RT-DETR，模式相同，但有两个改进。主干网络长度从 `model.yaml["backbone"]` 读取，因此相同的训练器可以在 RT-DETR 变体（RT-DETR-L、RT-DETR-X、ResNet-50/101 主干网络）之间工作，而无需硬编码层数。参数还在每个部分内分为权重、BatchNorm 和偏置组，以便从 BatchNorm 参数和偏置中排除权重衰减，这与默认训练器的策略相匹配。这对于 RT-DETR 微调特别有用，其中解码器头通常是随机初始化的，而主干网络携带预训练特征，这些特征受益于较低的学习率：

```python
import torch
from torch import nn

from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import unwrap_model


class RTDETRBackboneLRTrainer(RTDETRTrainer):
    """主干网络参数使用较低学习率的 RT-DETR 训练器。"""

    backbone_lr_ratio = 0.1  # 主干网络学习率占检测头学习率的比例

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """构建一个 AdamW 优化器，包含六个参数组：检测头和主干网络 × {权重, bn, 偏置}。"""
        # 解析优化器名称；"auto" 映射到具有 RT-DETR 风格默认值的 AdamW
        canonical = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "auto"}
        name = {x.lower(): x for x in canonical}.get(name.lower(), name)
        if name == "auto":
            name, lr, momentum = "AdamW", 1e-4, 0.9
        self.args.warmup_bias_lr = 0.0  # RT-DETR 从 0 开始预热偏置，不同于 YOLO 的 0.1
        if name not in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            raise NotImplementedError(f"此训练器仅支持 AdamW 系列优化器；收到 {name}")

        # 从 model.yaml 识别主干网络参数，并将每个参数路由到（部分, 类型）组
        unwrapped = unwrap_model(model)
        backbone_len = len(unwrapped.yaml["backbone"])
        norm_types = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        groups = {f"{s}_{k}": [] for s in ("head", "backbone") for k in ("weight", "bn", "bias")}

        for module_name, module in unwrapped.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                parts = fullname.split(".")
                section = (
                    "backbone"
                    if len(parts) > 1 and parts[0] == "model" and parts[1].isdigit() and int(parts[1]) < backbone_len
                    else "head"
                )
                if "bias" in param_name:
                    kind = "bias"
                elif isinstance(module, norm_types) or "logit_scale" in fullname:
                    kind = "bn"
                else:
                    kind = "weight"
                groups[f"{section}_{kind}"].append(param)

        # 构建优化器，每组具有不同的学习率和权重衰减；主干网络组使用 lr * backbone_lr_ratio
        backbone_lr = lr * self.backbone_lr_ratio
        param_groups = [
            {"params": groups["head_weight"], "lr": lr, "weight_decay": decay, "param_group": "weight"},
            {"params": groups["head_bn"], "lr": lr, "weight_decay": 0.0, "param_group": "bn"},
            {"params": groups["head_bias"], "lr": lr, "weight_decay": 0.0, "param_group": "bias"},
            {"params": groups["backbone_weight"], "lr": backbone_lr, "weight_decay": decay, "param_group": "weight"},
            {"params": groups["backbone_bn"], "lr": backbone_lr, "weight_decay": 0.0, "param_group": "bn"},
            {"params": groups["backbone_bias"], "lr": backbone_lr, "weight_decay": 0.0, "param_group": "bias"},
        ]
        param_groups = [pg for pg in param_groups if pg["params"]]  # 删除空组
        optimizer = getattr(torch.optim, name)(param_groups, betas=(momentum, 0.999))

        LOGGER.info(
            f"{colorstr('optimizer:')} {name}(lr={lr}, backbone_lr={backbone_lr}) 参数组\n"
            f"  检测头:     {len(groups['head_bn'])} bn, {len(groups['head_weight'])} weight(decay={decay}), "
            f"{len(groups['head_bias'])} bias (lr={lr})\n"
            f"  主干网络: {len(groups['backbone_bn'])} bn, {len(groups['backbone_weight'])} weight(decay={decay}), "
            f"{len(groups['backbone_bias'])} bias (lr={backbone_lr})"
        )
        return optimizer


model = RTDETR("rtdetr-l.pt")
model.train(data="coco8.yaml", epochs=20, trainer=RTDETRBackboneLRTrainer)
```

!!! tip "选择 `backbone_lr_ratio`"

    常见的起点是 `backbone_lr_ratio = 0.1`，与原始使用 HGNetV2 主干网络的 RT-DETR 设置相匹配。文献建议根据主干网络大小和预训练数据规模按比例调整该比例：在非常大的数据集上预训练的大型主干网络（例如，使用 DINO、CLIP 或 MAE 在数亿张图像上训练的 ViT-L/H）通常使用较小的比例，如 `0.01` 或更低，以保留已学习良好的特征；而预训练较轻的较小主干网络可以容忍较大的比例，如 `0.5` 或更高。

!!! note "学习率调度器"

    内置的学习率调度器（`cosine` 或 `linear`）仍然应用于每组基础学习率之上。主干网络和检测头学习率将遵循相同的衰减计划，在整个训练过程中保持它们之间的比例。

!!! tip "组合技术"

    这些自定义方法可以通过覆盖多个方法和根据需要添加回调来组合到一个训练器类中。

## 为多 GPU 训练同步 BatchNorm

当使用 DistributedDataParallel 在多个 GPU 上进行训练时，默认的 `BatchNorm2d` 层在每个 GPU 上独立计算统计信息。对于 RT-DETR 微调和其他使用较小每 GPU 批量大小的配方，每 GPU 批量统计信息可能噪声较大。PyTorch 的 `SyncBatchNorm` 在所有 rank 之间同步均值和方差，以获得单个全局批量统计信息，这通常可以提高收敛性，但代价是较小的 GPU 间通信开销。

转换必须在模型位于 GPU 之后、DDP 包装它之前进行。最干净的钩子是 `set_model_attributes()`，`BaseTrainer` 恰好在该时间窗口调用它：

```python
from torch import nn

from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer


class SyncBNTrainer(RTDETRTrainer):
    """将 BatchNorm 转换为 SyncBatchNorm 以进行多 GPU 训练的 RT-DETR 训练器。"""

    def set_model_attributes(self):
        """运行父类设置，然后在多 GPU 训练时将 BN 转换为 SyncBatchNorm。"""
        super().set_model_attributes()
        if self.world_size > 1:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)


model = RTDETR("rtdetr-l.pt")
model.train(data="coco8.yaml", epochs=20, device=[0, 1], trainer=SyncBNTrainer)
```

`world_size > 1` 保护确保训练器在单 GPU 运行中也可以安全使用；在单 GPU 上，转换被跳过，训练继续使用常规的 `BatchNorm2d`。相同的模式适用于 YOLO，只需将父类切换为 `DetectionTrainer`。

!!! tip "何时使用 SyncBatchNorm"

    | 场景                                       | 推荐           |
    | ---------------------------------------------- | ------------------------ |
    | 多 GPU 训练，每 GPU 批量较小（≤ 16） | 启用                   |
    | 多 GPU 训练，每 GPU 批量较大（≥ 32） | 可选；收益较小  |
    | 单 GPU 训练                            | 不适用（跳过） |

## 可配置的梯度裁剪

默认训练器在 `optimizer_step()` 中将梯度裁剪到 `max_norm=10.0`，这是一个宽松的值，针对 YOLO 模型调整，其中梯度很少超过该值。DETR 系列检测器（RT-DETR、DEIM、DINO）通常使用更严格的值，如 `0.1`，以稳定解码器的交叉注意力层，其中梯度幅度可能会激增。要覆盖裁剪值，请子类化训练器并覆盖 `optimizer_step()`：

```python
import torch

from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer


class CustomClipTrainer(RTDETRTrainer):
    """具有可配置梯度裁剪的 RT-DETR 训练器。"""

    clip_grad_norm = 0.1  # 最大梯度范数；设置为 0 以禁用裁剪

    def optimizer_step(self):
        """运行优化器步骤，使用可配置的梯度范数裁剪。"""
        self.scaler.unscale_(self.optimizer)
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)


model = RTDETR("rtdetr-l.pt")
model.train(data="coco8.yaml", epochs=20, trainer=CustomClipTrainer)
```

相同的训练器适用于 YOLO，只需将父类切换为 `DetectionTrainer`（`from ultralytics.models.yolo.detect import DetectionTrainer`）并使用 `YOLO("yolo26n.pt")` 加载 YOLO 检查点。`optimizer_step` 主体保持不变。

!!! tip "典型的 `clip_grad_norm` 值"

    | 架构系列          | 典型的 `max_norm` |
    | ---------------------------- | ------------------ |
    | RT-DETR / DEIM / DETR 系列 | `0.1`              |
    | YOLO（Ultralytics 默认）   | `10.0`             |
    | 禁用裁剪             | `0`                |

## 常见问题解答

### 如何将自定义训练器传递给 YOLO？

将您的自定义训练器类（不是实例）传递给 `model.train()` 中的 `trainer` 参数：

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", trainer=MyCustomTrainer)
```

`YOLO` 类在内部处理训练器实例化。有关训练器架构的更多详细信息，请参阅 [高级自定义](../usage/engine.md)页面。

### 我可以覆盖哪些 BaseTrainer 方法？

可用于自定义的关键方法：

| 方法               | 目的                           |
| -------------------- | --------------------------------- |
| `validate()`         | 运行验证并返回指标 |
| `build_optimizer()`  | 构建优化器           |
| `save_model()`       | 保存训练检查点         |
| `get_model()`        | 返回模型实例         |
| `get_validator()`    | 返回验证器实例     |
| `get_dataloader()`   | 构建数据加载器              |
| `preprocess_batch()` | 预处理输入批次            |
| `label_loss_items()` | 格式化损失项以进行日志记录     |

有关完整的 API 参考，请参阅 [`BaseTrainer` 文档](../reference/engine/trainer.md)。

### 我可以使用回调而不是子类化训练器吗？

是的，对于更简单的自定义，[回调](../usage/callbacks.md)通常就足够了。可用的回调事件包括 `on_train_start`、`on_train_epoch_start`、`on_train_epoch_end`、`on_fit_epoch_end` 和 `on_model_save`。这些允许您挂钩到训练循环中，而无需子类化。上面的主干网络冻结示例演示了这种方法。

### 如何在不子类化模型的情况下自定义损失函数？

如果您的更改更简单（例如调整损失增益），可以直接修改[超参数](https://www.ultralytics.com/glossary/hyperparameter-tuning)：

```python
model.train(data="coco8.yaml", box=10.0, cls=1.5, dfl=2.0)
```

对于损失的结构性更改（例如添加类别权重），您需要子类化损失和模型，如[类别权重部分](#添加类别权重)所示。