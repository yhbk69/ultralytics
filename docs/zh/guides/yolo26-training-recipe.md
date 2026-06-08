---
comments: true
description: 了解 YOLO26 基础模型在 COCO 上的训练方法，包括优化器设置、数据增强流水线、损失权重，以及针对各模型尺寸的实用微调指导。
keywords: YOLO26, 训练配方, 预训练, 微调, MuSGD, 数据增强, 损失权重, COCO, 模型卡, 超参数, Ultralytics, 目标检测, 深度学习, 数据增强
---

# YOLO26 训练配方

## 简介

本文档记录了用于在 [COCO](../datasets/detect/coco.md) 上生成官方 [YOLO26](../models/yolo26.md) 预训练权重的确切[训练](../modes/train.md)配方。此处展示的每个[超参数](https://www.ultralytics.com/glossary/hyperparameter-tuning)均已嵌入已发布的 `.pt` 权重中，可通过编程方式查看。

了解基础模型的训练方式有助于你在[微调](https://www.ultralytics.com/glossary/fine-tuning)时做出更好的决策：保留哪些[数据增强](./yolo-data-augmentation.md)，调整哪些[损失函数](https://www.ultralytics.com/glossary/loss-function)权重，以及哪些优化器设置最适合你的数据集规模。

!!! tip "本指南面向谁？"

    本指南面向希望了解官方 YOLO26 权重背后训练细节的实践者——不仅仅是架构，还有塑造其性能的[学习率](https://www.ultralytics.com/glossary/learning-rate)调度、数据增强流水线和损失权重。利用这些信息，在用自己的数据微调时做出明智的选择。

## 查看训练参数

每个 Ultralytics 权重文件都存储了生成它时使用的完整训练配置。你可以随时查看这些设置：

!!! example "查看权重文件的训练参数"

    === "Ultralytics API"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        print(model.ckpt["train_args"])
        ```

    === "PyTorch"

        ```python
        import torch

        # 加载任意官方权重
        ckpt = torch.load("yolo26n.pt", map_location="cpu", weights_only=False)

        # 打印所有训练参数
        for k, v in sorted(ckpt["train_args"].items()):
            print(f"{k}: {v}")
        ```

此方法适用于任意 `.pt` 权重文件——无论是官方发布版还是你自己微调的模型。完整的可配置训练参数列表请参见[训练配置参考](../usage/cfg.md)。

## 训练概览

所有 YOLO26 基础模型均在 COCO 上以 **640x640** 分辨率训练，使用 **MuSGD** 优化器，**[批量大小](https://www.ultralytics.com/glossary/batch-size)为 128**。模型从中间预训练权重初始化，并通过进化搜索找到的超参数进行精炼。各模型尺寸的完整训练日志和指标可在 [Ultralytics 平台](https://platform.ultralytics.com/ultralytics/yolo26)上查看：

<iframe
  src="https://platform.ultralytics.com/embed/ultralytics/yolo26"
  scrolling="no"
  width="100%"
  height="290px"
  style="border:none"
></iframe>

所有尺寸的关键设计选择：

- **端到端训练**（`end2end=True`），采用无 NMS 的一对一头部
- **MuSGD 优化器**，将 SGD 与针对卷积权重的 Muon 风格正交化更新结合
- **高强度的 [Mosaic](./yolo-data-augmentation.md#mosaic-mosaic) 增强**（约 0.9-1.0 概率），在最后 10 个 epoch 禁用（`close_mosaic=10`）
- **激进的缩放增强**（0.56-0.95），以处理不同尺寸的目标
- **最小的旋转/剪切**，大多数尺寸保持低几何失真

## 各模型尺寸的超参数

### 优化器和学习率

| 设置            | N       | S       | M       | L       | X       |
| --------------- | ------- | ------- | ------- | ------- | ------- |
| `optimizer`     | MuSGD   | MuSGD   | MuSGD   | MuSGD   | MuSGD   |
| `lr0`           | 0.0054  | 0.00038 | 0.00038 | 0.00038 | 0.00038 |
| `lrf`           | 0.0495  | 0.882   | 0.882   | 0.882   | 0.882   |
| `momentum`      | 0.947   | 0.948   | 0.948   | 0.948   | 0.948   |
| `weight_decay`  | 0.00064 | 0.00027 | 0.00027 | 0.00027 | 0.00027 |
| `warmup_epochs` | 0.98    | 0.99    | 0.99    | 0.99    | 0.99    |
| `epochs`        | 245     | 70      | 80      | 60      | 40      |
| `batch`         | 128     | 128     | 128     | 128     | 128     |
| `imgsz`         | 640     | 640     | 640     | 640     | 640     |

!!! info "学习率策略"

    N 模型使用较高的初始学习率，配合陡峭衰减（`lrf=0.0495`），而 S/M/L/X 模型使用低得多的初始学习率并配合更平缓的调度（`lrf=0.882`）。这反映了小型模型与大型模型不同的收敛动态——小型模型需要更激进的更新才能有效学习。

### 损失权重

| 设置  | N    | S    | M    | L    | X    |
| ----- | ---- | ---- | ---- | ---- | ---- |
| `box` | 5.63 | 9.83 | 9.83 | 9.83 | 9.83 |
| `cls` | 0.56 | 0.65 | 0.65 | 0.65 | 0.65 |
| `dfl` | 9.04 | 0.96 | 0.96 | 0.96 | 0.96 |

N 模型优先考虑 DFL 损失，而 S/M/L/X 模型将重点转移到[边界框](https://www.ultralytics.com/glossary/bounding-box)回归上。分类损失在所有尺寸上相对一致。

### 数据增强流水线

有关每种技术的详细说明，请参阅 [YOLO 数据增强指南](./yolo-data-augmentation.md)。

| 设置                                                               | N     | S     | M     | L     | X     |
| ------------------------------------------------------------------ | ----- | ----- | ----- | ----- | ----- |
| [`mosaic`](./yolo-data-augmentation.md#mosaic-mosaic)              | 0.909 | 0.992 | 0.992 | 0.992 | 0.992 |
| [`mixup`](./yolo-data-augmentation.md#mixup-mixup)                 | 0.012 | 0.05  | 0.427 | 0.427 | 0.427 |
| [`copy_paste`](./yolo-data-augmentation.md#copy-paste-copy_paste)  | 0.075 | 0.404 | 0.304 | 0.404 | 0.404 |
| [`scale`](./yolo-data-augmentation.md#scale-scale)                 | 0.562 | 0.9   | 0.95  | 0.95  | 0.95  |
| [`fliplr`](./yolo-data-augmentation.md#flip-left-right-fliplr)     | 0.606 | 0.304 | 0.304 | 0.304 | 0.304 |
| [`degrees`](./yolo-data-augmentation.md#rotation-degrees)          | 1.11  | ~0    | ~0    | ~0    | ~0    |
| [`shear`](./yolo-data-augmentation.md#shear-shear)                 | 1.46  | ~0    | ~0    | ~0    | ~0    |
| [`translate`](./yolo-data-augmentation.md#translation-translate)   | 0.071 | 0.275 | 0.275 | 0.275 | 0.275 |
| [`hsv_h`](./yolo-data-augmentation.md#hue-adjustment-hsv_h)        | 0.014 | 0.013 | 0.013 | 0.013 | 0.013 |
| [`hsv_s`](./yolo-data-augmentation.md#saturation-adjustment-hsv_s) | 0.645 | 0.353 | 0.353 | 0.353 | 0.353 |
| [`hsv_v`](./yolo-data-augmentation.md#brightness-adjustment-hsv_v) | 0.566 | 0.194 | 0.194 | 0.194 | 0.194 |
| [`bgr`](./yolo-data-augmentation.md#bgr-channel-swap-bgr)          | 0.106 | 0.0   | 0.0   | 0.0   | 0.0   |

较大模型整体使用更激进的数据增强（更高的 [mixup](./yolo-data-augmentation.md#mixup-mixup)、[copy-paste](./yolo-data-augmentation.md#copy-paste-copy_paste) 和 [scale](./yolo-data-augmentation.md#scale-scale)），因为它们具有更大的容量，能从更强的[正则化](https://www.ultralytics.com/glossary/regularization)中受益。N 模型是唯一具有显著[旋转](./yolo-data-augmentation.md#rotation-degrees)、[剪切](./yolo-data-augmentation.md#shear-shear)和 [BGR](./yolo-data-augmentation.md#bgr-channel-swap-bgr) 增强的尺寸。

### 内部训练参数

??? note "高级：内部流水线参数"

    权重文件中还包含内部训练流水线中使用的参数，但这些参数**不**作为 `default.yaml` 中用户可配置的设置暴露：

    | 设置    | 描述                      | N    | S     | M     | L     | X     |
    |--------|---------------------------|------|-------|-------|-------|-------|
    | `muon_w` | MuSGD 中的 Muon 更新权重    | 0.528 | 0.436 | 0.436 | 0.436 | 0.436 |
    | `sgd_w`  | MuSGD 中的 SGD 更新权重     | 0.674 | 0.479 | 0.479 | 0.479 | 0.479 |
    | `cls_w`  | 内部分类权重                | 2.74  | 3.48  | 3.48  | 3.48  | 3.48  |
    | `o2m`    | 一对多头部的损失权重          | 1.0   | 0.705 | 0.705 | 0.705 | 0.705 |
    | `topk`   | Top-k 标签分配              | 8     | 5     | 5     | 5     | 5     |

    这些参数仅为可复现性而记录，微调时无需设置。更多细节请参见[常见问题](#常见问题)。

## 微调指导

在自己的数据集上微调 YOLO26 时，无需完全复现完整的预训练配方。预训练权重已经编码了来自 COCO 训练的数据增强和优化知识。更通用的训练最佳实践请参见[模型训练技巧](./model-training-tips.md)。

### 从简单开始

!!! example "使用默认设置微调"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        results = model.train(data="your-dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo train model=yolo26n.pt data=your-dataset.yaml epochs=100 imgsz=640
        ```

使用默认设置微调是一个强有力的基线。除非有特定原因，否则只调整超参数。

### 何时调整

**小数据集（< 1,000 张图像）：**

- 降低数据增强强度：`mosaic=0.5`，`mixup=0.0`，`copy_paste=0.0`
- 降低学习率：`lr0=0.001`
- 使用更少的 [epoch](https://www.ultralytics.com/glossary/epoch) 并配合耐心值：`epochs=50`，`patience=20`
- 考虑冻结骨干网络层：`freeze=10`

**大数据集（> 50,000 张图像）：**

- 更贴近预训练配方
- 在较长的训练中考虑 `optimizer=MuSGD`
- 增强数据增强：`mosaic=1.0`，`mixup=0.3`，`scale=0.9`

**特定领域的图像**（航拍、医学、水下）：

- 如果垂直方向变化较大，增加 `flipud=0.5`
- 如果目标以任意角度出现，增加 `degrees`
- 如果光照条件与 COCO 差异显著，调整 `hsv_s` 和 `hsv_v`

自动化超参数优化请参见[超参数调优指南](./hyperparameter-tuning.md)。

### 选择模型尺寸

| 模型     | 最适合的场景                           | 批量大小建议                            |
| -------- | -------------------------------------- | --------------------------------------- |
| YOLO26n  | 边缘设备、移动端、CPU 实时推理          | 消费级 GPU 上使用大批量（64-128）        |
| YOLO26s  | 平衡的速度与精度                        | 中等批量（32-64）                        |
| YOLO26m  | 适度计算资源下更高的精度                | 较小批量（16-32）                        |
| YOLO26l  | GPU 可用时的高精度                      | 小批量（8-16）或多 GPU                   |
| YOLO26x  | 最高精度，服务器部署                    | 小批量（4-8）或多 GPU                    |

导出和部署选项请参见[导出指南](../modes/export.md)和[模型部署选项](./model-deployment-options.md)。

## 常见问题

### 如何查看任意权重文件使用的确切超参数？

使用 `torch.load()` 加载权重文件并访问 `train_args` 键，或通过 Ultralytics API 使用 `model.ckpt["train_args"]`。完整示例请参见[查看训练参数](#查看训练参数)。

### 为什么不同模型尺寸的 epoch 数不同？

更大的模型在 COCO 上收敛更快，因为它们具有更大的容量。N 模型需要 245 个 epoch，而 X 模型仅需 40 个。在自己的数据集上微调时，最优 epoch 数取决于数据集的规模和复杂度，而非模型尺寸。使用早停（`patience`）自动找到合适的停止点。

### 微调时应该使用 MuSGD 吗？

当 `optimizer=auto`（默认值）时，Ultralytics 会自动为较长训练（>10,000 次迭代）选择 **MuSGD**，为较短训练选择 **AdamW**。如果愿意，也可以显式设置 `optimizer=MuSGD`。更多优化器选择信息请参见[训练文档](../modes/train.md)。

### 权重文件中的 `muon_w`、`sgd_w`、`cls_w`、`o2m` 和 `topk` 是什么？

这些是生成基础权重时训练流水线中的内部参数。它们为可复现性而存储，但**不**是 `default.yaml` 中用户可配置的设置。微调时无需设置它们。详情请参见[内部训练参数](#内部训练参数)。

### 是否可以从头开始精确复现预训练？

这些权重是使用包含公开代码库中不具备的额外功能（如可配置的 `o2m` 权重和 `cls_w`）的内部训练分支生成的。使用本页面记录的超参数配合公开的 Ultralytics 包可以获得非常接近的结果，但精确复现需要内部分支。
