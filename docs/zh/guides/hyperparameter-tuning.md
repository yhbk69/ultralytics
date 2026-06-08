---
comments: true
description: 掌握 Ultralytics YOLO 的超参数调优，通过本综合指南优化模型性能。立即提升您的机器学习模型！
keywords: Ultralytics YOLO, 超参数调优, 机器学习, 模型优化, 遗传算法, 学习率, 批次大小, 训练轮数
---

# Ultralytics YOLO [超参数调优](https://www.ultralytics.com/glossary/hyperparameter-tuning) 指南

## 简介

超参数调优并非一次性设置，而是一个迭代过程，旨在优化[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型的性能指标，如准确率、精确率和召回率。在 Ultralytics YOLO 中，这些超参数的范围从学习率到架构细节，例如层数或使用的激活函数类型。[Ultralytics 平台](https://platform.ultralytics.com)也支持[云端训练](../platform/train/cloud-training.md)，可配置超参数并实时跟踪指标。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/j0MOGKBqx7E"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何调优超参数以获得更好的模型性能 🚀
</p>

### 什么是超参数？

超参数是算法的高层结构设置。它们在训练阶段之前设定，并在训练过程中保持不变。以下是 Ultralytics YOLO 中一些常用的调优超参数：

- **学习率** `lr0`：决定每次迭代中向[损失函数](https://www.ultralytics.com/glossary/loss-function)最小值移动的步长。
- **[批次大小](https://www.ultralytics.com/glossary/batch-size)** `batch`：一次前向传播中同时处理的图像数量。
- **[训练轮数](https://www.ultralytics.com/glossary/epoch)** `epochs`：一个 epoch 是对所有训练样本完成一次完整的前向和反向传播。
- **架构细节**：如通道数、层数、激活函数类型等。

<p align="center">
  <img width="640" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/hyperparameter-tuning-visual.avif" alt="超参数优化搜索空间可视化">
</p>

有关 YOLO26 中使用的完整数据增强超参数列表，请参阅[配置页面](../usage/cfg.md#augmentation-settings)。

### 遗传进化与变异

Ultralytics YOLO 使用[遗传算法](https://en.wikipedia.org/wiki/Genetic_algorithm)来优化超参数。遗传算法的灵感来源于自然选择和遗传学机制。

- **交叉**：每次迭代使用 BLX-α 交叉和适应度加权父代选择，组合来自迄今为止最多九个最高适应度配置的基因。
- **变异**：重组后的候选方案随后通过对每个超参数应用对数正态乘法因子（每个参数概率为 0.5）进行扰动。变异强度 sigma 在前 300 次迭代中从 0.2 线性衰减到 0.1，因此算法在早期广泛探索，在收敛时逐步细化。第 1 次迭代没有父代可供交叉，使用默认训练超参数作为基线。

## 超参数调优准备

在开始调优过程之前，重要的是：

1. **确定评估指标**：确定用于评估模型性能的指标。可以是 AP50、F1-score 或其他指标。
2. **设定调优预算**：明确您愿意分配多少计算资源。超参数调优可能计算量很大。

## 涉及步骤

### 初始化超参数

从一组合理的初始超参数开始。这可以是 Ultralytics YOLO 设置的默认超参数，也可以是基于您的领域知识或先前实验的结果。

### 变异超参数

使用 `_mutate` 方法基于现有集合生成一组新的超参数。[Tuner 类](https://docs.ultralytics.com/reference/engine/tuner)会自动处理此过程。

### 训练模型

使用变异后的超参数集合进行训练。然后使用您选择的指标评估训练性能。

### 评估模型

使用 AP50、F1-score 或自定义指标来评估模型性能。[评估过程](https://docs.ultralytics.com/modes/val)有助于确定当前超参数是否优于之前的超参数。

### 记录结果

记录性能指标和相应的超参数以供将来参考至关重要。Ultralytics YOLO 会自动以 NDJSON 格式保存这些结果。

### 重复

重复此过程，直到达到设定的迭代次数或性能指标令人满意。每次迭代都建立在先前运行中获得的知识之上。

## 默认搜索空间说明

下表列出了 YOLO26 中超参数调优的默认搜索空间参数。每个参数都有一个由元组 `(min, max)` 定义的特定值范围。

| 参数 | 类型 | 值范围 | 说明 |
| ----------------- | ------- | -------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `lr0` | `float` | `(1e-5, 1e-2)` | 训练开始时的初始学习率。较低的值提供更稳定的训练但收敛较慢 |
| `lrf` | `float` | `(0.01, 1.0)` | 最终学习率因子，作为 lr0 的分数。控制训练期间学习率下降的程度 |
| `momentum` | `float` | `(0.7, 0.98)` | SGD 动量因子。较高的值有助于保持一致的梯度方向，可加速收敛 |
| `weight_decay` | `float` | `(0.0, 0.001)` | L2 正则化因子，用于防止过拟合。较大的值施加更强的正则化 |
| `warmup_epochs` | `float` | `(0.0, 5.0)` | 线性学习率预热的 epoch 数。有助于防止早期训练不稳定 |
| `warmup_momentum` | `float` | `(0.0, 0.95)` | 预热阶段的初始动量。逐渐增加到最终动量值 |
| `box` | `float` | `(1.0, 20.0)` | 总损失函数中边界框损失的权重。平衡框回归与分类 |
| `cls` | `float` | `(0.1, 4.0)` | 总损失函数中分类损失的权重。较高的值强调正确的类别预测 |
| `cls_pw` | `float` | `(0.0, 1.0)` | 处理类别不平衡的类别加权幂。较高的值增加稀有类别的权重 |
| `dfl` | `float` | `(0.4, 12.0)` | 总损失函数中 DFL（分布焦点损失）的权重。较高的值强调精确的边界框定位 |
| `hsv_h` | `float` | `(0.0, 0.1)` | HSV 色彩空间中随机色调增强范围。帮助模型泛化到不同颜色变化 |
| `hsv_s` | `float` | `(0.0, 0.9)` | HSV 空间中随机饱和度增强范围。模拟不同的光照条件 |
| `hsv_v` | `float` | `(0.0, 0.9)` | 随机明度（亮度）增强范围。帮助模型处理不同的曝光水平 |
| `degrees` | `float` | `(0.0, 45.0)` | 最大旋转增强角度（度）。帮助模型对物体方向具有不变性 |
| `translate` | `float` | `(0.0, 0.9)` | 最大平移增强，以图像尺寸的分数表示。提高对物体位置的鲁棒性 |
| `scale` | `float` | `(0.0, 0.95)` | 随机缩放增强范围。帮助模型检测不同大小的物体 |
| `shear` | `float` | `(0.0, 10.0)` | 最大剪切增强角度（度）。为训练图像添加类似透视的变形 |
| `perspective` | `float` | `(0.0, 0.001)` | 随机透视增强范围。模拟不同的观察角度 |
| `flipud` | `float` | `(0.0, 1.0)` | 训练期间垂直翻转图像的概率。适用于俯视/航拍图像 |
| `fliplr` | `float` | `(0.0, 1.0)` | 水平翻转图像的概率。帮助模型对物体方向具有不变性 |
| `bgr` | `float` | `(0.0, 1.0)` | 使用 BGR 增强的概率，即交换颜色通道。有助于颜色不变性 |
| `mosaic` | `float` | `(0.0, 1.0)` | 使用马赛克增强的概率，将 4 张图像组合在一起。特别适用于小物体检测 |
| `mixup` | `float` | `(0.0, 1.0)` | 使用 mixup 增强的概率，混合两张图像。可提高模型鲁棒性 |
| `cutmix` | `float` | `(0.0, 1.0)` | 使用 cutmix 增强的概率。在保持局部特征的同时组合图像区域 |
| `copy_paste` | `float` | `(0.0, 1.0)` | 使用复制粘贴增强的概率。有助于提高实例分割性能 |
| `close_mosaic` | `float` | `(0.0, 10.0)` | 在最后 N 个 epoch 中禁用马赛克增强，以在训练完成前稳定训练 |

## 自定义搜索空间示例

以下是如何定义搜索空间并使用 `model.tune()` 方法利用 `Tuner` 类在 COCO8 上对 YOLO26n 进行 30 个 epoch 的超参数调优，使用 AdamW 优化器，并跳过绘图、检查点和除最后一个 epoch 外的验证以加快调优速度。

!!! warning

    此示例仅用于**演示**。从短期或小规模调优运行中得出的超参数很少对实际训练是最优的。在实践中，调优应在与完整训练相似的设置下进行——包括可比的数据集、epoch 和数据增强——以确保结果可靠且可迁移。快速调优可能会使参数偏向更快的收敛或短期验证收益，而这些收益并不具有泛化性。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 初始化 YOLO 模型
        model = YOLO("yolo26n.pt")

        # 定义搜索空间
        search_space = {
            "lr0": (1e-5, 1e-2),
            "degrees": (0.0, 45.0),
        }

        # 在 COCO8 上调优超参数，训练 30 个 epoch
        model.tune(
            data="coco8.yaml",
            epochs=30,
            iterations=300,
            optimizer="AdamW",
            space=search_space,
            plots=False,
            save=False,
            val=False,
        )
        ```

## 恢复中断的超参数调优会话

您可以通过传递 `resume=True` 来恢复中断的超参数调优会话。您可以选择传递 `runs/{task}` 下使用的目录 `name` 来恢复。否则，它将恢复最后一个中断的会话。您还需要提供所有先前的训练参数，包括 `data`、`epochs`、`iterations` 和 `space`。

!!! example "在 `model.tune()` 中使用 `resume=True`"

    ```python
    from ultralytics import YOLO

    # 定义 YOLO 模型
    model = YOLO("yolo26n.pt")

    # 定义搜索空间
    search_space = {
        "lr0": (1e-5, 1e-2),
        "degrees": (0.0, 45.0),
    }

    # 恢复之前的运行
    results = model.tune(data="coco8.yaml", epochs=50, iterations=300, space=search_space, resume=True)

    # 恢复名为 'tune_exp' 的调优运行
    results = model.tune(data="coco8.yaml", epochs=50, iterations=300, space=search_space, name="tune_exp", resume=True)
    ```

## 结果

成功完成超参数调优过程后，您将获得若干文件和目录，其中包含了调优结果。以下逐一说明：

### 文件结构

以下是结果目录结构的示例。训练目录如 `train1/` 包含单个调优迭代，即使用一组超参数训练的一个模型。`tune/` 目录包含所有单个模型训练的调优结果：

```plaintext
runs/
└── detect/
    ├── train1/
    ├── train2/
    ├── ...
    └── tune/
        ├── best_hyperparameters.yaml
        ├── tune_fitness.png
        ├── tune_results.ndjson
        ├── tune_scatter_plots.png
        └── weights/
            ├── last.pt
            └── best.pt
```

### 文件说明

#### best_hyperparameters.yaml

此 YAML 文件包含调优过程中找到的最佳性能超参数。您可以使用此文件以这些优化设置初始化未来的训练。

- **格式**：YAML
- **用途**：超参数结果
- **示例**：

    ```yaml
    # 558/900 次迭代完成 ✅ (45536.81s)
    # 结果已保存至 /usr/src/ultralytics/runs/detect/tune
    # 在第 498 次迭代观察到最佳适应度=0.64297
    # 最佳适应度指标为 {'metrics/precision(B)': 0.87247, 'metrics/recall(B)': 0.71387, 'metrics/mAP50(B)': 0.79106, 'metrics/mAP50-95(B)': 0.62651, 'val/box_loss': 2.79884, 'val/cls_loss': 2.72386, 'val/dfl_loss': 0.68503, 'fitness': 0.64297}
    # 最佳适应度模型为 /usr/src/ultralytics/runs/detect/train498
    # 最佳适应度超参数如下所示。

    lr0: 0.00269
    lrf: 0.00288
    momentum: 0.73375
    weight_decay: 0.00015
    warmup_epochs: 1.22935
    warmup_momentum: 0.1525
    box: 18.27875
    cls: 1.32899
    dfl: 0.56016
    hsv_h: 0.01148
    hsv_s: 0.53554
    hsv_v: 0.13636
    degrees: 0.0
    translate: 0.12431
    scale: 0.07643
    shear: 0.0
    perspective: 0.0
    flipud: 0.0
    fliplr: 0.08631
    mosaic: 0.42551
    mixup: 0.0
    copy_paste: 0.0
    ```

#### tune_fitness.png

这是一个显示适应度与迭代次数关系的图表。它帮助您可视化遗传算法随时间推移的表现。

- **格式**：PNG
- **用途**：性能可视化

<p align="center">
  <img width="640" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/best-fitness.avif" alt="超参数调优适应度与迭代关系图">
</p>

图表包含：

- **每个数据集每次迭代一个标记点**，因此单数据集运行每次迭代显示一个点，多数据集运行每次迭代每个数据集显示一个点。
- **一条虚线"平滑均值"线**，通过对每次迭代的顶层适应度值进行高斯平滑（`sigma=3`）计算得出。

#### tune_results.ndjson

一个 NDJSON 文件，包含每次调优迭代的详细结果。每行是一个 JSON 对象，包含聚合适应度、调优后的超参数和每个数据集的指标。单数据集和多数据集调优使用相同的文件格式。

- **格式**：NDJSON
- **用途**：每次迭代结果跟踪。
- **示例**：

以下为便于阅读而展示的格式化示例。在实际的 `.ndjson` 文件中，每个对象存储在一行中。

```json
{
    "iteration": 1,
    "fitness": 0.48628,
    "hyperparameters": {
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005
    },
    "datasets": {
        "coco8": {
            "metrics/precision(B)": 0.65666,
            "metrics/recall(B)": 0.85,
            "metrics/mAP50(B)": 0.85086,
            "metrics/mAP50-95(B)": 0.64104,
            "val/box_loss": 1.57958,
            "val/cls_loss": 1.04986,
            "val/dfl_loss": 1.32641,
            "fitness": 0.64104
        },
        "coco8-grayscale": {
            "metrics/precision(B)": 0.6582,
            "metrics/recall(B)": 0.51667,
            "metrics/mAP50(B)": 0.59106,
            "metrics/mAP50-95(B)": 0.33152,
            "val/box_loss": 1.95424,
            "val/cls_loss": 1.64059,
            "val/dfl_loss": 1.70226,
            "fitness": 0.33152
        }
    },
    "save_dirs": {
        "coco8": "runs/detect/coco8",
        "coco8-grayscale": "runs/detect/coco8-grayscale"
    }
}
```

顶层 `fitness` 是每个数据集 `fitness` 值的算术平均值。对于单数据集调优，`datasets` 字典中有一个条目，其 `fitness` 等于顶层 `fitness`。每次完成的迭代记录一个 JSON 对象。实际的 `save_dirs` 路径是绝对路径；为便于阅读，上文中进行了缩写。

#### tune_scatter_plots.png

此文件包含从 `tune_results.ndjson` 生成的散点图，帮助您可视化不同超参数与性能指标之间的关系。默认值为 0 的超参数（例如下面的 `degrees` 和 `shear`）可能只能从其初始种子缓慢演化，因为乘法变异因子从接近零的值开始几乎没有扩展空间。

- **格式**：PNG
- **用途**：探索性数据分析

<p align="center">
  <img width="1000" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/tune-scatter-plots.avif" alt="超参数调优结果散点图分析">
</p>

#### weights/

此目录包含超参数调优过程中保存的最后一次和最佳迭代的 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 模型。

- **`last.pt`**：last.pt 是训练最后一个 epoch 的权重。
- **`best.pt`**：best.pt 是获得最佳适应度分数的迭代的权重。

利用这些结果，您可以为未来的模型训练和分析做出更明智的决策。请随时查阅这些产物，以了解您的模型表现如何以及如何进一步改进。

## 结论

Ultralytics YOLO 中的超参数调优过程简化而强大，这得益于其基于遗传算法的方法，结合了 BLX-α 交叉和对数正态变异。遵循本指南中概述的步骤将帮助您系统地调优模型以获得更好的性能。

### 延伸阅读

1. [维基百科中的超参数优化](https://en.wikipedia.org/wiki/Hyperparameter_optimization)
2. [YOLOv5 超参数进化指南](../yolov5/tutorials/hyperparameter_evolution.md)
3. [使用 Ray Tune 和 YOLO26 进行高效超参数调优](../integrations/ray-tune.md)

如需更深入的了解，您可以探索 [`Tuner` 类](https://docs.ultralytics.com/reference/engine/tuner)的源代码和配套文档。如果您有任何问题、功能请求或需要进一步帮助，请随时通过 [GitHub](https://github.com/ultralytics/ultralytics/issues/new/choose) 或 [Discord](https://discord.com/invite/ultralytics) 联系我们。

## 常见问题

### 如何在超参数调优期间优化 Ultralytics YOLO 的[学习率](https://www.ultralytics.com/glossary/learning-rate)？

要优化 Ultralytics YOLO 的学习率，首先使用 `lr0` 参数设置初始学习率。常见值范围从 `0.001` 到 `0.01`。在超参数调优过程中，此值将被变异以找到最优设置。您可以使用 `model.tune()` 方法自动化此过程。例如：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 初始化 YOLO 模型
        model = YOLO("yolo26n.pt")

        # 在 COCO8 上调优超参数，训练 30 个 epoch
        model.tune(data="coco8.yaml", epochs=30, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)
        ```

有关更多详细信息，请查看 [Ultralytics YOLO 配置页面](../usage/cfg.md#augmentation-settings)。

### 在 YOLO26 中使用遗传算法进行超参数调优有哪些好处？

Ultralytics YOLO26 中的遗传算法提供了一种探索超参数空间的稳健方法，从而实现高度优化的模型性能。主要好处包括：

- **高效搜索**：BLX-α 交叉组合来自最高适应度父代的基因，而对数正态变异对结果进行扰动以发现新的候选方案。
- **避免局部最小值**：通过引入随机性，有助于避免局部最小值，确保更好的全局优化。
- **性能指标**：它们基于特定任务的适应度分数（检测任务为 mAP50-95）进行自适应。

要了解遗传算法如何优化超参数，请查看[超参数进化指南](../yolov5/tutorials/hyperparameter_evolution.md)。

### Ultralytics YOLO 的超参数调优过程需要多长时间？

Ultralytics YOLO 的超参数调优所需时间很大程度上取决于多个因素，例如数据集的大小、模型架构的复杂性、迭代次数以及可用的计算资源。例如，在 COCO8 这样的数据集上调优 YOLO26n 训练 30 个 epoch 可能需要数小时到数天，具体取决于硬件。

要有效管理调优时间，请事先定义明确的调优预算（[内部章节链接](#超参数调优准备)）。这有助于平衡资源分配和优化目标。

### 在 YOLO 超参数调优期间应使用哪些指标来评估模型性能？

在 YOLO 超参数调优期间评估模型性能时，您可以使用几个关键指标：

- **AP50**：IoU 阈值为 0.50 时的平均精度。
- **F1-Score**：精确率和召回率的调和平均值。
- **精确率和召回率**：指示模型在识别真正例与假正例和假负例方面的[准确率](https://www.ultralytics.com/glossary/accuracy)的单独指标。

这些指标帮助您了解模型性能的不同方面。请参阅 [Ultralytics YOLO 性能指标](../guides/yolo-performance-metrics.md)指南获取全面概述。

### 我可以使用 Ray Tune 对 YOLO26 进行高级超参数优化吗？

可以，Ultralytics YOLO26 集成了 [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) 以进行高级超参数优化。Ray Tune 提供复杂的搜索算法，如贝叶斯优化和 Hyperband，以及并行执行能力以加速调优过程。

要将 Ray Tune 与 YOLO26 一起使用，只需在 `model.tune()` 方法调用中设置 `use_ray=True` 参数。有关更多详细信息和示例，请查看 [Ray Tune 集成指南](../integrations/ray-tune.md)。
