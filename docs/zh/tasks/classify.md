---
comments: true
description: 掌握使用 YOLO26 进行图像分类。学习如何高效地训练、验证、预测和导出模型。
keywords: YOLO26, 图像分类, AI, 机器学习, 预训练模型, ImageNet, 模型导出, 预测, 训练, 验证
model_name: yolo26n-cls
---

# 图像分类

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/image-classification-examples.avif" alt="YOLO 对物体和场景的图像分类">

[图像分类](https://www.ultralytics.com/glossary/image-classification)是三项任务中最简单的一项，涉及将整张图像归类到一组预定义类别中的某一个。

图像分类器的输出是一个类别标签和一个置信度分数。当你只需要知道图像属于哪个类别，而不需要知道该类别的物体位于何处或其确切形状时，图像分类非常有用。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/5BO0Il_YYAg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>探索 Ultralytics YOLO 任务：使用 Ultralytics 平台进行图像分类
</p>

!!! tip

    YOLO26 分类模型使用 `-cls` 后缀，即 `yolo26n-cls.pt`，并在 [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) 上进行了预训练。

## [模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

此处展示了 YOLO26 预训练分类模型。检测、分割和姿态模型在 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 数据集上预训练，而分类模型在 [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) 数据集上预训练。

[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)在首次使用时从最新的 Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)自动下载。

{% include "macros/yolo-cls-perf.md" %}

- **acc** 值是模型在 [ImageNet](https://www.image-net.org/) 数据集验证集上的准确率。<br>可通过 `yolo val classify data=path/to/ImageNet device=0` 复现
- **速度** 是在 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例上对 ImageNet 验证集图像的平均速度。<br>可通过 `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu` 复现
- **Params** 和 **FLOPs** 值是 `model.fuse()` 后融合模型的值，该操作会合并 Conv 和 BatchNorm 层。预训练检查点保留完整的训练架构，可能显示更高的数值。

## 训练

在 MNIST160 数据集上以图像尺寸 64 训练 YOLO26n-cls 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)。完整参数列表请参见[配置](../usage/cfg.md)页面。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-cls.yaml")  # 从 YAML 构建新模型
        model = YOLO("yolo26n-cls.pt")  # 加载预训练模型（推荐用于训练）
        model = YOLO("yolo26n-cls.yaml").load("yolo26n-cls.pt")  # 从 YAML 构建并迁移权重

        # 训练模型
        results = model.train(data="mnist160", epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        # 从 YAML 构建新模型并从头开始训练
        yolo classify train data=mnist160 model=yolo26n-cls.yaml epochs=100 imgsz=64

        # 从预训练的 *.pt 模型开始训练
        yolo classify train data=mnist160 model=yolo26n-cls.pt epochs=100 imgsz=64

        # 从 YAML 构建新模型，将预训练权重迁移到其中并开始训练
        yolo classify train data=mnist160 model=yolo26n-cls.yaml pretrained=yolo26n-cls.pt epochs=100 imgsz=64
        ```

!!! tip

    Ultralytics YOLO 分类在训练时使用 [`torchvision.transforms.RandomResizedCrop`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomResizedCrop.html)，在验证和推理时使用 [`torchvision.transforms.CenterCrop`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.CenterCrop.html)。
    这些基于裁剪的变换假设输入为正方形，可能会无意中裁剪掉极端宽高比图像中的重要区域，从而可能在训练期间丢失关键的视觉信息。
    为了在保持图像比例的同时保留完整图像，可以考虑使用 [`torchvision.transforms.Resize`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html) 替代裁剪变换。

    你可以通过自定义 `ClassificationDataset` 和 `ClassificationTrainer` 来定制数据增强流水线以实现此目的。


    ```python
    import torch
    import torchvision.transforms as T

    from ultralytics import YOLO
    from ultralytics.data.dataset import ClassificationDataset
    from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator


    class CustomizedDataset(ClassificationDataset):
        """一个用于图像分类的自定义数据集类，具有增强的数据增强变换。"""

        def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
            """初始化一个具有增强数据增强变换的自定义分类数据集。"""
            super().__init__(root, args, augment, prefix)

            # 在此处添加自定义训练变换
            train_transforms = T.Compose(
                [
                    T.Resize((args.imgsz, args.imgsz)),
                    T.RandomHorizontalFlip(p=args.fliplr),
                    T.RandomVerticalFlip(p=args.flipud),
                    T.RandAugment(interpolation=T.InterpolationMode.BILINEAR),
                    T.ColorJitter(brightness=args.hsv_v, contrast=args.hsv_v, saturation=args.hsv_s, hue=args.hsv_h),
                    T.ToTensor(),
                    T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
                    T.RandomErasing(p=args.erasing, inplace=True),
                ]
            )

            # 在此处添加自定义验证变换
            val_transforms = T.Compose(
                [
                    T.Resize((args.imgsz, args.imgsz)),
                    T.ToTensor(),
                    T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
                ]
            )
            self.torch_transforms = train_transforms if augment else val_transforms


    class CustomizedTrainer(ClassificationTrainer):
        """一个用于 YOLO 分类模型的自定义训练器类，具有增强的数据集处理能力。"""

        def build_dataset(self, img_path: str, mode: str = "train", batch=None):
            """为分类训练和训练期间的验证构建自定义数据集。"""
            return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)


    class CustomizedValidator(ClassificationValidator):
        """一个用于 YOLO 分类模型的自定义验证器类，具有增强的数据集处理能力。"""

        def build_dataset(self, img_path: str, mode: str = "train"):
            """为分类独立验证构建自定义数据集。"""
            return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=self.args.split)


    model = YOLO("yolo26n-cls.pt")
    model.train(data="imagenet1000", trainer=CustomizedTrainer, epochs=10, imgsz=224, batch=64)
    model.val(data="imagenet1000", validator=CustomizedValidator, imgsz=224, batch=64)
    ```

### 数据集格式

YOLO 分类数据集格式的详细信息可在[数据集指南](../datasets/classify/index.md)中找到。分类数据集也可以在 [Ultralytics 平台](https://platform.ultralytics.com)上进行管理和标注。

## 验证

在 MNIST160 数据集上验证已训练的 YOLO26n-cls 模型的[准确率](https://www.ultralytics.com/glossary/accuracy)。无需任何参数，因为 `model` 会将其训练 `data` 和参数作为模型属性保留。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-cls.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 验证模型
        metrics = model.val()  # 无需参数，数据集和设置已被记住
        metrics.top1  # top1 准确率
        metrics.top5  # top5 准确率
        ```

    === "CLI"

        ```bash
        yolo classify val model=yolo26n-cls.pt  # 验证官方模型
        yolo classify val model=path/to/best.pt # 验证自定义模型
        ```

!!! tip

    如[训练部分](#训练)所述，你可以通过使用自定义 `ClassificationTrainer` 来处理训练期间的极端宽高比问题。在调用 `val()` 方法时，你需要通过实现自定义 `ClassificationValidator` 来应用相同的方法以获得一致的验证结果。实现细节请参考[训练部分](#训练)中的完整代码示例。

## 预测

使用已训练的 YOLO26n-cls 模型对图像进行预测。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-cls.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 使用模型进行预测
        results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
        ```

    === "CLI"

        ```bash
        yolo classify predict model=yolo26n-cls.pt source='https://ultralytics.com/images/bus.jpg'  # 使用官方模型预测
        yolo classify predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # 使用自定义模型预测
        ```

完整 `predict` 模式详情请参见[预测](../modes/predict.md)页面。

## 导出

将 YOLO26n-cls 模型导出为不同格式，如 ONNX、CoreML 等。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-cls.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义训练的模型

        # 导出模型
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-cls.pt format=onnx  # 导出官方模型
        yolo export model=path/to/best.pt format=onnx # 导出自定义训练的模型
        ```

可用的 YOLO26-cls 导出格式如下表所示。你可以使用 `format` 参数导出为任意格式，即 `format='onnx'` 或 `format='engine'`。你可以直接对导出的模型进行预测或验证，即 `yolo predict model=yolo26n-cls.onnx`。导出完成后会显示你的模型的使用示例。

{% include "macros/export-table.md" %}

完整 `export` 详情请参见[导出](../modes/export.md)页面。

## 常见问题

### YOLO26 在图像分类中的用途是什么？

YOLO26 模型（如 `yolo26n-cls.pt`）专为高效的图像分类而设计。它们为整张图像分配一个类别标签以及一个置信度分数。这对于只需要知道图像属于哪个特定类别，而不需要识别图像中物体的位置或形状的应用场景特别有用。

### 如何训练 YOLO26 模型进行图像分类？

要训练 YOLO26 模型，你可以使用 Python 或 CLI 命令。例如，在 MNIST160 数据集上以图像尺寸 64 训练 `yolo26n-cls` 模型 100 个 epoch：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="mnist160", epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        yolo classify train data=mnist160 model=yolo26n-cls.pt epochs=100 imgsz=64
        ```

更多配置选项请访问[配置](../usage/cfg.md)页面。

### 在哪里可以找到预训练的 YOLO26 分类模型？

预训练的 YOLO26 分类模型可以在[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)部分找到。像 `yolo26n-cls.pt`、`yolo26s-cls.pt`、`yolo26m-cls.pt` 等模型在 [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) 数据集上进行了预训练，可以轻松下载并用于各种图像分类任务。

### 如何将训练好的 YOLO26 模型导出为不同格式？

你可以使用 Python 或 CLI 命令将训练好的 YOLO26 模型导出为各种格式。例如，将模型导出为 ONNX 格式：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-cls.pt")  # 加载训练好的模型

        # 将模型导出为 ONNX
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-cls.pt format=onnx # 将训练好的模型导出为 ONNX 格式
        ```

详细导出选项请参考[导出](../modes/export.md)页面。

### 如何验证训练好的 YOLO26 分类模型？

要在 MNIST160 等数据集上验证训练好的模型的准确率，可以使用以下 Python 或 CLI 命令：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-cls.pt")  # 加载训练好的模型

        # 验证模型
        metrics = model.val()  # 无需参数，使用训练时的数据集和设置
        metrics.top1  # top1 准确率
        metrics.top5  # top5 准确率
        ```

    === "CLI"

        ```bash
        yolo classify val model=yolo26n-cls.pt # 验证训练好的模型
        ```

更多信息请访问[验证](#验证)部分。
