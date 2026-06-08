---
comments: true
description: 探索 Ultralytics 针对检测、分割、分类等视觉任务的多样化数据集。利用高质量标注数据提升您的项目。
keywords: Ultralytics, 数据集, 计算机视觉, 目标检测, 实例分割, 姿态估计, 图像分类, 多目标跟踪
---

# 数据集概述

Ultralytics 支持多种数据集，以促进计算机视觉任务的开展，例如检测、[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)、姿态估计、分类和多目标跟踪。以下是主要的 Ultralytics 数据集列表，以及每个计算机视觉任务及其对应数据集的摘要。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/YDXKa1EljmU"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> Ultralytics 数据集概述
</p>

## [目标检测](detect/index.md)

[边界框](https://www.ultralytics.com/glossary/bounding-box)目标检测是一种计算机视觉技术，通过在图像中每个物体周围绘制边界框来检测和定位物体。

- [African-wildlife](detect/african-wildlife.md)：一个包含非洲野生动物图像的数据集，包括水牛、大象、犀牛和斑马。
- [Argoverse](detect/argoverse.md)：一个包含来自城市场景的 3D 跟踪和运动预测数据的数据集，具有丰富的标注信息。
- [Brain-tumor](detect/brain-tumor.md)：一个用于检测脑肿瘤的数据集，包含 MRI 或 CT 扫描图像，并提供肿瘤存在、位置和特征等详细信息。
- [COCO](detect/coco.md)：Common Objects in Context (COCO) 是一个大规模的目标检测、分割和描述数据集，包含 80 个物体类别。
- [COCO8](detect/coco8.md)：COCO 训练集和验证集前 4 张图像的一个较小子集，适用于快速测试。
- [COCO8-Grayscale](detect/coco8-grayscale.md)：COCO8 的灰度版本，通过将 RGB 转换为灰度创建，适用于单通道模型评估。
- [COCO8-Multispectral](detect/coco8-multispectral.md)：COCO8 的 10 通道多光谱版本，通过对 RGB 波长进行插值创建，适用于光谱感知模型评估。
- [COCO128](detect/coco128.md)：COCO 训练集和验证集前 128 张图像的一个较小子集，适用于测试。
- [Construction-PPE](detect/construction-ppe.md)：一个建筑工地图像数据集，标注了关键安全装备（如安全帽、背心、手套、靴子和护目镜），以及缺失装备的标签，支持开发用于合规和工人保护的 AI 模型。
- [Global Wheat 2020](detect/globalwheat2020.md)：一个包含小麦穗图像的数据集，用于 2020 年全球小麦挑战赛。
- [HomeObjects-3K](detect/homeobjects-3k.md)：一个标注的室内场景数据集，包含 12 种常见家居物品，非常适合在智能家居系统、机器人和增强现实中开发和测试计算机视觉模型。
- [KITTI](detect/kitti.md) 新：一个著名的自动驾驶数据集，包含立体视觉、LiDAR 和 GPS/IMU 输入，用于各种道路场景中的 2D 目标检测。
- [LVIS](detect/lvis.md)：一个大规模的目标检测、分割和描述数据集，包含 1203 个物体类别。
- [Medical-pills](detect/medical-pills.md)：一个包含标记药丸图像的数据集，旨在辅助药品质量控制、分拣以及确保符合行业标准等任务。
- [Objects365](detect/objects365.md)：一个高质量的大规模目标检测数据集，包含 365 个物体类别和超过 60 万张标注图像。
- [OpenImagesV7](detect/open-images-v7.md)：一个由 Google 提供的综合数据集，包含 170 万张训练图像和 4.2 万张验证图像。
- [RF100](detect/roboflow-100.md)：一个多样化的目标检测基准，包含 100 个数据集，涵盖七个图像领域，用于全面的模型评估。
- [Signature](detect/signature.md)：一个包含各种带有签名标注的文档图像的数据集，支持文档验证和欺诈检测研究。
- [SKU-110K](detect/sku-110k.md)：一个包含零售环境中密集目标检测的数据集，拥有超过 1.1 万张图像和 170 万个边界框。
- [VisDrone](detect/visdrone.md)：一个包含无人机拍摄图像的目标检测和多目标跟踪数据的数据集，拥有超过 1 万张图像和视频序列。
- [VOC](detect/voc.md)：Pascal Visual Object Classes (VOC) 数据集，用于目标检测和分割，包含 20 个物体类别和超过 1.1 万张图像。
- [xView](detect/xview.md)：一个用于俯视图目标检测的数据集，包含 60 个物体类别和超过 100 万个标注目标。

## [实例分割](segment/index.md)

实例分割是一种计算机视觉技术，涉及在像素级别识别和定位图像中的物体。与仅对每个像素进行分类的语义分割不同，[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)能够区分同一类的不同实例。

- [Carparts-seg](segment/carparts-seg.md)：专门用于识别车辆部件的数据集，满足设计、制造和研究需求。适用于目标检测和分割任务。
- [COCO](segment/coco.md)：一个大规模数据集，专为目标检测、分割和描述任务设计，包含超过 20 万张标注图像。
- [COCO8-seg](segment/coco8-seg.md)：一个用于实例分割任务的较小数据集，包含 8 张 COCO 图像的子集及其分割标注。
- [COCO128-seg](segment/coco128-seg.md)：一个用于实例分割任务的较小数据集，包含 128 张 COCO 图像的子集及其分割标注。
- [Crack-seg](segment/crack-seg.md)：专门用于检测道路和墙壁裂缝的数据集，适用于目标检测和分割任务。
- [Package-seg](segment/package-seg.md)：专门用于识别仓库或工业场景中包裹的数据集，适用于目标检测和分割应用。

## [姿态估计](pose/index.md)

姿态估计是一种用于确定物体相对于相机或世界坐标系姿态的技术。这涉及识别物体（尤其是人类或动物）上的关键点或关节。

- [COCO](pose/coco.md)：一个包含人体姿态标注的大规模数据集，专为姿态估计任务设计。
- [COCO8-pose](pose/coco8-pose.md)：一个用于姿态估计任务的较小数据集，包含 8 张 COCO 图像的子集及其人体姿态标注。
- [Dog-pose](pose/dog-pose.md)：一个全面的数据集，包含约 6,000 张以狗为主题的图像，每只狗标注了 24 个关键点，专为姿态估计任务设计。
- [Hand-Keypoints](pose/hand-keypoints.md)：一个简洁的数据集，包含超过 26,000 张以人手为中心的图像，每只手标注了 21 个关键点，专为姿态估计任务设计。
- [Tiger-pose](pose/tiger-pose.md)：一个紧凑的数据集，包含 263 张以老虎为主题的图像，每只老虎标注了 12 个关键点，用于姿态估计任务。

## [分类](classify/index.md)

[图像分类](https://www.ultralytics.com/glossary/image-classification)是一种计算机视觉任务，涉及根据图像的视觉内容将其归类为一个或多个预定义的类别。

- [Caltech 101](classify/caltech101.md)：一个包含 101 个物体类别图像的数据集，用于图像分类任务。
- [Caltech 256](classify/caltech256.md)：Caltech 101 的扩展版本，包含 256 个物体类别和更具挑战性的图像。
- [CIFAR-10](classify/cifar10.md)：一个包含 6 万张 32x32 彩色图像的数据集，分为 10 个类别，每个类别 6000 张图像。
- [CIFAR-100](classify/cifar100.md)：CIFAR-10 的扩展版本，包含 100 个物体类别，每个类别 600 张图像。
- [Fashion-MNIST](classify/fashion-mnist.md)：一个包含 7 万张灰度图像的数据集，涵盖 10 个时尚类别，用于图像分类任务。
- [ImageNet](classify/imagenet.md)：一个大规模的目标检测和图像分类数据集，包含超过 1400 万张图像和 2 万个类别。
- [ImageNet-10](classify/imagenet10.md)：ImageNet 的一个较小子集，包含 10 个类别，用于更快的实验和测试。
- [Imagenette](classify/imagenette.md)：ImageNet 的一个较小子集，包含 10 个易于区分的类别，用于更快速的训练和测试。
- [Imagewoof](classify/imagewoof.md)：ImageNet 中一个更具挑战性的子集，包含 10 个狗品种类别，用于图像分类任务。
- [MNIST](classify/mnist.md)：一个包含 7 万张手写数字灰度图像的数据集，用于图像分类任务。
- [MNIST160](classify/mnist.md)：MNIST 数据集中每个类别的前 8 张图像，共包含 160 张图像。

## [旋转边界框 (OBB)](obb/index.md)

旋转边界框 (OBB) 是计算机视觉中的一种方法，使用旋转的边界框检测图像中的倾斜物体，通常应用于航拍和卫星图像。与传统的边界框不同，OBB 可以更好地贴合不同方向的物体。

- [DOTA-v2](obb/dota-v2.md)：一个流行的 OBB 航拍图像数据集，包含 170 万个实例和 11,268 张图像。
- [DOTA8](obb/dota8.md)：DOTAv1 分割集前 8 张图像的一个较小子集，4 张用于训练，4 张用于验证，适用于快速测试。
- [DOTA128](obb/dota128.md)：DOTA 数据集的 128 张图像子集，用于训练和验证，在测试 OBB 模型时提供了规模与多样性之间的良好平衡。

## [多目标跟踪](track/index.md)

多目标跟踪是一种计算机视觉技术，涉及在视频序列中随时间检测和跟踪多个物体。此任务通过跨帧保持物体的一致身份来扩展目标检测。

- [Argoverse](detect/argoverse.md)：一个包含来自城市场景的 3D 跟踪和运动预测数据的数据集，具有丰富的标注信息，用于多目标跟踪任务。
- [VisDrone](detect/visdrone.md)：一个包含无人机拍摄图像的目标检测和多目标跟踪数据的数据集，拥有超过 1 万张图像和视频序列。

## 贡献新数据集

贡献新数据集需要几个步骤，以确保其与现有基础设施良好对齐。以下是必要的步骤：

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yMR7BgwHQ3g?start=427"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何为 Ultralytics 贡献数据集
</p>

### 贡献新数据集的步骤

1. **收集图像**：收集属于该数据集的图像。这些图像可以从各种来源收集，例如公共数据库或您自己的收藏。
2. **标注图像**：根据任务类型，使用边界框、分割掩码或关键点对这些图像进行标注。
3. **导出标注**：将这些标注转换为 Ultralytics 支持的 YOLO `*.txt` 文件格式。
4. **组织数据集**：将数据集按正确的文件夹结构排列。您应有 `images/` 和 `labels/` 顶层目录，每个目录内包含 `train/` 和 `val/` 子目录。

    ```
    dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    ```

5. **创建 `data.yaml` 文件**：在数据集的根目录中，创建一个描述数据集、类别和其他必要信息的 `data.yaml` 文件。
6. **优化图像（可选）**：如果要减小数据集的大小以实现更高效的处理，可以使用以下代码优化图像。这不是必需的，但建议这样做以减少数据集大小并提高下载速度。
7. **压缩数据集**：将整个数据集文件夹压缩为一个 zip 文件。
8. **文档与 PR**：创建一个描述您的数据集以及它如何融入现有框架的文档页面。然后提交 Pull Request (PR)。有关如何提交 PR 的更多详细信息，请参阅 [Ultralytics 贡献指南](https://docs.ultralytics.com/help/contributing)。

### 优化和压缩数据集的示例代码

!!! example "优化和压缩数据集"

    === "Python"

       ```python
       from pathlib import Path

       from ultralytics.data.utils import compress_one_image
       from ultralytics.utils.downloads import zip_directory

       # Define dataset directory
       path = Path("path/to/dataset")

       # Optimize images in dataset (optional)
       for f in path.rglob("*.jpg"):
           compress_one_image(f)

       # Zip dataset into 'path/to/dataset.zip'
       zip_directory(path)
       ```

按照这些步骤，您可以贡献一个与 Ultralytics 现有结构良好集成的新数据集。

## 常见问题

### Ultralytics 支持哪些用于目标检测的数据集？

Ultralytics 支持多种用于[目标检测](https://www.ultralytics.com/glossary/object-detection)的数据集，包括：

- [COCO](detect/coco.md)：一个大规模的目标检测、分割和描述数据集，包含 80 个物体类别。
- [LVIS](detect/lvis.md)：一个广泛的数据集，包含 1203 个物体类别，专为更细粒度的目标检测和分割而设计。
- [Argoverse](detect/argoverse.md)：一个包含来自城市场景的 3D 跟踪和运动预测数据的数据集，具有丰富的标注信息。
- [VisDrone](detect/visdrone.md)：一个包含无人机拍摄图像的目标检测和多目标跟踪数据的数据集。
- [SKU-110K](detect/sku-110k.md)：包含零售环境中密集目标检测的数据集，拥有超过 1.1 万张图像。

这些数据集有助于训练适用于各种目标检测应用的鲁棒 [Ultralytics YOLO](https://docs.ultralytics.com/models) 模型。

### 如何为 Ultralytics 贡献新数据集？

贡献新数据集需要几个步骤：

1. **收集图像**：从公共数据库或个人收藏中收集图像。
2. **标注图像**：根据任务类型应用边界框、分割掩码或关键点。
3. **导出标注**：将标注转换为 YOLO `*.txt` 格式。
4. **组织数据集**：使用包含 `train/` 和 `val/` 目录的文件夹结构，每个目录包含 `images/` 和 `labels/` 子目录。
5. **创建 `data.yaml` 文件**：包含数据集描述、类别和其他相关信息。
6. **优化图像（可选）**：减小数据集大小以提高效率。
7. **压缩数据集**：将数据集压缩为 zip 文件。
8. **文档与 PR**：描述您的数据集并按照 [Ultralytics 贡献指南](https://docs.ultralytics.com/help/contributing) 提交 Pull Request。

请访问[贡献新数据集](#贡献新数据集)获取完整指南。

### 为什么应该使用 Ultralytics 平台来管理我的数据集？

[Ultralytics 平台](https://platform.ultralytics.com/) 为数据集管理和分析提供了强大的功能，包括：

- **无缝的数据集管理**：在一个地方上传、组织和管理您的数据集。
- **即时训练集成**：直接使用上传的数据集进行模型训练，无需额外设置。
- **可视化工具**：探索和可视化您的数据集图像和标注。
- **数据集分析**：获取关于数据集分布和特征的洞察。

该平台简化了从数据集管理到模型训练的过渡，使整个过程更加高效。了解更多关于 [Ultralytics 平台数据集](https://docs.ultralytics.com/platform/data) 的信息。

### Ultralytics YOLO 模型在计算机视觉方面有哪些独特功能？

Ultralytics YOLO 模型为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务提供了若干独特功能：

- **实时性能**：高速推理和训练能力，适用于时间敏感的应用。
- **多功能性**：在统一框架中支持检测、分割、分类和姿态估计任务。
- **预训练模型**：为各种应用提供高性能的预训练模型，缩短训练时间。
- **广泛的社区支持**：活跃的社区和全面的文档，用于故障排除和开发。
- **易于集成**：提供简单的 API，便于与现有项目和工作流集成。

在 [Ultralytics 模型](https://docs.ultralytics.com/models) 页面上了解更多关于 YOLO 模型的信息。

### 如何使用 Ultralytics 工具优化和压缩数据集？

要使用 Ultralytics 工具优化和压缩数据集，请参考以下示例代码：

!!! example "优化和压缩数据集"

    === "Python"

        ```python
        from pathlib import Path

        from ultralytics.data.utils import compress_one_image
        from ultralytics.utils.downloads import zip_directory

        # Define dataset directory
        path = Path("path/to/dataset")

        # Optimize images in dataset (optional)
        for f in path.rglob("*.jpg"):
            compress_one_image(f)

        # Zip dataset into 'path/to/dataset.zip'
        zip_directory(path)
        ```

此过程有助于减小数据集大小，实现更高效的存储和更快的下载速度。了解更多关于[优化和压缩数据集](#优化和压缩数据集的示例代码)的信息。