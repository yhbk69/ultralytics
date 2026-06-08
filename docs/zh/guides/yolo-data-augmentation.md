---
comments: true
description: 了解 Ultralytics YOLO 中的核心数据增强技术。探索各种变换、其影响以及如何有效实施以提升模型性能。
keywords: YOLO 数据增强, 计算机视觉, 深度学习, 图像变换, 模型训练, Ultralytics YOLO, HSV 调整, 几何变换, Mosaic 增强
---

# 使用 Ultralytics YOLO 进行数据增强

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/albumentations-augmentation.avif" alt="YOLO 数据增强示例，展示原始图像和增强后的训练图像">
</p>

## 简介

[数据增强](https://www.ultralytics.com/glossary/data-augmentation)是计算机视觉中的一项关键技术，通过对现有图像应用各种变换来人工扩展训练数据集。在训练 Ultralytics YOLO 等[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型时，数据增强有助于提高模型的鲁棒性、减少过拟合并增强对真实场景的泛化能力。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/e-TwqFtay90"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Mosaic、MixUp 等数据增强技术帮助 Ultralytics YOLO 模型更好地泛化 🚀
</p>

### 为什么数据增强很重要

数据增强在训练计算机视觉模型时具有多重关键作用：

- **扩展数据集**：通过创建现有图像的变体，可以在不收集新数据的情况下有效增加训练数据集的大小。
- **改善泛化能力**：模型学会在各种条件下识别物体，使其在真实应用中更加鲁棒。
- **减少过拟合**：通过在训练数据中引入变化，模型不太可能记忆特定的图像特征。
- **提升性能**：经过适当增强训练的模型通常在验证集和测试集上取得更好的[准确度](https://www.ultralytics.com/glossary/accuracy)。

Ultralytics YOLO 的实现提供了一套全面的增强技术，每种技术都有其特定用途，并以不同方式提升模型性能。本指南将详细探讨每个增强参数，帮助您理解何时以及如何在项目中有效使用它们。

### 配置示例

您可以使用 Python API、命令行界面（CLI）或配置文件来自定义每个参数。以下是每种方法中设置数据增强的示例。

!!! example "配置示例"

    === "Python"

        ```python
        import albumentations as A

        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")

        # 使用自定义增强参数进行训练
        model.train(data="coco.yaml", epochs=100, hsv_h=0.03, hsv_s=0.6, hsv_v=0.5)

        # 不使用任何增强进行训练（为简洁起见省略了已禁用的值）
        model.train(
            data="coco.yaml",
            epochs=100,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            translate=0.0,
            scale=0.0,
            fliplr=0.0,
            mosaic=0.0,
            erasing=0.0,
            auto_augment=None,
        )

        # 使用自定义 Albumentations 变换进行训练（仅限 Python API）
        custom_transforms = [
            A.Blur(blur_limit=7, p=0.5),
            A.CLAHE(clip_limit=4.0, p=0.5),
        ]
        model.train(data="coco.yaml", epochs=100, augmentations=custom_transforms)
        ```

    === "CLI"

        ```bash
        # 使用自定义增强参数进行训练
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 hsv_h=0.03 hsv_s=0.6 hsv_v=0.5
        ```

#### 使用配置文件

您可以在 YAML 配置文件（例如 `train_custom.yaml`）中定义所有训练参数，包括增强参数。`mode` 参数仅在使用 CLI 时需要。这个新的 YAML 文件将覆盖 `ultralytics` 包中[默认配置文件](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml)中的设置。

```yaml
# train_custom.yaml
# 'mode' 仅在使用 CLI 时需要
mode: train
data: coco8.yaml
model: yolo26n.pt
epochs: 100
hsv_h: 0.03
hsv_s: 0.6
hsv_v: 0.5
```

然后使用 Python API 启动训练：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLO26n 模型
        model = YOLO("yolo26n.pt")

        # 使用自定义配置训练模型
        model.train(cfg="train_custom.yaml")
        ```

    === "CLI"

        ```bash
        # 使用自定义配置训练模型
        yolo detect train model="yolo26n.pt" cfg=train_custom.yaml
        ```

## 色彩空间增强

### 色调调整 (`hsv_h`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ hsv_h }}`
- **用法**：在保持颜色关系的同时偏移图像颜色。`hsv_h` 超参数定义偏移幅度，最终调整值在 `-hsv_h` 和 `hsv_h` 之间随机选择。例如，`hsv_h=0.3` 时，偏移量在 `-0.3` 到 `0.3` 之间随机选取。当值超过 `0.5` 时，色调偏移会在色轮上循环，这就是为什么 `0.5` 和 `-0.5` 的增强效果相同。
- **用途**：对于户外场景特别有用，因为光照条件会显著影响物体的外观。例如，香蕉在明亮阳光下可能看起来更黄，而在室内则偏绿。
- **Ultralytics 实现**：[RandomHSV](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.RandomHSV)

|                                                              **`-0.5`**                                                              |                                                              **`-0.25`**                                                               |                                                                 **`0.0`**                                                                  |                                                              **`0.25`**                                                              |                                                              **`0.5`**                                                              |
| :----------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_hsv_h_-0.5.avif" alt="色调偏移 -0.5 增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_hsv_h_-0.25.avif" alt="色调偏移 -0.25 增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_identity.avif" alt="未增强的原始图像"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_hsv_h_0.25.avif" alt="色调偏移 0.25 增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_hsv_h_0.5.avif" alt="色调偏移 0.5 增强效果"/> |

### 饱和度调整 (`hsv_s`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ hsv_s }}`
- **用法**：修改图像中颜色的强度。`hsv_s` 超参数定义偏移幅度，最终调整值在 `-hsv_s` 和 `hsv_s` 之间随机选择。例如，`hsv_s=0.7` 时，强度在 `-0.7` 到 `0.7` 之间随机选取。
- **用途**：帮助模型应对不同的天气条件和相机设置。例如，红色交通标志在晴天可能非常鲜艳，但在雾天则显得暗淡褪色。
- **Ultralytics 实现**：[RandomHSV](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.RandomHSV)

|                                                                  **`-1.0`**                                                                   |                                                              **`-0.5`**                                                               |                                                                 **`0.0`**                                                                  |                                                              **`0.5`**                                                              |                                                                **`1.0`**                                                                |
| :-------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_hsv_s_-1.avif" alt="饱和度 -1.0 灰度增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_hsv_s_-0.5.avif" alt="饱和度 -0.5 增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_identity.avif" alt="未增强的原始图像"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_hsv_s_0.5.avif" alt="饱和度 0.5 增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_hsv_s_1.avif" alt="饱和度 1.0 鲜艳增强效果"/> |

### 亮度调整 (`hsv_v`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ hsv_v }}`
- **用法**：改变图像的亮度。`hsv_v` 超参数定义偏移幅度，最终调整值在 `-hsv_v` 和 `hsv_v` 之间随机选择。例如，`hsv_v=0.4` 时，强度在 `-0.4` 到 `0.4` 之间随机选取。
- **用途**：对于需要在不同光照条件下运行的模型训练至关重要。例如，红苹果在阳光下可能看起来很亮，但在阴影中则暗得多。
- **Ultralytics 实现**：[RandomHSV](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.RandomHSV)

|                                                                **`-1.0`**                                                                |                                                              **`-0.5`**                                                               |                                                                 **`0.0`**                                                                  |                                                              **`0.5`**                                                              |                                                                **`1.0`**                                                                 |
| :--------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_hsv_v_-1.avif" alt="亮度 -1.0 变暗增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_hsv_v_-0.5.avif" alt="亮度 -0.5 增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_identity.avif" alt="未增强的原始图像"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_hsv_v_0.5.avif" alt="亮度 0.5 增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_hsv_v_1.avif" alt="亮度 1.0 变亮增强效果"/> |

## 几何变换

### 旋转 (`degrees`)

- **范围**：`0.0` 到 `180`
- **默认值**：`{{ degrees }}`
- **用法**：在指定范围内随机旋转图像。`degrees` 超参数定义旋转角度，最终调整值在 `-degrees` 和 `degrees` 之间随机选择。例如，`degrees=10.0` 时，旋转角度在 `-10.0` 到 `10.0` 之间随机选取。
- **用途**：对于物体可能以不同方向出现的应用至关重要。例如，在无人机航拍图像中，车辆可以朝向任意方向，这要求模型能够识别各种旋转角度的物体。
- **Ultralytics 实现**：[RandomPerspective](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.RandomPerspective)

|                                                                       **`-180`**                                                                        |                                                                       **`-90`**                                                                       |                                                                 **`0.0`**                                                                  |                                                                      **`90`**                                                                       |                                                                       **`180`**                                                                       |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_degrees_-180.avif" alt="旋转 -180 度增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_degrees_-90.avif" alt="旋转 -90 度增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_identity.avif" alt="未增强的原始图像"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_degrees_90.avif" alt="旋转 90 度增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_degrees_180.avif" alt="旋转 180 度增强效果"/> |

### 平移 (`translate`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ translate }}`
- **用法**：以图像大小的随机比例在水平和垂直方向上平移图像。`translate` 超参数定义平移幅度，最终调整值在 `-translate` 和 `translate` 之间随机选择两次（每个轴一次）。例如，`translate=0.5` 时，x 轴平移量在 `-0.5` 到 `0.5` 之间随机选取，y 轴平移量在同样范围内独立随机选取。
- **用途**：帮助模型学习检测部分可见的物体，并提高对物体位置的鲁棒性。例如，在车辆损伤评估应用中，车辆部件可能根据拍摄者的位置和距离而完整或部分出现在画面中，平移增强将教会模型无论完整性或位置如何都能识别这些特征。
- **Ultralytics 实现**：[RandomPerspective](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.RandomPerspective)
- **注意**：为简洁起见，下面应用的平移在 x 轴和 y 轴上每次都相同。不显示 `-1.0` 和 `1.0` 的值，因为它们会将图像完全移出画面。

|                                                                           **`-0.5`**                                                                           |                                                                         **`-0.25`**                                                                          |                                                                 **`0.0`**                                                                  |                                                                         **`0.25`**                                                                         |                                                                        **`0.5`**                                                                         |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_translate_-0.5.avif" alt="平移 -0.5 偏移增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_translate_-0.25.avif" alt="平移 -0.25 偏移增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_identity.avif" alt="未增强的原始图像"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_translate_0.25.avif" alt="平移 0.25 偏移增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_translate_0.5.avif" alt="平移 0.5 偏移增强效果"/> |

### 缩放 (`scale`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ scale }}`
- **用法**：在指定范围内按随机比例调整图像大小。`scale` 超参数定义缩放因子，最终调整值在 `1-scale` 和 `1+scale` 之间随机选择。例如，`scale=0.5` 时，缩放比例在 `0.5` 到 `1.5` 之间随机选取。
- **用途**：使模型能够处理不同距离和尺寸的物体。例如，在自动驾驶应用中，车辆可能出现在距离相机的不同距离处，这要求模型能够识别不同大小的物体。
- **Ultralytics 实现**：[RandomPerspective](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.RandomPerspective)
- **注意**：
    - 不显示 `-1.0` 的值，因为它会使图像消失，而 `1.0` 仅产生 2 倍放大效果。
    - 下表中显示的值是通过超参数 `scale` 应用的值，而非最终的缩放因子。
    - 如果 `scale` 大于 `1.0`，图像可能变得非常小或被翻转，因为缩放因子在 `1-scale` 和 `1+scale` 之间随机选择。例如，`scale=3.0` 时，缩放比例在 `-2.0` 到 `4.0` 之间随机选取。如果选中负值，图像将被翻转。

|                                                                     **`-0.5`**                                                                      |                                                                      **`-0.25`**                                                                      |                                                                 **`0.0`**                                                                  |                                                                     **`0.25`**                                                                      |                                                                     **`0.5`**                                                                     |
| :-------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_scale_-0.5.avif" alt="缩放 0.5x 缩小增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_scale_-0.25.avif" alt="缩放 0.75x 缩小增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_identity.avif" alt="未增强的原始图像"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_scale_0.25.avif" alt="缩放 1.25x 放大增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_scale_0.5.avif" alt="缩放 1.5x 放大增强效果"/> |

### 错切 (`shear`)

- **范围**：`-180` 到 `+180`
- **默认值**：`{{ shear }}`
- **用法**：引入沿 x 轴和 y 轴倾斜图像的几何变换，在保持平行线的前提下将图像部分沿一个方向偏移。`shear` 超参数定义错切角度，最终调整值在 `-shear` 和 `shear` 之间随机选择。例如，`shear=10.0` 时，x 轴错切角度在 `-10` 到 `10` 之间随机选取，y 轴错切角度在同样范围内独立随机选取。
- **用途**：帮助模型泛化到因轻微倾斜或斜视角引起的视角变化。例如，在交通监控中，由于相机非垂直安装，汽车和路标等物体可能看起来倾斜。应用错切增强可确保模型学习识别这些物体，尽管存在倾斜变形。
- **Ultralytics 实现**：[RandomPerspective](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.RandomPerspective)
- **注意**：
    - `shear` 值会快速扭曲图像，建议从较小的值开始，逐渐增加。
    - 与透视变换不同，错切不会引入深度或消失点，而是通过改变角度同时保持对边平行来扭曲物体的形状。

|                                                                    **`-10`**                                                                     |                                                                    **`-5`**                                                                    |                                                                 **`0.0`**                                                                  |                                                                   **`5`**                                                                    |                                                                    **`10`**                                                                    |
| :----------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_shear_-10.avif" alt="错切 -10 度增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_shear_-5.avif" alt="错切 -5 度增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_identity.avif" alt="未增强的原始图像"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_shear_5.avif" alt="错切 5 度增强效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_shear_10.avif" alt="错切 10 度增强效果"/> |

### 透视 (`perspective`)

- **范围**：`0.0` - `0.001`
- **默认值**：`{{ perspective }}`
- **用法**：沿 x 轴和 y 轴应用完整的透视变换，模拟物体从不同深度或角度观看时的效果。`perspective` 超参数定义透视幅度，最终调整值在 `-perspective` 和 `perspective` 之间随机选择。例如，`perspective=0.001` 时，x 轴透视量在 `-0.001` 到 `0.001` 之间随机选取，y 轴透视量在同样范围内独立随机选取。
- **用途**：透视增强对于处理极端视角变化至关重要，特别是在物体因透视偏移而显得缩短或扭曲的场景中。例如，在基于无人机的物体检测中，建筑物、道路和车辆可能会根据无人机的倾斜角度和高度而显得拉伸或压缩。通过应用透视变换，模型可以学习识别这些物体，尽管存在透视引起的变形，从而提高在实际部署中的鲁棒性。
- **Ultralytics 实现**：[RandomPerspective](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.RandomPerspective)

|                                                                         **`-0.001`**                                                                         |                                                                         **`-0.0005`**                                                                          |                                                                 **`0.0`**                                                                  |                                                                         **`0.0005`**                                                                         |                                                                        **`0.001`**                                                                         |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_perspective_-0.001.avif" alt="透视 -0.001 变换效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_perspective_-0.0005.avif" alt="透视 -0.0005 变换效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_identity.avif" alt="未增强的原始图像"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_perspective_0.0005.avif" alt="透视 0.0005 变换效果"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_geometric_perspective_0.001.avif" alt="透视 0.001 变换效果"/> |

### 上下翻转 (`flipud`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ flipud }}`
- **用法**：沿 y 轴反转图像执行垂直翻转。此变换将整个图像上下镜像，但保留物体之间的所有空间关系。`flipud` 超参数定义应用变换的概率，`flipud=1.0` 时确保所有图像都被翻转，`flipud=0.0` 时完全禁用该变换。例如，`flipud=0.5` 时，每张图像有 50% 的概率被上下翻转。
- **用途**：适用于物体可能颠倒出现的场景。例如，在机器人视觉系统中，传送带或机械臂上的物体可能以各种方向被拾取和放置。垂直翻转帮助模型识别物体，无论其上下方向如何。
- **Ultralytics 实现**：[RandomFlip](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.RandomFlip)

|                                                                    **`flipud` 关闭**                                                                    |                                                                       **`flipud` 开启**                                                                        |
| :----------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_identity.avif" alt="未增强的原始图像" width="38%"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_flip_vertical_1.avif" alt="垂直翻转增强已启用" width="38%"/> |

### 左右翻转 (`fliplr`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ fliplr }}`
- **用法**：沿 x 轴镜像图像执行水平翻转。此变换交换左右两侧同时保持空间一致性，帮助模型泛化到镜像方向的物体。`fliplr` 超参数定义应用变换的概率，`fliplr=1.0` 时确保所有图像都被翻转，`fliplr=0.0` 时完全禁用该变换。例如，`fliplr=0.5` 时，每张图像有 50% 的概率被左右翻转。
- **用途**：水平翻转广泛用于物体检测、姿态估计和人脸识别，以提高对左右变化的鲁棒性。例如，在自动驾驶中，车辆和行人可能出现在道路的任意一侧，水平翻转帮助模型在两种方向下都能同样准确地识别它们。
- **Ultralytics 实现**：[RandomFlip](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.RandomFlip)

|                                                                    **`fliplr` 关闭**                                                                    |                                                                         **`fliplr` 开启**                                                                          |
| :----------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_identity.avif" alt="未增强的原始图像" width="38%"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_flip_horizontal_1.avif" alt="水平翻转增强已启用" width="38%"/> |

### BGR 通道交换 (`bgr`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ bgr }}`
- **用法**：将图像的颜色通道从 RGB 交换为 BGR，改变颜色的表示顺序。`bgr` 超参数定义应用变换的概率，`bgr=1.0` 时确保所有图像都进行通道交换，`bgr=0.0` 时禁用该变换。例如，`bgr=0.5` 时，每张图像有 50% 的概率从 RGB 转换为 BGR。
- **用途**：提高对不同颜色通道顺序的鲁棒性。例如，当训练的模型需要在 RGB 和 BGR 格式可能不一致使用的各种相机系统和图像库中工作时，或者将模型部署到输入颜色格式可能与训练数据不同的环境中时。
- **Ultralytics 实现**：[Format](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.Format)

|                                                                     **`bgr` 关闭**                                                                      |                                                                        **`bgr` 开启**                                                                        |
| :----------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_identity.avif" alt="未增强的原始图像" width="38%"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_bgr_channel_swap_1.avif" alt="BGR 通道交换增强效果" width="38%"/> |

### 马赛克 (`mosaic`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ mosaic }}`
- **用法**：将四张训练图像合并为一张。`mosaic` 超参数定义应用变换的概率，`mosaic=1.0` 时确保所有图像都被合并，`mosaic=0.0` 时禁用该变换。例如，`mosaic=0.5` 时，每张图像有 50% 的概率与其他三张图像合并。
- **用途**：对于改善小物体检测和上下文理解非常有效。例如，在野生动物保护项目中，动物可能以不同距离和尺度出现，马赛克增强通过从有限数据中人工创建多样化的训练样本来帮助模型学习在不同大小、部分遮挡和环境背景下识别同一物种。
- **Ultralytics 实现**：[Mosaic](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.Mosaic)
- **注意**：
    - 即使 `mosaic` 增强使模型更鲁棒，它也可能使训练过程更具挑战性。
    - 可以在训练接近结束时通过将 `close_mosaic` 设置为训练结束前应禁用的 epoch 数来关闭 `mosaic` 增强。例如，如果 `epochs` 设置为 `200`，`close_mosaic` 设置为 `20`，则 `mosaic` 增强将在 `180` 个 epoch 后禁用。如果 `close_mosaic` 设置为 `0`，则 `mosaic` 增强将在整个训练过程中启用。
    - 生成马赛克的中心位置使用随机值确定，既可能在图像内部，也可能在图像外部。
    - 当前实现的 `mosaic` 增强从数据集中随机选择 4 张图像进行合并。如果数据集较小，同一张图像可能在同一马赛克中被多次使用。

|                                                                    **`mosaic` 关闭**                                                                    |                                                                     **`mosaic` 开启**                                                                     |
| :----------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_identity.avif" alt="未增强的原始图像" width="38%"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_mosaic_on.avif" alt="4 图像马赛克增强已启用" width="55%"/> |

### 混合 (`mixup`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ mixup }}`
- **用法**：以给定概率混合两张图像及其标签。`mixup` 超参数定义应用变换的概率，`mixup=1.0` 时确保所有图像都被混合，`mixup=0.0` 时禁用该变换。例如，`mixup=0.5` 时，每张图像有 50% 的概率与另一张图像混合。
- **用途**：提高模型鲁棒性并减少过拟合。例如，在零售产品识别系统中，混合增强通过将不同产品的图像混合，教会模型即使在拥挤货架上部分可见或被其他产品遮挡时也能识别商品，从而帮助模型学习更鲁棒的特征。
- **Ultralytics 实现**：[MixUp](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.MixUp)
- **注意**：
    - 混合比例是从 `np.random.beta(32.0, 32.0)` Beta 分布中随机选取的，意味着每张图像贡献约 50%，有轻微变化。

|                                                           **第一张图像，`mixup` 关闭**                                                            |                                                               **第二张图像，`mixup` 关闭**                                                                |                                                                     **`mixup` 开启**                                                                     |
| :-----------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_identity.avif" alt="混合增强的第一张图像" width="60%"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_mixup_identity_2.avif" alt="混合增强的第二张图像" width="60%"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_mixup_on.avif" alt="混合增强已启用" width="85%"/> |

### CutMix (`cutmix`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ cutmix }}`
- **用法**：以给定概率从一张图像中剪切一个矩形区域并粘贴到另一张图像上。`cutmix` 超参数定义应用变换的概率，`cutmix=1.0` 时确保所有图像都经过此变换，`cutmix=0.0` 时完全禁用。例如，`cutmix=0.5` 时，每张图像有 50% 的概率其某个区域被来自另一张图像的补丁替换。
- **用途**：通过创建真实的遮挡场景同时保持局部特征完整性来增强模型性能。例如，在自动驾驶系统中，CutMix 帮助模型学习识别车辆或行人，即使它们被其他物体部分遮挡，从而提高复杂真实环境中重叠物体场景下的检测精度。
- **Ultralytics 实现**：[CutMix](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.CutMix)
- **注意**：
    - 剪切区域的大小和位置每次应用时随机确定。
    - 与全局混合像素值的 mixup 不同，`cutmix` 保持剪切区域内原始像素强度不变，保留局部特征。
    - 只有当补丁区域不与任何现有边界框重叠时，才会将区域粘贴到目标图像中。此外，只保留在粘贴区域内至少保留 `0.1`（10%）原始面积的边界框。
    - 在当前实现中，此最小边界框面积阈值无法更改，默认设置为 `0.1`。

|                                                           **第一张图像，`cutmix` 关闭**                                                            |                                                           **第二张图像，`cutmix` 关闭**                                                            |                                                                 **`cutmix` 开启**                                                                 |
| :------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_cutmix_identity_1.avif" alt="CutMix 的第一张图像" width="85%"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_cutmix_identity_2.avif" alt="CutMix 的第二张图像" width="85%"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_cutmix_on.avif" alt="CutMix 增强已启用" width="85%"/> |

## 分割专用增强

### 复制-粘贴 (`copy_paste`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ copy_paste }}`
- **用法**：仅适用于分割任务，此增强根据指定概率在图像内或图像之间复制物体，由 [`copy_paste_mode`](#copy-paste-mode-copy_paste_mode) 参数控制。`copy_paste` 超参数定义应用变换的概率，`copy_paste=1.0` 时确保所有图像都进行复制，`copy_paste=0.0` 时禁用该变换。例如，`copy_paste=0.5` 时，每张图像有 50% 的概率从另一张图像复制物体。
- **用途**：对于实例分割任务和稀有物体类别特别有用。例如，在工业缺陷检测中，某些类型的缺陷出现频率很低，复制-粘贴增强可以通过将缺陷从一张图像复制到另一张图像来人工增加这些稀有缺陷的出现频率，帮助模型更好地学习这些代表性不足的案例，无需额外的缺陷样本。
- **Ultralytics 实现**：[CopyPaste](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.CopyPaste)
- **注意**：
    - 如下面的动图所示，`copy_paste` 增强可用于将物体从一张图像复制到另一张图像。
    - 一旦复制了物体，无论 `copy_paste_mode` 如何，都会计算其与源图像中所有物体的面积交并比（IoA）。如果所有 IoA 都低于 `0.3`（30%），则将该物体粘贴到目标图像中。只要有一个 IoA 高于 `0.3`，该物体就不会被粘贴到目标图像中。
    - IoA 阈值在当前实现中无法更改，默认设置为 `0.3`。

|                                                                     **`copy_paste` 关闭**                                                                     |                                                     **`copy_paste` 开启，`copy_paste_mode=flip`**                                                     |                                                               可视化 `copy_paste` 过程                                                                |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_copy_paste_off.avif" alt="未增强的原始图像" width="80%"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_copy_paste_on.avif" alt="复制-粘贴增强已启用" width="80%"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_copy_paste_demo.avif" alt="复制-粘贴增强动画演示" width="97%"/> |

### 复制-粘贴模式 (`copy_paste_mode`)

- **选项**：`'flip'`、`'mixup'`
- **默认值**：`'{{ copy_paste_mode }}'`
- **用法**：确定[复制-粘贴](#copy-paste-copy_paste)增强所使用的方法。设置为 `'flip'` 时，物体来自同一张图像；设置为 `'mixup'` 时，允许从不同图像复制物体。
- **用途**：为复制的物体如何集成到目标图像中提供灵活性。
- **Ultralytics 实现**：[CopyPaste](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.CopyPaste)
- **注意**：
    - 两种 `copy_paste_mode` 的 IoA 原则相同，但复制物体的方式不同。
    - 取决于图像大小，物体有时可能部分或完全被复制到画面之外。
    - 取决于多边形标注的质量，复制的物体可能与原始物体有轻微的形状变化。

|                                                                    **参考图像**                                                                     |                                                              **`copy_paste` 选中的图像**                                                               |                                                   **`copy_paste` 开启，`copy_paste_mode=mixup`**                                                    |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_mixup_identity_2.avif" alt="混合增强的第二张图像" width="77%"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_copy_paste_off.avif" alt="未增强的原始图像" width="80%"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_copy_paste_mixup.avif" alt="使用 MixUp 模式的复制-粘贴" width="77%"/> |

## 分类专用增强

### 自动增强 (`auto_augment`)

- **选项**：`'randaugment'`、`'autoaugment'`、`'augmix'`、`None`
- **默认值**：`'{{ auto_augment }}'`
- **用法**：为分类任务应用自动化增强策略。`'randaugment'` 选项使用 RandAugment，`'autoaugment'` 使用 AutoAugment，`'augmix'` 使用 AugMix。设置为 `None` 则禁用自动增强。
- **用途**：自动优化分类任务的增强策略。三者的区别如下：
    - **AutoAugment**：此模式应用从 ImageNet、CIFAR10 和 SVHN 等数据集学习到的预定义增强策略。用户可以选用这些现有策略，但无法在 Torchvision 中训练新策略。要为特定数据集发现最优增强策略，需要使用外部库或自定义实现。参考 [AutoAugment 论文](https://arxiv.org/abs/1805.09501)。
    - **RandAugment**：以统一幅度随机选择变换。此方法减少了对大量搜索阶段的需求，计算效率更高，同时仍能增强模型鲁棒性。参考 [RandAugment 论文](https://arxiv.org/abs/1909.13719)。
    - **AugMix**：AugMix 是一种通过简单变换的随机组合创建多样化图像变体来增强模型鲁棒性的数据增强方法。参考 [AugMix 论文](https://arxiv.org/abs/1912.02781)。
- **Ultralytics 实现**：[classify_augmentations()](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.classify_augmentations)
- **注意**：
    - 本质而言，三种方法的主要区别在于增强策略的定义和应用方式。
    - 您可以参考[这篇文章](https://sebastianraschka.com/blog/2023/data-augmentation-pytorch.html)详细比较这三种方法。

### 随机擦除 (`erasing`)

- **范围**：`0.0` - `0.9`
- **默认值**：`{{ erasing }}`
- **用法**：在分类训练期间随机擦除图像的部分区域。`erasing` 超参数定义应用变换的概率，`erasing=0.9` 时确保几乎所有图像都被擦除，`erasing=0.0` 时禁用该变换。例如，`erasing=0.5` 时，每张图像有 50% 的概率被擦除部分区域。
- **用途**：帮助模型学习鲁棒的特征，防止过度依赖特定的图像区域。例如，在人脸识别系统中，随机擦除帮助模型对太阳镜、口罩或其他可能部分遮挡面部特征的物体更加鲁棒。这通过迫使模型使用多种面部特征来识别个体，而不是仅仅依赖可能被遮挡的显著特征，从而提高真实场景性能。
- **Ultralytics 实现**：[classify_augmentations()](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.classify_augmentations)
- **注意**：
    - `erasing` 增强附带有 `scale`、`ratio` 和 `value` 超参数，在当前[实现](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py#L2502)中无法更改。根据 PyTorch [文档](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandomErasing.html)，它们的默认值分别为 `(0.02, 0.33)`、`(0.3, 3.3)` 和 `0`。
    - `erasing` 超参数的上限设置为 `0.9`，以避免对所有图像应用该变换。

|                                                                   **`erasing` 关闭**                                                                    |                                                          **`erasing` 开启（示例 1）**                                                          |                                                          **`erasing` 开启（示例 2）**                                                          |                                                          **`erasing` 开启（示例 3）**                                                          |
| :----------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_identity.avif" alt="未增强的原始图像" width="85%"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_erasing_ex1.avif" alt="随机擦除示例 1" width="85%"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_erasing_ex2.avif" alt="随机擦除示例 2" width="85%"/> | <img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/augmentation_erasing_ex3.avif" alt="随机擦除示例 3" width="85%"/> |

## 高级增强功能

### 自定义 Albumentations 变换 (`augmentations`)

- **类型**：Albumentations 变换的 `list`
- **默认值**：`None`
- **用法**：允许您使用 Python API 提供自定义的 [Albumentations](https://albumentations.ai/) 变换用于数据增强。此参数接受一个 Albumentations 变换对象列表，这些变换将在训练期间替代默认的 Albumentations 变换进行应用。
- **用途**：通过利用 Albumentations 丰富的变换库，提供对数据增强策略的细粒度控制。当您需要超越 YOLO 内置选项的专业增强时特别有用，例如高级颜色调整、噪声注入或领域特定的变换。
- **Ultralytics 实现**：[Albumentations](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.Albumentations)

!!! example "自定义 Albumentations 示例"

    === "Python API"

        ```python
        import albumentations as A

        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")

        # 定义自定义 Albumentations 变换
        custom_transforms = [
            A.Blur(blur_limit=7, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ]

        # 使用自定义 Albumentations 变换进行训练
        model.train(
            data="coco8.yaml",
            epochs=100,
            augmentations=custom_transforms,  # 传入自定义变换
            imgsz=640,
        )
        ```

    === "更高级的示例"

        ```python
        import albumentations as A

        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")

        # 定义带特定参数的高级自定义 Albumentations 变换
        advanced_transforms = [
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.MedianBlur(blur_limit=7, p=1.0),
                    A.GaussianBlur(blur_limit=7, p=1.0),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                ],
                p=0.2,
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.2
            ),
        ]

        # 使用高级自定义变换进行训练
        model.train(
            data="coco8.yaml",
            epochs=100,
            augmentations=advanced_transforms,
            imgsz=640,
        )
        ```

**关键点：**

- **仅限 Python API**：自定义 Albumentations 变换目前仅支持通过 Python API 使用。不能通过 CLI 或 YAML 配置文件指定。
- **替换默认变换**：当您通过 `augmentations` 参数提供自定义变换时，它们会完全替换默认的 Albumentations 变换。**默认的 YOLO 增强（如 `mosaic`、`hsv_h`、`hsv_s`、`degrees` 等）保持活跃并独立应用**。
- **边界框兼容性**：在使用空间变换（改变图像几何形状的变换）时需谨慎。Ultralytics 会自动处理边界框调整，但某些复杂变换可能需要额外配置。
- **丰富的变换库**：Albumentations 提供超过 70 种不同的变换。探索 [Albumentations 文档](https://albumentations.ai/docs/)发现所有可用选项。
- **性能考虑**：添加过多增强或使用计算成本高的变换会减慢训练速度。从少量开始，监控训练速度。

**常见用例：**

- **医学影像**：应用弹性变形或网格扭曲等专业变换用于 X 光或 MRI 图像增强
- **航拍/卫星图像**：使用针对俯视视角优化的变换
- **低光条件**：应用噪声和亮度调整来模拟具有挑战性的光照
- **工业检测**：为质量控制应用添加缺陷模拟图案或纹理变化

**兼容性说明：**

- 需要 Albumentations 1.0.3 或更高版本
- 兼容所有 YOLO 检测和分割任务
- 不适用于分类任务（分类使用不同的增强流水线）

有关 Albumentations 和可用变换的更多信息，请访问[官方 Albumentations 文档](https://albumentations.ai/docs/)。

## 常见问题

### 可供选择的增强太多了。我怎么知道该用哪些？

选择合适的增强取决于您的具体用例和数据集。以下是一些帮助您决策的通用指南：

- 在大多数情况下，颜色和亮度的轻微变化是有益的。`hsv_h`、`hsv_s` 和 `hsv_v` 的默认值是一个可靠的起点。
- 如果相机视角一致且在模型部署后不会改变，您可能可以跳过几何变换，如 `rotation`、`translation`、`scale`、`shear` 或 `perspective`。但是，如果相机角度可能变化，且您需要模型更鲁棒，最好保留这些增强。
- 仅在有部分遮挡物体或每张图像多个物体是可接受的且不改变标签值时使用 `mosaic` 增强。或者，您可以保持 `mosaic` 活跃但增大 `close_mosaic` 值以在训练过程中更早禁用它。

简而言之：保持简单。从少量增强开始，根据需要逐渐添加更多。目标是提高模型的泛化能力和鲁棒性，而不是使训练过程过于复杂。此外，确保您应用的增强反映了模型在生产环境中将遇到的数据分布。

### 开始训练时，我看到一个 `albumentations: Blur[...]` 引用。这是否意味着 Ultralytics YOLO 运行了额外的增强，如模糊？

如果安装了 `albumentations` 包，Ultralytics 会自动使用它应用一组额外的图像增强。这些增强在内部处理，无需额外配置。

您可以在我们的[技术文档](https://docs.ultralytics.com/reference/data/augment#ultralytics.data.augment.Albumentations)以及 [Albumentations 集成指南](https://docs.ultralytics.com/integrations/albumentations)中找到完整的变换列表。请注意，只有概率 `p` 大于 `0` 的增强才会被激活。这些增强有意以低频应用，以模拟真实世界的视觉伪影，如模糊或灰度效果。

您也可以使用 Python API 提供自己的自定义 Albumentations 变换。请参阅[高级增强功能](#advanced-augmentation-features)部分了解更多详情。

### 开始训练时，我没有看到任何对 albumentations 的引用。为什么？

检查是否安装了 `albumentations` 包。如果没有安装，可以通过运行 `pip install albumentations` 安装。安装后，该包应被 Ultralytics 自动检测并使用。

### 如何自定义我的增强？

您可以通过创建自定义数据集类和训练器来自定义增强。例如，您可以用 PyTorch 的 [torchvision.transforms.Resize](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html) 或其他变换替换默认的 Ultralytics 分类增强。有关实现细节，请参阅分类文档中的[自定义训练示例](../tasks/classify.md#train)。
