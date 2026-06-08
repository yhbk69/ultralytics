---
comments: true
description: 学习如何使用 Ultralytics 预测模式从推理结果中提取孤立对象。分割对象隔离的分步指南。
keywords: Ultralytics, 分割, 对象隔离, 预测模式, YOLO26, 机器学习, 对象检测, 二值掩码, 图像处理
---

# 隔离分割对象

在执行[分割任务](../tasks/segment.md)后，有时需要从推理结果中提取孤立对象。本指南提供了使用 Ultralytics [预测模式](../modes/predict.md)实现此目标的通用方法。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/5HBB5IBuJ6c"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何使用 Ultralytics YOLO 分割与 OpenCV 在 Python 中移除背景并隔离对象 🚀
</p>

## 方法步骤

1.  查看 [Ultralytics 快速入门安装部分](../quickstart.md)，了解安装所需库的快速指南。

    ***

2.  加载模型并在源数据上运行 `predict()` 方法。

    ```python
    from ultralytics import YOLO

    # 加载模型
    model = YOLO("yolo26n-seg.pt")

    # 执行推理
    results = model.predict()
    ```

    !!! question "没有预测参数？"

        如果不指定源数据，将使用库中的示例图片：

        ```
        'ultralytics/assets/bus.jpg'
        'ultralytics/assets/zidane.jpg'
        ```

        这对于使用 `predict()` 方法快速测试非常有用。

    有关分割模型的更多信息，请访问[分割任务](../tasks/segment.md#models)页面。要了解有关 `predict()` 方法的更多信息，请参阅文档的[预测模式](../modes/predict.md)部分。

    ***

3.  现在遍历结果和轮廓。对于需要将图像保存到文件的工作流，可以先获取源图像的基本名称（`base-name`）和检测类别标签（`class-label`）以备后续使用（可选）。

    ```{ .py .annotate }
    from pathlib import Path

    import numpy as np

    # (2) 遍历检测结果（适用于多张图像）
    for r in results:
        img = np.copy(r.orig_img)
        img_name = Path(r.path).stem  # 源图像基本名称

        # 遍历每个对象轮廓（多个检测）
        for ci, c in enumerate(r):
            # (1) 获取检测类别名称
            label = c.names[c.boxes.cls.tolist().pop()]
    ```

    1. 要了解有关处理检测结果的更多信息，请参阅[预测模式的 Boxes 部分](../modes/predict.md#boxes)。
    2. 要了解有关 `predict()` 结果的更多信息，请参阅[预测模式的结果处理](../modes/predict.md#working-with-results)。

    ??? info "For 循环"

        单张图像只会迭代第一个循环一次。只有单个检测的单张图像每个循环只会迭代一次。

    ***

4.  首先从源图像生成二值掩码，然后在掩码上绘制填充轮廓。这将使对象能够与图像的其他部分隔离开来。右侧显示了 `bus.jpg` 中检测到的 `person` 类别对象之一的示例。

    ![二值掩码图像](https://github.com/ultralytics/ultralytics/assets/62214284/59bce684-fdda-4b17-8104-0b4b51149aca){ width="240", align="right" }

    ```{ .py .annotate }
    import cv2

    # 创建二值掩码
    b_mask = np.zeros(img.shape[:2], np.uint8)

    # (1) 提取轮廓结果
    contour = c.masks.xy.pop()
    # (2) 更改类型
    contour = contour.astype(np.int32)
    # (3) 重塑形状
    contour = contour.reshape(-1, 1, 2)


    # 将轮廓绘制到掩码上
    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
    ```

    1. 有关 `c.masks.xy` 的更多信息，请参阅[预测模式的掩码部分](../modes/predict.md#masks)。

    2. 此处将值转换为 `np.int32`，以便与 [OpenCV](https://www.ultralytics.com/glossary/opencv) 的 `drawContours()` 函数兼容。

    3. OpenCV 的 `drawContours()` 函数期望轮廓的形状为 `[N, 1, 2]`，展开下方部分了解更多详情。

    <details>
    <summary> 展开以了解定义 <code>contour</code> 变量时发生了什么。</summary>
    <p>
    - `c.masks.xy` :: 以 `(x, y)` 格式提供掩码轮廓点的坐标。有关更多详细信息，请参阅[预测模式的掩码部分](../modes/predict.md#masks)。
    - `.pop()` :: 由于 `masks.xy` 是包含单个元素的列表，使用 `pop()` 方法提取该元素。
    - `.astype(np.int32)` :: 使用 `masks.xy` 将返回 `float32` 数据类型，但这与 OpenCV 的 `drawContours()` 函数不兼容，因此将其数据类型更改为 `int32` 以兼容。
    - `.reshape(-1, 1, 2)` :: 将数据重新格式化为所需的 `[N, 1, 2]` 形状，其中 `N` 是轮廓点数量，每个点由单个条目 `1` 表示，该条目由 `2` 个值组成。`-1` 表示此维度上的值数量是灵活的。

    </details>
    <p></p>
    <details>
    <summary> 展开以了解 <code>drawContours()</code> 配置的说明。</summary>
    <p>
    - 在测试过程中，将 `contour` 变量包裹在方括号 `[contour]` 中可以有效地生成所需的轮廓掩码。
    - 为 `drawContours()` 参数指定的值 `-1` 指示函数绘制图像中存在的所有轮廓。
    - 元组 `(255, 255, 255)` 表示白色，即在此二值掩码中绘制轮廓所需的颜色。
    - 添加 `cv2.FILLED` 将使轮廓边界包围的所有像素着色为相同颜色，在此例中，所有包围的像素将为白色。
    - 有关更多信息，请参阅 [OpenCV 的 `drawContours()` 文档](https://docs.opencv.org/4.8.0/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc)。

    </details>
    <p></p>

    ***

5.  接下来，从此步骤开始有两种处理图像的选项，每种选项都有后续的子选项。

    ### 对象隔离选项

    !!! example

        === "黑色背景像素"

            ```python
            # 创建三通道掩码
            mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)

            # 使用二值掩码隔离对象
            isolated = cv2.bitwise_and(mask3ch, img)
            ```

            ??? question "这是如何工作的？"

                - 首先，将二值掩码从单通道图像转换为三通道图像。此转换对于后续将掩码与原始图像合并的步骤是必要的。两幅图像必须具有相同数量的通道才能兼容混合操作。

                - 使用 OpenCV 的 `bitwise_and()` 函数将原始图像与三通道二值掩码合并。此操作仅保留两幅图像中大于零 `(> 0)` 的像素值。由于掩码像素仅在轮廓区域内大于零 `(> 0)`，因此从原始图像中保留的像素是那些与轮廓重叠的像素。

            ### 黑色像素隔离：子选项

            ??? info "全尺寸图像"

                如果保留全尺寸图像，则不需要额外步骤。

                <figure markdown>
                    ![全尺寸隔离对象图像示例 - 黑色背景](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/full-size-isolated-object-black-background.avif){ width=240 }
                    <figcaption>全尺寸输出示例</figcaption>
                </figure>

            ??? info "裁剪对象图像"

                需要额外步骤将图像裁剪为仅包含对象区域。

                ![裁剪隔离对象图像示例 - 黑色背景](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/example-crop-isolated-object-image-black-background.avif){ align="right" }
                ```{ .py .annotate }
                # (1) 边界框坐标
                x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
                # 将图像裁剪到对象区域
                iso_crop = isolated[y1:y2, x1:x2]
                ```

                1.  有关[边界框](https://www.ultralytics.com/glossary/bounding-box)结果的更多信息，请参阅[预测模式的 Boxes 部分](../modes/predict.md#boxes)。

                ??? question "这段代码做了什么？"

                    - `c.boxes.xyxy.cpu().numpy()` 调用以 `xyxy` 格式将边界框作为 NumPy 数组检索，其中 `xmin`、`ymin`、`xmax` 和 `ymax` 表示边界框矩形的坐标。有关更多详细信息，请参阅[预测模式的 Boxes 部分](../modes/predict.md#boxes)。

                    - `squeeze()` 操作移除 NumPy 数组中所有不必要的维度，确保其具有预期的形状。

                    - 使用 `.astype(np.int32)` 转换坐标值将边界框坐标数据类型从 `float32` 更改为 `int32`，使其与使用索引切片进行图像裁剪兼容。

                    - 最后，使用索引切片从图像中裁剪出边界框区域。边界由检测边界框的 `[ymin:ymax, xmin:xmax]` 坐标定义。

        === "透明背景像素"

            ```python
            # 使用透明背景隔离对象（保存为 PNG 时）
            isolated = np.dstack([img, b_mask])
            ```

            ??? question "这是如何工作的？"

                - 使用 NumPy 的 `dstack()` 函数（沿深度轴堆叠数组）与生成的二值掩码结合，将创建具有四个通道的图像。这使得在保存为 `PNG` 文件时，对象轮廓之外的所有像素变为透明。

            ### 透明像素隔离：子选项

            ??? info "全尺寸图像"

                如果保留全尺寸图像，则不需要额外步骤。

                <figure markdown>
                    ![全尺寸隔离对象图像示例 - 无背景](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/example-full-size-isolated-object-image-no-background.avif){ width=240 }
                    <figcaption>全尺寸输出示例 + 透明背景</figcaption>
                </figure>

            ??? info "裁剪对象图像"

                需要额外步骤将图像裁剪为仅包含对象区域。

                ![裁剪隔离对象图像示例 - 无背景](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/example-crop-isolated-object-image-no-background.avif){ align="right" }
                ```{ .py .annotate }
                # (1) 边界框坐标
                x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
                # 将图像裁剪到对象区域
                iso_crop = isolated[y1:y2, x1:x2]
                ```

                1.  有关边界框结果的更多信息，请参阅[预测模式的 Boxes 部分](../modes/predict.md#boxes)。

                ??? question "这段代码做了什么？"

                    - 使用 `c.boxes.xyxy.cpu().numpy()` 时，边界框以 NumPy 数组形式返回，使用 `xyxy` 框坐标格式，对应边界框（矩形）的点 `xmin, ymin, xmax, ymax`，有关更多信息，请参阅[预测模式的 Boxes 部分](../modes/predict.md#boxes)。

                    - 添加 `squeeze()` 确保从 NumPy 数组中移除所有多余维度。

                    - 使用 `.astype(np.int32)` 转换坐标值将边界框坐标数据类型从 `float32` 更改为 `int32`，以便在使用索引切片裁剪图像时兼容。

                    - 最后，使用索引切片裁剪边界框的图像区域，边界由检测边界框的 `[ymin:ymax, xmin:xmax]` 坐标设置。

    ??? question "如果我想裁剪对象时**包含**背景怎么办？"

        这是 Ultralytics 库的内置功能。请参阅[预测模式推理参数](../modes/predict.md#inference-arguments)中的 `save_crop` 参数了解详情。

    ***

6.  <u>接下来做什么完全由作为开发者的你决定。</u>下面展示了一个可能的后续步骤的基本示例（将图像保存到文件以供将来使用）。
    - **注意：** 此步骤是可选的，如果您的特定使用场景不需要，可以跳过。

    ??? example "最终步骤示例"

        ```python
        # 将隔离对象保存到文件
        _ = cv2.imwrite(f"{img_name}_{label}-{ci}.png", iso_crop)
        ```

        - 在此示例中，`img_name` 是源图像文件的基本名称，`label` 是检测到的类别名称，`ci` 是[对象检测](https://www.ultralytics.com/glossary/object-detection)的索引（在具有相同类别名称的多个实例的情况下）。

## 完整示例代码

以下是前面部分所有步骤组合成的单个代码块。对于重复使用，最好定义一个函数来执行 `for` 循环中包含的部分或全部命令，但这是留给读者的练习。

```{ .py .annotate }
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

m = YOLO("yolo26n-seg.pt")  # (4)!
res = m.predict(source="path/to/image.jpg")  # (3)!

# 遍历检测结果 (5)
for r in res:
    img = np.copy(r.orig_img)
    img_name = Path(r.path).stem

    # 遍历每个对象轮廓 (6)
    for ci, c in enumerate(r):
        label = c.names[c.boxes.cls.tolist().pop()]

        b_mask = np.zeros(img.shape[:2], np.uint8)

        # 创建轮廓掩码 (1)
        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

        # 选择以下之一：

        # 选项 1：使用黑色背景隔离对象
        mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
        isolated = cv2.bitwise_and(mask3ch, img)

        # 选项 2：使用透明背景隔离对象（保存为 PNG 时）
        isolated = np.dstack([img, b_mask])

        # 可选：检测裁剪（从选项 1 或选项 2 中）
        x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
        iso_crop = isolated[y1:y2, x1:x2]

        # 在此处添加您的自定义后处理 (2)
```

1. 此处填充 `contour` 的行合并为一行，而上文中是拆分为多行的。
2. {==此处的内容由您决定！==}
3. 有关更多信息，请参阅[预测模式](../modes/predict.md)。
4. 有关更多信息，请参阅[分割任务](../modes/segment.md#models)。
5. 了解更多关于[结果处理](../modes/predict.md#working-with-results)的信息。
6. 了解更多关于[分割掩码结果](../modes/predict.md#masks)的信息。

## 常见问题

### 如何使用 Ultralytics YOLO26 为分割任务隔离对象？

使用 Ultralytics YOLO26 隔离对象，请按照以下步骤操作：

1. **加载模型并运行推理：**

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n-seg.pt")
    results = model.predict(source="path/to/your/image.jpg")
    ```

2. **生成二值掩码并绘制轮廓：**

    ```python
    import cv2
    import numpy as np

    img = np.copy(results[0].orig_img)
    b_mask = np.zeros(img.shape[:2], np.uint8)
    contour = results[0].masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
    cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
    ```

3. **使用二值掩码隔离对象：**
    ```python
    mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
    isolated = cv2.bitwise_and(mask3ch, img)
    ```

有关更多信息，请参阅[预测模式](../modes/predict.md)和[分割任务](../tasks/segment.md)指南。

### 分割后保存隔离对象有哪些选项？

Ultralytics YOLO26 提供了两种保存隔离对象的主要选项：

1. **使用黑色背景：**

    ```python
    mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
    isolated = cv2.bitwise_and(mask3ch, img)
    ```

2. **使用透明背景：**
    ```python
    isolated = np.dstack([img, b_mask])
    ```

有关更多详细信息，请访问[预测模式](../modes/predict.md)部分。

### 如何使用 Ultralytics YOLO26 将隔离对象裁剪到其边界框？

要将隔离对象裁剪到其边界框：

1. **获取边界框坐标：**

    ```python
    x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy().astype(np.int32)
    ```

2. **裁剪隔离图像：**
    ```python
    iso_crop = isolated[y1:y2, x1:x2]
    ```

在[预测模式](../modes/predict.md#boxes)文档中了解有关边界框结果的更多信息。

### 为什么应该使用 Ultralytics YOLO26 进行分割任务中的对象隔离？

Ultralytics YOLO26 提供：

- **高速**实时对象检测和分割。
- **精确的边界框和掩码生成**，实现精确的对象隔离。
- **全面的文档**和易于使用的 API，实现高效开发。

在[分割任务文档](../tasks/segment.md)中探索使用 YOLO 的优势。

### 我可以使用 Ultralytics YOLO26 保存包含背景的隔离对象吗？

是的，这是 Ultralytics YOLO26 的内置功能。在 `predict()` 方法中使用 `save_crop` 参数。例如：

```python
results = model.predict(source="path/to/your/image.jpg", save_crop=True)
```

在[预测模式推理参数](../modes/predict.md#inference-arguments)部分阅读有关 `save_crop` 参数的更多信息。