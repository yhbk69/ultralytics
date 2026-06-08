---
comments: true
description: 了解如何在 Ultralytics 平台中上传、管理和组织数据集，以进行 YOLO 模型训练，支持自动处理和统计。
keywords: Ultralytics 平台, 数据集, 数据集管理, 数据集版本管理, YOLO, 数据上传, 训练数据, 计算机视觉, 机器学习
---

# 数据集

[Ultralytics 平台](https://platform.ultralytics.com) 数据集为管理训练数据提供了简化的解决方案。上传后，平台会自动处理图像、标签和统计数据。数据处理完成且满足以下条件后，数据集即可用于训练：`train` 拆分中至少有一张图像，`val` 或 `test` 拆分中至少有一张图像，至少有一张已标注图像，且总共至少有两张图像。

## 上传数据集

Ultralytics 平台支持多种上传格式，提供灵活性。

### 支持的格式

=== "图像"

    | 格式   | 扩展名           | 说明               | 最大大小 |
    | ------ | ---------------- | ------------------ | -------- |
    | JPEG   | `.jpg`, `.jpeg`  | 最常用，推荐       | 50 MB    |
    | PNG    | `.png`           | 支持透明度         | 50 MB    |
    | WebP   | `.webp`          | 现代格式，压缩效果好 | 50 MB    |
    | BMP    | `.bmp`           | 未压缩             | 50 MB    |
    | TIFF   | `.tiff`, `.tif`  | 高质量             | 50 MB    |
    | HEIC   | `.heic`          | iPhone 照片        | 50 MB    |
    | AVIF   | `.avif`          | 新一代格式         | 50 MB    |
    | JP2    | `.jp2`           | JPEG 2000          | 50 MB    |
    | DNG    | `.dng`           | 原始相机格式       | 50 MB    |
    | MPO    | `.mpo`           | 多图像对象         | 50 MB    |

=== "视频"

    视频会在客户端以每秒 1 帧的速度自动提取帧（每个视频最多 100 帧）。

    | 格式 | 扩展名  | 提取方式                | 最大大小 |
    | ---- | ------- | ----------------------- | -------- |
    | MP4  | `.mp4`  | 1 FPS，最多 100 帧      | 1 GB     |
    | WebM | `.webm` | 1 FPS，最多 100 帧      | 1 GB     |
    | MOV  | `.mov`  | 1 FPS，最多 100 帧      | 1 GB     |
    | AVI  | `.avi`  | 1 FPS，最多 100 帧      | 1 GB     |
    | MKV  | `.mkv`  | 1 FPS，最多 100 帧      | 1 GB     |
    | M4V  | `.m4v`  | 1 FPS，最多 100 帧      | 1 GB     |

    !!! info "视频帧提取"

        视频帧在上传前于浏览器中以每秒 1 帧的速度提取。60 秒的视频生成 60 帧。每个视频最多 100 帧——对于超过约 100 秒的视频，会从整个时长中均匀采样 100 帧。

=== "压缩包"

    压缩包会自动解压并处理。

    | 格式   | 扩展名                   | 说明           | 免费版 | 专业版 | 企业版 |
    | ------ | ------------------------ | -------------- | ------ | ------ | ------ |
    | ZIP    | `.zip`                   | 最常用         | 10 GB  | 20 GB  | 50 GB  |
    | TAR    | `.tar` `.tar.gz` `.tgz`  | 压缩或原始格式 | 10 GB  | 20 GB  | 50 GB  |
    | NDJSON | `.ndjson`                | 数据集导出     | 10 GB  | 20 GB  | 50 GB  |

### 准备数据集

平台支持 [Ultralytics YOLO](../../datasets/detect/index.md#ultralytics-yolo-format)、[COCO](https://cocodataset.org/#format-data)、[Ultralytics NDJSON](../../datasets/detect/index.md#ultralytics-ndjson-format) 以及原始（未标注）上传：

=== "YOLO 格式"

    使用标准的 YOLO 目录结构和 `data.yaml` 文件：

    ```
    my-dataset/
    ├── images/
    │   ├── train/
    │   │   ├── img001.jpg
    │   │   └── img002.jpg
    │   └── val/
    │       ├── img003.jpg
    │       └── img004.jpg
    ├── labels/
    │   ├── train/
    │   │   ├── img001.txt
    │   │   └── img002.txt
    │   └── val/
    │       ├── img003.txt
    │       └── img004.txt
    └── data.yaml
    ```

    YAML 文件定义数据集配置：

    ```yaml
    # data.yaml
    path: .
    train: images/train
    val: images/val

    names:
        0: person
        1: car
        2: dog
    ```

=== "COCO 格式"

    使用 JSON 标注文件和标准 [COCO 结构](https://cocodataset.org/#format-data)：

    ```
    my-coco-dataset/
    ├── train/
    │   ├── _annotations.coco.json
    │   ├── img001.jpg
    │   └── img002.jpg
    └── val/
        ├── _annotations.coco.json
        ├── img003.jpg
        └── img004.jpg
    ```

    JSON 文件包含 `images`、`annotations` 和 `categories` 数组：

    ```json
    {
        "images": [{ "id": 1, "file_name": "img001.jpg", "width": 640, "height": 480 }],
        "annotations": [{ "id": 1, "image_id": 1, "category_id": 0, "bbox": [100, 50, 200, 300] }],
        "categories": [{ "id": 0, "name": "person" }]
    }
    ```

    COCO 标注在上传时自动转换。支持检测（`bbox`）、分割（`segmentation` 多边形）和姿态（`keypoints`）任务。类别 ID 会在所有标注文件中重新映射为密集的从 0 开始的序列。有关格式转换，请参阅[格式转换工具](../../datasets/detect/index.md#port-or-convert-label-formats)。

=== "分类目录布局"

    分类上传会从常见的文件夹布局中自动检测：

    ```
    split/class/image.jpg
    class/split/image.jpg
    class/image.jpg
    ```

    示例：

    ```
    my-classify-dataset/
    ├── train/
    │   ├── cats/
    │   └── dogs/
    └── val/
        ├── cats/
        └── dogs/
    ```

=== "NDJSON"

    Ultralytics NDJSON 导出可以直接重新上传到平台。这对于在不同工作区之间迁移数据集非常有用，同时可以保留元数据、类别、拆分和标注。

!!! tip "原始上传"

    **原始**：上传未标注的图像（无标签）。适用于计划使用[标注编辑器](annotation.md)在平台上直接标注的场景。

!!! tip "扁平目录结构"

    你也可以上传没有显式拆分文件夹的图像。平台会遵循上传时的活动拆分目标，对于非分类数据集，如果没有提供拆分信息，平台可能会自动从训练集中创建验证拆分。你始终可以在后续使用批量移动到拆分或拆分重新分配功能来重新分配图像。

!!! tip "格式自动检测"

    格式会自动检测：包含 `names`、`train` 或 `val` 键的 `data.yaml` 文件的数据集被视为 YOLO 格式。包含 COCO JSON 文件（含有 `images`、`annotations` 和 `categories` 数组）的数据集被视为 COCO 格式。`.ndjson` 导出会作为 Ultralytics NDJSON 导入。只有图像而没有标注的数据集被视为原始格式。

关于任务特定的格式详情，请参阅[支持的任务](index.md#supported-tasks)和[数据集概述](../../datasets/index.md)。

### 上传流程

1. 在侧边栏中导航到 `Datasets`
2. 点击 `New Dataset` 或将文件拖入上传区域
3. 选择任务类型（参见[支持的任务](index.md#supported-tasks)）
4. 添加名称和可选的描述
5. 设置可见性（公开或私有）和可选的许可证（参见[可用许可证](#available-licenses)）
6. 点击 `Create`

![Ultralytics 平台数据集上传对话框任务选择器](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-datasets-upload-dialog-task-selector.avif)

上传后，平台通过多阶段流水线处理数据：

```mermaid
graph LR
    A[上传] --> B[验证]
    B --> C[标准化]
    C --> D[缩略图]
    D --> E[解析标签]
    E --> F[统计]

    style A fill:#4CAF50,color:#fff
    style B fill:#2196F3,color:#fff
    style C fill:#2196F3,color:#fff
    style D fill:#2196F3,color:#fff
    style E fill:#2196F3,color:#fff
    style F fill:#9C27B0,color:#fff
```

1. **验证**：格式和大小检查
2. **标准化**：大图像调整大小（最大 4096px，最小边长 28px）
3. **缩略图**：生成 256px WebP 预览图
4. **标签解析**：提取 [YOLO](../../datasets/detect/index.md#ultralytics-yolo-format) 和 COCO 格式标签
5. **统计**：计算类别分布和图像尺寸

![Ultralytics 平台数据集上传进度条](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-datasets-upload-progress-bar.avif)

??? tip "上传前验证"

    你可以在上传前本地验证数据集：

    ```python
    from ultralytics.data.utils import check_det_dataset

    check_det_dataset("path/to/data.yaml")
    ```

!!! warning "图像大小要求"

    图像的最短边必须至少为 28px。小于此值的图像会在处理过程中被拒绝。最长边超过 4096px 的图像会自动调整大小，同时保持宽高比。

## 浏览图像

以多种布局查看数据集图像。

从图库工具栏打开[聚类](#clustering)面板，以交互式二维散点图探索数据集。

| 视图       | 说明                                       |
| ---------- | ------------------------------------------ |
| **网格**   | 带标注叠加的缩略图网格（默认）               |
| **紧凑**   | 较小的缩略图，适合快速浏览                   |
| **表格**   | 包含缩略图、文件名、尺寸、大小、拆分、类别和标注数量的列表 |

![Ultralytics 平台数据集图库网格视图（带标注）](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-datasets-gallery-grid-view-with-annotations.avif)

### 排序与筛选

图像可以排序和筛选，以便高效浏览：

=== "排序选项"

    | 排序方式             | 说明                   |
    | -------------------- | ---------------------- |
    | 最新 / 最旧          | 上传 / 创建顺序        |
    | 名称 A-Z / Z-A       | 文件名字母顺序         |
    | 高度 ↑/↓             | 图像高度（像素）       |
    | 宽度 ↑/↓             | 图像宽度（像素）       |
    | 大小 ↑/↓             | 磁盘上的文件大小       |
    | 标注数量 ↑/↓         | 每张图像的标注数量     |

    !!! note "大型数据集"

        对于超过 100,000 张图像的数据集，名称/大小/宽度/高度排序会被禁用以保持图库的响应速度。最新、最旧和标注数量排序仍然可用。

=== "筛选器"

    | 筛选器             | 选项                         |
    | ------------------ | ---------------------------- |
    | **拆分筛选器**     | 训练、验证、测试或全部       |
    | **标签筛选器**     | 全部、已标注或未标注         |
    | **类别筛选器**     | 按类别名称筛选               |
    | **搜索**           | 按文件名筛选图像             |

!!! tip "查找未标注图像"

    将标签筛选器设置为 `Unlabeled`，可以快速找到仍需标注的图像。这对于大型数据集中追踪标注进度特别有用。

### 全屏查看器

点击任意图像打开全屏查看器，功能包括：

- **导航**：使用箭头键或缩略图预览浏览
- **元数据**：文件名、尺寸、拆分标签、标注数量
- **标注**：切换标注叠加层的可见性
- **类别分布**：每个类别的标签数量，带颜色指示器
- **编辑**：进入标注模式以添加或修改标签
- **下载**：下载原始图像文件
- **删除**：从数据集中删除该图像
- **缩放**：`Cmd/Ctrl+滚轮`、`Cmd/Ctrl++` 或 `Cmd/Ctrl+=` 放大，`Cmd/Ctrl+-` 缩小
- **重置视图**：`Cmd/Ctrl + 0` 或重置按钮，使图像适应查看器
- **平移**：缩放后按住 `Space` 并拖动来平移画布
- **像素视图**：切换像素化渲染以进行仔细检查

![Ultralytics 平台数据集全屏查看器（带元数据面板）](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-datasets-fullscreen-viewer-with-metadata-panel.avif)

### 按拆分筛选

按数据集的拆分筛选图像：

| 拆分       | 用途                 |
| ---------- | -------------------- |
| **训练**   | 用于模型训练         |
| **验证**   | 用于训练期间的验证   |
| **测试**   | 用于最终评估         |

## 聚类

`Clustering` 面板将数据集投射到交互式二维散点图中，视觉上相似的图像会聚在一起。可用于发现聚类、识别重复和异常值，以及检查拆分或类别在数据中的分布——无需离开图库。在任何数据集页面的图库工具栏中点击散点图图标即可打开。

![Ultralytics 平台数据集聚类空状态](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-datasets-clustering-empty-state.avif)

### 运行分析

启动分析：

1. 打开数据集，点击图库工具栏中的散点图图标
2. 点击 `Analyze Dataset`
3. 等待进度条完成——结果会显示在同一面板中

分析在后台运行，根据数据集大小可能需要几分钟。你可以关闭面板或离开页面，稍后再回来查看。

### 可视化

分析完成后，面板会显示所有已分析图像的二维散点图。图库筛选器（拆分、类别、已标注/未标注）会使不符合筛选条件的点变暗，让你可以专注于关心的子集。

![Ultralytics 平台数据集聚类散点图](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-datasets-clustering.avif)

#### 按颜色区分

使用面板工具栏中的 `Color by` 下拉菜单更改数据点的着色方式。随时切换视图模式——图表会即时重新着色，方便查看拆分、类别或图像属性在聚类中的分布：

| 选项              | 着色方式                         |
| ----------------- | -------------------------------- |
| **拆分**          | 训练 / 验证 / 测试               |
| **类别**          | 每张图像上的第一个标注类别       |
| **宽度**          | 图像宽度                         |
| **高度**          | 图像高度                         |
| **大小**          | 文件大小                         |
| **标注**          | 每张图像的标注数量               |

![Ultralytics 平台数据集聚类颜色模式](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-datasets-clustering-color-modes.avif)

#### 套索选择

在区域周围绘制自由形状的选择框以高亮图表上的点。图库会筛选出匹配的图像，你可以使用常规的[图像操作](#image-operations)来检查、重新标注、移动或删除它们。

!!! tip "清除选择"

    图表上方有一个标签显示选中的点数量——点击 `×` 清除套索选择并返回完整图库视图。

#### 平移与缩放

直接用鼠标和键盘导航大型散点图：

| 操作                  | 动作                             |
| --------------------- | -------------------------------- |
| **滚轮**              | 平移图表（二维）                 |
| **Cmd/Ctrl+滚轮**     | 以光标位置为锚点放大或缩小       |
| **按住 Space**        | 切换到拖动平移模式               |

### 重新分析

如果数据集在分析后发生了变化，`Re-analyze` 按钮会出现在面板顶部，对拥有者和编辑者可见。

点击 `Re-analyze` 可以从头重新计算嵌入和二维投影。

## 数据集标签页

每个数据集页面最多显示六个标签页，取决于数据集状态和你的权限：

### 图像标签页

显示图像图库的默认视图，带有标注叠加层。支持网格、紧凑和表格视图模式。在此处拖放文件以添加更多图像。

### 类别标签页

当数据集有图像时，此标签页会出现。

管理数据集的标注类别：

- **类别直方图**：显示每个类别标注数量的柱状图，支持线性/对数刻度切换
- **类别表格**：可排序、可搜索的表格，包含类别名称、标签数量和图像数量
- **编辑类别名称**：点击任意类别名称进行内联重命名
- **编辑类别颜色**：点击色块更改类别颜色
- **添加新类别**：使用底部的输入框添加类别

![Ultralytics 平台数据集类别标签页直方图和表格](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-datasets-classes-tab-histogram-and-table.avif)

!!! note "不均衡数据集的对数刻度"

    如果数据集存在类别不均衡（例如 10,000 个 "person" 标注但只有 50 个 "bicycle"），使用类别直方图上的 `Log Scale` 切换可以清晰地可视化所有类别。

### 图表标签页

当数据集有图像时，此标签页会出现。

从数据集自动计算的统计信息：

| 图表                       | 说明                                           |
| -------------------------- | ---------------------------------------------- |
| **拆分分布**               | 训练/验证/测试图像数量和已标注百分比的环形图   |
| **热门类别**               | 10 个最频繁标注类别的环形图                     |
| **图像宽度**               | 图像宽度分布的直方图，含均值                   |
| **图像高度**               | 图像高度分布的直方图，含均值                   |
| **每个实例的点数**         | 每个标注的多边形顶点或关键点数量（分割/姿态）   |
| **标注位置**               | 边界框中心位置的二维热力图                     |
| **图像尺寸**               | 宽度 vs 高度的二维热力图，含宽高比参考线       |

![Ultralytics 平台数据集图表标签页统计网格](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-datasets-charts-tab-statistics-grid.avif)

!!! tip "统计缓存"

    统计数据缓存 5 分钟。标注的更改会在缓存过期后反映出来。

!!! info "全屏热力图"

    点击任意热力图上的展开按钮，以全屏模式查看。这提供更大、更详细的视图——有助于理解大型数据集中的空间模式。

### 模型标签页

查看在此数据集上训练的所有模型，以可搜索的表格展示：

| 列         | 说明                     |
| ---------- | ------------------------ |
| 名称       | 模型名称，带链接         |
| 项目       | 父项目，带图标           |
| 状态       | 训练状态标签             |
| 任务       | YOLO 任务类型            |
| 轮数       | 最佳轮数 / 总轮数        |
| mAP50-95   | 平均精度                 |
| mAP50      | IoU 0.50 时的 mAP        |
| 创建时间   | 创建日期                 |

![Ultralytics 平台数据集模型标签页已训练模型表格](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-datasets-models-tab-trained-models-table.avif)

### 错误标签页

仅当一个或多个文件处理失败时，此标签页才会出现。

处理失败的图像列在此处，包含：

- **错误横幅**：失败图像的总数和指导信息
- **错误表格**：文件名、用户友好的错误描述、修复提示和预览缩略图
- 常见错误包括文件损坏、不支持的格式、图像太小（最小 28px）以及不支持的颜色模式

![Ultralytics 平台数据集错误标签页处理失败](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-datasets-errors-tab-processing-failures.avif)

??? info "常见处理错误"

    | 错误                       | 原因                       | 修复方法                         |
    | -------------------------- | -------------------------- | -------------------------------- |
    | 无法读取图像文件           | 文件损坏或不支持的格式     | 从图像编辑器重新导出             |
    | 不完整或已损坏             | 传输过程中文件被截断       | 重新下载原始文件                 |
    | 图像太小                   | 最小边长低于 28px          | 使用更高分辨率的源图像           |
    | 不支持的颜色模式           | CMYK 或索引颜色模式        | 转换为 RGB 模式                  |

### 版本标签页

为数据集创建不可变的 NDJSON 快照，以实现可复现的训练。每个版本会捕获创建时的图像数量、类别数量、标注数量和文件大小。

| 列         | 说明                         |
| ---------- | ---------------------------- |
| 版本       | 版本号（v1, v2, ...）        |
| 描述       | 用户提供的描述（可编辑）     |
| 图像       | 快照时的图像数量             |
| 类别       | 快照时的类别数量             |
| 标注       | 快照时的标注数量             |
| 大小       | NDJSON 导出文件大小          |
| 创建时间   | 版本创建的时间               |

创建版本：

1. 打开 **版本** 标签页
2. （可选）输入描述（例如 "Added 500 training images" 或 "Fixed mislabeled classes"）
3. 点击 **+ New Version**
4. 新版本出现在表格中
5. 需要时，从表格中单独下载版本

每个版本按顺序编号（v1, v2, v3...）并永久存储。你可以随时从版本表格下载任何历史版本。

!!! note "仅就绪数据集"

    版本创建在数据集达到 `ready` 状态后才可用。

!!! tip "何时创建版本"

    在数据集发生重大更改之前和之后创建版本——添加图像、修复标注或重新平衡拆分。这样可以比较不同数据集状态下的模型性能。

!!! note "NDJSON 文件大小"

    显示的大小是 NDJSON 导出文件大小，其中包含图像 URL 和标注，而不是图像本身。实际图像数据单独存储，通过签名 URL 访问。

## 导出数据集

通过数据集标题栏或版本标签页下载 NDJSON，导出数据集以供离线使用。

导出步骤：

1. 点击数据集标题栏中的 **Export** 按钮
2. 直接下载当前的 NDJSON 快照
3. 如需可重新下载的不可变编号快照，请使用 **版本** 标签页

![Ultralytics 平台数据集导出 NDJSON 下载](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-datasets-export-ndjson-download.avif)

NDJSON 格式每行存储一个 JSON 对象。第一行包含数据集元数据，之后每行对应一张图像：

```json
{"type": "dataset", "task": "detect", "name": "my-dataset", "description": "...", "url": "https://platform.ultralytics.com/...", "class_names": {"0": "person", "1": "car"}, "version": 1, "created_at": "2026-01-15T10:00:00Z", "updated_at": "2026-02-20T14:30:00Z"}
{"type": "image", "file": "img001.jpg", "url": "https://...", "width": 640, "height": 480, "split": "train", "annotations": {"boxes": [[0, 0.5, 0.5, 0.2, 0.3]]}}
{"type": "image", "file": "img002.jpg", "url": "https://...", "width": 1280, "height": 720, "split": "val"}
```

!!! note "签名 URL"

    导出的 NDJSON 中的图像 URL 是签名 URL，有效期 7 天。如需新的 URL，请重新导出数据集或创建新版本。

完整规范请参阅 [Ultralytics NDJSON 格式文档](../../datasets/detect/index.md#ultralytics-ndjson-format)。

## 图像操作

### 快速操作

在 **网格** 或 **紧凑** 视图中右键点击任意图像以访问快速操作：

| 操作                 | 说明                                   |
| -------------------- | -------------------------------------- |
| **移动到拆分**       | 将图像重新分配到训练、验证或测试拆分   |
| **下载**             | 下载原始图像文件                       |
| **删除**             | 从数据集中删除该图像                   |

![Ultralytics 平台数据集图像卡片右键菜单](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-datasets-image-card-context-menu.avif)

!!! tip "单张 vs 批量"

    图像右键菜单操作的是**单张图像**。如需批量操作多张图像，请使用带复选框选择的**表格**视图。

### 批量移动到拆分

将选中的图像重新分配到同一数据集的另一个拆分：

1. 切换到 **表格** 视图
2. 使用复选框选择图像
3. 右键打开右键菜单
4. 选择 `Move to split` > **训练**、**验证** 或 **测试**

你也可以在网格视图中将图像拖放到拆分筛选标签页上。

!!! tip "组织训练/验证拆分"

    将所有图像上传到一个数据集，然后使用批量移动到拆分为训练、验证和测试拆分配置子集。

### 拆分重新分配

使用自定义比例将所有图像重新分配到训练、验证和测试拆分：

1. 点击数据集工具栏中的 **拆分栏**，打开 **重新分配拆分** 对话框
2. 使用以下任意方法调整拆分百分比
3. 查看实时图像数量预览以确认分布
4. 点击 **Apply** 按照你的百分比随机重新分配所有图像

![Ultralytics 平台数据集拆分重新分配对话框](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/platform-datasets-split-redistribution-dialog.avif)

对话框提供三种设置目标拆分比例的方式：

| 方法       | 说明                                                     |
| ---------- | -------------------------------------------------------- |
| **拖动**   | 拖动彩色段之间的手柄，直观调整拆分边界                   |
| **输入**   | 编辑任意拆分的百分比输入（其他两个拆分自动按比例重新平衡）|
| **自动**   | 一键设置 80/20 训练/验证拆分，测试拆分为 0%             |

实时预览会在应用前准确显示每个拆分中将有多少张图像。

!!! tip "快速 80/20 拆分"

    点击 **Auto** 按钮可立即设置推荐的 80/20 训练/验证拆分。这是训练中最常用的比例。

### 批量删除

一次删除多张图像：

1. 在表格视图中选择图像
2. 右键点击并选择 `Delete`
3. 确认删除

## 数据集 URI

使用 `ul://` URI 格式引用平台数据集（参见[使用平台数据集](../api/index.md#using-platform-datasets)）：

```
ul://username/datasets/dataset-slug
```

使用此 URI 从任何地方训练模型：

=== "CLI"

    ```bash
    export ULTRALYTICS_API_KEY="YOUR_API_KEY"
    yolo train model=yolo26n.pt data=ul://username/datasets/my-dataset epochs=100
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    model.train(data="ul://username/datasets/my-dataset", epochs=100)
    ```

!!! example "使用平台数据在任何地方训练"

    `ul://` URI 适用于任何环境：

    - **本地机器**：在你的硬件上训练，数据自动下载
    - **Google Colab**：在 Notebook 中访问你的平台数据集
    - **远程服务器**：在云虚拟机上训练，拥有完整的数据集访问权限

## 可用许可证

平台支持数据集的以下许可证：

| 许可证         | 类型             |
| -------------- | ---------------- |
| None           | 未选择许可证     |
| CC0-1.0        | 公共领域         |
| CC-BY-2.5      | 宽松             |
| CC-BY-4.0      | 宽松             |
| CC-BY-SA-4.0   | Copyleft         |
| CC-BY-NC-4.0   | 非商业           |
| CC-BY-NC-SA-4.0| Copyleft         |
| CC-BY-ND-4.0   | 禁止演绎         |
| CC-BY-NC-ND-4.0| 非商业           |
| Apache-2.0     | 宽松             |
| MIT            | 宽松             |
| AGPL-3.0       | Copyleft         |
| GPL-3.0        | Copyleft         |
| Research-Only  | 限制             |
| Other          | 自定义           |

!!! note "Copyleft 许可证"

    克隆具有 Copyleft 许可证（AGPL-3.0、GPL-3.0、CC-BY-SA-4.0、CC-BY-NC-SA-4.0）的数据集时，克隆会继承该许可证且许可证选择器会被锁定。

## 可见性设置

控制谁可以查看你的数据集：

| 设置         | 说明                         |
| ------------ | ---------------------------- |
| **私有**     | 只有你可以访问               |
| **公开**     | 任何人都可以在探索页面查看   |

可见性在创建数据集时通过 `New Dataset` 对话框中的切换开关设置。公开数据集在[探索](../explore.md)页面上可见。

## 编辑数据集

数据集元数据直接在数据集页面上内联编辑——无需对话框：

- **名称**：点击数据集名称进行编辑。更改在失焦或按 `Enter` 时自动保存。
- **描述**：点击描述（或 "Add a description..." 占位符）进行编辑。更改会自动保存。
- **任务类型**：点击任务标签选择不同的任务类型。
- **许可证**：点击许可证选择器更改数据集许可证。

!!! info "更改任务类型"

    每张图像同时存储所有任务类型的标注。更改数据集任务类型会控制在编辑器中显示哪些标注，以及导出和训练中包含哪些标注。其他任务类型的标注保留在数据库中，切换回来时会重新出现。

## 克隆数据集

查看不属于你的公开数据集时，点击 `Clone Dataset` 在你的工作区中创建副本。克隆包含所有图像、标注和类别定义。如果原始数据集具有 Copyleft 许可证，克隆会继承该许可证且许可证选择器会被锁定。

## 收藏与分享

- **收藏**：点击星标按钮收藏数据集。星标数对所有用户可见。
- **分享**：对于公开数据集，点击分享按钮复制链接或分享到社交平台。

## 删除数据集

删除不再需要的数据集：

1. 打开数据集操作菜单
2. 点击 `Delete`
3. 在对话框中确认："This will move [name] to trash. You can restore it within 30 days."

!!! note "回收站与恢复"

    删除的数据集会移至回收站——而非永久删除。你可以在 30 天内从 [`Settings > Trash`](../account/trash.md) 恢复。

## 在数据集上训练

直接从数据集开始训练：

1. 在数据集页面上点击 `New Model`
2. 选择一个项目或创建新项目
3. 配置训练参数
4. 开始训练

```mermaid
graph LR
    A[数据集] --> B[新模型]
    B --> C[选择项目]
    C --> D[配置]
    D --> E[开始训练]

    style A fill:#2196F3,color:#fff
    style E fill:#4CAF50,color:#fff
```

详情请参见[云端训练](../train/cloud-training.md)。

## 常见问题

### 上传后我的数据会怎样？

你的数据会在你选择的区域（美国、欧盟或亚太）中处理和存储。图像会经历以下步骤：

1. 验证格式和大小
2. 如果最小边长低于 28px 则被拒绝
3. 如果大于 4096px 则进行标准化（保持宽高比；为优化存储进行编码）
4. 使用基于内容的地址存储（CAS）和 XXH3-128 哈希存储
5. 生成 256px WebP 缩略图以便快速浏览

### 存储如何运作？

Ultralytics 平台使用**基于内容的地址存储（CAS）**以实现高效存储：

- **去重**：不同用户上传的相同图像只存储一次
- **完整性**：XXH3-128 哈希确保数据完整性
- **效率**：降低存储成本并加快处理速度
- **区域性**：数据保留在你选择的区域（美国、欧盟或亚太）

### 可以向现有数据集添加图像吗？

可以，将文件拖放到数据集页面或使用上传按钮添加更多图像。新的统计数据会自动计算。

### 如何在拆分之间移动图像？

使用批量移动到拆分功能：

1. 在表格视图中选择图像
2. 右键点击并选择 `Move to split`
3. 选择目标拆分（训练、验证或测试）

### 支持哪些标签格式？

Ultralytics 平台支持 YOLO 标签、COCO JSON、Ultralytics NDJSON 和原始图像上传：

=== "YOLO 格式"

    每张图像一个 `.txt` 文件，使用归一化坐标（0-1 范围）：

    | 任务     | 格式                             | 示例                               |
    | -------- | -------------------------------- | ---------------------------------- |
    | 检测     | `class cx cy w h`                | `0 0.5 0.5 0.2 0.3`                |
    | 分割     | `class x1 y1 x2 y2 ...`          | `0 0.1 0.1 0.9 0.1 0.9 0.9`        |
    | 姿态     | `class cx cy w h kx1 ky1 v1 ...` | `0 0.5 0.5 0.2 0.3 0.6 0.7 2`     |
    | OBB      | `class x1 y1 x2 y2 x3 y3 x4 y4`  | `0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9`|
    | 分类     | 目录结构                         | `train/cats/`, `train/dogs/`       |

    姿态可见性标志：0=未标注，1=已标注但被遮挡，2=已标注且可见。

=== "COCO 格式"

    包含 `images`、`annotations` 和 `categories` 数组的 JSON 文件。支持检测（`bbox`）、分割（多边形）和姿态（`keypoints`）任务。COCO 使用绝对像素坐标，在上传时自动转换为归一化格式。

=== "NDJSON"

    Ultralytics NDJSON 导出可以重新导入到平台。这是在不同工作区之间迁移数据集元数据、拆分和标注的最完整方式。

### 可以为多种任务类型标注同一个数据集吗？

可以。每张图像同时存储所有 5 种任务类型（检测、分割、姿态、OBB、分类）的标注。你可以随时切换数据集的活动任务类型，而不会丢失现有标注。只有与活动任务类型匹配的标注才会显示在编辑器中，并包含在导出和训练中——其他任务的标注会被保留，切换回来时会重新出现。
